// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — KVCache  (kv_cache.hpp)
//
// Per-layer key/value store for autoregressive generation, with a choice of
// storage dtype so the cache can be shrunk when available RAM is tight:
//
//   F32   — full precision, baseline (32 bits/element)
//   F16   — IEEE half float, lossless-enough for K/V in practice (16 bits/element,
//           2x smaller than F32). Reuses the same f16<->f32 conversion as the
//           .nfw weight loader, so there's no new numerical code to trust.
//   Q8_0  — int8 with a per-token (per-row) f32 scale, one scale per
//           [n_kv_heads * d_head] row (8 bits/element + ~0.4% scale overhead,
//           4x smaller than F32). Same scheme GGML/llama.cpp calls "Q8_0":
//           scale = max(|row|) / 127, q[i] = round(row[i] / scale).
//
// All three dtypes are appended to and read from with the same interface
// (append() / dequant_row()), so callers (attention.hpp) don't need to know
// which one is active. Quantization/dequantization happens per-token, which
// keeps the extra CPU cost proportional to the (already tiny) K/V row size
// rather than the full context — cheap relative to the GEMMs around it, and
// a good trade when the bottleneck is memory bandwidth/capacity, not compute.
// =============================================================================
#include "tensor.hpp"
#include "weights.hpp"   // reuses detail::f16_to_f32 / f32_to_f16
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vlrf {

enum class KVCacheDType : int {
    F32  = 0,
    F16  = 1,
    Q8_0 = 2,
};

inline const char* kv_cache_dtype_name(KVCacheDType t) {
    switch (t) {
        case KVCacheDType::F32:  return "f32";
        case KVCacheDType::F16:  return "f16";
        case KVCacheDType::Q8_0: return "q8_0";
    }
    return "f32";
}

// Bytes/element for a given cache dtype, ignoring the small per-row scale
// overhead of Q8_0 (~4 bytes per row of `row_dim` elements — negligible
// unless row_dim is very small).
inline double kv_cache_bytes_per_element(KVCacheDType t) {
    switch (t) {
        case KVCacheDType::F32:  return 4.0;
        case KVCacheDType::F16:  return 2.0;
        case KVCacheDType::Q8_0: return 1.0;
    }
    return 4.0;
}

namespace detail {

// f32 -> f16, software round-to-nearest-even-ish (round-to-nearest is enough
// for KV cache values; this mirrors the precision f16_to_f32 in weights.hpp
// already assumes elsewhere in the codebase).
inline uint16_t f32_to_f16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) {
        // Inf / NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        // Underflow to zero (KV cache values are never denormal-critical)
        return (uint16_t)sign;
    }
    if (exp >= 0x1F) {
        // Overflow to inf
        return (uint16_t)(sign | 0x7C00u);
    }
    // Round mantissa from 23 -> 10 bits (round-to-nearest, ties away handled
    // implicitly by the +0x00000FFF/2 style bias below).
    uint32_t rounded_mant = (mant + 0x00001000u) >> 13;
    if (rounded_mant & 0x0400u) {
        // Mantissa rounded up into the next exponent
        rounded_mant = 0;
        exp += 1;
        if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | rounded_mant);
}

// Quantize one row of `n` floats to int8 with a single per-row scale.
// Returns the scale used (0.0f for an all-zero row, in which case the
// int8 output is all zeros too and dequant will correctly reproduce zero).
inline float quantize_row_q8_0(const float* src, int8_t* dst, int n) noexcept {
    float amax = 0.f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    if (amax == 0.f) {
        std::fill(dst, dst + n, 0);
        return 0.f;
    }
    float scale = amax / 127.f;
    float inv_scale = 1.f / scale;
    for (int i = 0; i < n; ++i) {
        int32_t q = (int32_t)std::lround(src[i] * inv_scale);
        q = std::min(127, std::max(-127, q));
        dst[i] = (int8_t)q;
    }
    return scale;
}

inline void dequantize_row_q8_0(const int8_t* src, float scale, float* dst, int n) noexcept {
    for (int i = 0; i < n; ++i) dst[i] = (float)src[i] * scale;
}

} // namespace detail

// -----------------------------------------------------------------------------
// KVCache
// -----------------------------------------------------------------------------
// Fixed-capacity ring-free cache: allocated once for max_seq_len tokens,
// filled left-to-right as generation proceeds (matches existing usage in
// transformer.hpp — reset_kv_cache() / kv_caches[l].used).
// -----------------------------------------------------------------------------
struct KVCache {
    KVCacheDType dtype{KVCacheDType::F32};
    int max_seq_len{0};
    int n_kv_heads{0};
    int d_head{0};
    int row_dim{0};   // n_kv_heads * d_head
    int used{0};      // tokens stored so far

    // --- F32 storage --------------------------------------------------
    Tensor K, V;

    // --- F16 storage ----------------------------------------------------
    std::vector<uint16_t> K_f16, V_f16;

    // --- Q8_0 storage -----------------------------------------------------
    std::vector<int8_t> K_q8, V_q8;
    std::vector<float>  K_scale, V_scale;  // one scale per token row

    KVCache() = default;

    KVCache(int max_seq_len_, int n_kv_heads_, int d_head_,
            KVCacheDType dtype_ = KVCacheDType::F32) {
        dtype       = dtype_;
        max_seq_len = max_seq_len_;
        n_kv_heads  = n_kv_heads_;
        d_head      = d_head_;
        row_dim     = n_kv_heads_ * d_head_;

        switch (dtype) {
            case KVCacheDType::F32:
                K = Tensor({max_seq_len, row_dim});
                V = Tensor({max_seq_len, row_dim});
                K.zero(); V.zero();
                break;
            case KVCacheDType::F16:
                K_f16.assign((size_t)max_seq_len * row_dim, 0);
                V_f16.assign((size_t)max_seq_len * row_dim, 0);
                break;
            case KVCacheDType::Q8_0:
                K_q8.assign((size_t)max_seq_len * row_dim, 0);
                V_q8.assign((size_t)max_seq_len * row_dim, 0);
                K_scale.assign(max_seq_len, 0.f);
                V_scale.assign(max_seq_len, 0.f);
                break;
        }
    }

    void reset() {
        used = 0;
        switch (dtype) {
            case KVCacheDType::F32:
                K.zero(); V.zero();
                break;
            case KVCacheDType::F16:
                std::fill(K_f16.begin(), K_f16.end(), (uint16_t)0);
                std::fill(V_f16.begin(), V_f16.end(), (uint16_t)0);
                break;
            case KVCacheDType::Q8_0:
                std::fill(K_q8.begin(), K_q8.end(), (int8_t)0);
                std::fill(V_q8.begin(), V_q8.end(), (int8_t)0);
                std::fill(K_scale.begin(), K_scale.end(), 0.f);
                std::fill(V_scale.begin(), V_scale.end(), 0.f);
                break;
        }
    }

    // Approximate resident bytes for this cache (all layers combined is
    // this times n_layers — see TransformerModel::kv_cache_bytes()).
    int64_t bytes() const {
        double bpe = kv_cache_bytes_per_element(dtype);
        int64_t data_bytes = (int64_t)(2.0 * max_seq_len * row_dim * bpe);
        int64_t scale_bytes = (dtype == KVCacheDType::Q8_0)
            ? (int64_t)(2 * max_seq_len * sizeof(float))
            : 0;
        return data_bytes + scale_bytes;
    }

    // -------------------------------------------------------------------------
    // append() — write T new rows of K and V (each row_dim floats) starting
    // at the current `used` position, then advance `used` by T.
    //   k_src, v_src : [T, row_dim] contiguous f32, as produced by the
    //                  Q/K/V projections in attention.hpp.
    // -------------------------------------------------------------------------
    void append(const float* k_src, const float* v_src, int T) {
        if (used + T > max_seq_len)
            throw std::runtime_error("KVCache::append: exceeds max_seq_len");

        switch (dtype) {
            case KVCacheDType::F32: {
                float* Kd = K.data_ptr() + (int64_t)used * row_dim;
                float* Vd = V.data_ptr() + (int64_t)used * row_dim;
                std::copy(k_src, k_src + (size_t)T * row_dim, Kd);
                std::copy(v_src, v_src + (size_t)T * row_dim, Vd);
                break;
            }
            case KVCacheDType::F16: {
                uint16_t* Kd = K_f16.data() + (int64_t)used * row_dim;
                uint16_t* Vd = V_f16.data() + (int64_t)used * row_dim;
                int64_t n = (int64_t)T * row_dim;
                for (int64_t i = 0; i < n; ++i) {
                    Kd[i] = detail::f32_to_f16(k_src[i]);
                    Vd[i] = detail::f32_to_f16(v_src[i]);
                }
                break;
            }
            case KVCacheDType::Q8_0: {
                for (int t = 0; t < T; ++t) {
                    int row = used + t;
                    K_scale[row] = detail::quantize_row_q8_0(
                        k_src + (int64_t)t * row_dim,
                        K_q8.data() + (int64_t)row * row_dim, row_dim);
                    V_scale[row] = detail::quantize_row_q8_0(
                        v_src + (int64_t)t * row_dim,
                        V_q8.data() + (int64_t)row * row_dim, row_dim);
                }
                break;
            }
        }
        used += T;
    }

    // -------------------------------------------------------------------------
    // dequant_into() — materialize rows [0, ctx_len) of K and V as
    // contiguous f32 buffers for the attention GEMMs. For the F32 dtype
    // this is a plain view (no copy needed — see fast path in attention.hpp
    // via raw_f32() below); for F16/Q8_0 it always copies+converts.
    // -------------------------------------------------------------------------
    void dequant_into(int ctx_len, float* K_out, float* V_out) const {
        int64_t n = (int64_t)ctx_len * row_dim;
        switch (dtype) {
            case KVCacheDType::F32: {
                std::copy(K.data_ptr(), K.data_ptr() + n, K_out);
                std::copy(V.data_ptr(), V.data_ptr() + n, V_out);
                break;
            }
            case KVCacheDType::F16: {
                for (int64_t i = 0; i < n; ++i) {
                    K_out[i] = detail::f16_to_f32(K_f16[i]);
                    V_out[i] = detail::f16_to_f32(V_f16[i]);
                }
                break;
            }
            case KVCacheDType::Q8_0: {
                for (int row = 0; row < ctx_len; ++row) {
                    detail::dequantize_row_q8_0(
                        K_q8.data() + (int64_t)row * row_dim, K_scale[row],
                        K_out + (int64_t)row * row_dim, row_dim);
                    detail::dequantize_row_q8_0(
                        V_q8.data() + (int64_t)row * row_dim, V_scale[row],
                        V_out + (int64_t)row * row_dim, row_dim);
                }
                break;
            }
        }
    }

    // True if K/V for [0, used) can be read as a contiguous f32 buffer
    // without copying (only the F32 dtype supports this).
    bool has_raw_f32() const { return dtype == KVCacheDType::F32; }
    const float* raw_f32_K() const { return K.data_ptr(); }
    const float* raw_f32_V() const { return V.data_ptr(); }
};

} // namespace vlrf
