// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Attention Block  (attention.hpp)
// Implements Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
// Supports KV-cache for O(1) incremental decoding.
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include "compute.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "rope.hpp"
#include "kv_cache.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace vlrf {

// -----------------------------------------------------------------------------
// AttentionBlock
// Weights layout (loaded from WeightMap under a given prefix):
//   {prefix}.q_proj.weight  [d_model, d_model]
//   {prefix}.k_proj.weight  [d_kv, d_model]        where d_kv = n_kv_heads * d_head
//   {prefix}.v_proj.weight  [d_kv, d_model]
//   {prefix}.o_proj.weight  [d_model, d_model]
//   {prefix}.q_proj.bias    [d_model]               (optional)
//   {prefix}.k_proj.bias    [d_kv]                  (optional)
//   {prefix}.v_proj.bias    [d_kv]                  (optional)
//   {prefix}.o_proj.bias    [d_model]               (optional)
// -----------------------------------------------------------------------------
class AttentionBlock {
public:
    // Weights (pointers into WeightMap — no copy)
    Tensor* Wq{nullptr};  Tensor* bq{nullptr};
    Tensor* Wk{nullptr};  Tensor* bk{nullptr};
    Tensor* Wv{nullptr};  Tensor* bv{nullptr};
    Tensor* Wo{nullptr};  Tensor* bo{nullptr};

    // Config cache
    int   n_heads{};
    int   n_kv_heads{};
    int   d_model{};
    int   d_head{};
    int   d_kv{};          // n_kv_heads * d_head
    bool  use_rope{false};
    bool  use_causal_mask{true};
    float rope_theta{10000.f};
    float attention_multiplier{0.0f};  // MuP override (IBM Granite 4.0+); 0.0 = "unset"

    void bind(WeightMap& wm, std::string const& prefix, ModelConfig const& cfg) {
        n_heads    = cfg.n_heads;
        n_kv_heads = cfg.n_kv_heads;
        d_model    = cfg.d_model;
        d_head     = cfg.d_head();
        d_kv       = n_kv_heads * d_head;

        Wq = &wm.get(prefix + ".q_proj.weight");
        Wk = &wm.get(prefix + ".k_proj.weight");
        Wv = &wm.get(prefix + ".v_proj.weight");
        Wo = &wm.get(prefix + ".o_proj.weight");

        if (wm.has(prefix + ".q_proj.bias")) bq = &wm.get(prefix + ".q_proj.bias");
        if (wm.has(prefix + ".k_proj.bias")) bk = &wm.get(prefix + ".k_proj.bias");
        if (wm.has(prefix + ".v_proj.bias")) bv = &wm.get(prefix + ".v_proj.bias");
        if (wm.has(prefix + ".o_proj.bias")) bo = &wm.get(prefix + ".o_proj.bias");

        use_rope   = (cfg.pos_embedding == PosEmbedding::ROTARY);
        use_causal_mask = cfg.use_causal_mask;
        rope_theta = cfg.rope_theta;
        attention_multiplier = cfg.attention_multiplier;

        if (use_rope) {
            // Initialise the global RoPE cache (idempotent)
            global_rope_cache().init(cfg.max_seq_len, d_head, rope_theta);
        }
    }

    // -------------------------------------------------------------------------
    // Forward pass
    //   x        : [T, d_model]    input (T = sequence length for this call)
    //   kv_cache : optional KV cache (nullptr = no cache / full context)
    //   returns  : [T, d_model]
    // -------------------------------------------------------------------------
    Tensor forward(Tensor const& x, KVCache* kv_cache = nullptr) const {
        int T = x.dim(0);

        // --- Projections ---------------------------------------------------
        // Q: [T, d_model]   K,V: [T, d_kv]
        // GEMM itself routes to the GPU backend when one is active (see
        // compute::matmul); the transpose stays on CPU (IBackend has no
        // transpose op), so there's still a small CPU step here even in the
        // GPU case.
        Tensor Q = compute::linear(x, *Wq); // x*Wq^T
        Tensor K = compute::linear(x, *Wk);
        Tensor V = compute::linear(x, *Wv);

        // add_bias_inplace has no IBackend equivalent — always CPU.
        if (bq) ops::add_bias_inplace(Q, *bq);
        if (bk) ops::add_bias_inplace(K, *bk);
        if (bv) ops::add_bias_inplace(V, *bv);

        // --- RoPE: rotate Q and K based on position -----------------------
        // pos_offset = number of tokens already in cache (for incremental decode)
        if (use_rope) {
            int pos_offset = kv_cache ? kv_cache->used : 0;
            auto& rc = global_rope_cache();
            // Q: [T, n_heads * d_head]
            rc.apply(Q.data_ptr(), T, n_heads,    pos_offset);
            // K: [T, n_kv_heads * d_head]
            rc.apply(K.data_ptr(), T, n_kv_heads, pos_offset);
        }

        // --- KV-Cache append -----------------------------------------------
        int ctx_len = T;
        Tensor K_ctx, V_ctx;
        // Owns dequantized storage when the cache dtype isn't F32 (F16/Q8_0
        // paths materialize a contiguous f32 buffer for the GEMMs below;
        // F32 caches skip this copy entirely via has_raw_f32()). Declared
        // here so its lifetime covers the Tensor::from_ptr views into it.
        std::vector<float> K_dequant, V_dequant;

        if (kv_cache) {
            kv_cache->append(K.data_ptr(), V.data_ptr(), T);
            ctx_len = kv_cache->used;

            if (kv_cache->has_raw_f32()) {
                // No copy needed — view directly into the cache's f32 storage.
                K_ctx = Tensor::from_ptr(const_cast<float*>(kv_cache->raw_f32_K()), {ctx_len, d_kv});
                V_ctx = Tensor::from_ptr(const_cast<float*>(kv_cache->raw_f32_V()), {ctx_len, d_kv});
            } else {
                // F16/Q8_0: dequantize the whole used range into scratch
                // buffers. Cost is O(ctx_len * d_kv) per forward call —
                // small next to the GEMMs it feeds, and the point of
                // quantizing is to shrink resident memory, not compute.
                K_dequant.resize((size_t)ctx_len * d_kv);
                V_dequant.resize((size_t)ctx_len * d_kv);
                kv_cache->dequant_into(ctx_len, K_dequant.data(), V_dequant.data());
                K_ctx = Tensor::from_ptr(K_dequant.data(), {ctx_len, d_kv});
                V_ctx = Tensor::from_ptr(V_dequant.data(), {ctx_len, d_kv});
            }
        } else {
            K_ctx = K;
            V_ctx = V;
        }

        // --- Reshape for multi-head attention ------------------------------
        // Q  : [T,        n_heads,    d_head]  -> [n_heads, T,       d_head]
        // K,V: [ctx_len, n_kv_heads, d_head]  -> [n_kv_heads, ctx_len, d_head]
        Tensor Qr = reshape_heads(Q, T,       n_heads,    d_head);
        Tensor Kr = reshape_heads(K_ctx, ctx_len, n_kv_heads, d_head);
        Tensor Vr = reshape_heads(V_ctx, ctx_len, n_kv_heads, d_head);

        // --- GQA: expand K,V heads if needed -------------------------------
        Tensor Kr_exp = Kr, Vr_exp = Vr;
        if (n_kv_heads != n_heads) {
            Kr_exp = expand_kv_heads(Kr, n_heads / n_kv_heads);
            Vr_exp = expand_kv_heads(Vr, n_heads / n_kv_heads);
        }

        // --- Scaled dot-product attention ----------------------------------
        bool causal = use_causal_mask;
        if (causal) {
            // Causal mask is only meaningful for autoregressive decoding.
            // When using KV cache with T==1, the new token should attend to all past tokens,
            // so no mask is applied in that single-token decode case.
            if (kv_cache && T == 1) causal = false;
            else if (kv_cache == nullptr || T > 1) causal = true;
            else causal = false;
        }

        Tensor attn_out = sdp_heads(Qr, Kr_exp, Vr_exp, T, ctx_len, causal);

        // --- Merge heads  [T, d_model] -------------------------------------
        Tensor merged({T, d_model});
        merge_heads(attn_out, merged, T, n_heads, d_head);

        // --- Output projection ---------------------------------------------
        Tensor out = compute::linear(merged, *Wo);
        if (bo) ops::add_bias_inplace(out, *bo);
        return out;
    }

    // Flash-attention-style forward for a single head, single thread.
    // Q: [T, d_h]  K,V: [ctx_len, d_h]  ->  O: [T, d_h]
    //
    // Tiles queries into blocks of kBr rows and keys/values into blocks of
    // kBc rows. For each query block, walks key blocks left-to-right
    // maintaining, per query row:
    //   m   = running max of raw (scaled) scores seen so far
    //   l   = running sum of exp(score - m)          (softmax denominator)
    //   acc = running sum of softmax_weight * V       (unnormalized output)
    // Standard "online softmax" rescaling (Milakov & Gimelshein / FlashAttention):
    // when a new block raises the running max from m to m_new, the old
    // l and acc are rescaled by exp(m - m_new) before adding the new
    // block's contribution, so the result is mathematically identical to
    // computing full softmax over all of [0, ctx_len) at once — just
    // without ever holding all of it in memory simultaneously.
    //
    // Peak extra memory per call: kBr*kBc (score tile) + kBr*d_h (accum) +
    // 2*kBr (m, l) floats — independent of T and ctx_len. Compare to the
    // old per_head path's T*ctx_len score buffer.
    static void flash_attn_head(float const* Q, float const* K, float const* V,
                                 float* O, int T, int ctx_len, int d_h,
                                 float scale, bool causal) {
        constexpr int kBr = 64;   // query rows per tile
        constexpr int kBc = 64;   // key/value rows per tile

        std::vector<float> m(kBr), l(kBr);
        std::vector<float> acc((size_t)kBr * d_h);
        std::vector<float> s_tile((size_t)kBr * kBc);

        for (int q0 = 0; q0 < T; q0 += kBr) {
            int br = std::min(kBr, T - q0);

            std::fill(m.begin(), m.begin() + br, -std::numeric_limits<float>::infinity());
            std::fill(l.begin(), l.begin() + br, 0.f);
            std::fill(acc.begin(), acc.begin() + (size_t)br * d_h, 0.f);

            // With a causal mask, query row q0+qi can only see keys up to
            // (ctx_len - T) + q0 + qi (the "past" offset accounts for
            // queries being the tail end of a longer KV-cache context).
            // No key block beyond that index needs to be visited at all
            // for this query tile, which additionally skips work that the
            // old dense implementation always computed (score + mask to
            // -1e30 + exp(-1e30)=0) — this is a real compute saving on
            // top of the memory saving.
            int past = ctx_len - T;
            int k_limit = causal ? std::min(ctx_len, q0 + br + past) : ctx_len;

            for (int k0 = 0; k0 < k_limit; k0 += kBc) {
                int bc = std::min(kBc, k_limit - k0);

                // s_tile[qi][kj] = scale * dot(Q[q0+qi], K[k0+kj])
                for (int qi = 0; qi < br; ++qi) {
                    float const* qrow = Q + (size_t)(q0 + qi) * d_h;
                    for (int kj = 0; kj < bc; ++kj) {
                        float const* krow = K + (size_t)(k0 + kj) * d_h;
                        float dot = 0.f;
                        for (int d = 0; d < d_h; ++d) dot += qrow[d] * krow[d];
                        s_tile[(size_t)qi * kBc + kj] = dot * scale;
                    }
                }

                // Causal masking within this tile (only the boundary block
                // needs it — blocks fully inside the causal window need no
                // per-element check, but we keep it simple and correct
                // rather than special-casing the boundary block).
                if (causal) {
                    for (int qi = 0; qi < br; ++qi) {
                        int qpos = q0 + qi + past;
                        for (int kj = 0; kj < bc; ++kj) {
                            int kpos = k0 + kj;
                            if (kpos > qpos) s_tile[(size_t)qi * kBc + kj] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }

                // Online softmax update, block-local max/sum then merge
                // with the running (m, l, acc) — this is the core FA trick.
                for (int qi = 0; qi < br; ++qi) {
                    float* srow = &s_tile[(size_t)qi * kBc];

                    float blk_max = -std::numeric_limits<float>::infinity();
                    for (int kj = 0; kj < bc; ++kj) blk_max = std::max(blk_max, srow[kj]);
                    if (blk_max == -std::numeric_limits<float>::infinity()) continue; // fully masked block

                    float m_prev = m[qi];
                    float m_new  = std::max(m_prev, blk_max);

                    float blk_sum = 0.f;
                    for (int kj = 0; kj < bc; ++kj) {
                        float p = std::exp(srow[kj] - m_new);
                        srow[kj] = p;               // reuse s_tile as the exp'd weights
                        blk_sum += p;
                    }

                    // Rescale previous accumulators to the new max.
                    float correction = (m_prev == -std::numeric_limits<float>::infinity())
                                        ? 0.f : std::exp(m_prev - m_new);
                    l[qi] = l[qi] * correction + blk_sum;

                    float* arow = &acc[(size_t)qi * d_h];
                    if (correction != 1.f)
                        for (int d = 0; d < d_h; ++d) arow[d] *= correction;

                    for (int kj = 0; kj < bc; ++kj) {
                        float w = srow[kj];
                        float const* vrow = V + (size_t)(k0 + kj) * d_h;
                        for (int d = 0; d < d_h; ++d) arow[d] += w * vrow[d];
                    }

                    m[qi] = m_new;
                }
            }

            // Normalize and write out this query tile.
            for (int qi = 0; qi < br; ++qi) {
                float inv_l = (l[qi] > 0.f) ? (1.f / l[qi]) : 0.f;
                float const* arow = &acc[(size_t)qi * d_h];
                float* orow = O + (size_t)(q0 + qi) * d_h;
                for (int d = 0; d < d_h; ++d) orow[d] = arow[d] * inv_l;
            }
        }
    }

private:
    // Reshape [T, n_h * d_h] -> [n_h, T, d_h]  (new allocation)
    static Tensor reshape_heads(Tensor const& x, int T, int n_h, int d_h) {
        // x data is [T, n_h * d_h] row-major
        // We want [n_h, T, d_h] so element [h,t,j] = x[t, h*d_h + j]
        Tensor out({n_h, T, d_h});
        float const* src = x.data_ptr();
        float*       dst = out.data_ptr();
        int stride = n_h * d_h;
        for (int h = 0; h < n_h; ++h)
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d_h; ++j)
                    dst[h * T * d_h + t * d_h + j] = src[t * stride + h * d_h + j];
        return out;
    }

    // Expand K or V from [n_kv, T, d_h] to [n_heads, T, d_h] by repeating
    static Tensor expand_kv_heads(Tensor const& kv, int rep) {
        int n_kv = kv.dim(0), T = kv.dim(1), d_h = kv.dim(2);
        int n_h  = n_kv * rep;
        Tensor out({n_h, T, d_h});
        float const* src = kv.data_ptr();
        float*       dst = out.data_ptr();
        for (int k = 0; k < n_kv; ++k)
            for (int r = 0; r < rep; ++r) {
                int h = k * rep + r;
                std::copy(src + k * T * d_h,
                          src + (k + 1) * T * d_h,
                          dst + h * T * d_h);
            }
        return out;
    }

    // Scaled dot-product attention across heads
    // Q:  [n_h, T, d_h]   K,V: [n_h, ctx, d_h]  -> [n_h, T, d_h]
    //
    // GPU path: dispatches through the backend's fused flash-attention
    // kernel (active_backend().sdp_attention() — SYCLBackend::dispatch_attention
    // / OpenCLBackend::dispatch_attention), which fuses QK^T + online softmax
    // + weighted-V into a single kernel launch and never materializes the
    // full [T, ctx_len] score matrix, mirroring the CPU tiling below. Falls
    // back to the non-fused gemm/mask/softmax/gemm sequence (still GPU-
    // routed via compute::) when the fused kernel can't handle this call —
    // either because the shape falls outside what dispatch_attention
    // supports (see its own guards) or because attention_multiplier (MuP)
    // is set: the fused kernels hardcode the standard 1/sqrt(d_head) scale
    // and have no parameter for a caller-supplied override.
    //
    // CPU path: flash-attention style tiling (flash_attn_head below).
    // Never materializes the full [T, ctx_len] score matrix — bounds
    // extra memory to O(kBr * kBc) per thread regardless of context
    // length, which matters once ctx_len gets into the thousands during
    // prefill (a naive [T,ctx_len] f32 buffer at T=ctx_len=8192 is 256MB,
    // per head, per layer, times however many heads run concurrently).
    Tensor sdp_heads(Tensor const& Q, Tensor const& K, Tensor const& V,
                     int T, int ctx_len, bool causal) const {
        int n_h = n_heads, d_h = d_head;
        // attention_multiplier (MuP, IBM Granite 4.0+) REPLACES the standard
        // 1/sqrt(d_head) scaling entirely — confirmed from
        // modeling_granitemoehybrid.py: "self.scaling = config.attention_multiplier
        // # Only diff with llama". The two are never combined.
        // Sentinel 0.0f means "not set by config" -> fall back to standard scaling.
        bool has_custom_scale = (attention_multiplier != 0.0f);
        float scale = has_custom_scale ? attention_multiplier : 1.f / std::sqrt((float)d_h);

        if (compute::gpu_active()) {
            // The fused kernel's contract (see SYCLBackend/OpenCLBackend
            // dispatch_attention) is the standard 1/sqrt(d_head) scale only,
            // so a custom attention_multiplier must skip it entirely and use
            // the gemm-based path below instead.
            if (!has_custom_scale) {
                Tensor out = vlrf::backend::active_backend().sdp_attention(Q, K, V, causal);
                // active_backend().sdp_attention() itself falls back to
                // ops::sdp_attention (CPU) on any dispatch failure, so `out`
                // is always valid here — no separate failure path to handle.
                if (out.shape == Shape({n_h, T, d_h})) return out;
                // Defensive: an unexpected shape (e.g. a future backend
                // returning something other than [n_h,T,d_h]) falls through
                // to the explicit gemm-based path rather than propagating a
                // shape mismatch to the caller.
            }

            // GPU backends (sycl_backend.hpp / OpenCLBackend's dispatch_*)
            // use a single command queue / SYCL queue with no internal
            // locking. Calling compute::gemm/softmax_inplace concurrently
            // from multiple CPU threads (as ops::parallel_for would do)
            // races on that queue. Since the actual math work is offloaded
            // to the GPU here anyway, running the per-head loop
            // sequentially on the calling thread is the correct tradeoff —
            // losing CPU-side head parallelism is expected once the GEMMs
            // themselves are no longer running on CPU.
            Tensor out({n_h, T, d_h});
            auto per_head_gpu = [&](int64_t h) {
                Tensor Qh = Tensor::from_ptr(
                    const_cast<float*>(Q.data_ptr()) + h * T * d_h, {T, d_h});
                Tensor Kh = Tensor::from_ptr(
                    const_cast<float*>(K.data_ptr()) + h * ctx_len * d_h, {ctx_len, d_h});
                Tensor Vh = Tensor::from_ptr(
                    const_cast<float*>(V.data_ptr()) + h * ctx_len * d_h, {ctx_len, d_h});

                Tensor Kht = ops::transpose_last2(Kh);     // [d_h, ctx_len] — CPU (no IBackend transpose)
                Tensor scores({T, ctx_len});
                compute::gemm(Qh, Kht, scores, scale, 0.f, false, false);

                if (causal) {
                    float* ps = scores.data_ptr();
                    int past = ctx_len - T;
                    for (int qi = 0; qi < T; ++qi)
                        for (int ki = qi + past + 1; ki < ctx_len; ++ki)
                            ps[qi * ctx_len + ki] = -1e30f;
                }

                compute::softmax_inplace(scores);
                Tensor outh = Tensor::from_ptr(out.data_ptr() + h * T * d_h, {T, d_h});
                compute::gemm(scores, Vh, outh, 1.f, 0.f, false, false);
            };
            for (int64_t h = 0; h < n_h; ++h) per_head_gpu(h);
            return out;
        }

        Tensor out({n_h, T, d_h});
        // --- CPU: tiled / online-softmax attention, per head, in parallel ---
        ops::parallel_for(n_h, [&](int64_t h0, int64_t h1) {
            for (int64_t h = h0; h < h1; ++h) {
                float const* Qh = Q.data_ptr() + h * T * d_h;
                float const* Kh = K.data_ptr() + h * ctx_len * d_h;
                float const* Vh = V.data_ptr() + h * ctx_len * d_h;
                float*       Oh = out.data_ptr() + h * T * d_h;
                flash_attn_head(Qh, Kh, Vh, Oh, T, ctx_len, d_h, scale, causal);
            }
        });
        return out;
    }


    // Merge [n_h, T, d_h] -> [T, n_h * d_h]
    static void merge_heads(Tensor const& x, Tensor& dst, int T, int n_h, int d_h) {
        float const* src = x.data_ptr();
        float*       out = dst.data_ptr();
        int D = n_h * d_h;
        for (int h = 0; h < n_h; ++h)
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d_h; ++j)
                    out[t * D + h * d_h + j] = src[h * T * d_h + t * d_h + j];
    }
};

} // namespace vlrf
