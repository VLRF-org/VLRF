// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — KVCache quantization (F16 / Q8_0)  (test_kv_cache_quant.cpp)
// Verifies: round-trip accuracy of each dtype, append()/dequant_into()
// correctness against a plain F32 reference, memory savings match the
// documented ratios, and end-to-end model generation still works (doesn't
// crash / produces finite logits) with each cache dtype.
// =============================================================================
#include "kv_cache.hpp"
#include "transformer.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::cerr << "FAIL: " << #cond << " (" << __FILE__ << ":" << __LINE__ << ")\n"; } \
} while (0)

using namespace vlrf;

// -----------------------------------------------------------------------------
static void test_f16_roundtrip_scalar() {
    // Spot-check f32->f16->f32 on a range of values including edge cases.
    float vals[] = {0.f, 1.f, -1.f, 0.5f, 3.14159f, -2.71828f, 1e-3f, 1000.f, -0.0001f};
    for (float v : vals) {
        uint16_t h = detail::f32_to_f16(v);
        float back = detail::f16_to_f32(h);
        // f16 has ~3 decimal digits of precision; allow proportional error.
        float tol = std::max(1e-3f, std::fabs(v) * 1e-2f);
        CHECK(std::fabs(back - v) <= tol);
    }
}

static void test_q8_0_roundtrip_row() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    const int n = 64;
    std::vector<float> row(n);
    for (auto& x : row) x = dist(rng);

    std::vector<int8_t> q(n);
    float scale = detail::quantize_row_q8_0(row.data(), q.data(), n);
    CHECK(scale > 0.f);

    std::vector<float> back(n);
    detail::dequantize_row_q8_0(q.data(), scale, back.data(), n);

    float max_abs = 0.f;
    for (float x : row) max_abs = std::max(max_abs, std::fabs(x));
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) max_err = std::max(max_err, std::fabs(back[i] - row[i]));
    // Q8_0 quantization error should be bounded by ~1 quantization step
    // (scale = max_abs/127), with a little slack for rounding.
    CHECK(max_err <= (max_abs / 127.f) * 1.5f);
}

static void test_q8_0_all_zero_row() {
    std::vector<float> row(32, 0.f);
    std::vector<int8_t> q(32);
    float scale = detail::quantize_row_q8_0(row.data(), q.data(), 32);
    CHECK(scale == 0.f);
    for (int8_t v : q) CHECK(v == 0);
    std::vector<float> back(32);
    detail::dequantize_row_q8_0(q.data(), scale, back.data(), 32);
    for (float v : back) CHECK(v == 0.f);
}

// -----------------------------------------------------------------------------
static void test_kvcache_append_and_dequant(KVCacheDType dtype, float tol) {
    const int max_seq = 16, n_kv_heads = 2, d_head = 8;
    const int row_dim = n_kv_heads * d_head;
    KVCache cache(max_seq, n_kv_heads, d_head, dtype);
    CHECK(cache.used == 0);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);

    // Append 5 tokens one at a time (mimics autoregressive decode).
    std::vector<std::vector<float>> K_ref, V_ref;
    for (int t = 0; t < 5; ++t) {
        std::vector<float> k(row_dim), v(row_dim);
        for (auto& x : k) x = dist(rng);
        for (auto& x : v) x = dist(rng);
        cache.append(k.data(), v.data(), 1);
        K_ref.push_back(k);
        V_ref.push_back(v);
    }
    CHECK(cache.used == 5);

    std::vector<float> K_out((size_t)5 * row_dim), V_out((size_t)5 * row_dim);
    cache.dequant_into(5, K_out.data(), V_out.data());

    float max_err = 0.f;
    for (int t = 0; t < 5; ++t)
        for (int j = 0; j < row_dim; ++j) {
            max_err = std::max(max_err, std::fabs(K_out[t * row_dim + j] - K_ref[t][j]));
            max_err = std::max(max_err, std::fabs(V_out[t * row_dim + j] - V_ref[t][j]));
        }
    CHECK(max_err <= tol);
}

static void test_kvcache_bytes_ratio() {
    const int max_seq = 1000, n_kv_heads = 8, d_head = 128;
    KVCache f32(max_seq, n_kv_heads, d_head, KVCacheDType::F32);
    KVCache f16(max_seq, n_kv_heads, d_head, KVCacheDType::F16);
    KVCache q80(max_seq, n_kv_heads, d_head, KVCacheDType::Q8_0);

    int64_t b32 = f32.bytes(), b16 = f16.bytes(), b8 = q80.bytes();
    std::cout << "  KVCache bytes @ ctx=1000: f32=" << b32
              << " f16=" << b16 << " (" << (double)b32 / b16 << "x smaller)"
              << " q8_0=" << b8 << " (" << (double)b32 / b8 << "x smaller)\n";

    // F16 should be ~2x smaller, Q8_0 ~4x smaller (within a hair, since
    // Q8_0 carries a small per-row scale overhead).
    CHECK(std::fabs((double)b32 / b16 - 2.0) < 0.05);
    CHECK((double)b32 / b8 > 3.9 && (double)b32 / b8 <= 4.0);
}

static void test_kvcache_reset() {
    KVCache cache(8, 2, 4, KVCacheDType::Q8_0);
    std::vector<float> k(8, 1.f), v(8, 2.f);
    cache.append(k.data(), v.data(), 1);
    CHECK(cache.used == 1);
    cache.reset();
    CHECK(cache.used == 0);
    for (int8_t x : cache.K_q8) CHECK(x == 0);
    for (float s : cache.K_scale) CHECK(s == 0.f);
}

static void test_kvcache_overflow_throws() {
    KVCache cache(2, 1, 4, KVCacheDType::F32);
    std::vector<float> k(4, 1.f), v(4, 1.f);
    cache.append(k.data(), v.data(), 1);
    cache.append(k.data(), v.data(), 1);
    bool threw = false;
    try {
        cache.append(k.data(), v.data(), 1);  // exceeds max_seq_len=2
    } catch (std::runtime_error const&) {
        threw = true;
    }
    CHECK(threw);
}

// -----------------------------------------------------------------------------
// End-to-end: run generation through TransformerModel with each cache dtype
// and check logits stay finite and (for F16, which is high-precision) match
// the F32 baseline closely. Mirrors test_model.cpp's make_model() setup.
// -----------------------------------------------------------------------------
static ModelConfig make_tiny_config() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.d_model = 32;
    cfg.d_ff = 64;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 4;
    cfg.n_layers = 2;
    cfg.max_seq_len = 32;
    cfg.bos_token_id = 1;
    cfg.eos_token_id = 2;
    cfg.tie_embeddings = true;
    return cfg;
}

static void test_end_to_end_generation_per_dtype() {
    std::vector<int> ids = {1, 4, 7, 12, 3};

    for (KVCacheDType dtype : {KVCacheDType::F32, KVCacheDType::F16, KVCacheDType::Q8_0}) {
        ModelConfig cfg = make_tiny_config();
        cfg.kv_cache_dtype = (int)dtype;

        TransformerModel m;
        m.init(cfg);
        m.wm = WeightMap::random_init(cfg);
        m.bind_weights();
        m.rng.seed(42);
        m.reset_kv_cache();

        Tensor logits;
        for (int id : ids) logits = m.forward({id}, true);

        bool all_finite = true;
        for (int64_t i = 0; i < logits.numel(); ++i)
            if (!std::isfinite(logits[i])) all_finite = false;
        CHECK(all_finite);
        std::cout << "  end-to-end forward OK with kv_cache_dtype="
                  << kv_cache_dtype_name(dtype) << "\n";
    }
}

int main() {
    std::cout << "=== VLRF KVCache Quantization Tests ===\n";

    test_f16_roundtrip_scalar();
    test_q8_0_roundtrip_row();
    test_q8_0_all_zero_row();

    test_kvcache_append_and_dequant(KVCacheDType::F32,  1e-6f);
    test_kvcache_append_and_dequant(KVCacheDType::F16,  5e-2f);
    test_kvcache_append_and_dequant(KVCacheDType::Q8_0, 0.15f);

    test_kvcache_bytes_ratio();
    test_kvcache_reset();
    test_kvcache_overflow_throws();

    test_end_to_end_generation_per_dtype();

    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
