// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Flash-attention SDPA (attention.hpp: flash_attn_head)  (test_flash_attention.cpp)
//
// Two levels of coverage:
//   1. Unit-level: flash_attn_head() vs. an independent naive dense
//      softmax(QK^T*scale + causal_mask) @ V reference, directly on raw
//      buffers — covers tile-boundary cases (T/ctx_len not multiples of
//      the 64-row tile size), T==1 decode, and KV-cache continuation
//      (T < ctx_len, causal offset).
//   2. Integration-level: a real AttentionBlock bound to random weights,
//      run once over a full prompt (prefill) vs. token-by-token through a
//      KVCache (incremental decode) — the two must agree, since that's
//      the actual invariant the rest of the engine depends on.
// =============================================================================
#include "attention.hpp"
#include "kv_cache.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace vlrf;

static int g_pass = 0, g_fail = 0;

#define EXPECT(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n"; \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: |" #a " - " #b "| = " \
                  << std::abs(_a-_b) << " > " << (tol) << "  (" << _a << " vs " << _b << ")\n"; \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

// -----------------------------------------------------------------------------
// 1. Unit-level: flash_attn_head vs. naive dense reference
// -----------------------------------------------------------------------------
static void dense_ref(std::vector<float> const& Q, std::vector<float> const& K,
                       std::vector<float> const& V, std::vector<float>& O,
                       int T, int ctx_len, int d_h, float scale, bool causal) {
    int past = ctx_len - T;
    std::vector<float> scores(ctx_len);
    for (int qi = 0; qi < T; ++qi) {
        int qpos = qi + past;
        float mx = -std::numeric_limits<float>::infinity();
        for (int kj = 0; kj < ctx_len; ++kj) {
            float dot = 0.f;
            for (int d = 0; d < d_h; ++d) dot += Q[qi*d_h+d]*K[kj*d_h+d];
            float s = dot * scale;
            if (causal && kj > qpos) s = -std::numeric_limits<float>::infinity();
            scores[kj] = s;
            mx = std::max(mx, s);
        }
        double sum = 0.0;
        for (int kj = 0; kj < ctx_len; ++kj) {
            scores[kj] = std::exp(scores[kj] - mx);
            sum += scores[kj];
        }
        for (int d = 0; d < d_h; ++d) {
            double acc = 0.0;
            for (int kj = 0; kj < ctx_len; ++kj) acc += scores[kj]*V[kj*d_h+d];
            O[qi*d_h+d] = (float)(acc / sum);
        }
    }
}

static void run_flash_vs_dense(int T, int ctx_len, int d_h, bool causal, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> Q((size_t)T*d_h), K((size_t)ctx_len*d_h), V((size_t)ctx_len*d_h);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    float scale = 1.f / std::sqrt((float)d_h);
    std::vector<float> O_fa((size_t)T*d_h), O_ref((size_t)T*d_h);

    AttentionBlock::flash_attn_head(Q.data(), K.data(), V.data(), O_fa.data(),
                                     T, ctx_len, d_h, scale, causal);
    dense_ref(Q, K, V, O_ref, T, ctx_len, d_h, scale, causal);

    for (size_t i = 0; i < O_fa.size(); ++i)
        EXPECT_NEAR(O_fa[i], O_ref[i], 1e-4f);
}

void test_flash_attn_exact_tile_multiple() {
    run_flash_vs_dense(64, 64, 32, false, 1);
    run_flash_vs_dense(128, 128, 32, true, 2);
}

void test_flash_attn_tile_tail() {
    run_flash_vs_dense(37, 37, 8, true, 5);       // odd sizes, causal
    run_flash_vs_dense(130, 130, 64, true, 6);    // spans 3 tiles of 64
}

void test_flash_attn_decode_single_token() {
    run_flash_vs_dense(1, 1, 16, false, 3);
    run_flash_vs_dense(1, 200, 16, false, 4);     // decode against a long KV cache
    run_flash_vs_dense(1, 1, 4, true, 8);
}

void test_flash_attn_kv_continuation() {
    // T < ctx_len with causal=true: the shape produced when a KV cache
    // already holds `ctx_len - T` prior tokens and this forward() call
    // processes `T` new ones (e.g. speculative decode / chunked prefill).
    run_flash_vs_dense(200, 500, 64, true, 7);
}

void test_flash_attn_large_head_dim_no_mask() {
    run_flash_vs_dense(300, 300, 128, false, 9);
}

// -----------------------------------------------------------------------------
// 2. Integration-level: AttentionBlock prefill vs. incremental KV-cache decode
// -----------------------------------------------------------------------------
static void fill_random(Tensor& t, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (int64_t i = 0; i < t.numel(); ++i) t.data_ptr()[i] = nd(rng);
}

void test_attention_block_prefill_matches_incremental_decode() {
    ModelConfig cfg = ModelConfig::tiny();
    cfg.pos_embedding = PosEmbedding::ROTARY;
    cfg.use_causal_mask = true;
    int d_model = cfg.d_model, n_heads = cfg.n_heads, n_kv_heads = cfg.n_kv_heads;
    int d_head = cfg.d_head();
    int d_kv = n_kv_heads * d_head;

    WeightMap wm;
    std::mt19937 rng(42);
    auto make = [&](std::string name, std::vector<int32_t> shape) -> Tensor& {
        Tensor t{Shape(shape)};
        fill_random(t, rng);
        wm.tensors.emplace(name, std::move(t));
        return wm.get(name);
    };
    make("layer.q_proj.weight", {d_model, d_model});
    make("layer.k_proj.weight", {d_kv, d_model});
    make("layer.v_proj.weight", {d_kv, d_model});
    make("layer.o_proj.weight", {d_model, d_model});

    AttentionBlock attn;
    attn.bind(wm, "layer", cfg);

    int T = 9; // prompt length; deliberately not a multiple of the 64-row tile
    Tensor prompt({T, d_model});
    fill_random(prompt, rng);

    // --- Full prefill in one shot, no cache ---
    Tensor out_prefill = attn.forward(prompt, nullptr);

    // --- Token-by-token incremental decode through a KVCache ---
    KVCache cache(cfg.max_seq_len, n_kv_heads, d_head, KVCacheDType::F32);
    Tensor out_incremental({T, d_model});
    for (int t = 0; t < T; ++t) {
        Tensor tok({1, d_model});
        std::copy(prompt.data_ptr() + (size_t)t * d_model,
                  prompt.data_ptr() + (size_t)(t + 1) * d_model,
                  tok.data_ptr());
        Tensor step_out = attn.forward(tok, &cache);
        std::copy(step_out.data_ptr(), step_out.data_ptr() + d_model,
                  out_incremental.data_ptr() + (size_t)t * d_model);
    }

    // Prefill (masked, all-at-once) and incremental decode (unmasked
    // single-token steps against a growing cache) must produce the same
    // result per position — this is the invariant the flash-attention
    // rewrite has to preserve, since kv_cache-based generation depends on
    // it matching prefill bit-for-bit (up to float rounding).
    for (int64_t i = 0; i < out_prefill.numel(); ++i)
        EXPECT_NEAR(out_prefill.data_ptr()[i], out_incremental.data_ptr()[i], 1e-3f);
}

int main() {
    test_flash_attn_exact_tile_multiple();
    test_flash_attn_tile_tail();
    test_flash_attn_decode_single_token();
    test_flash_attn_kv_continuation();
    test_flash_attn_large_head_dim_no_mask();
    test_attention_block_prefill_matches_incremental_decode();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
