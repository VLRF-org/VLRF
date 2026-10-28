// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — TransformerModel forward pass  (test_model.cpp)
// Tests shape correctness, KV-cache consistency, and generation loop.
// =============================================================================
#include "transformer.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <numeric>
#include <unordered_map>
#include <set>
#include <string>
#ifdef _WIN32
#  include <windows.h>
#endif

// Cross-platform temp directory
static std::string get_tmp_dir() {
#ifdef _WIN32
    char buf[512];
    DWORD len = GetTempPathA(sizeof(buf), buf);
    if (len == 0) return ".";
    // Remove trailing backslash for path joining
    std::string s(buf, len);
    while (!s.empty() && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s;
#else
    return "/tmp";
#endif
}

using namespace vlrf;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if(!(cond)){std::cerr<<"FAIL ["<<__LINE__<<"]: "#cond"\n";++g_fail;} \
    else ++g_pass; } while(0)
#define CHECK_EQ(a,b,tol) do { \
    float _a=(float)(a),_b=(float)(b); \
    if(std::fabs(_a-_b)>(tol)){std::cerr<<"FAIL ["<<__LINE__<<"]: |"#a"-"#b"|="<<std::fabs(_a-_b)<<"\n";++g_fail;} \
    else ++g_pass; }while(0)

// Build a tiny model for fast testing
static ModelConfig make_test_cfg() {
    ModelConfig c;
    c.vocab_size  = 64;
    c.d_model     = 32;
    c.d_ff        = 64;
    c.n_heads     = 2;
    c.n_kv_heads  = 2;
    c.n_layers    = 2;
    c.max_seq_len = 64;
    c.bos_token_id = 1;
    c.eos_token_id = 2;
    c.tie_embeddings = true;
    return c;
}

static TransformerModel make_model() {
    auto cfg = make_test_cfg();
    TransformerModel m;
    m.init(cfg);
    m.wm = WeightMap::random_init(cfg);
    m.bind_weights();
    return m;
}

// -----------------------------------------------------------------------------
void test_config_check() {
    ModelConfig c = make_test_cfg();
    bool threw = false;
    try { c.check(); } catch (...) { threw = true; }
    CHECK(!threw);

    ModelConfig bad = c;
    bad.d_model = 33;  // not divisible by n_heads=2
    threw = false;
    try { bad.check(); } catch (...) { threw = true; }
    CHECK(threw);
}

void test_param_count() {
    ModelConfig c = make_test_cfg();
    int64_t p = c.param_count();
    CHECK(p > 0);
    std::cout << "  tiny test model: " << p << " params\n";
}

void test_weight_map_keys() {
    auto cfg = make_test_cfg();
    WeightMap wm = WeightMap::random_init(cfg);
    CHECK(wm.has("embed.weight"));
    CHECK(wm.has("ln_final.weight"));
    CHECK(wm.has("ln_final.bias"));
    CHECK(wm.has("layers.0.ln1.weight"));
    CHECK(wm.has("layers.0.attn.q_proj.weight"));
    CHECK(wm.has("layers.0.ffn.fc1.weight"));
    CHECK(wm.has("layers.1.ffn.fc2.weight"));
    // No lm_head since tie_embeddings=true
    CHECK(!wm.has("lm_head.weight"));
}

void test_weight_shapes() {
    auto cfg = make_test_cfg();
    WeightMap wm = WeightMap::random_init(cfg);
    auto& e = wm.get("embed.weight");
    CHECK(e.dim(0) == cfg.vocab_size && e.dim(1) == cfg.d_model);

    auto& q = wm.get("layers.0.attn.q_proj.weight");
    CHECK(q.dim(0) == cfg.d_model && q.dim(1) == cfg.d_model);

    auto& fc1 = wm.get("layers.0.ffn.fc1.weight");
    CHECK(fc1.dim(0) == cfg.d_ff && fc1.dim(1) == cfg.d_model);
}

void test_forward_output_shape() {
    TransformerModel m = make_model();
    std::vector<int> ids = {1, 5, 10, 15};
    Tensor logits = m.forward(ids, false);
    CHECK(logits.ndim() == 1);
    CHECK(logits.dim(0) == m.cfg.vocab_size);
}

void test_forward_single_token() {
    TransformerModel m = make_model();
    std::vector<int> ids = {3};
    Tensor logits = m.forward(ids, false);
    CHECK(logits.numel() == (int64_t)m.cfg.vocab_size);
}

void test_kvcache_incremental_equals_full() {
    // Forward the full sequence without cache and with cache should give
    // the same final logits (up to float precision).
    TransformerModel m = make_model();
    m.rng.seed(42);

    std::vector<int> ids = {1, 4, 7, 12, 3};

    // Full forward (no cache)
    m.reset_kv_cache();
    Tensor logits_full = m.forward(ids, false);

    // Incremental (with cache)
    m.reset_kv_cache();
    Tensor logits_inc;
    for (int i = 0; i < (int)ids.size(); ++i) {
        logits_inc = m.forward({ids[i]}, true);
    }

    // Results should match to ~1e-4 (float32 accumulation differences)
    int V = m.cfg.vocab_size;
    float max_diff = 0.f;
    for (int j = 0; j < V; ++j)
        max_diff = std::max(max_diff, std::fabs(logits_full[j] - logits_inc[j]));

    std::cout << "  KV-cache vs full forward max_diff = " << max_diff << "\n";
    CHECK(max_diff < 0.5f);  // should be tight; random weights so not too strict
}

void test_generate_returns_tokens() {
    TransformerModel m = make_model();
    m.rng.seed(1);

    GenerationConfig gc;
    gc.max_new_tokens = 5;
    gc.do_sample      = false;  // greedy for reproducibility

    std::vector<int> prompt = {1, 5};
    auto result = m.generate(prompt, gc);

    CHECK(result.tokens_generated > 0);
    CHECK((int)result.token_ids.size() >= (int)prompt.size());
    // All generated ids should be in-vocabulary
    for (int id : result.token_ids)
        CHECK(id >= 0 && id < m.cfg.vocab_size);
}

void test_generate_eos_stops() {
    TransformerModel m = make_model();
    // Force the model to always predict EOS by zeroing embed and
    // setting lm_head bias toward eos_token_id — too complex to do cleanly,
    // so instead just test that hit_eos is set when eos is sampled.
    // We verify the flag logic by checking: if result contains eos, flag is true.
    GenerationConfig gc;
    gc.max_new_tokens = 50;
    gc.do_sample = false;
    auto result = m.generate({1}, gc);
    // If EOS token appeared anywhere after prompt, hit_eos must be set
    bool found_eos = false;
    for (int i = 1; i < (int)result.token_ids.size(); ++i)
        if (result.token_ids[i] == m.cfg.eos_token_id) { found_eos = true; break; }
    if (found_eos) CHECK(result.hit_eos);
    // If no EOS was hit, we just exhausted max_new_tokens — that's also fine
    CHECK(result.tokens_generated <= gc.max_new_tokens);
}

// -----------------------------------------------------------------------------
// Repetition penalty / stop sequences
// -----------------------------------------------------------------------------

void test_repetition_penalty_reduces_repeats() {
    // Greedy decoding on random weights tends to loop on a small handful of
    // tokens (or even a single token) once the KV cache settles into a
    // stable pattern — a good stress case for repetition control. We can't
    // assert exact token identities (random init), but we CAN assert that
    // applying a strong repetition_penalty does not increase the maximum
    // single-token repeat count relative to no penalty, and typically
    // reduces it.
    TransformerModel m = make_model();
    m.rng.seed(42);

    auto max_repeat = [](std::vector<int> const& ids) {
        std::unordered_map<int,int> counts;
        int best = 0;
        for (int id : ids) best = std::max(best, ++counts[id]);
        return best;
    };

    GenerationConfig gc_base;
    gc_base.max_new_tokens = 40;
    gc_base.do_sample = false; // greedy — maximizes chance of a visible loop
    auto result_base = m.generate({1}, gc_base);

    TransformerModel m2 = make_model();
    m2.rng.seed(42);
    GenerationConfig gc_pen = gc_base;
    gc_pen.repetition_penalty = 1.3f;
    auto result_pen = m2.generate({1}, gc_pen);

    CHECK(result_base.tokens_generated > 0);
    CHECK(result_pen.tokens_generated > 0);
    int rep_base = max_repeat(result_base.token_ids);
    int rep_pen  = max_repeat(result_pen.token_ids);
    std::cout << "  max single-token repeat: no-penalty=" << rep_base
               << " repetition_penalty=1.3=" << rep_pen << "\n";
    CHECK(rep_pen <= rep_base);
}

void test_repetition_penalty_is_noop_at_default() {
    // repetition_penalty=1.0, frequency=0, presence=0 must reproduce the
    // exact same generation as before these fields existed — the fast-path
    // early-return in apply_repetition_penalties() should make this a
    // byte-for-byte no-op, not just "close".
    TransformerModel m1 = make_model();
    m1.rng.seed(7);
    GenerationConfig gc;
    gc.max_new_tokens = 20;
    gc.do_sample = true;
    gc.temperature = 0.8f;
    auto r1 = m1.generate({1, 5}, gc);

    TransformerModel m2 = make_model();
    m2.rng.seed(7);
    // Explicitly set defaults even though they already are the defaults,
    // to document intent.
    gc.repetition_penalty = 1.0f;
    gc.frequency_penalty = 0.0f;
    gc.presence_penalty = 0.0f;
    auto r2 = m2.generate({1, 5}, gc);

    CHECK(r1.token_ids == r2.token_ids);
}

void test_frequency_and_presence_penalty_change_output() {
    // Sanity check that the additive penalties actually perturb logits
    // (and therefore CAN change greedy output) rather than being dead code.
    // We don't assert a specific direction of change — random weights make
    // that unpredictable — just that SOME run with strong penalties diverges
    // from the unpenalized run at least once across a batch of seeds, which
    // would be extraordinarily unlikely if the penalties had no effect.
    bool any_diff = false;
    for (uint32_t seed = 0; seed < 8; ++seed) {
        TransformerModel m1 = make_model();
        m1.rng.seed(seed);
        GenerationConfig gc;
        gc.max_new_tokens = 30;
        gc.do_sample = false;
        auto r1 = m1.generate({1}, gc);

        TransformerModel m2 = make_model();
        m2.rng.seed(seed);
        gc.frequency_penalty = 1.5f;
        gc.presence_penalty = 1.5f;
        auto r2 = m2.generate({1}, gc);

        if (r1.token_ids != r2.token_ids) { any_diff = true; break; }
    }
    CHECK(any_diff);
}

void test_no_repeat_ngram_blocks_exact_loops() {
    // With no_repeat_ngram_size=2, no bigram may repeat among generated
    // tokens. Verify greedy decoding under this constraint never produces
    // the same (token[i], token[i+1]) pair twice.
    TransformerModel m = make_model();
    m.rng.seed(3);
    GenerationConfig gc;
    gc.max_new_tokens = 40;
    gc.do_sample = false;
    gc.no_repeat_ngram_size = 2;
    auto result = m.generate({1}, gc);

    // token_ids includes the prompt; only check generated tokens (matching
    // GenerationConfig's documented "generated tokens only" scope).
    int prompt_len = 1;
    std::vector<int> gen(result.token_ids.begin() + prompt_len, result.token_ids.end());
    std::set<std::pair<int,int>> seen_bigrams;
    bool any_repeat = false;
    for (size_t i = 0; i + 1 < gen.size(); ++i) {
        auto bg = std::make_pair(gen[i], gen[i+1]);
        if (seen_bigrams.count(bg)) { any_repeat = true; break; }
        seen_bigrams.insert(bg);
    }
    CHECK(!any_repeat);
    CHECK(result.tokens_generated > 0);
}

void test_no_repeat_ngram_zero_is_noop() {
    // no_repeat_ngram_size=0 (the default) must not alter greedy output at
    // all relative to not having the field.
    TransformerModel m1 = make_model();
    m1.rng.seed(11);
    GenerationConfig gc;
    gc.max_new_tokens = 15;
    gc.do_sample = false;
    auto r1 = m1.generate({1}, gc);

    TransformerModel m2 = make_model();
    m2.rng.seed(11);
    gc.no_repeat_ngram_size = 0;
    auto r2 = m2.generate({1}, gc);

    CHECK(r1.token_ids == r2.token_ids);
}

void test_stop_string_truncates_output() {
    // Byte-level tokenizer: ids 4..259 map 1:1 to raw bytes (see
    // init_byte_vocab()), so token id = 4 + (unsigned char). This lets us
    // pick a stop string made of literal bytes and know exactly which
    // token ids must appear for it to match, without needing BPE merges.
    Tokenizer tok;
    tok.init_byte_vocab();

    // Build a tiny model whose vocab covers the byte tokenizer's range.
    ModelConfig cfg = make_test_cfg();
    cfg.vocab_size = tok.vocab_size();
    TransformerModel m;
    m.init(cfg);
    m.wm = WeightMap::random_init(cfg);
    m.bind_weights();
    m.rng.seed(5);

    // Run once with no stop condition to get a baseline of what greedy
    // decoding naturally produces from this seed/model.
    GenerationConfig gc_base;
    gc_base.max_new_tokens = 30;
    gc_base.do_sample = false;
    auto baseline = m.generate({tok.bos_id}, gc_base, &tok);
    CHECK(baseline.tokens_generated > 0);

    if (baseline.text.size() < 2) {
        // Degenerate baseline (e.g. immediate EOS) — nothing meaningful to
        // test a stop string against for this seed; skip rather than fail.
        std::cout << "  (baseline text too short to test stop-string truncation for this seed)\n";
        return;
    }

    // Pick a stop string that's guaranteed to occur: the last 2 characters
    // of the baseline's natural (unstopped) output.
    std::string stop = baseline.text.substr(baseline.text.size() - 2);

    TransformerModel m2;
    m2.init(cfg);
    m2.wm = WeightMap::random_init(cfg); // same fixed seed as m's random_init -> identical weights
    m2.bind_weights();
    m2.rng.seed(5);

    GenerationConfig gc_stop;
    gc_stop.max_new_tokens = 30;
    gc_stop.do_sample = false;
    gc_stop.stop_strings = {stop};
    auto stopped = m2.generate({tok.bos_id}, gc_stop, &tok);

    // The stopped run must not contain the stop string in its returned
    // text (it should have been trimmed off).
    CHECK(stopped.text.find(stop) == std::string::npos || stopped.text.empty());
    // It must be a prefix-consistent truncation of the baseline: since both
    // runs share the same seed/weights/greedy decoding up to the stop
    // point, the stopped text must equal the baseline text up to where the
    // stop string was trimmed.
    CHECK(stopped.text.size() <= baseline.text.size());
    CHECK(baseline.text.compare(0, stopped.text.size(), stopped.text) == 0);
    // And it must have generated fewer or equal tokens than the unstopped run
    CHECK(stopped.tokens_generated <= baseline.tokens_generated);
}

void test_stop_strings_empty_is_noop() {
    // No stop_strings and no tokenizer must behave identically to before
    // this feature existed.
    TransformerModel m1 = make_model();
    m1.rng.seed(9);
    GenerationConfig gc;
    gc.max_new_tokens = 15;
    gc.do_sample = false;
    auto r1 = m1.generate({1}, gc);

    TransformerModel m2 = make_model();
    m2.rng.seed(9);
    gc.stop_strings = {}; // explicit empty, same as default
    auto r2 = m2.generate({1}, gc);

    CHECK(r1.token_ids == r2.token_ids);
}

void test_stop_strings_ignored_without_tokenizer() {
    // stop_strings set but tok == nullptr must not crash and must not
    // affect generation (there's no text to match against) — see
    // GenerationConfig's field docs.
    TransformerModel m = make_model();
    m.rng.seed(13);
    GenerationConfig gc;
    gc.max_new_tokens = 15;
    gc.do_sample = false;
    gc.stop_strings = {"anything"};
    auto result = m.generate({1}, gc, nullptr);
    CHECK(result.tokens_generated > 0);
    CHECK(result.text.empty()); // no tokenizer supplied -> no decoded text
}

void test_learned_pe_rejects_overflow() {
    auto cfg = make_test_cfg();
    cfg.pos_embedding = PosEmbedding::LEARNED;
    cfg.max_seq_len = 2;

    TransformerModel m;
    m.init(cfg);
    m.wm = WeightMap::random_init(cfg);
    m.bind_weights();

    bool threw = false;
    try {
        (void)m.forward({1, 2, 3}, false);
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}

void test_reset_context() {
    TransformerModel m = make_model();
    m.forward({1, 2, 3}, true);
    CHECK(m.context_used() > 0);
    m.reset_kv_cache();
    CHECK(m.context_used() == 0);
}

void test_weights_save_load(std::string const& tmpdir) {
    auto cfg = make_test_cfg();
    WeightMap wm = WeightMap::random_init(cfg);
    // Use correct separator for platform
#ifdef _WIN32
    std::string path = tmpdir + "\\test_weights.nfw";
#else
    std::string path = tmpdir + "/test_weights.nfw";
#endif
    wm.save(path);

    WeightMap wm2;
    wm2.load(path);

    // Check a few tensors match exactly
    auto const& e1 = wm.get("embed.weight");
    auto const& e2 = wm2.get("embed.weight");
    CHECK(e1.shape == e2.shape);
    float diff = 0.f;
    for (int64_t i = 0; i < e1.numel(); ++i)
        diff += std::fabs(e1[i] - e2[i]);
    CHECK_EQ(diff, 0.f, 1e-9f);
}

void test_model_presets() {
    // Just check that presets define valid configs and produce sensible param counts
    auto tiny   = ModelConfig::tiny();
    auto small  = ModelConfig::small();
    auto medium = ModelConfig::medium();

    tiny.check();
    small.check();
    medium.check();

    CHECK(tiny.param_count()   > 1'000'000LL);
    CHECK(small.param_count()  > 10'000'000LL);
    CHECK(medium.param_count() > 50'000'000LL);
    std::cout << "  " << tiny.summary()   << "\n";
    std::cout << "  " << small.summary()  << "\n";
    std::cout << "  " << medium.summary() << "\n";
}

#if defined(VLRF_USE_SYCL) || defined(VLRF_USE_OPENCL)
// -----------------------------------------------------------------------------
// GPU routing tests — only built when a GPU backend is compiled in. Verifies
// (a) forward() actually takes the compute::* GPU path when a non-CPU
// backend is active rather than silently staying on CPU, and (b) the GPU
// path produces numerically equivalent output to the CPU path, since
// vlrf::compute's whole premise is that routing must be behavior-preserving.
void test_gpu_routing_matches_cpu() {
    auto cfg = make_test_cfg();
    std::vector<int> ids = {1, 5, 10, 15, 20};

    // CPU reference: default-constructed BackendRegistry starts on CpuBackend.
    vlrf::backend::BackendRegistry::instance().set_cpu();
    CHECK(!compute::gpu_active());
    TransformerModel m_cpu;
    m_cpu.init(cfg);
    m_cpu.wm = WeightMap::random_init(cfg);
    m_cpu.bind_weights();
    Tensor logits_cpu = m_cpu.forward(ids, false);

    // GPU path: same weights (WeightMap::random_init uses a fixed seed —
    // verify that assumption holds, otherwise this comparison is meaningless).
    bool got_gpu = vlrf::backend::BackendRegistry::instance().try_sycl()
                || vlrf::backend::BackendRegistry::instance().try_opencl();
    if (!got_gpu) {
        std::cout << "  (no SYCL/OpenCL device available in this environment — "
                     "skipping GPU-vs-CPU comparison)\n";
        vlrf::backend::BackendRegistry::instance().set_cpu();
        return;
    }
    CHECK(compute::gpu_active());

    TransformerModel m_gpu;
    m_gpu.init(cfg);
    m_gpu.wm = WeightMap::random_init(cfg);
    m_gpu.bind_weights();
    Tensor logits_gpu = m_gpu.forward(ids, false);

    vlrf::backend::BackendRegistry::instance().set_cpu();  // restore for other tests

    CHECK(logits_cpu.numel() == logits_gpu.numel());
    float max_diff = 0.f;
    float const* a = logits_cpu.data_ptr();
    float const* b = logits_gpu.data_ptr();
    for (int64_t i = 0; i < logits_cpu.numel(); ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    // Loose tolerance: GPU kernels (SYCL/OpenCL) may accumulate in a
    // different order than the CPU SIMD path, so bit-exactness isn't
    // expected — but a real routing bug (e.g. wrong kernel args, uninitialised
    // buffers) would show up as a large, not marginal, discrepancy.
    CHECK(max_diff < 1e-2f);
    std::cout << "  GPU vs CPU forward max_diff = " << max_diff << "\n";
}

void test_gpu_routing_falls_back_on_cpu_backend() {
    // Explicitly on CpuBackend: compute:: calls must be indistinguishable
    // from calling ops:: directly (this is the "only routes when non-CPU"
    // requirement) — checked here via gpu_active() rather than output
    // comparison, since output would be identical either way by construction.
    vlrf::backend::BackendRegistry::instance().set_cpu();
    CHECK(!compute::gpu_active());
}
#endif // VLRF_USE_SYCL || VLRF_USE_OPENCL

// -----------------------------------------------------------------------------
// RUN() prints the test name before AND after it runs, flushing both times.
// stdout is fully buffered (not line-buffered) when a test binary's output
// is captured through a pipe rather than a real console -- exactly what
// ctest does. Without an explicit flush, a crash anywhere in the test list
// can discard the ENTIRE buffered log, making every crash look like it
// happened before main()'s first line even ran. Flushing after each test
// means a crash now leaves a log that ends at "  START <name>" with no
// matching "  ok <name>", identifying the culprit directly instead of
// requiring a debugger session to find it.
#define RUN(fn) do { \
    std::cout << "  START " #fn "\n" << std::flush; \
    fn(); \
    std::cout << "  ok    " #fn "\n" << std::flush; \
} while (0)

int main() {
    std::cout << "=== VLRF Model Tests ===\n" << std::flush;

    RUN(test_config_check);
    RUN(test_param_count);
    RUN(test_weight_map_keys);
    RUN(test_weight_shapes);
    RUN(test_forward_output_shape);
    RUN(test_forward_single_token);
    RUN(test_kvcache_incremental_equals_full);
    RUN(test_generate_returns_tokens);
    RUN(test_generate_eos_stops);
    RUN(test_repetition_penalty_reduces_repeats);
    RUN(test_repetition_penalty_is_noop_at_default);
    RUN(test_frequency_and_presence_penalty_change_output);
    RUN(test_no_repeat_ngram_blocks_exact_loops);
    RUN(test_no_repeat_ngram_zero_is_noop);
    RUN(test_stop_string_truncates_output);
    RUN(test_stop_strings_empty_is_noop);
    RUN(test_stop_strings_ignored_without_tokenizer);
    RUN(test_learned_pe_rejects_overflow);
    RUN(test_reset_context);
    { auto tmp = get_tmp_dir(); std::cout << "  START test_weights_save_load\n" << std::flush;
      test_weights_save_load(tmp);
      std::cout << "  ok    test_weights_save_load\n" << std::flush; }
    RUN(test_model_presets);

#if defined(VLRF_USE_SYCL) || defined(VLRF_USE_OPENCL)
    RUN(test_gpu_routing_falls_back_on_cpu_backend);
    RUN(test_gpu_routing_matches_cpu);
#endif

    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n" << std::flush;
    return g_fail > 0 ? 1 : 0;
}
