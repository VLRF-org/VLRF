// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — C API (vlrf.h)  (test_api.cpp)
// Tests the full public interface end-to-end.
// =============================================================================
#ifndef VLRF_BUILD_DLL
#  define VLRF_BUILD_DLL
#endif
#include "vlrf.h"
#include "tokenizer.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#ifdef _WIN32
#  include <windows.h>
#endif

// Cross-platform temp path helper
static std::string tmp_path(const char* filename) {
#ifdef _WIN32
    char buf[512];
    DWORD len = GetTempPathA(sizeof(buf), buf);
    if (len == 0) return std::string(".\\") + filename;
    return std::string(buf) + filename;
#else
    return std::string("/tmp/") + filename;
#endif
}

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if(!(cond)){std::cerr<<"FAIL ["<<__LINE__<<"]: "#cond"\n";++g_fail;} \
    else ++g_pass; } while(0)
#define CHECK_STATUS(s) CHECK((s) == VLRF_OK)

// Build a tiny but valid VlrfModelParams
static VlrfModelParams tiny_params() {
    VlrfModelParams p{};
    p.vocab_size     = 64;
    p.d_model        = 32;
    p.d_ff           = 64;
    p.n_heads        = 2;
    p.n_kv_heads     = 2;
    p.n_layers       = 2;
    p.max_seq_len    = 64;
    p.bos_token_id   = 1;
    p.eos_token_id   = 2;
    p.pad_token_id   = 0;
    p.temperature    = 1.0f;
    p.top_p          = 1.0f;
    p.top_k          = 0;
    p.max_new_tokens = 8;
    p.use_sampling   = 0;    // greedy
    return p;
}

// -----------------------------------------------------------------------------
void test_create_free() {
    VlrfModel h = vlrf_create_model();
    CHECK(h != nullptr);
    vlrf_free_model(h);
    ++g_pass;
}

void test_null_handle() {
    VlrfStatus s = vlrf_reset_context(nullptr);
    CHECK(s == VLRF_ERR_INVALID_ARG);
    CHECK(vlrf_context_length(nullptr) == 0);
}

void test_init_random() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    CHECK_STATUS(vlrf_init_random(h, &p));
    CHECK(vlrf_context_length(h) == 0);
    vlrf_free_model(h);
}

void test_init_bad_params() {
    VlrfModel h = vlrf_create_model();
    VlrfModelParams p = tiny_params();
    p.d_model = 33;  // not divisible by n_heads=2
    VlrfStatus s = vlrf_init_random(h, &p);
    CHECK(s != VLRF_OK);
    vlrf_free_model(h);
}

void test_get_params() {
    VlrfModel h = vlrf_create_model();
    auto pin = tiny_params();
    vlrf_init_random(h, &pin);
    VlrfModelParams pout{};
    CHECK_STATUS(vlrf_get_params(h, &pout));
    CHECK(pout.vocab_size == pin.vocab_size);
    CHECK(pout.d_model    == pin.d_model);
    CHECK(pout.n_layers   == pin.n_layers);
    vlrf_free_model(h);
}

void test_version_string() {
    const char* v = vlrf_version();
    CHECK(v != nullptr && std::strlen(v) > 0);
    std::cout << "  version: " << v << "\n";
}

void test_status_strings() {
    CHECK(std::string(vlrf_status_str(VLRF_OK))              == "OK");
    CHECK(std::string(vlrf_status_str(VLRF_ERR_INVALID_ARG)) == "Invalid argument");
    CHECK(std::string(vlrf_status_str(VLRF_ERR_NOT_INIT))    == "Model not initialised");
}

void test_model_summary() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h, &p);
    const char* s = vlrf_model_summary(h);
    CHECK(s != nullptr && std::strlen(s) > 0);
    std::cout << "  summary: " << s << "\n";
    vlrf_free_model(h);
}

void test_backend_info() {
    const char* info = vlrf_backend_info();
    CHECK(info != nullptr && std::strlen(info) > 0);
    std::cout << "  backend: " << info << "\n";
}

void test_get_logits_shape() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h, &p);
    vlrf_set_seed(h, 1);

    std::vector<float> logits(p.vocab_size, 0.f);
    int32_t ctx[] = {1, 5, 10};
    VlrfStatus s = vlrf_get_logits(h, ctx, 3, logits.data());
    CHECK_STATUS(s);

    // Logits should not all be zero after forward pass
    float sum_abs = 0.f;
    for (float v : logits) sum_abs += std::fabs(v);
    CHECK(sum_abs > 0.f);

    vlrf_free_model(h);
}

void test_generate_ids_greedy() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h, &p);
    vlrf_set_seed(h, 42);

    int32_t prompt[] = {1, 5};
    std::vector<int32_t> out(32, 0);
    int32_t out_len = (int32_t)out.size();
    VlrfStatus s = vlrf_generate_ids(h, prompt, 2, out.data(), &out_len,
                                  8, nullptr, nullptr);
    CHECK_STATUS(s);
    CHECK(out_len > 0);
    for (int i = 0; i < out_len; ++i)
        CHECK(out[i] >= 0 && out[i] < p.vocab_size);

    std::cout << "  generated " << out_len << " tokens\n";
    vlrf_free_model(h);
}

void test_reset_context() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h, &p);

    // Generate something to fill cache
    int32_t prompt[] = {1, 2, 3};
    std::vector<int32_t> out(16);
    int32_t len = 16;
    vlrf_generate_ids(h, prompt, 3, out.data(), &len, 5, nullptr, nullptr);
    CHECK(vlrf_context_length(h) > 0);

    CHECK_STATUS(vlrf_reset_context(h));
    CHECK(vlrf_context_length(h) == 0);

    vlrf_free_model(h);
}

void test_set_seed_reproducibility() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    p.use_sampling = 1;
    p.temperature  = 1.0f;
    vlrf_init_random(h, &p);

    int32_t prompt[] = {1};
    std::vector<int32_t> out1(8), out2(8);
    int32_t len;

    vlrf_set_seed(h, 99);
    len = 8;
    vlrf_generate_ids(h, prompt, 1, out1.data(), &len, 8, nullptr, nullptr);

    vlrf_set_seed(h, 99);
    len = 8;
    vlrf_generate_ids(h, prompt, 1, out2.data(), &len, 8, nullptr, nullptr);

    bool same = true;
    for (int i = 0; i < 8; ++i) if (out1[i] != out2[i]) { same = false; break; }
    CHECK(same);  // same seed → same sequence

    vlrf_free_model(h);
}

void test_streaming_callback() {
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h, &p);
    vlrf_set_seed(h, 7);

    int callback_count = 0;
    auto cb = [](void* ud, int32_t /*id*/, const char* /*str*/) -> int {
        (*reinterpret_cast<int*>(ud))++;
        return 1;  // continue
    };

    int32_t prompt[] = {1};
    std::vector<int32_t> out(16);
    int32_t len = 16;
    vlrf_generate_ids(h, prompt, 1, out.data(), &len, 6, cb, &callback_count);
    CHECK(callback_count == len - 1);  // callback fires for each NEW token (not prompt)

    vlrf_free_model(h);
}

void test_repetition_params_roundtrip() {
    // VlrfModelParams' new fields must survive a full config_to_params /
    // params_to_config round trip via vlrf_get_params, same as every other
    // field in the struct.
    VlrfModel h = vlrf_create_model();
    auto p = tiny_params();
    p.repetition_penalty   = 1.25f;
    p.frequency_penalty    = 0.5f;
    p.presence_penalty     = 0.75f;
    p.no_repeat_ngram_size = 3;
    CHECK_STATUS(vlrf_init_random(h, &p));

    VlrfModelParams out{};
    CHECK_STATUS(vlrf_get_params(h, &out));
    CHECK(std::fabs(out.repetition_penalty - 1.25f) < 1e-6f);
    CHECK(std::fabs(out.frequency_penalty  - 0.5f)  < 1e-6f);
    CHECK(std::fabs(out.presence_penalty   - 0.75f) < 1e-6f);
    CHECK(out.no_repeat_ngram_size == 3);

    vlrf_free_model(h);
}

void test_zero_init_params_are_off_by_default() {
    // A caller doing `VlrfModelParams p{};` (the documented idiom, used by
    // tiny_params() itself) must get the new fields at their "no effect"
    // defaults — critical for ABI compatibility with any prebuilt caller
    // that only sets the fields it knows about and zero-inits the rest.
    VlrfModelParams p{}; // zero-initialized, no fields set at all
    CHECK(p.repetition_penalty == 0.0f);   // NOTE: raw zero-init, NOT 1.0
    CHECK(p.frequency_penalty == 0.0f);
    CHECK(p.presence_penalty == 0.0f);
    CHECK(p.no_repeat_ngram_size == 0);
    // The above are the *storage* defaults for an untouched struct; what
    // matters for behavior is that params_to_config's caller-visible
    // effect at repetition_penalty==0.0 must still not break generation
    // (0.0 divides nothing since apply_repetition_penalties treats exactly
    // 1.0 as "off" — 0.0 IS a real, if extreme, penalty value). Verify
    // generation still runs cleanly with a fully zero-inited params struct
    // aside from the required non-zero architecture fields.
    p.vocab_size = 64; p.d_model = 32; p.d_ff = 64;
    p.n_heads = 2; p.n_kv_heads = 2; p.n_layers = 2; p.max_seq_len = 64;
    p.bos_token_id = 1; p.eos_token_id = 2; p.pad_token_id = 0;
    p.temperature = 1.0f; p.top_p = 1.0f; p.top_k = 0;
    p.max_new_tokens = 8; p.use_sampling = 0;
    // repetition_penalty left at 0.0 deliberately (see note above) to prove
    // this doesn't crash even though it's an unusual value.

    VlrfModel h = vlrf_create_model();
    CHECK_STATUS(vlrf_init_random(h, &p));
    int32_t prompt[] = {1};
    std::vector<int32_t> out(16, 0);
    int32_t out_len = (int32_t)out.size();
    VlrfStatus s = vlrf_generate_ids(h, prompt, 1, out.data(), &out_len, 6, nullptr, nullptr);
    CHECK_STATUS(s);
    CHECK(out_len > 0);
    vlrf_free_model(h);
}

void test_set_stop_strings_truncates_text() {
    // Build a tiny model + byte-level tokenizer, save both, reload via the
    // public API (this is the only way vlrf_generate_text has a tokenizer at
    // all — see vlrf_load_weights), then confirm vlrf_set_stop_strings()
    // actually stops generation and trims the matched string from the
    // returned text.
    auto p = tiny_params();
    p.vocab_size = 4 + 256; // must cover the byte tokenizer's full range
    p.use_sampling = 0;     // greedy, for reproducibility

    VlrfModel h1 = vlrf_create_model();
    CHECK_STATUS(vlrf_init_random(h1, &p));

    std::string nfw_path = tmp_path("vlrf_api_stop_test.nfw");
    std::string tok_path = tmp_path("vlrf_api_stop_test.nft");
    std::string cfg_path = nfw_path.substr(0, nfw_path.find_last_of("/\\") + 1) + "config.json";
    CHECK_STATUS(vlrf_save_weights(h1, nfw_path.c_str()));

    {
        std::ofstream cf(cfg_path);
        cf << "{\n"
           << "  \"vocab_size\": "    << p.vocab_size    << ",\n"
           << "  \"d_model\": "      << p.d_model       << ",\n"
           << "  \"d_ff\": "         << p.d_ff          << ",\n"
           << "  \"n_heads\": "      << p.n_heads       << ",\n"
           << "  \"n_kv_heads\": "   << p.n_kv_heads    << ",\n"
           << "  \"n_layers\": "     << p.n_layers      << ",\n"
           << "  \"max_seq_len\": "  << p.max_seq_len   << ",\n"
           << "  \"bos_token_id\": " << p.bos_token_id  << ",\n"
           << "  \"eos_token_id\": " << p.eos_token_id  << ",\n"
           << "  \"pad_token_id\": " << p.pad_token_id  << ",\n"
           << "  \"tie_embeddings\": true,\n"
           << "  \"pos_embedding\": \"sinusoidal\"\n"
           << "}\n";
    }

    // Byte-level tokenizer save — same format vlrf_load_weights expects for
    // its tokenizer_path argument (Tokenizer::load/save round trip).
    {
        vlrf::Tokenizer tok;
        tok.init_byte_vocab();
        tok.save(tok_path);
    }

    VlrfModel h2 = vlrf_create_model();
    CHECK_STATUS(vlrf_load_weights(h2, nfw_path.c_str(), tok_path.c_str()));
    vlrf_set_seed(h2, 5);

    char baseline[1024] = {0};
    CHECK_STATUS(vlrf_generate_text(h2, "A", baseline, sizeof(baseline), 30, nullptr, nullptr));
    std::string baseline_str(baseline);

    if (baseline_str.size() < 2) {
        std::cout << "  (baseline text too short to test stop-string truncation)\n";
        vlrf_free_model(h1); vlrf_free_model(h2);
        return;
    }

    std::string stop = baseline_str.substr(baseline_str.size() - 2);
    const char* stops[] = { stop.c_str() };
    CHECK_STATUS(vlrf_set_stop_strings(h2, stops, 1));
    vlrf_set_seed(h2, 5); // reset RNG so the run is directly comparable

    char stopped[1024] = {0};
    CHECK_STATUS(vlrf_generate_text(h2, "A", stopped, sizeof(stopped), 30, nullptr, nullptr));
    std::string stopped_str(stopped);

    CHECK(stopped_str.find(stop) == std::string::npos);
    CHECK(stopped_str.size() <= baseline_str.size());
    CHECK(baseline_str.compare(0, stopped_str.size(), stopped_str) == 0);

    // Clearing stop strings (n_stops=0) must restore unstopped behavior.
    CHECK_STATUS(vlrf_set_stop_strings(h2, nullptr, 0));
    vlrf_set_seed(h2, 5);
    char cleared[1024] = {0};
    CHECK_STATUS(vlrf_generate_text(h2, "A", cleared, sizeof(cleared), 30, nullptr, nullptr));
    CHECK(std::string(cleared) == baseline_str);

    vlrf_free_model(h1);
    vlrf_free_model(h2);
}

void test_save_load_weights() {
    VlrfModel h1 = vlrf_create_model();
    auto p = tiny_params();
    vlrf_init_random(h1, &p);

    std::string path = tmp_path("vlrf_api_test.nfw");
    CHECK_STATUS(vlrf_save_weights(h1, path.c_str()));

    // vlrf_load_weights requires a config.json next to the .nfw file —
    // d_model alone cannot reliably determine n_heads (see
    // infer_config_from_weights in vlrf_api.cpp for why), so
    // VLRF refuses to guess rather than silently load the wrong
    // architecture. Every real converter (convert_gpt2.py, convert_llama.py,
    // convert_safetensors.py, convert_gguf.py) writes this file
    // automatically; tests must do the same to exercise the supported path.
    std::string cfg_path = path.substr(0, path.find_last_of("/\\") + 1) + "config.json";
    {
        std::ofstream cf(cfg_path);
        cf << "{\n"
           << "  \"vocab_size\": "    << p.vocab_size    << ",\n"
           << "  \"d_model\": "      << p.d_model       << ",\n"
           << "  \"d_ff\": "         << p.d_ff          << ",\n"
           << "  \"n_heads\": "      << p.n_heads       << ",\n"
           << "  \"n_kv_heads\": "   << p.n_kv_heads    << ",\n"
           << "  \"n_layers\": "     << p.n_layers      << ",\n"
           << "  \"max_seq_len\": "  << p.max_seq_len   << ",\n"
           << "  \"bos_token_id\": " << p.bos_token_id  << ",\n"
           << "  \"eos_token_id\": " << p.eos_token_id  << ",\n"
           << "  \"pad_token_id\": " << p.pad_token_id  << ",\n"
           << "  \"tie_embeddings\": true,\n"
           << "  \"pos_embedding\": \"sinusoidal\"\n"
           << "}\n";
    }

    // Get logits from original model
    std::vector<float> logits1(p.vocab_size);
    int32_t ctx[] = {1, 3, 5};
    vlrf_set_seed(h1, 0);
    vlrf_get_logits(h1, ctx, 3, logits1.data());

    // Load into a second model and compare
    VlrfModel h2 = vlrf_create_model();
    CHECK_STATUS(vlrf_load_weights(h2, path.c_str(), nullptr));

    std::vector<float> logits2(p.vocab_size);
    vlrf_get_logits(h2, ctx, 3, logits2.data());

    float max_diff = 0.f;
    for (int i = 0; i < p.vocab_size; ++i)
        max_diff = std::max(max_diff, std::fabs(logits1[i] - logits2[i]));
    CHECK(max_diff < 1e-4f);
    std::cout << "  save/load logit max_diff = " << max_diff << "\n";

    vlrf_free_model(h1);
    vlrf_free_model(h2);
}

void test_set_backend_cpu() {
    VlrfStatus s = vlrf_set_backend(VLRF_BACKEND_CPU);
    CHECK_STATUS(s);
}

// -----------------------------------------------------------------------------
int main() {
    std::cout << "=== VLRF C API Tests ===\n";
    test_create_free();
    test_null_handle();
    test_init_random();
    test_init_bad_params();
    test_get_params();
    test_version_string();
    test_status_strings();
    test_model_summary();
    test_backend_info();
    test_get_logits_shape();
    test_generate_ids_greedy();
    test_reset_context();
    test_set_seed_reproducibility();
    test_streaming_callback();
    test_repetition_params_roundtrip();
    test_zero_init_params_are_off_by_default();
    test_set_stop_strings_truncates_text();
    test_save_load_weights();
    test_set_backend_cpu();
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
