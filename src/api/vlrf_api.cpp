// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — API Implementation  (vlrf_api.cpp)
// Bridges the pure-C interface in vlrf.h to the C++ model internals.
// VLRF_BUILD_DLL is set by CMake via target_compile_definitions — do not redefine.
// =============================================================================
#ifndef VLRF_BUILD_DLL
#  define VLRF_BUILD_DLL
#endif
#include "vlrf.h"
#include "transformer.hpp"
#include "tokenizer.hpp"
#include "generation_config.hpp"
#include "kv_cache.hpp"
#include "backend.hpp"
#include "simd_gemm.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Forward declaration of helper (defined at bottom of this file)
// -----------------------------------------------------------------------------
static vlrf::ModelConfig infer_config_from_weights(vlrf::WeightMap const& wm);
static vlrf::ModelConfig load_config_json(std::string const& path,
                                         vlrf::WeightMap const& wm);

// Cap max_seq_len so KV cache fits in available RAM.
// Leaves at least 1 GB headroom beyond the model weights.
static int cap_ctx_to_ram(int desired_ctx, vlrf::ModelConfig const& cfg) {
    int d_head  = (cfg.n_heads > 0) ? (cfg.d_model / cfg.n_heads) : 64;
    // KV cache per token (all layers, K+V, both heads), at whichever dtype
    // cfg.kv_cache_dtype selects — F16/Q8_0 shrink this and so let more
    // context fit in the same RAM budget than the old fixed-F32 assumption did.
    auto  kv_dtype = static_cast<vlrf::KVCacheDType>(cfg.kv_cache_dtype);
    double bytes_per_elem = vlrf::kv_cache_bytes_per_element(kv_dtype);
    int64_t bytes_per_token = (int64_t)(cfg.n_layers * 2.0 *
                               cfg.n_kv_heads * d_head * bytes_per_elem);
    // Q8_0 also carries a small per-token scale (2 floats/token/layer,
    // K+V) — fold it in so the estimate isn't optimistic.
    if (kv_dtype == vlrf::KVCacheDType::Q8_0)
        bytes_per_token += (int64_t)cfg.n_layers * 2 * sizeof(float);
    if (bytes_per_token <= 0) return desired_ctx;

    int64_t avail    = vlrf::available_ram_bytes();
    // Reserve 1.5 GB for OS + app + activations, rest available for KV cache
    int64_t headroom = (int64_t)1536 * 1024 * 1024;
    int64_t kv_budget = avail - headroom;
    if (kv_budget <= 0) kv_budget = (int64_t)512 * 1024 * 1024; // floor 512 MB

    int max_ctx = (int)(kv_budget / bytes_per_token);
    // Round down to nearest 256
    max_ctx = (max_ctx / 256) * 256;
    if (max_ctx < 256) max_ctx = 256;

    if (max_ctx < desired_ctx) {
        fprintf(stderr,
            "[VLRF] Context capped %d -> %d tokens "
            "(KV cache would need %.1f GB, only %.1f GB available)\n",
            desired_ctx, max_ctx,
            (bytes_per_token * desired_ctx) / 1e9,
            avail / 1e9);
        return max_ctx;
    }
    return desired_ctx;
}

// -----------------------------------------------------------------------------
// Internal per-model context
// -----------------------------------------------------------------------------
struct VlrfModel_t {
    std::unique_ptr<vlrf::TransformerModel> model;
    std::unique_ptr<vlrf::Tokenizer>        tokenizer;
    std::string                           summary_cache;
    std::mutex                            mtx;
    // Set via vlrf_set_stop_strings(); consumed by vlrf_generate_text/_ids.
    // Persists across generate calls until changed/cleared, matching the
    // header doc — unaffected by vlrf_reset_context() since that only
    // touches the KV cache, not generation config.
    std::vector<std::string>              stop_strings;

    bool ready() const { return model != nullptr; }
};

// -----------------------------------------------------------------------------
// Conversion helpers
// -----------------------------------------------------------------------------
static VlrfModelParams config_to_params(vlrf::ModelConfig const& c) {
    VlrfModelParams p{};
    p.vocab_size     = c.vocab_size;
    p.d_model        = c.d_model;
    p.d_ff           = c.d_ff;
    p.n_heads        = c.n_heads;
    p.n_kv_heads     = c.n_kv_heads;
    p.n_layers       = c.n_layers;
    p.max_seq_len    = c.max_seq_len;
    p.bos_token_id   = c.bos_token_id;
    p.eos_token_id   = c.eos_token_id;
    p.pad_token_id   = c.pad_token_id;
    p.temperature    = c.temperature;
    p.top_p          = c.top_p;
    p.top_k          = c.top_k;
    p.max_new_tokens = c.max_new_tokens;
    p.use_sampling   = c.do_sample ? 1 : 0;
    p.kv_cache_dtype = c.kv_cache_dtype;
    p.repetition_penalty   = c.repetition_penalty;
    p.frequency_penalty    = c.frequency_penalty;
    p.presence_penalty     = c.presence_penalty;
    p.no_repeat_ngram_size = c.no_repeat_ngram_size;
    return p;
}

static vlrf::ModelConfig params_to_config(VlrfModelParams const& p) {
    vlrf::ModelConfig c;
    c.vocab_size     = p.vocab_size;
    c.d_model        = p.d_model;
    c.d_ff           = p.d_ff;
    c.n_heads        = p.n_heads;
    c.n_kv_heads     = p.n_kv_heads;
    c.n_layers       = p.n_layers;
    c.max_seq_len    = p.max_seq_len;
    c.bos_token_id   = p.bos_token_id;
    c.eos_token_id   = p.eos_token_id;
    c.pad_token_id   = p.pad_token_id;
    c.temperature    = p.temperature;
    c.top_p          = p.top_p;
    c.top_k          = p.top_k;
    c.max_new_tokens = p.max_new_tokens;
    c.do_sample      = p.use_sampling != 0;
    c.kv_cache_dtype = p.kv_cache_dtype;
    c.repetition_penalty   = p.repetition_penalty;
    c.frequency_penalty    = p.frequency_penalty;
    c.presence_penalty     = p.presence_penalty;
    c.no_repeat_ngram_size = p.no_repeat_ngram_size;
    return c;
}

// Error-wrapping macros
#define VLRF_GUARD(h) do { if (!(h)) return VLRF_ERR_INVALID_ARG; } while(0)
#define VLRF_READY(h) do { if (!(h)->ready()) return VLRF_ERR_NOT_INIT; } while(0)

#define VLRF_TRY    try {
#define VLRF_CATCH  } catch (std::bad_alloc const&)  { return VLRF_ERR_OUT_OF_MEMORY; } \
                    catch (std::ios_base::failure const& e) { \
                        fprintf(stderr, "[VLRF] IO error: %s\n", e.what()); \
                        return VLRF_ERR_IO; } \
                    catch (std::exception const& e) { \
                        fprintf(stderr, "[VLRF] Error: %s\n", e.what()); \
                        return VLRF_ERR_UNKNOWN; } \
                    catch (...) { \
                        fprintf(stderr, "[VLRF] Unknown exception\n"); \
                        return VLRF_ERR_UNKNOWN; }

// -----------------------------------------------------------------------------
// C API implementations
// -----------------------------------------------------------------------------
extern "C" {

// ---- Lifecycle ------------------------------------------------------------

VLRF_API VlrfModel vlrf_create_model(void) {
    try { return new VlrfModel_t(); }
    catch (...) { return nullptr; }
}

VLRF_API VlrfStatus vlrf_init_random(VlrfModel h, VlrfModelParams const* params) {
    VLRF_GUARD(h); VLRF_GUARD(params);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        vlrf::ModelConfig cfg = params_to_config(*params);
        cfg.max_seq_len = cap_ctx_to_ram(cfg.max_seq_len, cfg);
        cfg.check();

        h->model = std::make_unique<vlrf::TransformerModel>();
        h->model->init(cfg);
        h->model->wm = vlrf::WeightMap::random_init(cfg);
        h->model->bind_weights();
        h->summary_cache = cfg.summary();
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_load_weights(VlrfModel     h,
                                  const char* weights_path,
                                  const char* tokenizer_path)
{
    VLRF_GUARD(h); VLRF_GUARD(weights_path);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);

        vlrf::WeightMap wm;
        wm.load(weights_path);

        // Try loading config.json from same directory as the .nfw file
        // This is the recommended path — avoids fragile shape inference.
        vlrf::ModelConfig cfg;
        bool cfg_loaded = false;
        {
            std::string cfg_path = weights_path;
            // Replace filename with config.json
            size_t slash = cfg_path.find_last_of("/\\");
            if (slash != std::string::npos)
                cfg_path = cfg_path.substr(0, slash + 1) + "config.json";
            else
                cfg_path = "config.json";

            std::ifstream cf(cfg_path);
            if (cf.good()) {
                cfg = load_config_json(cfg_path, wm);
                cfg_loaded = true;
                fprintf(stderr, "[VLRF] Loaded config from %s\n", cfg_path.c_str());
            }
        }
        if (!cfg_loaded) {
            fprintf(stderr, "[VLRF] config.json not found, inferring from weights\n");
            cfg = infer_config_from_weights(wm);
        }
        // Cap context window to what fits in RAM before allocating KV caches
        cfg.max_seq_len = cap_ctx_to_ram(cfg.max_seq_len, cfg);
        cfg.check();

        h->model = std::make_unique<vlrf::TransformerModel>();
        h->model->init(cfg);
        h->model->wm = std::move(wm);
        h->model->bind_weights();
        h->summary_cache = cfg.summary();

        if (tokenizer_path && tokenizer_path[0] != '\0') {
            h->tokenizer = std::make_unique<vlrf::Tokenizer>();
            h->tokenizer->load(tokenizer_path);
        }
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_save_weights(VlrfModel h, const char* weights_path) {
    VLRF_GUARD(h); VLRF_GUARD(weights_path); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        h->model->wm.save(weights_path);
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_get_params(VlrfModel h, VlrfModelParams* out) {
    VLRF_GUARD(h); VLRF_GUARD(out); VLRF_READY(h);
    *out = config_to_params(h->model->cfg);
    return VLRF_OK;
}

VLRF_API VlrfStatus vlrf_set_generation_params(VlrfModel h, VlrfModelParams const* p) {
    VLRF_GUARD(h); VLRF_GUARD(p); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        auto& c = h->model->cfg;
        // Only touch generation/sampling fields — see header doc for why
        // architecture and kv_cache_dtype are deliberately excluded here.
        c.temperature            = p->temperature;
        c.top_p                  = p->top_p;
        c.top_k                  = p->top_k;
        c.max_new_tokens         = p->max_new_tokens;
        c.do_sample              = p->use_sampling != 0;
        c.repetition_penalty     = p->repetition_penalty;
        c.frequency_penalty      = p->frequency_penalty;
        c.presence_penalty       = p->presence_penalty;
        c.no_repeat_ngram_size   = p->no_repeat_ngram_size;
        return VLRF_OK;
    VLRF_CATCH
}

// Override the context window at runtime (must be called before first inference).
// Useful when you know you only need a short context and want to save RAM.
VLRF_API VlrfStatus vlrf_set_ctx(VlrfModel h, int32_t ctx_len) {
    VLRF_GUARD(h); VLRF_READY(h);
    if (ctx_len <= 0) return VLRF_ERR_INVALID_ARG;
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        // Allow explicit user override — they take responsibility for RAM
        h->model->cfg.max_seq_len = ctx_len;
        h->model->reset_kv_cache();
        h->summary_cache = h->model->cfg.summary();
        return VLRF_OK;
    VLRF_CATCH
}

// Change KV cache storage dtype at runtime (reallocates all layers' caches,
// discarding any in-progress conversation — same caveat as vlrf_set_ctx).
VLRF_API VlrfStatus vlrf_set_kv_cache_type(VlrfModel h, VlrfKVCacheType dtype) {
    VLRF_GUARD(h); VLRF_READY(h);
    if (dtype != VLRF_KV_F32 && dtype != VLRF_KV_F16 && dtype != VLRF_KV_Q8_0)
        return VLRF_ERR_INVALID_ARG;
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        h->model->cfg.kv_cache_dtype = (int32_t)dtype;
        h->model->reset_kv_cache();
        h->summary_cache = h->model->cfg.summary();
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API int64_t vlrf_kv_cache_bytes(VlrfModel h) {
    if (!h || !h->ready()) return -1;
    std::lock_guard<std::mutex> lk(h->mtx);
    return h->model->kv_cache_bytes();
}

VLRF_API void vlrf_free_model(VlrfModel h) {
    delete h;
}

// ---- Tokenizer ------------------------------------------------------------

VLRF_API VlrfStatus vlrf_encode(VlrfModel      h,
                            const char*  text,
                            int32_t*     out_ids,
                            int32_t*     in_out_len,
                            int32_t      add_bos)
{
    VLRF_GUARD(h); VLRF_GUARD(text); VLRF_GUARD(out_ids); VLRF_GUARD(in_out_len);
    VLRF_READY(h);
    if (!h->tokenizer) return VLRF_ERR_NOT_INIT;
    VLRF_TRY
        auto ids = h->tokenizer->encode(std::string(text), add_bos != 0);
        int32_t cap = *in_out_len;
        int32_t n   = std::min((int32_t)ids.size(), cap);
        for (int32_t i = 0; i < n; ++i) out_ids[i] = (int32_t)ids[i];
        *in_out_len = n;
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_decode(VlrfModel        h,
                            const int32_t* ids,
                            int32_t        n_ids,
                            char*          out_text,
                            int32_t        buf_len)
{
    VLRF_GUARD(h); VLRF_GUARD(ids); VLRF_GUARD(out_text); VLRF_READY(h);
    if (!h->tokenizer) return VLRF_ERR_NOT_INIT;
    VLRF_TRY
        std::vector<int> iv(ids, ids + n_ids);
        std::string text = h->tokenizer->decode(iv);
        std::strncpy(out_text, text.c_str(), (size_t)buf_len - 1);
        out_text[buf_len - 1] = '\0';
        return VLRF_OK;
    VLRF_CATCH
}

// ---- Inference ------------------------------------------------------------

VLRF_API VlrfStatus vlrf_generate_text(VlrfModel         h,
                                   const char*     prompt,
                                   char*           out_text,
                                   int32_t         buf_len,
                                   int32_t         max_tokens,
                                   VlrfTokenCallback callback,
                                   void*           user_data)
{
    VLRF_GUARD(h); VLRF_GUARD(prompt); VLRF_GUARD(out_text); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);

        // Encode prompt
        std::vector<int> prompt_ids;
        if (h->tokenizer) {
            prompt_ids = h->tokenizer->encode(std::string(prompt), true);
        } else {
            // Fallback: raw bytes, offset by special-token count (4)
            for (unsigned char c : std::string(prompt))
                prompt_ids.push_back(
                    std::min((int)c + 4, h->model->cfg.vocab_size - 1));
        }

        // Build GenerationConfig
        vlrf::GenerationConfig gc;
        gc.max_new_tokens = max_tokens > 0 ? max_tokens : h->model->cfg.max_new_tokens;
        gc.temperature    = h->model->cfg.temperature;
        gc.top_p          = h->model->cfg.top_p;
        gc.top_k          = h->model->cfg.top_k;
        gc.do_sample      = h->model->cfg.do_sample;
        gc.repetition_penalty   = h->model->cfg.repetition_penalty;
        gc.frequency_penalty    = h->model->cfg.frequency_penalty;
        gc.presence_penalty     = h->model->cfg.presence_penalty;
        gc.no_repeat_ngram_size = h->model->cfg.no_repeat_ngram_size;
        gc.stop_strings         = h->stop_strings;

        if (callback) {
            gc.on_token = [&](int token_id) -> bool {
                std::string tok_str = h->tokenizer
                    ? h->tokenizer->decode_token(token_id) : "";
                return callback(user_data, (int32_t)token_id, tok_str.c_str()) != 0;
            };
        }

        auto result = h->model->generate(prompt_ids, gc, h->tokenizer.get());

        // Write output (full text or decoded ids)
        std::string out_str = result.text.empty() && h->tokenizer
            ? h->tokenizer->decode(result.token_ids)
            : result.text;

        std::strncpy(out_text, out_str.c_str(), (size_t)buf_len - 1);
        out_text[buf_len - 1] = '\0';
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_generate_ids(VlrfModel         h,
                                  const int32_t*  prompt_ids,
                                  int32_t         n_prompt,
                                  int32_t*        out_ids,
                                  int32_t*        in_out_len,
                                  int32_t         max_tokens,
                                  VlrfTokenCallback callback,
                                  void*           user_data)
{
    VLRF_GUARD(h); VLRF_GUARD(out_ids); VLRF_GUARD(in_out_len); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);

        std::vector<int> pids;
        if (prompt_ids && n_prompt > 0)
            pids.assign(prompt_ids, prompt_ids + n_prompt);

        vlrf::GenerationConfig gc;
        gc.max_new_tokens = max_tokens > 0 ? max_tokens : h->model->cfg.max_new_tokens;
        gc.temperature    = h->model->cfg.temperature;
        gc.top_p          = h->model->cfg.top_p;
        gc.top_k          = h->model->cfg.top_k;
        gc.do_sample      = h->model->cfg.do_sample;
        gc.repetition_penalty   = h->model->cfg.repetition_penalty;
        gc.frequency_penalty    = h->model->cfg.frequency_penalty;
        gc.presence_penalty     = h->model->cfg.presence_penalty;
        gc.no_repeat_ngram_size = h->model->cfg.no_repeat_ngram_size;
        gc.stop_strings         = h->stop_strings;

        if (callback) {
            gc.on_token = [&](int id) -> bool {
                std::string tok = h->tokenizer
                    ? h->tokenizer->decode_token(id) : "";
                return callback(user_data, (int32_t)id, tok.c_str()) != 0;
            };
        }

        auto result = h->model->generate(pids, gc, h->tokenizer.get());

        int32_t cap = *in_out_len;
        int32_t n   = std::min((int32_t)result.token_ids.size(), cap);
        for (int32_t i = 0; i < n; ++i) out_ids[i] = (int32_t)result.token_ids[i];
        *in_out_len = n;
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API VlrfStatus vlrf_get_logits(VlrfModel        h,
                                const int32_t* context_ids,
                                int32_t        n_context,
                                float*         out_logits)
{
    VLRF_GUARD(h); VLRF_GUARD(context_ids); VLRF_GUARD(out_logits); VLRF_READY(h);
    if (n_context <= 0) return VLRF_ERR_INVALID_ARG;
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        std::vector<int> ids(context_ids, context_ids + n_context);
        h->model->reset_kv_cache();
        vlrf::Tensor logits = h->model->forward(ids, false);
        std::copy(logits.data_ptr(), logits.data_ptr() + logits.numel(), out_logits);
        return VLRF_OK;
    VLRF_CATCH
}

// ---- Context --------------------------------------------------------------

VLRF_API VlrfStatus vlrf_reset_context(VlrfModel h) {
    VLRF_GUARD(h); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        h->model->reset_kv_cache();
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API int32_t vlrf_context_length(VlrfModel h) {
    if (!h || !h->ready()) return 0;
    return (int32_t)h->model->context_used();
}

// ---- Backend --------------------------------------------------------------

VLRF_API VlrfStatus vlrf_set_backend(VlrfBackend backend) {
    VLRF_TRY
        auto& reg = vlrf::backend::BackendRegistry::instance();
        switch (backend) {
            case VLRF_BACKEND_AUTO:   reg.auto_select(); break;
            case VLRF_BACKEND_CPU:    reg.set_cpu();     break;
            case VLRF_BACKEND_OPENCL: reg.try_opencl();  break;
            case VLRF_BACKEND_SYCL:   reg.try_sycl();    break;
            default: return VLRF_ERR_INVALID_ARG;
        }
        return VLRF_OK;
    VLRF_CATCH
}

VLRF_API const char* vlrf_backend_info(void) {
    static std::string s;
    s = vlrf::backend::active_backend().device_info();
    return s.c_str();
}

VLRF_API const char* vlrf_backend_capabilities(void) {
    static std::string s;
    s = vlrf::backend::active_backend().capabilities();
    return s.c_str();
}

VLRF_API const char* vlrf_simd_kernel_info(void) {
    return vlrf::simd::active_kernel_name();
}

// ---- Utility --------------------------------------------------------------

VLRF_API const char* vlrf_status_str(VlrfStatus s) {
    switch (s) {
        case VLRF_OK:                return "OK";
        case VLRF_ERR_INVALID_ARG:   return "Invalid argument";
        case VLRF_ERR_OUT_OF_MEMORY: return "Out of memory";
        case VLRF_ERR_IO:            return "I/O error";
        case VLRF_ERR_BAD_MAGIC:     return "Bad file magic";
        case VLRF_ERR_NOT_INIT:      return "Model not initialised";
        case VLRF_ERR_CTX_FULL:      return "Context window full";
        default:                   return "Unknown error";
    }
}

VLRF_API const char* vlrf_version(void) { return "VLRF 1.1.2"; }

VLRF_API const char* vlrf_model_summary(VlrfModel h) {
    if (!h || !h->ready()) return "(no model loaded)";
    return h->summary_cache.c_str();
}

VLRF_API VlrfStatus vlrf_set_seed(VlrfModel h, uint32_t seed) {
    VLRF_GUARD(h); VLRF_READY(h);
    h->model->rng.seed(seed);
    return VLRF_OK;
}

VLRF_API VlrfStatus vlrf_set_stop_strings(VlrfModel            h,
                                      const char* const* stops,
                                      int32_t             n_stops)
{
    VLRF_GUARD(h); VLRF_READY(h);
    VLRF_TRY
        std::lock_guard<std::mutex> lk(h->mtx);
        h->stop_strings.clear();
        if (stops && n_stops > 0) {
            h->stop_strings.reserve((size_t)n_stops);
            for (int32_t i = 0; i < n_stops; ++i) {
                if (stops[i]) h->stop_strings.emplace_back(stops[i]);
            }
        }
        return VLRF_OK;
    VLRF_CATCH
}

} // extern "C"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// load_config_json — read the config.json written by the Python converter
// Parses a minimal subset: the fields we care about for inference.
// -----------------------------------------------------------------------------
static vlrf::ModelConfig load_config_json(std::string const& path,
                                         vlrf::WeightMap const& wm) {
    std::ifstream f(path);
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    auto get_int = [&](std::string const& key, int def) -> int {
        std::string pat = "\"" + key + "\"";
        size_t p = json.find(pat);
        if (p == std::string::npos) return def;
        size_t colon = json.find(':', p + pat.size());
        if (colon == std::string::npos) return def;
        size_t vs = colon + 1;
        while (vs < json.size() && (json[vs]==' '||json[vs]=='\t')) ++vs;
        return std::stoi(json.substr(vs));
    };
    auto get_float = [&](std::string const& key, float def) -> float {
        std::string pat = "\"" + key + "\"";
        size_t p = json.find(pat);
        if (p == std::string::npos) return def;
        size_t colon = json.find(':', p + pat.size());
        if (colon == std::string::npos) return def;
        size_t vs = colon + 1;
        while (vs < json.size() && (json[vs]==' '||json[vs]=='\t')) ++vs;
        return std::stof(json.substr(vs));
    };
    auto get_bool = [&](std::string const& key, bool def) -> bool {
        std::string pat = "\"" + key + "\"";
        size_t p = json.find(pat);
        if (p == std::string::npos) return def;
        size_t colon = json.find(':', p + pat.size());
        if (colon == std::string::npos) return def;
        size_t vs = colon + 1;
        while (vs < json.size() && (json[vs]==' '||json[vs]=='\t')) ++vs;
        return json.substr(vs, 4) == "true";
    };
    auto get_str = [&](std::string const& key, std::string def) -> std::string {
        std::string pat = "\"" + key + "\"";
        size_t p = json.find(pat);
        if (p == std::string::npos) return def;
        size_t colon = json.find(':', p + pat.size());
        if (colon == std::string::npos) return def;
        size_t q1 = json.find('"', colon + 1);
        if (q1 == std::string::npos) return def;
        size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) return def;
        return json.substr(q1 + 1, q2 - q1 - 1);
    };

    vlrf::ModelConfig cfg;
    cfg.vocab_size    = get_int("vocab_size",    32000);
    cfg.d_model       = get_int("d_model",       512);
    cfg.d_ff          = get_int("d_ff",          2048);
    cfg.n_heads       = get_int("n_heads",       8);
    cfg.n_kv_heads    = get_int("n_kv_heads",    cfg.n_heads);
    cfg.n_layers      = get_int("n_layers",      6);
    cfg.max_seq_len   = get_int("max_seq_len",   2048);
    cfg.bos_token_id  = get_int("bos_token_id",  1);
    cfg.eos_token_id  = get_int("eos_token_id",  2);
    cfg.pad_token_id  = get_int("pad_token_id",  0);
    cfg.max_new_tokens= get_int("max_new_tokens",256);
    cfg.top_k         = get_int("top_k",         50);
    cfg.layer_norm_eps= get_float("layer_norm_eps", 1e-5f);
    cfg.temperature   = get_float("temperature", 0.8f);
    cfg.top_p         = get_float("top_p",       0.9f);
    cfg.repetition_penalty   = get_float("repetition_penalty",   1.0f);
    cfg.frequency_penalty    = get_float("frequency_penalty",    0.0f);
    cfg.presence_penalty     = get_float("presence_penalty",     0.0f);
    cfg.no_repeat_ngram_size = get_int("no_repeat_ngram_size",   0);
    cfg.pre_norm      = get_bool("pre_norm",     true);
    cfg.use_rms_norm  = get_bool("use_rms_norm", false);
    cfg.tie_embeddings= get_bool("tie_embeddings",true);
    cfg.do_sample     = true;

    std::string act = get_str("activation", "gelu");
    if (act == "silu") cfg.activation = vlrf::Activation::SILU;
    else if (act == "relu") cfg.activation = vlrf::Activation::RELU;
    else cfg.activation = vlrf::Activation::GELU;

    std::string pos = get_str("pos_embedding", "sinusoidal");
    if (pos == "learned")   cfg.pos_embedding = vlrf::PosEmbedding::LEARNED;
    else if (pos == "rotary") cfg.pos_embedding = vlrf::PosEmbedding::ROTARY;
    else                    cfg.pos_embedding = vlrf::PosEmbedding::SINUSOIDAL;

    cfg.rope_theta = get_float("rope_theta", 10000.f);

    // MuP scaling (IBM Granite 4.0+ "granitemoehybrid"/"granitemoe" family).
    // Defaults are true no-ops, so reading these for every other
    // architecture's config.json (which won't have these keys) safely
    // falls back to "behave exactly as before this feature was added".
    cfg.embedding_multiplier = get_float("embedding_multiplier", 1.0f);
    cfg.attention_multiplier = get_float("attention_multiplier", 0.0f);
    cfg.residual_multiplier  = get_float("residual_multiplier",  1.0f);
    cfg.logits_scaling       = get_float("logits_scaling",       1.0f);

    // Override tie_embeddings based on actual weights
    cfg.tie_embeddings = !wm.has("lm_head.weight");

    return cfg;
}

// -----------------------------------------------------------------------------
// infer_config_from_weights — fallback when config.json is not present.
// Reconstructs ModelConfig by examining weight tensor shapes.
// -----------------------------------------------------------------------------
static vlrf::ModelConfig infer_config_from_weights(vlrf::WeightMap const& wm) {
    vlrf::ModelConfig cfg;

    // Vocabulary size and model dimension from embedding table
    {
        auto const& e = wm.get("embed.weight");
        cfg.vocab_size = e.dim(0);
        cfg.d_model    = e.dim(1);
    }

    // Count layers by probing for layer norm weights
    cfg.n_layers = 0;
    while (wm.has("layers." + std::to_string(cfg.n_layers) + ".ln1.weight"))
        ++cfg.n_layers;
    if (cfg.n_layers == 0)
        throw std::runtime_error(
            "infer_config: could not detect n_layers from weight keys");

    // FFN dimension from first layer's fc1 weight  [d_ff, d_model]
    {
        auto const& f = wm.get("layers.0.ffn.fc1.weight");
        cfg.d_ff = f.dim(0);
    }

    // Head count: CANNOT be reliably inferred from d_model alone.
    //
    // d_head is not recoverable from d_model + k_proj shape — many valid
    // (n_heads, d_head) pairs divide the same d_model. Concretely:
    //   d_model=2560 (Granite-3B, real d_head=80, n_heads=32)
    //     is ALSO divisible by 64 -> would wrongly infer n_heads=40
    //   d_model=1280 (GPT-2-large, real d_head=64, n_heads=20)
    //     is ALSO divisible by 128 -> would wrongly infer n_heads=10
    // No fixed priority order gets every known architecture right; any
    // such heuristic silently produces a WRONG model that still "loads"
    // and "runs" but generates garbage, with no error to indicate why.
    //
    // All VLRF Python converters (convert_gpt2.py, convert_llama.py,
    // convert_safetensors.py, convert_gguf.py) write a config.json with
    // the real n_heads value next to every .nfw — vlrf_load_weights() reads
    // that file FIRST and only reaches this code path if it's missing.
    // Given the silent-wrong-answer risk, we refuse to guess: fail loudly
    // and tell the user exactly what's missing and how to fix it.
    throw std::runtime_error(
        "infer_config_from_weights: cannot reliably determine n_heads "
        "from weight shapes alone (d_model=" + std::to_string(cfg.d_model) +
        " is divisible by multiple plausible head dimensions). "
        "config.json is missing or could not be read from the model "
        "directory. Re-run the Python converter (tools/converter/convert*.py) "
        "to regenerate it, or place a config.json with the correct "
        "\"n_heads\" and \"n_kv_heads\" fields next to the .nfw file.");

    cfg.tie_embeddings = !wm.has("lm_head.weight");

    cfg.pos_embedding = wm.has("pos_embed.weight")
                      ? vlrf::PosEmbedding::LEARNED
                      : vlrf::PosEmbedding::SINUSOIDAL;

    if (cfg.pos_embedding == vlrf::PosEmbedding::LEARNED)
        cfg.max_seq_len = wm.get("pos_embed.weight").dim(0);

    return cfg;
}
