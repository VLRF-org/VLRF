// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Public DLL API  (vlrf.h)
// Pure C interface for maximum ABI stability and cross-language use.
// Include this header in any application that loads vlrf.dll.
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// -----------------------------------------------------------------------------
// DLL export / import macros
// -----------------------------------------------------------------------------
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(VLRF_BUILD_DLL)
#    define VLRF_API __declspec(dllexport)
#  else
#    define VLRF_API __declspec(dllimport)
#  endif
#else
#  define VLRF_API __attribute__((visibility("default")))
#endif

// -----------------------------------------------------------------------------
// Status codes
// -----------------------------------------------------------------------------
typedef enum VlrfStatus {
    VLRF_OK                = 0,
    VLRF_ERR_INVALID_ARG   = 1,
    VLRF_ERR_OUT_OF_MEMORY = 2,
    VLRF_ERR_IO            = 3,
    VLRF_ERR_BAD_MAGIC     = 4,
    VLRF_ERR_NOT_INIT      = 5,
    VLRF_ERR_CTX_FULL      = 6,
    VLRF_ERR_UNKNOWN       = 99,
} VlrfStatus;

// -----------------------------------------------------------------------------
// Opaque model handle
// -----------------------------------------------------------------------------
typedef struct VlrfModel_t* VlrfModel;

// -----------------------------------------------------------------------------
// Configuration structure (subset exposed to C callers)
// Mirrors ModelConfig but stable across builds.
// -----------------------------------------------------------------------------
typedef struct VlrfModelParams {
    int32_t vocab_size;
    int32_t d_model;
    int32_t d_ff;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t n_layers;
    int32_t max_seq_len;
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t pad_token_id;
    float   temperature;
    float   top_p;
    int32_t top_k;
    int32_t max_new_tokens;
    int32_t use_sampling;    // 0 = greedy, 1 = sample
    int32_t kv_cache_dtype;  // VlrfKVCacheType: 0=F32, 1=F16, 2=Q8_0 (see below)
    // ---- Repetition control (added — see GenerationConfig in
    // generation_config.hpp for full semantics of each field) ----
    // These occupy 4 of the previously-reserved slots below. Struct size
    // and the layout of every field above is unchanged, so any existing
    // caller that does `VlrfModelParams p{};` (zero-init) continues to work
    // unmodified and gets these at their "off" defaults (0.0 / 0), and any
    // caller that was previously writing zeros into _reserved is
    // unaffected either way.
    float   repetition_penalty;   // 1.0 = off (default). See ModelConfig.
    float   frequency_penalty;    // 0.0 = off (default).
    float   presence_penalty;     // 0.0 = off (default).
    int32_t no_repeat_ngram_size; // 0 = off (default).
    int32_t _reserved[11];
} VlrfModelParams;

// -----------------------------------------------------------------------------
// KV cache storage type
// Lower precision shrinks the KV cache's resident memory footprint
// (F16 = 2x smaller, Q8_0 = 4x smaller than F32), letting a longer context
// window fit in the same RAM budget. Trades a small amount of numerical
// precision in attention for that memory headroom. Set via
// VlrfModelParams::kv_cache_dtype or vlrf_set_kv_cache_type().
// -----------------------------------------------------------------------------
typedef enum VlrfKVCacheType {
    VLRF_KV_F32  = 0,   // full precision, baseline
    VLRF_KV_F16  = 1,   // half float, ~2x smaller, effectively lossless in practice
    VLRF_KV_Q8_0 = 2,   // int8 + per-token scale, ~4x smaller, small quantization noise
} VlrfKVCacheType;

// -----------------------------------------------------------------------------
// Token streaming callback
//   user_data : opaque pointer passed through to callback
//   token_id  : the token that was just generated
//   token_str : UTF-8 string representation (may be multi-byte, null-terminated)
//   return 1  : continue generation
//   return 0  : stop generation
// -----------------------------------------------------------------------------
typedef int (*VlrfTokenCallback)(void* user_data,
                                int32_t token_id,
                                const char* token_str);

// -----------------------------------------------------------------------------
// Core API
// -----------------------------------------------------------------------------

// Create an uninitialised model handle.
// Call vlrf_load_weights() or vlrf_init_random() before using.
VLRF_API VlrfModel vlrf_create_model(void);

// Initialise with explicit params (random weights — for testing).
// After this call, weights are random and the model is usable.
VLRF_API VlrfStatus vlrf_init_random(VlrfModel model, VlrfModelParams const* params);

// Load model weights and config from a .nfw file.
// tokenizer_path may be NULL (disables string decode).
VLRF_API VlrfStatus vlrf_load_weights(VlrfModel model,
                                  const char* weights_path,
                                  const char* tokenizer_path);

// Save current weights to file.
VLRF_API VlrfStatus vlrf_save_weights(VlrfModel model,
                                  const char* weights_path);

// Get a copy of the model's current params.
VLRF_API VlrfStatus vlrf_get_params(VlrfModel model, VlrfModelParams* out_params);

// Update generation/sampling settings on an already-loaded model:
// temperature, top_p, top_k, max_new_tokens, use_sampling,
// repetition_penalty, frequency_penalty, presence_penalty,
// no_repeat_ngram_size. Deliberately narrower than a full "set params" —
// architecture fields (vocab_size, d_model, n_heads, ...) and
// kv_cache_dtype are IGNORED even if changed in *params, since altering
// those after weights are loaded and bound would require re-initializing
// the model's tensors and KV cache, not just flipping a config value. Use
// vlrf_set_kv_cache_type() for that field instead. Safe to call at any time,
// including mid-conversation (does not reset the KV cache).
VLRF_API VlrfStatus vlrf_set_generation_params(VlrfModel model, VlrfModelParams const* params);

// -----------------------------------------------------------------------------
// Tokenizer API  (optional; only valid if tokenizer was loaded)
// -----------------------------------------------------------------------------

// Encode text to token ids.
//   out_ids   : caller-allocated buffer
//   in_out_len: on input = capacity; on output = number of tokens written
VLRF_API VlrfStatus vlrf_encode(VlrfModel model,
                            const char* text,
                            int32_t* out_ids,
                            int32_t* in_out_len,
                            int32_t add_bos);

// Decode token ids to text.
//   out_text  : caller-allocated buffer (recommend >= 4096 bytes)
//   buf_len   : capacity of out_text
VLRF_API VlrfStatus vlrf_decode(VlrfModel model,
                            const int32_t* ids,
                            int32_t n_ids,
                            char*    out_text,
                            int32_t  buf_len);

// -----------------------------------------------------------------------------
// Inference API
// -----------------------------------------------------------------------------

// Generate text from a prompt (string → string).
//   prompt      : null-terminated UTF-8 input
//   out_text    : caller-allocated output buffer
//   buf_len     : capacity
//   max_tokens  : max new tokens (0 = use model default)
//   callback    : optional streaming callback (may be NULL)
//   user_data   : passed to callback
VLRF_API VlrfStatus vlrf_generate_text(VlrfModel          model,
                                   const char*      prompt,
                                   char*            out_text,
                                   int32_t          buf_len,
                                   int32_t          max_tokens,
                                   VlrfTokenCallback  callback,
                                   void*            user_data);

// Generate from pre-tokenized ids.
VLRF_API VlrfStatus vlrf_generate_ids(VlrfModel          model,
                                  const int32_t*   prompt_ids,
                                  int32_t          n_prompt,
                                  int32_t*         out_ids,
                                  int32_t*         in_out_len,
                                  int32_t          max_tokens,
                                  VlrfTokenCallback  callback,
                                  void*            user_data);

// Get raw logits for the next token given a context.
// out_logits must be pre-allocated with vocab_size floats.
VLRF_API VlrfStatus vlrf_get_logits(VlrfModel        model,
                                const int32_t* context_ids,
                                int32_t        n_context,
                                float*         out_logits);

// -----------------------------------------------------------------------------
// Context management
// -----------------------------------------------------------------------------

// Reset the KV cache (start a new conversation).
VLRF_API VlrfStatus vlrf_reset_context(VlrfModel model);

// Get number of tokens currently in the KV cache.
VLRF_API int32_t  vlrf_context_length(VlrfModel model);

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------
typedef enum VlrfBackend {
    VLRF_BACKEND_AUTO   = 0,
    VLRF_BACKEND_CPU    = 1,
    VLRF_BACKEND_OPENCL = 2,
    VLRF_BACKEND_SYCL   = 3,
} VlrfBackend;

VLRF_API VlrfStatus vlrf_set_backend(VlrfBackend backend);
VLRF_API const char* vlrf_backend_info(void);

// Returns a per-op breakdown of what actually runs on the active backend's
// device vs. falling back to CPU — e.g. for OpenCL, gemm/softmax may run on
// GPU while layer_norm/gelu still run on CPU (no kernel implemented yet for
// those). This is a multi-line string; each line after the first describes
// one op. For the CPU backend this just reports the active SIMD GEMM
// kernel (same info as vlrf_simd_kernel_info(), included here for symmetry
// with the GPU backends). Complements vlrf_backend_info(), which only
// reports the device name/init status, not per-op dispatch.
VLRF_API const char* vlrf_backend_capabilities(void);

// Returns which SIMD GEMM kernel is active for matrix multiplication:
// e.g. "AVX2+FMA (hand-written 8-wide micro-kernel)",
//      "AVX1 (hand-written 8-wide micro-kernel, no FMA — e.g. Ivy Bridge/Sandy Bridge)",
//      "Scalar fallback (no AVX detected)", or "oneMKL sgemm" if built with VLRF_USE_ONEMKL.
// Useful for confirming the correct kernel was selected at runtime,
// especially when deploying the same binary across machines with
// different CPU generations.
VLRF_API const char* vlrf_simd_kernel_info(void);

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

// Override context window length (call before first inference).
// The model auto-caps based on available RAM at load time,
// but you can reduce further with this call.
// Cannot increase beyond the value set at load time.
VLRF_API VlrfStatus vlrf_set_ctx(VlrfModel model, int32_t ctx_len);

// Set the KV cache storage dtype and reallocate the cache accordingly.
// Call before generation starts (or after vlrf_reset_context — this call
// itself resets the cache, discarding any in-progress conversation).
// Lowering precision (F16/Q8_0) shrinks resident memory, which — combined
// with vlrf_set_ctx — lets a longer context fit in a fixed RAM budget than
// F32 would allow. See VlrfKVCacheType above for the tradeoffs.
VLRF_API VlrfStatus vlrf_set_kv_cache_type(VlrfModel model, VlrfKVCacheType dtype);

// Return the KV cache's current total resident bytes across all layers
// (reflects whatever dtype/ctx length is currently configured).
VLRF_API int64_t  vlrf_kv_cache_bytes(VlrfModel model);

// Free all model resources.
VLRF_API void     vlrf_free_model(VlrfModel model);

// Return a human-readable error string.
VLRF_API const char* vlrf_status_str(VlrfStatus status);

// Return the library version string.
VLRF_API const char* vlrf_version(void);

// Return model summary string (architecture, param count, etc.)
VLRF_API const char* vlrf_model_summary(VlrfModel model);

// Set stop sequences: generation halts as soon as decoded output ends with
// any of these strings (the matched string is trimmed from the returned
// text). Pass n_stops=0 (or stops=NULL) to clear. Copies the strings
// internally, so the caller's buffers need not outlive this call. Requires
// a tokenizer to have been loaded via vlrf_load_weights (stop strings need
// decoded text to match against) — silently has no effect otherwise, same
// as GenerationConfig::stop_strings without a tokenizer.
// Persists across calls to vlrf_generate_text/vlrf_generate_ids until changed
// or cleared; unaffected by vlrf_reset_context().
VLRF_API VlrfStatus vlrf_set_stop_strings(VlrfModel            model,
                                      const char* const* stops,
                                      int32_t             n_stops);

// Set RNG seed for sampling reproducibility.
VLRF_API VlrfStatus vlrf_set_seed(VlrfModel model, uint32_t seed);

#ifdef __cplusplus
} // extern "C"
#endif
