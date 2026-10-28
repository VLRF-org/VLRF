// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — ModelConfig  (config.hpp)
// All hyperparameters for the Transformer model in one place.
// Designed to be serialised to/from JSON or a flat binary header.
// =============================================================================
#include <cstdint>
#include <string>
#include <stdexcept>

namespace vlrf {

// -----------------------------------------------------------------------------
// Activation function enum
// -----------------------------------------------------------------------------
enum class Activation : int { GELU = 0, SILU = 1, RELU = 2 };

// -----------------------------------------------------------------------------
// Position embedding type
// -----------------------------------------------------------------------------
enum class PosEmbedding : int {
    SINUSOIDAL  = 0,   // fixed sin/cos
    LEARNED     = 1,   // trainable table
    ROTARY      = 2,   // RoPE
};

// -----------------------------------------------------------------------------
// ModelConfig
// -----------------------------------------------------------------------------
struct ModelConfig {
    // ---- Vocabulary / tokenizer -------------------------------------------
    int vocab_size   = 32000;   // BPE vocabulary size
    int pad_token_id = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;

    // ---- Architecture ----------------------------------------------------
    int d_model      = 512;    // embedding / residual stream dimension
    int d_ff         = 2048;   // feedforward hidden dimension (typically 4 * d_model)
    int n_heads      = 8;      // number of attention heads
    int n_kv_heads   = 8;      // for grouped-query attention; == n_heads → MHA
    int n_layers     = 6;      // transformer blocks
    int max_seq_len  = 2048;   // maximum context window

    // ---- KV cache storage --------------------------------------------------
    // Dtype used to store the K/V cache during generation. Lower precision
    // shrinks resident memory (F16 = 2x smaller, Q8_0 = 4x smaller than F32)
    // at the cost of a small per-forward dequantize step and a little
    // numerical noise in attention scores — see include/model/kv_cache.hpp.
    // Values match vlrf::KVCacheDType (kept as int here to avoid a circular
    // include between config.hpp and kv_cache.hpp): 0=F32, 1=F16, 2=Q8_0.
    int kv_cache_dtype = 0;

    // ---- Head size (derived, validated in check())
    int d_head() const { return d_model / n_heads; }

    // ---- Normalisation / regularisation ----------------------------------
    float layer_norm_eps    = 1e-5f;
    bool  pre_norm          = true;   // pre-LN (GPT-style) vs post-LN
    bool  use_rms_norm      = false;  // use RMSNorm instead of LayerNorm
    float dropout           = 0.0f;   // training only; 0 = disabled

    // ---- Activations -----------------------------------------------------
    Activation  activation      = Activation::GELU;
    PosEmbedding pos_embedding  = PosEmbedding::SINUSOIDAL;

    // ---- RoPE settings --------------------------------------------------
    float rope_theta     = 10000.f; // RoPE base frequency
    //   LLaMA 2:  10000.0
    //   LLaMA 3:  500000.0
    //   Granite:  10000000.0  (128k context)
    //   Mistral:  10000.0

    // ---- MuP (Maximal Update Parametrization) scaling -----------------
    // Used by IBM Granite 4.0+ (granitemoehybrid / granitemoe architecture
    // family). All four default to "no-op" values so models that don't
    // use MuP (LLaMA, Mistral, GPT-2, Granite 3.x) are completely
    // unaffected — these multipliers only matter when explicitly set by
    // the model's config.json.
    //
    // Verified against transformers 5.12.1's actual
    // modeling_granitemoehybrid.py source (not docs/guesses):
    //   embedding_multiplier : inputs_embeds = embed(ids) * embedding_multiplier
    //   attention_multiplier : REPLACES the standard 1/sqrt(d_head) attention
    //                           scale entirely — do not combine with it.
    //   residual_multiplier  : hidden = residual + sublayer_out * residual_multiplier
    //                           (applied after BOTH attention and FFN sublayers)
    //   logits_scaling       : final_logits = lm_head(x) / logits_scaling
    //                           (division, applied AFTER the LM head matmul)
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;   // 0.0 = "unset", use 1/sqrt(d_head) instead
    float residual_multiplier  = 1.0f;
    float logits_scaling       = 1.0f;

    // ---- Attention flags -------------------------------------------------
    bool use_causal_mask = true;    // autoregressive causal masking
    bool use_flash_style = false;   // use chunked/flash-style loop (future)

    // ---- Weight tying -----------------------------------------------------
    bool tie_embeddings = true;    // share embed / lm_head weights

    // ---- Generation defaults ---------------------------------------------
    int   max_new_tokens     = 256;
    float temperature        = 0.8f;
    float top_p              = 0.9f;
    int   top_k              = 50;
    bool  do_sample          = true;
    // See GenerationConfig (generation_config.hpp) for full documentation
    // of these — same semantics, just the persisted/default-config copies.
    // Deliberately excludes stop_strings: a vector<string> doesn't fit this
    // struct's role as a flat, .nfw/JSON-persistable set of numeric knobs;
    // stop strings are set per-call via GenerationConfig or via
    // vlrf_set_stop_strings() in the C API instead.
    float repetition_penalty = 1.0f;
    float frequency_penalty  = 0.0f;
    float presence_penalty   = 0.0f;
    int   no_repeat_ngram_size = 0;

    // ---- Metadata --------------------------------------------------------
    std::string model_name    = "vlrf-base";
    std::string model_version = "1.0.0";
    int         model_magic   = 0x4E464D44;  // "NFMD"

    // ---- Validation -------------------------------------------------------
    void check() const {
        if (d_model <= 0 || d_model % n_heads != 0)
            throw std::runtime_error("d_model must be divisible by n_heads");
        if (n_layers <= 0)
            throw std::runtime_error("n_layers must be positive");
        if (max_seq_len <= 0)
            throw std::runtime_error("max_seq_len must be positive");
        if (vocab_size <= 0)
            throw std::runtime_error("vocab_size must be positive");
        if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0)
            throw std::runtime_error("n_heads must be divisible by n_kv_heads");
        if (d_ff <= 0)
            throw std::runtime_error("d_ff must be positive");
    }

    // ---- Preset configs for common model sizes ----------------------------
    static ModelConfig tiny() {       // ~6M params
        ModelConfig c;
        c.d_model = 256; c.d_ff = 1024; c.n_heads = 4; c.n_kv_heads = 4; c.n_layers = 4;
        c.vocab_size = 32000; c.max_seq_len = 1024;
        c.model_name = "vlrf-tiny";
        return c;
    }

    static ModelConfig small() {      // ~25M params
        ModelConfig c;
        c.d_model = 512; c.d_ff = 2048; c.n_heads = 8; c.n_kv_heads = 8; c.n_layers = 6;
        c.vocab_size = 32000; c.max_seq_len = 2048;
        c.model_name = "vlrf-small";
        return c;
    }

    static ModelConfig medium() {     // ~117M params
        ModelConfig c;
        c.d_model = 768; c.d_ff = 3072; c.n_heads = 12; c.n_kv_heads = 12; c.n_layers = 12;
        c.vocab_size = 32000; c.max_seq_len = 2048;
        c.model_name = "vlrf-medium";
        return c;
    }

    static ModelConfig large() {      // ~350M params
        ModelConfig c;
        c.d_model = 1024; c.d_ff = 4096; c.n_heads = 16; c.n_kv_heads = 16; c.n_layers = 24;
        c.vocab_size = 32000; c.max_seq_len = 4096;
        c.model_name = "vlrf-large";
        return c;
    }

    // ---- Estimate parameter count -----------------------------------------
    int64_t param_count() const {
        // embedding
        int64_t p = (int64_t)vocab_size * d_model;
        if (pos_embedding == PosEmbedding::LEARNED)
            p += (int64_t)max_seq_len * d_model;
        // each layer: attn QKV + O proj + FFN + 2x LN
        int64_t d_kv = (int64_t)(d_model / n_heads) * n_kv_heads;
        int64_t attn = (int64_t)d_model * (d_model + 2 * d_kv + d_model);  // qkv + o
        int64_t ffn  = (int64_t)d_model * d_ff + (int64_t)d_ff * d_model;
        int64_t ln   = (int64_t)2 * 2 * d_model;
        p += (int64_t)n_layers * (attn + ffn + ln);
        // final LN
        p += 2 * d_model;
        // lm_head (shared with embed if tie_embeddings)
        if (!tie_embeddings) p += (int64_t)d_model * vocab_size;
        return p;
    }

    std::string summary() const {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "%s | d=%d ff=%d h=%d L=%d ctx=%d vocab=%d | ~%.1fM params",
            model_name.c_str(),
            d_model, d_ff, n_heads, n_layers, max_seq_len, vocab_size,
            param_count() / 1e6f);
        return buf;
    }
};

} // namespace vlrf
