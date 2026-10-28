// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — TransformerBlock  (transformer_block.hpp)
// One layer of the Transformer: LN → Attention → residual → LN → FFN → residual
// Supports both Pre-LN (GPT-2 / modern) and Post-LN (original) ordering.
//
// Weight names under {prefix} = "layers.N":
//   layers.N.ln1.weight       [d_model]
//   layers.N.ln1.bias         [d_model]
//   layers.N.ln2.weight       [d_model]
//   layers.N.ln2.bias         [d_model]
//   layers.N.attn.*           → AttentionBlock
//   layers.N.ffn.*            → FeedForward
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include "compute.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "attention.hpp"
#include "feedforward.hpp"
#include <string>

namespace vlrf {

class TransformerBlock {
public:
    AttentionBlock attn;
    FeedForward    ffn;

    // LayerNorm / RMSNorm weights
    Tensor* ln1_w{nullptr};  Tensor* ln1_b{nullptr};
    Tensor* ln2_w{nullptr};  Tensor* ln2_b{nullptr};

    bool pre_norm{true};
    bool use_rms{false};
    float norm_eps{1e-5f};
    float residual_multiplier{1.0f};   // MuP scaling (IBM Granite 4.0+); 1.0 = no-op

    // Stored for re-binding in streaming mode
    WeightMap*   wm_ptr {nullptr};
    std::string  prefix_;
    ModelConfig const* cfg_ptr {nullptr};

    void bind(WeightMap& wm, std::string const& prefix, ModelConfig const& cfg) {
        wm_ptr  = &wm;
        prefix_ = prefix;
        cfg_ptr = &cfg;
        pre_norm = cfg.pre_norm;
        use_rms  = cfg.use_rms_norm;
        norm_eps = cfg.layer_norm_eps;
        residual_multiplier = cfg.residual_multiplier;
        bind_ptrs(wm, prefix, cfg);
    }

    // Re-bind raw pointers after streaming layer load
    void rebind() {
        if (wm_ptr && cfg_ptr)
            bind_ptrs(*wm_ptr, prefix_, *cfg_ptr);
    }

private:
    void bind_ptrs(WeightMap& wm, std::string const& prefix, ModelConfig const& cfg) {
        ln1_w = &wm.get(prefix + ".ln1.weight");
        ln2_w = &wm.get(prefix + ".ln2.weight");

        if (!use_rms) {
            ln1_b = &wm.get(prefix + ".ln1.bias");
            ln2_b = &wm.get(prefix + ".ln2.bias");
        }

        attn.bind(wm, prefix + ".attn", cfg);
        ffn.bind(wm,  prefix + ".ffn",  cfg);
    }
public:

    // -------------------------------------------------------------------------
    // Forward
    //   x        : [T, d_model]
    //   kv_cache : optional (layer-specific cache, may be nullptr)
    //   returns  : [T, d_model]
    // -------------------------------------------------------------------------
    Tensor forward(Tensor const& x, KVCache* kv_cache = nullptr) const {
        Tensor out = like(x);

        if (pre_norm) {
            // Pre-LN: norm → attn → residual → norm → ffn → residual
            Tensor h1 = normalise1(x);
            Tensor a  = attn.forward(h1, kv_cache);
            compute::add_inplace(a, x);              // residual

            Tensor h2 = normalise2(a);
            Tensor f  = ffn.forward(h2);
            compute::add_inplace(f, a);              // residual
            return f;
        } else {
            // Post-LN: attn → residual → norm → ffn → residual → norm
            Tensor a = attn.forward(x, kv_cache);
            compute::add_inplace(a, x);              // residual
            Tensor h1 = normalise1(a);

            Tensor f = ffn.forward(h1);
            compute::add_inplace(f, h1);             // residual
            return normalise2(f);
        }
    }

private:
    // Unused by forward()/forward_v2() directly (they call normalise1/2
    // instead) — kept for any external caller expecting a single-norm
    // entry point. Defaults to the ln1 weights.
    Tensor normalise(Tensor const& x) const {
        Tensor out = like(x);
        if (use_rms) {
            // RMSNorm has no IBackend equivalent — always CPU.
            ops::rms_norm(x, *ln1_w, out, norm_eps);
        } else {
            compute::layer_norm(x, *ln1_w, *ln1_b, out, norm_eps);
        }
        return out;
    }

    // Exposed for use in pre_norm path (first vs second norm)
    Tensor normalise1(Tensor const& x) const {
        Tensor out = like(x);
        if (use_rms)  ops::rms_norm(x, *ln1_w, out, norm_eps);
        else          compute::layer_norm(x, *ln1_w, *ln1_b, out, norm_eps);
        return out;
    }
    Tensor normalise2(Tensor const& x) const {
        Tensor out = like(x);
        if (use_rms)  ops::rms_norm(x, *ln2_w, out, norm_eps);
        else          compute::layer_norm(x, *ln2_w, *ln2_b, out, norm_eps);
        return out;
    }

public:
    // Override forward to use separate norm1/norm2 correctly
    Tensor forward_v2(Tensor const& x, KVCache* kv_cache = nullptr) const {
        // residual_multiplier (MuP, IBM Granite 4.0+): scales the sublayer
        // output BEFORE adding to the residual stream, applied identically
        // after both attention and FFN. Verified against
        // modeling_granitemoehybrid.py:
        //   hidden_states = residual + hidden_states * self.residual_multiplier
        // residual_multiplier defaults to 1.0 (no-op) for every other
        // architecture, so this is always safe to apply unconditionally.
        if (pre_norm) {
            Tensor h1 = normalise1(x);
            Tensor a  = attn.forward(h1, kv_cache);
            if (residual_multiplier != 1.0f) ops::scale_inplace(a, residual_multiplier);
            compute::add_inplace(a, x);

            Tensor h2 = normalise2(a);
            Tensor f  = ffn.forward(h2);
            if (residual_multiplier != 1.0f) ops::scale_inplace(f, residual_multiplier);
            compute::add_inplace(f, a);
            return f;
        } else {
            Tensor a = attn.forward(x, kv_cache);
            if (residual_multiplier != 1.0f) ops::scale_inplace(a, residual_multiplier);
            compute::add_inplace(a, x);
            Tensor h1 = normalise1(a);

            Tensor f  = ffn.forward(h1);
            if (residual_multiplier != 1.0f) ops::scale_inplace(f, residual_multiplier);
            compute::add_inplace(f, h1);
            return normalise2(f);
        }
    }
};

} // namespace vlrf
