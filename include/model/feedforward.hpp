// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — FeedForward Block  (feedforward.hpp)
// Two-layer MLP with configurable activation (GELU / SiLU / ReLU).
// Supports standard and gated variants (SwiGLU / GeGLU).
//
// Standard:  FFN(x) = act(x W1 + b1) W2 + b2
// Gated:     FFN(x) = (act(x Wg) * (x W1)) W2   (SwiGLU style)
//
// Weight names under prefix:
//   {prefix}.fc1.weight  [d_ff, d_model]
//   {prefix}.fc1.bias    [d_ff]           (optional)
//   {prefix}.fc2.weight  [d_model, d_ff]
//   {prefix}.fc2.bias    [d_model]         (optional)
//   {prefix}.gate.weight [d_ff, d_model]   (optional, for gated)
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include "compute.hpp"
#include "config.hpp"
#include "weights.hpp"
#include <string>

namespace vlrf {

class FeedForward {
public:
    Tensor* W1{nullptr};  Tensor* b1{nullptr};
    Tensor* W2{nullptr};  Tensor* b2{nullptr};
    Tensor* Wg{nullptr};  // gate projection (optional, for SwiGLU)

    Activation act{Activation::GELU};
    bool       gated{false};

    void bind(WeightMap& wm, std::string const& prefix, ModelConfig const& cfg) {
        act   = cfg.activation;
        gated = (act == Activation::SILU) && wm.has(prefix + ".gate.weight");

        W1 = &wm.get(prefix + ".fc1.weight");
        W2 = &wm.get(prefix + ".fc2.weight");

        if (wm.has(prefix + ".fc1.bias")) b1 = &wm.get(prefix + ".fc1.bias");
        if (wm.has(prefix + ".fc2.bias")) b2 = &wm.get(prefix + ".fc2.bias");
        if (gated) Wg = &wm.get(prefix + ".gate.weight");
    }

    // x: [T, d_model]  -> [T, d_model]
    Tensor forward(Tensor const& x) const {
        // h = x @ W1^T  + b1    [T, d_ff]
        // GEMM routes to GPU when active; transpose and bias-add have no
        // IBackend equivalent and stay on CPU either way.
        Tensor h = compute::linear(x, *W1);
        if (b1) ops::add_bias_inplace(h, *b1);

        if (gated) {
            // gate path for SwiGLU — silu_inplace/multiply_inplace have no
            // IBackend equivalent, always CPU.
            Tensor g = compute::linear(x, *Wg);
            ops::silu_inplace(g);        // sigmoid-gated linear unit
            ops::multiply_inplace(h, g); // element-wise gate
        } else {
            // Standard activation. Only GELU has an IBackend method;
            // SiLU/ReLU stay CPU regardless of active backend.
            switch (act) {
                case Activation::GELU: compute::gelu_inplace(h); break;
                case Activation::SILU: ops::silu_inplace(h); break;
                case Activation::RELU: ops::relu_inplace(h); break;
            }
        }

        // out = h @ W2^T + b2   [T, d_model]
        Tensor out = compute::linear(h, *W2);
        if (b2) ops::add_bias_inplace(out, *b2);
        return out;
    }
};

} // namespace vlrf
