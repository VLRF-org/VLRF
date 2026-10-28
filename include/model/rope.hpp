// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Rotary Positional Encoding (RoPE)  (rope.hpp)
//
// RoPE rotates query and key vectors in complex space based on position.
// This gives the model a relative position signal that generalises well
// to sequences longer than those seen during training.
//
// Used by: LLaMA, Mistral, Granite, Falcon, Qwen, and most modern LLMs.
//
// Algorithm (Su et al. 2021):
//   For a vector x of dimension d_head at position pos:
//   Split x into pairs: (x_0,x_1), (x_2,x_3), ..., (x_{d-2}, x_{d-1})
//   For pair i at position pos:
//     theta_i = pos / (rope_theta ^ (2i / d_head))
//     x'_{2i}   = x_{2i}   * cos(theta_i) - x_{2i+1} * sin(theta_i)
//     x'_{2i+1} = x_{2i+1} * cos(theta_i) + x_{2i}   * sin(theta_i)
//
// Config:
//   rope_theta : base frequency (10000.0 for LLaMA, 10000000.0 for Granite 128k)
//   d_head     : head dimension
//
// Applied to Q and K AFTER projection, BEFORE KV cache storage.
// V is NOT rotated.
// =============================================================================
#include "tensor.hpp"
#include <cmath>
#include <vector>
#include <mutex>

namespace vlrf {

// -----------------------------------------------------------------------------
// RoPE frequency cache
// Precomputes cos/sin tables for all positions up to max_seq_len.
// Thread-safe lazy initialisation.
// -----------------------------------------------------------------------------
struct RopeCache {
    std::vector<float> cos_table;  // [max_seq_len * d_head/2]
    std::vector<float> sin_table;  // [max_seq_len * d_head/2]
    int   max_seq_len {0};
    int   d_head      {0};
    float rope_theta  {10000.f};
    bool  ready       {false};
    std::mutex mtx;

    void init(int max_seq, int dh, float theta) {
        std::lock_guard<std::mutex> lk(mtx);
        if (ready && max_seq_len == max_seq && d_head == dh &&
            rope_theta == theta) return;

        max_seq_len = max_seq;
        d_head      = dh;
        rope_theta  = theta;
        int half    = dh / 2;

        cos_table.resize(max_seq * half);
        sin_table.resize(max_seq * half);

        for (int pos = 0; pos < max_seq; ++pos) {
            for (int i = 0; i < half; ++i) {
                float freq = 1.f / std::pow(theta, (float)(2*i) / (float)dh);
                float angle = (float)pos * freq;
                cos_table[pos * half + i] = std::cos(angle);
                sin_table[pos * half + i] = std::sin(angle);
            }
        }
        ready = true;
    }

    // Apply RoPE to tensor x: [T, n_heads * d_head] in-place
    // pos_offset: position of the first token in x (0 for prefill, past for decode)
    void apply(float* x, int T, int n_heads, int pos_offset) const {
        int half = d_head / 2;
        int stride = n_heads * d_head;   // elements per token

        for (int t = 0; t < T; ++t) {
            int pos = pos_offset + t;
            if (pos >= max_seq_len) break;

            float const* cos_row = cos_table.data() + pos * half;
            float const* sin_row = sin_table.data() + pos * half;

            // Process each head independently
            for (int h = 0; h < n_heads; ++h) {
                float* head = x + t * stride + h * d_head;

                // Rotate pairs
                for (int i = 0; i < half; ++i) {
                    float x0 = head[i];
                    float x1 = head[i + half];
                    head[i]        = x0 * cos_row[i] - x1 * sin_row[i];
                    head[i + half] = x1 * cos_row[i] + x0 * sin_row[i];
                }
            }
        }
    }
};

// Global cache — shared across all attention layers (same d_head and theta)
inline RopeCache& global_rope_cache() {
    static RopeCache cache;
    return cache;
}

} // namespace vlrf
