// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — TransformerModel  (transformer.hpp)
// Full autoregressive decoder-only Transformer.
//
// Weight layout in WeightMap:
//   embed.weight                     [vocab_size, d_model]
//   pos_embed.weight                 [max_seq_len, d_model]  (if LEARNED)
//   layers.N.ln1.weight/bias         [d_model]
//   layers.N.ln2.weight/bias         [d_model]
//   layers.N.attn.{q,k,v,o}_proj.{weight,bias}
//   layers.N.ffn.{fc1,fc2}.{weight,bias}
//   ln_final.weight/bias             [d_model]
//   lm_head.weight                   [vocab_size, d_model]  (may == embed.weight)
// =============================================================================
#include "config.hpp"
#include "generation_config.hpp"
#include "weights.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include "compute.hpp"
#include "attention.hpp"
#include "feedforward.hpp"
#include "transformer_block.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

namespace vlrf {

// -----------------------------------------------------------------------------
// Sampler helpers
// -----------------------------------------------------------------------------
namespace sampler {

inline int argmax(Tensor const& logits) {
    float const* p = logits.data_ptr();
    int n = (int)logits.numel();
    return (int)(std::max_element(p, p + n) - p);
}

// Apply repetition/frequency/presence penalties to raw logits in-place,
// based on token_counts (a histogram over GENERATED tokens only — see
// GenerationConfig's field comments for why prompt tokens are excluded).
// Order matches common inference-server convention: multiplicative
// repetition_penalty first, then additive frequency/presence penalties.
inline void apply_repetition_penalties(Tensor& logits,
                                        std::vector<int> const& token_counts,
                                        float repetition_penalty,
                                        float frequency_penalty,
                                        float presence_penalty)
{
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return; // fast path: no-op when all penalties are at their defaults

    float* p = logits.data_ptr();
    int n = (int)logits.numel();
    int hist_n = (int)token_counts.size();
    for (int id = 0; id < n && id < hist_n; ++id) {
        int count = token_counts[id];
        if (count <= 0) continue;

        if (repetition_penalty != 1.0f) {
            // CTRL-style: penalize positive logits by dividing, negative
            // logits by multiplying, so the penalty always pushes the
            // logit DOWN regardless of its sign.
            p[id] = (p[id] > 0.f) ? (p[id] / repetition_penalty)
                                   : (p[id] * repetition_penalty);
        }
        if (frequency_penalty != 0.0f) {
            p[id] -= frequency_penalty * (float)count;
        }
        if (presence_penalty != 0.0f) {
            p[id] -= presence_penalty;
        }
    }
}

// Blocks any token that would complete a repeat of an n-gram already seen
// among `generated` (the tokens produced so far this generation, NOT
// including the prompt). Zeroes out (-inf) the logit of every candidate
// token that would recreate an n-gram whose first (n-1) tokens match the
// most recent (n-1) generated tokens. No-op if n <= 0 or not enough tokens
// have been generated yet to form a window.
inline void block_repeated_ngrams(Tensor& logits,
                                   std::vector<int> const& generated,
                                   int n)
{
    if (n <= 1) return;
    int gen_n = (int)generated.size();
    if (gen_n < n - 1) return;

    // The (n-1)-token suffix that would become the start of a new n-gram
    // if we picked some next token.
    std::vector<int> suffix(generated.end() - (n - 1), generated.end());

    float* p = logits.data_ptr();
    int vocab_n = (int)logits.numel();

    // Scan every historical n-gram; if its first (n-1) tokens match
    // `suffix`, its last token is now forbidden as the next pick.
    for (int start = 0; start + n <= gen_n; ++start) {
        bool match = true;
        for (int k = 0; k < n - 1; ++k) {
            if (generated[start + k] != suffix[k]) { match = false; break; }
        }
        if (!match) continue;
        int forbidden = generated[start + n - 1];
        if (forbidden >= 0 && forbidden < vocab_n) {
            p[forbidden] = -std::numeric_limits<float>::infinity();
        }
    }
}

// Apply temperature, top-k, top-p filtering, then draw one sample.
// logits tensor is modified in-place.
inline int sample(Tensor& logits, float temperature, float top_p, int top_k,
                  std::mt19937& rng)
{
    int n = (int)logits.numel();
    float* p = logits.data_ptr();

    if (temperature <= 0.f) return argmax(logits);

    // Temperature scaling
    for (int i = 0; i < n; ++i) p[i] /= temperature;

    // Softmax into probability space. Deliberately CPU (ops::, not
    // compute::) even when a GPU backend is active: this is a single
    // [vocab_size] vector, run once per generated token, already back on
    // host after the LM-head matmul above — round-tripping it to device and
    // back would add transfer overhead with no parallelism win at this size.
    ops::softmax_inplace(logits);

    // Top-k: zero out everything outside the k highest-probability tokens
    if (top_k > 0 && top_k < n) {
        std::vector<std::pair<float,int>> sv(n);
        for (int i = 0; i < n; ++i) sv[i] = {p[i], i};
        std::partial_sort(sv.begin(), sv.begin() + top_k, sv.end(),
                          [](auto const& a, auto const& b){ return a.first > b.first; });
        for (int i = top_k; i < n; ++i) p[sv[i].second] = 0.f;
    }

    // Top-p (nucleus): keep the smallest set of tokens whose cumulative
    // probability exceeds top_p.
    if (top_p < 1.f) {
        std::vector<std::pair<float,int>> sv(n);
        for (int i = 0; i < n; ++i) sv[i] = {p[i], i};
        std::sort(sv.begin(), sv.end(),
                  [](auto const& a, auto const& b){ return a.first > b.first; });
        float cum = 0.f;
        for (int i = 0; i < n; ++i) {
            cum += sv[i].first;
            if (cum > top_p && i > 0) {
                for (int j = i; j < n; ++j) p[sv[j].second] = 0.f;
                break;
            }
        }
    }

    // Renormalise
    float sum = 0.f;
    for (int i = 0; i < n; ++i) sum += p[i];
    if (sum < 1e-9f) return argmax(logits);
    for (int i = 0; i < n; ++i) p[i] /= sum;

    // Categorical sample
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float u = dist(rng), cum = 0.f;
    for (int i = 0; i < n; ++i) {
        cum += p[i];
        if (u <= cum) return i;
    }
    return n - 1;
}

} // namespace sampler

// -----------------------------------------------------------------------------
// TransformerModel
// -----------------------------------------------------------------------------
class TransformerModel {
public:
    ModelConfig                   cfg;
    std::vector<TransformerBlock> layers;
    std::vector<KVCache>          kv_caches;

    // Raw pointers into wm — valid as long as wm is alive
    Tensor* embed_w    {nullptr};
    Tensor* pos_embed_w{nullptr};   // null when using sinusoidal PE
    Tensor* ln_final_w {nullptr};
    Tensor* ln_final_b {nullptr};   // null when using RMSNorm
    Tensor* lm_head_w  {nullptr};   // may alias embed_w when tie_embeddings=true

    WeightMap    wm;
    std::mt19937 rng{42};

    TransformerModel() = default;

    // TransformerModel is move-only (WeightMap itself is move-only — see
    // weights.hpp), but move can't just be the implicit member-wise
    // default here. bind_weights() hands out raw pointers that are
    // self-referential in two distinct ways:
    //   - embed_w/ln_final_w/.../layers[l].ln1_w etc. point AT elements of
    //     this->wm.tensors (an unordered_map) — these stay valid across a
    //     move because the map's own move steals its node/bucket storage
    //     rather than relocating individual elements.
    //   - layers[l].wm_ptr/cfg_ptr point AT this->wm / this->cfg
    //     THEMSELVES (the WeightMap/ModelConfig objects, not their
    //     contents) — these do NOT survive a move: after `other` is
    //     moved-from, its wm/cfg member addresses are gone, but wm_ptr/
    //     cfg_ptr inside the (also moved) `layers` still hold those old
    //     addresses. Left unfixed, the very first rebind() call after the
    //     move (or any streaming forward pass) dereferences a pointer into
    //     a destroyed object — undefined behaviour that shows up as heap
    //     corruption / a crash inside unordered_map's internals, often far
    //     from this constructor and easy to misdiagnose as an STL bug.
    // The fix: move the plain data members, then re-run
    // bind_weight_pointers() so every raw pointer (both kinds above) is
    // re-derived fresh against *this rather than trusting stale addresses
    // to still be correct. Deliberately NOT bind_weights(): that also
    // calls reset_kv_cache(), which would silently wipe the kv_caches we
    // just moved in (e.g. discarding an in-progress generation's context
    // if the owning TransformerModel gets relocated mid-use). Cheap:
    // bind_weight_pointers() only does map lookups + pointer arithmetic,
    // no tensor data is copied or reallocated.
    TransformerModel(TransformerModel&& o) noexcept
        : cfg(std::move(o.cfg))
        , kv_caches(std::move(o.kv_caches))
        , wm(std::move(o.wm))
        , rng(o.rng)
    {
        layers.resize(cfg.n_layers);
        if (!wm.tensors.empty()) bind_weight_pointers();
    }

    TransformerModel& operator=(TransformerModel&& o) noexcept {
        if (this == &o) return *this;
        cfg       = std::move(o.cfg);
        kv_caches = std::move(o.kv_caches);
        wm        = std::move(o.wm);
        rng       = o.rng;
        layers.clear();
        layers.resize(cfg.n_layers);
        embed_w = pos_embed_w = ln_final_w = ln_final_b = lm_head_w = nullptr;
        if (!wm.tensors.empty()) bind_weight_pointers();
        return *this;
    }

    TransformerModel(TransformerModel const&)            = delete;
    TransformerModel& operator=(TransformerModel const&) = delete;

    // -------------------------------------------------------------------------
    // init() — set config, size layer array. Does NOT allocate weights.
    // Follow with:  wm = WeightMap::random_init(cfg)  OR  wm.load(path)
    // then call:    bind_weights()
    // -------------------------------------------------------------------------
    void init(ModelConfig const& c) {
        cfg = c;
        cfg.check();
        layers.resize(cfg.n_layers);
        kv_caches.resize(cfg.n_layers);
    }

    // -------------------------------------------------------------------------
    // bind_weights() — hook weight pointers after wm is populated
    // -------------------------------------------------------------------------
    void bind_weights() {
        bind_weight_pointers();
        reset_kv_cache();
    }

    // Re-derives every raw pointer this object hands out (embed_w, ln_*_w,
    // lm_head_w, and each layer's own bound pointers, including
    // TransformerBlock::wm_ptr/cfg_ptr) against *this's own wm/cfg, without
    // touching kv_caches. Split out from bind_weights() so the move
    // constructor/assignment can re-point everything after relocating wm
    // and cfg without discarding kv_caches (which it moves in directly,
    // preserving any tokens already generated) the way calling the
    // KV-cache-resetting bind_weights() would.
    void bind_weight_pointers() {
        embed_w    = &wm.get("embed.weight");
        ln_final_w = &wm.get("ln_final.weight");

        if (!cfg.use_rms_norm)
            ln_final_b = &wm.get("ln_final.bias");

        if (cfg.pos_embedding == PosEmbedding::LEARNED)
            pos_embed_w = &wm.get("pos_embed.weight");

        lm_head_w = cfg.tie_embeddings
                  ? embed_w
                  : &wm.get("lm_head.weight");

        for (int l = 0; l < cfg.n_layers; ++l) {
            std::string prefix = "layers." + std::to_string(l);
            layers[l].bind(wm, prefix, cfg);
        }
    }

    // -------------------------------------------------------------------------
    // KV-cache helpers
    // -------------------------------------------------------------------------
    void reset_kv_cache() {
        auto dtype = static_cast<KVCacheDType>(cfg.kv_cache_dtype);
        for (int l = 0; l < cfg.n_layers; ++l)
            kv_caches[l] = KVCache(cfg.max_seq_len, cfg.n_kv_heads, cfg.d_head(), dtype);
    }

    // Total resident bytes across all layers' KV caches at the current dtype.
    int64_t kv_cache_bytes() const {
        int64_t total = 0;
        for (auto const& c : kv_caches) total += c.bytes();
        return total;
    }

    int context_used() const {
        return kv_caches.empty() ? 0 : kv_caches[0].used;
    }

    // -------------------------------------------------------------------------
    // forward() — run T tokens through all layers, return logits [vocab_size]
    //             for the LAST token position.
    //
    //  use_cache = true  → append to KV cache (autoregressive decode mode)
    //  use_cache = false → full recompute, no cache update
    // -------------------------------------------------------------------------
    Tensor forward(std::vector<int> const& token_ids, bool use_cache = true) {
        int T = (int)token_ids.size();
        if (T == 0) throw std::runtime_error("TransformerModel::forward: empty input");

        // 1. Token embedding  [T, d_model]
        Tensor x = ops::embedding_lookup(*embed_w, token_ids);

        // MuP embedding scaling (IBM Granite 4.0+ "granitemoehybrid" family).
        // No-op (multiplier=1.0) for every other architecture.
        if (cfg.embedding_multiplier != 1.0f)
            ops::scale_inplace(x, cfg.embedding_multiplier);

        // 2. Positional encoding
        //    RoPE: applied inside each attention layer (not here)
        //    Sinusoidal / Learned: applied here to the embedding
        int pos_offset = use_cache ? context_used() : 0;

        if (cfg.pos_embedding == PosEmbedding::SINUSOIDAL) {
            ops::add_sinusoidal_pe(x, pos_offset);
        } else if (cfg.pos_embedding == PosEmbedding::LEARNED && pos_embed_w) {
            if (pos_offset + T > cfg.max_seq_len) {
                throw std::runtime_error("TransformerModel::forward: learned positional embeddings exceed max_seq_len");
            }
            int safe_pos = (pos_offset < 0) ? 0 : pos_offset;
            Tensor pe = pos_embed_w->row_slice(safe_pos, T);
            compute::add_inplace(x, pe);
        }
        // PosEmbedding::ROTARY: nothing to add here; RoPE runs in AttentionBlock

        // 3. Transformer layers
        // In streaming mode:
        //   - load_layer(l)  : reads layer l weights from disk into RAM
        //   - layers[l].rebind(): refreshes raw pointers (tensors may have moved)
        //   - free_layer(l)  : evicts layer l from RAM after use
        // In mmap/full mode these are all no-ops.
        for (int l = 0; l < cfg.n_layers; ++l) {
            wm.load_layer(l);           // no-op unless streaming
            // rebind() re-derives layers[l]'s raw Tensor* pointers from
            // wm_ptr (the WeightMap this layer was originally bound
            // against). Only meaningful in streaming mode, where
            // load_layer() just replaced those tensors' backing storage —
            // in mmap/full mode the tensors bound at bind_weights() time
            // never move, so this is not just unnecessary but actively
            // unsafe to call unconditionally: wm_ptr was captured as
            // &(this->wm) at bind_weights() time, and if this
            // TransformerModel was itself relocated since then (e.g.
            // returned by value from a factory function without NRVO —
            // legal and NOT guaranteed to be elided for a named local,
            // unlike a function's own prvalue return), wm_ptr is left
            // pointing at the address of the pre-move object's `wm`
            // member, not this one. Dereferencing it then reads freed
            // memory. Guarding on is_streaming() (the only mode that
            // actually needs rebind()) avoids that dereference entirely
            // in the common (non-streaming) case.
            if (wm.is_streaming())
                layers[l].rebind();     // refresh ptrs after load (streaming only)
            KVCache* cache = use_cache ? &kv_caches[l] : nullptr;
            x = layers[l].forward_v2(x, cache);
            if (wm.is_streaming() && !use_cache)
                wm.free_layer(l);       // free immediately if no KV cache needed
        }

        // 4. Final layer normalisation
        Tensor normed = like(x);
        if (cfg.use_rms_norm)
            ops::rms_norm(x, *ln_final_w, normed, cfg.layer_norm_eps);  // no IBackend equivalent
        else
            compute::layer_norm(x, *ln_final_w, *ln_final_b, normed, cfg.layer_norm_eps);

        // 5. LM head: last token row  [1, d_model] × W^T → [1, vocab_size]
        // Single-row matmul run once per generated token — small enough that
        // the host<->device transfer cost of routing it to GPU (each
        // compute:: call does its own independent transfer; see
        // sycl_backend.hpp's per-op residency limitation) may outweigh the
        // benefit versus just running it on CPU. Routed anyway for
        // consistency now that a GPU backend, once selected, drives the
        // whole forward pass; revisit if profiling shows this specific call
        // is a bottleneck.
        Tensor last   = normed.row_slice(T - 1, 1);
        Tensor logits = compute::linear(last, *lm_head_w);

        // MuP logits scaling (IBM Granite 4.0+): divide AFTER the LM head
        // matmul, matching modeling_granitemoehybrid.py's
        // "logits = logits / self.config.logits_scaling" exactly.
        // No-op (multiplier=1.0) for every other architecture.
        if (cfg.logits_scaling != 1.0f)
            ops::scale_inplace(logits, 1.0f / cfg.logits_scaling);

        return logits.view({cfg.vocab_size});
    }

    // -------------------------------------------------------------------------
    // GenerationResult
    // -------------------------------------------------------------------------
    struct GenerationResult {
        std::vector<int> token_ids;          // prompt + all generated tokens
        std::string      text;               // decoded (if tokenizer supplied)
        int              tokens_generated{0};
        bool             hit_eos{false};
    };

    // -------------------------------------------------------------------------
    // generate() — prefill + autoregressive decode loop
    // -------------------------------------------------------------------------
    GenerationResult generate(std::vector<int> const& prompt_ids,
                               GenerationConfig const& gc,
                               Tokenizer const*        tok = nullptr)
    {
        GenerationResult result;
        reset_kv_cache();

        // ---- Prefill: run full prompt to populate the KV cache
        if (!prompt_ids.empty())
            forward(prompt_ids, true);

        int last_token = prompt_ids.empty()
                       ? cfg.bos_token_id
                       : prompt_ids.back();
        result.token_ids = prompt_ids;

        // Histogram of GENERATED tokens only (excludes the prompt — see
        // GenerationConfig's field comments for why), sized to vocab so the
        // penalty loop can index it directly without a hash map.
        std::vector<int> token_counts(cfg.vocab_size, 0);
        std::vector<int> generated_ids; // generated-only, for n-gram blocking
        generated_ids.reserve(gc.max_new_tokens);

        // Tail of decoded text, used to check stop_strings without
        // re-decoding the whole sequence every step. Only the longest
        // stop string's length of history needs to be kept.
        std::string decoded_tail;
        size_t max_stop_len = 0;
        for (auto const& s : gc.stop_strings) max_stop_len = std::max(max_stop_len, s.size());
        bool hit_stop = false;
        size_t stop_match_len = 0;

        // ---- Decode loop
        for (int step = 0; step < gc.max_new_tokens; ++step) {
            if (context_used() >= cfg.max_seq_len) break;

            // Streaming decode: free the last active layer between tokens.
            // Each token causes all layers to be loaded one by one (in forward()),
            // so peak RAM = size of one layer, not the whole model.
            if (wm.is_streaming() && wm.streamed_layer >= 0)
                wm.free_layer(wm.streamed_layer);

            // Single-token forward; each layer is load→rebind→run→free inside
            Tensor logits = forward({last_token}, true);

            sampler::apply_repetition_penalties(logits, token_counts,
                                                 gc.repetition_penalty,
                                                 gc.frequency_penalty,
                                                 gc.presence_penalty);
            sampler::block_repeated_ngrams(logits, generated_ids, gc.no_repeat_ngram_size);

            int next_id = gc.do_sample
                ? sampler::sample(logits, gc.temperature, gc.top_p, gc.top_k, rng)
                : sampler::argmax(logits);

            result.token_ids.push_back(next_id);
            generated_ids.push_back(next_id);
            if (next_id >= 0 && next_id < (int)token_counts.size())
                token_counts[next_id]++;
            result.tokens_generated++;
            last_token = next_id;

            // Optional streaming callback — returning false stops generation
            if (gc.on_token && !gc.on_token(next_id)) break;

            if (next_id == cfg.eos_token_id) {
                result.hit_eos = true;
                break;
            }

            // Stop-string check: decode just this token's text and append
            // to a rolling tail, trimmed to the longest stop string's
            // length so this stays O(1) per step instead of re-decoding
            // the whole sequence. Requires a tokenizer; silently skipped
            // (not an error) if tok == nullptr — see GenerationConfig.
            if (tok && !gc.stop_strings.empty()) {
                decoded_tail += tok->decode_token(next_id);
                if (decoded_tail.size() > max_stop_len * 2 + 64) {
                    // Keep the tail bounded; +64 slack avoids trimming
                    // mid-match on multi-byte UTF-8 sequences.
                    decoded_tail.erase(0, decoded_tail.size() - (max_stop_len * 2 + 64));
                }
                for (auto const& stop : gc.stop_strings) {
                    if (stop.empty()) continue;
                    auto pos = decoded_tail.rfind(stop);
                    if (pos != std::string::npos && pos + stop.size() == decoded_tail.size()) {
                        hit_stop = true;
                        stop_match_len = stop.size();
                        break;
                    }
                }
                if (hit_stop) break;
            }
        }

        if (tok) {
            result.text = tok->decode(result.token_ids);
            // Trim the matched stop string itself off the end of the
            // decoded text so callers don't have to strip it themselves —
            // it's an instruction to STOP, not content the caller asked
            // for. Token ids are left as-is (the stop string may span a
            // token boundary oddly and there's no clean way to "un-append"
            // a partial token), only the text is trimmed.
            if (hit_stop && result.text.size() >= stop_match_len) {
                result.text.erase(result.text.size() - stop_match_len);
            }
        }
        return result;
    }

    // Convenience overload — uses generation defaults from cfg
    GenerationResult generate(std::vector<int> const& prompt_ids,
                               Tokenizer const*        tok = nullptr)
    {
        GenerationConfig gc;
        gc.max_new_tokens = cfg.max_new_tokens;
        gc.temperature    = cfg.temperature;
        gc.top_p          = cfg.top_p;
        gc.top_k          = cfg.top_k;
        gc.do_sample      = cfg.do_sample;
        return generate(prompt_ids, gc, tok);
    }
};

// -----------------------------------------------------------------------------
// WeightMap::random_init — defined here (after ModelConfig/PosEmbedding complete)
// -----------------------------------------------------------------------------
inline WeightMap WeightMap::random_init(ModelConfig const& cfg) {
    WeightMap wm;
    std::mt19937 rng(1234);
    float init_std = 0.02f;

    // Helper: create a tensor with given shape and optional fill
    auto add = [&](std::string const& key, Shape const& s,
                   bool randn = false, float std_dev = 0.02f, float fill_v = 0.f)
    {
        Tensor t(s);
        if (randn) {
            std::normal_distribution<float> dist(0.f, std_dev);
            float* p = t.data_ptr();
            for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
        } else {
            t.fill(fill_v);
        }
        wm.tensors[key] = std::move(t);
    };

    // Embeddings
    add("embed.weight", {cfg.vocab_size, cfg.d_model}, true, init_std);
    if (cfg.pos_embedding == PosEmbedding::LEARNED)
        add("pos_embed.weight", {cfg.max_seq_len, cfg.d_model}, true, init_std);

    int d_kv = cfg.n_kv_heads * cfg.d_head();

    // Per-layer weights
    for (int l = 0; l < cfg.n_layers; ++l) {
        std::string pfx = "layers." + std::to_string(l);

        // Layer norms: weight=1, bias=0
        add(pfx+".ln1.weight", {cfg.d_model}, false, 0.f, 1.f);
        add(pfx+".ln1.bias",   {cfg.d_model});
        add(pfx+".ln2.weight", {cfg.d_model}, false, 0.f, 1.f);
        add(pfx+".ln2.bias",   {cfg.d_model});

        // Attention projections
        add(pfx+".attn.q_proj.weight", {cfg.d_model, cfg.d_model}, true, init_std);
        add(pfx+".attn.k_proj.weight", {d_kv,        cfg.d_model}, true, init_std);
        add(pfx+".attn.v_proj.weight", {d_kv,        cfg.d_model}, true, init_std);
        add(pfx+".attn.o_proj.weight", {cfg.d_model, cfg.d_model}, true, init_std);
        add(pfx+".attn.q_proj.bias",   {cfg.d_model});
        add(pfx+".attn.k_proj.bias",   {d_kv});
        add(pfx+".attn.v_proj.bias",   {d_kv});
        add(pfx+".attn.o_proj.bias",   {cfg.d_model});

        // Feed-forward
        add(pfx+".ffn.fc1.weight", {cfg.d_ff,    cfg.d_model}, true, init_std);
        add(pfx+".ffn.fc2.weight", {cfg.d_model, cfg.d_ff},    true, init_std);
        add(pfx+".ffn.fc1.bias",   {cfg.d_ff});
        add(pfx+".ffn.fc2.bias",   {cfg.d_model});
    }

    // Final layer norm
    add("ln_final.weight", {cfg.d_model}, false, 0.f, 1.f);
    add("ln_final.bias",   {cfg.d_model});

    // LM head (only when NOT tied to embedding)
    if (!cfg.tie_embeddings)
        add("lm_head.weight", {cfg.vocab_size, cfg.d_model}, true, init_std);

    return wm;
}

} // namespace vlrf
