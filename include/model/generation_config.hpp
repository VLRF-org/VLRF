// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — GenerationConfig  (generation_config.hpp)
// Extracted here so transformer.hpp and vlrf_api.cpp can both include it
// without circular dependency.
// =============================================================================
#include <functional>
#include <string>
#include <vector>

namespace vlrf {

struct GenerationConfig {
    int   max_new_tokens = 128;
    float temperature    = 0.8f;
    float top_p          = 0.9f;
    int   top_k          = 50;
    bool  do_sample      = true;

    // ---- Repetition control ------------------------------------------------
    // Applied to raw logits, before temperature/top-k/top-p, in this order:
    // repetition_penalty first (multiplicative, CTRL/HF-style), then
    // frequency_penalty and presence_penalty (additive, OpenAI-style).
    // All are computed over the token counts of the CURRENTLY GENERATED
    // sequence only (not the prompt), matching common inference-server
    // convention — penalizing prompt tokens tends to fight instruction
    // formats and role tags baked into the prompt.
    //
    //   repetition_penalty : 1.0 = no effect (default, off). >1.0 discourages
    //     repeats by dividing the logit of any token seen so far by this
    //     factor if positive, multiplying if negative (CTRL paper approach:
    //     logit>0 -> logit/penalty, logit<=0 -> logit*penalty). Typical
    //     useful range 1.05–1.3; values much above ~1.5 degrade coherence.
    //   frequency_penalty : 0.0 = no effect (default, off). Subtracts
    //     frequency_penalty * count(token) from that token's logit — scales
    //     with HOW OFTEN a token has repeated, so frequent repeats are
    //     suppressed harder than a single-use token. Typical range 0.0–2.0.
    //   presence_penalty : 0.0 = no effect (default, off). Subtracts a FLAT
    //     presence_penalty from any token's logit if it has appeared at all
    //     (count > 0), regardless of how many times — encourages introducing
    //     new tokens rather than suppressing repeats proportionally. Typical
    //     range 0.0–2.0.
    float repetition_penalty = 1.0f;
    float frequency_penalty  = 0.0f;
    float presence_penalty   = 0.0f;

    // Blocks any repeat of an n-gram that has already occurred among the
    // generated tokens (checked against the last no_repeat_ngram_size-1
    // generated tokens as context). 0 = disabled (default). Complements the
    // penalties above: penalties make repeats less likely, this makes exact
    // n-gram loops impossible. Common values: 3 or 4. Applied only to
    // GENERATED tokens (matching the penalty fields above), and only once
    // at least no_repeat_ngram_size tokens have been generated.
    int no_repeat_ngram_size = 0;

    // ---- Stop sequences -----------------------------------------------------
    // Generation halts as soon as the decoded text generated so far ends
    // with any of these strings (checked after each token is decoded via
    // Tokenizer::decode_token — cheap, no need to re-decode the whole
    // sequence). The matched stop string is NOT included in
    // GenerationResult::text or token_ids beyond the token(s) that produced
    // it — see TransformerModel::generate() for the exact truncation
    // behavior. Requires a tokenizer to be supplied to generate(); ignored
    // (with no effect, not an error) if tok == nullptr, since token-only
    // generation has no text to match against.
    std::vector<std::string> stop_strings;

    // Optional streaming callback: called after each new token is sampled.
    // Returning false stops generation early.
    std::function<bool(int /*token_id*/)> on_token;
};

} // namespace vlrf
