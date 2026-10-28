// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Tokenizer fixes  (test_tokenizer_fixes.cpp)
// Verifies the tokenizer byte-escaping fix and the O(1) merge-rank
// performance fix.
// =============================================================================
#include "tokenizer.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <cassert>

using namespace vlrf;

static int g_pass = 0, g_fail = 0;
#define CHECK(label, cond) do { \
    if(cond){printf("  PASS  %s\n",label);++g_pass;} \
    else{printf("  FAIL  %s\n",label);++g_fail;} \
} while(0)

// -----------------------------------------------------------------------------
// Test 1: byte-escaping detection and round-trip
// -----------------------------------------------------------------------------
void test_byte_escaping_roundtrip() {
    printf("\n=== Test 1: byte-escaping round-trip ===\n");

    Tokenizer tok;
    // Build a GPT-2-style vocab: space -> 'Ġ' (U+0120), no literal space
    auto add = [&](std::string s) {
        tok.vocab_index[s] = (int)tok.vocab.size();
        tok.vocab.push_back(s);
        };
    add("<pad>"); add("<bos>"); add("<eos>"); add("<unk>");
    for (char c : std::string("intmach(){")) add(std::string(1, c));
    std::string gspace = bpe_byte_escape::byte_to_escaped_utf8(' ');
    add(gspace);
    add("in"); add("int");
    add(gspace + "m"); add(gspace + "main");

    tok.pad_id = 0; tok.bos_id = 1; tok.eos_id = 2; tok.unk_id = 3;
    tok.merges = {
        {tok.vocab_index["i"], tok.vocab_index["n"]},
        {tok.vocab_index["in"], tok.vocab_index["t"]},
        {tok.vocab_index[gspace], tok.vocab_index["m"]},
        {tok.vocab_index[gspace + "m"], tok.vocab_index["a"]},
    };
    for (size_t i = 0;i < tok.merges.size();++i) {
        auto [a, b] = tok.merges[i];
        tok.merge_rank[{a, b}] = (int)i;
        std::string merged = tok.vocab[a] + tok.vocab[b];
        auto it = tok.vocab_index.find(merged);
        if (it != tok.vocab_index.end()) tok.merge_map[{a, b}] = it->second;
    }
    tok.detect_byte_escaping();

    CHECK("uses_byte_escaping auto-detected true", tok.uses_byte_escaping);

    std::string original = "int main() {";
    auto ids = tok.encode(original, false);
    std::string decoded = tok.decode(ids, true);

    CHECK("encode produces non-trivial token count", ids.size() > 0 && ids.size() < original.size());
    CHECK("round-trip encode->decode matches original", decoded == original);

    // Verify space byte specifically resolves correctly (not <unk>)
    bool any_unk = false;
    for (int id : ids) if (id == tok.unk_id) any_unk = true;
    CHECK("no <unk> tokens produced for simple ASCII text", !any_unk);
}

// -----------------------------------------------------------------------------
// Test 2: backward compatibility -- plain byte vocab (no escaping)
// -----------------------------------------------------------------------------
void test_plain_vocab_backward_compat() {
    printf("\n=== Test 2: plain vocab backward compatibility ===\n");

    Tokenizer tok;
    tok.init_byte_vocab();

    CHECK("plain byte vocab: uses_byte_escaping is false", !tok.uses_byte_escaping);

    std::string original = "Hello, World! int main() {";
    auto ids = tok.encode(original, false);
    std::string decoded = tok.decode(ids, false);

    CHECK("plain vocab round-trip matches (byte-for-byte, no merges)", decoded == original);
    CHECK("plain vocab: 1 token per byte (no merges defined)", ids.size() == original.size());
}

// -----------------------------------------------------------------------------
// Test 3: make_char_level still works correctly
// -----------------------------------------------------------------------------
void test_char_level_tokenizer() {
    printf("\n=== Test 3: make_char_level ===\n");

    std::string chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    Tokenizer tok = Tokenizer::make_char_level(chars);

    CHECK("char-level: uses_byte_escaping false", !tok.uses_byte_escaping);

    std::string text = "Hello World";
    auto ids = tok.encode(text, false);
    for (int id : ids) CHECK("char-level: no unk for known chars", id != tok.unk_id);
    std::string back = tok.decode(ids, true);
    CHECK("char-level round-trip", back == text);
}

// -----------------------------------------------------------------------------
// Test 4: performance -- O(1) rank lookup at realistic scale
// -----------------------------------------------------------------------------
void test_merge_performance() {
    printf("\n=== Test 4: O(1) merge-rank performance at scale ===\n");

    Tokenizer tok;
    // Build a large synthetic vocab: enough base chars + many merge levels
    // to exercise realistic BPE reduction depth.
    auto add = [&](std::string s) {
        if (tok.vocab_index.count(s)) return;
        tok.vocab_index[s] = (int)tok.vocab.size();
        tok.vocab.push_back(s);
        };
    add("<pad>"); add("<bos>"); add("<eos>"); add("<unk>");
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    for (char c : alphabet) add(std::string(1, c));

    // Build a large merge chain: a,b->ab ab,c->abc etc, PLUS many
    // unrelated filler merges to simulate a 50,000+ entry real merge table.
    std::mt19937 rng(1);
    for (int i = 0; i < 50000; ++i) {
        int a = rng() % (int)tok.vocab.size();
        int b = rng() % (int)tok.vocab.size();
        std::string merged = tok.vocab[a] + tok.vocab[b];
        if (merged.size() > 20) continue;  // avoid runaway string growth
        add(merged);
        tok.merges.push_back({ a, b });
    }
    for (size_t i = 0;i < tok.merges.size();++i) {
        auto [a, b] = tok.merges[i];
        tok.merge_rank[{a, b}] = (int)i;
        std::string merged = tok.vocab[a] + tok.vocab[b];
        auto it = tok.vocab_index.find(merged);
        if (it != tok.vocab_index.end()) tok.merge_map[{a, b}] = it->second;
    }
    tok.detect_byte_escaping();

    printf("  Built synthetic tokenizer: %d vocab, %zu merges\n",
        tok.vocab_size(), tok.merges.size());

    std::string prompt = "the quick brown fox jumps over the lazy dog and runs away into the forest";

    auto t0 = std::chrono::steady_clock::now();
    auto ids = tok.encode(prompt, false);
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Encoded %zu-char prompt into %zu tokens in %.3f ms\n",
        prompt.size(), ids.size(), ms);

    // With O(1) rank lookup this should be near-instant (<50ms even with
    // 50k merges); the old O(n*m) linear scan would take seconds to
    // minutes at this merge-table size for even a short prompt.
    CHECK("encoding completes in well under 1 second (O(1) rank lookup working)",
        ms < 1000.0);
}

int main() {
    printf("=== VLRF Tokenizer Fix Verification ===\n");

    test_byte_escaping_roundtrip();
    test_plain_vocab_backward_compat();
    test_char_level_tokenizer();
    test_merge_performance();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}