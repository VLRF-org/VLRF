// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Real tokenizer encode  (test_real_tokenizer_encode.cpp)
// Encodes a prompt with a real .nft file and inspects the resulting token
// ids, byte-escaping detection, and per-byte vocabulary lookup — useful for
// comparing against the real HuggingFace tokenizer's output for the same
// prompt.
// Usage:
//   test_real_tokenizer_encode.exe <tokenizer.nft> ["prompt text"]
// =============================================================================
#include "tokenizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <tokenizer.nft> [\"prompt text\"]\n", argv[0]);
        return 1;
    }

    std::string prompt = (argc >= 3) ? argv[2] : "int main() {";

    vlrf::Tokenizer tok;
    tok.load(argv[1]);
    printf("vocab_size=%d n_merges=%zu\n\n", tok.vocab_size(), tok.merges.size());

    auto it_space = tok.vocab_index.find(" ");
    printf("Literal space ' ' in vocab_index: %s",
        it_space != tok.vocab_index.end() ? "YES" : "NO");
    if (it_space != tok.vocab_index.end()) printf("  (id=%d)", it_space->second);
    printf("\n");

    std::string gpt2_space = vlrf::bpe_byte_escape::byte_to_escaped_utf8(' ');
    auto it_gspace = tok.vocab_index.find(gpt2_space);
    printf("GPT-2 escaped space (Ġ) in vocab_index: %s",
        it_gspace != tok.vocab_index.end() ? "YES" : "NO");
    if (it_gspace != tok.vocab_index.end()) printf("  (id=%d)", it_gspace->second);
    printf("\n\n");

    printf("uses_byte_escaping = %s\n\n", tok.uses_byte_escaping ? "true" : "false");

    auto ids = tok.encode(prompt, false);

    printf("Encoding \"%s\":\n", prompt.c_str());
    printf("Token count: %zu\n", ids.size());
    printf("Token ids: [");
    for (size_t i = 0; i < ids.size(); ++i)
        printf("%d%s", ids[i], i + 1 < ids.size() ? ", " : "");
    printf("]\n\n");

    printf("Decoded tokens:\n");
    for (size_t i = 0; i < ids.size() && i < 30; ++i)
        printf("  id=%d -> %s\n", ids[i], tok.decode_token(ids[i]).c_str());

    std::string decoded = tok.decode(ids, true);
    printf("\nFull decode: \"%s\"\n", decoded.c_str());
    printf("Round-trip match: %s\n", (decoded == prompt) ? "PASS" : "FAIL");

    printf("\nIndividual raw byte -> initial token id mapping\n");
    printf("(using the SAME escaping logic as the real encode() path):\n");
    for (unsigned char c : prompt) {
        std::string s = tok.uses_byte_escaping
            ? vlrf::bpe_byte_escape::byte_to_escaped_utf8(c)
            : std::string(1, (char)c);
        auto it = tok.vocab_index.find(s);
        printf("  byte 0x%02X ('%c') -> %s\n", c, (c >= 32 && c < 127) ? c : '.',
            it != tok.vocab_index.end() ?
            ("id=" + std::to_string(it->second)).c_str() : "NOT FOUND (unk)");
    }

    return 0;
}