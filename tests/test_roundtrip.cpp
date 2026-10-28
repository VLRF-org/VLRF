// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Tokenizer round-trip  (test_roundtrip.cpp)
// =============================================================================
#include "tokenizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <tokenizer.nft> [\"text\"]\n", argv[0]);
        return 1;
    }

    std::string original = (argc >= 3) ? argv[2] : "int main() {";

    vlrf::Tokenizer tok;
    tok.load(argv[1]);

    auto ids = tok.encode(original, false);
    std::string decoded = tok.decode(ids, true);

    printf("Original: %s\n", original.c_str());
    printf("Decoded:  %s\n", decoded.c_str());
    printf("Round-trip match: %s\n",
        (original == decoded) ? "PASS" : "FAIL (see diff below)");

    if (original != decoded) {
        printf("Original bytes: ");
        for (unsigned char c : original) printf("%02X ", c);
        printf("\nDecoded  bytes: ");
        for (unsigned char c : decoded) printf("%02X ", c);
        printf("\n");
    }

    return (original == decoded) ? 0 : 1;
}