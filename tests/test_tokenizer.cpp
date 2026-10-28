// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Tokenizer  (test_tokenizer.cpp)
// =============================================================================
#include "tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#ifdef _WIN32
#  include <windows.h>
#endif

// Cross-platform temp path helper
static std::string tmp_path(const char* filename) {
#ifdef _WIN32
    char buf[512];
    DWORD len = GetTempPathA(sizeof(buf), buf);
    if (len == 0) return std::string(".\\") + filename;
    return std::string(buf) + filename;
#else
    return std::string("/tmp/") + filename;
#endif
}

using namespace vlrf;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if(!(cond)){std::cerr<<"FAIL ["<<__LINE__<<"]: "#cond"\n";++g_fail;} \
    else ++g_pass; } while(0)

// -----------------------------------------------------------------------------
void test_byte_vocab_init() {
    Tokenizer tok;
    tok.init_byte_vocab();
    CHECK(tok.vocab_size() == 4 + 256);   // 4 specials + 256 byte tokens
    CHECK(tok.pad_id == 0);
    CHECK(tok.bos_id == 1);
    CHECK(tok.eos_id == 2);
    CHECK(tok.unk_id == 3);
}

void test_special_token_names() {
    Tokenizer tok;
    tok.init_byte_vocab();
    CHECK(tok.vocab[0] == "<pad>");
    CHECK(tok.vocab[1] == "<bos>");
    CHECK(tok.vocab[2] == "<eos>");
    CHECK(tok.vocab[3] == "<unk>");
}

void test_encode_decode_roundtrip() {
    Tokenizer tok;
    tok.init_byte_vocab();
    std::string text = "Hello, World!";
    auto ids = tok.encode(text, false);
    CHECK(ids.size() == text.size());  // byte-level: 1 id per byte
    std::string decoded = tok.decode(ids, false);
    CHECK(decoded == text);
}

void test_encode_add_bos() {
    Tokenizer tok;
    tok.init_byte_vocab();
    auto ids = tok.encode("Hi", true);
    CHECK(ids[0] == tok.bos_id);
    CHECK((int)ids.size() == 3);  // <bos> + 'H' + 'i'
}

void test_decode_skip_special() {
    Tokenizer tok;
    tok.init_byte_vocab();
    std::vector<int> ids = {tok.bos_id, (int)'A'+4, (int)'B'+4, tok.eos_id};
    // skip_special=true should hide bos/eos
    std::string out = tok.decode(ids, true);
    CHECK(out.find("<bos>") == std::string::npos);
    CHECK(out.find("<eos>") == std::string::npos);
}

void test_decode_token() {
    Tokenizer tok;
    tok.init_byte_vocab();
    CHECK(tok.decode_token(0) == "<pad>");
    CHECK(tok.decode_token(1) == "<bos>");
    CHECK(tok.decode_token(-1) == "<unk>");
    CHECK(tok.decode_token(9999) == "<unk>");
}

void test_char_level_tokenizer() {
    std::string chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    Tokenizer tok = Tokenizer::make_char_level(chars);
    std::string text = "Hello World";
    auto ids = tok.encode(text, false);
    // Every character in text should be encodable (no unk for ascii letters/space)
    for (int id : ids) CHECK(id != tok.unk_id);
    std::string back = tok.decode(ids, true);
    CHECK(back == text);
}

void test_save_load() {
    Tokenizer tok;
    tok.init_byte_vocab();
    std::string path = tmp_path("test_tok.nft");
    tok.save(path);

    Tokenizer tok2;
    tok2.load(path);
    CHECK(tok2.vocab_size() == tok.vocab_size());
    CHECK(tok2.pad_id == tok.pad_id);
    CHECK(tok2.bos_id == tok.bos_id);
    CHECK(tok2.eos_id == tok.eos_id);

    // Verify a few vocab entries match
    for (int i = 0; i < std::min(10, tok.vocab_size()); ++i)
        CHECK(tok2.vocab[i] == tok.vocab[i]);
}

void test_vocab_index_lookup() {
    Tokenizer tok;
    tok.init_byte_vocab();
    auto it = tok.vocab_index.find("<eos>");
    CHECK(it != tok.vocab_index.end());
    CHECK(it->second == tok.eos_id);
}

void test_encode_empty_string() {
    Tokenizer tok;
    tok.init_byte_vocab();
    auto ids = tok.encode("", false);
    CHECK(ids.empty());
    auto ids_bos = tok.encode("", true);
    CHECK(ids_bos.size() == 1 && ids_bos[0] == tok.bos_id);
}

// -----------------------------------------------------------------------------
int main() {
    std::cout << "=== VLRF Tokenizer Tests ===\n";
    test_byte_vocab_init();
    test_special_token_names();
    test_encode_decode_roundtrip();
    test_encode_add_bos();
    test_decode_skip_special();
    test_decode_token();
    test_char_level_tokenizer();
    test_save_load();
    test_vocab_index_lookup();
    test_encode_empty_string();
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
