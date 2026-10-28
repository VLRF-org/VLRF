// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — BPE Tokenizer  (tokenizer.hpp)
// Byte-Pair Encoding tokenizer compatible with GPT-2 / LLaMA / tiktoken
// -derived style vocabularies.
//
// File format (.nft):
//   [magic:u32] [version:u32] [vocab_size:u32] [pad:u32]
//   Vocab entries:  [len:u16][bytes:u8*len]    (index = position)
//   Merge table:    [n_merges:u32]
//                   per merge: [a:u32][b:u32]   (merge(a,b) → a+1 in vocab)
//
// -----------------------------------------------------------------------------
// GPT-2 / tiktoken-style byte<->unicode escaping
// -----------------------------------------------------------------------------
// Many modern BPE tokenizers (GPT-2, GPT-NeoX, IBM Granite, and most
// tiktoken-derived vocabularies) do not store raw bytes directly in their
// vocabulary. Bytes that would be visually ambiguous or unsafe as literal
// vocabulary strings — space (0x20), newline (0x0A), and other control
// characters — are instead mapped to unused Unicode codepoints (space
// becomes U+0120 'Ġ', newline becomes U+010A 'Ċ', etc.) so every token can
// be represented as an ordinary, printable string.
//
// This mapping MUST be applied to raw input bytes before vocabulary lookup
// during encode(), and reversed during decode(). Skipping this step causes
// every space/newline/control-byte in real text to fail vocabulary lookup
// (falling through to <unk>), which in turn prevents nearly every
// word-boundary BPE merge from ever firing — silently producing a
// completely different (and wrong) token sequence for any real sentence,
// even though the vocabulary and merge table both load and "look" correct
// on inspection.
//
// Not every tokenizer uses this scheme (some genuinely store raw bytes),
// so VLRF auto-detects it per-tokenizer: if the vocabulary contains
// the escaped space form ('Ġ') but NOT a literal space character, byte
// escaping is assumed to be in use.
// =============================================================================
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <functional>
#include <array>
#include <cstring>

namespace vlrf {

    static constexpr uint32_t NFT_MAGIC = 0x4E465446; // "NFTF"
    static constexpr uint32_t NFT_VERSION = 1;

    // Pair hash for unordered_map
    struct PairHash {
        size_t operator()(std::pair<int, int> const& p) const {
            return std::hash<int64_t>()((int64_t)p.first << 32 | (uint32_t)p.second);
        }
    };

    // -------------------------------------------------------------------------
    // GPT-2 / tiktoken-style byte<->unicode escaping tables
    // -------------------------------------------------------------------------
    namespace bpe_byte_escape {

        // Byte -> unicode codepoint mapping (identical construction to
        // HuggingFace's / OpenAI's canonical `bytes_to_unicode()`).
        inline std::array<int, 256> const& byte_to_unicode_table() {
            static std::array<int, 256> table = [] {
                std::array<int, 256> t{};
                std::vector<int> bs;
                for (int b = (int)'!'; b <= (int)'~'; ++b) bs.push_back(b);
                for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
                for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
                std::vector<int> cs = bs;
                int n = 0;
                for (int b = 0; b < 256; ++b) {
                    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                        bs.push_back(b);
                        cs.push_back(256 + n);
                        ++n;
                    }
                }
                for (size_t i = 0; i < bs.size(); ++i) t[bs[i]] = cs[i];
                return t;
                }();
            return table;
        }

        // Encode a single raw byte as its escaped UTF-8 string form.
        inline std::string byte_to_escaped_utf8(unsigned char b) {
            int cp = byte_to_unicode_table()[b];
            std::string out;
            if (cp < 0x80) {
                out += (char)cp;
            }
            else {
                // cp is always < 0x400 for this table, so at most 2 UTF-8 bytes.
                out += (char)(0xC0 | (cp >> 6));
                out += (char)(0x80 | (cp & 0x3F));
            }
            return out;
        }

        // Reverse mapping: unicode codepoint -> original byte value.
        inline std::unordered_map<int, int> const& unicode_to_byte_table() {
            static std::unordered_map<int, int> table = [] {
                std::unordered_map<int, int> t;
                auto const& fwd = byte_to_unicode_table();
                for (int b = 0; b < 256; ++b) t[fwd[b]] = b;
                return t;
                }();
            return table;
        }

    } // namespace bpe_byte_escape

    // -------------------------------------------------------------------------
    // GPT-2 regex pre-tokenizer
    // -------------------------------------------------------------------------
    // Before byte-level BPE ever runs, GPT-2 (and every tiktoken-derived
    // vocabulary descended from it) first splits the input into chunks using
    // the pattern:
    //
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // BPE merges are then applied independently within each chunk -- merges
    // never cross a chunk boundary. Skipping this step doesn't break vocab
    // lookup (byte-escaping already handles that), but it silently produces
    // a different token stream than the reference tokenizer: e.g. "Hello,
    // world" and "Hello,  world" (two spaces) will merge differently, and
    // punctuation runs attach to the wrong side of a word, because nothing
    // stops a merge from spanning a word/punctuation/whitespace boundary.
    //
    // `\p{L}` / `\p{N}` require actual Unicode letter/number classification,
    // which std::regex (byte-oriented, no Unicode property escapes) cannot
    // express -- so this is a small hand-rolled UTF-8-aware scanner, which
    // is also how tiktoken/HF implement this internally (not a generic
    // regex engine).
    namespace gpt2_pretokenize {

        // Decode one UTF-8 codepoint starting at s[i]. Returns the codepoint
        // and advances `len` to the number of bytes consumed. Malformed
        // sequences are treated as a single raw byte (fail-safe passthrough,
        // matching unescape_bytes()'s philosophy elsewhere in this file).
        inline int decode_utf8(std::string const& s, size_t i, int& len) {
            unsigned char c0 = (unsigned char)s[i];
            if (c0 < 0x80) { len = 1; return c0; }
            if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
                len = 2;
                return ((c0 & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
            }
            if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
                len = 3;
                return ((c0 & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6)
                     | ((unsigned char)s[i + 2] & 0x3F);
            }
            if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
                len = 4;
                return ((c0 & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12)
                     | (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
            }
            len = 1;
            return c0;
        }

        // Sorted, non-overlapping codepoint range tables approximating the
        // Unicode general categories used by the GPT-2 pre-tokenizer regex.
        // They cover every major script block (Latin incl. extended, Greek,
        // Cyrillic, Armenian, Hebrew, Arabic, Indic, Thai/Lao/Tibetan,
        // Georgian, Hangul, Kana, Bopomofo, CJK incl. supplementary
        // ideographs, fullwidth forms). Looked up by binary search.
        struct CpRange { int lo, hi; };

        inline bool in_ranges(CpRange const* r, size_t n, int cp) {
            size_t lo = 0, hi = n;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (cp < r[mid].lo)      hi = mid;
                else if (cp > r[mid].hi) lo = mid + 1;
                else return true;
            }
            return false;
        }

        // Approximation of Unicode \p{L} (letter).
        inline bool is_letter(int cp) {
            if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
            if (cp < 0x80) return false; // other ASCII: digits/punct handled elsewhere
            static const CpRange kLetters[] = {
                {0x00AA,0x00AA},{0x00B5,0x00B5},{0x00BA,0x00BA},
                {0x00C0,0x00D6},{0x00D8,0x00F6},{0x00F8,0x02C1},  // Latin-1 .. Latin Ext-B, IPA
                {0x02C6,0x02D1},{0x02E0,0x02E4},{0x02EC,0x02EC},{0x02EE,0x02EE},
                {0x0370,0x0374},{0x0376,0x0377},{0x037A,0x037D},{0x037F,0x037F},
                {0x0386,0x0386},{0x0388,0x03F5},{0x03F7,0x0481},  // Greek, Cyrillic
                {0x048A,0x052F},                                  // Cyrillic ext + supplement
                {0x0531,0x0556},{0x0560,0x0588},                  // Armenian
                {0x05D0,0x05EA},{0x05EF,0x05F2},                  // Hebrew
                {0x0620,0x064A},{0x066E,0x066F},{0x0671,0x06D3},  // Arabic
                {0x06D5,0x06D5},{0x06E5,0x06E6},{0x06EE,0x06EF},{0x06FA,0x06FC},{0x06FF,0x06FF},
                {0x0710,0x0710},{0x0712,0x072F},{0x074D,0x07A5},  // Syriac, Thaana
                {0x07B1,0x07B1},{0x07CA,0x07EA},                  // NKo
                {0x0800,0x0815},{0x0840,0x0858},{0x0860,0x086A},{0x08A0,0x08C9},
                {0x0904,0x0939},{0x093D,0x093D},{0x0950,0x0950},  // Devanagari
                {0x0958,0x0961},{0x0971,0x0980},
                {0x0985,0x09B9},{0x09BD,0x09BD},{0x09CE,0x09CE},{0x09DC,0x09E1},{0x09F0,0x09F1}, // Bengali
                {0x0A05,0x0A39},{0x0A59,0x0A5E},{0x0A72,0x0A74},  // Gurmukhi
                {0x0A85,0x0AB9},{0x0ABD,0x0ABD},{0x0AD0,0x0AE1},  // Gujarati
                {0x0B05,0x0B39},{0x0B3D,0x0B3D},{0x0B5C,0x0B61},{0x0B71,0x0B71}, // Oriya
                {0x0B83,0x0BB9},{0x0BD0,0x0BD0},                  // Tamil
                {0x0C05,0x0C39},{0x0C3D,0x0C3D},{0x0C58,0x0C61},{0x0C80,0x0C80}, // Telugu
                {0x0C85,0x0CB9},{0x0CBD,0x0CBD},{0x0CDD,0x0CE1},{0x0CF1,0x0CF2}, // Kannada
                {0x0D05,0x0D3A},{0x0D3D,0x0D3D},{0x0D4E,0x0D4E},{0x0D54,0x0D56},{0x0D5F,0x0D61},{0x0D7A,0x0D7F}, // Malayalam
                {0x0D85,0x0DC6},                                  // Sinhala
                {0x0E01,0x0E30},{0x0E32,0x0E33},{0x0E40,0x0E46},  // Thai
                {0x0E81,0x0EB0},{0x0EB2,0x0EB3},{0x0EBD,0x0EC6},  // Lao
                {0x0EDC,0x0EDF},{0x0F00,0x0F00},{0x0F40,0x0F6C},{0x0F88,0x0F8C}, // Tibetan
                {0x1000,0x102A},{0x103F,0x103F},{0x1050,0x1055},  // Myanmar
                {0x10A0,0x10C5},{0x10C7,0x10C7},{0x10CD,0x10CD},  // Georgian
                {0x10D0,0x10FA},{0x10FC,0x1248},                  // Georgian, Hangul Jamo, Ethiopic
                {0x124A,0x135A},{0x1380,0x138F},{0x13A0,0x13F5},{0x13F8,0x13FD}, // Ethiopic, Cherokee
                {0x1401,0x166C},{0x166F,0x167F},{0x1681,0x169A},  // Canadian Syllabics, Ogham
                {0x16A0,0x16EA},{0x16F1,0x16F8},{0x1700,0x1711},  // Runic, Tagalog
                {0x1780,0x17B3},{0x17D7,0x17D7},{0x17DC,0x17DC},  // Khmer
                {0x1820,0x1878},{0x1880,0x18A8},                  // Mongolian
                {0x1D00,0x1DBF},                                  // Phonetic extensions
                {0x1E00,0x1F15},{0x1F18,0x1F1D},{0x1F20,0x1F45},{0x1F48,0x1F4D}, // Latin Ext Additional, Greek Ext
                {0x1F50,0x1F57},{0x1F59,0x1F59},{0x1F5B,0x1F5B},{0x1F5D,0x1F5D},{0x1F5F,0x1F7D},
                {0x1F80,0x1FB4},{0x1FB6,0x1FBC},{0x1FBE,0x1FBE},{0x1FC2,0x1FC4},{0x1FC6,0x1FCC},
                {0x1FD0,0x1FD3},{0x1FD6,0x1FDB},{0x1FE0,0x1FEC},{0x1FF2,0x1FF4},{0x1FF6,0x1FFC},
                {0x2071,0x2071},{0x207F,0x207F},{0x2090,0x209C},
                {0x2102,0x2102},{0x2107,0x2107},{0x210A,0x2113},{0x2115,0x2115},{0x2119,0x211D},
                {0x2124,0x2124},{0x2126,0x2126},{0x2128,0x2128},{0x212A,0x212D},{0x212F,0x2139},
                {0x213C,0x213F},{0x2145,0x2149},{0x214E,0x214E},{0x2183,0x2184},
                {0x2C00,0x2CE4},{0x2CEB,0x2CEE},{0x2CF2,0x2CF3},  // Glagolitic, Latin Ext-C, Coptic
                {0x2D00,0x2D25},{0x2D27,0x2D27},{0x2D2D,0x2D2D},{0x2D30,0x2D67},{0x2D6F,0x2D6F},
                {0x2D80,0x2DDE},
                {0x3005,0x3006},{0x3031,0x3035},{0x303B,0x303C},  // CJK iteration/ideographic marks
                {0x3041,0x3096},{0x309D,0x309F},                  // Hiragana
                {0x30A1,0x30FA},{0x30FC,0x30FF},                  // Katakana
                {0x3105,0x312F},{0x3131,0x318E},{0x31A0,0x31BF},  // Bopomofo, Hangul compat Jamo
                {0x31F0,0x31FF},
                {0x3400,0x4DBF},{0x4E00,0xA48C},                  // CJK Ext-A, Unified, Yi
                {0xA4D0,0xA4FD},{0xA500,0xA60C},{0xA610,0xA61F},{0xA62A,0xA62B},
                {0xA640,0xA66E},{0xA67F,0xA69D},{0xA6A0,0xA6E5},
                {0xA717,0xA71F},{0xA722,0xA788},{0xA78B,0xA7CA},{0xA7D0,0xA7D9},{0xA7F2,0xA801},
                {0xAC00,0xD7A3},{0xD7B0,0xD7C6},{0xD7CB,0xD7FB},  // Hangul syllables + Jamo ext
                {0xF900,0xFAD9},                                  // CJK compatibility ideographs
                {0xFB00,0xFB06},{0xFB13,0xFB17},{0xFB1D,0xFB1D},{0xFB1F,0xFB28},{0xFB2A,0xFB4F},
                {0xFB50,0xFBB1},{0xFBD3,0xFD3D},{0xFD50,0xFDFB},{0xFE70,0xFEFC}, // Arabic presentation forms
                {0xFF21,0xFF3A},{0xFF41,0xFF5A},{0xFF66,0xFFDC},  // Fullwidth / halfwidth letters
                {0x10000,0x100FA},{0x10280,0x1031F},{0x10330,0x10340},{0x10400,0x1049D}, // Linear B, Gothic, Deseret
                {0x1D400,0x1D7CB},                                // Mathematical alphanumerics (letters)
                {0x20000,0x2A6DF},{0x2A700,0x2EBE0},{0x2F800,0x2FA1D},{0x30000,0x3134A}, // CJK Ext-B..G
            };
            return in_ranges(kLetters, sizeof(kLetters) / sizeof(kLetters[0]), cp);
        }

        // Approximation of Unicode \p{N} (number: Nd + Nl + No).
        inline bool is_number(int cp) {
            if (cp >= '0' && cp <= '9') return true;
            if (cp < 0x80) return false;
            static const CpRange kNumbers[] = {
                {0x00B2,0x00B3},{0x00B9,0x00B9},{0x00BC,0x00BE},  // superscripts, vulgar fractions
                {0x0660,0x0669},{0x06F0,0x06F9},{0x07C0,0x07C9},  // Arabic-Indic, Extended, NKo
                {0x0966,0x096F},{0x09E6,0x09EF},{0x09F4,0x09F9},
                {0x0A66,0x0A6F},{0x0AE6,0x0AEF},{0x0B66,0x0B6F},{0x0B72,0x0B77},
                {0x0BE6,0x0BF2},{0x0C66,0x0C6F},{0x0C78,0x0C7E},{0x0CE6,0x0CEF},
                {0x0D58,0x0D5E},{0x0D66,0x0D78},{0x0DE6,0x0DEF},
                {0x0E50,0x0E59},{0x0ED0,0x0ED9},{0x0F20,0x0F33},  // Thai, Lao, Tibetan
                {0x1040,0x1049},{0x1090,0x1099},{0x1369,0x137C},{0x16EE,0x16F0},
                {0x17E0,0x17E9},{0x17F0,0x17F9},{0x1810,0x1819},
                {0x2070,0x2070},{0x2074,0x2079},{0x2080,0x2089},  // super/subscripts
                {0x2150,0x2182},{0x2185,0x2189},                  // number forms (roman numerals, fractions)
                {0x2460,0x249B},{0x24EA,0x24FF},{0x2776,0x2793},  // enclosed / dingbat numbers
                {0x2CFD,0x2CFD},
                {0x3007,0x3007},{0x3021,0x3029},{0x3038,0x303A},  // CJK numerals
                {0x3192,0x3195},{0x3220,0x3229},{0x3248,0x324F},{0x3251,0x325F},
                {0x3280,0x3289},{0x32B1,0x32BF},
                {0xA620,0xA629},{0xA6E6,0xA6EF},{0xA830,0xA835},{0xA8D0,0xA8D9},
                {0xFF10,0xFF19},                                  // fullwidth digits
                {0x1D7CE,0x1D7FF},                                // mathematical digits
            };
            return in_ranges(kNumbers, sizeof(kNumbers) / sizeof(kNumbers[0]), cp);
        }

        // Unicode \s (White_Space property), matching the regex class used by
        // the GPT-2 pre-tokenizer pattern.
        inline bool is_space(int cp) {
            if (cp == ' ' || (cp >= '\t' && cp <= '\r')) return true; // \t \n \v \f \r
            if (cp < 0x80) return false;
            return cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
                   (cp >= 0x2000 && cp <= 0x200A) ||
                   cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
                   cp == 0x205F || cp == 0x3000;
        }

        // Split `text` into chunks following the GPT-2 pattern's precedence
        // order: contractions, then " ?letters+", " ?digits+",
        // " ?punct-run+", then whitespace (with the "\s+(?!\S)" vs "\s+"
        // distinction: a run of whitespace not followed by a non-space
        // character keeps its last space attached to itself rather than to
        // the next word).
        inline std::vector<std::string> split(std::string const& text) {
            std::vector<std::string> chunks;
            size_t n = text.size();
            size_t i = 0;

            // Contractions checked literally (case-sensitive, as GPT-2 does)
            // at the current position before falling through to the
            // general-category rules.
            static const char* contractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

            while (i < n) {
                // 1) Literal contraction match.
                bool matched_contraction = false;
                for (const char* c : contractions) {
                    size_t clen = std::strlen(c);
                    if (i + clen <= n && text.compare(i, clen, c) == 0) {
                        chunks.push_back(text.substr(i, clen));
                        i += clen;
                        matched_contraction = true;
                        break;
                    }
                }
                if (matched_contraction) continue;

                int len = 0;
                int cp = decode_utf8(text, i, len);

                // 2) " ?\p{L}+" -- optional leading space, then a run of letters.
                if (cp == ' ' && i + len < n) {
                    int len2 = 0;
                    int cp2 = decode_utf8(text, i + len, len2);
                    if (is_letter(cp2)) {
                        size_t start = i;
                        size_t j = i + len;
                        while (j < n) {
                            int l = 0; int c = decode_utf8(text, j, l);
                            if (!is_letter(c)) break;
                            j += l;
                        }
                        chunks.push_back(text.substr(start, j - start));
                        i = j;
                        continue;
                    }
                    if (is_number(cp2)) {
                        size_t start = i;
                        size_t j = i + len;
                        while (j < n) {
                            int l = 0; int c = decode_utf8(text, j, l);
                            if (!is_number(c)) break;
                            j += l;
                        }
                        chunks.push_back(text.substr(start, j - start));
                        i = j;
                        continue;
                    }
                    if (!is_space(cp2)) {
                        // " ?[^\s\p{L}\p{N}]+" -- leading space + punctuation run.
                        size_t start = i;
                        size_t j = i + len;
                        while (j < n) {
                            int l = 0; int c = decode_utf8(text, j, l);
                            if (is_space(c) || is_letter(c) || is_number(c)) break;
                            j += l;
                        }
                        chunks.push_back(text.substr(start, j - start));
                        i = j;
                        continue;
                    }
                }

                if (is_letter(cp)) {
                    size_t start = i;
                    size_t j = i + len;
                    while (j < n) {
                        int l = 0; int c = decode_utf8(text, j, l);
                        if (!is_letter(c)) break;
                        j += l;
                    }
                    chunks.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }

                if (is_number(cp)) {
                    size_t start = i;
                    size_t j = i + len;
                    while (j < n) {
                        int l = 0; int c = decode_utf8(text, j, l);
                        if (!is_number(c)) break;
                        j += l;
                    }
                    chunks.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }

                if (!is_space(cp)) {
                    // "[^\s\p{L}\p{N}]+" -- punctuation/symbol run with no
                    // leading space (letter/number cases above have already
                    // been ruled out by this point, so `cp` here is
                    // necessarily punctuation).
                    size_t start = i;
                    size_t j = i + len;
                    while (j < n) {
                        int l = 0; int c = decode_utf8(text, j, l);
                        if (is_space(c) || is_letter(c) || is_number(c)) break;
                        j += l;
                    }
                    chunks.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }

                {
                    // is_space(cp) at this point (only remaining case).
                    // "\s+(?!\S)" vs "\s+": consume a whitespace run, but if
                    // it's followed by more (non-space) text, leave the last
                    // space behind to start the next chunk -- this is what
                    // makes " ?\p{L}+" able to pick up a leading space on
                    // the following word instead of it being swallowed here.
                    size_t start = i;
                    size_t j = i + len;
                    while (j < n) {
                        int l = 0; int c = decode_utf8(text, j, l);
                        if (!is_space(c)) break;
                        j += l;
                    }
                    bool followed_by_more = (j < n);
                    if (followed_by_more) {
                        // "\s+(?!\S)" only matches a whitespace run that is
                        // NOT followed by non-space text; since this run IS
                        // followed by more text, the regex instead falls
                        // through to plain "\s+" matching everything except
                        // its last codepoint -- that last codepoint is left
                        // behind to become a leading " ?" for the next
                        // chunk (e.g. " ?\p{L}+"). Find where the last
                        // codepoint of [start, j) begins.
                        size_t last_cp_start = start;
                        size_t k = start;
                        while (k < j) {
                            last_cp_start = k;
                            int l = 0; decode_utf8(text, k, l);
                            k += l;
                        }
                        if (last_cp_start > start) {
                            // Run has more than one codepoint: emit
                            // everything before the last one as its own
                            // chunk, then continue the loop with `i` at the
                            // last codepoint (single-space run, always
                            // makes progress next iteration).
                            chunks.push_back(text.substr(start, last_cp_start - start));
                            i = last_cp_start;
                        } else {
                            // Run is exactly one codepoint wide and is
                            // followed by more text: nothing to peel off
                            // ahead of it, so it stands alone as its own
                            // single-codepoint chunk (this is what lets the
                            // NEXT chunk's " ?\p{L}+"/" ?\p{N}+"/" ?punct+"
                            // rule attach it as a leading space instead).
                            chunks.push_back(text.substr(start, len));
                            i = start + len;
                        }
                        continue;
                    } else {
                        chunks.push_back(text.substr(start, j - start));
                        i = j;
                        continue;
                    }
                }

                // Defensive fallback: every codepoint is classified as
                // either space or non-space above and both branches always
                // advance `i` and `continue`, so this is unreachable in
                // practice. Kept so that any future edit which narrows one
                // of those branches fails safe -- emitting the codepoint on
                // its own -- rather than looping forever or dropping input.
                chunks.push_back(text.substr(i, len));
                i += len;
            }

            return chunks;
        }

    } // namespace gpt2_pretokenize

    class Tokenizer {
    public:
        // ---- Special tokens ---------------------------------------------------
        int pad_id{ 0 };
        int bos_id{ 1 };
        int eos_id{ 2 };
        int unk_id{ 3 };

        // ---- Vocabulary -------------------------------------------------------
        std::vector<std::string>              vocab;           // id → string
        std::unordered_map<std::string, int>  vocab_index;     // string → id
        std::vector<std::pair<int, int>>       merges;          // BPE merge rules (ordered)
        std::unordered_map<std::pair<int, int>, int, PairHash> merge_map;  // pair -> merged token id
        // pair -> priority rank (lower = higher priority). O(1) lookup during
        // apply_merges() instead of an O(n) linear scan through `merges` --
        // with real vocabularies (50k-100k+ merges) the linear scan made
        // encoding a single short prompt take tens of seconds to minutes.
        std::unordered_map<std::pair<int, int>, int, PairHash> merge_rank;

        // Whether this vocabulary uses GPT-2/tiktoken-style byte escaping.
        // Auto-detected in load() / init_byte_vocab().
        bool uses_byte_escaping{ false };

        // Whether encode() should run the GPT-2 regex pre-tokenizer before
        // BPE. Tied to the same signal as byte escaping: every vocabulary
        // that uses the byte<->unicode escape table (GPT-2, GPT-NeoX,
        // Granite, most tiktoken-derived vocabs) also uses this splitting
        // scheme, since both come from the same reference implementation.
        // Plain byte vocabs and the char-level test tokenizer (which have no
        // notion of "words") leave this off and BPE the whole input as one
        // chunk, same as before.
        bool uses_gpt2_pretokenizer{ false };

        void detect_byte_escaping() {
            bool has_literal_space = vocab_index.count(" ") > 0;
            std::string escaped_space = bpe_byte_escape::byte_to_escaped_utf8(' ');
            bool has_escaped_space = vocab_index.count(escaped_space) > 0;
            uses_byte_escaping = has_escaped_space && !has_literal_space;
            uses_gpt2_pretokenizer = uses_byte_escaping;
        }

        // ---- Init from scratch (for testing / bootstrap) ---------------------
        void init_byte_vocab() {
            // First 256 tokens = raw bytes, then special tokens
            vocab.clear();
            vocab_index.clear();

            auto add = [&](std::string s) {
                vocab_index[s] = static_cast<int>(vocab.size());
                vocab.push_back(s);
                };
            add("<pad>");
            add("<bos>");
            add("<eos>");
            add("<unk>");

            for (int i = 0; i < 256; ++i) {
                std::string s = std::string(1, (char)i);
                add(s);
            }

            detect_byte_escaping();  // will be false for this plain-byte vocab
        }

        int vocab_size() const { return static_cast<int>(vocab.size()); }

        // ---- Encode -----------------------------------------------------------
        std::vector<int> encode(std::string const& text, bool add_bos = false) const {
            std::vector<int> ids;
            if (add_bos) ids.push_back(bos_id);

            // GPT-2 regex pre-tokenization first (if this vocab uses it):
            // split the input into chunks (contractions / word runs / digit
            // runs / punctuation runs / whitespace runs) and BPE each chunk
            // independently -- merges never cross a chunk boundary. This
            // matches the reference tokenizer's behavior at word/punctuation
            // boundaries; without it, BPE runs over the raw input as one
            // continuous stream and can merge across boundaries the
            // reference implementation never would (e.g. across a
            // punctuation/whitespace/word transition), producing a
            // different-but-plausible-looking token sequence.
            //
            // When this vocab doesn't use the scheme (plain byte vocabs,
            // char-level test tokenizers), fall back to the previous
            // behavior of treating the whole input as a single chunk.
            std::vector<std::string> chunks = uses_gpt2_pretokenizer
                ? gpt2_pretokenize::split(text)
                : std::vector<std::string>{ text };

            for (std::string const& chunk : chunks) {
                // Byte-level tokenization within this chunk.
                //
                // If this vocabulary uses GPT-2/tiktoken-style byte escaping
                // (detected in load()), each raw input byte must be converted
                // to its escaped Unicode form BEFORE lookup -- e.g. a literal
                // space (0x20) becomes 'Ġ' (U+0120) because that is how the
                // vocabulary actually stores it. Without this, every
                // space/newline/control byte in real text falls through to
                // <unk>, and essentially all word-boundary BPE merges never
                // fire.
                std::vector<int> tokens;
                tokens.reserve(chunk.size());
                for (unsigned char c : chunk) {
                    std::string s = uses_byte_escaping
                        ? bpe_byte_escape::byte_to_escaped_utf8(c)
                        : std::string(1, (char)c);
                    auto it = vocab_index.find(s);
                    tokens.push_back(it != vocab_index.end() ? it->second : unk_id);
                }

                // Apply BPE merges within this chunk only.
                tokens = apply_merges(tokens);

                ids.insert(ids.end(), tokens.begin(), tokens.end());
            }

            return ids;
        }

        // ---- Decode -----------------------------------------------------------
        std::string decode(std::vector<int> const& ids, bool skip_special = true) const {
            std::string escaped_result;
            for (int id : ids) {
                if (id < 0 || id >= (int)vocab.size()) continue;
                if (skip_special && (id == pad_id || id == bos_id || id == eos_id)) continue;
                escaped_result += vocab[id];
            }

            if (!uses_byte_escaping) return escaped_result;

            // Reverse the byte escaping: walk the UTF-8 string, decode each
            // codepoint, map it back through unicode_to_byte_table(), and
            // emit the original raw byte. Without this, output would show
            // literal 'Ġ'/'Ċ' etc. instead of spaces/newlines.
            return unescape_bytes(escaped_result);
        }

        std::string decode_token(int id) const {
            if (id < 0 || id >= (int)vocab.size()) return "<unk>";
            if (!uses_byte_escaping) return vocab[id];
            return unescape_bytes(vocab[id]);
        }

    private:
        // Decode a UTF-8-encoded, byte-escaped string back to raw bytes by
        // mapping each Unicode codepoint through unicode_to_byte_table().
        // Codepoints not present in the reverse table are passed through as
        // their raw UTF-8 encoding, to fail safe rather than silently drop data.
        static std::string unescape_bytes(std::string const& s) {
            std::string out;
            out.reserve(s.size());
            auto const& rev = bpe_byte_escape::unicode_to_byte_table();

            size_t i = 0;
            while (i < s.size()) {
                unsigned char c0 = (unsigned char)s[i];
                int cp; int len;
                if (c0 < 0x80) { cp = c0; len = 1; }
                else if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size())
                {
                    cp = ((c0 & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F); len = 2;
                }
                else if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size())
                {
                    cp = ((c0 & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6)
                        | ((unsigned char)s[i + 2] & 0x3F); len = 3;
                }
                else { cp = c0; len = 1; }  // invalid/unexpected lead byte, pass through

                auto it = rev.find(cp);
                if (it != rev.end()) {
                    out += (char)(unsigned char)it->second;
                }
                else {
                    out.append(s, i, len);
                }
                i += len;
            }
            return out;
        }

    public:
        // ---- File I/O --------------------------------------------------------
        void save(std::string const& path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Tokenizer: cannot write '" + path + "'");

            uint32_t vs = vocab_size(), nm = static_cast<uint32_t>(merges.size()), pad = 0;
            f.write(reinterpret_cast<char const*>(&NFT_MAGIC), 4);
            f.write(reinterpret_cast<char const*>(&NFT_VERSION), 4);
            f.write(reinterpret_cast<char const*>(&vs), 4);
            f.write(reinterpret_cast<char const*>(&pad), 4);

            for (auto const& s : vocab) {
                uint16_t l = static_cast<uint16_t>(s.size());
                f.write(reinterpret_cast<char const*>(&l), 2);
                f.write(s.data(), l);
            }

            f.write(reinterpret_cast<char const*>(&nm), 4);
            for (auto const& m : merges) {
                uint32_t a = m.first, b = m.second;
                f.write(reinterpret_cast<char const*>(&a), 4);
                f.write(reinterpret_cast<char const*>(&b), 4);
            }
        }

        void load(std::string const& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Tokenizer: cannot open '" + path + "'");

            uint32_t magic, version, vs, pad;
            f.read(reinterpret_cast<char*>(&magic), 4);
            f.read(reinterpret_cast<char*>(&version), 4);
            f.read(reinterpret_cast<char*>(&vs), 4);
            f.read(reinterpret_cast<char*>(&pad), 4);

            if (magic != NFT_MAGIC)
                throw std::runtime_error("Tokenizer: bad magic");

            vocab.resize(vs);
            vocab_index.clear();
            for (uint32_t i = 0; i < vs; ++i) {
                uint16_t l; f.read(reinterpret_cast<char*>(&l), 2);
                vocab[i].resize(l);
                f.read(vocab[i].data(), l);
                vocab_index[vocab[i]] = i;
            }

            uint32_t nm; f.read(reinterpret_cast<char*>(&nm), 4);
            merges.resize(nm);
            merge_map.clear();
            merge_rank.clear();
            merge_rank.reserve(nm);
            for (uint32_t i = 0; i < nm; ++i) {
                uint32_t a, b;
                f.read(reinterpret_cast<char*>(&a), 4);
                f.read(reinterpret_cast<char*>(&b), 4);
                merges[i] = { (int)a, (int)b };
                merge_rank[{(int)a, (int)b}] = (int)i;
                // merged token id: find vocab entry for vocab[a]+vocab[b]
                std::string merged = vocab[a] + vocab[b];
                auto it = vocab_index.find(merged);
                if (it != vocab_index.end())
                    merge_map[{(int)a, (int)b}] = it->second;
            }

            detect_byte_escaping();
        }

        // ---- Build a minimal test tokenizer (character-level fallback) --------
        static Tokenizer make_char_level(std::string const& chars) {
            Tokenizer t;
            t.vocab.clear();
            t.vocab_index.clear();
            auto add = [&](std::string s) {
                t.vocab_index[s] = (int)t.vocab.size();
                t.vocab.push_back(s);
                };
            add("<pad>"); add("<bos>"); add("<eos>"); add("<unk>");
            for (char c : chars) {
                std::string s(1, c);
                if (t.vocab_index.find(s) == t.vocab_index.end()) add(s);
            }
            t.pad_id = 0; t.bos_id = 1; t.eos_id = 2; t.unk_id = 3;
            t.detect_byte_escaping();
            return t;
        }

    private:
        std::vector<int> apply_merges(std::vector<int> tokens) const {
            if (merges.empty()) return tokens;

            bool changed = true;
            while (changed && tokens.size() > 1) {
                changed = false;
                // Find the highest-priority (lowest rank) applicable merge.
                // O(1) rank lookup via merge_rank -- see the field comment
                // above for why this matters at real vocabulary scale.
                int best_pos = -1;
                int best_rank = (int)merges.size();
                int best_id = -1;

                for (int i = 0; i + 1 < (int)tokens.size(); ++i) {
                    auto rit = merge_rank.find({ tokens[i], tokens[i + 1] });
                    if (rit == merge_rank.end()) continue;
                    int rank = rit->second;
                    if (rank < best_rank) {
                        auto mit = merge_map.find({ tokens[i], tokens[i + 1] });
                        if (mit == merge_map.end()) continue;  // rank known but no target id
                        best_rank = rank;
                        best_pos = i;
                        best_id = mit->second;
                    }
                }

                if (best_pos >= 0) {
                    tokens[best_pos] = best_id;
                    tokens.erase(tokens.begin() + best_pos + 1);
                    changed = true;
                }
            }
            return tokens;
        }
    };

} // namespace vlrf