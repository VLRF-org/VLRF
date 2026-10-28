// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Tiny JSON implementation  (tiny_json.cpp)
// See include/agent/tiny_json.hpp for scope/limitations.
// =============================================================================
#include "tiny_json.hpp"
#include <cctype>
#include <sstream>

namespace vlrf {
namespace agent {
namespace json {

Value const& Value::get(std::string const& key) const {
    static const Value kNull{};
    if (type != Type::Object || !obj) return kNull;
    for (auto const& kv : *obj) {
        if (kv.first == key) return kv.second;
    }
    return kNull;
}

namespace {

class Parser {
public:
    explicit Parser(std::string const& s) : s_(s), i_(0), n_(s.size()) {}

    Value parse_document() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (i_ != n_) fail("trailing content after JSON value");
        return v;
    }

    Value parse_value() {
        skip_ws();
        if (i_ >= n_) fail("unexpected end of input");
        char c = s_[i_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
        return Value{};
    }

    size_t pos() const { return i_; }

private:
    std::string const& s_;
    size_t i_;
    size_t n_;

    [[noreturn]] void fail(std::string const& msg) {
        std::ostringstream oss;
        oss << "tiny_json parse error at offset " << i_ << ": " << msg;
        throw std::runtime_error(oss.str());
    }

    void skip_ws() {
        while (i_ < n_ && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }

    char peek() { return i_ < n_ ? s_[i_] : '\0'; }

    void expect(char c) {
        if (i_ >= n_ || s_[i_] != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }

    Value parse_object() {
        Value v;
        v.type = Type::Object;
        v.obj = std::make_shared<Object>();
        expect('{');
        skip_ws();
        if (peek() == '}') { ++i_; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected string key");
            std::string key = parse_raw_string();
            skip_ws();
            expect(':');
            Value val = parse_value();
            v.obj->emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == '}') { ++i_; break; }
            fail("expected ',' or '}' in object");
        }
        return v;
    }

    Value parse_array() {
        Value v;
        v.type = Type::Array;
        v.arr = std::make_shared<Array>();
        expect('[');
        skip_ws();
        if (peek() == ']') { ++i_; return v; }
        while (true) {
            Value elem = parse_value();
            v.arr->push_back(std::move(elem));
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == ']') { ++i_; break; }
            fail("expected ',' or ']' in array");
        }
        return v;
    }

    // Appends codepoint `cp` to `out` as UTF-8. Supports the full Unicode
    // range (1-4 byte sequences). Lone/invalid surrogates and out-of-range
    // values are replaced with U+FFFD so the output is always valid UTF-8.
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    // Reads exactly 4 hex digits at the cursor (after "\\u") and returns the value.
    uint32_t parse_hex4() {
        if (i_ + 4 > n_) fail("truncated \\u escape");
        uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
            char h = s_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
            else fail("invalid \\u escape digit");
        }
        return cp;
    }

    std::string parse_raw_string() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= n_) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                if (i_ >= n_) fail("unterminated escape");
                char e = s_[i_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // High surrogate: combine with a following \uDC00-\uDFFF
                            // low surrogate into one astral codepoint.
                            if (i_ + 1 < n_ && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                                size_t save = i_;
                                i_ += 2;
                                uint32_t lo = parse_hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    i_ = save; // not a pair; lone high -> U+FFFD below
                                }
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape character");
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    Value parse_string_value() {
        Value v;
        v.type = Type::String;
        v.str = parse_raw_string();
        return v;
    }

    Value parse_bool() {
        Value v;
        v.type = Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v.b = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; }
        else fail("invalid literal (expected true/false)");
        return v;
    }

    Value parse_null() {
        if (s_.compare(i_, 4, "null") != 0) fail("invalid literal (expected null)");
        i_ += 4;
        return Value{}; // default-constructed Value is Type::Null
    }

    Value parse_number() {
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < n_ && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        if (i_ < n_ && s_[i_] == '.') {
            ++i_;
            while (i_ < n_ && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ < n_ && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < n_ && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            while (i_ < n_ && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ == start) fail("invalid number");
        Value v;
        v.type = Type::Number;
        v.num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

} // namespace

Value parse(std::string const& text) {
    Parser p(text);
    return p.parse_document();
}

Value find_and_parse_first_object(std::string const& text) {
    // Strip the most common fence forms models emit around JSON so we
    // don't have to special-case them at every call site. This is a
    // plain substring scan, not a Markdown parser.
    size_t search_start = 0;
    while (true) {
        size_t brace = text.find('{', search_start);
        if (brace == std::string::npos) return Value{}; // Type::Null: no tool call present

        // Find the matching closing brace by tracking depth, respecting
        // string literals so a '{' or '}' inside a JSON string doesn't
        // throw off the count.
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        size_t end = std::string::npos;
        for (size_t k = brace; k < text.size(); ++k) {
            char c = text[k];
            if (in_string) {
                if (escaped) { escaped = false; }
                else if (c == '\\') { escaped = true; }
                else if (c == '"') { in_string = false; }
                continue;
            }
            if (c == '"') { in_string = true; continue; }
            if (c == '{') ++depth;
            else if (c == '}') {
                --depth;
                if (depth == 0) { end = k; break; }
            }
        }

        if (end == std::string::npos) {
            // Unbalanced braces from this point on — nothing further to try.
            return Value{};
        }

        std::string candidate = text.substr(brace, end - brace + 1);
        try {
            Value v = parse(candidate);
            if (v.is_object()) return v;
            // Parsed fine but wasn't an object (shouldn't happen given we
            // anchored on '{'), fall through and keep scanning.
        } catch (std::exception const&) {
            // This '{...}' span wasn't valid JSON on its own (e.g. a code
            // sample containing a brace). Keep scanning after it rather
            // than treating the whole text as a parse error, since prose
            // legitimately contains braces (C++ code blocks, etc.).
        }
        search_start = brace + 1;
    }
}

} // namespace json
} // namespace agent
} // namespace vlrf
