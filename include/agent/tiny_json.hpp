// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Tiny JSON  (tiny_json.hpp)
//
// A small, dependency-free JSON reader used only by the agent module
// (agent_config.cpp, tool_protocol.cpp). NOT a general-purpose JSON
// library: no writer beyond simple string escaping, no comments/trailing
// commas, no big-number edge cases. \uXXXX escapes (including surrogate pairs) -> UTF-8.
// This is deliberate — the schemas it parses (agent.json, tool-call
// blocks) are small and fixed, and pulling in a full third-party JSON
// dependency for that isn't worth the build-system churn.
//
// Do not reach for this in new code outside include/agent/ or src/agent/;
// if VLRF grows a general JSON need elsewhere, vendor a real library
// instead of extending this one.
// =============================================================================
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <variant>
#include <cstdint>

namespace vlrf {
namespace agent {
namespace json {

struct Value;
using Array  = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>; // preserves insertion order

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value {
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::shared_ptr<Array> arr;
    std::shared_ptr<Object> obj;

    bool is_null()   const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array()  const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }

    // Object lookup; returns a Null Value (not a throw) if the object
    // doesn't have the key or this Value isn't an object, so callers can
    // chain .get("a").get("b") without null-checking every hop.
    Value const& get(std::string const& key) const;

    std::string as_string(std::string const& fallback = "") const {
        return type == Type::String ? str : fallback;
    }
    double as_number(double fallback = 0.0) const {
        return type == Type::Number ? num : fallback;
    }
    bool as_bool(bool fallback = false) const {
        return type == Type::Bool ? b : fallback;
    }
};

// Parses a full JSON document from `text`. Throws std::runtime_error with
// a position-annotated message on malformed input.
Value parse(std::string const& text);

// Finds and parses the first top-level JSON object in `text`, tolerating
// a surrounding ```json ... ``` or ``` ... ``` fence and leading/trailing
// prose. Returns Type::Null (not a throw) if no `{ ... }` object could be
// located at all — that's the normal "model didn't call a tool" case.
// Throws std::runtime_error only if a `{...}` span was found but failed
// to parse as valid JSON (this is surfaced by tool_protocol.cpp as
// ToolKind::ParseError instead of propagating the exception).
Value find_and_parse_first_object(std::string const& text);

} // namespace json
} // namespace agent
} // namespace vlrf
