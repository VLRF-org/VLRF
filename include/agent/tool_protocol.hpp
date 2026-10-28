// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Agent Tool-Call Protocol  (tool_protocol.hpp)
//
// VLRF/vlrf_cli have no built-in notion of "tools" — the model just
// emits text. This header defines the small JSON convention the agent loop
// looks for in that text, and the parser that extracts it.
//
// Convention: the model is prompted (see agent_loop.cpp build_system_prompt)
// to emit a single JSON object on its own line, fenced or unfenced, of the
// shape:
//
//   {"tool": "exec", "args": {"argv": ["ls", "-la"]}}
//   {"tool": "write_file", "args": {"path": "main.cpp", "content": "..."}}
//   {"tool": "read_file", "args": {"path": "main.cpp"}}
//   {"tool": "finish", "args": {"summary": "Implemented the feature."}}
//
// The parser is deliberately tolerant of ```json ... ``` fences (models
// habitually add them) but otherwise strict: malformed or unrecognized
// tool calls are surfaced as a ToolCall with kind=Unknown rather than
// thrown, so the agent loop can report the parse error back to the model
// as the next turn's input instead of crashing the session.
// =============================================================================
#include <string>
#include <vector>
#include <optional>

namespace vlrf {
namespace agent {

enum class ToolKind {
    Exec,
    WriteFile,
    ReadFile,
    Finish,
    Unknown,      // recognized JSON, unrecognized/missing "tool" field
    NoToolCall,   // no JSON tool-call block found in the text at all
    ParseError,   // looked like a tool-call block but JSON was malformed
};

struct ToolCall {
    ToolKind kind = ToolKind::NoToolCall;

    // Exec
    std::vector<std::string> argv;

    // WriteFile / ReadFile
    std::string path;
    std::string content;      // WriteFile only

    // Finish
    std::string summary;

    // Diagnostics
    std::string raw_json;     // the exact JSON block that was parsed
    std::string error;        // set when kind == ParseError or Unknown
};

// Scans `model_output` for the first JSON tool-call object (optionally
// inside a ```json fence) and parses it. If no candidate JSON object is
// found at all, returns a ToolCall with kind == NoToolCall — this is the
// normal case for a turn where the model just produced prose/code without
// requesting a tool.
ToolCall parse_tool_call(std::string const& model_output);

// Serializes a tool result back into the small text block appended to the
// running transcript before the next generate() call. Kept as plain
// labeled text (not JSON) on the way *back* to the model, since it's much
// less likely to be misparsed as a nested tool call than round-tripping
// JSON-in-JSON.
std::string format_tool_result_for_context(ToolKind kind,
                                            bool success,
                                            std::string const& output,
                                            std::string const& error);

} // namespace agent
} // namespace vlrf
