// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Agent Tool-Call Protocol implementation  (tool_protocol.cpp)
// See include/agent/tool_protocol.hpp for the protocol description.
// =============================================================================
#include "tool_protocol.hpp"
#include "tiny_json.hpp"
#include <sstream>

namespace vlrf {
namespace agent {

using json::Value;
using json::Type;

ToolCall parse_tool_call(std::string const& model_output) {
    ToolCall call;

    Value root;
    try {
        root = json::find_and_parse_first_object(model_output);
    } catch (std::exception const& e) {
        call.kind = ToolKind::ParseError;
        call.error = e.what();
        return call;
    }

    if (root.is_null()) {
        call.kind = ToolKind::NoToolCall;
        return call;
    }

    // Re-serialize isn't available in tiny_json (no writer), so keep the
    // raw text unavailable here; callers that want the exact substring
    // can re-run find_and_parse on their own copy. We do want *some*
    // diagnostic string though, so record the tool name we found instead.
    std::string tool_name = root.get("tool").as_string();
    Value const& args = root.get("args");

    if (tool_name.empty()) {
        call.kind = ToolKind::Unknown;
        call.error = "tool-call JSON object found but \"tool\" field is missing or not a string";
        return call;
    }

    if (tool_name == "exec") {
        Value const& argv_val = args.get("argv");
        if (!argv_val.is_array() || argv_val.arr->empty()) {
            call.kind = ToolKind::ParseError;
            call.error = "\"exec\" requires args.argv to be a non-empty array of strings";
            return call;
        }
        for (auto const& elem : *argv_val.arr) {
            if (!elem.is_string()) {
                call.kind = ToolKind::ParseError;
                call.error = "\"exec\" args.argv must contain only strings";
                return call;
            }
            call.argv.push_back(elem.as_string());
        }
        call.kind = ToolKind::Exec;
        return call;
    }

    if (tool_name == "write_file") {
        Value const& path_val = args.get("path");
        Value const& content_val = args.get("content");
        if (!path_val.is_string()) {
            call.kind = ToolKind::ParseError;
            call.error = "\"write_file\" requires args.path to be a string";
            return call;
        }
        call.path = path_val.as_string();
        call.content = content_val.as_string(""); // empty content is valid (create empty file)
        call.kind = ToolKind::WriteFile;
        return call;
    }

    if (tool_name == "read_file") {
        Value const& path_val = args.get("path");
        if (!path_val.is_string()) {
            call.kind = ToolKind::ParseError;
            call.error = "\"read_file\" requires args.path to be a string";
            return call;
        }
        call.path = path_val.as_string();
        call.kind = ToolKind::ReadFile;
        return call;
    }

    if (tool_name == "finish") {
        call.summary = args.get("summary").as_string("(no summary provided)");
        call.kind = ToolKind::Finish;
        return call;
    }

    call.kind = ToolKind::Unknown;
    call.error = "unrecognized tool \"" + tool_name + "\" (expected exec|write_file|read_file|finish)";
    return call;
}

std::string format_tool_result_for_context(ToolKind kind,
                                            bool success,
                                            std::string const& output,
                                            std::string const& error) {
    std::ostringstream oss;
    oss << "[tool_result";
    switch (kind) {
        case ToolKind::Exec:      oss << " tool=exec"; break;
        case ToolKind::WriteFile: oss << " tool=write_file"; break;
        case ToolKind::ReadFile:  oss << " tool=read_file"; break;
        case ToolKind::Finish:    oss << " tool=finish"; break;
        case ToolKind::Unknown:   oss << " tool=unknown"; break;
        case ToolKind::ParseError: oss << " tool=parse_error"; break;
        case ToolKind::NoToolCall: oss << " tool=none"; break;
    }
    oss << " success=" << (success ? "true" : "false") << "]\n";
    if (!output.empty()) oss << output << "\n";
    if (!error.empty())  oss << "error: " << error << "\n";
    oss << "[/tool_result]";
    return oss.str();
}

} // namespace agent
} // namespace vlrf
