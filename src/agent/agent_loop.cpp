// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Coding Agent Loop implementation  (agent_loop.cpp)
// =============================================================================
#include "agent_loop.hpp"
#include <sstream>
#include <vector>
#include <cstring>

namespace vlrf {
namespace agent {

std::string build_system_prompt(AgentConfig const& cfg) {
    std::ostringstream oss;
    oss <<
        "You are a coding agent running inside VLRF's sandboxed agent loop.\n"
        "You have access to exactly these tools. To use one, output a single JSON\n"
        "object on its own (optionally inside a ```json fence) and nothing else\n"
        "in that turn:\n"
        "\n"
        "  {\"tool\": \"exec\", \"args\": {\"argv\": [\"<program>\", \"<arg1>\", ...]}}\n"
        "    Runs a command in the sandbox, cwd = /workspace. argv is a literal\n"
        "    argument list, not a shell string; for shell features use\n"
        "    {\"argv\": [\"/bin/sh\", \"-c\", \"your command\"]} explicitly.\n"
        "\n"
        "  {\"tool\": \"write_file\", \"args\": {\"path\": \"relative/path\", \"content\": \"...\"}}\n"
        "  {\"tool\": \"read_file\", \"args\": {\"path\": \"relative/path\"}}\n"
        "    Paths are relative to the workspace root; you cannot access anything\n"
        "    outside it.\n"
        "\n"
        "  {\"tool\": \"finish\", \"args\": {\"summary\": \"what you did\"}}\n"
        "    Ends the session. Call this once the task is complete.\n"
        "\n"
        "Sandbox backend: " << to_string(cfg.sandbox_backend) << "\n"
        "Command timeout: " << cfg.limits.timeout_sec << "s. "
        "Network access: " << (cfg.limits.network_enabled ? "enabled" : "disabled") << ".\n"
        "You have at most " << cfg.max_agent_turns << " turns total. Work efficiently.\n";
    return oss.str();
}

namespace {

// Bridges VlrfTokenCallback (C ABI, one token at a time) into a std::string
// accumulator so the agent loop can work with whole-turn text, matching
// how tool-call parsing needs to see the complete JSON object at once
// rather than token-by-token.
struct GenerationCapture {
    std::string text;
};

int capture_callback(void* user_data, int32_t /*token_id*/, const char* token_str) {
    auto* capture = static_cast<GenerationCapture*>(user_data);
    if (token_str) capture->text += token_str;
    return 1; // continue generation
}

} // namespace

AgentSessionResult run_agent_session(VlrfModel model,
                                      ISandbox& sandbox,
                                      AgentConfig const& cfg,
                                      std::string const& task_description,
                                      AgentTraceFn trace) {
    AgentSessionResult session;

    auto log_trace = [&](std::string const& line) {
        if (trace) trace(line);
    };

    std::string transcript = build_system_prompt(cfg) + "\nTask: " + task_description + "\n";

    bool stopped_due_to_error = false;

    for (int turn = 0; turn < cfg.max_agent_turns; ++turn) {
        AgentTurnLog turn_log;
        turn_log.turn_index = turn;

        // --- Generate ---
        // Buffer sizing: vlrf_generate_text requires a caller-allocated
        // output buffer (see vlrf.h); 64KiB comfortably covers a
        // single agent turn's worth of reasoning + one tool call. If a
        // model runs right up against this, that's a sign max_tokens
        // should be tightened in AgentConfig, not a silent truncation
        // hazard, so we treat VLRF_ERR_* here as a hard stop for the turn.
        constexpr int32_t kOutBufSize = 65536;
        std::vector<char> out_buf(kOutBufSize, '\0');
        GenerationCapture capture;

        VlrfStatus status;
        try {
            status = vlrf_generate_text(model,
                                       transcript.c_str(),
                                       out_buf.data(),
                                       kOutBufSize,
                                       /*max_tokens=*/0, // model default
                                       capture_callback,
                                       &capture);
        } catch (std::exception const& e) {
            // The core KV-cache/context-window path currently throws a
            // raw C++ exception on overflow instead of returning
            // VLRF_ERR_CTX_FULL through the C ABI (see include/model/
            // kv_cache.hpp KVCache::append) -- guard against that here
            // so a long-running agent session degrades to a clean stop
            // instead of an uncaught-exception crash. This growing
            // transcript is exactly the workload that hits context
            // limits in practice, so the agent loop is the right place
            // to defend against it even though the root cause is
            // upstream of this module.
            log_trace("[turn " + std::to_string(turn) + "] generation threw: " + std::string(e.what()) +
                      " (likely context window exhausted by the growing transcript -- "
                      "consider a smaller max_agent_turns, a larger model context, or "
                      "summarizing/trimming older turns before this point)");
            turn_log.model_output = "";
            session.turns.push_back(turn_log);
            stopped_due_to_error = true;
            break;
        }
        if (status != VLRF_OK) {
            log_trace("[turn " + std::to_string(turn) + "] generation failed: " +
                      std::string(vlrf_status_str(status)));
            turn_log.model_output = ""; // generation never produced text this turn
            session.turns.push_back(turn_log);
            stopped_due_to_error = true;
            break;
        }

        // Prefer the streamed capture (token-accurate) but fall back to
        // the buffer vlrf_generate_text itself filled if no callback text
        // was captured for some reason (defensive; both should agree).
        std::string model_output = !capture.text.empty() ? capture.text : std::string(out_buf.data());
        turn_log.model_output = model_output;
        transcript += model_output + "\n";

        // --- Parse ---
        ToolCall call = parse_tool_call(model_output);
        turn_log.tool_call = call;

        if (call.kind == ToolKind::NoToolCall) {
            log_trace("[turn " + std::to_string(turn) + "] model produced text with no tool call; "
                      "prompting it to continue or finish");
            transcript += "[system] No tool call detected. Call a tool (exec/write_file/"
                          "read_file) or finish.\n";
            session.turns.push_back(turn_log);
            continue;
        }

        if (call.kind == ToolKind::ParseError || call.kind == ToolKind::Unknown) {
            log_trace("[turn " + std::to_string(turn) + "] tool call parse error: " + call.error);
            transcript += format_tool_result_for_context(call.kind, false, "", call.error) + "\n";
            session.turns.push_back(turn_log);
            continue;
        }

        if (call.kind == ToolKind::Finish) {
            log_trace("[turn " + std::to_string(turn) + "] finish: " + call.summary);
            turn_log.session_ended = true;
            session.completed = true;
            session.final_summary = call.summary;
            session.turns.push_back(turn_log);
            break;
        }

        // --- Execute ---
        if (call.kind == ToolKind::Exec) {
            log_trace("[turn " + std::to_string(turn) + "] exec: " +
                      [&] { std::string j; for (auto& a : call.argv) j += a + " "; return j; }());
            ExecResult result = sandbox.exec(call.argv);
            turn_log.exec_result = result;

            bool success = result.launched && !result.timed_out && result.exit_code == 0;
            std::string output = result.truncated_for_model(cfg.max_tool_output_bytes);
            std::string err = result.timed_out ? "command timed out"
                             : !result.launched ? sandbox.last_error()
                             : "";
            transcript += format_tool_result_for_context(ToolKind::Exec, success, output, err) + "\n";
        } else if (call.kind == ToolKind::WriteFile) {
            bool ok = sandbox.write_file(call.path, call.content);
            log_trace("[turn " + std::to_string(turn) + "] write_file " + call.path +
                      (ok ? " OK" : " FAILED"));
            transcript += format_tool_result_for_context(
                ToolKind::WriteFile, ok, ok ? ("wrote " + std::to_string(call.content.size()) + " bytes") : "",
                ok ? "" : sandbox.last_error()) + "\n";
        } else if (call.kind == ToolKind::ReadFile) {
            std::string content;
            bool ok = sandbox.read_file(call.path, content);
            log_trace("[turn " + std::to_string(turn) + "] read_file " + call.path +
                      (ok ? " OK" : " FAILED"));
            std::string truncated = content.size() > cfg.max_tool_output_bytes
                ? content.substr(0, cfg.max_tool_output_bytes) + "\n...[truncated]"
                : content;
            transcript += format_tool_result_for_context(
                ToolKind::ReadFile, ok, ok ? truncated : "", ok ? "" : sandbox.last_error()) + "\n";
        }

        session.turns.push_back(turn_log);
    }

    if (!session.completed && !stopped_due_to_error) {
        session.hit_turn_limit = true;
        log_trace("agent session hit max_agent_turns (" + std::to_string(cfg.max_agent_turns) +
                  ") without calling finish");
    }

    return session;
}

} // namespace agent
} // namespace vlrf
