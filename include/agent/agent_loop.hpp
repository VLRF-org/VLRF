// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Coding Agent Loop  (agent_loop.hpp)
//
// Wires together: a loaded VlrfModel, an ISandbox, and the tool-call
// protocol into a generate -> parse -> execute -> feed-back loop, driven
// by `vlrf_cli agent`. This is orchestration only — it does not know about
// HyperV or Podman specifically, only the ISandbox interface, so backend
// selection happens once (see sandbox_factory.hpp) before the loop starts.
// =============================================================================
#include "sandbox.hpp"
#include "agent_config.hpp"
#include "tool_protocol.hpp"
#include "vlrf.h"
#include <string>
#include <vector>
#include <functional>

namespace vlrf {
namespace agent {

// Called once per turn with a human-readable status line, e.g.
// "[turn 3/25] exec: cargo build" — wired to std::cout in vlrf_cli, but
// kept as a callback so tests can capture it instead.
using AgentTraceFn = std::function<void(std::string const& line)>;

struct AgentTurnLog {
    int turn_index = 0;
    std::string model_output;
    ToolCall tool_call;
    ExecResult exec_result;   // only meaningful if tool_call.kind == Exec
    bool session_ended = false;
};

struct AgentSessionResult {
    bool completed = false;        // true if the model emitted "finish"
    bool hit_turn_limit = false;
    std::string final_summary;     // from the "finish" tool call, if any
    std::vector<AgentTurnLog> turns;
};

// Runs the full agent session for one task description against an
// already-loaded model handle and an already-created sandbox. Ownership
// of both `model` and `sandbox` stays with the caller (vlrf_cli frees the
// model, and the sandbox's destroy() should be called by the caller after
// this returns so partial state is inspectable on failure).
AgentSessionResult run_agent_session(VlrfModel model,
                                      ISandbox& sandbox,
                                      AgentConfig const& cfg,
                                      std::string const& task_description,
                                      AgentTraceFn trace = nullptr);

// Builds the system/instruction preamble that teaches the model the
// tool-call JSON convention. Exposed separately so tests and `vlrf_cli
// agent --dry-run` can inspect exactly what gets prepended to the prompt.
std::string build_system_prompt(AgentConfig const& cfg);

} // namespace agent
} // namespace vlrf
