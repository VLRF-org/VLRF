// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Sandbox Factory  (sandbox_factory.hpp)
//
// Single point that turns an AgentConfig into a concrete ISandbox. Keeps
// the #ifdef VLRF_AGENT_USE_HYPERV / VLRF_AGENT_USE_PODMAN compile-guard logic
// in one .cpp instead of scattered across the agent loop.
// =============================================================================
#include "sandbox.hpp"
#include "agent_config.hpp"
#include <memory>

namespace vlrf {
namespace agent {

// Resolves cfg.sandbox_backend (including Auto) and constructs the
// matching sandbox implementation. Returns nullptr if the resolved
// backend was not compiled into this binary (check vlrf_agent_capabilities()
// beforehand to give the user a clear error instead of a null deref).
std::unique_ptr<ISandbox> make_sandbox(AgentConfig const& cfg);

// Human-readable summary of which sandbox backends this binary was built
// with, e.g. "podman: yes, hyperv: no (built on non-Windows)". Mirrors the
// style of vlrf_backend_capabilities() in the core API.
const char* agent_sandbox_capabilities();

} // namespace agent
} // namespace vlrf
