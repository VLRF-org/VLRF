// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Agent Configuration  (agent_config.hpp)
//
// Loads agent.json (or a user-specified path) and resolves the sandbox
// backend choice. This is the single place that maps
// `"sandbox_backend": "auto"` to a concrete backend based on host OS, so
// vlrf_cli, the agent loop, and the sandbox factory all agree.
// =============================================================================
#include "sandbox.hpp"
#include <string>
#include <stdexcept>

namespace vlrf {
namespace agent {

enum class SandboxBackendKind {
    Auto,     // resolve by OS: HyperV on Windows, Podman on Linux
    HyperV,
    Podman,
    None,     // agent loop disabled / dry-run parsing only, no exec
};

struct AgentConfig {
    SandboxBackendKind sandbox_backend = SandboxBackendKind::Auto;

    // Hyper-V specific
    std::string hyperv_base_vhd;       // path to the base utility-VM VHDX
    int64_t     hyperv_vm_memory_mb = 2048;
    int         hyperv_vm_cpu_count = 2;
    // Guest logon used for exec(). Required: a full Windows VM guest has
    // no HcsCreateProcess-compatible init process (that API path only
    // works against container/minimal-VM guests, confirmed by hand
    // against a real Windows 11 host — HcsCreateProcess returns
    // ERROR_NOT_SUPPORTED (0x80070032) every time against a normal
    // desktop/server Windows guest, regardless of boot wait or Guest
    // Service Interface state). exec() instead shells out to
    // `Invoke-Command -VMId ... -Credential ...` (PowerShell Direct),
    // which was confirmed working end-to-end against a real guest. The
    // guest account's password must be non-blank: PowerShell Direct
    // rejects blank-password accounts the same way normal network logon
    // does, even though an interactive console session with no password
    // works fine.
    std::string hyperv_guest_username;
    std::string hyperv_guest_password;

    // Podman specific
    std::string podman_image = "docker.io/library/alpine:3.20";
    std::string podman_binary_path = "podman"; // resolved via PATH by default

    // Shared
    std::string workspace_dir = "./vlrf_agent_workspace";
    SandboxLimits limits;

    int max_agent_turns = 25;          // hard cap on the generate<->exec loop
    size_t max_tool_output_bytes = 8192; // per-call output fed back into context

    // Resolve "auto" to a concrete backend for the *current* build+host.
    // Throws std::runtime_error if the resolved backend was compiled out
    // (e.g. asking for HyperV in a Linux-only build).
    SandboxBackendKind resolve_backend() const;

    // Parse from a JSON file on disk. Throws std::runtime_error with a
    // descriptive message on missing file / malformed JSON / unknown keys
    // in strict mode. Unknown keys are ignored (forward-compat) unless
    // strict=true.
    static AgentConfig load_from_file(std::string const& path, bool strict = false);

    // Parse from an in-memory JSON string (used by load_from_file and by
    // tests). See agent_config.cpp for the accepted schema.
    static AgentConfig load_from_json_string(std::string const& json_text, bool strict = false);
};

const char* to_string(SandboxBackendKind k);

} // namespace agent
} // namespace vlrf
