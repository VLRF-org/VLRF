// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Agent Sandbox Interface  (sandbox.hpp)
//
// Abstract execution sandbox used by the coding-agent loop (agent_loop.hpp)
// to run model-requested shell commands and file operations in isolation
// from the host. Two concrete backends implement this interface:
//
//   HyperVSandbox  (src/agent/sandbox_hyperv.cpp)  — Windows only, uses the
//                  Host Compute Service (HCS) API to run a lightweight
//                  utility VM per session.
//   PodmanSandbox  (src/agent/sandbox_podman.cpp)  — Linux (and anywhere
//                  Podman is installed), drives `podman` as a subprocess
//                  to run a rootless container per session.
//
// Neither backend is compiled unless its platform/feature guard is enabled
// (VLRF_AGENT_USE_HYPERV / VLRF_AGENT_USE_PODMAN — see CMakeLists.txt). The
// factory in sandbox_factory.hpp picks a backend at runtime based on
// AgentConfig::sandbox_backend ("auto" resolves by OS at compile+runtime).
//
// SECURITY NOTE: this interface intentionally does not expose raw host
// path access, network configuration, or privilege escalation. Every
// concrete backend MUST confine command execution and file I/O to the
// session's workspace directory. Callers should treat command exit codes,
// stdout, and stderr as untrusted text — do not eval() or shell-interpret
// them further up the stack.
// =============================================================================
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace vlrf {
namespace agent {

// -----------------------------------------------------------------------------
// Result of a single command execution inside the sandbox.
// -----------------------------------------------------------------------------
struct ExecResult {
    bool        launched = false;   // false if the sandbox couldn't even start the process
    int         exit_code = -1;     // process exit code (meaningless if !launched or timed_out)
    bool        timed_out = false;
    std::string stdout_text;
    std::string stderr_text;

    // Combined output truncated to a caller-supplied byte budget, with a
    // marker appended when truncation happened. Kept separate from the
    // untruncated fields above so callers can log the full output while
    // only feeding the truncated version back to the model's context.
    std::string truncated_for_model(size_t max_bytes) const;
};

// -----------------------------------------------------------------------------
// Resource limits applied to every command run in the sandbox. Backends
// enforce what they can (Podman: cgroups + `timeout`; Hyper-V: job-object
// style VM resource caps where available) and MUST fail closed — if a
// backend cannot enforce a limit at all, it should refuse to start the
// sandbox rather than silently ignore the limit.
// -----------------------------------------------------------------------------
struct SandboxLimits {
    int      cpu_count      = 2;
    int64_t  memory_mb      = 2048;
    int      timeout_sec    = 60;     // per-command wall clock timeout
    int64_t  disk_quota_mb  = 4096;   // workspace size cap, 0 = backend default
    bool     network_enabled = false; // default deny; agent tasks rarely need it
};

// -----------------------------------------------------------------------------
// ISandbox — one isolated, disposable execution environment bound to a
// single host workspace directory. Lifetime: create() -> N x exec()/
// write_file()/read_file() -> destroy() (or destructor).
//
// Implementations must be safe to use from a single thread at a time
// (the agent loop is single-threaded per session); they do not need to be
// internally thread-safe.
// -----------------------------------------------------------------------------
class ISandbox {
public:
    virtual ~ISandbox() = default;

    // Bring the sandbox up: create the container/VM, mount the workspace.
    // workspace_host_dir must already exist on the host. Returns false on
    // failure (check last_error()).
    virtual bool create(std::string const& workspace_host_dir,
                         SandboxLimits const& limits) = 0;

    // Run a command inside the sandbox, cwd'd to the mounted workspace.
    // `command` is passed as an argv vector (never a shell string) to
    // avoid a second layer of shell-quoting bugs; if the model wants shell
    // features (pipes, globs) the agent loop wraps it as
    // {"/bin/sh", "-c", command_string} explicitly and visibly — see
    // agent_loop.cpp build_exec_argv().
    virtual ExecResult exec(std::vector<std::string> const& argv) = 0;

    // Write a file into the workspace. path is relative to the workspace
    // root; backends must reject (return false) any path that escapes the
    // workspace (e.g. via "..", absolute paths, or symlink tricks).
    virtual bool write_file(std::string const& relative_path,
                             std::string const& content) = 0;

    // Read a file back out of the workspace, subject to the same path
    // confinement as write_file(). Returns false if the path is invalid,
    // escapes the workspace, or the file doesn't exist.
    virtual bool read_file(std::string const& relative_path,
                            std::string& out_content) = 0;

    // Tear down the container/VM. Safe to call multiple times.
    virtual void destroy() = 0;

    // Human-readable identifier of the concrete backend, e.g.
    // "podman" or "hyperv". Used for logging and for the
    // `{"tool":"exec", ...}` responses' "backend" field.
    virtual const char* backend_name() const = 0;

    // Last human-readable error, valid after any method above returns
    // false / a !launched ExecResult.
    virtual std::string const& last_error() const = 0;
};

} // namespace agent
} // namespace vlrf
