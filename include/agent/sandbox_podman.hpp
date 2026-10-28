// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Podman Sandbox  (sandbox_podman.hpp)
//
// ISandbox implementation backed by a rootless Podman container. Drives
// the `podman` CLI as a subprocess rather than linking libpod directly —
// this keeps VLRF's build free of Podman's Go toolchain/library
// dependency and matches how most tooling scripts against Podman anyway.
//
// Lifecycle:
//   create()  -> `podman run -d --name <session> -v <workspace>:/workspace:Z
//                 --cpus N --memory Mmb --network none <image> sleep infinity`
//   exec()    -> `podman exec --workdir /workspace <session> <argv...>`
//                 wrapped with `timeout <N>` inside the container so a
//                 hung process can't wedge the session.
//   write/read_file() -> direct host filesystem I/O against the bind-mounted
//                 workspace dir (faster and simpler than `podman cp`, and
//                 exactly as safe since it's the same bind mount either way).
//   destroy() -> `podman rm -f <session>`
//
// Path confinement: write_file/read_file resolve the requested relative
// path against workspace_host_dir using a canonicalization check (not
// just a string prefix check) so ".." components and symlinks that would
// escape the workspace are rejected before touching the filesystem.
// =============================================================================
#include "sandbox.hpp"
#include <string>

namespace vlrf {
namespace agent {

class PodmanSandbox : public ISandbox {
public:
    explicit PodmanSandbox(std::string podman_binary_path, std::string image);
    ~PodmanSandbox() override;

    bool create(std::string const& workspace_host_dir, SandboxLimits const& limits) override;
    ExecResult exec(std::vector<std::string> const& argv) override;
    bool write_file(std::string const& relative_path, std::string const& content) override;
    bool read_file(std::string const& relative_path, std::string& out_content) override;
    void destroy() override;
    const char* backend_name() const override { return "podman"; }
    std::string const& last_error() const override { return last_error_; }

    // Exposed for unit testing without a real podman daemon: builds the
    // argv for the `podman run` invocation from limits/workspace/image,
    // without executing it.
    static std::vector<std::string> build_run_argv(std::string const& podman_binary_path,
                                                     std::string const& container_name,
                                                     std::string const& workspace_host_dir,
                                                     std::string const& image,
                                                     SandboxLimits const& limits);

    // Exposed for unit testing: builds the argv for `podman exec`, wrapping
    // the user's command with `timeout <N>` so a runaway process is killed
    // even if the host-side subprocess wait logic has a bug.
    static std::vector<std::string> build_exec_argv(std::string const& podman_binary_path,
                                                      std::string const& container_name,
                                                      std::vector<std::string> const& user_argv,
                                                      int timeout_sec);

    // Resolves `relative_path` against `workspace_root`, rejecting any
    // path that would escape the workspace (via "..", an absolute path,
    // or a symlink pointing outside). Returns empty string on rejection.
    // Exposed for unit testing path-confinement logic in isolation.
    static std::string resolve_workspace_path(std::string const& workspace_root,
                                                std::string const& relative_path);

private:
    std::string podman_binary_path_;
    std::string image_;
    std::string container_name_;
    std::string workspace_host_dir_;
    SandboxLimits limits_;
    bool created_ = false;
    mutable std::string last_error_;

    // Runs a host-side subprocess (used for the `podman` CLI invocations
    // themselves, e.g. `podman run`, `podman exec`, `podman rm`).
    // Captures stdout/stderr/exit code; does not go through a shell.
    static ExecResult run_host_process(std::vector<std::string> const& argv, int timeout_sec);
};

} // namespace agent
} // namespace vlrf
