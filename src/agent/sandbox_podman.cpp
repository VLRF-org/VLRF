// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Podman Sandbox implementation  (sandbox_podman.cpp)
// Linux-only (compiled when VLRF_AGENT_USE_PODMAN is defined).
// =============================================================================
#include "sandbox_podman.hpp"
#include <sstream>
#include <fstream>
#include <random>
#include <chrono>
#include <cstring>
#include <climits>

#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <cstdlib>
#endif

namespace vlrf {
namespace agent {

namespace {

std::string make_session_name() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream oss;
    oss << "vlrf-agent-" << std::hex << rng();
    return oss.str();
}

} // namespace

PodmanSandbox::PodmanSandbox(std::string podman_binary_path, std::string image)
    : podman_binary_path_(std::move(podman_binary_path)), image_(std::move(image)) {}

PodmanSandbox::~PodmanSandbox() {
    if (created_) destroy();
}

std::vector<std::string> PodmanSandbox::build_run_argv(std::string const& podman_binary_path,
                                                         std::string const& container_name,
                                                         std::string const& workspace_host_dir,
                                                         std::string const& image,
                                                         SandboxLimits const& limits) {
    std::vector<std::string> argv = {
        podman_binary_path, "run", "-d", "--name", container_name,
        "-v", workspace_host_dir + ":/workspace:Z",
        "--workdir", "/workspace",
        "--cpus", std::to_string(limits.cpu_count),
        "--memory", std::to_string(limits.memory_mb) + "m",
        // Default-deny network unless explicitly enabled — most coding
        // agent tasks (compile, test, lint) don't need it, and it closes
        // off a large exfiltration/attack surface by default.
        "--network", (limits.network_enabled ? "bridge" : "none"),
        // Drop privileges as far as podman allows for a general-purpose
        // build/test container: no new privileges, no extra capabilities.
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
    };
    if (limits.disk_quota_mb > 0) {
        argv.push_back("--storage-opt");
        argv.push_back("size=" + std::to_string(limits.disk_quota_mb) + "m");
    }
    argv.push_back(image);
    // Keep the container alive so we can `podman exec` into it repeatedly
    // instead of paying container-start latency per tool call.
    argv.push_back("sleep");
    argv.push_back("infinity");
    return argv;
}

std::vector<std::string> PodmanSandbox::build_exec_argv(std::string const& podman_binary_path,
                                                          std::string const& container_name,
                                                          std::vector<std::string> const& user_argv,
                                                          int timeout_sec) {
    std::vector<std::string> argv = {
        podman_binary_path, "exec", "--workdir", "/workspace", container_name,
        // `timeout` runs inside the container so a wedged process is
        // killed even if the host-side wait has a bug; --kill-after gives
        // it a grace period before SIGKILL if SIGTERM is ignored.
        "timeout", "--kill-after=5", std::to_string(timeout_sec),
    };
    for (auto const& a : user_argv) argv.push_back(a);
    return argv;
}

std::string PodmanSandbox::resolve_workspace_path(std::string const& workspace_root,
                                                    std::string const& relative_path) {
#ifdef _WIN32
    (void)workspace_root; (void)relative_path;
    return {}; // this backend only runs on Linux
#else
    if (relative_path.empty()) return {};
    if (relative_path.front() == '/') return {}; // reject absolute paths outright

    // Canonicalize the workspace root itself (must already exist).
    char root_real[PATH_MAX];
    if (::realpath(workspace_root.c_str(), root_real) == nullptr) return {};
    std::string root_str(root_real);

    // Build the candidate path and canonicalize its *parent directory*
    // (the file itself may not exist yet, e.g. for write_file creating a
    // new file) so we can still catch ".." / symlink escapes.
    std::string candidate = root_str + "/" + relative_path;

    size_t last_slash = candidate.find_last_of('/');
    std::string parent = (last_slash == std::string::npos) ? root_str : candidate.substr(0, last_slash);
    std::string filename = (last_slash == std::string::npos) ? candidate : candidate.substr(last_slash + 1);

    // Walk up until we find an existing ancestor directory to canonicalize
    // (parent may itself not exist yet for a nested write_file call that
    // also needs to create directories — callers of write_file are
    // expected to mkdir -p via the sandbox exec path first for nested
    // paths; here we just need enough of the ancestor chain to exist to
    // validate containment).
    std::string probe = parent;
    while (true) {
        char probe_real[PATH_MAX];
        if (::realpath(probe.c_str(), probe_real) != nullptr) {
            std::string probe_str(probe_real);
            // The resolved ancestor must be root itself or strictly under it.
            if (probe_str != root_str &&
                probe_str.compare(0, root_str.size() + 1, root_str + "/") != 0) {
                return {}; // escaped via symlink somewhere in the ancestor chain
            }
            break;
        }
        size_t slash = probe.find_last_of('/');
        if (slash == std::string::npos || probe == "/") return {}; // couldn't resolve any ancestor
        probe = probe.substr(0, slash);
        if (probe.empty()) probe = "/";
    }

    // Reassemble the *unresolved* tail path (since intermediate dirs
    // beyond `probe` may not exist yet) but re-verify it purely
    // lexically now that we know the resolvable ancestor is contained:
    // reject any ".." component anywhere in relative_path outright,
    // which is the actual escape vector realpath-of-missing-file can't
    // check for us.
    {
        std::istringstream iss(relative_path);
        std::string component;
        while (std::getline(iss, component, '/')) {
            if (component == "..") return {};
        }
    }

    return root_str + "/" + relative_path + (last_slash == std::string::npos ? "" : "");
    (void)filename;
#endif
}

ExecResult PodmanSandbox::run_host_process(std::vector<std::string> const& argv, int timeout_sec) {
    ExecResult result;
#ifdef _WIN32
    (void)argv; (void)timeout_sec;
    result.stderr_text = "PodmanSandbox is not supported on Windows";
    return result;
#else
    if (argv.empty()) {
        result.stderr_text = "empty argv";
        return result;
    }

    int stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        result.stderr_text = "pipe() failed";
        return result;
    }

    pid_t pid = fork();
    if (pid < 0) {
        result.stderr_text = "fork() failed";
        return result;
    }

    if (pid == 0) {
        // Child: redirect stdio, exec argv directly (no shell involved).
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        std::vector<char*> c_argv;
        c_argv.reserve(argv.size() + 1);
        for (auto const& a : argv) c_argv.push_back(const_cast<char*>(a.c_str()));
        c_argv.push_back(nullptr);

        execvp(c_argv[0], c_argv.data());
        // execvp only returns on failure.
        _exit(127);
    }

    // Parent
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    result.launched = true;

    auto read_all = [](int fd, std::string& out) {
        char buf[4096];
        ssize_t n;
        while ((n = read(fd, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
    };

    // Simple approach: poll waitpid with WNOHANG while draining pipes, so
    // a hung child doesn't block us past timeout_sec. This function is
    // used for the `podman` CLI calls themselves (run/exec/rm), which
    // have their own in-container `timeout` wrapper for exec; this host-
    // side timeout is a second, coarser safety net for the CLI call
    // itself hanging (e.g. podman daemon unresponsive).
    auto start = std::chrono::steady_clock::now();
    int status = 0;
    bool exited = false;
    while (true) {
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) { exited = true; break; }
        read_all(stdout_pipe[0], result.stdout_text);
        read_all(stderr_pipe[0], result.stderr_text);
        if (timeout_sec > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_sec) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                result.timed_out = true;
                exited = true;
                break;
            }
        }
        usleep(20 * 1000);
    }
    // Final drain after exit.
    read_all(stdout_pipe[0], result.stdout_text);
    read_all(stderr_pipe[0], result.stderr_text);
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    if (exited && !result.timed_out) {
        result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    return result;
#endif
}

bool PodmanSandbox::create(std::string const& workspace_host_dir, SandboxLimits const& limits) {
#ifdef _WIN32
    last_error_ = "PodmanSandbox is not supported on Windows";
    return false;
#else
    workspace_host_dir_ = workspace_host_dir;
    limits_ = limits;
    container_name_ = make_session_name();

    struct stat st{};
    if (::stat(workspace_host_dir_.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        last_error_ = "workspace_host_dir does not exist or is not a directory: " + workspace_host_dir_;
        return false;
    }

    auto argv = build_run_argv(podman_binary_path_, container_name_, workspace_host_dir_, image_, limits_);
    ExecResult r = run_host_process(argv, /*timeout_sec=*/60);
    if (!r.launched) {
        last_error_ = "failed to launch podman: " + r.stderr_text;
        return false;
    }
    if (r.exit_code != 0) {
        last_error_ = "podman run failed (exit " + std::to_string(r.exit_code) + "): " + r.stderr_text;
        return false;
    }
    created_ = true;
    return true;
#endif
}

ExecResult PodmanSandbox::exec(std::vector<std::string> const& argv) {
    ExecResult result;
    if (!created_) {
        last_error_ = "exec() called before create() succeeded";
        return result;
    }
    auto full_argv = build_exec_argv(podman_binary_path_, container_name_, argv, limits_.timeout_sec);
    // Add a small grace period on top of the in-container `timeout` so the
    // host-side watchdog doesn't race the container-side one.
    result = run_host_process(full_argv, limits_.timeout_sec + 10);
    if (!result.launched) {
        last_error_ = "failed to launch podman exec: " + result.stderr_text;
        return result;
    }
    // The command is always wrapped with the coreutils `timeout` inside
    // the container (see build_exec_argv); that program's own exit code
    // is the primary timeout signal; the host-side watchdog in
    // run_host_process is a secondary net for a hung `podman exec` call
    // itself (e.g. daemon unresponsive), not for the wrapped command.
    // coreutils timeout: 124 = command timed out, 137 = killed via
    // --kill-after (SIGKILL, 128+9).
    if (!result.timed_out && (result.exit_code == 124 || result.exit_code == 137)) {
        result.timed_out = true;
    }
    return result;
}

bool PodmanSandbox::write_file(std::string const& relative_path, std::string const& content) {
#ifdef _WIN32
    (void)relative_path; (void)content;
    last_error_ = "PodmanSandbox is not supported on Windows";
    return false;
#else
    std::string resolved = resolve_workspace_path(workspace_host_dir_, relative_path);
    if (resolved.empty()) {
        last_error_ = "write_file: path escapes workspace or is invalid: " + relative_path;
        return false;
    }
    std::ofstream f(resolved, std::ios::binary | std::ios::trunc);
    if (!f) {
        last_error_ = "write_file: could not open for writing: " + resolved;
        return false;
    }
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    return static_cast<bool>(f);
#endif
}

bool PodmanSandbox::read_file(std::string const& relative_path, std::string& out_content) {
#ifdef _WIN32
    (void)relative_path; (void)out_content;
    last_error_ = "PodmanSandbox is not supported on Windows";
    return false;
#else
    std::string resolved = resolve_workspace_path(workspace_host_dir_, relative_path);
    if (resolved.empty()) {
        last_error_ = "read_file: path escapes workspace or is invalid: " + relative_path;
        return false;
    }
    std::ifstream f(resolved, std::ios::binary);
    if (!f) {
        last_error_ = "read_file: could not open: " + resolved;
        return false;
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    out_content = oss.str();
    return true;
#endif
}

void PodmanSandbox::destroy() {
#ifndef _WIN32
    if (created_) {
        run_host_process({podman_binary_path_, "rm", "-f", container_name_}, 30);
        created_ = false;
    }
#endif
}

} // namespace agent
} // namespace vlrf
