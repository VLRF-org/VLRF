// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Agent Configuration implementation  (agent_config.cpp)
// =============================================================================
#include "agent_config.hpp"
#include "tiny_json.hpp"
#include <fstream>
#include <sstream>

namespace vlrf {
namespace agent {

using json::Value;
using json::Type;

const char* to_string(SandboxBackendKind k) {
    switch (k) {
        case SandboxBackendKind::Auto:   return "auto";
        case SandboxBackendKind::HyperV: return "hyperv";
        case SandboxBackendKind::Podman: return "podman";
        case SandboxBackendKind::None:   return "none";
    }
    return "unknown";
}

static SandboxBackendKind parse_backend_name(std::string const& s) {
    if (s == "auto")   return SandboxBackendKind::Auto;
    if (s == "hyperv") return SandboxBackendKind::HyperV;
    if (s == "podman") return SandboxBackendKind::Podman;
    if (s == "none")   return SandboxBackendKind::None;
    throw std::runtime_error("agent config: unknown sandbox_backend \"" + s +
                              "\" (expected auto|hyperv|podman|none)");
}

SandboxBackendKind AgentConfig::resolve_backend() const {
    SandboxBackendKind resolved = sandbox_backend;
    if (resolved == SandboxBackendKind::Auto) {
#ifdef _WIN32
        resolved = SandboxBackendKind::HyperV;
#else
        resolved = SandboxBackendKind::Podman;
#endif
    }

    if (resolved == SandboxBackendKind::HyperV) {
#ifndef VLRF_AGENT_USE_HYPERV
        throw std::runtime_error(
            "agent config resolved to \"hyperv\" but this binary was built without "
            "VLRF_AGENT_USE_HYPERV (Windows-only backend). Rebuild with -DVLRF_AGENT_USE_HYPERV=ON "
            "on Windows, or set sandbox_backend to \"podman\"/\"none\".");
#endif
    }
    if (resolved == SandboxBackendKind::Podman) {
#ifndef VLRF_AGENT_USE_PODMAN
        throw std::runtime_error(
            "agent config resolved to \"podman\" but this binary was built without "
            "VLRF_AGENT_USE_PODMAN. Rebuild with -DVLRF_AGENT_USE_PODMAN=ON, or set "
            "sandbox_backend to \"hyperv\"/\"none\".");
#endif
    }
    return resolved;
}

AgentConfig AgentConfig::load_from_json_string(std::string const& json_text, bool strict) {
    AgentConfig cfg;
    Value root = json::parse(json_text); // throws on malformed JSON

    if (!root.is_object()) {
        throw std::runtime_error("agent config: top-level JSON value must be an object");
    }

    static const char* kKnownKeys[] = {
        "sandbox_backend", "hyperv_base_vhd", "hyperv_vm_memory_mb", "hyperv_vm_cpu_count",
        "hyperv_guest_username", "hyperv_guest_password",
        "podman_image", "podman_binary_path", "workspace_dir", "limits",
        "max_agent_turns", "max_tool_output_bytes"
    };

    if (strict) {
        for (auto const& kv : *root.obj) {
            bool known = false;
            for (auto const* k : kKnownKeys) if (kv.first == k) { known = true; break; }
            if (!known) {
                throw std::runtime_error("agent config: unknown key \"" + kv.first +
                                          "\" (strict mode)");
            }
        }
    }

    if (Value const& v = root.get("sandbox_backend"); v.is_string()) {
        cfg.sandbox_backend = parse_backend_name(v.as_string());
    }
    if (Value const& v = root.get("hyperv_base_vhd"); v.is_string()) {
        cfg.hyperv_base_vhd = v.as_string();
    }
    if (Value const& v = root.get("hyperv_vm_memory_mb"); v.type == Type::Number) {
        cfg.hyperv_vm_memory_mb = static_cast<int64_t>(v.as_number());
    }
    if (Value const& v = root.get("hyperv_vm_cpu_count"); v.type == Type::Number) {
        cfg.hyperv_vm_cpu_count = static_cast<int>(v.as_number());
    }
    if (Value const& v = root.get("hyperv_guest_username"); v.is_string()) {
        cfg.hyperv_guest_username = v.as_string();
    }
    if (Value const& v = root.get("hyperv_guest_password"); v.is_string()) {
        cfg.hyperv_guest_password = v.as_string();
    }
    if (Value const& v = root.get("podman_image"); v.is_string()) {
        cfg.podman_image = v.as_string();
    }
    if (Value const& v = root.get("podman_binary_path"); v.is_string()) {
        cfg.podman_binary_path = v.as_string();
    }
    if (Value const& v = root.get("workspace_dir"); v.is_string()) {
        cfg.workspace_dir = v.as_string();
    }
    if (Value const& v = root.get("max_agent_turns"); v.type == Type::Number) {
        cfg.max_agent_turns = static_cast<int>(v.as_number());
    }
    if (Value const& v = root.get("max_tool_output_bytes"); v.type == Type::Number) {
        cfg.max_tool_output_bytes = static_cast<size_t>(v.as_number());
    }

    if (Value const& lim = root.get("limits"); lim.is_object()) {
        if (Value const& v = lim.get("cpu_count"); v.type == Type::Number)
            cfg.limits.cpu_count = static_cast<int>(v.as_number());
        if (Value const& v = lim.get("memory_mb"); v.type == Type::Number)
            cfg.limits.memory_mb = static_cast<int64_t>(v.as_number());
        if (Value const& v = lim.get("timeout_sec"); v.type == Type::Number)
            cfg.limits.timeout_sec = static_cast<int>(v.as_number());
        if (Value const& v = lim.get("disk_quota_mb"); v.type == Type::Number)
            cfg.limits.disk_quota_mb = static_cast<int64_t>(v.as_number());
        if (Value const& v = lim.get("network_enabled"); v.type == Type::Bool)
            cfg.limits.network_enabled = v.as_bool();
    }

    // Fail fast on obviously-bad values rather than letting a sandbox
    // backend discover them later with a more confusing error.
    if (cfg.max_agent_turns <= 0) {
        throw std::runtime_error("agent config: max_agent_turns must be > 0");
    }
    if (cfg.limits.timeout_sec <= 0) {
        throw std::runtime_error("agent config: limits.timeout_sec must be > 0");
    }

    return cfg;
}

AgentConfig AgentConfig::load_from_file(std::string const& path, bool strict) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("agent config: could not open \"" + path + "\"");
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    try {
        return load_from_json_string(buf.str(), strict);
    } catch (std::exception const& e) {
        throw std::runtime_error("agent config: error parsing \"" + path + "\": " + e.what());
    }
}

} // namespace agent
} // namespace vlrf
