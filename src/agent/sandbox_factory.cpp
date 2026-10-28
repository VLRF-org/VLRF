// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Sandbox Factory implementation  (sandbox_factory.cpp)
// =============================================================================
#include "sandbox_factory.hpp"

#ifdef VLRF_AGENT_USE_PODMAN
#include "sandbox_podman.hpp"
#endif
#ifdef VLRF_AGENT_USE_HYPERV
#include "sandbox_hyperv.hpp"
#endif

namespace vlrf {
namespace agent {

std::unique_ptr<ISandbox> make_sandbox(AgentConfig const& cfg) {
    SandboxBackendKind resolved = cfg.resolve_backend(); // throws if unavailable

    switch (resolved) {
        case SandboxBackendKind::Podman:
#ifdef VLRF_AGENT_USE_PODMAN
            return std::make_unique<PodmanSandbox>(cfg.podman_binary_path, cfg.podman_image);
#else
            return nullptr; // unreachable: resolve_backend() already threw above
#endif
        case SandboxBackendKind::HyperV:
#ifdef VLRF_AGENT_USE_HYPERV
            return std::make_unique<HyperVSandbox>(cfg.hyperv_base_vhd,
                                                     cfg.hyperv_guest_username,
                                                     cfg.hyperv_guest_password);
#else
            return nullptr; // unreachable: resolve_backend() already threw above
#endif
        case SandboxBackendKind::None:
        case SandboxBackendKind::Auto: // resolve_backend() never returns Auto
        default:
            return nullptr;
    }
}

const char* agent_sandbox_capabilities() {
    static const std::string info = std::string("podman: ")
#ifdef VLRF_AGENT_USE_PODMAN
        + "yes"
#else
        + "no (build with -DVLRF_AGENT_USE_PODMAN=ON)"
#endif
        + ", hyperv: "
#ifdef VLRF_AGENT_USE_HYPERV
        + "yes"
#else
        + "no (Windows-only; build with -DVLRF_AGENT_USE_HYPERV=ON on Windows)"
#endif
        ;
    return info.c_str();
}

} // namespace agent
} // namespace vlrf
