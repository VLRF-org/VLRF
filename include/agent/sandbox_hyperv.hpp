// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Hyper-V Sandbox  (sandbox_hyperv.hpp)
//
// ISandbox implementation backed by a Hyper-V VM, driven entirely through
// PowerShell (New-VM/Start-VM/Remove-VM, Invoke-Command -VMId for exec,
// Copy-VMFile for file I/O) rather than the raw Host Compute Service
// (HCS) API.
//
// THIS REPLACES AN EARLIER HCS-BASED VERSION. That version used
// HcsCreateComputeSystem/HcsStartComputeSystem directly, which does work
// (confirmed by hand: schema accepted, VM boots) but creates a compute
// system that VMMS never registers -- `Get-VM`, `vmconnect`, and,
// critically, `Invoke-Command -VMId` all query VMMS, not raw HCS, and all
// fail against an HCS-only compute system with errors like "Hyper-V was
// unable to find a virtual machine with id ..." (confirmed by hand).
// Since exec() needs PowerShell Direct (see below for why), and
// PowerShell Direct needs a VMMS-registered VM, create()/destroy() were
// switched to go through VMMS too (New-VM/Remove-VM), so the whole
// lifecycle uses one consistent, mutually-compatible registration path
// instead of straddling two separate, non-interoperating VM registries.
//
// Lifecycle:
//   create()  -> New-VHD -Differencing (copy-on-write off the base VHDX,
//                same as before), then New-VM -VHDPath <differencing
//                disk> -Generation 2 ... plus Set-VMProcessor/
//                Set-VMMemory for the CPU/memory limits, then Start-VM.
//                All via a pwsh.exe subprocess (see create_differencing_disk()'s
//                comment for why pwsh.exe specifically, not powershell.exe).
//   exec()    -> Invoke-Command -VMId <guid> -Credential <guest cred>
//                -ScriptBlock {...} over PowerShell Direct. Confirmed by
//                hand that this requires (a) a VMMS-registered VM -- see
//                above -- and (b) a non-blank guest account password:
//                PowerShell Direct rejects blank-password accounts the
//                same way normal network logon does, even though an
//                interactive console session with no password works
//                fine. HcsCreateProcess (the more "direct" HCS API for
//                this) was tried first and confirmed, by hand, to fail
//                every time with ERROR_NOT_SUPPORTED (0x80070032) against
//                a full Windows VM guest -- that API path is built for
//                container/minimal-utility-VM guests with a purpose-built
//                init process listening for HCS exec requests, which a
//                normal Windows install doesn't have.
//   write/read_file() -> write_file() uses Copy-VMFile -FileSource Host
//                (guest<->host file copy over PowerShell Direct's
//                channel) -- but Copy-VMFile has NO -VMId parameter
//                (confirmed by hand: only -VMName), so compute_system_id_
//                is used there instead of runtime_vm_guid_. read_file()
//                CANNOT use Copy-VMFile at all: confirmed by hand that
//                -FileSource Guest is not a valid value on this Hyper-V
//                version (only "Host" is accepted), meaning Copy-VMFile
//                here is Host->Guest only. read_file() instead reads the
//                guest file's bytes via Invoke-Command (base64-encoded
//                with the guest's own [Convert]::ToBase64String, decoded
//                host-side), reusing the same PowerShell Direct channel
//                exec() already has working. VSMB shares (the original
//                HCS-based approach) are not used at all -- they're an
//                HCS-compute-system-document concept with no equivalent
//                in the New-VM/VMMS model used here.
//   destroy() -> Stop-VM -TurnOff, Remove-VM, then delete the
//                differencing disk.
//
// IMPORTANT — confirmed working end-to-end via a standalone probe
// (tests/newvm_probe.cpp), with one real limitation found and fixed:
// New-VHD -Differencing, New-VM/Start-VM, PowerShell Direct readiness
// polling, exec() via Invoke-Command, and write_file() via Copy-VMFile
// all passed by hand. Copy-VMFile's -FileSource Guest direction does
// NOT exist on the tested Hyper-V version (only "Host" is a valid
// value), so read_file() uses a different mechanism entirely (base64
// over Invoke-Command) rather than Copy-VMFile -- see read_file()'s own
// comment. If retargeting to a different Hyper-V version, rerun
// newvm_probe.exe there before assuming Copy-VMFile's Guest direction is
// still unavailable; it may differ by version.
// =============================================================================
#include "sandbox.hpp"
#include <string>

namespace vlrf {
namespace agent {

class HyperVSandbox : public ISandbox {
public:
    // base_vhd_path: path to the golden/parent VHDX for the utility VM.
    // Must already have a guest OS installed. guest_username/guest_password:
    // logon used by exec() via PowerShell Direct (see exec()'s comment for
    // why this replaced HcsCreateProcess). The guest account must have a
    // non-blank password -- PowerShell Direct rejects blank-password
    // accounts.
    HyperVSandbox(std::string base_vhd_path, std::string guest_username, std::string guest_password);
    ~HyperVSandbox() override;

    bool create(std::string const& workspace_host_dir, SandboxLimits const& limits) override;
    ExecResult exec(std::vector<std::string> const& argv) override;
    bool write_file(std::string const& relative_path, std::string const& content) override;
    bool read_file(std::string const& relative_path, std::string& out_content) override;
    void destroy() override;
    const char* backend_name() const override { return "hyperv"; }
    std::string const& last_error() const override { return last_error_; }

    // Same path-confinement contract as PodmanSandbox::resolve_workspace_path
    // (".." / absolute-path / symlink-escape rejection); duplicated rather
    // than shared across the Windows/Linux compile boundary since the two
    // backends are never linked into the same TU guard.
    static std::string resolve_workspace_path(std::string const& workspace_root,
                                                std::string const& relative_path);

private:
    std::string base_vhd_path_;
    std::string guest_username_;
    std::string guest_password_;
    std::string differencing_vhd_path_;
    std::string workspace_host_dir_;
    // Used as the VM's -Name in New-VM. Unlike the earlier HCS-based
    // version, this name IS registered with VMMS (New-VM does that), so
    // `Get-VM <compute_system_id_>` etc. will find it -- unlike the old
    // HCS-only compute system, which existed outside VMMS entirely.
    std::string compute_system_id_;
    // The GUID PowerShell Direct (Invoke-Command -VMId) needs. Captured
    // in create() via `(Get-VM -Name compute_system_id_).Id` after
    // New-VM. Kept as a separate field from compute_system_id_ (rather
    // than just using the name everywhere) because Invoke-Command
    // -VMName was observed to be less reliable than -VMId in early
    // testing of the old HCS-based approach's PowerShell Direct calls;
    // -VMId is used throughout for consistency.
    std::string runtime_vm_guid_;
    SandboxLimits limits_;
    bool created_ = false;
    mutable std::string last_error_;

    bool create_differencing_disk();
    void delete_differencing_disk();
};

} // namespace agent
} // namespace vlrf
