// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Hyper-V Sandbox implementation  (sandbox_hyperv.cpp)
// Windows-only (compiled when VLRF_AGENT_USE_HYPERV is defined).
//
// See sandbox_hyperv.hpp for the full lifecycle description and the
// "untested end-to-end" caveat (individual pieces confirmed working by
// hand; the full sequence wired up here has not been run start-to-finish
// yet, and Copy-VMFile specifically has not been tried by hand at all).
// =============================================================================
#include "sandbox_hyperv.hpp"
#include "tiny_json.hpp"
#include <sstream>
#include <fstream>
#include <random>
#include <array>
#include <thread>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <shlwapi.h>
#  pragma comment(lib, "shlwapi.lib")
#endif

namespace vlrf {
namespace agent {

namespace {

std::string make_session_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream oss;
    oss << "vlrf-agent-" << std::hex << rng();
    return oss.str();
}

// Fixed guest-side workspace directory that write_file()/read_file()
// copy into/out of via Copy-VMFile, and that exec() sets as its working
// directory. Unlike the old VSMB-share approach (which mapped
// workspace_host_dir_ directly into the guest at \\workspace\),
// Copy-VMFile always targets an explicit guest-side path rather than
// exposing a live host directory, so a fixed, dedicated path inside the
// guest is used instead of trying to mirror workspace_host_dir_'s name/
// structure. Chosen to be unlikely to collide with anything already on
// a stock Windows guest image.
constexpr char kGuestWorkspaceDir[] = "C:\\vlrf_workspace";

#ifdef _WIN32
// utf8_to_wide is needed for building the UTF-16LE payload
// base64_encode_utf16le encodes for powershell.exe -EncodedCommand (see
// that function's own comment for why -EncodedCommand + UTF-16LE
// specifically). wide_to_utf8 is unused after removing the HCS calls
// that used to need it, but is left in as a general-purpose pair with
// utf8_to_wide in case future changes to this file need it back rather
// than re-deriving MultiByteToWideChar/WideCharToMultiByte boilerplate
// again.
std::wstring utf8_to_wide(std::string const& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), len);
    return w;
}

std::string wide_to_utf8(PCWSTR w) {
    if (!w || !*w) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string s(static_cast<size_t>(len - 1), '\0'); // len includes the null terminator
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
    return s;
}

// Quotes a UTF-8 string as a PowerShell single-quoted string literal
// ('...'), the only quoting style with no backtick/escape-character
// ambiguity in PowerShell -- the sole special case inside a single-
// quoted literal is a literal single quote, escaped by doubling it.
std::string ps_quote(std::string const& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else out += c;
    }
    out += "'";
    return out;
}

// Encodes a UTF-16LE string as base64, the exact input format
// powershell.exe -EncodedCommand requires (a base64'd UTF-16LE byte
// sequence, NOT UTF-8 -- this is a well-documented but easy-to-miss
// PowerShell quirk; base64-encoding UTF-8 bytes here would make
// powershell.exe silently misinterpret the script).
std::string base64_encode_utf16le(std::wstring const& w) {
    static const char* kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    // wchar_t is UTF-16LE-compatible in practice on Windows (LE-native
    // platform), so we can treat each wchar_t as two raw bytes directly
    // rather than doing a separate UTF-16LE serialization pass.
    std::vector<unsigned char> bytes(w.size() * 2);
    for (size_t i = 0; i < w.size(); ++i) {
        bytes[i * 2] = static_cast<unsigned char>(w[i] & 0xFF);
        bytes[i * 2 + 1] = static_cast<unsigned char>((w[i] >> 8) & 0xFF);
    }
    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        unsigned int n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        out += kAlphabet[(n >> 18) & 0x3F];
        out += kAlphabet[(n >> 12) & 0x3F];
        out += kAlphabet[(n >> 6) & 0x3F];
        out += kAlphabet[n & 0x3F];
    }
    size_t remaining = bytes.size() - i;
    if (remaining == 1) {
        unsigned int n = bytes[i] << 16;
        out += kAlphabet[(n >> 18) & 0x3F];
        out += kAlphabet[(n >> 12) & 0x3F];
        out += "==";
    } else if (remaining == 2) {
        unsigned int n = (bytes[i] << 16) | (bytes[i + 1] << 8);
        out += kAlphabet[(n >> 18) & 0x3F];
        out += kAlphabet[(n >> 12) & 0x3F];
        out += kAlphabet[(n >> 6) & 0x3F];
        out += "=";
    }
    return out;
}

// Decodes a standard base64 string (as produced by PowerShell's own
// [Convert]::ToBase64String, which read_file() uses to get raw file
// bytes out of the guest over the same JSON-over-PowerShell-Direct
// channel exec() uses) back to raw bytes. This is plain base64, NOT the
// UTF-16LE-specific encoding base64_encode_utf16le above produces --
// different purpose, different input shape, so a separate function
// rather than trying to generalize one into the other.
std::string base64_decode(std::string const& in) {
    auto decode_char = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1; // '=' padding or invalid -- both handled by caller skipping
    };
    std::string out;
    out.reserve(in.size() / 4 * 3 + 3);
    int val = 0, bits = -8;
    for (char c : in) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
        int d = decode_char(c);
        if (d < 0) continue; // skip anything unexpected rather than throwing
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0) {
            out += static_cast<char>((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return out;
}

// Runs `script` as a pwsh.exe -EncodedCommand host-side subprocess,
// capturing its combined stdout+stderr into out_captured and enforcing
// timeout_ms by killing the process on expiry. Returns false only for a
// launch-level failure (CreatePipe/CreateProcessW itself failing) or a
// timeout; a script that runs to completion but reports its own failure
// (e.g. our scripts' `{ok:false, error:"..."}` JSON convention) still
// returns true here, since from this function's point of view the
// subprocess executed and produced output -- the caller is responsible
// for inspecting that output's content for a logical failure.
//
// Factored out of what was originally duplicated inline in exec()'s
// first version, once create() needed the exact same
// launch+capture+timeout pattern for its own New-VM driver script.
bool run_pwsh_capture(std::string const& script, DWORD timeout_ms,
                       std::string& out_captured, bool* out_timed_out = nullptr) {
    if (out_timed_out) *out_timed_out = false;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE stdout_read = nullptr, stdout_write = nullptr;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        return false;
    }
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stdout_write; // merged; simpler than a second pipe
                                  // for a value this codebase doesn't
                                  // otherwise need split.
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi{};

    // -EncodedCommand (base64 UTF-16LE), not -Command "..." -- callers'
    // scripts embed PowerShell single-quoted literals (via ps_quote())
    // that may themselves contain host-shell-significant characters;
    // -EncodedCommand sidesteps a second, painful layer of host-shell
    // quoting entirely by passing the whole script as one opaque base64
    // blob with no shell-metacharacter surface at all.
    std::wstring script_w = utf8_to_wide(script);
    std::string encoded = base64_encode_utf16le(script_w);
    std::string cmdline_str = "pwsh.exe -NoProfile -NonInteractive -EncodedCommand " + encoded;
    std::wstring cmdline_w = utf8_to_wide(cmdline_str);
    std::vector<wchar_t> cmdline_buf(cmdline_w.begin(), cmdline_w.end());
    cmdline_buf.push_back(L'\0');

    BOOL created = CreateProcessW(nullptr, cmdline_buf.data(), nullptr, nullptr,
                                   TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(stdout_write);
    if (!created) {
        CloseHandle(stdout_read);
        return false;
    }

    // Drain the pipe on a background thread while waiting with a timeout
    // on the process handle, so a chatty/hung remote command can't
    // deadlock this thread by filling the pipe buffer before
    // WaitForSingleObject returns.
    std::string captured;
    std::thread reader([&]() {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(stdout_read, buf, sizeof(buf), &n, nullptr) && n > 0) {
            captured.append(buf, n);
        }
    });

    DWORD wait_rc = WaitForSingleObject(pi.hProcess, timeout_ms);
    bool timed_out = false;
    if (wait_rc == WAIT_TIMEOUT) {
        timed_out = true;
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, 5000);
    }
    // reader's ReadFile loop unblocks on its own via EOF once the
    // process (and its inherited copy of stdout_write, already closed
    // on our side above) has no more open write handles -- true whether
    // the process exited normally or was just terminated above, so
    // join() here is always safe to call unconditionally.
    reader.join();
    CloseHandle(stdout_read);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    out_captured = captured;
    if (out_timed_out) *out_timed_out = timed_out;
    return !timed_out;
}
#endif

} // namespace

HyperVSandbox::HyperVSandbox(std::string base_vhd_path, std::string guest_username, std::string guest_password)
    : base_vhd_path_(std::move(base_vhd_path)),
      guest_username_(std::move(guest_username)),
      guest_password_(std::move(guest_password)) {}

HyperVSandbox::~HyperVSandbox() {
    if (created_) destroy();
}

std::string HyperVSandbox::resolve_workspace_path(std::string const& workspace_root,
                                                    std::string const& relative_path) {
    // Same contract/logic family as PodmanSandbox::resolve_workspace_path:
    // reject absolute paths and any ".." path component outright (this
    // check alone is sufficient and doesn't require realpath-style
    // canonicalization games, since Windows path semantics make the
    // POSIX symlink-ancestor-walk approach used on Linux inapplicable —
    // NTFS symlinks/junctions are handled by requiring the workspace dir
    // itself to be created fresh per-session with no pre-existing
    // reparse points, enforced by create() below rather than here).
#ifndef _WIN32
    (void)workspace_root; (void)relative_path;
    return {}; // this backend only compiles/runs on Windows
#else
    if (relative_path.empty()) return {};
    if (relative_path.size() >= 2 && relative_path[1] == ':') return {}; // "C:\..." absolute
    if (relative_path.front() == '\\' || relative_path.front() == '/') return {};

    std::istringstream iss(relative_path);
    std::string component;
    char sep = (relative_path.find('\\') != std::string::npos) ? '\\' : '/';
    while (std::getline(iss, component, sep)) {
        if (component == "..") return {};
    }

    std::string root = workspace_root;
    if (!root.empty() && root.back() != '\\' && root.back() != '/') root += '\\';
    std::string normalized_rel = relative_path;
    for (auto& c : normalized_rel) if (c == '/') c = '\\';
    return root + normalized_rel;
#endif
}

bool HyperVSandbox::create_differencing_disk() {
#ifdef _WIN32
    differencing_vhd_path_ = base_vhd_path_.substr(0, base_vhd_path_.find_last_of("\\/") + 1)
                              + compute_system_id_ + ".vhdx";
    // Created via PowerShell (New-VHD -Differencing) rather than the
    // VirtDisk COM API (IVhdImage/OpenVirtualDisk) to keep the Win32
    // surface this file depends on smaller; PowerShell's Hyper-V module
    // is present on any host with Hyper-V enabled, which is already a
    // precondition for this backend.
    //
    // Uses pwsh.exe (PowerShell 7+), not powershell.exe (Windows
    // PowerShell 5.1). Confirmed by hand: on the machine this was tested
    // against, powershell.exe's own built-in Microsoft.PowerShell.Security
    // module failed to autoload (ConvertTo-SecureString errored with
    // "the module could not be loaded"), while pwsh.exe's copy of the
    // same module worked fine. This machine had pwsh 7 installed and in
    // regular use; if pwsh.exe isn't guaranteed present in your
    // deployment target, either bundle/require it or first diagnose why
    // powershell.exe's module store is broken there instead of switching
    // binaries.
    std::ostringstream cmd;
    cmd << "pwsh.exe -NoProfile -NonInteractive -Command "
        << "\"New-VHD -Differencing -Path '" << differencing_vhd_path_
        << "' -ParentPath '" << base_vhd_path_ << "'\"";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        last_error_ = "New-VHD differencing disk creation failed (exit " + std::to_string(rc) + ")";
        return false;
    }
    return true;
#else
    return false;
#endif
}

void HyperVSandbox::delete_differencing_disk() {
#ifdef _WIN32
    if (!differencing_vhd_path_.empty()) {
        DeleteFileA(differencing_vhd_path_.c_str());
    }
#endif
}

bool HyperVSandbox::create(std::string const& workspace_host_dir, SandboxLimits const& limits) {
#ifndef _WIN32
    (void)workspace_host_dir; (void)limits;
    last_error_ = "HyperVSandbox is only supported on Windows";
    return false;
#else
    workspace_host_dir_ = workspace_host_dir;
    limits_ = limits;
    compute_system_id_ = make_session_id();

    if (!PathFileExistsA(workspace_host_dir_.c_str())) {
        last_error_ = "workspace_host_dir does not exist: " + workspace_host_dir_;
        return false;
    }
    if (!PathFileExistsA(base_vhd_path_.c_str())) {
        last_error_ = "hyperv_base_vhd not found: " + base_vhd_path_;
        return false;
    }

    if (!create_differencing_disk()) return false;

    // One PowerShell script does New-VM (registers with VMMS, unlike the
    // old HcsCreateComputeSystem approach -- see header comment for why
    // that matters), sets CPU/memory limits, starts it, and reports back
    // the VM's real GUID as JSON -- all in one pwsh.exe invocation rather
    // than one std::system() call per step, so we get structured,
    // parseable results and can distinguish exactly which step failed.
    //
    // -SwitchName: network_enabled selects whether the VM gets a NIC at
    // all. "Default Switch" is the standard built-in Hyper-V switch
    // present on any host with Hyper-V enabled (confirmed present on the
    // test machine this was developed against via `Get-VM nfvm |
    // Get-VMNetworkAdapter`, which showed it connected there) -- if a
    // deployment target has renamed or removed it, this needs to become
    // a configurable switch name rather than this hardcoded default.
    // network_enabled=false: New-VM's -NoVMNetworkAdapter equivalent
    // (omitting -SwitchName controls adapter presence per how New-VM's
    // parameter behavior works -- confirm; some New-VM verb sets require
    // 'Remove-VMNetworkAdapter' after creation instead of not creating
    // one in the first place, which would be a documented gap if
    // network_enabled=false turns out to leave a NIC attached anyway).
    std::ostringstream script;
    script << "$ErrorActionPreference='Stop';"
           << "try {"
           << "  New-VM -Name " << ps_quote(compute_system_id_)
           << " -Generation 2 -MemoryStartupBytes " << (static_cast<int64_t>(limits.memory_mb) * 1024 * 1024)
           << " -VHDPath " << ps_quote(differencing_vhd_path_)
           << (limits.network_enabled ? std::string(" -SwitchName 'Default Switch'") : std::string())
           << " | Out-Null;"
           << "  Set-VMProcessor -VMName " << ps_quote(compute_system_id_)
           << " -Count " << limits.cpu_count << ";"
           << "  Set-VMFirmware -VMName " << ps_quote(compute_system_id_) << " -EnableSecureBoot Off;"
           // Secure Boot disabled: the differencing disk's parent
           // (base_vhd_path_) was validated in earlier manual testing
           // with Secure Boot ON and it worked, but that was against the
           // ORIGINAL registered VM (nfvm), not a fresh New-VM -- turning
           // it off here removes one more variable from a
           // not-yet-fully-tested path. Revisit if the base image
           // actually requires Secure Boot to boot correctly.
           << "  Get-VMIntegrationService -VMName " << ps_quote(compute_system_id_)
           << " | Where-Object { $_.Id -like '*6C09BB55-D683-4DA0-8931-C9BF705F6480*' }"
           << " | Enable-VMIntegrationService;"
           // Matched by the Guest Service Interface's fixed service GUID
           // (6C09BB55-D683-4DA0-8931-C9BF705F6480), NOT by its display
           // name: the name is localized (a Polish host shows "Interfejs
           // usługi gościa"), so `-Name 'Guest Service Interface'` would
           // fail on non-English hosts and, under
           // $ErrorActionPreference='Stop', abort the whole create().
           // Guest Service Interface is believed to be OFF BY DEFAULT on a
           // freshly New-VM'd VM (it was off by default on the original
           // nfvm VM). Observed: Copy-VMFile against a fresh VM failed
           // with "device not ready", 0x80070015 -- "cannot initiate
           // file copy to guest". That this integration service being
           // disabled is the cause is an inference from that error plus
           // the nfvm history, NOT yet verified on a fresh VM -- the
           // newvm_probe rerun after adding this line is what confirms
           // or refutes it. The host-side per-VM toggle and the
           // guest-side service start type (already Automatic inside the
           // guest image from an earlier manual fix) are two separate
           // settings, which is why this is needed even though the
           // guest side was already fixed.
           << "  Start-VM -Name " << ps_quote(compute_system_id_) << ";"
           << "  $vm = Get-VM -Name " << ps_quote(compute_system_id_) << ";"
           << "  [PSCustomObject]@{ ok=$true; vm_id=$vm.Id.ToString(); error=$null } | ConvertTo-Json -Compress"
           << "} catch {"
           << "  [PSCustomObject]@{ ok=$false; vm_id=$null; error=$_.Exception.Message } | ConvertTo-Json -Compress"
           << "}";

    std::string create_result_json;
    if (!run_pwsh_capture(script.str(), 120000, create_result_json)) {
        last_error_ = "create(): failed to launch pwsh.exe driver for New-VM";
        delete_differencing_disk();
        return false;
    }

    bool ok = false;
    std::string vm_id, err;
    try {
        json::Value parsed = json::find_and_parse_first_object(create_result_json);
        ok = parsed.get("ok").as_bool();
        vm_id = parsed.get("vm_id").as_string();
        err = parsed.get("error").as_string();
    } catch (std::exception const& e) {
        last_error_ = std::string("create(): could not parse New-VM driver output: ") + e.what() +
                      "; raw output: " + create_result_json;
        delete_differencing_disk();
        return false;
    }

    if (!ok || vm_id.empty()) {
        last_error_ = "create(): New-VM/Start-VM failed: " + (err.empty() ? create_result_json : err);
        // Best-effort cleanup: New-VM may have partially registered the
        // VM even if a later step (Set-VMProcessor, Start-VM, ...)
        // failed inside the same try block -- Remove-VM here prevents an
        // orphaned, never-started VM object from accumulating in VMMS
        // across repeated failed create() calls. Errors from this best-
        // effort cleanup are intentionally swallowed (run_pwsh_capture's
        // return value is ignored) since create() is already returning
        // failure regardless.
        std::string cleanup_result;
        run_pwsh_capture("Remove-VM -Name " + ps_quote(compute_system_id_) + " -Force -ErrorAction SilentlyContinue",
                          15000, cleanup_result);
        delete_differencing_disk();
        return false;
    }

    runtime_vm_guid_ = vm_id;

    // Start-VM returns once the VM begins powering on, NOT once the
    // guest OS has booted far enough for PowerShell Direct to respond
    // (confirmed by hand in earlier HCS-based testing: HcsStartComputeSystem
    // returning success similarly only meant "powering on", and the
    // guest took well over a minute afterward before Invoke-Command
    // could reach it, worse under host memory pressure). Rather than
    // guess a fixed sleep duration here, poll with a trivial
    // Invoke-Command call until it succeeds or a generous ceiling is
    // hit, so a fast-booting guest doesn't wait needlessly and a slow
    // one still gets a real chance rather than a hardcoded timeout that
    // might be wrong for a different host/guest combination.
    {
        const int max_poll_attempts = 24;      // 24 * 10s = 240s ceiling
        bool ready = false;
        for (int attempt = 0; attempt < max_poll_attempts; ++attempt) {
            std::ostringstream probe;
            probe << "$ErrorActionPreference='Stop';"
                  << "try {"
                  << "  $sec = ConvertTo-SecureString " << ps_quote(guest_password_) << " -AsPlainText -Force;"
                  << "  $cred = New-Object System.Management.Automation.PSCredential("
                  << ps_quote(guest_username_) << ", $sec);"
                  << "  Invoke-Command -VMId " << ps_quote(runtime_vm_guid_)
                  << " -Credential $cred -ScriptBlock { $true } -ErrorAction Stop | Out-Null;"
                  << "  [PSCustomObject]@{ ready=$true } | ConvertTo-Json -Compress"
                  << "} catch {"
                  << "  [PSCustomObject]@{ ready=$false } | ConvertTo-Json -Compress"
                  << "}";
            std::string probe_result;
            run_pwsh_capture(probe.str(), 15000, probe_result);
            try {
                json::Value parsed = json::find_and_parse_first_object(probe_result);
                if (parsed.get("ready").as_bool()) { ready = true; break; }
            } catch (std::exception const&) {
                // Treat a parse failure as "not ready yet" and keep polling
                // rather than aborting create() over a transient hiccup.
            }
            Sleep(10000);
        }
        if (!ready) {
            last_error_ = "create(): VM started but PowerShell Direct never became reachable "
                          "within the polling ceiling -- the guest may still be booting (check "
                          "Guest Service Interface / vmicguestinterface status inside the guest "
                          "by hand), or hyperv_guest_username/hyperv_guest_password may be wrong.";
            // Not returning false here: the VM IS up and registered, and
            // some callers might still get value from write_file() via
            // Copy-VMFile even if this particular readiness probe
            // (Invoke-Command) failed for a transient reason. exec() and
            // write_file()/read_file() will each surface their own clear
            // error on first real use if the guest genuinely isn't
            // reachable.
        }
    }

    created_ = true;
    return true;
#endif
}

ExecResult HyperVSandbox::exec(std::vector<std::string> const& argv) {
    ExecResult result;
#ifndef _WIN32
    (void)argv;
    last_error_ = "HyperVSandbox is only supported on Windows";
    return result;
#else
    if (!created_) {
        last_error_ = "exec() called before create() succeeded";
        return result;
    }
    if (runtime_vm_guid_.empty()) {
        last_error_ = "exec() unavailable: runtime VM GUID was not determined during "
                       "create() (see that method's last_error_ for why)";
        return result;
    }
    if (guest_username_.empty()) {
        last_error_ = "exec() requires hyperv_guest_username/hyperv_guest_password to be "
                       "set in agent config -- PowerShell Direct needs an authenticated "
                       "guest logon, and a blank-password guest account will be rejected "
                       "the same way (confirmed by hand: `net user <name> *` to set a real "
                       "password fixed an identical rejection during manual testing)";
        return result;
    }
    if (argv.empty()) {
        last_error_ = "exec() called with empty argv (need at least an executable name)";
        return result;
    }

    // argv is still passed as a real array into the guest script block
    // (via ArgumentList, not string-concatenated), so this does NOT
    // reintroduce the "shell-like string" quoting concern the old
    // HcsCreateProcess path had -- PowerShell's own -ArgumentList
    // marshals each element as a distinct argument into the script
    // block's $args, and & is invoked with them expanded positionally,
    // so no quoting/escaping of argv's contents happens on the host side
    // at all.
    //
    // The script block itself needs argv.size() named parameters
    // ($a0, $a1, ...) declared via `param(...)`, since PowerShell script
    // blocks don't have a variadic "just splat ArgumentList" shorthand
    // that also captures stdout/stderr/exit code together the way this
    // needs.
    std::ostringstream script_args_decl;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i) script_args_decl << ',';
        script_args_decl << "$a" << i;
    }

    // $a0 is the executable itself (ProcessStartInfo.FileName); the
    // remaining $a1.. are its arguments, built as an explicit
    // comma-separated PowerShell array rather than string-slicing
    // script_args_call (a prior version of this used
    // script_args_call.str().substr(3) to strip "$a0 " off the front,
    // which silently breaks once there are 10+ args and $a0 is no longer
    // exactly 3 characters before the space -- e.g. it would instead cut
    // into "$a10").
    std::ostringstream script_rest_args;
    for (size_t i = 1; i < argv.size(); ++i) {
        if (i > 1) script_rest_args << ',';
        script_rest_args << "$a" << i;
    }

    // The remote script itself: run argv as a native command (not through
    // cmd.exe, so argv[0] must be an executable path/name the guest can
    // resolve — same contract PodmanSandbox::exec has), capture stdout/
    // stderr separately, and report the process exit code as JSON on a
    // single line so the host side can parse it without scraping mixed
    // text. Working directory is kGuestWorkspaceDir (see write_file's
    // comment for why this fixed path replaced the old VSMB share) --
    // created with -Force so exec() doesn't fail on a session's very
    // first command if no write_file() call has happened yet to
    // implicitly create it via Copy-VMFile's -CreateFullPath.
    std::string remote_script =
        "param(" + script_args_decl.str() + ")\n"
        "$ErrorActionPreference = 'Continue'\n"
        "$cwdErr = $null\n"
        "try {\n"
        "  New-Item -ItemType Directory -Force -Path '" + std::string(kGuestWorkspaceDir) + "' | Out-Null\n"
        "  Set-Location '" + std::string(kGuestWorkspaceDir) + "'\n"
        "} catch { $cwdErr = $_.Exception.Message }\n"
        "$stdout = $null; $stderr = $null; $exitCode = -1\n"
        "try {\n"
        "  $psi = New-Object System.Diagnostics.ProcessStartInfo\n"
        "  $psi.FileName = $a0\n"
        "  $restArgs = @(" + script_rest_args.str() + ")\n"
        "  $psi.Arguments = ($restArgs | ForEach-Object { '\"' + ($_ -replace '\"','\\\"') + '\"' }) -join ' '\n"
        "  $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true\n"
        "  $psi.UseShellExecute = $false\n"
        "  $p = [System.Diagnostics.Process]::Start($psi)\n"
        "  $stdout = $p.StandardOutput.ReadToEnd(); $stderr = $p.StandardError.ReadToEnd()\n"
        "  $p.WaitForExit(); $exitCode = $p.ExitCode\n"
        "} catch { $stderr = $_.Exception.Message; $exitCode = -1 }\n"
        "[PSCustomObject]@{ stdout = $stdout; stderr = $stderr; exit_code = $exitCode; cwd_error = $cwdErr } | ConvertTo-Json -Compress\n";

    // The host-side driver script: build the credential, dot-source the
    // remote script into an Invoke-Command call, print ONLY the inner
    // JSON line to stdout (everything else PowerShell might emit --
    // progress, warnings -- goes to its own stderr so it doesn't corrupt
    // the one line this side actually parses).
    std::ostringstream driver;
    driver << "$ErrorActionPreference = 'Stop';"
           << "$sec = ConvertTo-SecureString " << ps_quote(guest_password_) << " -AsPlainText -Force;"
           << "$cred = New-Object System.Management.Automation.PSCredential("
           << ps_quote(guest_username_) << ", $sec);"
           << "$sb = [ScriptBlock]::Create(" << ps_quote(remote_script) << ");"
           << "$argList = @(" ;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i) driver << ',';
        driver << ps_quote(argv[i]);
    }
    driver << ");"
           << "try {"
           << "  $r = Invoke-Command -VMId " << ps_quote(runtime_vm_guid_)
           << " -Credential $cred -ScriptBlock $sb -ArgumentList $argList -ErrorAction Stop;"
           << "  Write-Output $r"
           << "} catch {"
           << "  [PSCustomObject]@{ stdout=''; stderr=$_.Exception.Message; exit_code=-1; cwd_error=$null }"
           << "    | ConvertTo-Json -Compress | Write-Output"
           << "}";

    std::string driver_script = driver.str();

    // Run the driver via the shared helper (factored out after create()
    // needed the identical launch+capture+timeout pattern for its own
    // New-VM driver script -- see run_pwsh_capture's doc comment).
    std::string captured_stdout;
    bool timed_out = false;
    DWORD wait_ms = static_cast<DWORD>(limits_.timeout_sec) * 1000;
    if (!run_pwsh_capture(driver_script, wait_ms, captured_stdout, &timed_out)) {
        if (timed_out) {
            result.timed_out = true;
            last_error_ = "exec(): command exceeded timeout_sec=" + std::to_string(limits_.timeout_sec) +
                          " (host-side deadline on the pwsh.exe/Invoke-Command driver -- "
                          "this remains the primary timeout enforcement, same as the old "
                          "HcsCreateProcess path's rationale, since PowerShell Direct has no "
                          "built-in per-invocation timeout of its own either)";
        } else {
            last_error_ = "exec(): failed to launch host-side pwsh.exe driver, "
                          "GetLastError=" + std::to_string(GetLastError());
        }
        return result;
    }

    // Parse the single JSON line the driver script prints. Tolerate
    // leading/trailing whitespace and any stray PowerShell banner text
    // by using find_and_parse_first_object rather than a strict full-
    // document parse, since -EncodedCommand + -NoProfile suppresses most
    // but not guaranteed all extraneous output.
    try {
        json::Value parsed = json::find_and_parse_first_object(captured_stdout);
        if (parsed.is_null()) {
            last_error_ = "exec(): could not find a JSON result object in the driver's output: " +
                          captured_stdout;
            return result;
        }
        result.stdout_text = parsed.get("stdout").as_string();
        result.stderr_text = parsed.get("stderr").as_string();
        result.exit_code = static_cast<int>(parsed.get("exit_code").as_number(-1));
        std::string cwd_error = parsed.get("cwd_error").as_string();
        if (!cwd_error.empty()) {
            // Surfaced but not fatal: the command may still have run in
            // whatever the guest's default cwd was. See remote_script's
            // comment above on why \\workspace\ is unverified.
            result.stderr_text = "[cwd warning: " + cwd_error + "] " + result.stderr_text;
        }
        result.launched = true;
    } catch (std::exception const& e) {
        last_error_ = std::string("exec(): failed to parse driver output as JSON: ") + e.what() +
                      "; raw output: " + captured_stdout;
    }

    return result;
#endif
}

bool HyperVSandbox::write_file(std::string const& relative_path, std::string const& content) {
#ifndef _WIN32
    (void)relative_path; (void)content;
    last_error_ = "HyperVSandbox is only supported on Windows";
    return false;
#else
    if (!created_ || runtime_vm_guid_.empty()) {
        last_error_ = "write_file() called before create() succeeded";
        return false;
    }
    // resolve_workspace_path's ".."/absolute-path rejection is reused
    // here purely for its path-confinement validation logic, applied
    // against workspace_host_dir_ as before -- write_file/read_file
    // still reason about relative_path the same way, even though the
    // actual write now happens through a temp host file + Copy-VMFile
    // rather than a direct write into workspace_host_dir_.
    std::string resolved_host_check = resolve_workspace_path(workspace_host_dir_, relative_path);
    if (resolved_host_check.empty()) {
        last_error_ = "write_file: path escapes workspace or is invalid: " + relative_path;
        return false;
    }

    // Copy-VMFile copies an existing HOST file into the guest; it has no
    // "write these bytes directly" form, so `content` is first staged to
    // a temp file on the host, then copied in. The temp file is placed
    // under workspace_host_dir_ (already known to exist and be under our
    // control) rather than the system temp dir, purely so cleanup is one
    // DeleteFileA call using a path we already validated.
    std::string temp_host_path = workspace_host_dir_;
    if (!temp_host_path.empty() && temp_host_path.back() != '\\' && temp_host_path.back() != '/') {
        temp_host_path += '\\';
    }
    temp_host_path += "._vlrf_write_staging.tmp";
    {
        std::ofstream f(temp_host_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            last_error_ = "write_file: could not create staging file: " + temp_host_path;
            return false;
        }
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!f) {
            last_error_ = "write_file: failed writing staging file: " + temp_host_path;
            DeleteFileA(temp_host_path.c_str());
            return false;
        }
    }

    // Guest-side destination: kGuestWorkspaceDir + the same relative
    // path structure the caller asked for, so a caller writing
    // "sub/dir/file.txt" ends up at C:\vlrf_workspace\sub\dir\file.txt in
    // the guest, mirroring how the old VSMB approach preserved relative
    // structure under \\workspace\.
    std::string guest_rel = relative_path;
    for (auto& c : guest_rel) if (c == '/') c = '\\';
    std::string guest_dest = std::string(kGuestWorkspaceDir) + "\\" + guest_rel;

    std::ostringstream script;
    script << "$ErrorActionPreference='Stop';"
           << "try {"
           // Copy-VMFile has NO -VMId parameter (confirmed by hand:
           // "Cannot find a parameter matching parameter name 'VMId'"),
           // unlike Invoke-Command which supports both -VMId and
           // -VMName. Use -VMName (compute_system_id_) here instead.
           << "  Copy-VMFile -VMName " << ps_quote(compute_system_id_)
           << " -SourcePath " << ps_quote(temp_host_path)
           << " -DestinationPath " << ps_quote(guest_dest)
           << " -FileSource Host -CreateFullPath -Force;"
           << "  [PSCustomObject]@{ ok=$true; error=$null } | ConvertTo-Json -Compress"
           << "} catch {"
           << "  [PSCustomObject]@{ ok=$false; error=$_.Exception.Message } | ConvertTo-Json -Compress"
           << "}";

    std::string captured;
    bool ok_launch = run_pwsh_capture(script.str(), 30000, captured);
    DeleteFileA(temp_host_path.c_str()); // staging file cleanup either way

    if (!ok_launch) {
        last_error_ = "write_file(): failed to launch pwsh.exe driver for Copy-VMFile";
        return false;
    }
    try {
        json::Value parsed = json::find_and_parse_first_object(captured);
        bool ok = parsed.get("ok").as_bool();
        if (!ok) {
            last_error_ = "write_file(): Copy-VMFile failed: " + parsed.get("error").as_string();
            return false;
        }
        return true;
    } catch (std::exception const& e) {
        last_error_ = std::string("write_file(): could not parse Copy-VMFile driver output: ") +
                      e.what() + "; raw output: " + captured;
        return false;
    }
#endif
}

bool HyperVSandbox::read_file(std::string const& relative_path, std::string& out_content) {
#ifndef _WIN32
    (void)relative_path; (void)out_content;
    last_error_ = "HyperVSandbox is only supported on Windows";
    return false;
#else
    if (!created_ || runtime_vm_guid_.empty()) {
        last_error_ = "read_file() called before create() succeeded";
        return false;
    }
    std::string resolved_host_check = resolve_workspace_path(workspace_host_dir_, relative_path);
    if (resolved_host_check.empty()) {
        last_error_ = "read_file: path escapes workspace or is invalid: " + relative_path;
        return false;
    }

    std::string guest_rel = relative_path;
    for (auto& c : guest_rel) if (c == '/') c = '\\';
    std::string guest_src = std::string(kGuestWorkspaceDir) + "\\" + guest_rel;

    // Copy-VMFile has NO guest-to-host direction on this Hyper-V version
    // at all -- confirmed by hand: "-FileSource Guest" is rejected
    // outright ("Cannot bind parameter 'FileSource'... valid values:
    // Host"), meaning Copy-VMFile here only supports Host->Guest, which
    // is exactly the gap flagged as a risk in this file's header
    // comment. Retrieval instead reuses the same JSON-over-PowerShell-
    // Direct channel exec() already has working: Invoke-Command reads
    // the guest file's bytes, base64-encodes them with PowerShell's own
    // [Convert]::ToBase64String, and this side decodes them back with
    // base64_decode() above. Text-safe JSON transport (base64 has no
    // characters ConvertTo-Json needs to escape) means this works for
    // arbitrary binary content too, not just text files.
    std::ostringstream script;
    script << "$ErrorActionPreference='Stop';"
           << "$sec = ConvertTo-SecureString " << ps_quote(guest_password_) << " -AsPlainText -Force;"
           << "$cred = New-Object System.Management.Automation.PSCredential("
           << ps_quote(guest_username_) << ", $sec);"
           << "try {"
           << "  $r = Invoke-Command -VMId " << ps_quote(runtime_vm_guid_)
           << " -Credential $cred -ScriptBlock {"
           << "    param($p);"
           << "    $bytes = [System.IO.File]::ReadAllBytes($p);"
           << "    [Convert]::ToBase64String($bytes)"
           << "  } -ArgumentList " << ps_quote(guest_src) << " -ErrorAction Stop;"
           << "  [PSCustomObject]@{ ok=$true; data_b64=$r; error=$null } | ConvertTo-Json -Compress"
           << "} catch {"
           << "  [PSCustomObject]@{ ok=$false; data_b64=$null; error=$_.Exception.Message } | ConvertTo-Json -Compress"
           << "}";

    std::string captured;
    if (!run_pwsh_capture(script.str(), 30000, captured)) {
        last_error_ = "read_file(): failed to launch pwsh.exe driver for Invoke-Command";
        return false;
    }
    try {
        json::Value parsed = json::find_and_parse_first_object(captured);
        bool ok = parsed.get("ok").as_bool();
        if (!ok) {
            last_error_ = "read_file(): could not read file from guest: " + parsed.get("error").as_string();
            return false;
        }
        std::string data_b64 = parsed.get("data_b64").as_string();
        out_content = base64_decode(data_b64);
        return true;
    } catch (std::exception const& e) {
        last_error_ = std::string("read_file(): could not parse Invoke-Command driver output: ") +
                      e.what() + "; raw output: " + captured;
        return false;
    }
#endif
}

void HyperVSandbox::destroy() {
#ifdef _WIN32
    if (created_) {
        // Stop-VM -TurnOff (not the default graceful shutdown): mirrors
        // the old HcsTerminateComputeSystem's immediate-stop semantics
        // rather than waiting on a graceful in-guest shutdown, which
        // matters for a differencing disk that's about to be deleted
        // regardless of clean shutdown state.
        std::string result_json;
        run_pwsh_capture(
            "Stop-VM -Name " + ps_quote(compute_system_id_) + " -TurnOff -Force -ErrorAction SilentlyContinue; " +
            "Remove-VM -Name " + ps_quote(compute_system_id_) + " -Force -ErrorAction SilentlyContinue",
            30000, result_json);
        delete_differencing_disk();
        created_ = false;
    }
#endif
}

} // namespace agent
} // namespace vlrf
