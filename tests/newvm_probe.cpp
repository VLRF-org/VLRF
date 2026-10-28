// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Hyper-V new-VM probe  (newvm_probe.cpp)
// Standalone diagnostic, NOT part of VLRF proper.
//
// Replaces hcs_probe.cpp entirely. That probe validated the raw-HCS
// approach (HcsCreateComputeSystem/HcsStartComputeSystem/HcsCreateProcess),
// which was abandoned after confirming by hand that an HCS-only compute
// system is invisible to VMMS -- and PowerShell Direct (Invoke-Command
// -VMId), which a full Windows guest's exec() needs, only works against
// a VMMS-registered VM. sandbox_hyperv.cpp was rewritten to use
// New-VM/Start-VM/Remove-VM (all VMMS-registering) instead of raw HCS;
// this probe exercises that exact same sequence standalone, so a failure
// shows up here first rather than three layers deep in the real agent
// loop.
//
// This probe uses NO Win32/HCS APIs directly at all -- it's a thin C++
// driver that shells out to pwsh.exe for every single step, mirroring
// exactly what sandbox_hyperv.cpp's create()/exec()/write_file()/
// read_file()/destroy() do. If you'd rather iterate purely in
// PowerShell without rebuilding this each time, the printed script text
// for each step (this probe echoes every script it runs) can be copied
// directly into a pwsh window instead.
//
// Build:
//   cl /std:c++17 /EHsc newvm_probe.cpp shlwapi.lib user32.lib
//
// Usage:
//   newvm_probe.exe <base.vhdx> <host-workspace-dir> <guest-username> <guest-password>
//
// Sequence exercised (matches sandbox_hyperv.cpp exactly):
//   1. New-VHD -Differencing off the base VHDX
//   2. New-VM -Generation 2 -VHDPath <diff disk>, Set-VMProcessor,
//      Set-VMFirmware -EnableSecureBoot Off, Start-VM
//   3. Poll Invoke-Command -VMId until PowerShell Direct responds
//   4. exec()-equivalent: run `whoami` and `hostname` in the guest via
//      Invoke-Command
//   5. write_file()-equivalent: Copy-VMFile a small text file INTO the
//      guest workspace dir (Host -> Guest -- NOT yet tried by hand
//      before this probe)
//   6. read_file()-equivalent: Copy-VMFile that same file back OUT
//      (Guest -> Host -- also not yet tried by hand before this probe;
//      this is the direction most likely to need a fallback if
//      Copy-VMFile's -FileSource Guest isn't supported on this Hyper-V
//      version)
//   7. Stop-VM -TurnOff, Remove-VM, delete the differencing disk
//
// Each step prints PASS/FAIL plus the raw JSON/output so a failure is
// diagnosable without re-running under a debugger.
// =============================================================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "user32.lib")

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <thread>

std::wstring utf8_to_wide(std::string const& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), len);
    return w;
}

// Same PowerShell single-quoted-literal quoting as sandbox_hyperv.cpp's
// ps_quote() -- only a literal ' needs doubling inside '...'.
std::string ps_quote(std::string const& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else out += c;
    }
    out += "'";
    return out;
}

// Same UTF-16LE base64 encoding as sandbox_hyperv.cpp's
// base64_encode_utf16le() -- required format for pwsh.exe -EncodedCommand.
std::string base64_encode_utf16le(std::wstring const& w) {
    static const char* kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> bytes(w.size() * 2);
    for (size_t i = 0; i < w.size(); ++i) {
        bytes[i * 2] = static_cast<unsigned char>(w[i] & 0xFF);
        bytes[i * 2 + 1] = static_cast<unsigned char>((w[i] >> 8) & 0xFF);
    }
    std::string out;
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

// Minimal, non-general JSON string/bool field extractor -- same
// deliberately-dumb approach as hcs_probe.cpp used, since pulling in
// tiny_json.hpp here would mean linking more of the VLRF tree into
// what's meant to be a single self-contained probe file.
std::string extract_json_string_field(std::string const& json, std::string const& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return {};
    pos = json.find('"', pos);
    if (pos == std::string::npos) return {};
    ++pos;
    std::string out;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            char next = json[pos + 1];
            if (next == '\\' || next == '"') { out += next; pos += 2; continue; }
        }
        out += json[pos];
        ++pos;
    }
    return out;
}

bool extract_json_bool_field(std::string const& json, std::string const& key, bool fallback) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < json.size() && json[pos] == ' ') ++pos;
    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return fallback;
}

// Plain base64 decode (not the UTF-16LE-specific encoding above) --
// matches sandbox_hyperv.cpp's base64_decode(), needed here too since
// this probe's step 6 was rewritten to use the same base64-over-
// Invoke-Command approach after confirming Copy-VMFile has no
// guest-to-host direction on this Hyper-V version.
std::string base64_decode(std::string const& in) {
    auto decode_char = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    out.reserve(in.size() / 4 * 3 + 3);
    int val = 0, bits = -8;
    for (char c : in) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
        int d = decode_char(c);
        if (d < 0) continue;
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
// capturing combined stdout+stderr, with a timeout. Mirrors
// sandbox_hyperv.cpp's run_pwsh_capture() exactly, duplicated here
// rather than shared since this probe is meant to stay a single
// self-contained file.
bool run_pwsh_capture(std::string const& script, DWORD timeout_ms, std::string& out_captured) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE stdout_read = nullptr, stdout_write = nullptr;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) return false;
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stdout_write;
    PROCESS_INFORMATION pi{};

    std::wstring script_w = utf8_to_wide(script);
    std::string encoded = base64_encode_utf16le(script_w);
    std::string cmdline_str = "pwsh.exe -NoProfile -NonInteractive -EncodedCommand " + encoded;
    std::wstring cmdline_w = utf8_to_wide(cmdline_str);
    std::vector<wchar_t> cmdline_buf(cmdline_w.begin(), cmdline_w.end());
    cmdline_buf.push_back(L'\0');

    BOOL created = CreateProcessW(nullptr, cmdline_buf.data(), nullptr, nullptr,
                                   TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(stdout_write);
    if (!created) { CloseHandle(stdout_read); return false; }

    std::string captured;
    std::thread reader([&]() {
        char buf[4096]; DWORD n = 0;
        while (ReadFile(stdout_read, buf, sizeof(buf), &n, nullptr) && n > 0) captured.append(buf, n);
    });

    DWORD wait_rc = WaitForSingleObject(pi.hProcess, timeout_ms);
    bool timed_out = (wait_rc == WAIT_TIMEOUT);
    if (timed_out) { TerminateProcess(pi.hProcess, 1); WaitForSingleObject(pi.hProcess, 5000); }
    reader.join();
    CloseHandle(stdout_read);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    out_captured = captured;
    return !timed_out;
}

void print_step(char const* label) {
    std::cout << "\n=== " << label << " ===\n";
}

void print_result(bool pass, std::string const& detail) {
    std::cout << (pass ? "PASS" : "FAIL") << ": " << detail << "\n";
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: newvm_probe.exe <base.vhdx> <host-workspace-dir> <guest-username> <guest-password>\n";
        return 1;
    }
    std::string base_vhd = argv[1];
    std::string workspace = argv[2];
    std::string guest_user = argv[3];
    std::string guest_pass = argv[4];

    if (!PathFileExistsA(base_vhd.c_str())) {
        std::cerr << "base VHDX not found: " << base_vhd << "\n";
        return 1;
    }
    if (!PathFileExistsA(workspace.c_str())) {
        std::cerr << "workspace dir not found: " << workspace << "\n";
        return 1;
    }

    std::mt19937_64 rng{ std::random_device{}() };
    std::ostringstream idoss;
    idoss << "vlrf-probe-" << std::hex << rng();
    std::string vm_name = idoss.str();
    std::string diff_vhd = base_vhd.substr(0, base_vhd.find_last_of("\\/") + 1) + vm_name + ".vhdx";
    std::string const kGuestWorkspaceDir = "C:\\vlrf_workspace";

    bool any_failed = false;

    // ---- Step 1: New-VHD -Differencing --------------------------------
    print_step("1. New-VHD -Differencing");
    {
        std::ostringstream script;
        script << "New-VHD -Differencing -Path " << ps_quote(diff_vhd)
               << " -ParentPath " << ps_quote(base_vhd) << " | Out-Null; "
               << "Write-Output 'OK'";
        std::cout << "script: " << script.str() << "\n";
        std::string out;
        bool launched = run_pwsh_capture(script.str(), 30000, out);
        bool pass = launched && out.find("OK") != std::string::npos;
        print_result(pass, out);
        if (!pass) { any_failed = true; std::cerr << "\nSTOP HERE.\n"; return 1; }
    }

    // ---- Step 2: New-VM / Set-VMProcessor / Set-VMFirmware / Start-VM -
    print_step("2. New-VM + Set-VMProcessor + Set-VMFirmware + Start-VM");
    std::string vm_guid;
    {
        std::ostringstream script;
        script << "$ErrorActionPreference='Stop';"
               << "try {"
               << "  New-VM -Name " << ps_quote(vm_name)
               << " -Generation 2 -MemoryStartupBytes 2147483648"
               << " -VHDPath " << ps_quote(diff_vhd)
               << " -SwitchName 'Default Switch' | Out-Null;"
               << "  Set-VMProcessor -VMName " << ps_quote(vm_name) << " -Count 2;"
               << "  Set-VMFirmware -VMName " << ps_quote(vm_name) << " -EnableSecureBoot Off;"
               << "  Get-VMIntegrationService -VMName " << ps_quote(vm_name)
               << " | Where-Object { $_.Id -like '*6C09BB55-D683-4DA0-8931-C9BF705F6480*' }"
               << " | Enable-VMIntegrationService;"
               << "  Start-VM -Name " << ps_quote(vm_name) << ";"
               << "  $vm = Get-VM -Name " << ps_quote(vm_name) << ";"
               << "  [PSCustomObject]@{ ok=$true; vm_id=$vm.Id.ToString(); error=$null } | ConvertTo-Json -Compress"
               << "} catch {"
               << "  [PSCustomObject]@{ ok=$false; vm_id=$null; error=$_.Exception.Message } | ConvertTo-Json -Compress"
               << "}";
        std::cout << "script: " << script.str() << "\n";
        std::string out;
        bool launched = run_pwsh_capture(script.str(), 60000, out);
        bool ok = launched && extract_json_bool_field(out, "ok", false);
        vm_guid = extract_json_string_field(out, "vm_id");
        print_result(ok && !vm_guid.empty(), out);
        if (!ok || vm_guid.empty()) {
            any_failed = true;
            std::cerr << "\nSTOP HERE. Cleaning up before exit...\n";
            std::string cleanup;
            run_pwsh_capture("Remove-VM -Name " + ps_quote(vm_name) + " -Force -ErrorAction SilentlyContinue",
                              15000, cleanup);
            DeleteFileA(diff_vhd.c_str());
            return 1;
        }
        std::cout << "  -> VM GUID: " << vm_guid << "\n";
    }

    // ---- Step 3: poll for PowerShell Direct readiness ------------------
    print_step("3. Poll Invoke-Command -VMId until guest responds");
    bool guest_ready = false;
    {
        const int max_attempts = 24; // 24 * 10s = 240s ceiling
        for (int attempt = 1; attempt <= max_attempts; ++attempt) {
            std::ostringstream script;
            script << "$ErrorActionPreference='Stop';"
                   << "try {"
                   << "  $sec = ConvertTo-SecureString " << ps_quote(guest_pass) << " -AsPlainText -Force;"
                   << "  $cred = New-Object System.Management.Automation.PSCredential("
                   << ps_quote(guest_user) << ", $sec);"
                   << "  Invoke-Command -VMId " << ps_quote(vm_guid)
                   << " -Credential $cred -ScriptBlock { $true } -ErrorAction Stop | Out-Null;"
                   << "  [PSCustomObject]@{ ready=$true } | ConvertTo-Json -Compress"
                   << "} catch {"
                   << "  [PSCustomObject]@{ ready=$false; error=$_.Exception.Message } | ConvertTo-Json -Compress"
                   << "}";
            std::string out;
            run_pwsh_capture(script.str(), 15000, out);
            bool ready = extract_json_bool_field(out, "ready", false);
            std::cout << "  attempt " << attempt << "/" << max_attempts << ": "
                      << (ready ? "ready" : "not yet") << "\n";
            if (ready) { guest_ready = true; break; }
            Sleep(10000);
        }
        print_result(guest_ready, guest_ready ? "guest reachable via PowerShell Direct"
                                               : "guest never became reachable within 240s");
        if (!guest_ready) any_failed = true;
    }

    // ---- Step 4: exec()-equivalent -------------------------------------
    if (guest_ready) {
        print_step("4. exec()-equivalent: whoami / hostname via Invoke-Command");
        std::ostringstream script;
        script << "$ErrorActionPreference='Stop';"
               << "$sec = ConvertTo-SecureString " << ps_quote(guest_pass) << " -AsPlainText -Force;"
               << "$cred = New-Object System.Management.Automation.PSCredential("
               << ps_quote(guest_user) << ", $sec);"
               << "try {"
               << "  $r = Invoke-Command -VMId " << ps_quote(vm_guid)
               << " -Credential $cred -ScriptBlock { \"$(whoami) | $(hostname)\" } -ErrorAction Stop;"
               << "  [PSCustomObject]@{ ok=$true; output=$r; error=$null } | ConvertTo-Json -Compress"
               << "} catch {"
               << "  [PSCustomObject]@{ ok=$false; output=$null; error=$_.Exception.Message } | ConvertTo-Json -Compress"
               << "}";
        std::cout << "script: " << script.str() << "\n";
        std::string out;
        run_pwsh_capture(script.str(), 30000, out);
        bool ok = extract_json_bool_field(out, "ok", false);
        print_result(ok, out);
        if (!ok) any_failed = true;
    }

    // ---- Step 5: write_file()-equivalent: Copy-VMFile Host -> Guest ----
    std::string local_marker_path = workspace;
    if (!local_marker_path.empty() && local_marker_path.back() != '\\' && local_marker_path.back() != '/')
        local_marker_path += '\\';
    local_marker_path += "newvm_probe_marker.txt";
    std::string marker_content = "hello from newvm_probe";
    if (guest_ready) {
        print_step("5. write_file()-equivalent: Copy-VMFile Host -> Guest");
        {
            std::ofstream f(local_marker_path, std::ios::binary | std::ios::trunc);
            f << marker_content;
        }
        std::string guest_dest = kGuestWorkspaceDir + "\\marker.txt";
        std::ostringstream script;
        script << "$ErrorActionPreference='Stop';"
               << "try {"
               // Copy-VMFile has NO -VMId parameter (confirmed by hand
               // in the first run of this probe) -- use -VMName instead.
               << "  Copy-VMFile -VMName " << ps_quote(vm_name)
               << " -SourcePath " << ps_quote(local_marker_path)
               << " -DestinationPath " << ps_quote(guest_dest)
               << " -FileSource Host -CreateFullPath -Force;"
               << "  [PSCustomObject]@{ ok=$true; error=$null } | ConvertTo-Json -Compress"
               << "} catch {"
               << "  [PSCustomObject]@{ ok=$false; error=$_.Exception.Message } | ConvertTo-Json -Compress"
               << "}";
        std::cout << "script: " << script.str() << "\n";
        std::string out;
        run_pwsh_capture(script.str(), 30000, out);
        bool ok = extract_json_bool_field(out, "ok", false);
        print_result(ok, out);
        if (!ok) {
            any_failed = true;
            std::cerr << "  NOTE: Copy-VMFile Host->Guest failing means write_file() in the\n"
                "  real sandbox_hyperv.cpp will not work as written -- see that file's\n"
                "  header comment for the base64-over-exec() fallback approach.\n";
        }
    }

    // ---- Step 6: read_file()-equivalent: base64-over-Invoke-Command ----
    if (guest_ready) {
        print_step("6. read_file()-equivalent: base64-encoded read via Invoke-Command "
                   "(Copy-VMFile -FileSource Guest confirmed unavailable on this Hyper-V version)");
        std::string guest_src = kGuestWorkspaceDir + "\\marker.txt";

        std::ostringstream script;
        script << "$ErrorActionPreference='Stop';"
               << "$sec = ConvertTo-SecureString " << ps_quote(guest_pass) << " -AsPlainText -Force;"
               << "$cred = New-Object System.Management.Automation.PSCredential("
               << ps_quote(guest_user) << ", $sec);"
               << "try {"
               << "  $r = Invoke-Command -VMId " << ps_quote(vm_guid)
               << " -Credential $cred -ScriptBlock {"
               << "    param($p);"
               << "    $bytes = [System.IO.File]::ReadAllBytes($p);"
               << "    [Convert]::ToBase64String($bytes)"
               << "  } -ArgumentList " << ps_quote(guest_src) << " -ErrorAction Stop;"
               << "  [PSCustomObject]@{ ok=$true; data_b64=$r; error=$null } | ConvertTo-Json -Compress"
               << "} catch {"
               << "  [PSCustomObject]@{ ok=$false; data_b64=$null; error=$_.Exception.Message } | ConvertTo-Json -Compress"
               << "}";
        std::cout << "script: " << script.str() << "\n";
        std::string out;
        run_pwsh_capture(script.str(), 30000, out);
        bool ok = extract_json_bool_field(out, "ok", false);
        if (ok) {
            std::string data_b64 = extract_json_string_field(out, "data_b64");
            std::string decoded = base64_decode(data_b64);
            bool content_matches = (decoded == marker_content);
            print_result(content_matches,
                         content_matches ? "content matches: \"" + decoded + "\""
                                          : "content MISMATCH, got: \"" + decoded + "\"");
            if (!content_matches) any_failed = true;
        } else {
            print_result(false, out);
            any_failed = true;
        }
    }

    // ---- Step 7: teardown ------------------------------------------------
    print_step("7. Stop-VM + Remove-VM + delete differencing disk");
    {
        std::string out;
        run_pwsh_capture(
            "Stop-VM -Name " + ps_quote(vm_name) + " -TurnOff -Force -ErrorAction SilentlyContinue; " +
            "Remove-VM -Name " + ps_quote(vm_name) + " -Force -ErrorAction SilentlyContinue; " +
            "Write-Output 'done'",
            30000, out);
        DeleteFileA(diff_vhd.c_str());
        DeleteFileA(local_marker_path.c_str());
        print_result(true, "torn down (best-effort; check Get-VM by hand if unsure)");
    }

    std::cout << "\n" << (any_failed ? "ONE OR MORE STEPS FAILED -- see FAIL lines above.\n"
                                      : "ALL STEPS PASSED.\n");
    return any_failed ? 1 : 0;
}
