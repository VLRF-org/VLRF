// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Sandbox support types implementation  (sandbox.cpp)
// =============================================================================
#include "sandbox.hpp"

namespace vlrf {
namespace agent {

std::string ExecResult::truncated_for_model(size_t max_bytes) const {
    std::string combined;
    if (!stdout_text.empty()) combined += "stdout:\n" + stdout_text;
    if (!stderr_text.empty()) {
        if (!combined.empty()) combined += "\n";
        combined += "stderr:\n" + stderr_text;
    }
    combined += "\nexit_code: " + std::to_string(exit_code);
    if (timed_out) combined += " (timed out)";

    if (combined.size() <= max_bytes) return combined;

    // Truncate from the middle-out isn't worth the complexity here: the
    // tail of stdout/stderr (most recent output, often the actual error)
    // is usually more useful to a coding agent than the head, so keep the
    // last max_bytes rather than the first.
    std::string truncated = combined.substr(combined.size() - max_bytes);
    return "...[truncated, showing last " + std::to_string(max_bytes) + " bytes]...\n" + truncated;
}

} // namespace agent
} // namespace vlrf
