// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Command-line interface  (vlrf_cli.cpp)
// Usage:
//   vlrf_cli init    [disabled] throws — download weights from huggingface.co
//                  and use tools/converter/ instead.
//   vlrf_cli run     --model model.nfw [--tok tok.nft] --prompt "..." [--tokens 128]
//   vlrf_cli info    --model model.nfw
//   vlrf_cli agent   --model model.nfw [--tok tok.nft] --task "..." [--agent-config agent.json]
// =============================================================================
#include "transformer.hpp"
#include "tokenizer.hpp"
#include "vlrf.h"
#include "agent_config.hpp"
#include "agent_loop.hpp"
#include "sandbox_factory.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <memory>

// Simple arg parser
struct Args {
    std::string command;
    std::string model_path;
    std::string tok_path;
    std::string out_path;
    std::string prompt;
    std::string config_name = "small";
    int max_tokens = 128;
    int ctx_len = 0;        // 0 = use model default (capped by RAM)
    float temperature = 0.8f;
    bool greedy = false;
    unsigned int seed = 42;
    std::string backend = "auto";   // auto|cpu|opencl|sycl
    std::string kv_cache = "f32";   // f32|f16|q8_0 — see VlrfKVCacheType
    float repeat_penalty = 1.0f;    // 1.0 = off. See GenerationConfig.
    float freq_penalty = 0.0f;      // 0.0 = off.
    float presence_penalty = 0.0f;  // 0.0 = off.
    int no_repeat_ngram = 0;        // 0 = off.
    std::vector<std::string> stop_strings; // repeatable --stop "..."

    // agent command
    std::string task;               // --task "..." (the coding task for the agent)
    std::string agent_config_path;  // --agent-config agent.json (optional; defaults below)
    std::string sandbox_backend_override; // --sandbox auto|hyperv|podman|none (overrides config file)
};

// Maps the --kv-cache string to an VlrfKVCacheType. Returns false (and prints
// an error) on an unrecognized value.
static bool parse_kv_cache_type(std::string const& name, VlrfKVCacheType& out) {
    if (name == "f32")       out = VLRF_KV_F32;
    else if (name == "f16")  out = VLRF_KV_F16;
    else if (name == "q8_0" || name == "q8") out = VLRF_KV_Q8_0;
    else {
        std::cerr << "Unknown --kv-cache '" << name << "' (expected f32|f16|q8_0)\n";
        return false;
    }
    return true;
}

// Maps the --backend string to an VlrfBackend and applies it via
// vlrf_set_backend(). Prints what was requested and what actually ended up
// active (auto/opencl/sycl can silently fall back to CPU if unavailable
// or not compiled in), so the user isn't left guessing from
// vlrf_backend_capabilities() output alone.
static bool apply_backend(std::string const& name) {
    VlrfBackend b;
    if (name == "auto")   b = VLRF_BACKEND_AUTO;
    else if (name == "cpu")    b = VLRF_BACKEND_CPU;
    else if (name == "opencl") b = VLRF_BACKEND_OPENCL;
    else if (name == "sycl")   b = VLRF_BACKEND_SYCL;
    else {
        std::cerr << "Unknown --backend '" << name << "' (expected auto|cpu|opencl|sycl)\n";
        return false;
    }

    VlrfStatus st = vlrf_set_backend(b);
    if (st != VLRF_OK) {
        std::cerr << "Warning: vlrf_set_backend(\"" << name << "\") failed: "
            << vlrf_status_str(st) << " — staying on previous backend\n";
    }

    std::cout << "Backend requested: " << name << "\n";
    std::cout << "Backend active   : " << vlrf_backend_info() << "\n";
    return true;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) return a;
    a.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--model" && i + 1 < argc) a.model_path = argv[++i];
        if (s == "--tok" && i + 1 < argc) a.tok_path = argv[++i];
        if (s == "--out" && i + 1 < argc) a.out_path = argv[++i];
        if (s == "--prompt" && i + 1 < argc) a.prompt = argv[++i];
        if (s == "--config" && i + 1 < argc) a.config_name = argv[++i];
        if (s == "--tokens" && i + 1 < argc) a.max_tokens = std::stoi(argv[++i]);
        if (s == "--ctx" && i + 1 < argc) a.ctx_len = std::stoi(argv[++i]);
        if (s == "--temp" && i + 1 < argc) a.temperature = std::stof(argv[++i]);
        if (s == "--seed" && i + 1 < argc) a.seed = std::stoul(argv[++i]);
        if (s == "--backend" && i + 1 < argc) a.backend = argv[++i];
        if (s == "--kv-cache" && i + 1 < argc) a.kv_cache = argv[++i];
        if (s == "--repeat-penalty" && i + 1 < argc) a.repeat_penalty = std::stof(argv[++i]);
        if (s == "--freq-penalty" && i + 1 < argc) a.freq_penalty = std::stof(argv[++i]);
        if (s == "--presence-penalty" && i + 1 < argc) a.presence_penalty = std::stof(argv[++i]);
        if (s == "--no-repeat-ngram" && i + 1 < argc) a.no_repeat_ngram = std::stoi(argv[++i]);
        if (s == "--stop" && i + 1 < argc) a.stop_strings.push_back(argv[++i]);
        if (s == "--greedy")               a.greedy = true;
        if (s == "--task" && i + 1 < argc) a.task = argv[++i];
        if (s == "--agent-config" && i + 1 < argc) a.agent_config_path = argv[++i];
        if (s == "--sandbox" && i + 1 < argc) a.sandbox_backend_override = argv[++i];
    }
    return a;
}

static void print_usage() {
    std::cout <<
        "VLRF CLI v1.1.2\n\n"
        "Commands:\n"
        "  init   [disabled] Random weights aren't a usable model.\n"
        "         Download pretrained weights from https://huggingface.co and\n"
        "         convert them with tools/converter/ instead (see below).\n\n"
        "  run    --model <file.nfw> [--tok <file.nft>] --prompt \"...\"\n"
        "         [--tokens N] [--ctx N] [--temp F] [--seed N] [--greedy]\n"
        "         [--backend auto|cpu|opencl|sycl] [--kv-cache f32|f16|q8_0]\n"
        "         [--repeat-penalty F] [--freq-penalty F] [--presence-penalty F]\n"
        "         [--no-repeat-ngram N] [--stop \"...\"]  (--stop is repeatable)\n"
        "         Run inference on a prompt.\n\n"
        "  info   --model <file.nfw> [--backend auto|cpu|opencl|sycl] [--kv-cache f32|f16|q8_0]\n"
        "         Print model architecture summary.\n\n"
        "  agent  --model <file.nfw> [--tok <file.nft>] --task \"...\"\n"
        "         [--agent-config agent.json] [--sandbox auto|hyperv|podman|none]\n"
        "         Run the coding-agent loop: the model calls exec/write_file/\n"
        "         read_file/finish as JSON tool-call blocks; each exec runs inside\n"
        "         an isolated sandbox (Hyper-V on Windows, Podman on Linux by\n"
        "         default -- see --sandbox and agent-config-example.json).\n\n"
        "Getting a model:\n"
        "  Download pretrained weights from https://huggingface.co and use the\n"
        "  converters in tools/converter/ (convert_safetensors.py, convert_gguf.py,\n"
        "  convert_llama.py, convert_gpt2.py) to produce a .nfw file.\n\n"
        "KV cache dtype (--kv-cache):\n"
        "  f32   full precision, default, largest memory footprint\n"
        "  f16   half precision, ~2x smaller, effectively lossless\n"
        "  q8_0  8-bit + per-token scale, ~4x smaller, small quantization noise\n"
        "  Lower precision lets a longer --ctx fit in the same RAM budget.\n\n"
        "Repetition control (no hard limits enforced — practical ranges):\n"
        "  --repeat-penalty    1.0 = off (default). Useful range 1.0-1.5;\n"
        "                      above ~2.0 tends to degrade coherence.\n"
        "                      Negative values invert the effect (encourage\n"
        "                      repeats instead of discouraging them) — avoid.\n"
        "  --freq-penalty      0.0 = off (default). Useful range 0.0-2.0.\n"
        "  --presence-penalty  0.0 = off (default). Useful range 0.0-2.0.\n"
        "                      Both scale a raw logit subtraction, so very\n"
        "                      large values (100+) can permanently exclude\n"
        "                      a token; negative values invert the effect.\n"
        "  --no-repeat-ngram   0 = off (default). Useful range 2-4. Values\n"
        "                      <=1 are always a no-op (by design).\n\n"
        "Examples:\n"
        "  vlrf_cli run    --model small.nfw --prompt \"int main\" --tokens 64\n"
        "  vlrf_cli run    --model small.nfw --prompt \"int main\" --kv-cache q8_0 --ctx 8192\n"
        "  vlrf_cli info   --model small.nfw\n";
}

// -----------------------------------------------------------------------------
// Command: init
// -----------------------------------------------------------------------------
static int cmd_init(Args const& a) {
    throw std::runtime_error(
        "vlrf_cli init is disabled: randomly-initialised weights are not a usable model.\n"
        "Download pretrained weights from https://huggingface.co and convert them with\n"
        "the scripts in tools/converter/ (e.g. convert_safetensors.py, convert_gguf.py,\n"
        "convert_llama.py, convert_gpt2.py) to produce a .nfw file for `vlrf_cli run`."
    );
}

// -----------------------------------------------------------------------------
// Command: run
// -----------------------------------------------------------------------------
static int cmd_run(Args const& a) {
    if (a.model_path.empty()) {
        std::cerr << "No --model specified\n";
        return 1;
    }
    if (a.prompt.empty()) {
        std::cerr << "No --prompt specified\n";
        return 1;
    }

    // ---- Load model -------------------------------------------------------
    std::cout << "Loading model from " << a.model_path << " ...\n";

    VlrfModel h = vlrf_create_model();
    if (!h) { std::cerr << "Failed to allocate model\n"; return 1; }

    // Select compute backend before loading weights / running inference.
    // The backend registry is a process-wide singleton (not per-model), so
    // it just needs to be set once, before the first op runs.
    if (!apply_backend(a.backend)) { vlrf_free_model(h); return 1; }

    // If --ctx specified, patch config.json or override after load
    // by setting max_seq_len before KV cache allocation.
    // We do this by writing a temporary override via vlrf_set_ctx (added below),
    // or simply by capping in vlrf_load_weights via params.
    // Simplest: override max_new_tokens and context via VlrfModelParams after load.
    VlrfStatus st = vlrf_load_weights(h, a.model_path.c_str(),
        a.tok_path.empty() ? nullptr : a.tok_path.c_str());
    if (st != VLRF_OK) {
        std::cerr << "Load failed: " << vlrf_status_str(st) << "\n";
        vlrf_free_model(h);
        return 1;
    }

    // Apply --ctx override (auto-cap already happened inside vlrf_load_weights)
    if (a.ctx_len > 0) {
        VlrfStatus cs = vlrf_set_ctx(h, a.ctx_len);
        if (cs == VLRF_OK)
            std::cout << "Context set to " << a.ctx_len << " tokens\n";
    }

    // Apply --kv-cache dtype (default f32 is a no-op vs. what vlrf_load_weights
    // already allocated; f16/q8_0 reallocate the cache at the smaller size).
    {
        VlrfKVCacheType kvt;
        if (!parse_kv_cache_type(a.kv_cache, kvt)) { vlrf_free_model(h); return 1; }
        VlrfStatus ks = vlrf_set_kv_cache_type(h, kvt);
        if (ks != VLRF_OK) {
            std::cerr << "Warning: vlrf_set_kv_cache_type(\"" << a.kv_cache
                << "\") failed: " << vlrf_status_str(ks) << "\n";
        }
        else {
            std::cout << "KV cache dtype   : " << a.kv_cache
                << "  (" << (vlrf_kv_cache_bytes(h) / 1024 / 1024) << " MB)\n";
        }
    }

    std::cout << vlrf_model_summary(h) << "\n";
    std::cout << "SIMD kernel: " << vlrf_simd_kernel_info() << "\n";
    std::cout << "Capabilities (GPU/CPU): " << vlrf_backend_capabilities() << "\n";
    vlrf_set_seed(h, a.seed);

    // Apply repetition-control overrides. Only touches generation/sampling
    // fields (see vlrf_set_generation_params doc) — start from whatever the
    // model's config.json set as defaults, then layer the CLI's explicit
    // choices on top so a config.json without these fields still works
    // (they default to "off" — see VlrfModelParams/GenerationConfig).
    {
        VlrfModelParams p{};
        vlrf_get_params(h, &p); // seed p with the model's current settings
        p.repetition_penalty = a.repeat_penalty;
        p.frequency_penalty = a.freq_penalty;
        p.presence_penalty = a.presence_penalty;
        p.no_repeat_ngram_size = a.no_repeat_ngram;
        if (a.greedy) p.use_sampling = 0;
        VlrfStatus gs = vlrf_set_generation_params(h, &p);
        if (gs != VLRF_OK) {
            std::cerr << "Warning: vlrf_set_generation_params failed: "
                << vlrf_status_str(gs) << "\n";
        }
    }

    if (!a.stop_strings.empty()) {
        std::vector<const char*> stop_ptrs;
        stop_ptrs.reserve(a.stop_strings.size());
        for (auto const& s : a.stop_strings) stop_ptrs.push_back(s.c_str());
        VlrfStatus ss = vlrf_set_stop_strings(h, stop_ptrs.data(), (int32_t)stop_ptrs.size());
        if (ss != VLRF_OK) {
            std::cerr << "Warning: vlrf_set_stop_strings failed: "
                << vlrf_status_str(ss) << "\n";
        }
        else {
            std::cout << "Stop strings     : " << a.stop_strings.size() << " configured\n";
        }
    }

    // ---- Generate ---------------------------------------------------------
    char out_buf[65536] = {};
    int max_tokens = a.max_tokens;

    // Streaming callback: prints each token's text as it's generated. Note
    // that with --stop configured, the matched stop string itself gets
    // printed here in real time (tokens are streamed as they're sampled,
    // before VLRF knows a stop condition will be hit) — it's only
    // trimmed from the RETURNED text (out_buf / stopped.text), which is
    // what any programmatic caller actually receives. This is a real,
    // unavoidable property of streaming: you can't un-print output that's
    // already been flushed to the terminal or sent to a caller.
    auto cb = [](void* /*ud*/, int32_t /*id*/, const char* tok_str) -> int {
        if (tok_str && tok_str[0]) std::cout << tok_str << std::flush;
        return 1;
        };

    std::cout << "\n--- Prompt ---\n" << a.prompt << "\n--- Output ---\n";

    auto t0 = std::chrono::steady_clock::now();
    st = vlrf_generate_text(h, a.prompt.c_str(), out_buf, sizeof(out_buf),
        max_tokens, cb, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "\n";

    if (st != VLRF_OK)
        std::cerr << "Generation error: " << vlrf_status_str(st) << "\n";

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    int ctx = vlrf_context_length(h);
    std::cout << "\n[" << ctx << " tokens in context, "
        << std::fixed << elapsed << "s]\n";

    vlrf_free_model(h);
    return st == VLRF_OK ? 0 : 1;
}

// -----------------------------------------------------------------------------
// Command: info
// -----------------------------------------------------------------------------
static int cmd_info(Args const& a) {
    if (a.model_path.empty()) {
        std::cerr << "No --model specified\n";
        return 1;
    }

    VlrfModel h = vlrf_create_model();
    if (!h) { std::cerr << "Failed to allocate model\n"; return 1; }
    if (!apply_backend(a.backend)) { vlrf_free_model(h); return 1; }

    VlrfStatus st = vlrf_load_weights(h, a.model_path.c_str(), nullptr);
    if (st != VLRF_OK) {
        std::cerr << "Load failed: " << vlrf_status_str(st) << "\n";
        vlrf_free_model(h);
        return 1;
    }

    // Apply --kv-cache so the printed estimate reflects the dtype the user
    // is actually asking about, not just whatever vlrf_load_weights defaulted to.
    {
        VlrfKVCacheType kvt;
        if (!parse_kv_cache_type(a.kv_cache, kvt)) { vlrf_free_model(h); return 1; }
        vlrf_set_kv_cache_type(h, kvt);  // best-effort; falls back silently on error
    }

    VlrfModelParams p{};
    vlrf_get_params(h, &p);

    std::cout << "=== VLRF Model Info ===\n";
    std::cout << vlrf_model_summary(h) << "\n\n";
    std::cout << "Architecture:\n";
    std::cout << "  Vocabulary size  : " << p.vocab_size << "\n";
    std::cout << "  Embedding dim    : " << p.d_model << "\n";
    std::cout << "  FFN hidden dim   : " << p.d_ff << "\n";
    std::cout << "  Attention heads  : " << p.n_heads << "\n";
    std::cout << "  KV heads         : " << p.n_kv_heads << "\n";
    std::cout << "  Layers           : " << p.n_layers << "\n";
    std::cout << "  Context window   : " << p.max_seq_len << " tokens\n";

    // Show KV cache memory at the currently-configured dtype (accurate for
    // f32/f16/q8_0, unlike a hardcoded sizeof(float) estimate).
    std::cout << "  KV cache dtype   : " << a.kv_cache << "\n";
    std::cout << "  KV cache size    : " << (vlrf_kv_cache_bytes(h) / 1024 / 1024)
        << " MB  (at max context)\n\n";
    std::cout << "Sampling defaults:\n";
    std::cout << "  Temperature      : " << p.temperature << "\n";
    std::cout << "  Top-p            : " << p.top_p << "\n";
    std::cout << "  Top-k            : " << p.top_k << "\n";
    std::cout << "  Max new tokens   : " << p.max_new_tokens << "\n";
    std::cout << "\nBackend: " << vlrf_backend_info() << "\n";
    std::cout << "SIMD kernel: " << vlrf_simd_kernel_info() << "\n";
    std::cout << "\nPer-op dispatch:\n  " << vlrf_backend_capabilities() << "\n";

    vlrf_free_model(h);
    return 0;
}

// -----------------------------------------------------------------------------
// Command: agent
// -----------------------------------------------------------------------------
// Runs the coding-agent loop: load model -> resolve+create a sandbox
// (HyperV on Windows, Podman on Linux, chosen by AgentConfig -- see
// include/agent/agent_config.hpp for how "auto" resolves) -> generate,
// parse tool calls out of the model's own output, execute them in the
// sandbox, feed results back, repeat until the model calls "finish" or
// max_agent_turns is hit.
static int cmd_agent(Args const& a) {
    if (a.model_path.empty()) {
        std::cerr << "No --model specified\n";
        return 1;
    }
    if (a.task.empty()) {
        std::cerr << "No --task specified (what should the agent do?)\n";
        return 1;
    }

    // ---- Load agent config --------------------------------------------
    vlrf::agent::AgentConfig cfg;
    try {
        if (!a.agent_config_path.empty()) {
            cfg = vlrf::agent::AgentConfig::load_from_file(a.agent_config_path);
            std::cout << "Loaded agent config from " << a.agent_config_path << "\n";
        } else {
            std::cout << "No --agent-config given; using built-in defaults "
                          "(sandbox_backend=auto, workspace=./vlrf_agent_workspace)\n";
        }
        if (!a.sandbox_backend_override.empty()) {
            if (a.sandbox_backend_override == "auto")        cfg.sandbox_backend = vlrf::agent::SandboxBackendKind::Auto;
            else if (a.sandbox_backend_override == "hyperv") cfg.sandbox_backend = vlrf::agent::SandboxBackendKind::HyperV;
            else if (a.sandbox_backend_override == "podman") cfg.sandbox_backend = vlrf::agent::SandboxBackendKind::Podman;
            else if (a.sandbox_backend_override == "none")   cfg.sandbox_backend = vlrf::agent::SandboxBackendKind::None;
            else {
                std::cerr << "Unknown --sandbox '" << a.sandbox_backend_override
                    << "' (expected auto|hyperv|podman|none)\n";
                return 1;
            }
        }
    } catch (std::exception const& e) {
        std::cerr << "Agent config error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Sandbox capabilities: " << vlrf::agent::agent_sandbox_capabilities() << "\n";
    std::cout << "Sandbox backend requested: " << vlrf::agent::to_string(cfg.sandbox_backend) << "\n";

    // ---- Create sandbox (before loading the model -- fail fast if the
    // chosen backend isn't available on this host/build, rather than
    // paying model-load time first) --------------------------------------
    std::unique_ptr<vlrf::agent::ISandbox> sandbox;
    try {
        sandbox = vlrf::agent::make_sandbox(cfg);
    } catch (std::exception const& e) {
        std::cerr << "Sandbox resolution failed: " << e.what() << "\n";
        return 1;
    }
    if (!sandbox) {
        std::cerr << "sandbox_backend=\"none\" -- agent loop requires a sandbox to execute "
                      "tool calls. Set sandbox_backend to \"hyperv\" or \"podman\" in the "
                      "agent config, or pass --sandbox.\n";
        return 1;
    }

    if (!sandbox->create(cfg.workspace_dir, cfg.limits)) {
        std::cerr << "Failed to create " << sandbox->backend_name()
            << " sandbox: " << sandbox->last_error() << "\n";
        return 1;
    }
    std::cout << "Sandbox active: " << sandbox->backend_name()
        << " (workspace: " << cfg.workspace_dir << ")\n";

    // ---- Load model ------------------------------------------------------
    std::cout << "Loading model from " << a.model_path << " ...\n";
    VlrfModel h = vlrf_create_model();
    if (!h) {
        std::cerr << "Failed to allocate model\n";
        sandbox->destroy();
        return 1;
    }
    if (!apply_backend(a.backend)) { vlrf_free_model(h); sandbox->destroy(); return 1; }

    VlrfStatus st = vlrf_load_weights(h, a.model_path.c_str(),
        a.tok_path.empty() ? nullptr : a.tok_path.c_str());
    if (st != VLRF_OK) {
        std::cerr << "Load failed: " << vlrf_status_str(st) << "\n";
        vlrf_free_model(h);
        sandbox->destroy();
        return 1;
    }
    std::cout << vlrf_model_summary(h) << "\n";

    // ---- Run the agent loop ------------------------------------------
    std::cout << "\n--- Agent task ---\n" << a.task << "\n--- Session ---\n";

    auto trace = [](std::string const& line) { std::cout << line << "\n"; };
    auto t0 = std::chrono::steady_clock::now();
    vlrf::agent::AgentSessionResult result =
        vlrf::agent::run_agent_session(h, *sandbox, cfg, a.task, trace);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n--- Result ---\n";
    if (result.completed) {
        std::cout << "Finished after " << result.turns.size() << " turn(s): "
            << result.final_summary << "\n";
    } else if (result.hit_turn_limit) {
        std::cout << "Stopped: hit max_agent_turns (" << cfg.max_agent_turns
            << ") without the model calling finish.\n";
    }
    std::cout << "[" << std::fixed << elapsed << "s elapsed]\n";

    // Workspace is left on disk for inspection even on failure/turn-limit;
    // only the sandbox process/container/VM itself is torn down here.
    sandbox->destroy();
    vlrf_free_model(h);
    return result.completed ? 0 : 1;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 0; }
    Args a = parse_args(argc, argv);

    try {
        if (a.command == "init") return cmd_init(a);
        if (a.command == "run")  return cmd_run(a);
        if (a.command == "info") return cmd_info(a);
        if (a.command == "agent") return cmd_agent(a);
    }
    catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "Unknown command '" << a.command << "'\n\n";
    print_usage();
    return 1;
}