#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
convert.py — VLRF unified weight converter (auto-detects format)

Detects the input format automatically and dispatches to the right converter.

Usage:
  python convert.py --model <path_or_name> --out <dir> [options]

Format auto-detection:
  *.gguf                     → convert_gguf.py
  *.safetensors or directory → convert_safetensors.py
  HF model name (no path)    → convert_gpt2.py or convert_llama.py

Options:
  --model   PATH or HF model name  (required)
  --out     DIR                    output directory (required)
  --arch    auto|llama|gpt2|...    override architecture detection
  --dtype   float32|float16        load dtype (default: float32)
  --no-tok  skip tokenizer conversion
  --verify  round-trip verify after writing
  --trust-remote-code  pass to HF from_pretrained

Examples:
  # From HuggingFace Hub (GPT-2)
  python convert.py --model gpt2 --out ./out/gpt2

  # From HuggingFace Hub (LLaMA-style)
  python convert.py --model meta-llama/Llama-2-7b-hf --out ./out/llama2

  # From local GGUF
  python convert.py --model ./models/mistral-7b.Q4_K_M.gguf --out ./out/mistral

  # From local safetensors directory
  python convert.py --model ./models/codellama-7b --out ./out/codellama

  # From single safetensors file
  python convert.py --model ./model.safetensors --arch llama --out ./out/model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def detect_format(model_path: str) -> str:
    p = Path(model_path)

    # Single GGUF file
    if p.suffix == ".gguf" and p.is_file():
        return "gguf"

    # Single or multi safetensors file
    if p.suffix == ".safetensors" and p.is_file():
        return "safetensors"

    # Local directory
    if p.is_dir():
        if list(p.glob("*.gguf")):
            return "gguf"
        if list(p.glob("*.safetensors")):
            return "safetensors"
        if list(p.glob("pytorch_model*.bin")) or list(p.glob("model*.bin")):
            return "pytorch"
        # HF checkpoint with config
        if (p / "config.json").exists():
            return "huggingface"

    # Glob pattern
    if "*" in model_path:
        if ".gguf" in model_path:
            return "gguf"
        if ".safetensors" in model_path:
            return "safetensors"

    # Assume HuggingFace model name (no local path exists)
    return "huggingface"


def detect_hf_arch(model_name: str) -> str:
    """Guess architecture from HF model name string."""
    name = model_name.lower()
    if any(x in name for x in ("gpt2", "gpt-2", "gpt_neo", "distilgpt")):
        return "gpt2"
    if any(x in name for x in ("llama", "codellama", "vicuna", "alpaca",
                                 "mistral", "mixtral", "qwen", "gemma",
                                 "deepseek", "solar", "openchat", "zephyr")):
        return "llama"
    # Default to llama-style (most modern models)
    return "llama"


def main():
    parser = argparse.ArgumentParser(
        description="VLRF unified weight converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("--model",  required=True,
                        help="HF model name, local path, .gguf file, or glob")
    parser.add_argument("--out",    required=True,
                        help="Output directory")
    parser.add_argument("--arch",   default="auto",
                        help="Architecture hint (default: auto)")
    parser.add_argument("--dtype",  default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Load dtype (default: float32)")
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer conversion")
    parser.add_argument("--verify", action="store_true",
                        help="Round-trip verify weights after writing")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code (HuggingFace)")
    args = parser.parse_args()

    fmt = detect_format(args.model)
    print(f"Detected format: {fmt}")

    # ---- Dispatch -----------------------------------------------------------

    if fmt == "gguf":
        from convert_gguf import main as gguf_main
        # Rebuild sys.argv for the sub-converter
        sys.argv = ["convert_gguf.py",
                    "--model", args.model,
                    "--out",   args.out]
        if args.no_tok:  sys.argv.append("--no-tok")
        if args.verify:  sys.argv.append("--verify")
        gguf_main()

    elif fmt == "safetensors":
        from convert_safetensors import main as st_main
        sys.argv = ["convert_safetensors.py",
                    "--model", args.model,
                    "--arch",  args.arch,
                    "--out",   args.out]
        if args.no_tok:  sys.argv.append("--no-tok")
        if args.verify:  sys.argv.append("--verify")
        st_main()

    elif fmt in ("huggingface", "pytorch"):
        # Decide GPT-2 vs LLaMA-style
        arch = args.arch if args.arch != "auto" else detect_hf_arch(args.model)

        if arch == "gpt2":
            from convert_gpt2 import main as gpt2_main
            sys.argv = ["convert_gpt2.py",
                        "--model", args.model,
                        "--out",   args.out,
                        "--dtype", args.dtype]
            if args.no_tok:  sys.argv.append("--no-tok")
            if args.verify:  sys.argv.append("--verify")
            gpt2_main()
        else:
            from convert_llama import main as llama_main
            sys.argv = ["convert_llama.py",
                        "--model", args.model,
                        "--out",   args.out,
                        "--dtype", args.dtype]
            if args.no_tok:  sys.argv.append("--no-tok")
            if args.verify:  sys.argv.append("--verify")
            if args.trust_remote_code: sys.argv.append("--trust-remote-code")
            llama_main()

    else:
        sys.exit(f"Unknown format '{fmt}' for model '{args.model}'")


if __name__ == "__main__":
    main()
