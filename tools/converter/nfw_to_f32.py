#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
nfw_to_f32.py — Convert float16 .nfw → float32 .nfw for the C++ VLRF model

The C++ WeightMap reader (weights.hpp) expects float32 format (v1).
Large models are first converted to float16 to save RAM during conversion,
then this tool converts them to float32 for actual inference.

Only one tensor is held in RAM at a time — safe for any model size.

Usage:
  python nfw_to_f32.py --input  model_f16.nfw
                       --output model.nfw

  # Or in-place (replaces the original):
  python nfw_to_f32.py --input model_f16.nfw --inplace
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import convert_nfw_f16_to_f32


def main():
    parser = argparse.ArgumentParser(
        description="Convert float16 .nfw to float32 .nfw for C++ VLRF")
    parser.add_argument("--input",   required=True,
                        help="Input float16 .nfw file")
    parser.add_argument("--output",  default="",
                        help="Output float32 .nfw file (default: input_f32.nfw)")
    parser.add_argument("--inplace", action="store_true",
                        help="Replace the input file with the float32 version")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Error: file not found: {src}")

    if args.inplace:
        tmp = src.with_suffix(".tmp.nfw")
        convert_nfw_f16_to_f32(str(src), str(tmp))
        src.unlink()
        tmp.rename(src)
        print(f"Replaced {src} with float32 version")
    else:
        dst = Path(args.output) if args.output else src.with_name(src.stem + "_f32.nfw")
        convert_nfw_f16_to_f32(str(src), str(dst))

    print("Done. You can now load this file with vlrf_cli or vlrf_load_weights().")


if __name__ == "__main__":
    main()