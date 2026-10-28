#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
convert_gpt2.py — Convert GPT-2 HuggingFace weights to VLRF (.nfw + .nft)

Supports:
  gpt2          (124M)
  gpt2-medium   (355M)
  gpt2-large    (774M)
  gpt2-xl       (1.5B)
  Any local HF checkpoint with GPT-2 architecture

Usage:
  # From HuggingFace Hub (downloads automatically):
  python convert_gpt2.py --model gpt2 --out ./gpt2

  # From a local directory:
  python convert_gpt2.py --model ./my_gpt2_checkpoint --out ./gpt2_vlrf

Output:
  ./gpt2/model.nfw       ← weights
  ./gpt2/tokenizer.nft   ← BPE tokenizer
  ./gpt2/config.json     ← human-readable config summary

Notes on weight layout differences:
  GPT-2 uses Conv1D (transposed Linear) so c_attn/c_proj/c_fc weights are
  stored as [in, out] instead of [out, in]. We transpose them here.
  GPT-2 fuses Q, K, V into a single c_attn weight [d, 3*d] — we split it.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("torch not installed: pip install torch --index-url https://download.pytorch.org/whl/cpu")

try:
    from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
except ImportError:
    sys.exit("transformers not installed: pip install transformers")

# Add parent dir so we can import nfw_writer
sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import write_nfw, write_nft, verify_nfw


# ---------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------

def convert_gpt2_weights(state_dict: dict, cfg: GPT2Config) -> dict:
    """
    Remap GPT-2 HuggingFace state_dict → VLRF WeightMap keys.

    GPT-2 quirks:
      • Conv1D weights are transposed: shape [in_features, out_features]
        VLRF expects Linear: [out_features, in_features]
      • c_attn fuses QKV: weight [d_model, 3*d_model], bias [3*d_model]
        We split into q/k/v each [d_model, d_model]
      • No separate lm_head weight (tied to wte)
    """
    vlrf = {}
    d = cfg.n_embd
    n_heads = cfg.n_head
    n_layers = cfg.n_layer

    def t(x):
        """Transpose a 2-D Conv1D weight to Linear layout."""
        return x.T if x.ndim == 2 else x

    # ---- Detect key naming convention -------------------------------------
    # transformers' GPT2Model.from_pretrained() returns BARE keys
    # (e.g. "wte.weight", "h.0.ln_1.weight") in current versions, but older
    # versions / GPT2LMHeadModel wrap everything under a "transformer."
    # prefix (e.g. "transformer.wte.weight"). Detect which is present rather
    # than hardcoding one, so this keeps working across transformers versions.
    if "transformer.wte.weight" in state_dict:
        prefix = "transformer."
    elif "wte.weight" in state_dict:
        prefix = ""
    else:
        raise KeyError(
            "convert_gpt2_weights: could not find 'wte.weight' or "
            "'transformer.wte.weight' in the loaded state_dict. "
            f"Available top-level keys (first 10): {list(state_dict.keys())[:10]}\n"
            "This usually means the model architecture isn't GPT-2-family, "
            "or transformers changed its key naming again — check the keys "
            "above and update convert_gpt2.py's prefix detection accordingly."
        )

    # ---- Embeddings
    vlrf["embed.weight"]    = state_dict[f"{prefix}wte.weight"].numpy()
    vlrf["pos_embed.weight"]= state_dict[f"{prefix}wpe.weight"].numpy()

    # ---- Layers
    for l in range(n_layers):
        src = f"{prefix}h.{l}"
        dst = f"layers.{l}"

        # Layer norms
        vlrf[f"{dst}.ln1.weight"] = state_dict[f"{src}.ln_1.weight"].numpy()
        vlrf[f"{dst}.ln1.bias"]   = state_dict[f"{src}.ln_1.bias"].numpy()
        vlrf[f"{dst}.ln2.weight"] = state_dict[f"{src}.ln_2.weight"].numpy()
        vlrf[f"{dst}.ln2.bias"]   = state_dict[f"{src}.ln_2.bias"].numpy()

        # Fused QKV → split
        # c_attn.weight shape in GPT-2: [d, 3*d]  (Conv1D → transpose to [3*d, d])
        c_attn_w = t(state_dict[f"{src}.attn.c_attn.weight"].numpy())  # [3d, d]
        c_attn_b = state_dict[f"{src}.attn.c_attn.bias"].numpy()       # [3d]

        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=0)   # each [d, d]
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=0)   # each [d]

        vlrf[f"{dst}.attn.q_proj.weight"] = q_w   # [d, d]
        vlrf[f"{dst}.attn.k_proj.weight"] = k_w   # [d, d]
        vlrf[f"{dst}.attn.v_proj.weight"] = v_w   # [d, d]
        vlrf[f"{dst}.attn.q_proj.bias"]   = q_b
        vlrf[f"{dst}.attn.k_proj.bias"]   = k_b
        vlrf[f"{dst}.attn.v_proj.bias"]   = v_b

        # Output projection: c_proj [d, d] (Conv1D) → transpose → [d, d]
        vlrf[f"{dst}.attn.o_proj.weight"] = t(state_dict[f"{src}.attn.c_proj.weight"].numpy())
        vlrf[f"{dst}.attn.o_proj.bias"]   = state_dict[f"{src}.attn.c_proj.bias"].numpy()

        # FFN: c_fc [d, 4d] → [4d, d],  c_proj [4d, d] → [d, 4d]
        vlrf[f"{dst}.ffn.fc1.weight"] = t(state_dict[f"{src}.mlp.c_fc.weight"].numpy())
        vlrf[f"{dst}.ffn.fc1.bias"]   = state_dict[f"{src}.mlp.c_fc.bias"].numpy()
        vlrf[f"{dst}.ffn.fc2.weight"] = t(state_dict[f"{src}.mlp.c_proj.weight"].numpy())
        vlrf[f"{dst}.ffn.fc2.bias"]   = state_dict[f"{src}.mlp.c_proj.bias"].numpy()

    # ---- Final layer norm
    vlrf["ln_final.weight"] = state_dict[f"{prefix}ln_f.weight"].numpy()
    vlrf["ln_final.bias"]   = state_dict[f"{prefix}ln_f.bias"].numpy()

    # ---- LM head — GPT-2 ties lm_head to wte, so we don't store it separately
    # (VLRF will use embed.weight via tie_embeddings=true)

    print(f"  Mapped {len(vlrf)} tensors from {len(state_dict)} source keys")
    return vlrf


def build_vlrf_config(cfg: GPT2Config) -> dict:
    """Produce the VLRF config JSON for this GPT-2 model."""
    return {
        "model_name":       "gpt2",
        "model_version":    "1.0.0",
        "vocab_size":       cfg.vocab_size,        # 50257
        "d_model":          cfg.n_embd,
        "d_ff":             cfg.n_inner or 4 * cfg.n_embd,
        "n_heads":          cfg.n_head,
        "n_kv_heads":       cfg.n_head,            # GPT-2 = MHA (no GQA)
        "n_layers":         cfg.n_layer,
        "max_seq_len":      cfg.n_positions,       # 1024
        "bos_token_id":     cfg.bos_token_id or 50256,
        "eos_token_id":     cfg.eos_token_id or 50256,
        "pad_token_id":     0,
        "layer_norm_eps":   cfg.layer_norm_epsilon,
        "pre_norm":         True,
        "use_rms_norm":     False,
        "tie_embeddings":   True,
        "pos_embedding":    "learned",
        "activation":       "gelu",
        "temperature":      0.8,
        "top_p":            0.9,
        "top_k":            50,
        "max_new_tokens":   256,
    }


def convert_tokenizer(hf_tokenizer, out_dir: Path) -> None:
    """
    Convert GPT-2 BPE tokenizer to .nft format.

    GPT-2 tokenizer stores merges as strings like "Ġ h" → we convert to
    integer pair (id_a, id_b) using the vocab index.
    """
    vocab_size = hf_tokenizer.vocab_size
    # Build ordered vocab list  (id → string)
    id2tok = [""] * vocab_size
    for tok, idx in hf_tokenizer.get_vocab().items():
        if idx < vocab_size:
            id2tok[idx] = tok

    # GPT-2 uses byte-level BPE; decode the Ġ/Ċ etc. representations back to
    # display strings. We keep them as-is for the .nft since our decoder just
    # concatenates token strings.

    # Add VLRF special tokens at the end (don't shift existing ids)
    # GPT-2 has a single <|endoftext|> = 50256; we map:
    #   pad = eos = 50256 for GPT-2 (it has no explicit pad)
    special = {
        "<pad>": hf_tokenizer.eos_token_id,
        "<bos>": hf_tokenizer.bos_token_id,
        "<eos>": hf_tokenizer.eos_token_id,
        "<unk>": hf_tokenizer.eos_token_id,  # GPT-2 has no unk
    }

    # Convert merge rules: pairs of (tok_a_str, tok_b_str) -> (id_a, id_b)
    #
    # transformers removed the convenient `tokenizer.bpe_ranks` attribute at
    # some point (confirmed absent in transformers 5.x) — GPT2Tokenizer no
    # longer exposes merge rules as a simple Python dict. The merges DO
    # still exist, but only inside the Rust `tokenizers` backend that
    # GPT2Tokenizer wraps internally. We extract them via that backend's
    # stable JSON serialisation (`backend_tokenizer.to_str()`), which is
    # public, documented API of the `tokenizers` library and unlikely to
    # be removed the way the convenience attribute was.
    vocab_index = {tok: idx for idx, tok in enumerate(id2tok)}
    merges = []

    merge_pairs = None
    if hasattr(hf_tokenizer, "bpe_ranks"):
        # Older transformers versions: convenient dict {(a,b): rank}
        merge_pairs = list(hf_tokenizer.bpe_ranks.keys())
    elif hasattr(hf_tokenizer, "backend_tokenizer"):
        # Current transformers versions: extract from the Rust tokenizer's
        # JSON state. model.merges is a list of [tok_a_str, tok_b_str] pairs,
        # in priority order (earlier = higher priority), which matches what
        # VLRF's tokenizer.hpp expects.
        try:
            state = json.loads(hf_tokenizer.backend_tokenizer.to_str())
            raw_merges = state.get("model", {}).get("merges", [])
            # Each entry may be a 2-element list [a, b] or (in some
            # tokenizers versions) a single "a b" string — handle both.
            merge_pairs = []
            for m in raw_merges:
                if isinstance(m, (list, tuple)) and len(m) == 2:
                    merge_pairs.append((m[0], m[1]))
                elif isinstance(m, str) and " " in m:
                    a, b = m.split(" ", 1)
                    merge_pairs.append((a, b))
        except Exception as e:
            print(f"  WARNING: could not parse backend_tokenizer merges: {e}")
            merge_pairs = None

    if merge_pairs:
        for a, b in merge_pairs:
            ia = vocab_index.get(a, -1)
            ib = vocab_index.get(b, -1)
            if ia >= 0 and ib >= 0:
                merges.append((ia, ib))
        print(f"  Extracted {len(merges)} BPE merge rules")
    else:
        print("  WARNING: no merge rules found via bpe_ranks or "
              "backend_tokenizer — tokenizer will fall back to byte-level "
              "encoding without BPE merging (works, but produces more "
              "tokens per text than the original tokenizer would).")

    nft_path = out_dir / "tokenizer.nft"
    write_nft(str(nft_path), id2tok, merges,
              pad_id=special["<pad>"],
              bos_id=special["<bos>"],
              eos_id=special["<eos>"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-2 HuggingFace weights to VLRF .nfw format")
    parser.add_argument("--model",  default="gpt2",
                        help="HF model name or local path (default: gpt2)")
    parser.add_argument("--out",    default="./converted/gpt2",
                        help="Output directory (default: ./converted/gpt2)")
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer conversion")
    parser.add_argument("--verify", action="store_true",
                        help="Round-trip verify weights after writing")
    parser.add_argument("--dtype",  default="float32",
                        choices=["float32", "float16"],
                        help="Source dtype to load (default: float32)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" GPT-2 → VLRF Converter")
    print(f"{'='*60}")
    print(f" Source : {args.model}")
    print(f" Output : {out_dir}")
    print()

    # ---- Load HF model
    print("Loading HuggingFace model...")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model  = GPT2Model.from_pretrained(args.model, torch_dtype=torch_dtype)
    hf_cfg = model.config
    state  = {k: v.float().cpu() for k, v in model.state_dict().items()}

    # Check for lm_head (sometimes present as separate weight)
    try:
        from transformers import GPT2LMHeadModel
        lm = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=torch_dtype)
        if "lm_head.weight" in lm.state_dict():
            state["lm_head.weight"] = lm.state_dict()["lm_head.weight"].float().cpu()
    except Exception:
        pass

    print(f"  {hf_cfg.n_layer} layers, d={hf_cfg.n_embd}, "
          f"h={hf_cfg.n_head}, ctx={hf_cfg.n_positions}, "
          f"vocab={hf_cfg.vocab_size}")

    # ---- Convert weights
    print("\nConverting weights...")
    vlrf_tensors = convert_gpt2_weights(state, hf_cfg)

    # ---- Write .nfw
    weights_path = out_dir / "model.nfw"
    write_nfw(str(weights_path), vlrf_tensors)

    if args.verify:
        print("\nVerifying round-trip...")
        verify_nfw(vlrf_tensors, str(weights_path))

    # ---- Write config JSON
    vlrf_cfg = build_vlrf_config(hf_cfg)
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vlrf_cfg, f, indent=2)
    print(f"  Config → {cfg_path}")

    # ---- Convert tokenizer
    if not args.no_tok:
        print("\nConverting tokenizer...")
        try:
            tok = GPT2Tokenizer.from_pretrained(args.model)
            convert_tokenizer(tok, out_dir)
        except Exception as e:
            print(f"  WARNING: tokenizer conversion failed: {e}")
            print("  You can still use the model without a tokenizer.")

    # ---- Summary
    print(f"\n{'='*60}")
    print(f" Conversion complete!")
    print(f"{'='*60}")
    print(f" Weights   : {weights_path}")
    print(f" Tokenizer : {out_dir / 'tokenizer.nft'}")
    print(f" Config    : {cfg_path}")
    print()
    print("To use with vlrf_cli:")
    print(f"  vlrf_cli run --model {weights_path} \\")
    print(f"             --tok {out_dir / 'tokenizer.nft'} \\")
    print(f'             --prompt "Your prompt here"')
    print()


if __name__ == "__main__":
    main()