#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
convert_llama.py — Convert LLaMA / Mistral / Phi / Granite safetensors weights to VLRF

Supports any model with the standard LLaMA-style HuggingFace layout:
  - LLaMA 2 / LLaMA 3  (Meta)
  - Mistral 7B / Mixtral
  - Phi-2 / Phi-3
  - CodeLlama
  - IBM Granite Code (attention_bias=true, mlp_bias=true)
  - Any GGUF-compatible model exported to safetensors

Usage:
  # From HuggingFace Hub:
  python convert_llama.py --model meta-llama/Llama-2-7b-hf --out ./llama2_7b

  # From local directory (already downloaded):
  python convert_llama.py --model ./my_llama_dir --out ./llama_vlrf

  # IBM Granite (local files):
  python convert_llama.py --model C:/Temp/llm_data --out ./granite_weights --dtype float16

Output:
  <out>/model.nfw        weights in VLRF binary format
  <out>/tokenizer.nft    BPE tokenizer
  <out>/config.json      human-readable config
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError:
    sys.exit("transformers not installed: pip install transformers")

sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import write_nfw, write_nft, verify_nfw


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

def detect_variant(hf_cfg) -> str:
    arch = getattr(hf_cfg, "model_type", "llama")
    if "mistral" in arch:  return "mistral"
    if "phi"     in arch:  return "phi"
    if "falcon"  in arch:  return "falcon"
    return "llama"


def get_cfg_attr(hf_cfg, *candidates, default=None):
    """Try multiple attribute names, return first found."""
    for name in candidates:
        val = getattr(hf_cfg, name, None)
        if val is not None:
            return val
    return default


# ---------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------

def convert_llama_weights(state_dict: dict, hf_cfg) -> dict:
    """
    Remap LLaMA-style HuggingFace state_dict to VLRF WeightMap keys.

    Handles:
      - RMSNorm (no bias) — zero bias written for VLRF compatibility
      - SwiGLU FFN: gate_proj + up_proj + down_proj
      - GQA: n_kv_heads < n_heads
      - Optional attention/FFN biases (Granite Code sets attention_bias=true)
      - No positional embedding table (RoPE approximated as sinusoidal)
    """
    vlrf = {}
    n_layers = hf_cfg.num_hidden_layers
    d = hf_cfg.hidden_size
    kv_heads = get_cfg_attr(hf_cfg, "num_key_value_heads",
                             "num_kv_heads", default=hf_cfg.num_attention_heads)
    head_dim = d // hf_cfg.num_attention_heads
    d_kv = kv_heads * head_dim
    d_ff = get_cfg_attr(hf_cfg, "intermediate_size", "ffn_dim", default=4 * d)

    variant = detect_variant(hf_cfg)
    has_attn_bias = getattr(hf_cfg, "attention_bias", False)
    has_mlp_bias  = getattr(hf_cfg, "mlp_bias", False)

    print(f"  Detected variant  : {variant}")
    print(f"  attention_bias    : {has_attn_bias}")
    print(f"  mlp_bias          : {has_mlp_bias}")

    def get(key_patterns: list, src: dict):
        """Return first matching key as float32 numpy array."""
        for k in key_patterns:
            if k in src:
                return src[k].float().cpu().numpy()
        raise KeyError(f"None of {key_patterns} found in state dict")

    def get_bias(key: str, size: int) -> np.ndarray:
        """Read a bias tensor if it exists, otherwise return zeros."""
        if key in state_dict:
            return state_dict[key].float().cpu().numpy()
        return np.zeros(size, dtype=np.float32)

    # ---- Token embedding
    vlrf["embed.weight"] = get([
        "model.embed_tokens.weight",
        "transformer.word_embeddings.weight",
        "gpt_neox.embed_in.weight",
    ], state_dict)

    # ---- Transformer layers
    for l in range(n_layers):
        src_pre = f"model.layers.{l}"
        dst_pre = f"layers.{l}"

        # Layer norm 1 (pre-attention) — RMSNorm has no bias
        vlrf[f"{dst_pre}.ln1.weight"] = get([
            f"{src_pre}.input_layernorm.weight",
            f"{src_pre}.ln_1.weight",
            f"{src_pre}.norm1.weight",
        ], state_dict)
        vlrf[f"{dst_pre}.ln1.bias"] = np.zeros(d, dtype=np.float32)

        # Layer norm 2 (pre-FFN)
        vlrf[f"{dst_pre}.ln2.weight"] = get([
            f"{src_pre}.post_attention_layernorm.weight",
            f"{src_pre}.ln_2.weight",
            f"{src_pre}.norm2.weight",
        ], state_dict)
        vlrf[f"{dst_pre}.ln2.bias"] = np.zeros(d, dtype=np.float32)

        # Attention weights
        vlrf[f"{dst_pre}.attn.q_proj.weight"] = get([
            f"{src_pre}.self_attn.q_proj.weight",
            f"{src_pre}.attention.q_proj.weight",
        ], state_dict)

        vlrf[f"{dst_pre}.attn.k_proj.weight"] = get([
            f"{src_pre}.self_attn.k_proj.weight",
            f"{src_pre}.attention.k_proj.weight",
        ], state_dict)

        vlrf[f"{dst_pre}.attn.v_proj.weight"] = get([
            f"{src_pre}.self_attn.v_proj.weight",
            f"{src_pre}.attention.v_proj.weight",
        ], state_dict)

        vlrf[f"{dst_pre}.attn.o_proj.weight"] = get([
            f"{src_pre}.self_attn.o_proj.weight",
            f"{src_pre}.attention.o_proj.weight",
            f"{src_pre}.self_attn.dense.weight",
        ], state_dict)

        # Attention biases — real values for Granite, zeros for standard LLaMA
        vlrf[f"{dst_pre}.attn.q_proj.bias"] = get_bias(f"{src_pre}.self_attn.q_proj.bias", d)
        vlrf[f"{dst_pre}.attn.k_proj.bias"] = get_bias(f"{src_pre}.self_attn.k_proj.bias", d_kv)
        vlrf[f"{dst_pre}.attn.v_proj.bias"] = get_bias(f"{src_pre}.self_attn.v_proj.bias", d_kv)
        vlrf[f"{dst_pre}.attn.o_proj.bias"] = get_bias(f"{src_pre}.self_attn.o_proj.bias", d)

        # FFN weights — SwiGLU: gate_proj + up_proj + down_proj
        # VLRF layout: fc1=up_proj, gate=gate_proj, fc2=down_proj
        vlrf[f"{dst_pre}.ffn.fc1.weight"] = get([
            f"{src_pre}.mlp.up_proj.weight",
            f"{src_pre}.mlp.fc1.weight",
            f"{src_pre}.mlp.dense_h_to_4h.weight",
        ], state_dict)

        vlrf[f"{dst_pre}.ffn.gate.weight"] = get([
            f"{src_pre}.mlp.gate_proj.weight",
            f"{src_pre}.mlp.w1.weight",
        ], state_dict)

        vlrf[f"{dst_pre}.ffn.fc2.weight"] = get([
            f"{src_pre}.mlp.down_proj.weight",
            f"{src_pre}.mlp.fc2.weight",
            f"{src_pre}.mlp.dense_4h_to_h.weight",
        ], state_dict)

        # FFN biases — real values for Granite, zeros for standard LLaMA
        vlrf[f"{dst_pre}.ffn.fc1.bias"] = get_bias(f"{src_pre}.mlp.up_proj.bias",   d_ff)
        vlrf[f"{dst_pre}.ffn.fc2.bias"] = get_bias(f"{src_pre}.mlp.down_proj.bias", d)

    # ---- Final norm
    vlrf["ln_final.weight"] = get([
        "model.norm.weight",
        "transformer.norm_f.weight",
        "gpt_neox.final_layer_norm.weight",
    ], state_dict)
    vlrf["ln_final.bias"] = np.zeros(d, dtype=np.float32)

    # ---- LM head
    try:
        vlrf["lm_head.weight"] = get(["lm_head.weight"], state_dict)
        print("  LM head: separate (not tied)")
    except KeyError:
        print("  LM head: tied to embedding")

    print(f"  Mapped {len(vlrf)} tensors")
    return vlrf


def build_vlrf_config(hf_cfg) -> dict:
    d        = hf_cfg.hidden_size
    n_heads  = hf_cfg.num_attention_heads
    kv_heads = get_cfg_attr(hf_cfg, "num_key_value_heads",
                             "num_kv_heads", default=n_heads)
    d_ff     = get_cfg_attr(hf_cfg, "intermediate_size", "ffn_dim", default=4 * d)
    ctx      = get_cfg_attr(hf_cfg, "max_position_embeddings",
                             "max_seq_len", default=4096)
    vocab    = hf_cfg.vocab_size
    n_layers = hf_cfg.num_hidden_layers
    bos      = get_cfg_attr(hf_cfg, "bos_token_id", default=1) or 1
    eos      = get_cfg_attr(hf_cfg, "eos_token_id", default=2) or 2
    tied     = getattr(hf_cfg, "tie_word_embeddings", False)

    return {
        "model_name":     getattr(hf_cfg, "model_type", "llama"),
        "model_version":  "1.0.0",
        "vocab_size":     vocab,
        "d_model":        d,
        "d_ff":           d_ff,
        "n_heads":        n_heads,
        "n_kv_heads":     kv_heads,
        "n_layers":       n_layers,
        "max_seq_len":    ctx,
        "bos_token_id":   bos,
        "eos_token_id":   eos,
        "pad_token_id":   getattr(hf_cfg, "pad_token_id", 0) or 0,
        "layer_norm_eps": getattr(hf_cfg, "rms_norm_eps", 1e-5),
        "pre_norm":       True,
        "use_rms_norm":   True,
        "tie_embeddings": tied,
        "pos_embedding":  "sinusoidal",
        "activation":     "silu",
        "temperature":    0.8,
        "top_p":          0.9,
        "top_k":          50,
        "max_new_tokens": 256,
        "attention_bias": getattr(hf_cfg, "attention_bias", False),
        "mlp_bias":       getattr(hf_cfg, "mlp_bias", False),
        "NOTE": "RoPE is approximated with sinusoidal PE.",
    }


# ---------------------------------------------------------------------------
# Tokenizer conversion
# ---------------------------------------------------------------------------

def convert_tokenizer_llama(hf_tokenizer, out_dir: Path) -> None:
    """Convert HuggingFace tokenizer to .nft format."""
    vocab_size = hf_tokenizer.vocab_size
    id2tok = ["<unk>"] * vocab_size

    try:
        vocab_dict = hf_tokenizer.get_vocab()
        for tok, idx in vocab_dict.items():
            if 0 <= idx < vocab_size:
                id2tok[idx] = tok
    except Exception as e:
        print(f"  WARNING: could not extract vocab: {e}")

    # Extract BPE merge rules if available
    merges = []
    try:
        if hasattr(hf_tokenizer, "bpe_ranks"):
            vocab_index = {t: i for i, t in enumerate(id2tok)}
            for (a, b) in hf_tokenizer.bpe_ranks.keys():
                ia = vocab_index.get(a, -1)
                ib = vocab_index.get(b, -1)
                if ia >= 0 and ib >= 0:
                    merges.append((ia, ib))
    except Exception:
        pass

    bos = getattr(hf_tokenizer, "bos_token_id", 1) or 1
    eos = getattr(hf_tokenizer, "eos_token_id", 2) or 2

    nft_path = out_dir / "tokenizer.nft"
    write_nft(str(nft_path), id2tok, merges, bos_id=bos, eos_id=eos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaMA/Mistral/Granite HuggingFace weights to VLRF .nfw")
    parser.add_argument("--model",  required=True,
                        help="HF model name or local directory path")
    parser.add_argument("--out",    required=True,
                        help="Output directory")
    parser.add_argument("--dtype",  default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Load dtype (float16 halves RAM usage)")
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer conversion")
    parser.add_argument("--verify", action="store_true",
                        help="Round-trip verify weights after writing")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to from_pretrained")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"\n{'='*60}")
    print(f" LLaMA/Mistral/Granite → VLRF Converter")
    print(f"{'='*60}")
    print(f" Source  : {args.model}")
    print(f" Output  : {out_dir}")
    print(f" dtype   : {args.dtype}")
    print()

    # ---- Load model
    print("Loading model (this may take a while for large models)...")
    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    model  = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    hf_cfg = model.config

    d        = hf_cfg.hidden_size
    n_heads  = hf_cfg.num_attention_heads
    kv_heads = get_cfg_attr(hf_cfg, "num_key_value_heads", default=n_heads)
    n_layers = hf_cfg.num_hidden_layers
    ctx      = get_cfg_attr(hf_cfg, "max_position_embeddings", default=4096)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_layers} layers, d={d}, h={n_heads}, kv_h={kv_heads}, "
          f"ctx={ctx}, vocab={hf_cfg.vocab_size}")
    print(f"  Total params : {total_params/1e9:.2f}B")
    print(f"  attention_bias: {getattr(hf_cfg, 'attention_bias', False)}")
    print(f"  mlp_bias      : {getattr(hf_cfg, 'mlp_bias', False)}")

    # ---- Convert weights
    print("\nConverting weights...")
    state = {k: v.float().cpu() for k, v in model.state_dict().items()}
    del model
    import gc; gc.collect()

    vlrf_tensors = convert_llama_weights(state, hf_cfg)
    del state
    gc.collect()

    # ---- Write .nfw
    weights_path = out_dir / "model.nfw"
    write_nfw(str(weights_path), vlrf_tensors)

    if args.verify:
        print("\nVerifying round-trip...")
        verify_nfw(vlrf_tensors, str(weights_path))

    # ---- Write config
    vlrf_cfg   = build_vlrf_config(hf_cfg)
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vlrf_cfg, f, indent=2)
    print(f"  Config → {cfg_path}")

    # ---- Convert tokenizer
    if not args.no_tok:
        print("\nConverting tokenizer...")
        try:
            tok = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code)
            convert_tokenizer_llama(tok, out_dir)
        except Exception as e:
            print(f"  WARNING: tokenizer failed: {e}")
            print("  You can still run inference without a tokenizer.")

    # ---- Summary
    print(f"\n{'='*60}")
    print(" Conversion complete!")
    print(f"{'='*60}")
    print(f" Weights   : {weights_path}")
    print(f" Tokenizer : {out_dir / 'tokenizer.nft'}")
    print(f" Config    : {cfg_path}")
    print()
    print("To use with vlrf_cli:")
    print(f"  vlrf_cli run --model {weights_path} \\")
    print(f"             --tok   {out_dir / 'tokenizer.nft'} \\")
    print(f'             --prompt "Your prompt here"')
    print()


if __name__ == "__main__":
    main()