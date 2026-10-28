#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
convert_safetensors.py — Convert .safetensors files to VLRF (.nfw + .nft)

Fixed issues vs previous version:
  - No duplicate tensors: speculative zero-bias writes removed from remap_llama
  - Missing zeros added in a single post-pass after all shards processed
  - StreamingNFWWriter deduplication guard
  - n_tensors header written correctly (exact count, no dry-run mismatch)

Supports:
  - Single .safetensors file
  - Directory containing shards (model-00001-of-00002.safetensors etc.)
  - GPT-2 and LLaMA / Mistral / Granite weight layouts
"""

import argparse
import glob
import json
import struct
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import write_nfw, write_nft, verify_nfw

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Safetensors reader
# ---------------------------------------------------------------------------

SAFETENSORS_DTYPES = {
    "F32":  (np.float32, 4),
    "F16":  (np.float16, 2),
    "BF16": (None,        2),
    "I32":  (np.int32,   4),
    "I16":  (np.int16,   2),
    "I8":   (np.int8,    1),
    "U8":   (np.uint8,   1),
    "BOOL": (np.bool_,   1),
}


class SafetensorsReader:
    def __init__(self, path: str):
        self.path = Path(path)
        self.f    = open(path, "rb")
        header_size = struct.unpack("<Q", self.f.read(8))[0]
        header_json = self.f.read(header_size).decode("utf-8")
        self.header: dict = json.loads(header_json)
        self._data_start  = 8 + header_size

    def close(self):
        self.f.close()

    def tensor_names(self) -> List[str]:
        return [k for k in self.header if k != "__metadata__"]

    def load(self, name: str) -> np.ndarray:
        meta      = self.header[name]
        dtype_str = meta["dtype"]
        shape     = meta["shape"]
        start, end= meta["data_offsets"]

        self.f.seek(self._data_start + start)
        raw = self.f.read(end - start)

        if dtype_str == "BF16":
            n       = len(raw) // 2
            f32     = np.zeros(n, dtype=np.float32)
            raw_arr = np.frombuffer(raw, dtype=np.uint8)
            f32_u8  = f32.view(np.uint8)
            f32_u8[2::4] = raw_arr[0::2]
            f32_u8[3::4] = raw_arr[1::2]
            return f32.reshape(shape)

        np_dtype, _ = SAFETENSORS_DTYPES.get(dtype_str, (np.float32, 4))
        return np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape).astype(np.float32)


def find_shards(pattern: str) -> List[str]:
    p = Path(pattern)
    if p.is_dir():
        paths = sorted(str(x) for x in p.glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"No .safetensors files in: {pattern}")
        return paths
    if p.is_file():
        return [str(p)]
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_arch(model_path: str, explicit_arch: str) -> tuple:
    # ALWAYS try to load the real config.json first, regardless of whether
    # --arch was passed explicitly. An explicit --arch only overrides which
    # ARCH_REMAPPER function is used for tensor-name remapping; it must
    # NEVER discard the actual model dimensions (hidden_size, num_heads,
    # num_hidden_layers, vocab_size, MuP multipliers, etc.) that only exist
    # in the real config.json. The old code returned an empty {} whenever
    # --arch was explicit, which silently fell back to hardcoded defaults
    # (vocab_size=32000, d_model=4096, n_layers=32, ...) for EVERY field --
    # producing a .nfw + config.json that "loads" and "runs" but is a
    # completely wrong architecture with randomly-initialised-looking
    # garbage weights reinterpreted at the wrong shapes. This was caught
    # by a real user converting IBM Granite-4.0-350M, where every single
    # field in the output config.json turned out to be a hardcoded default.
    hf_cfg = {}
    found_path = None
    for d in [Path(model_path), Path(model_path).parent]:
        cfg_path = d / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                hf_cfg = json.load(f)
            found_path = cfg_path
            break

    if explicit_arch != "auto":
        if found_path:
            print(f"  Using explicit --arch='{explicit_arch}' "
                  f"(config values still read from {found_path})")
        else:
            print(f"  Using explicit --arch='{explicit_arch}' "
                  f"(WARNING: no config.json found — all model dimensions "
                  f"will fall back to hardcoded defaults, which is almost "
                  f"certainly wrong. Place a config.json next to the model "
                  f"weights, or convert will silently produce a broken model.)")
        return explicit_arch, hf_cfg

    if found_path:
        arch = hf_cfg.get("model_type", "llama").lower()
        print(f"  Detected arch '{arch}' from {found_path}")
        return arch, hf_cfg

    print("  WARNING: config.json not found, defaulting to 'llama' "
          "with hardcoded fallback dimensions (almost certainly wrong)")
    return "llama", {}


def build_vlrf_config(hf_cfg: dict, arch: str) -> dict:
    def g(key, default): return hf_cfg.get(key, default)
    d        = g("hidden_size", 4096)
    n_heads  = g("num_attention_heads", 32)
    kv_heads = g("num_key_value_heads", n_heads)
    n_layers = g("num_hidden_layers", 32)

    # IBM Granite 4.0+ "granitemoehybrid"/"granitemoe": the DENSE FFN path
    # (GraniteMoeHybridMLP, always active) uses shared_intermediate_size,
    # NOT intermediate_size (which sizes the MoE EXPERT FFN — irrelevant
    # here since num_local_experts=0 means no experts are active). Using
    # intermediate_size would be wrong for any checkpoint where these two
    # values differ, even though they happen to match in some configs.
    if arch in ("granitemoehybrid", "granitemoe") and "shared_intermediate_size" in hf_cfg:
        d_ff = hf_cfg["shared_intermediate_size"]
    else:
        d_ff = g("intermediate_size", g("n_inner", 4 * d))

    ctx      = g("max_position_embeddings", g("n_positions", 2048))
    vocab    = g("vocab_size", 32000)
    bos      = g("bos_token_id", 1) or 1
    eos      = g("eos_token_id", 2) or 2
    eps      = g("rms_norm_eps", g("layer_norm_epsilon", 1e-5))
    use_rms  = arch not in ("gpt2", "gpt_neo", "gpt_neox")
    tied     = g("tie_word_embeddings", arch in ("gpt2", "gpt_neo"))
    # GPT-2 uses learned PE; LLaMA/Granite-family use RoPE; default sinusoidal
    rope_archs = ("llama","mistral","granite","codellama","phi","phi3",
                  "qwen2","gemma","falcon","granitemoehybrid","granitemoe")
    if arch in ("gpt2", "gpt_neo", "gpt_neox"):
        pos_emb = "learned"
    elif arch in rope_archs:
        pos_emb = "rotary"
    else:
        pos_emb = "sinusoidal"
    act      = "gelu" if arch in ("gpt2", "gpt_neo", "gpt_neox") else "silu"
    rope_theta = g("rope_theta", 10000.0)

    return {
        "model_name":     hf_cfg.get("_name_or_path", arch),
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
        "pad_token_id":   g("pad_token_id", 0) or 0,
        "layer_norm_eps": eps,
        "pre_norm":       True,
        "use_rms_norm":   use_rms,
        # MuP scaling (IBM Granite 4.0+ only; defaults are true no-ops for
        # every other architecture — see ModelConfig in config.hpp for the
        # exact application points, verified against transformers' actual
        # modeling_granitemoehybrid.py source).
        "embedding_multiplier": g("embedding_multiplier", 1.0),
        "attention_multiplier": g("attention_multiplier", 0.0),
        "residual_multiplier":  g("residual_multiplier", 1.0),
        "logits_scaling":       g("logits_scaling", 1.0),
        "tie_embeddings": tied,
        "pos_embedding":  pos_emb,
        "activation":     act,
        "rope_theta":     rope_theta,
        "attention_bias": g("attention_bias", False),
        "mlp_bias":       g("mlp_bias", False),
        "temperature":    0.8,
        "top_p":          0.9,
        "top_k":          50,
        "max_new_tokens": 256,
    }


# ---------------------------------------------------------------------------
# Weight remapping — NO speculative zero-bias writes
# Each function adds only tensors it has real data for.
# Missing biases are filled with zeros in a post-pass.
# ---------------------------------------------------------------------------

def remap_gpt2(name: str, arr: np.ndarray, vlrf: dict,
               d: int, n_heads: int) -> None:
    def T(x): return x.T if x.ndim == 2 else x
    if name == "transformer.wte.weight":     vlrf["embed.weight"] = arr; return
    if name == "transformer.wpe.weight":     vlrf["pos_embed.weight"] = arr; return
    if name in ("transformer.ln_f.weight","ln_f.weight"): vlrf["ln_final.weight"] = arr; return
    if name in ("transformer.ln_f.bias",  "ln_f.bias"):   vlrf["ln_final.bias"]   = arr; return
    if name == "lm_head.weight":             vlrf["lm_head.weight"] = arr; return

    for prefix in ("transformer.h.", "h."):
        if not name.startswith(prefix): continue
        rest  = name[len(prefix):]
        dot   = rest.index(".")
        l, tail = int(rest[:dot]), rest[dot+1:]
        dst   = f"layers.{l}"
        if tail == "ln_1.weight": vlrf[f"{dst}.ln1.weight"] = arr; return
        if tail == "ln_1.bias":   vlrf[f"{dst}.ln1.bias"]   = arr; return
        if tail == "ln_2.weight": vlrf[f"{dst}.ln2.weight"] = arr; return
        if tail == "ln_2.bias":   vlrf[f"{dst}.ln2.bias"]   = arr; return
        if tail == "attn.c_attn.weight":
            w = T(arr); q,k,v = np.split(w, 3, axis=0)
            vlrf[f"{dst}.attn.q_proj.weight"] = q
            vlrf[f"{dst}.attn.k_proj.weight"] = k
            vlrf[f"{dst}.attn.v_proj.weight"] = v; return
        if tail == "attn.c_attn.bias":
            q,k,v = np.split(arr, 3, axis=0)
            vlrf[f"{dst}.attn.q_proj.bias"] = q
            vlrf[f"{dst}.attn.k_proj.bias"] = k
            vlrf[f"{dst}.attn.v_proj.bias"] = v; return
        if tail == "attn.c_proj.weight": vlrf[f"{dst}.attn.o_proj.weight"] = T(arr); return
        if tail == "attn.c_proj.bias":   vlrf[f"{dst}.attn.o_proj.bias"]   = arr;    return
        if tail == "mlp.c_fc.weight":    vlrf[f"{dst}.ffn.fc1.weight"]     = T(arr); return
        if tail == "mlp.c_fc.bias":      vlrf[f"{dst}.ffn.fc1.bias"]       = arr;    return
        if tail == "mlp.c_proj.weight":  vlrf[f"{dst}.ffn.fc2.weight"]     = T(arr); return
        if tail == "mlp.c_proj.bias":    vlrf[f"{dst}.ffn.fc2.bias"]       = arr;    return


def remap_llama(name: str, arr: np.ndarray, vlrf: dict,
                d: int, n_heads: int, kv_heads: int) -> None:
    """
    LLaMA / Mistral / Granite.
    IMPORTANT: does NOT write speculative zero biases.
    Missing biases are added as zeros in fill_missing_biases() after all shards.
    """
    if name == "model.embed_tokens.weight":
        vlrf["embed.weight"] = arr; return
    if name in ("model.norm.weight", "norm.weight"):
        vlrf["ln_final.weight"] = arr; return
    if name == "lm_head.weight":
        vlrf["lm_head.weight"] = arr; return

    for pfx in ("model.layers.", "layers."):
        if not name.startswith(pfx): continue
        rest  = name[len(pfx):]
        dot   = rest.index(".")
        l     = int(rest[:dot])
        tail  = rest[dot+1:]
        dst   = f"layers.{l}"

        # Layer norms (no bias in RMSNorm — added as zeros in post-pass)
        if tail == "input_layernorm.weight":
            vlrf[f"{dst}.ln1.weight"] = arr; return
        if tail == "post_attention_layernorm.weight":
            vlrf[f"{dst}.ln2.weight"] = arr; return

        # Attention weights
        if tail == "self_attn.q_proj.weight": vlrf[f"{dst}.attn.q_proj.weight"] = arr; return
        if tail == "self_attn.k_proj.weight": vlrf[f"{dst}.attn.k_proj.weight"] = arr; return
        if tail == "self_attn.v_proj.weight": vlrf[f"{dst}.attn.v_proj.weight"] = arr; return
        if tail in ("self_attn.o_proj.weight", "self_attn.dense.weight"):
            vlrf[f"{dst}.attn.o_proj.weight"] = arr; return

        # Attention biases (Granite / models with attention_bias=true)
        if tail == "self_attn.q_proj.bias": vlrf[f"{dst}.attn.q_proj.bias"] = arr; return
        if tail == "self_attn.k_proj.bias": vlrf[f"{dst}.attn.k_proj.bias"] = arr; return
        if tail == "self_attn.v_proj.bias": vlrf[f"{dst}.attn.v_proj.bias"] = arr; return
        if tail in ("self_attn.o_proj.bias", "self_attn.dense.bias"):
            vlrf[f"{dst}.attn.o_proj.bias"] = arr; return

        # FFN weights — SwiGLU (3-matrix: separate gate_proj / up_proj / down_proj)
        if tail == "mlp.gate_proj.weight": vlrf[f"{dst}.ffn.gate.weight"] = arr; return
        if tail in ("mlp.up_proj.weight",   "mlp.fc1.weight"): vlrf[f"{dst}.ffn.fc1.weight"] = arr; return
        if tail in ("mlp.down_proj.weight", "mlp.fc2.weight"): vlrf[f"{dst}.ffn.fc2.weight"] = arr; return

        # FFN biases (Granite mlp_bias=true)
        if tail == "mlp.up_proj.bias":   vlrf[f"{dst}.ffn.fc1.bias"] = arr; return
        if tail == "mlp.down_proj.bias": vlrf[f"{dst}.ffn.fc2.bias"] = arr; return
        if tail == "mlp.gate_proj.bias": vlrf[f"{dst}.ffn.gate.bias"] = arr; return

        # FFN weights — FUSED 2-matrix variant (IBM Granite 4.0+ "shared_mlp").
        # GraniteMoeHybridMLP.input_linear has shape [2*hidden, in_features]
        # (a single matmul producing BOTH the gate and value paths at once,
        # split via .chunk(2, dim=-1) AFTER the matmul). Verified against
        # transformers 5.12.1's modeling_granitemoehybrid.py:
        #   hidden_states = self.input_linear(hidden_states)
        #   chunk0, chunk1 = hidden_states.chunk(2, dim=-1)
        #   hidden_states = activation(chunk0) * chunk1   <- chunk0=GATE, chunk1=VALUE
        #   hidden_states = self.output_linear(hidden_states)
        # We split the weight matrix itself along the OUTPUT dimension (rows,
        # since nn.Linear weight is [out_features, in_features]) BEFORE
        # writing it out, so VLRF's existing 3-matrix FeedForward
        # (gate.weight / fc1.weight / fc2.weight) handles it with no
        # C++ changes needed.
        if tail == "shared_mlp.input_linear.weight":
            rows = arr.shape[0]
            assert rows % 2 == 0, (
                f"shared_mlp.input_linear.weight has odd row count {rows}, "
                "cannot split into gate/value halves")
            half = rows // 2
            gate_half, value_half = arr[:half], arr[half:]
            vlrf[f"{dst}.ffn.gate.weight"] = gate_half
            vlrf[f"{dst}.ffn.fc1.weight"]  = value_half
            return
        if tail == "shared_mlp.output_linear.weight":
            vlrf[f"{dst}.ffn.fc2.weight"] = arr
            return

        return  # unrecognised tail — skip silently


def fill_missing_biases(vlrf: dict, n_layers: int, arch: str) -> int:
    """
    After all shards are processed, fill in any missing bias tensors with zeros.
    Returns count of zero-bias tensors added.
    """
    added = 0

    def zero_like_weight(weight_key: str) -> np.ndarray:
        """Return zero array matching the first dim of the weight tensor."""
        w = vlrf.get(weight_key)
        if w is None: return None
        return np.zeros(w.shape[0], dtype=np.float32)

    # Global norms
    for k in ("ln_final.bias",):
        if k not in vlrf:
            w = vlrf.get("ln_final.weight")
            if w is not None:
                vlrf[k] = np.zeros(w.shape[0], dtype=np.float32)
                added += 1

    if arch in ("gpt2", "gpt_neo", "gpt_neox"):
        return added  # GPT-2 biases come explicitly from the file

    # Per-layer biases for LLaMA-style
    for l in range(n_layers):
        dst = f"layers.{l}"

        # Layer norm biases (always zero for RMSNorm)
        for norm in ("ln1", "ln2"):
            k = f"{dst}.{norm}.bias"
            if k not in vlrf:
                w = vlrf.get(f"{dst}.{norm}.weight")
                if w is not None:
                    vlrf[k] = np.zeros(w.shape[0], dtype=np.float32)
                    added += 1

        # Attention biases
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            k = f"{dst}.attn.{proj}.bias"
            if k not in vlrf:
                z = zero_like_weight(f"{dst}.attn.{proj}.weight")
                if z is not None: vlrf[k] = z; added += 1

        # FFN biases
        for fc in ("fc1", "fc2"):
            k = f"{dst}.ffn.{fc}.bias"
            if k not in vlrf:
                z = zero_like_weight(f"{dst}.ffn.{fc}.weight")
                if z is not None: vlrf[k] = z; added += 1

    return added


ARCH_REMAPPERS = {
    "gpt2": remap_gpt2, "gpt_neo": remap_gpt2, "gpt_neox": remap_gpt2,
    "llama": remap_llama, "mistral": remap_llama, "phi": remap_llama,
    "phi3": remap_llama, "codellama": remap_llama, "qwen2": remap_llama,
    "gemma": remap_llama, "granite": remap_llama, "falcon": remap_llama,
    # IBM Granite 4.0+ "granitemoehybrid" / "granitemoe": uses the same
    # self_attn.{q,k,v,o}_proj naming as LLaMA, but a fused 2-matrix MLP
    # (shared_mlp.input_linear/output_linear) handled inside remap_llama
    # itself (see the "shared_mlp" branch). Only verified against
    # checkpoints where every layer_type is "attention" (no Mamba layers,
    # no MoE experts active) — e.g. Granite-4.0-350M. A checkpoint with
    # active Mamba or MoE layers would need additional remapping logic
    # that does not exist yet.
    "granitemoehybrid": remap_llama, "granitemoe": remap_llama,
}


# ---------------------------------------------------------------------------
# Main conversion — two-phase: collect then write
# Phase 1: stream all shards into a dict (large weights + real biases)
# Phase 2: fill missing biases with zeros, then write .nfw in one pass
#
# Memory strategy:
#   Large tensors  (>= threshold_mb): immediately written to a temp file,
#                                     only the array for the current tensor in RAM
#   Small tensors  (< threshold_mb):  kept in dict until write phase
#
# This keeps peak RAM to: largest_single_tensor + all_small_tensors
# For Granite 3B: largest = fc1/gate [10240,2560] f16 = 50MB
#                 all small (biases/norms) ≈ 100MB
#                 Total peak ≈ 150MB during conversion
# ---------------------------------------------------------------------------

def convert_streaming(model_path: str, arch: str, vlrf_cfg: dict,
                       out_path: str, verbose: bool = True) -> int:
    """
    Stream all shards, collect tensors, fill missing biases, write .nfw.
    Returns number of tensors written.
    """
    paths    = find_shards(model_path)
    remapper = ARCH_REMAPPERS.get(arch, remap_llama)
    d        = vlrf_cfg["d_model"]
    n_heads  = vlrf_cfg["n_heads"]
    kv_heads = vlrf_cfg["n_kv_heads"]
    n_layers = vlrf_cfg["n_layers"]

    if verbose:
        print(f"  Found {len(paths)} shard(s)  [collecting all tensors]")

    vlrf: dict = {}
    skipped  = 0
    total_src= 0

    for path in paths:
        reader = SafetensorsReader(path)
        names  = reader.tensor_names()
        if verbose:
            print(f"    Loading {Path(path).name} ({len(names)} tensors)...")
        for name in names:
            arr    = reader.load(name)
            before = len(vlrf)
            if arch in ("gpt2", "gpt_neo", "gpt_neox"):
                remapper(name, arr, vlrf, d, n_heads)
            else:
                remapper(name, arr, vlrf, d, n_heads, kv_heads)
            del arr
            if len(vlrf) == before:
                skipped += 1
            total_src += 1
        reader.close()

    if verbose:
        print(f"  Collected {len(vlrf)} tensors from {total_src} source tensors "
              f"({skipped} skipped)")

    # Fill missing biases with zeros
    added = fill_missing_biases(vlrf, n_layers, arch)
    if verbose and added:
        print(f"  Added {added} zero-bias tensors")

    # Write — use float16 to save disk space
    if verbose:
        print(f"  Total tensors to write: {len(vlrf)}")
    write_nfw(out_path, vlrf, storage_dtype="float16", verbose=verbose)
    return len(vlrf)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def convert_tokenizer(model_path: str, out_dir: Path) -> None:
    try:
        from transformers import AutoTokenizer
        d = Path(model_path)
        if not d.is_dir(): d = d.parent
        tok = AutoTokenizer.from_pretrained(str(d))
        vocab_sorted = sorted(tok.get_vocab().items(), key=lambda x: x[1])
        vocab_list   = [t for t, _ in vocab_sorted]
        vocab_index  = {t: i for i, t in enumerate(vocab_list)}

        # Extract BPE merge rules.
        #
        # `tokenizer.bpe_ranks` was removed from current transformers
        # versions (confirmed absent in 5.x — GPT2Tokenizer and friends no
        # longer expose it). Checking hasattr() alone silently produces
        # ZERO merges on any current transformers install, which is a much
        # more severe bug than it looks: without merge rules, encoding
        # falls back to one-token-per-byte, producing a COMPLETELY
        # different (and wrong) token sequence for every input — even
        # though the vocab strings themselves are stored correctly and
        # look fine on inspection. A model that is otherwise 100% correct
        # will produce meaningless output when fed the wrong input tokens.
        #
        # Merges still exist, but only inside the Rust `tokenizers`
        # backend that wraps the "slow" Python tokenizer class internally.
        # We extract them via that backend's stable, documented JSON
        # serialisation (`backend_tokenizer.to_str()` -> model.merges),
        # which survives transformers version changes much better than
        # the removed convenience attribute did.
        merge_pairs = None
        if hasattr(tok, "bpe_ranks"):
            merge_pairs = list(tok.bpe_ranks.keys())
        elif hasattr(tok, "backend_tokenizer"):
            try:
                state = json.loads(tok.backend_tokenizer.to_str())
                raw_merges = state.get("model", {}).get("merges", [])
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

        merges = []
        if merge_pairs:
            for a, b in merge_pairs:
                ia, ib = vocab_index.get(a, -1), vocab_index.get(b, -1)
                if ia >= 0 and ib >= 0:
                    merges.append((ia, ib))
            print(f"  Extracted {len(merges)} BPE merge rules")
        else:
            print("  WARNING: no merge rules found via bpe_ranks or "
                  "backend_tokenizer — encoding will fall back to "
                  "one-token-per-byte, which will NOT match the token ids "
                  "the original model was trained on. Generation quality "
                  "will be severely degraded or nonsensical.")

        bos = tok.bos_token_id if tok.bos_token_id is not None else 1
        eos = tok.eos_token_id if tok.eos_token_id is not None else 2
        write_nft(str(out_dir / "tokenizer.nft"), vocab_list, merges,
                  bos_id=bos, eos_id=eos)
    except ImportError:
        print("  transformers not installed — skipping tokenizer")
        print("  pip install transformers")
    except Exception as e:
        print(f"  WARNING: tokenizer failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert .safetensors to VLRF .nfw (fixed: no duplicates)")
    parser.add_argument("--model",  required=True,
                        help="Directory, .safetensors file, or glob pattern")
    parser.add_argument("--arch",   default="auto",
                        choices=list(ARCH_REMAPPERS)+["auto"])
    parser.add_argument("--out",    required=True, help="Output directory")
    parser.add_argument("--no-tok", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Safetensors → VLRF Converter  (fixed)")
    print(f"{'='*60}")
    print(f" Source : {args.model}")
    print(f" Output : {out_dir}")
    print()

    arch, hf_cfg = detect_arch(args.model, args.arch)
    vlrf_cfg = build_vlrf_config(hf_cfg, arch)

    d = vlrf_cfg["d_model"]
    print(f"  {vlrf_cfg['n_layers']} layers, d={d}, h={vlrf_cfg['n_heads']}, "
          f"kv_h={vlrf_cfg['n_kv_heads']}, vocab={vlrf_cfg['vocab_size']}")

    weights_path = out_dir / "model.nfw"
    print("\nConverting weights...")
    n_written = convert_streaming(args.model, arch, vlrf_cfg,
                                   str(weights_path))

    if args.verify:
        print("\nVerifying file is readable...")
        from nfw_writer import read_nfw
        sample = read_nfw(str(weights_path))
        missing = []
        for k in ["embed.weight","ln_final.weight","ln_final.bias"]:
            if k not in sample: missing.append(k)
        if missing: print(f"  WARNING missing: {missing}")
        else:       print(f"  OK: {len(sample)} tensors readable")

    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vlrf_cfg, f, indent=2)
    print(f"  Config → {cfg_path}")

    if not args.no_tok:
        print("\nConverting tokenizer...")
        convert_tokenizer(args.model, out_dir)

    print(f"\n{'='*60}")
    print(" Conversion complete!")
    print(f"{'='*60}")
    print(f" Weights   : {weights_path}")
    print(f" Tokenizer : {out_dir / 'tokenizer.nft'}")
    print(f" Config    : {cfg_path}")
    print()
    print("Usage:")
    print(f"  vlrf_cli run --model {weights_path}")
    print(f"             --tok   {out_dir / 'tokenizer.nft'}")
    print(f'             --prompt "int main() {{"')


if __name__ == "__main__":
    main()
