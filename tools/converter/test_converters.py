#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
"""
test_converters.py — Unit tests for all VLRF weight converters

Tests run entirely with synthetic data — no downloads required.
Each test:
  1. Creates fake weights that match the source format's layout
  2. Runs the converter
  3. Reads back the .nfw file
  4. Validates all expected VLRF keys are present with correct shapes
  5. Validates round-trip float accuracy

Run:
  python test_converters.py
  python test_converters.py -v    (verbose)
"""

import sys
import json
import struct
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import read_nfw, write_nft, NFW_MAGIC, NFW_VERSION, write_nfw

# ============================================================
# Helpers
# ============================================================

g_pass = 0
g_fail = 0
VERBOSE = "-v" in sys.argv


def check(label: str, cond: bool, msg: str = ""):
    global g_pass, g_fail
    if cond:
        g_pass += 1
        if VERBOSE:
            print(f"  PASS  {label}")
    else:
        g_fail += 1
        print(f"  FAIL  {label}" + (f": {msg}" if msg else ""))


def check_key(tensors: dict, key: str, expected_shape=None):
    present = key in tensors
    check(f"key present: {key}", present, "missing")
    if present and expected_shape is not None:
        actual = tuple(tensors[key].shape)
        exp    = tuple(expected_shape)
        check(f"shape {key}", actual == exp,
              f"got {actual}, want {exp}")


def randn(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ============================================================
# Test: nfw_writer round-trip
# ============================================================

def test_nfw_writer():
    print("\n--- test_nfw_writer ---")
    tensors = {
        "a":       randn((4, 8)),
        "b.c":     randn((16,)),
        "layers.0.weight": randn((3, 3, 3)),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.nfw")
        write_nfw(path, tensors, verbose=False)

        loaded = read_nfw(path)

        for name, arr in tensors.items():
            check_key(loaded, name, arr.shape)
            if name in loaded:
                diff = np.abs(arr - loaded[name]).max()
                check(f"round-trip {name}", diff < 1e-6,
                      f"max_diff={diff:.2e}")


# ============================================================
# Test: GPT-2 converter
# ============================================================

def make_gpt2_state(d=64, n_heads=4, n_layers=2, vocab=100, ctx=32):
    """Synthetic GPT-2 state dict (Conv1D layout)."""
    s = {}
    s["transformer.wte.weight"] = randn((vocab, d))
    s["transformer.wpe.weight"] = randn((ctx, d))
    d_ff = 4 * d

    for l in range(n_layers):
        p = f"transformer.h.{l}"
        s[f"{p}.ln_1.weight"] = np.ones(d,  dtype=np.float32)
        s[f"{p}.ln_1.bias"]   = np.zeros(d, dtype=np.float32)
        s[f"{p}.ln_2.weight"] = np.ones(d,  dtype=np.float32)
        s[f"{p}.ln_2.bias"]   = np.zeros(d, dtype=np.float32)
        # Conv1D: weight shape [in, out] i.e. [d, 3d]
        s[f"{p}.attn.c_attn.weight"] = randn((d, 3*d))
        s[f"{p}.attn.c_attn.bias"]   = randn((3*d,))
        s[f"{p}.attn.c_proj.weight"] = randn((d, d))
        s[f"{p}.attn.c_proj.bias"]   = randn((d,))
        # causal mask buffer (should be skipped)
        s[f"{p}.attn.bias"]          = np.zeros((1,1,ctx,ctx))
        s[f"{p}.mlp.c_fc.weight"]    = randn((d, d_ff))
        s[f"{p}.mlp.c_fc.bias"]      = randn((d_ff,))
        s[f"{p}.mlp.c_proj.weight"]  = randn((d_ff, d))
        s[f"{p}.mlp.c_proj.bias"]    = randn((d,))

    s["transformer.ln_f.weight"] = np.ones(d,  dtype=np.float32)
    s["transformer.ln_f.bias"]   = np.zeros(d, dtype=np.float32)
    return s


def test_gpt2_converter():
    print("\n--- test_gpt2_converter ---")
    from convert_gpt2 import convert_gpt2_weights

    class FakeCfg:
        n_embd = 64; n_head = 4; n_layer = 2; n_inner = None
        vocab_size = 100; n_positions = 32
        bos_token_id = 50256; eos_token_id = 50256
        layer_norm_epsilon = 1e-5

    import torch
    state = {k: torch.tensor(v) for k, v in make_gpt2_state().items()}
    cfg   = FakeCfg()

    vlrf = convert_gpt2_weights(state, cfg)

    # Check all expected keys
    check_key(vlrf, "embed.weight",     (100, 64))
    check_key(vlrf, "pos_embed.weight", (32, 64))
    check_key(vlrf, "ln_final.weight",  (64,))
    check_key(vlrf, "ln_final.bias",    (64,))

    for l in range(2):
        p = f"layers.{l}"
        check_key(vlrf, f"{p}.ln1.weight", (64,))
        check_key(vlrf, f"{p}.ln2.weight", (64,))
        check_key(vlrf, f"{p}.attn.q_proj.weight", (64, 64))
        check_key(vlrf, f"{p}.attn.k_proj.weight", (64, 64))
        check_key(vlrf, f"{p}.attn.v_proj.weight", (64, 64))
        check_key(vlrf, f"{p}.attn.o_proj.weight", (64, 64))
        check_key(vlrf, f"{p}.ffn.fc1.weight", (256, 64))
        check_key(vlrf, f"{p}.ffn.fc2.weight", (64, 256))

    # Conv1D transpose check: q_proj should be transposed from c_attn[:d]
    orig_q = state["transformer.h.0.attn.c_attn.weight"].numpy()[:, :64].T  # [64,64]
    diff   = np.abs(orig_q - vlrf["layers.0.attn.q_proj.weight"]).max()
    check("GPT-2 q_proj transpose correct", diff < 1e-5, f"max_diff={diff:.2e}")

    # lm_head should NOT be in output (tied)
    check("GPT-2 no lm_head (tied)", "lm_head.weight" not in vlrf)


# ============================================================
# Test: LLaMA converter
# ============================================================

def make_llama_state(d=64, n_heads=4, kv_heads=2, n_layers=2,
                     vocab=100, d_ff=128):
    s = {}
    dh   = d // n_heads
    d_kv = kv_heads * dh

    s["model.embed_tokens.weight"] = randn((vocab, d))
    s["model.norm.weight"]         = np.ones(d, dtype=np.float32)
    s["lm_head.weight"]            = randn((vocab, d))

    for l in range(n_layers):
        p = f"model.layers.{l}"
        s[f"{p}.input_layernorm.weight"]          = np.ones(d, dtype=np.float32)
        s[f"{p}.post_attention_layernorm.weight"] = np.ones(d, dtype=np.float32)
        s[f"{p}.self_attn.q_proj.weight"] = randn((d,    d))
        s[f"{p}.self_attn.k_proj.weight"] = randn((d_kv, d))
        s[f"{p}.self_attn.v_proj.weight"] = randn((d_kv, d))
        s[f"{p}.self_attn.o_proj.weight"] = randn((d,    d))
        s[f"{p}.mlp.gate_proj.weight"]    = randn((d_ff, d))
        s[f"{p}.mlp.up_proj.weight"]      = randn((d_ff, d))
        s[f"{p}.mlp.down_proj.weight"]    = randn((d,    d_ff))
    return s


def test_llama_converter():
    print("\n--- test_llama_converter ---")
    from convert_llama import convert_llama_weights

    class FakeCfg:
        hidden_size            = 64
        num_attention_heads    = 4
        num_key_value_heads    = 2
        num_hidden_layers      = 2
        intermediate_size      = 128
        max_position_embeddings= 64
        vocab_size             = 100
        bos_token_id           = 1
        eos_token_id           = 2
        rms_norm_eps           = 1e-6
        model_type             = "llama"
        tie_word_embeddings    = False

    import torch
    state = {k: torch.tensor(v) for k, v in make_llama_state().items()}
    vlrf = convert_llama_weights(state, FakeCfg())

    check_key(vlrf, "embed.weight",    (100, 64))
    check_key(vlrf, "ln_final.weight", (64,))
    check_key(vlrf, "lm_head.weight",  (100, 64))

    dh   = 64 // 4   # 16
    d_kv = 2 * dh    # 32

    for l in range(2):
        p = f"layers.{l}"
        check_key(vlrf, f"{p}.ln1.weight",           (64,))
        check_key(vlrf, f"{p}.ln2.weight",           (64,))
        check_key(vlrf, f"{p}.attn.q_proj.weight",   (64,  64))
        check_key(vlrf, f"{p}.attn.k_proj.weight",   (d_kv, 64))
        check_key(vlrf, f"{p}.attn.v_proj.weight",   (d_kv, 64))
        check_key(vlrf, f"{p}.attn.o_proj.weight",   (64,  64))
        check_key(vlrf, f"{p}.ffn.gate.weight",      (128, 64))
        check_key(vlrf, f"{p}.ffn.fc1.weight",       (128, 64))
        check_key(vlrf, f"{p}.ffn.fc2.weight",       (64,  128))

    # Zero biases present
    check("llama q_bias zeros", np.all(vlrf["layers.0.attn.q_proj.bias"] == 0))
    check("llama ln1_bias zeros", np.all(vlrf["layers.0.ln1.bias"] == 0))


# ============================================================
# Test: safetensors converter (GPT-2 layout)
# ============================================================

def write_fake_safetensors(path: str, tensors: Dict[str, np.ndarray]):
    """Write a minimal .safetensors file from numpy arrays."""
    # Build header dict
    header = {}
    offset = 0
    data_parts = []
    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        flat = arr.flatten().tobytes()
        end  = offset + len(flat)
        header[name] = {
            "dtype":        "F32",
            "shape":        list(arr.shape),
            "data_offsets": [offset, end],
        }
        data_parts.append(flat)
        offset = end

    header_json = json.dumps(header).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


def test_safetensors_gpt2():
    print("\n--- test_safetensors_converter (gpt2) ---")
    from convert_safetensors import load_all_shards, remap_gpt2

    state = make_gpt2_state(d=64, n_heads=4, n_layers=2, vocab=100, ctx=32)

    with tempfile.TemporaryDirectory() as tmp:
        st_path = str(Path(tmp) / "model.safetensors")
        write_fake_safetensors(st_path, state)

        loaded = load_all_shards(st_path)
        check("safetensors loaded all keys",
              len(loaded) == len(state),
              f"got {len(loaded)}, want {len(state)}")

        # Spot-check a value
        diff = np.abs(loaded["transformer.wte.weight"] -
                      state["transformer.wte.weight"]).max()
        check("safetensors embed round-trip", diff < 1e-6, f"max_diff={diff}")

        # Remap
        vlrf = {}
        for name, arr in loaded.items():
            remap_gpt2(name, arr, vlrf, d=64, n_heads=4)

        check_key(vlrf, "embed.weight",     (100, 64))
        check_key(vlrf, "pos_embed.weight", (32,  64))
        for l in range(2):
            p = f"layers.{l}"
            check_key(vlrf, f"{p}.attn.q_proj.weight", (64, 64))
            check_key(vlrf, f"{p}.ffn.fc1.weight",     (256, 64))


def test_safetensors_llama():
    print("\n--- test_safetensors_converter (llama) ---")
    from convert_safetensors import remap_llama

    state = make_llama_state(d=64, n_heads=4, kv_heads=2, n_layers=2,
                              vocab=100, d_ff=128)
    vlrf = {}
    for name, arr in state.items():
        remap_llama(name, arr, vlrf, d=64, n_heads=4, kv_heads=2)

    check_key(vlrf, "embed.weight",           (100, 64))
    check_key(vlrf, "layers.0.ffn.gate.weight", (128, 64))
    check_key(vlrf, "layers.0.ffn.fc1.weight",  (128, 64))
    check_key(vlrf, "layers.0.ffn.fc2.weight",  (64,  128))
    check_key(vlrf, "lm_head.weight",           (100, 64))


# ============================================================
# Test: GGUF reader (synthetic Q8_0)
# ============================================================

def make_fake_gguf_q8_0(path: str, numel: int = 256):
    """Write a minimal GGUF file with one Q8_0 tensor for testing."""
    n_blocks   = numel // 32
    block_data = bytearray()
    expected   = np.zeros(numel, dtype=np.float32)

    rng = np.random.default_rng(42)
    for b in range(n_blocks):
        scale = np.float32(rng.uniform(0.01, 0.1))
        qs    = rng.integers(-127, 127, size=32, dtype=np.int8)
        scale_f16 = np.float16(scale)
        block_data += scale_f16.tobytes()
        block_data += qs.tobytes()
        expected[b*32:(b+1)*32] = qs.astype(np.float32) * float(scale_f16)

    # Minimal GGUF: magic, version, 1 tensor, 1 kv (general.architecture)
    # KV: "general.architecture" = "llama"
    arch_key   = "general.architecture"
    arch_val   = "llama"

    def encode_str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    kv_block = encode_str(arch_key)
    kv_block += struct.pack("<I", 8)                    # STRING type
    kv_block += encode_str(arch_val)

    tensor_name = "token_embd.weight"
    tname_b     = encode_str(tensor_name)
    tdims       = struct.pack("<I", 1) + struct.pack("<Q", numel)
    tdtype      = struct.pack("<I", 8)     # Q8_0
    toffset     = struct.pack("<Q", 0)

    tensor_desc = tname_b + tdims + tdtype + toffset

    header = struct.pack("<I", 0x46554747)  # magic
    header += struct.pack("<I", 3)           # version
    header += struct.pack("<Q", 1)           # n_tensors
    header += struct.pack("<Q", 1)           # n_kv
    header += kv_block
    header += tensor_desc

    # Align data section to 32 bytes
    pos = len(header)
    pad = (32 - pos % 32) % 32

    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * pad)
        f.write(block_data)

    return expected


def test_gguf_q8_0():
    print("\n--- test_gguf_q8_0_dequant ---")
    from convert_gguf import GGUFReader

    with tempfile.TemporaryDirectory() as tmp:
        path     = str(Path(tmp) / "test.gguf")
        expected = make_fake_gguf_q8_0(path, numel=256)

        reader   = GGUFReader(path)
        check("gguf magic parsed",    True)
        check("gguf n_tensors == 1",  len(reader.tensors) == 1,
              f"got {len(reader.tensors)}")
        check("gguf arch metadata",
              reader.meta.get("general.architecture") == "llama")

        desc = reader.tensors[0]
        check("gguf tensor name",
              desc["name"] == "token_embd.weight")

        arr  = reader.load_tensor_f32(desc)
        reader.close()

        check("gguf shape correct", arr.shape == (256,),
              f"got {arr.shape}")

        diff = np.abs(arr - expected).max()
        check("gguf Q8_0 dequant accurate", diff < 1e-4,
              f"max_diff={diff:.2e}")


def test_gguf_key_remapping():
    print("\n--- test_gguf_key_remapping ---")
    from convert_gguf import remap_key

    vlrf = {}
    d  = 64; n_heads = 4; kv_heads = 2; dh = d // n_heads; d_kv = kv_heads * dh

    remap_key("token_embd.weight", "llama", 2, vlrf,
              randn((100, d)), n_heads, kv_heads, d)
    check_key(vlrf, "embed.weight", (100, d))

    remap_key("output_norm.weight", "llama", 2, vlrf,
              np.ones(d, dtype=np.float32), n_heads, kv_heads, d)
    check_key(vlrf, "ln_final.weight", (d,))
    check_key(vlrf, "ln_final.bias",   (d,))

    remap_key("blk.0.attn_q.weight", "llama", 2, vlrf,
              randn((d, d)), n_heads, kv_heads, d)
    check_key(vlrf, "layers.0.attn.q_proj.weight", (d, d))

    remap_key("blk.0.attn_k.weight", "llama", 2, vlrf,
              randn((d_kv, d)), n_heads, kv_heads, d)
    check_key(vlrf, "layers.0.attn.k_proj.weight", (d_kv, d))

    remap_key("blk.0.ffn_gate.weight", "llama", 2, vlrf,
              randn((128, d)), n_heads, kv_heads, d)
    check_key(vlrf, "layers.0.ffn.gate.weight", (128, d))

    remap_key("blk.0.ffn_norm.weight", "llama", 2, vlrf,
              np.ones(d, dtype=np.float32), n_heads, kv_heads, d)
    check_key(vlrf, "layers.0.ln2.weight", (d,))


# ============================================================
# Test: unified converter dispatch
# ============================================================

def test_convert_dispatch():
    print("\n--- test_convert_dispatch ---")
    from convert import detect_format, detect_hf_arch

    with tempfile.TemporaryDirectory() as tmp:
        gguf_file = str(Path(tmp) / "model.gguf")
        st_file   = str(Path(tmp) / "model.safetensors")
        Path(gguf_file).touch()
        Path(st_file).touch()

        check("gguf detected",        detect_format(gguf_file) == "gguf")
        check("safetensors detected", detect_format(st_file)   == "safetensors")
        check("hf name detected",     detect_format("gpt2")    == "huggingface")
        check("hf path detected",     detect_format("meta-llama/Llama-2-7b") == "huggingface")

    check("gpt2 arch",    detect_hf_arch("gpt2")                == "gpt2")
    check("gpt2-xl arch", detect_hf_arch("gpt2-xl")             == "gpt2")
    check("llama arch",   detect_hf_arch("meta-llama/Llama-2-7b-hf") == "llama")
    check("mistral arch", detect_hf_arch("mistralai/Mistral-7B") == "llama")
    check("codellama",    detect_hf_arch("codellama-7b")         == "llama")


# ============================================================
# Test: full end-to-end write + read with GPT-2 fake model
# ============================================================

def test_end_to_end_gpt2():
    print("\n--- test_end_to_end_gpt2 (write→read→verify) ---")
    from convert_gpt2 import convert_gpt2_weights, build_vlrf_config
    import torch

    class FakeCfg:
        n_embd = 32; n_head = 2; n_layer = 2; n_inner = None
        vocab_size = 50; n_positions = 16
        bos_token_id = 50256; eos_token_id = 50256
        layer_norm_epsilon = 1e-5

    state = {k: torch.tensor(v)
             for k, v in make_gpt2_state(d=32, n_heads=2, n_layers=2,
                                          vocab=50, ctx=16).items()}
    cfg = FakeCfg()
    vlrf  = convert_gpt2_weights(state, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "gpt2.nfw")
        write_nfw(path, vlrf, verbose=False)
        loaded = read_nfw(path)

        for key in vlrf:
            check_key(loaded, key, vlrf[key].shape)
            if key in loaded:
                diff = np.abs(vlrf[key] - loaded[key]).max()
                check(f"e2e {key} accurate", diff < 1e-6,
                      f"max_diff={diff:.2e}")

        cfg_dict = build_vlrf_config(cfg)
        cfg_path = str(Path(tmp) / "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_dict, f)
        check("config written", Path(cfg_path).exists())


# ============================================================
# Run all tests
# ============================================================

def main():
    print("=" * 60)
    print(" VLRF Converter Tests")
    print("=" * 60)

    test_nfw_writer()
    test_gpt2_converter()
    test_llama_converter()
    test_safetensors_gpt2()
    test_safetensors_llama()
    test_gguf_q8_0()
    test_gguf_key_remapping()
    test_convert_dispatch()
    test_end_to_end_gpt2()

    print(f"\n{'='*60}")
    print(f"Results: {g_pass} passed, {g_fail} failed")
    print(f"{'='*60}")
    return 1 if g_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
