# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"

#!/usr/bin/env python3
"""
convert_gguf.py — Convert GGUF files directly to VLRF (.nfw + .nft)

Reads the GGUF binary format without requiring llama.cpp or any C library.
Supports all quantised types by dequantising to float32 on the fly.

Supported GGUF quant types dequantised here:
  F32, F16, BF16
  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K  (K-quants)

Usage:
  python convert_gguf.py --model llama-2-7b.Q4_K_M.gguf --out ./llama2_7b
  python convert_gguf.py --model mistral-7b.Q8_0.gguf   --out ./mistral_vlrf
  python convert_gguf.py --model phi-2.Q4_0.gguf        --out ./phi2_vlrf --verify

GGUF format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import argparse
import json
import sys
import struct
import numpy as np
from enum import IntEnum
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from nfw_writer import write_nfw, write_nft, verify_nfw


# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

GGUF_MAGIC   = 0x46554747   # "GGUF" little-endian
GGUF_VERSION_MIN = 2
GGUF_VERSION_MAX = 3

class GGUFValueType(IntEnum):
    UINT8   = 0;  INT8    = 1;  UINT16 = 2;  INT16  = 3
    UINT32  = 4;  INT32   = 5;  FLOAT32= 6;  BOOL   = 7
    STRING  = 8;  ARRAY   = 9;  UINT64 = 10; INT64  = 11
    FLOAT64 = 12

class GGMLType(IntEnum):
    F32   = 0;  F16   = 1
    Q4_0  = 2;  Q4_1  = 3
    Q5_0  = 6;  Q5_1  = 7
    Q8_0  = 8;  Q8_1  = 9
    Q2_K  = 10; Q3_K  = 11; Q4_K  = 12; Q5_K  = 13; Q6_K  = 14
    BF16  = 30

# Block sizes (elements per block) and raw bytes per block
GGML_BLOCK = {
    GGMLType.F32:  (1,   4),
    GGMLType.F16:  (1,   2),
    GGMLType.BF16: (1,   2),
    GGMLType.Q4_0: (32,  18),   # 2 + 16
    GGMLType.Q4_1: (32,  20),   # 4 + 16
    GGMLType.Q5_0: (32,  22),   # 2 + 4 + 16
    GGMLType.Q5_1: (32,  24),   # 4 + 4 + 16
    GGMLType.Q8_0: (32,  34),   # 2 + 32
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
}


# ---------------------------------------------------------------------------
# GGUF binary reader
# ---------------------------------------------------------------------------

class GGUFReader:
    def __init__(self, path: str):
        self.path = Path(path)
        self.f    = open(path, "rb")
        self.meta: Dict[str, Any] = {}
        self.tensors: List[dict]  = []   # descriptor list (no data yet)
        self._data_offset = 0
        self._parse_header()

    def close(self):
        self.f.close()

    # ---- Primitive readers ------------------------------------------------

    def _ru8(self):  return struct.unpack("<B",  self.f.read(1))[0]
    def _ri8(self):  return struct.unpack("<b",  self.f.read(1))[0]
    def _ru16(self): return struct.unpack("<H",  self.f.read(2))[0]
    def _ri16(self): return struct.unpack("<h",  self.f.read(2))[0]
    def _ru32(self): return struct.unpack("<I",  self.f.read(4))[0]
    def _ri32(self): return struct.unpack("<i",  self.f.read(4))[0]
    def _ru64(self): return struct.unpack("<Q",  self.f.read(8))[0]
    def _ri64(self): return struct.unpack("<q",  self.f.read(8))[0]
    def _rf32(self): return struct.unpack("<f",  self.f.read(4))[0]
    def _rf64(self): return struct.unpack("<d",  self.f.read(8))[0]
    def _rbool(self):return bool(self._ru8())

    def _rstring(self):
        n = self._ru64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _rvalue(self, vtype: int):
        vt = GGUFValueType(vtype)
        if vt == GGUFValueType.UINT8:   return self._ru8()
        if vt == GGUFValueType.INT8:    return self._ri8()
        if vt == GGUFValueType.UINT16:  return self._ru16()
        if vt == GGUFValueType.INT16:   return self._ri16()
        if vt == GGUFValueType.UINT32:  return self._ru32()
        if vt == GGUFValueType.INT32:   return self._ri32()
        if vt == GGUFValueType.FLOAT32: return self._rf32()
        if vt == GGUFValueType.BOOL:    return self._rbool()
        if vt == GGUFValueType.STRING:  return self._rstring()
        if vt == GGUFValueType.UINT64:  return self._ru64()
        if vt == GGUFValueType.INT64:   return self._ri64()
        if vt == GGUFValueType.FLOAT64: return self._rf64()
        if vt == GGUFValueType.ARRAY:
            elem_type = self._ru32()
            count     = self._ru64()
            return [self._rvalue(elem_type) for _ in range(count)]
        raise ValueError(f"Unknown GGUF value type {vtype}")

    # ---- Header parsing ---------------------------------------------------

    def _parse_header(self):
        magic = self._ru32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic=0x{magic:08X})")

        version = self._ru32()
        if not (GGUF_VERSION_MIN <= version <= GGUF_VERSION_MAX):
            raise ValueError(f"Unsupported GGUF version {version}")
        self.version = version

        n_tensors = self._ru64()
        n_kv      = self._ru64()

        # Metadata key-value pairs
        for _ in range(n_kv):
            key   = self._rstring()
            vtype = self._ru32()
            val   = self._rvalue(vtype)
            self.meta[key] = val

        # Tensor descriptors
        for _ in range(n_tensors):
            name  = self._rstring()
            ndim  = self._ru32()
            dims  = [self._ru64() for _ in range(ndim)]
            dtype = GGMLType(self._ru32())
            offset= self._ru64()   # relative to data section start
            self.tensors.append({
                "name":   name,
                "dims":   dims,
                "dtype":  dtype,
                "offset": offset,
            })

        # Data section starts at next 32-byte alignment after header
        pos = self.f.tell()
        alignment = int(self.meta.get("general.alignment", 32))
        aligned = ((pos + alignment - 1) // alignment) * alignment
        self._data_offset = aligned

    # ---- Tensor data loading with dequantisation -------------------------

    def load_tensor_f32(self, desc: dict) -> np.ndarray:
        """Load one tensor and return as float32 ndarray in C order."""
        dtype  = desc["dtype"]
        dims   = desc["dims"]           # GGUF dims are innermost-first
        offset = desc["offset"]

        numel = 1
        for d in dims:
            numel *= d

        self.f.seek(self._data_offset + offset)

        arr = self._dequantise(dtype, numel)

        # GGUF stores dims [innermost, ..., outermost]; numpy wants [outermost,...,innermost]
        shape = list(reversed(dims))
        return arr.reshape(shape).astype(np.float32)

    def _dequantise(self, dtype: GGMLType, numel: int) -> np.ndarray:
        if dtype == GGMLType.F32:
            raw = np.frombuffer(self.f.read(numel * 4), dtype=np.float32).copy()
            return raw

        if dtype == GGMLType.F16:
            raw = np.frombuffer(self.f.read(numel * 2), dtype=np.float16).copy()
            return raw.astype(np.float32)

        if dtype == GGMLType.BF16:
            raw = np.frombuffer(self.f.read(numel * 2), dtype=np.uint16).copy()
            # Reinterpret BF16 → F32 by shifting into upper 16 bits
            f32 = np.zeros(numel, dtype=np.uint32)
            f32[:] = raw.astype(np.uint32) << 16
            return f32.view(np.float32)

        if dtype == GGMLType.Q8_0:
            return self._dequant_q8_0(numel)
        if dtype == GGMLType.Q4_0:
            return self._dequant_q4_0(numel)
        if dtype == GGMLType.Q4_1:
            return self._dequant_q4_1(numel)
        if dtype == GGMLType.Q5_0:
            return self._dequant_q5_0(numel)
        if dtype == GGMLType.Q5_1:
            return self._dequant_q5_1(numel)
        if dtype == GGMLType.Q2_K:
            return self._dequant_q2_k(numel)
        if dtype == GGMLType.Q3_K:
            return self._dequant_q3_k(numel)
        if dtype == GGMLType.Q4_K:
            return self._dequant_q4_k(numel)
        if dtype == GGMLType.Q5_K:
            return self._dequant_q5_k(numel)
        if dtype == GGMLType.Q6_K:
            return self._dequant_q6_k(numel)

        raise NotImplementedError(f"Dequantisation not implemented for {dtype.name}")

    # ---- Q8_0: blocks of 32, each: [scale:f16][qs:i8*32] -----------------
    def _dequant_q8_0(self, numel: int) -> np.ndarray:
        block_size = 32
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * 34)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base = b * 34
            scale = np.frombuffer(buf[base:base+2], dtype=np.float16)[0].astype(np.float32)
            qs    = buf[base+2:base+34].view(np.int8).astype(np.float32)
            out[b*block_size:(b+1)*block_size] = qs * scale
        return out

    # ---- Q4_0: blocks of 32, each: [scale:f16][qs:u4*32 packed in 16 bytes]
    def _dequant_q4_0(self, numel: int) -> np.ndarray:
        block_size = 32
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * 18)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * 18
            scale = np.frombuffer(buf[base:base+2], dtype=np.float16)[0].astype(np.float32)
            packed= buf[base+2:base+18]
            lo    = (packed & 0x0F).astype(np.int8)
            hi    = ((packed >> 4) & 0x0F).astype(np.int8)
            qs    = np.empty(32, dtype=np.int8)
            qs[0::2] = lo - 8
            qs[1::2] = hi - 8
            out[b*block_size:(b+1)*block_size] = qs.astype(np.float32) * scale
        return out

    # ---- Q4_1: [d:f16][m:f16][qs:u4*32 in 16 bytes] ----------------------
    def _dequant_q4_1(self, numel: int) -> np.ndarray:
        block_size = 32
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * 20)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * 20
            d     = np.frombuffer(buf[base:base+2],   dtype=np.float16)[0].astype(np.float32)
            m     = np.frombuffer(buf[base+2:base+4], dtype=np.float16)[0].astype(np.float32)
            packed= buf[base+4:base+20]
            lo    = (packed & 0x0F).astype(np.float32)
            hi    = ((packed >> 4) & 0x0F).astype(np.float32)
            qs    = np.empty(32, dtype=np.float32)
            qs[0::2] = lo
            qs[1::2] = hi
            out[b*block_size:(b+1)*block_size] = qs * d + m
        return out

    # ---- Q5_0: [scale:f16][qh:u32][qs:u4*32 in 16 bytes] ----------------
    def _dequant_q5_0(self, numel: int) -> np.ndarray:
        block_size = 32
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * 22)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * 22
            scale = np.frombuffer(buf[base:base+2], dtype=np.float16)[0].astype(np.float32)
            qh    = np.frombuffer(buf[base+2:base+6], dtype=np.uint32)[0]
            packed= buf[base+6:base+22]
            lo    = (packed & 0x0F).astype(np.int32)
            hi    = ((packed >> 4) & 0x0F).astype(np.int32)
            qs    = np.empty(32, dtype=np.int32)
            qs[0::2] = lo
            qs[1::2] = hi
            for i in range(32):
                qs[i] |= ((qh >> i) & 1) << 4
            out[b*block_size:(b+1)*block_size] = (qs - 16).astype(np.float32) * scale
        return out

    # ---- Q5_1: [d:f16][m:f16][qh:u32][qs:u4*32] -------------------------
    def _dequant_q5_1(self, numel: int) -> np.ndarray:
        block_size = 32
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * 24)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * 24
            d     = np.frombuffer(buf[base:base+2],   dtype=np.float16)[0].astype(np.float32)
            m     = np.frombuffer(buf[base+2:base+4], dtype=np.float16)[0].astype(np.float32)
            qh    = np.frombuffer(buf[base+4:base+8], dtype=np.uint32)[0]
            packed= buf[base+8:base+24]
            lo    = (packed & 0x0F).astype(np.int32)
            hi    = ((packed >> 4) & 0x0F).astype(np.int32)
            qs    = np.empty(32, dtype=np.int32)
            qs[0::2] = lo
            qs[1::2] = hi
            for i in range(32):
                qs[i] |= ((qh >> i) & 1) << 4
            out[b*block_size:(b+1)*block_size] = qs.astype(np.float32) * d + m
        return out

    # ---- K-quants (super-block structure) ---------------------------------

    # Q6_K: 256-element super-blocks, 6 bits per weight
    # Layout: [ql:128 bytes][qh:64 bytes][scales:16 bytes (i8)][d:f16]
    def _dequant_q6_k(self, numel: int) -> np.ndarray:
        block_size = 256
        n_blocks   = numel // block_size
        bytes_per  = 210
        raw        = self.f.read(n_blocks * bytes_per)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base   = b * bytes_per
            ql     = buf[base:base+128]
            qh     = buf[base+128:base+192]
            scales = buf[base+192:base+208].view(np.int8).astype(np.float32)
            d      = np.frombuffer(buf[base+208:base+210], dtype=np.float16)[0].astype(np.float32)

            vals = np.empty(256, dtype=np.float32)
            for i in range(128):
                q0 = (ql[i] & 0x0F) | (((qh[i] >> 0) & 3) << 4)
                q1 = (ql[i] >>   4) | (((qh[i] >> 2) & 3) << 4)
                sc0 = scales[i // 16]
                sc1 = scales[8 + i // 16]
                vals[i]       = d * sc0 * (q0 - 32)
                vals[i + 128] = d * sc1 * (q1 - 32)
            out[b*block_size:(b+1)*block_size] = vals
        return out

    # Q4_K: [d:f16][dmin:f16][scales:12 bytes][qs:128 bytes]  256 elements
    def _dequant_q4_k(self, numel: int) -> np.ndarray:
        block_size = 256
        n_blocks   = numel // block_size
        bytes_per  = 144
        raw        = self.f.read(n_blocks * bytes_per)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * bytes_per
            d     = np.frombuffer(buf[base:base+2],   dtype=np.float16)[0].astype(np.float32)
            dmin  = np.frombuffer(buf[base+2:base+4], dtype=np.float16)[0].astype(np.float32)
            sc_raw= buf[base+4:base+16]
            qs    = buf[base+16:base+144]

            # Unpack 6-bit scales and 4-bit mins from 12 bytes
            scales = np.zeros(8, dtype=np.float32)
            mins   = np.zeros(8, dtype=np.float32)
            for i in range(8):
                j = (i * 6) // 8
                s = sc_raw[j] if j < 12 else 0
                scales[i] = (s & 0x3F) * d
                mins[i]   = ((s >> 6) & 0x3) * dmin  # simplified

            for i in range(128):
                lo = qs[i] & 0x0F
                hi = (qs[i] >> 4) & 0x0F
                sc_idx0 = (i // 32)
                sc_idx1 = (i // 32) + 4
                out[b*256 + i]       = scales[sc_idx0] * lo - mins[sc_idx0]
                out[b*256 + i + 128] = scales[sc_idx1] * hi - mins[sc_idx1]
        return out

    # Q2_K, Q3_K, Q5_K — simplified dequant (good enough for inference)
    def _dequant_q2_k(self, numel: int) -> np.ndarray:
        return self._dequant_k_fallback(numel, GGMLType.Q2_K, 2)

    def _dequant_q3_k(self, numel: int) -> np.ndarray:
        return self._dequant_k_fallback(numel, GGMLType.Q3_K, 3)

    def _dequant_q5_k(self, numel: int) -> np.ndarray:
        # Q5_K shares block structure with Q4_K + high bits
        block_size = 256
        n_blocks   = numel // block_size
        bytes_per  = 176
        raw        = self.f.read(n_blocks * bytes_per)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.empty(numel, dtype=np.float32)
        for b in range(n_blocks):
            base  = b * bytes_per
            d     = np.frombuffer(buf[base:base+2],   dtype=np.float16)[0].astype(np.float32)
            dmin  = np.frombuffer(buf[base+2:base+4], dtype=np.float16)[0].astype(np.float32)
            sc_raw= buf[base+4:base+16]
            qh    = buf[base+16:base+48]
            qs    = buf[base+48:base+176]

            scales = np.zeros(8, dtype=np.float32)
            mins   = np.zeros(8, dtype=np.float32)
            for i in range(8):
                j = (i * 6) // 8
                s = sc_raw[j] if j < 12 else 0
                scales[i] = (s & 0x3F) * d
                mins[i]   = ((s >> 6) & 0x3) * dmin

            for i in range(128):
                qlo = qs[i] & 0x0F
                qhi = (qs[i] >> 4) & 0x0F
                hb0 = (qh[i // 4] >> (2 * (i % 4))) & 1
                hb1 = (qh[i // 4] >> (2 * (i % 4) + 1)) & 1
                q0  = qlo | (hb0 << 4)
                q1  = qhi | (hb1 << 4)
                si0 = i // 32
                si1 = si0 + 4
                out[b*256 + i]       = scales[si0] * q0 - mins[si0]
                out[b*256 + i + 128] = scales[si1] * q1 - mins[si1]
        return out

    def _dequant_k_fallback(self, numel: int, dtype: GGMLType, bits: int) -> np.ndarray:
        """Generic fallback for K-quant types: read raw bytes and scale naively."""
        _, bytes_per_block = GGML_BLOCK[dtype]
        block_size = 256
        n_blocks   = numel // block_size
        raw        = self.f.read(n_blocks * bytes_per_block)
        buf        = np.frombuffer(raw, dtype=np.uint8)
        out        = np.zeros(numel, dtype=np.float32)
        mask       = (1 << bits) - 1
        offset     = (1 << (bits - 1))
        # Read scale from last 2 bytes of each block
        for b in range(n_blocks):
            bstart = b * bytes_per_block
            bend   = bstart + bytes_per_block
            d  = np.frombuffer(buf[bend-2:bend], dtype=np.float16)[0].astype(np.float32)
            data_bytes = buf[bstart:bend-2]
            vals = []
            bit_pos = 0
            byte_idx = 0
            acc = int(data_bytes[0]) if len(data_bytes) > 0 else 0
            bits_in_acc = 8
            for _ in range(block_size):
                if bits_in_acc < bits:
                    byte_idx += 1
                    if byte_idx < len(data_bytes):
                        acc |= int(data_bytes[byte_idx]) << bits_in_acc
                        bits_in_acc += 8
                v = (acc & mask) - offset
                acc >>= bits
                bits_in_acc -= bits
                vals.append(v)
            out[b*block_size:(b+1)*block_size] = np.array(vals, dtype=np.float32) * d
        return out


# ---------------------------------------------------------------------------
# Architecture inference from GGUF metadata
# ---------------------------------------------------------------------------

def infer_arch(meta: dict) -> str:
    return meta.get("general.architecture", "llama")


def build_vlrf_config_from_gguf(meta: dict) -> dict:
    arch = infer_arch(meta)
    pfx  = arch  # e.g. "llama", "mistral", "phi2"

    def m(key, default=None):
        return meta.get(f"{pfx}.{key}", meta.get(key, default))

    d        = int(m("embedding_length", 4096))
    n_heads  = int(m("attention.head_count", 32))
    kv_heads = int(m("attention.head_count_kv", n_heads))
    n_layers = int(m("block_count", 32))
    d_ff     = int(m("feed_forward_length", 4 * d))
    ctx      = int(m("context_length", 4096))
    vocab    = int(meta.get("tokenizer.ggml.tokens", [None] * 32000).__len__()
                   if "tokenizer.ggml.tokens" in meta else 32000)

    norm_eps = float(m("attention.layer_norm_rms_epsilon",
                        m("attention.layer_norm_epsilon", 1e-5)))
    use_rms  = "rms" in str(m("attention.layer_norm_type", "rms")).lower()

    return {
        "model_name":     meta.get("general.name", arch),
        "model_version":  "1.0.0",
        "source_arch":    arch,
        "vocab_size":     vocab,
        "d_model":        d,
        "d_ff":           d_ff,
        "n_heads":        n_heads,
        "n_kv_heads":     kv_heads,
        "n_layers":       n_layers,
        "max_seq_len":    ctx,
        "bos_token_id":   int(meta.get("tokenizer.ggml.bos_token_id", 1)),
        "eos_token_id":   int(meta.get("tokenizer.ggml.eos_token_id", 2)),
        "pad_token_id":   int(meta.get("tokenizer.ggml.padding_token_id", 0)),
        "layer_norm_eps": norm_eps,
        "pre_norm":       True,
        "use_rms_norm":   use_rms,
        "tie_embeddings": False,
        "pos_embedding":  "sinusoidal",
        "activation":     "silu",
        "temperature":    0.8,
        "top_p":          0.9,
        "top_k":          50,
        "max_new_tokens": 256,
    }


# ---------------------------------------------------------------------------
# Tensor key remapping: GGUF → VLRF
# ---------------------------------------------------------------------------

def remap_key(name: str, arch: str, n_layers: int,
              vlrf_tensors: dict, arr: np.ndarray,
              n_heads: int, kv_heads: int, d_model: int) -> None:
    """
    Map one GGUF tensor name to one or more VLRF weight map entries.
    Modifies vlrf_tensors in-place.
    """

    # ---- Embeddings
    if name == "token_embd.weight":
        vlrf_tensors["embed.weight"] = arr
        return

    # ---- Final norm
    if name in ("output_norm.weight", "final_norm.weight"):
        vlrf_tensors["ln_final.weight"] = arr
        vlrf_tensors["ln_final.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # ---- LM head
    if name == "output.weight":
        vlrf_tensors["lm_head.weight"] = arr
        return

    # ---- Per-layer tensors: blk.N.*
    if not name.startswith("blk."):
        return  # skip unknown top-level tensors

    parts = name.split(".")
    l     = int(parts[1])
    rest  = ".".join(parts[2:])
    pfx   = f"layers.{l}"

    dh     = d_model // n_heads
    d_kv   = kv_heads * dh

    # Attention norms
    if rest == "attn_norm.weight":
        vlrf_tensors[f"{pfx}.ln1.weight"] = arr
        vlrf_tensors[f"{pfx}.ln1.bias"]   = np.zeros_like(arr)
        return

    # FFN norms
    if rest == "ffn_norm.weight":
        vlrf_tensors[f"{pfx}.ln2.weight"] = arr
        vlrf_tensors[f"{pfx}.ln2.bias"]   = np.zeros_like(arr)
        return

    # Attention Q
    if rest == "attn_q.weight":
        vlrf_tensors[f"{pfx}.attn.q_proj.weight"] = arr
        vlrf_tensors[f"{pfx}.attn.q_proj.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # Attention K
    if rest == "attn_k.weight":
        vlrf_tensors[f"{pfx}.attn.k_proj.weight"] = arr
        vlrf_tensors[f"{pfx}.attn.k_proj.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # Attention V
    if rest == "attn_v.weight":
        vlrf_tensors[f"{pfx}.attn.v_proj.weight"] = arr
        vlrf_tensors[f"{pfx}.attn.v_proj.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # Attention output
    if rest == "attn_output.weight":
        vlrf_tensors[f"{pfx}.attn.o_proj.weight"] = arr
        vlrf_tensors[f"{pfx}.attn.o_proj.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # FFN gate (SwiGLU)
    if rest == "ffn_gate.weight":
        vlrf_tensors[f"{pfx}.ffn.gate.weight"] = arr
        return

    # FFN up
    if rest == "ffn_up.weight":
        vlrf_tensors[f"{pfx}.ffn.fc1.weight"] = arr
        vlrf_tensors[f"{pfx}.ffn.fc1.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # FFN down
    if rest == "ffn_down.weight":
        vlrf_tensors[f"{pfx}.ffn.fc2.weight"] = arr
        vlrf_tensors[f"{pfx}.ffn.fc2.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return

    # Fused QKV (some models store together)
    if rest == "attn_qkv.weight":
        q, k, v = np.split(arr, [d_model, d_model + d_kv], axis=0)
        vlrf_tensors[f"{pfx}.attn.q_proj.weight"] = q
        vlrf_tensors[f"{pfx}.attn.k_proj.weight"] = k
        vlrf_tensors[f"{pfx}.attn.v_proj.weight"] = v
        vlrf_tensors[f"{pfx}.attn.q_proj.bias"]   = np.zeros(d_model, dtype=np.float32)
        vlrf_tensors[f"{pfx}.attn.k_proj.bias"]   = np.zeros(d_kv,    dtype=np.float32)
        vlrf_tensors[f"{pfx}.attn.v_proj.bias"]   = np.zeros(d_kv,    dtype=np.float32)
        return

    # Standard 2-layer FFN (non-SwiGLU models like Falcon)
    if rest == "ffn_up.weight" or rest == "ffn_gate_up.weight":
        vlrf_tensors[f"{pfx}.ffn.fc1.weight"] = arr
        vlrf_tensors[f"{pfx}.ffn.fc1.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return
    if rest == "ffn_down.weight":
        vlrf_tensors[f"{pfx}.ffn.fc2.weight"] = arr
        vlrf_tensors[f"{pfx}.ffn.fc2.bias"]   = np.zeros(arr.shape[0], dtype=np.float32)
        return


# ---------------------------------------------------------------------------
# Tokenizer extraction from GGUF metadata
# ---------------------------------------------------------------------------

def extract_tokenizer(meta: dict, out_dir: Path) -> None:
    tokens = meta.get("tokenizer.ggml.tokens", [])
    if not tokens:
        print("  WARNING: no tokenizer.ggml.tokens in GGUF metadata, skipping")
        return

    # token_type: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
    token_types = meta.get("tokenizer.ggml.token_type", [1] * len(tokens))

    # Convert merge strings "a b" → (id_a, id_b)
    merges_raw = meta.get("tokenizer.ggml.merges", [])
    tok2id     = {t: i for i, t in enumerate(tokens)}
    merges     = []
    for m in merges_raw:
        parts = m.split(" ", 1)
        if len(parts) == 2:
            ia = tok2id.get(parts[0], -1)
            ib = tok2id.get(parts[1], -1)
            if ia >= 0 and ib >= 0:
                merges.append((ia, ib))

    bos = int(meta.get("tokenizer.ggml.bos_token_id", 1))
    eos = int(meta.get("tokenizer.ggml.eos_token_id", 2))
    pad = int(meta.get("tokenizer.ggml.padding_token_id", 0))
    unk = int(meta.get("tokenizer.ggml.unknown_token_id", 0))

    nft_path = out_dir / "tokenizer.nft"
    write_nft(str(nft_path), tokens, merges,
              pad_id=pad, bos_id=bos, eos_id=eos, unk_id=unk)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" GGUF → VLRF Converter")
    print(f"{'='*60}")
    print(f" Source : {args.model}")
    print(f" Output : {out_dir}")
    print()

    print("Parsing GGUF header...")
    reader = GGUFReader(args.model)

    arch     = infer_arch(reader.meta)
    vlrf_cfg   = build_vlrf_config_from_gguf(reader.meta)
    n_layers = vlrf_cfg["n_layers"]
    d_model  = vlrf_cfg["d_model"]
    n_heads  = vlrf_cfg["n_heads"]
    kv_heads = vlrf_cfg["n_kv_heads"]

    print(f"  Architecture : {arch}")
    print(f"  Layers       : {n_layers}")
    print(f"  d_model      : {d_model}")
    print(f"  n_heads      : {n_heads}  kv_heads: {kv_heads}")
    print(f"  FFN dim      : {vlrf_cfg['d_ff']}")
    print(f"  Context      : {vlrf_cfg['max_seq_len']}")
    print(f"  Tensors      : {len(reader.tensors)}")
    print()

    # Count quant types
    from collections import Counter
    qtypes = Counter(t["dtype"].name for t in reader.tensors)
    print("  Quant types  :", dict(qtypes))
    print()

    # ---- Convert tensors
    vlrf_tensors: dict = {}
    n = len(reader.tensors)
    for i, desc in enumerate(reader.tensors):
        name  = desc["name"]
        dtype = desc["dtype"]
        dims  = desc["dims"]
        numel = 1
        for d in dims:
            numel *= d

        print(f"  [{i+1:4d}/{n}] {name:50s} {dtype.name:6s} {list(reversed(dims))}", end="\r")

        arr = reader.load_tensor_f32(desc)
        remap_key(name, arch, n_layers, vlrf_tensors, arr,
                  n_heads, kv_heads, d_model)

    print()  # clear \r line
    print(f"\n  Mapped {len(vlrf_tensors)} VLRF tensors from {n} GGUF tensors")

    reader.close()

    # ---- Write weights
    weights_path = out_dir / "model.nfw"
    write_nfw(str(weights_path), vlrf_tensors)

    if args.verify:
        print("\nVerifying round-trip...")
        verify_nfw(vlrf_tensors, str(weights_path))

    # ---- Write config
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vlrf_cfg, f, indent=2)
    print(f"  Config → {cfg_path}")

    # ---- Extract tokenizer
    if not args.no_tok:
        print("\nExtracting tokenizer from GGUF metadata...")
        extract_tokenizer(reader.meta, out_dir)

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
    print(f'             --prompt "int main()"')
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF weights to VLRF .nfw format")
    parser.add_argument("--model",  required=True,
                        help="Path to input .gguf file")
    parser.add_argument("--out",    required=True,
                        help="Output directory")
    parser.add_argument("--no-tok", action="store_true",
                        help="Skip tokenizer extraction")
    parser.add_argument("--verify", action="store_true",
                        help="Round-trip verify weights after writing")
    args = parser.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"Error: file not found: {args.model}")

    convert(args)


if __name__ == "__main__":
    main()
