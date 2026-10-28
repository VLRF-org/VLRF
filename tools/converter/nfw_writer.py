# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
import struct, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

NFW_MAGIC=0x4E465746; NFW_VERSION=1; NFW_VERSION2=2
NFT_MAGIC=0x4E465446; NFT_VERSION=1
DTYPE_F32=0; DTYPE_F16=1

def write_nfw(path, tensors, storage_dtype="float32", verbose=True):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    n=len(tensors); use_f16=(storage_dtype=="float16")
    version=NFW_VERSION2 if use_f16 else NFW_VERSION
    total=sum(v.size for v in tensors.values())
    if verbose: print(f"Writing {n} tensors ({total/1e6:.1f}M params) → {path}")
    with open(path,"wb") as f:
        f.write(struct.pack("<IIII",NFW_MAGIC,version,n,0))
        for name,arr in tensors.items():
            arr=np.asarray(arr,dtype=np.float16 if use_f16 else np.float32)
            name_b=name.encode(); f.write(struct.pack("<H",len(name_b))); f.write(name_b)
            f.write(struct.pack("<B",arr.ndim))
            for d in arr.shape: f.write(struct.pack("<i",int(d)))
            f.write(struct.pack("<q",arr.size))
            if use_f16: f.write(struct.pack("<B",DTYPE_F16))
            f.write(arr.flatten().tobytes())
    if verbose: print(f"  Saved {path.stat().st_size/1024/1024:.0f} MB")

def read_nfw(path):
    t={}
    with open(path,"rb") as f:
        magic,version,n,_=struct.unpack("<IIII",f.read(16))
        for _ in range(n):
            nl,=struct.unpack("<H",f.read(2)); name=f.read(nl).decode()
            nd,=struct.unpack("<B",f.read(1)); dims=list(struct.unpack(f"<{nd}i",f.read(4*nd)))
            ne,=struct.unpack("<q",f.read(8))
            dtype=DTYPE_F32
            if version==NFW_VERSION2: dtype,=struct.unpack("<B",f.read(1))
            if dtype==DTYPE_F16:
                data=np.frombuffer(f.read(ne*2),dtype=np.float16).copy().astype(np.float32)
            else:
                data=np.frombuffer(f.read(ne*4),dtype=np.float32).copy()
            t[name]=data.reshape(dims)
    return t

def write_nft(path, vocab, merges, pad_id=0, bos_id=1, eos_id=2, unk_id=3, verbose=True):
    path=Path(path)
    if verbose: print(f"Writing tokenizer ({len(vocab)} tokens) → {path}")
    with open(path,"wb") as f:
        f.write(struct.pack("<IIII",NFT_MAGIC,NFT_VERSION,len(vocab),0))
        for tok in vocab:
            b=tok.encode("utf-8"); f.write(struct.pack("<H",len(b))); f.write(b)
        f.write(struct.pack("<I",len(merges)))
        for a,b in merges: f.write(struct.pack("<II",a,b))

def verify_nfw(original, path, atol=1e-3):
    loaded=read_nfw(path); ok=True
    for name,arr in original.items():
        if name not in loaded: print(f"  MISSING: {name}"); ok=False; continue
        diff=np.abs(arr.astype(np.float32)-loaded[name].astype(np.float32)).max()
        if diff>atol: print(f"  MISMATCH {name}: {diff:.4f}"); ok=False
    if ok: print(f"  Verification passed ({len(original)} tensors)")
    return ok
