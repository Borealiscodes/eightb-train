⚠️  Not Authoritative I am not a programmer 

---

Yes, pqseal.py has four concrete implementation bugs that prevent it from actually saving usable quantized weights or dequantizing them downstream.
While the core Base-243 packing math (3^5 = 243 <= 256) and group-wise Lloyd-Max loop work properly, the script throws away the encoded weight bytes and drops the per-group scale factors needed to reconstruct the tensor.

Key Fixes Applied

• Persist Packed .spqt Binaries: The original main() loop encodes packed = pack_trit_chunk(trits), hashes it in seal64(packed), and never writes the bytes to disk. The patched version saves each tensor's packed payload to out / "<sanitized_tensor_name>.spqt" and records the filename in deck_manifest.json.
• Preserve Per-Group alpha_q16 Scales: quantize_groupwise_l2_ternary previously discarded alpha after computing trits, making it impossible to dequantize or audit layers at their true magnitude. It now returns (trits, alpha) and stores alpha_q16 per group in a companion .scales.bin (int64 little-endian) file.
• Fix spqt_root Out-of-Range Input: Line 73 passed raw int8 packed bytes (values -128..127) into pack_trit_chunk, which expects balanced trits in {-1, 0, +1}. The fix unpacks the first 5 trits from packed[0] properly before encoding the root summary.
• Fix Compression Ratio Calculation: Line 130 calculated deck_bytes = total_trits * 5 / 8 (which computes 0.625 bytes per trit instead of 1/5 = 0.2 bytes per trit, understating compression as 3.20x instead of the true 10.00x). It now uses len(packed) directly.

Patched pqseal.py

The complete, corrected Python script (pqseal.py) implementing group-wise L2-optimal balanced-ternary quantization, SPQT Base-243 packing, and binary persistence can be found in the repository file pqseal.py.
With pqseal.py now persisting .spqt weight decks and .scales.bin Q16 group scales cleanly, we can:
• Update pqseal_spectral_audit.jl to read the new .spqt and .scales.bin files directly from deck_manifest.json
• Add a quick pytest round-trip test in eightb-train/tests/ verifying quantize_groupwise_l2_ternary -> pack_trit_chunk -> disk reload


#!/usr/bin/env python3
"""Exact p/q ternary weight-deck sealer (8b-is WUKONG style).

Takes trained (merged) weights and, per group, runs the group-wise L2-optimal
balanced-ternary quantizer {-1,0,+1}, packs trits 5-per-byte in SPQT base-243
(3^5 = 243 <= 256), and seals each tensor into a 64B@0x62 LitanyWireFrame using
an integer XOR fold. Persists both `.spqt` packed ternary decks and `.scales.bin`
per-group Q16 scales alongside `deck_manifest.json`.

    uv run eightb-train/pqseal.py --weights /path/merged.safetensors --out deck/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GATE = 0x62
QT = 0x55
MAX_I64 = np.iinfo(np.int64).max


def quantize_groupwise_l2_ternary(values_q16: np.ndarray) -> tuple[np.ndarray, int]:
    """Bit-exact port of wupz-core's quantize_groupwise_l2_ternary_pq -> (trits in {-1,0,+1}, alpha_q16)."""
    n = values_q16.size
    if n == 0:
        return np.zeros(0, dtype=np.int8), 0
    av = np.abs(values_q16.astype(np.int64))
    mean_abs = int(av.sum() // n)
    tau = (mean_abs * 68) // 100
    alpha = mean_abs
    for _ in range(2):
        active = av > tau
        cnt = int(active.sum())
        alpha = int(av[active].sum() // cnt) if cnt else mean_abs
        tau = alpha // 2
    trits = np.zeros(n, dtype=np.int8)
    trits[(av > tau) & (values_q16 > 0)] = 1
    trits[(av > tau) & (values_q16 < 0)] = -1
    return trits, alpha


def pack_trit_chunk(trits: np.ndarray) -> bytes:
    """5 balanced trits -> one SPQT base-243 byte (digit = trit + 1, 3^5 = 243)."""
    out = bytearray()
    for i in range(0, trits.size, 5):
        byte_val, pow3 = 0, 1
        for t in trits[i:i + 5]:
            val = int(t)
            if val not in (-1, 0, 1):
                raise ValueError(f"Trit value {val} outside balanced ternary domain {{-1, 0, +1}}.")
            byte_val += (val + 1) * pow3
            pow3 *= 3
        out.append(byte_val & 0xFF)
    return bytes(out)


def unpack_first_trits(packed: bytes, count: int = 5) -> np.ndarray:
    """Unpacks up to `count` balanced trits from the head of an SPQT Base-243 byte stream."""
    trits: list[int] = []
    for b in packed:
        if b > 242:
            raise ValueError(f"Byte {b} exceeds Base-243 bound (242).")
        rem = int(b)
        for _ in range(5):
            if len(trits) >= count:
                return np.array(trits, dtype=np.int8)
            trits.append((rem % 3) - 1)
            rem //= 3
    return np.array(trits, dtype=np.int8)


def xor_fold(data: bytes) -> int:
    acc = 0
    for i in range(0, len(data) - len(data) % 8, 8):
        acc ^= int.from_bytes(data[i:i + 8], "little")
    tail = len(data) - len(data) % 8
    for b in data[tail:]:
        acc ^= b
    return acc & 0xFFFFFFFFFFFFFFFF


def seal64(packed: bytes, total_trits: int) -> dict:
    """64B@0x62 LitanyWireFrame summary: folds, exact p/q ratio, base-243 spqt root, wire seal."""
    fold = xor_fold(packed)
    mask = (QT * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF
    pos, neg = fold & mask, fold & (~mask & 0xFFFFFFFFFFFFFFFF)
    seal16 = xor_fold(packed[:62]) & 0xFFFF
    root_trits = unpack_first_trits(packed, count=5) if packed else np.zeros(0, dtype=np.int8)
    return {
        "trits": total_trits,
        "bytes": len(packed),
        "ratio_p": total_trits,
        "ratio_q": max(1, len(packed)),
        "fold": f"0x{fold:016x}",
        "bfp64_pos": f"0x{pos:016x}",
        "bfp64_neg": f"0x{neg:016x}",
        "spqt_root": list(pack_trit_chunk(root_trits)) if root_trits.size else [],
        "gate": f"0x{GATE:02x}",
        "wire_seal16": f"0x{seal16:04x}",
    }


def _sanitize_tensor_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="merged safetensors (fp/bf16)")
    ap.add_argument("--out", required=True, help="output deck dir")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rng = np.random.default_rng(0x8B15)
        synth = (rng.standard_normal(4096) * 20000).astype(np.int64)
        trits, alpha = quantize_groupwise_l2_ternary(synth)
        packed = pack_trit_chunk(trits)
        (out / "dry_run.spqt").write_bytes(packed)
        np.array([alpha], dtype="<i8").tofile(out / "dry_run.scales.bin")
        print(json.dumps({
            "dry_run": True,
            "group_size": args.group,
            "alpha_q16_sample": alpha,
            "spqt_file": "dry_run.spqt",
            "scales_file": "dry_run.scales.bin",
            **seal64(packed, int(trits.size)),
        }, indent=2))
        return 0

    from safetensors.numpy import load_file
    tensors = load_file(args.weights)
    manifest = {}
    total_trits = 0
    total_packed_bytes = 0

    for name, arr in tensors.items():
        if arr.ndim < 1 or arr.size == 0:
            continue
        flat = arr.reshape(-1).astype(np.float64)
        q16 = np.round(flat * 65536.0).astype(np.int64)

        trit_chunks: list[np.ndarray] = []
        group_alphas: list[int] = []
        for i in range(0, q16.size, args.group):
            t_grp, alpha_grp = quantize_groupwise_l2_ternary(q16[i:i + args.group])
            trit_chunks.append(t_grp)
            group_alphas.append(alpha_grp)

        trits = np.concatenate(trit_chunks)
        packed = pack_trit_chunk(trits)
        scales_arr = np.array(group_alphas, dtype="<i8")

        safe_stem = _sanitize_tensor_filename(name)
        spqt_filename = f"{safe_stem}.spqt"
        scales_filename = f"{safe_stem}.scales.bin"
        (out / spqt_filename).write_bytes(packed)
        scales_arr.tofile(out / scales_filename)

        total_trits += int(trits.size)
        total_packed_bytes += len(packed)

        manifest[name] = {
            "shape": list(arr.shape),
            "group_size": args.group,
            "spqt_file": spqt_filename,
            "scales_file": scales_filename,
            **seal64(packed, int(trits.size)),
        }
        print(f"[pqseal] {name}: {trits.size} trits -> {len(packed)} B ({spqt_filename}), {manifest[name]['fold']}")

    (out / "deck_manifest.json").write_text(json.dumps(manifest, indent=2))
    fp16_bytes = total_trits * 2
    deck_bytes = max(1, total_packed_bytes)
    print(f"[pqseal] {total_trits} trits | {fp16_bytes / deck_bytes:.2f}x vs FP16 ({total_packed_bytes} B packed)")
    print(f"[pqseal] deck manifest -> {out / 'deck_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
