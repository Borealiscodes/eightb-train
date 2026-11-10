# ==========================================================================
# MODULE: test_pqseal.py (SPQT Base-243 & 64B@0x62 Persistence Test Suite)
# PROJECT: Borealiscodes / eightb-train
# PURPOSE: Verifies round-trip Base-243 packing, per-group alpha_q16 scale
#          persistence, and manifest checksum integrity.
# ==========================================================================

import json
import numpy as np
import pytest
from pqseal import (
    main,
    pack_trit_chunk,
    quantize_groupwise_l2_ternary,
    seal64,
    unpack_first_trits,
    xor_fold,
)


def test_base243_round_trip_bijection():
    """Proves 5-trit Base-243 packing and unpacking preserve every trit identically."""
    rng = np.random.default_rng(42)
    trits = rng.choice([-1, 0, 1], size=125).astype(np.int8)

    packed = pack_trit_chunk(trits)
    assert len(packed) == 25
    assert all(0 <= b <= 242 for b in packed)

    unpacked = unpack_first_trits(packed, count=125)
    np.testing.assert_array_equal(unpacked, trits)


def test_pack_trit_chunk_rejects_invalid_trits():
    """Ensures values outside {-1, 0, +1} raise ValueError instead of wrapping mod 256."""
    bad_trits = np.array([-1, 0, 2, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="outside balanced ternary domain"):
        pack_trit_chunk(bad_trits)


def test_dry_run_persists_spqt_and_scales(tmp_path):
    """Verifies `--dry-run` writes `.spqt` and `.scales.bin` files to the output directory."""
    out_dir = tmp_path / "dry_deck"
    rc = main(["pqseal.py", "--weights", "unused", "--out", str(out_dir), "--dry-run"])
    assert rc == 0

    spqt_path = out_dir / "dry_run.spqt"
    scales_path = out_dir / "dry_run.scales.bin"
    assert spqt_path.exists()
    assert scales_path.exists()

    packed = spqt_path.read_bytes()
    assert len(packed) == (4096 + 4) // 5  # 820 bytes for 4096 trits
    scales = np.fromfile(scales_path, dtype="<i8")
    assert scales.size == 1
    assert scales[0] > 0


def test_full_safetensors_deck_persistence_and_manifest(tmp_path):
    """End-to-end check: seals a multi-group tensor and verifies manifest hashes and scales."""
    safetensors = pytest.importorskip("safetensors.numpy")

    weights_path = tmp_path / "merged.safetensors"
    out_dir = tmp_path / "deck"

    rng = np.random.default_rng(106)
    w_proj = rng.normal(0.0, 0.25, size=(16, 32)).astype(np.float32)  # 512 weights -> 4 groups of 128
    safetensors.save_file({"model.layers.0.self_attn.q_proj.weight": w_proj}, str(weights_path))

    rc = main([
        "pqseal.py",
        "--weights", str(weights_path),
        "--out", str(out_dir),
        "--group", "128",
    ])
    assert rc == 0

    manifest_path = out_dir / "deck_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    entry = manifest["model.layers.0.self_attn.q_proj.weight"]
    assert entry["shape"] == [16, 32]
    assert entry["trits"] == 512
    assert entry["group_size"] == 128

    spqt_bytes = (out_dir / entry["spqt_file"]).read_bytes()
    scales = np.fromfile(out_dir / entry["scales_file"], dtype="<i8")

    assert len(spqt_bytes) == entry["bytes"]
    assert scales.size == 4  # 512 / 128 = 4 groups
    assert f"0x{xor_fold(spqt_bytes):016x}" == entry["fold"]
