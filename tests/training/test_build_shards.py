from __future__ import annotations

import pytest
import yaml

from cfm.data.training.build_shards import build_training_shards, compute_training_tile_ids
from cfm.eval.holdout.paths import eval_set_locked_marker, holdout_manifest_path

_RELEASE, _REGION = "2026-04-15.0", "singapore"


def test_training_set_is_validated_minus_holdout_by_id():
    """Holdout-source-identity: training = validated minus holdout, BY ID from the
    FROZEN manifest (single source). No holdout tile may appear in the training set."""
    ids = set(compute_training_tile_ids(_RELEASE, _REGION))
    holdout = yaml.safe_load(holdout_manifest_path(_RELEASE).read_text(encoding="utf-8"))
    held = {(int(t["tile_i"]), int(t["tile_j"])) for t in holdout["regions"][_REGION]["tiles"]}
    assert held.isdisjoint(ids)


def test_training_count_matches_marker_training_residual():
    """Gate-6 cross-check against the recorded property (the marker's 362)."""
    ids = compute_training_tile_ids(_RELEASE, _REGION)
    marker = yaml.safe_load(eval_set_locked_marker(_RELEASE).read_text(encoding="utf-8"))
    assert len(ids) == marker["training_residual"]


@pytest.mark.slow
def test_shards_stamp_real_lineage(tmp_path):
    """Lineage is STAMPED (not None) and points at the tile itself — so a missing
    lineage would be a genuine None at the audit (G-F4), never a synthesized value."""
    shards = build_training_shards(_RELEASE, _REGION, out_dir=tmp_path)
    assert len(shards) > 0
    for s in shards:
        assert s.lineage is not None
        assert (s.region, s.tile_i, s.tile_j) in s.lineage


@pytest.mark.slow
def test_build_is_byte_deterministic_build_twice_and_diff(tmp_path):
    """Determinism is ACROSS runs: build twice into separate dirs, diff bytes —
    not a one-build hash."""
    a, b = tmp_path / "a", tmp_path / "b"
    build_training_shards(_RELEASE, _REGION, out_dir=a)
    build_training_shards(_RELEASE, _REGION, out_dir=b)
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b and len(files_a) > 0
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"non-deterministic: {rel}"
