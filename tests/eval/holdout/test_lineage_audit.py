from __future__ import annotations

import pytest

from cfm.eval.holdout import lineage_audit as la


def _manifest(regions: dict[str, list[tuple[int, int]]]) -> dict:
    return {
        "regions": {
            r: {"tiles": [{"tile_i": i, "tile_j": j} for (i, j) in tiles]}
            for r, tiles in regions.items()
        }
    }


HOLDOUT = _manifest({"singapore": [(1, 7), (1, 8)]})


def _art(path: str, lineage) -> la.Artifact:
    return la.Artifact(path=path, lineage=lineage)


def test_clean_training_set_passes():
    arts = [_art("train/a.parquet", frozenset({("singapore", 2, 2)}))]
    la.audit_no_holdout_leak(HOLDOUT, arts)  # no raise


def test_GF1_held_out_tile_in_training_path_trips():
    arts = [_art("train/tile=EPSG3414_i1_j7/x.parquet", frozenset({("singapore", 1, 7)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF2_held_out_derived_artifact_trips():
    # A reference distribution whose lineage includes a held-out tile.
    arts = [_art("train/ref_dist.parquet", frozenset({("singapore", 2, 2), ("singapore", 1, 8)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF3_r2_on_real_baseline_referenced_from_training_trips():
    arts = [_art("train/r2_tokenizer_on_real.parquet", frozenset({("singapore", 1, 7)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF4_absent_lineage_trips_on_the_absence_fail_closed():
    # The completeness twin: a training-reachable artifact with NO recorded lineage
    # FAILS (fail-closed) - not pass - because an untracked derivative is exactly
    # where a held-out leak hides.
    arts = [_art("train/mystery.parquet", None)]
    with pytest.raises(la.HoldoutLeakError) as exc:
        la.audit_no_holdout_leak(HOLDOUT, arts)
    assert "absent lineage" in str(exc.value)


def test_REGION_SCALING_two_region_manifest_uses_identical_logic():
    # spec §B done-right test: a synthetic 2-region holdout -> the audit is
    # byte-identical (no per-region special-casing). Same leak shape trips in either.
    two = _manifest({"singapore": [(1, 7)], "regionD": [(5, 5)]})
    clean = [_art("train/a.parquet", frozenset({("singapore", 9, 9), ("regionD", 1, 1)}))]
    la.audit_no_holdout_leak(two, clean)  # no raise
    leak = [_art("train/b.parquet", frozenset({("regionD", 5, 5)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(two, leak)
