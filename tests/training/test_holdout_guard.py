from __future__ import annotations

import pytest

from cfm.data.training.holdout_guard import manifest_to_reachable, run_holdout_audit
from cfm.eval.holdout.lineage_audit import HoldoutLeakError

# A frozen-holdout stand-in: one held-out tile (singapore, 1, 7).
_HOLDOUT = {"regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}}}


def _train_manifest(tiles: list[dict]) -> dict:
    return {"region": "singapore", "tiles": tiles}


def test_f1f2_injected_holdout_ref_raises_and_clean_twin_passes():
    """A shard whose RECORDED lineage includes a held-out tile -> trips (F1/F2)."""
    leak = _train_manifest([{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 1, 7]]}])
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(_HOLDOUT, manifest_to_reachable(leak))
    clean = _train_manifest([{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 2, 2]]}])
    run_holdout_audit(_HOLDOUT, manifest_to_reachable(clean))  # twin: no raise


def test_f4_absent_lineage_reaches_audit_as_none_and_raises_present_twin_passes():
    """F4 (critical): a tile with NO lineage field -> Artifact.lineage is None
    (NOT synthesized) -> fail-closed raise. Present-lineage twin passes."""
    absent = _train_manifest([{"tile_i": 2, "tile_j": 2}])  # no 'lineage' key
    reachable = manifest_to_reachable(absent)
    assert reachable[0].lineage is None  # proves NO synthesis happened
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(_HOLDOUT, reachable)
    present = _train_manifest([{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 2, 2]]}])
    run_holdout_audit(_HOLDOUT, manifest_to_reachable(present))


def test_clean_passes_with_nonzero_count():
    clean = _train_manifest(
        [
            {"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 2, 2]]},
            {"tile_i": 3, "tile_j": 3, "lineage": [["singapore", 3, 3]]},
        ]
    )
    reachable = manifest_to_reachable(clean)
    assert len(reachable) > 0  # audited real shards, not zero (non-vacuous)
    run_holdout_audit(_HOLDOUT, reachable)


def test_stamped_lineage_integrity_real_training_tile_passes_and_counts():
    """A correct non-holdout lineage is accepted and counted, so 'passes' means
    'audited and cleared', not 'audited a degenerate/never-real stamp'."""
    m = _train_manifest([{"tile_i": 5, "tile_j": 9, "lineage": [["singapore", 5, 9]]}])
    reachable = manifest_to_reachable(m)
    assert reachable[0].lineage == frozenset({("singapore", 5, 9)})
    run_holdout_audit(_HOLDOUT, reachable)
