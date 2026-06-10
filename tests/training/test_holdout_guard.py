from __future__ import annotations

import pytest
import yaml

from cfm.data.training.holdout_guard import manifest_to_reachable, run_holdout_audit
from cfm.eval.holdout.lineage_audit import HoldoutLeakError
from cfm.eval.holdout.manifest import manifest_sha256

# A frozen-holdout stand-in: one held-out tile (singapore, 1, 7).
_HOLDOUT = {
    "manifest_schema_version": "2.0",
    "regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
}


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


def test_schema_refused_under_default_2_0():
    # a 1.0 manifest under the default-2.0 expectation -> REFUSED (the #16 backstop firing)
    sg_like = {
        "manifest_schema_version": "1.0",
        "regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
    }
    with pytest.raises(HoldoutLeakError, match="schema"):
        run_holdout_audit(sg_like, [])


def test_schema_accepted_with_explicit_matching_version():
    # explicit "1.0" -> schema accepted -> audit runs (clean -> no raise)
    sg_like = {
        "manifest_schema_version": "1.0",
        "regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
    }
    run_holdout_audit(sg_like, [], expected_schema_version="1.0")


# ----- Task 20 (F9): sha + lock-marker verification at read (manifest_path) -----


def _stamp_and_write(tmp, holdout: dict, *, with_marker: bool = True):
    """Stamp manifest_sha256 (the real freeze grammar) and write to disk; touch the
    _EVAL_SET_LOCKED marker beside it. Returns (path, stamped_dict)."""
    stamped = dict(holdout)
    stamped["manifest_sha256"] = manifest_sha256(stamped)
    p = tmp / "holdout_manifest.yaml"
    p.write_text(yaml.safe_dump(stamped), encoding="utf-8")
    if with_marker:
        (tmp / "_EVAL_SET_LOCKED").touch()
    return p, stamped


_CLEAN = [{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 2, 2]]}]


def test_f9_stamped_manifest_with_marker_passes(tmp_path):
    p, stamped = _stamp_and_write(tmp_path, _HOLDOUT)
    run_holdout_audit(stamped, manifest_to_reachable(_train_manifest(_CLEAN)), manifest_path=p)


def test_f9_sha_mismatch_raises_naming_the_manifest(tmp_path):
    """A manifest whose recomputed sha != its stored manifest_sha256 field is
    refused at read — and the error names the manifest path."""
    p, stamped = _stamp_and_write(tmp_path, _HOLDOUT)
    tampered = dict(stamped)
    tampered["regions"] = {"singapore": {"tiles": [{"tile_i": 9, "tile_j": 9}]}}  # post-stamp edit
    with pytest.raises(HoldoutLeakError, match=r"holdout_manifest\.yaml"):
        run_holdout_audit(tampered, [], manifest_path=p)


def test_f9_missing_sha_field_raises_fail_closed(tmp_path):
    """An UNSTAMPED manifest (no manifest_sha256 field) is unverifiable -> refused."""
    p = tmp_path / "holdout_manifest.yaml"
    p.write_text(yaml.safe_dump(_HOLDOUT), encoding="utf-8")
    (tmp_path / "_EVAL_SET_LOCKED").touch()
    with pytest.raises(HoldoutLeakError, match="manifest_sha256"):
        run_holdout_audit(dict(_HOLDOUT), [], manifest_path=p)


def test_f9_missing_lock_marker_raises(tmp_path):
    """sha verifies, but no _EVAL_SET_LOCKED beside the manifest -> refused."""
    p, stamped = _stamp_and_write(tmp_path, _HOLDOUT, with_marker=False)
    with pytest.raises(HoldoutLeakError, match="_EVAL_SET_LOCKED"):
        run_holdout_audit(stamped, [], manifest_path=p)


def test_f9_manifest_path_none_skips_sha_and_marker_checks():
    """Legacy/synthetic callers (manifest_path=None): the schema + leak audit still
    run, but the sha/marker checks are SKIPPED (pinned legacy behavior)."""
    run_holdout_audit(dict(_HOLDOUT), manifest_to_reachable(_train_manifest(_CLEAN)))


def test_f9_leak_still_detected_after_sha_and_marker_pass(tmp_path):
    """Order: schema -> sha/marker -> leak. A planted leak on a correctly stamped,
    marker-locked manifest must STILL raise (leak detection stays observable)."""
    p, stamped = _stamp_and_write(tmp_path, _HOLDOUT)
    leak = _train_manifest([{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 1, 7]]}])
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(stamped, manifest_to_reachable(leak), manifest_path=p)
