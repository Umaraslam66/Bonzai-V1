from __future__ import annotations

import pytest
import yaml

from cfm.data.training.build_shards import (
    _tile_conditioning_dict,
    build_training_shards,
    compute_training_tile_ids,
)
from cfm.eval.holdout.labels import MorphologyStratum, TileLabels
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


def test_tile_conditioning_distinguishes_admin_region_from_city():
    """F6 trap guard: the dict key is admin_region (the division, None for EU);
    the CITY name lives on TrainingShard.region — the two must never share a key."""
    labels = TileLabels(
        tile_i=0,
        tile_j=0,
        population_density_bucket=None,
        cell_density_buckets=(),
        morphology_stratum=MorphologyStratum(
            dominant_zoning_class=None, modal_road_skeleton_class=None
        ),
        coastal_inland_river=None,
        admin_region=None,  # the EU regime: admin division absent on every tile
        sub_c_morphology_class=None,
    )
    d = _tile_conditioning_dict(labels)
    assert "admin_region" in d and "region" not in d
    assert d["admin_region"] is None


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


# --- Task 11 review follow-up: verify_union_manifests (one-source G4 roll-up) ---


def _union_world() -> tuple[dict, dict]:
    """Synthetic G4 roll-up + multiregion holdout: 2 train cities (aaa, bbb),
    1 held-out (heldout1, validated but excluded), 1 unvalidated (dropped)."""
    g4 = {
        "per_city": [
            {"name": "bbb", "validated": True},
            {"name": "aaa", "validated": True},
            {"name": "heldout1", "validated": True},
            {"name": "unvalidated1", "validated": False},
        ]
    }
    holdout = {"holdout_schema_version": "2.0", "held_out_cities": ["heldout1"]}
    return g4, holdout


def test_verify_union_manifests_returns_sorted_cities_when_all_exist(tmp_path, monkeypatch):
    import cfm.data.training.build_shards as bs

    g4, holdout = _union_world()

    def fake_manifest_path(release, city):
        return tmp_path / release / city / "training_manifest.yaml"

    monkeypatch.setattr(bs, "training_manifest_path", fake_manifest_path)
    for city in ("aaa", "bbb"):
        p = fake_manifest_path(_RELEASE, city)
        p.parent.mkdir(parents=True)
        p.write_text(f"region: {city}\n", encoding="utf-8")

    cities = bs.verify_union_manifests(_RELEASE, g4_rollup=g4, holdout_manifest=holdout)
    # sorted train cities: held-out and unvalidated excluded, order deterministic
    assert cities == ["aaa", "bbb"]


def test_verify_union_manifests_raises_naming_all_missing_cities(tmp_path, monkeypatch):
    import cfm.data.training.build_shards as bs

    g4, holdout = _union_world()

    def fake_manifest_path(release, city):
        return tmp_path / release / city / "training_manifest.yaml"

    monkeypatch.setattr(bs, "training_manifest_path", fake_manifest_path)
    # only aaa exists -> bbb must be named in the error; with NEITHER on disk both are named
    with pytest.raises(ValueError) as exc:
        bs.verify_union_manifests(_RELEASE, g4_rollup=g4, holdout_manifest=holdout)
    assert "aaa" in str(exc.value) and "bbb" in str(exc.value)

    p = fake_manifest_path(_RELEASE, "aaa")
    p.parent.mkdir(parents=True)
    p.write_text("region: aaa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bbb"):
        bs.verify_union_manifests(_RELEASE, g4_rollup=g4, holdout_manifest=holdout)


def test_verify_union_manifests_strict_holdout_read_raises(tmp_path, monkeypatch):
    """Fail-closed: a holdout mapping WITHOUT held_out_cities must raise ValueError
    naming the key -- never fall through to train_cities' .get(..., []) and
    silently exclude nothing (same contract as _union_datamodule)."""
    import cfm.data.training.build_shards as bs

    g4, _ = _union_world()
    with pytest.raises(ValueError, match="held_out_cities"):
        bs.verify_union_manifests(
            _RELEASE, g4_rollup=g4, holdout_manifest={"holdout_schema_version": "2.0"}
        )
