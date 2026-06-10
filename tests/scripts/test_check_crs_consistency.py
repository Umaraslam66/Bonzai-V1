"""Tests for scripts/check_crs_consistency.py (Task 12, F2).

Three-way CRS consistency: region config ``projected_crs`` == sub-D manifest
``region_crs`` == on-disk ``tile=EPSG..._i_j`` dir labels (== holdout manifest
per-region ``crs`` where present), plus a meters guard (projected allowlist,
explicit geographic reject-set).

All tests drive the import-testable core against a synthetic tmp_path tree —
no real data, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_crs_consistency", _REPO / "scripts" / "check_crs_consistency.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic fixture tree: 2 fake regions (alpha=EPSG:3414, beta=EPSG:25833).
# ---------------------------------------------------------------------------

_RELEASE = "2026-04-15.0"


def _write_region_config(config_root: Path, region: str, projected_crs: str) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / f"{region}.yaml").write_text(
        yaml.safe_dump({"name": region, "projected_crs": projected_crs}),
        encoding="utf-8",
    )


def _write_sub_d_region(
    sub_d_root: Path,
    region: str,
    region_crs: str,
    tile_labels: list[str],
) -> None:
    rdir = sub_d_root / _RELEASE / region
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "manifest.yaml").write_text(
        yaml.safe_dump({"region": region, "region_crs": region_crs, "tiles": []}),
        encoding="utf-8",
    )
    for k, label in enumerate(tile_labels):
        (rdir / f"tile={label}_i{k}_j{k}").mkdir()


def _make_tree(tmp_path: Path) -> dict:
    """All-consistent 2-region tree. Returns the kwargs for the core check."""
    config_root = tmp_path / "configs" / "regions"
    sub_d_root = tmp_path / "sub_d"
    _write_region_config(config_root, "alpha", "EPSG:3414")
    _write_region_config(config_root, "beta", "EPSG:25833")
    _write_sub_d_region(sub_d_root, "alpha", "EPSG:3414", ["EPSG3414", "EPSG3414"])
    _write_sub_d_region(sub_d_root, "beta", "EPSG:25833", ["EPSG25833"])
    mr_holdout = {"regions": {"beta": {"crs": "EPSG:25833", "tiles": []}}}
    return {
        "config_root": config_root,
        "sub_d_root_fn": lambda release, region: sub_d_root / release / region,
        "sg_holdout": None,
        "mr_holdout": mr_holdout,
    }


# ---------------------------------------------------------------------------
# (i) all-consistent -> PASS, exit 0, report YAML written + re-readable
# ---------------------------------------------------------------------------


def test_all_consistent_passes_and_writes_report(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    report = tmp_path / "report.yaml"

    rc, summary = mod.run_check(_RELEASE, ["alpha", "beta"], report=report, **kwargs)

    assert rc == 0
    assert summary["n_regions"] == 2
    assert summary["n_pass"] == 2
    assert summary["n_fail"] == 0
    for region in ("alpha", "beta"):
        v = summary["per_region"][region]
        assert v["verdict"] == "PASS"
        assert v["diffs"] == []
        assert v["consistent"] is True
        assert v["projected_ok"] is True

    # alpha appears in NO holdout manifest -> leg skipped, recorded as null.
    assert summary["per_region"]["alpha"]["holdout_crs"] is None
    # beta's crs comes from the multiregion manifest.
    assert summary["per_region"]["beta"]["holdout_crs"] == "EPSG:25833"

    # Report is written and round-trips with per-region verdicts == 2.
    assert report.exists()
    loaded = yaml.safe_load(report.read_text(encoding="utf-8"))
    assert loaded["release"] == _RELEASE
    assert len(loaded["per_region"]) == 2
    assert loaded["per_region"]["alpha"]["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# (ii) sub-D region_crs mismatch -> FAIL, mismatch named in diffs
# ---------------------------------------------------------------------------


def test_sub_d_region_crs_mismatch_fails_named(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    # Corrupt alpha's sub-D manifest: region_crs disagrees with the config.
    mpath = kwargs["sub_d_root_fn"](_RELEASE, "alpha") / "manifest.yaml"
    m = yaml.safe_load(mpath.read_text(encoding="utf-8"))
    m["region_crs"] = "EPSG:25832"
    mpath.write_text(yaml.safe_dump(m), encoding="utf-8")

    rc, summary = mod.run_check(_RELEASE, ["alpha", "beta"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["alpha"]
    assert v["verdict"] == "FAIL"
    joined = " ".join(v["diffs"])
    assert "EPSG:3414" in joined and "EPSG:25832" in joined
    assert "region_crs" in joined
    # beta untouched -> still PASS.
    assert summary["per_region"]["beta"]["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# (iii) rogue tile-dir label -> FAIL naming BOTH labels
# ---------------------------------------------------------------------------


def test_rogue_tile_dir_label_fails_naming_both(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    rogue = kwargs["sub_d_root_fn"](_RELEASE, "alpha") / "tile=EPSG4326_i0_j99"
    rogue.mkdir()

    rc, summary = mod.run_check(_RELEASE, ["alpha"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["alpha"]
    assert v["verdict"] == "FAIL"
    joined = " ".join(v["diffs"])
    assert "EPSG3414" in joined and "EPSG4326" in joined
    assert sorted(v["tile_dir_labels"]) == ["EPSG3414", "EPSG4326"]


# ---------------------------------------------------------------------------
# (iv) geographic CRS in config -> FAIL with the meters-guard diff
# ---------------------------------------------------------------------------


def test_geographic_crs_in_config_fails_meters_guard(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    # Region whose config, manifest and tile dirs all agree on EPSG:4326 —
    # internally consistent, so ONLY the meters guard can catch it.
    _write_region_config(kwargs["config_root"], "gamma", "EPSG:4326")
    _write_sub_d_region(
        kwargs["sub_d_root_fn"](_RELEASE, "gamma").parents[1],
        "gamma",
        "EPSG:4326",
        ["EPSG4326"],
    )

    rc, summary = mod.run_check(_RELEASE, ["gamma"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["gamma"]
    assert v["verdict"] == "FAIL"
    assert v["projected_ok"] is False
    assert any("geographic" in d.lower() for d in v["diffs"])


def test_unknown_projected_crs_outside_allowlist_fails(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    # EPSG:32648 is projected/meters but NOT in the region-config allowlist:
    # the pin is "exactly the CRS values our configs declare", so it must fail.
    _write_region_config(kwargs["config_root"], "delta", "EPSG:32648")
    _write_sub_d_region(
        kwargs["sub_d_root_fn"](_RELEASE, "delta").parents[1],
        "delta",
        "EPSG:32648",
        ["EPSG32648"],
    )

    rc, summary = mod.run_check(_RELEASE, ["delta"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["delta"]
    assert v["projected_ok"] is False
    assert any("allowlist" in d for d in v["diffs"])


# ---------------------------------------------------------------------------
# (v) holdout crs present+disagreeing -> FAIL; absent -> skipped, still PASS
# ---------------------------------------------------------------------------


def test_holdout_crs_disagreement_fails(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    kwargs["mr_holdout"] = {"regions": {"beta": {"crs": "EPSG:25832", "tiles": []}}}

    rc, summary = mod.run_check(_RELEASE, ["beta"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["beta"]
    assert v["verdict"] == "FAIL"
    joined = " ".join(v["diffs"])
    assert "holdout" in joined and "EPSG:25832" in joined and "EPSG:25833" in joined


def test_holdout_crs_absent_is_skipped_not_failed(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    # SG-style manifest entry with NO crs key (the frozen Singapore shape).
    kwargs["mr_holdout"] = None
    kwargs["sg_holdout"] = {"regions": {"alpha": {"tiles": []}}}

    rc, summary = mod.run_check(_RELEASE, ["alpha"], report=None, **kwargs)

    assert rc == 0
    v = summary["per_region"]["alpha"]
    assert v["verdict"] == "PASS"
    assert v["holdout_crs"] is None


# ---------------------------------------------------------------------------
# Mandatory legs: missing sub-D manifest / zero tile dirs are FAIL, not skip
# ---------------------------------------------------------------------------


def test_missing_sub_d_manifest_is_fail(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    (kwargs["sub_d_root_fn"](_RELEASE, "alpha") / "manifest.yaml").unlink()

    rc, summary = mod.run_check(_RELEASE, ["alpha"], report=None, **kwargs)

    assert rc != 0
    assert summary["per_region"]["alpha"]["verdict"] == "FAIL"


def test_zero_tile_dirs_is_fail(tmp_path):
    mod = _load_module()
    kwargs = _make_tree(tmp_path)
    rdir = kwargs["sub_d_root_fn"](_RELEASE, "beta")
    for d in rdir.iterdir():
        if d.is_dir() and d.name.startswith("tile="):
            d.rmdir()

    rc, summary = mod.run_check(_RELEASE, ["beta"], report=None, **kwargs)

    assert rc != 0
    v = summary["per_region"]["beta"]
    assert v["verdict"] == "FAIL"
    assert v["tile_dir_labels"] == []
