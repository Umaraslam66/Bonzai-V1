"""Task 8 — tests for the multi-region train-shard build RUNNER.

The runner (``scripts/build_multiregion_train_shards.py``) is a thin driver over the
LOCKED build API: it resolves the train cities via ``train_cities`` (validated minus
held-out), builds each city's per-region ``training_manifest.yaml`` via
``build_training_shards``, and then VERIFIES the end-state by RE-READING every manifest
from disk (never trusting the build call's return value). It also asserts a NEGATIVE
end-state: no held-out city manifest was created by the run.

All tests are SYNTHETIC — no real sub-D/sub-F corpus. We monkeypatch the per-city build
and the path helpers so every write lands under ``tmp_path``. The fixtures mirror the
existing multi-region test (``test_build_shards_multiregion.py``):
``_synthetic_g4_rollup`` (validated ``prague``/``barcelona`` + the 4 held-out validated
+ 2 unvalidated) and ``_synthetic_multiregion_holdout`` (the 4 held-out cities).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# Load the runner module by file path (scripts/ is not an importable package root).
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_multiregion_train_shards.py"
_spec = importlib.util.spec_from_file_location("build_multiregion_train_shards", _SCRIPT)
assert _spec is not None and _spec.loader is not None
R = importlib.util.module_from_spec(_spec)
sys.modules["build_multiregion_train_shards"] = R
_spec.loader.exec_module(R)

_RELEASE = "2026-04-15.0"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (mirror tests/data/training/test_build_shards_multiregion.py)
# --------------------------------------------------------------------------- #
def _synthetic_g4_rollup() -> dict:
    """Mix of validated / unvalidated, including the 4 held-out (validated: True)."""
    return {
        "per_city": [
            {"name": "prague", "crs": "EPSG:25833", "validated": True},
            {"name": "barcelona", "crs": "EPSG:25831", "validated": True},
            {"name": "munich", "crs": "EPSG:25832", "validated": True},  # HELD-OUT
            {"name": "glasgow", "crs": "EPSG:25830", "validated": True},  # HELD-OUT
            {"name": "krakow", "crs": "EPSG:25834", "validated": True},  # HELD-OUT
            {"name": "eisenhuttenstadt", "crs": "EPSG:25833", "validated": True},  # HELD-OUT
            {"name": "half_baked", "crs": "EPSG:25832", "validated": False},  # UNVALIDATED
            {"name": "broken_city", "crs": "EPSG:25831", "validated": False},  # UNVALIDATED
        ]
    }


def _synthetic_multiregion_holdout() -> dict:
    """Schema-2.0 multiregion holdout: ``held_out_cities`` is the whole-city selector."""
    return {
        "manifest_schema_version": "2.0",
        "held_out_cities": ["eisenhuttenstadt", "glasgow", "krakow", "munich"],
        "regions": {
            c: {"tiles": [{"tile_i": 0, "tile_j": 0}]}
            for c in ("eisenhuttenstadt", "glasgow", "krakow", "munich")
        },
    }


# --------------------------------------------------------------------------- #
# Path + build monkeypatches: every write lands under tmp_path; the fake build
# writes a real schema-1.0 per-city manifest (region=city, n_training_tiles=len(city)).
# --------------------------------------------------------------------------- #
def _install_tmp_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    def region_dir(release: str, region: str) -> Path:
        return root / release / region

    def manifest_path(release: str, region: str) -> Path:
        return region_dir(release, region) / "training_manifest.yaml"

    # The runner references these names directly (imported into its namespace).
    monkeypatch.setattr(R, "training_region_dir", region_dir)
    monkeypatch.setattr(R, "training_manifest_path", manifest_path)


def _make_fake_build(monkeypatch, also_writes_heldout: str | None = None):
    """Return a fake ``build_training_shards`` that writes a real per-city manifest.

    ``n_training_tiles`` is a deterministic per-city count (``len(city)``). If
    ``also_writes_heldout`` is set, the fake ALSO writes a stray manifest for that
    (held-out) city — used to prove the negative end-state check bites.
    """

    def fake_build(release: str, region: str, *, out_dir: Path | None = None):
        d = R.training_region_dir(release, region)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_schema_version": "1.0",
            "release": release,
            "region": region,
            "n_training_tiles": len(region),
            "tiles": [],
        }
        R.training_manifest_path(release, region).write_text(
            yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
        )
        if also_writes_heldout is not None:
            h = also_writes_heldout
            hd = R.training_region_dir(release, h)
            hd.mkdir(parents=True, exist_ok=True)
            R.training_manifest_path(release, h).write_text(
                yaml.safe_dump(
                    {
                        "manifest_schema_version": "1.0",
                        "release": release,
                        "region": h,
                        "n_training_tiles": len(h),
                        "tiles": [],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return []

    monkeypatch.setattr(R, "build_training_shards", fake_build)
    return fake_build


# =========================================================================== #
# TEST 1 — builds exactly the train cities; held-out excluded
# =========================================================================== #
def test_builds_exactly_train_cities_and_excludes_heldout(tmp_path, monkeypatch):
    _install_tmp_paths(monkeypatch, tmp_path)
    _make_fake_build(monkeypatch)

    summary = R.build_all_train_cities(
        _RELEASE,
        g4_rollup=_synthetic_g4_rollup(),
        holdout_manifest=_synthetic_multiregion_holdout(),
    )

    assert summary["train_cities"] == ["barcelona", "prague"]  # sorted
    assert summary["n_train_cities"] == 2

    # Exactly 2 manifests on disk — and exactly for the train cities.
    built = sorted(p.parent.name for p in (tmp_path / _RELEASE).glob("*/training_manifest.yaml"))
    assert built == ["barcelona", "prague"]

    # NO manifest for any held-out city.
    for held in ("eisenhuttenstadt", "glasgow", "krakow", "munich"):
        assert not R.training_manifest_path(_RELEASE, held).exists()

    # Deterministic per-city tile counts (len of city name) summed.
    assert summary["total_training_tiles"] == len("barcelona") + len("prague")
    assert summary["per_city"] == {"barcelona": len("barcelona"), "prague": len("prague")}
    assert summary["held_out_cities"] == ["eisenhuttenstadt", "glasgow", "krakow", "munich"]


# =========================================================================== #
# TEST 2 — structural disjoint guard is non-vacuous (RED on a leak)
# =========================================================================== #
def test_structural_guard_fires_if_heldout_leaks(tmp_path, monkeypatch):
    _install_tmp_paths(monkeypatch, tmp_path)
    _make_fake_build(monkeypatch)

    # Force train_cities (as referenced by the runner) to LEAK a held-out city.
    monkeypatch.setattr(R, "train_cities", lambda *a, **k: ["barcelona", "munich", "prague"])

    with pytest.raises(AssertionError, match="munich"):
        R.build_all_train_cities(
            _RELEASE,
            g4_rollup=_synthetic_g4_rollup(),
            holdout_manifest=_synthetic_multiregion_holdout(),
        )


# =========================================================================== #
# TEST 3 — negative end-state detects a stray held-out manifest
# =========================================================================== #
def test_negative_endstate_detects_stray_heldout_manifest(tmp_path, monkeypatch):
    """The build (buggily) also writes a ``munich`` (held-out) manifest mid-run; the
    negative end-state check must FAIL loud because a held-out manifest APPEARED that
    was not present in the pre-snapshot."""
    _install_tmp_paths(monkeypatch, tmp_path)
    _make_fake_build(monkeypatch, also_writes_heldout="munich")

    with pytest.raises(AssertionError, match="munich"):
        R.build_all_train_cities(
            _RELEASE,
            g4_rollup=_synthetic_g4_rollup(),
            holdout_manifest=_synthetic_multiregion_holdout(),
        )


def test_negative_endstate_tolerates_preexisting_heldout_manifest(tmp_path, monkeypatch):
    """A held-out manifest that EXISTED BEFORE the run (pre-snapshot true) is tolerated
    — the check flags only a NEWLY-created one. This proves the snapshot is a delta
    check, not a blanket "no held-out manifest may exist" check."""
    _install_tmp_paths(monkeypatch, tmp_path)
    _make_fake_build(monkeypatch)

    # Pre-create a munich (held-out) manifest BEFORE the run.
    md = R.training_region_dir(_RELEASE, "munich")
    md.mkdir(parents=True, exist_ok=True)
    R.training_manifest_path(_RELEASE, "munich").write_text(
        yaml.safe_dump(
            {
                "manifest_schema_version": "1.0",
                "release": _RELEASE,
                "region": "munich",
                "n_training_tiles": 1,
                "tiles": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Run must NOT raise on the pre-existing held-out manifest.
    summary = R.build_all_train_cities(
        _RELEASE,
        g4_rollup=_synthetic_g4_rollup(),
        holdout_manifest=_synthetic_multiregion_holdout(),
    )
    assert summary["n_train_cities"] == 2


# =========================================================================== #
# TEST 4 — report YAML is written and byte-deterministic across two runs
# =========================================================================== #
def test_report_yaml_is_written_and_deterministic(tmp_path, monkeypatch):
    _install_tmp_paths(monkeypatch, tmp_path)
    _make_fake_build(monkeypatch)

    out1 = tmp_path / "r1.yaml"
    out2 = tmp_path / "r2.yaml"

    R.build_all_train_cities(
        _RELEASE,
        g4_rollup=_synthetic_g4_rollup(),
        holdout_manifest=_synthetic_multiregion_holdout(),
        report_out=out1,
    )
    R.build_all_train_cities(
        _RELEASE,
        g4_rollup=_synthetic_g4_rollup(),
        holdout_manifest=_synthetic_multiregion_holdout(),
        report_out=out2,
    )

    assert out1.exists() and out2.exists()
    assert out1.read_bytes() == out2.read_bytes()  # byte-identical -> deterministic
