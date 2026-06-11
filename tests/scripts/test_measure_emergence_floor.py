"""Tests for scripts/measure_emergence_floor.py (readiness-closure Task 13, F13/F15).

The measure script computes a region's emergence floor from the REAL holdout
density (``holdout_polygons_per_active_cell``) and writes a provenance-bearing
entry into configs/eval/emergence_floors.yaml. All tests are synthetic: the
density fn is injected (no real sub-F tile data, no Leonardo).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]

_RELEASE = "2026-04-15.0"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "measure_emergence_floor", _REPO / "scripts" / "measure_emergence_floor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Derived from the writer's constant — deliberately NOT a third hand-copied literal.
# Writer/resolver equality is pinned by test_writer_and_resolver_required_key_sets_match.
_REQUIRED_KEYS = frozenset(_load_module().REQUIRED_ENTRY_KEYS)


def _fake_density(*, release: str, region: str) -> float:
    assert release == _RELEASE
    return {"krakow": 8.0, "valencia": 4.0}[region]


def test_writer_and_resolver_required_key_sets_match() -> None:
    """Sync guard: the writer's REQUIRED_ENTRY_KEYS and the resolver's
    _FLOOR_ENTRY_REQUIRED_KEYS (scripts/train_scaffold.py) are duplicated constants
    (script file vs importable module); this pins them to the same set so a schema
    change in one cannot silently skew the other."""
    import scripts.train_scaffold as ts

    mod = _load_module()
    assert set(mod.REQUIRED_ENTRY_KEYS) == set(ts._FLOOR_ENTRY_REQUIRED_KEYS)


def test_writes_entry_with_full_schema_and_frac_times_density_floor(tmp_path) -> None:
    mod = _load_module()
    floors = tmp_path / "emergence_floors.yaml"
    entry = mod.measure_and_write(
        _RELEASE, "krakow", floors_path=floors, density_fn=_fake_density, git_sha="abc123"
    )
    assert entry["floor"] == pytest.approx(0.25 * 8.0)
    assert entry["holdout_density"] == pytest.approx(8.0)
    assert entry["frac"] == 0.25
    assert entry["derived_at"] == "abc123"
    assert entry["derivation_regime"] == {
        "cell_length": "full",
        "denominator": "all_nonempty_cells",
    }
    assert set(entry) == _REQUIRED_KEYS


def test_round_trips_through_yaml(tmp_path) -> None:
    mod = _load_module()
    floors = tmp_path / "emergence_floors.yaml"
    entry = mod.measure_and_write(
        _RELEASE, "krakow", floors_path=floors, density_fn=_fake_density, git_sha="abc123"
    )
    data = yaml.safe_load(floors.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["regions"]["krakow"] == entry


def test_second_region_preserves_first_entry(tmp_path) -> None:
    mod = _load_module()
    floors = tmp_path / "emergence_floors.yaml"
    first = mod.measure_and_write(
        _RELEASE, "krakow", floors_path=floors, density_fn=_fake_density, git_sha="sha1"
    )
    second = mod.measure_and_write(
        _RELEASE, "valencia", floors_path=floors, density_fn=_fake_density, git_sha="sha2"
    )
    data = yaml.safe_load(floors.read_text(encoding="utf-8"))
    assert data["regions"]["krakow"] == first
    assert data["regions"]["valencia"] == second
    assert second["floor"] == pytest.approx(0.25 * 4.0)


def test_rewrite_replaces_the_region_entry(tmp_path) -> None:
    mod = _load_module()
    floors = tmp_path / "emergence_floors.yaml"
    mod.measure_and_write(
        _RELEASE, "krakow", floors_path=floors, density_fn=_fake_density, git_sha="old"
    )
    entry = mod.measure_and_write(
        _RELEASE, "krakow", floors_path=floors, density_fn=_fake_density, git_sha="new"
    )
    data = yaml.safe_load(floors.read_text(encoding="utf-8"))
    assert data["regions"]["krakow"] == entry
    assert data["regions"]["krakow"]["derived_at"] == "new"


def test_writer_refuses_when_existing_entry_misses_required_key(tmp_path) -> None:
    # Schema validation on write: a corrupt pre-existing entry (missing keys) must
    # refuse the write rather than silently re-serialize a broken artifact.
    mod = _load_module()
    floors = tmp_path / "emergence_floors.yaml"
    floors.write_text(
        yaml.safe_dump({"schema_version": "1.0", "regions": {"krakow": {"floor": 2.0}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="krakow"):
        mod.measure_and_write(
            _RELEASE, "valencia", floors_path=floors, density_fn=_fake_density, git_sha="x"
        )


def test_cli_main_writes_entry(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "holdout_polygons_per_active_cell", _fake_density)
    monkeypatch.setattr(mod, "_git_sha", lambda: "clisha")
    floors = tmp_path / "emergence_floors.yaml"
    rc = mod.main(
        [
            "--release",
            _RELEASE,
            "--region",
            "valencia",
            "--floors-path",
            str(floors),
            "--frac",
            "0.25",
        ]
    )
    assert rc == 0
    data = yaml.safe_load(floors.read_text(encoding="utf-8"))
    assert data["regions"]["valencia"]["floor"] == pytest.approx(1.0)
    assert data["regions"]["valencia"]["derived_at"] == "clisha"
