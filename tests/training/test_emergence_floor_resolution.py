"""Emergence-floor resolution in the scaffold (readiness-closure Task 13, F13/F15).

The floor is a per-region, provenance-bearing artifact (configs/eval/
emergence_floors.yaml), resolved by ``cfg.region`` BEFORE training starts.
Fail-open is CLOSED: a cell-generating run (eval_cells > 0) whose region has no
entry raises at config time — the old ``--emergence-floor`` CLI literal is gone.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import yaml

import scripts.train_scaffold as ts
from cfm.data.training.datamodule import CellExample
from cfm.training.config import ScaffoldConfig


class _Boom(Exception):
    """Sentinel raised by the stubbed _datamodule to stop run_short after resolution."""


# --- (i) resolution against the REAL seeded artifact ---------------------------------


def test_resolves_seeded_singapore_floor_from_real_yaml() -> None:
    floor, prov = ts._resolve_emergence_floor("singapore")
    assert floor == pytest.approx(1.96)
    assert prov["region"] == "singapore"
    assert prov["holdout_density"] == pytest.approx(7.85)
    assert prov["frac"] == 0.25
    assert prov["derivation_regime"] == {
        "cell_length": "full",
        "denominator": "all_nonempty_cells",
    }
    assert "derived_at" in prov


# --- (ii) unknown region raises, naming region + path + the fix ----------------------


def test_unknown_region_raises_naming_region_path_and_fix() -> None:
    with pytest.raises(ValueError) as exc:
        ts._resolve_emergence_floor("atlantis")
    msg = str(exc.value)
    assert "atlantis" in msg
    assert "emergence_floors.yaml" in msg
    assert "measure_emergence_floor.py" in msg


def test_entry_missing_required_key_raises_on_load(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "emergence_floors.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"schema_version": "1.0", "regions": {"krakow": {"floor": 2.0, "frac": 0.25}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ts, "_EMERGENCE_FLOORS_PATH", bad)
    with pytest.raises(ValueError, match="krakow"):
        ts._resolve_emergence_floor("krakow")


def test_scalar_entry_raises_curated_error_not_attributeerror(tmp_path, monkeypatch) -> None:
    # A corrupted entry that is a bare scalar (`singapore: 1.96`) must produce the
    # same curated ValueError (region + path + fix hint), not a bare AttributeError
    # from `entry.keys()`.
    bad = tmp_path / "emergence_floors.yaml"
    bad.write_text(
        yaml.safe_dump({"schema_version": "1.0", "regions": {"singapore": 1.96}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ts, "_EMERGENCE_FLOORS_PATH", bad)
    with pytest.raises(ValueError) as exc:
        ts._resolve_emergence_floor("singapore")
    msg = str(exc.value)
    assert "singapore" in msg
    assert "emergence_floors.yaml" in msg
    assert "measure_emergence_floor.py" in msg  # the fix hint


# --- (iii) run_short: resolution happens BEFORE the datamodule / training ------------


def test_run_short_unknown_region_raises_before_datamodule(monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(ts, "_datamodule", lambda *a, **k: called.append(1))
    # eval_cells > 0 (F15: a config field now) => the run generates, so a floor is required
    cfg = ScaffoldConfig(devices=1, accelerator="cpu", region="atlantis", eval_cells=64)
    with pytest.raises(ValueError, match="atlantis"):
        ts.run_short(cfg, build_shards=False)
    assert called == []  # the failure pre-empts ALL data/training work


def test_run_short_resolves_floor_for_its_region_before_training(monkeypatch) -> None:
    seen: list[str] = []

    def fake_resolve(region: str):
        seen.append(region)
        return 1.96, {"region": region}

    def boom(*a, **k):
        raise _Boom

    monkeypatch.setattr(ts, "_resolve_emergence_floor", fake_resolve)
    monkeypatch.setattr(ts, "_datamodule", boom)
    # eval_cells > 0 (F15: a config field now) => the floor must resolve pre-training
    cfg = ScaffoldConfig(devices=1, accelerator="cpu", region="singapore", eval_cells=64)
    with pytest.raises(_Boom):
        ts.run_short(cfg, build_shards=False)
    assert seen == ["singapore"]  # resolved BEFORE _datamodule raised


def test_run_short_eval_cells_zero_skips_resolution(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        ts, "_resolve_emergence_floor", lambda region: seen.append(region) or (1.0, {})
    )

    def boom(*a, **k):
        raise _Boom

    monkeypatch.setattr(ts, "_datamodule", boom)
    # F15: eval_cells moved onto the config; 0 still means "generate nothing".
    cfg = ScaffoldConfig(devices=1, accelerator="cpu", region="atlantis", eval_cells=0)
    with pytest.raises(_Boom):
        ts.run_short(cfg, build_shards=False)
    assert seen == []  # no cells generated -> no floor needed


# --- the CLI literal is dead ----------------------------------------------------------


def test_cli_emergence_floor_flag_removed() -> None:
    flags = {opt for action in ts._build_parser()._actions for opt in action.option_strings}
    assert "--emergence-floor" not in flags


def test_run_short_signature_has_no_floor_kwarg() -> None:
    assert "emergence_floor_per_cell" not in inspect.signature(ts.run_short).parameters


# --- plumbing: floor + provenance flow through _generate_and_score to slice_eval -----


def test_generate_and_score_passes_floor_and_provenance_to_slice_eval(monkeypatch) -> None:
    captured: dict = {}

    def fake_slice_eval(blocks, geoms, strata, **kwargs):
        captured.update(kwargs)
        return {"n_decoded": len(blocks)}

    monkeypatch.setattr(ts, "slice_eval", fake_slice_eval)
    monkeypatch.setattr(
        ts,
        "generate_cell_tokens",
        lambda model, *, prefix, max_new, seed, char_stats=None: [7, 8, 9],
    )
    monkeypatch.setattr(ts, "split_cell_into_features", lambda tokens: [])
    example = CellExample(
        region="singapore",
        tile_i=0,
        tile_j=0,
        cell_i=0,
        cell_j=0,
        prefix_ids=(101, 102),
        tokens=(1, 2, 3),
        cell_density_bucket=2,
        character_stats=(0.0,) * 7,  # Task 24b required field
    )
    dm = SimpleNamespace(val_cells=[example])
    cfg = ScaffoldConfig(devices=1, accelerator="cpu")
    model = SimpleNamespace(model=object())
    prov = {"region": "singapore", "floor": 1.96}
    ts._generate_and_score(
        model,
        dm,
        cfg,
        n_cells=1,
        max_new=4,
        emergence_floor_per_cell=1.96,
        emergence_floor_provenance=prov,
    )
    assert captured["emergence_floor_per_cell"] == 1.96
    assert captured["emergence_floor_provenance"] == prov
