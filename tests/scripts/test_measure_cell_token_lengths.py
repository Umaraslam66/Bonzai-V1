"""Tests for scripts/measure_cell_token_lengths.py (readiness-closure Task 15, F13).

The measure script scans a region's sub-F ``tile=*/cells.parquet`` token lengths
and writes per-city stats (p50/p99/p99.9/max, frac over the cell-token budget) to
a deterministic YAML. All tests are synthetic: a tmp_path sub-F tree is built with
the PRODUCTION writer (``cfm.data.sub_f.io.write_cells_parquet``), so the fixture
schema can never drift from what ``read_sub_f_cells`` expects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from cfm.data.sub_f.io import CellRow, write_cells_parquet
from cfm.data.training.datamodule import DEFAULT_MAX_CELL_TOKENS

_REPO = Path(__file__).resolve().parents[2]

_RELEASE = "2026-04-15.0"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "measure_cell_token_lengths", _REPO / "scripts" / "measure_cell_token_lengths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tile(tile_dir: Path, lengths: list[int]) -> None:
    """One synthetic 64-cell tile: the first ``len(lengths)`` cells get token
    sequences of those lengths, the rest stay empty (token_sequence == [])."""
    assert len(lengths) <= 64
    rows = []
    for idx in range(64):
        ci, cj = idx // 8, idx % 8
        toks = [1] * lengths[idx] if idx < len(lengths) else []
        rows.append(
            CellRow(
                cell_i=ci,
                cell_j=cj,
                cell_slot_index=idx,
                token_sequence=toks,
                feature_count=1 if toks else 0,
                provenance_sha256="a" * 64,
            )
        )
    write_cells_parquet(tile_dir / "cells.parquet", rows)


@pytest.fixture()
def sub_f_tree(tmp_path: Path):
    """Synthetic 2-city sub-F tree + an injectable root fn (mirrors sub_f_region_dir).

    aaa: one tile, non-empty lengths [10, 20, 30, 40] (hand-computable stats).
    bbb: TWO tiles ([5, 5] and [7]) — proves multi-tile aggregation.
    """
    root = tmp_path / "sub_f"
    _write_tile(root / _RELEASE / "aaa" / "tile=EPSG3414_i0_j0", [10, 20, 30, 40])
    _write_tile(root / _RELEASE / "bbb" / "tile=EPSG3414_i0_j0", [5, 5])
    _write_tile(root / _RELEASE / "bbb" / "tile=EPSG3414_i0_j1", [7])

    def root_fn(release: str, region: str) -> Path:
        return root / release / region

    return root_fn


def test_measure_city_stats_on_known_lengths(sub_f_tree) -> None:
    mod = _load_module()
    stats = mod.measure_city(_RELEASE, "aaa", sub_f_root_fn=sub_f_tree, budget=30)
    # lengths [10, 20, 30, 40]; numpy linear-interpolation percentiles:
    # p50 = 25.0; p99 = 30 + 0.97*10 = 39.7; p99.9 = 30 + 0.997*10 = 39.97
    assert stats["n_cells"] == 4  # empty cells are NOT counted
    assert stats["p50"] == pytest.approx(25.0)
    assert stats["p99"] == pytest.approx(39.7)
    assert stats["p99_9"] == pytest.approx(39.97)
    assert stats["max"] == 40
    # STRICT > budget (matches flatten's `n > max_cell_tokens`): 30 == budget is NOT over
    assert stats["frac_over_budget"] == pytest.approx(0.25)
    assert stats["budget"] == 30


def test_measure_city_aggregates_across_tiles(sub_f_tree) -> None:
    mod = _load_module()
    stats = mod.measure_city(_RELEASE, "bbb", sub_f_root_fn=sub_f_tree, budget=30)
    assert stats["n_cells"] == 3  # [5, 5] + [7] across the two tiles
    assert stats["max"] == 7
    assert stats["frac_over_budget"] == 0.0


def test_frac_over_budget_is_strictly_greater_than(sub_f_tree) -> None:
    """budget == max length -> nothing is over (strict >, the flatten contract)."""
    mod = _load_module()
    stats = mod.measure_city(_RELEASE, "aaa", sub_f_root_fn=sub_f_tree, budget=40)
    assert stats["frac_over_budget"] == 0.0


def test_missing_region_dir_raises_loud(sub_f_tree, tmp_path: Path) -> None:
    """A region with NO sub-F data must fail closed, naming the dir — never a
    silent skip that under-reports the corpus."""
    mod = _load_module()
    with pytest.raises(FileNotFoundError, match="nope"):
        mod.measure_city(_RELEASE, "nope", sub_f_root_fn=sub_f_tree)


def test_measure_and_write_yaml_is_deterministic_and_round_trips(
    sub_f_tree, tmp_path: Path
) -> None:
    mod = _load_module()
    out = tmp_path / "cell_token_lengths.yaml"
    data = mod.measure_and_write(
        _RELEASE,
        ["aaa", "bbb"],
        out_path=out,
        sub_f_root_fn=sub_f_tree,
        budget=30,
        git_sha="abc123",
    )
    first_bytes = out.read_bytes()
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded["budget"] == 30
    assert loaded["derived_at"] == "abc123"  # git sha, never a timestamp
    assert set(loaded["cities"]) == {"aaa", "bbb"}
    assert loaded["cities"]["aaa"]["max"] == 40
    assert loaded["cities"] == data["cities"]
    # byte-determinism: a second write is identical
    mod.measure_and_write(
        _RELEASE,
        ["aaa", "bbb"],
        out_path=out,
        sub_f_root_fn=sub_f_tree,
        budget=30,
        git_sha="abc123",
    )
    assert out.read_bytes() == first_bytes


def test_cli_halt_threshold_exits_nonzero_naming_offending_city(
    sub_f_tree, tmp_path: Path, monkeypatch, capsys
) -> None:
    """The F13 action contract travels with the tool: any city whose frac over the
    DEFAULT budget exceeds --halt-threshold (default 0.005) makes the CLI exit
    nonzero, naming the city."""
    mod = _load_module()
    # city "ccc": one cell over the DEFAULT budget -> frac 0.5 >> 0.005
    base = tmp_path / "halt_tree"
    _write_tile(base / _RELEASE / "ccc" / "tile=EPSG3414_i0_j0", [100, DEFAULT_MAX_CELL_TOKENS + 1])
    _write_tile(base / _RELEASE / "ddd" / "tile=EPSG3414_i0_j0", [100, 200])
    monkeypatch.setattr(mod, "sub_f_region_dir", lambda release, region: base / release / region)
    monkeypatch.setattr(mod, "_git_sha", lambda: "clisha")
    out = tmp_path / "out.yaml"

    rc = mod.main(["--release", _RELEASE, "--regions", "ccc", "ddd", "--out", str(out)])
    assert rc != 0
    captured = capsys.readouterr().out
    halt_lines = [ln for ln in captured.splitlines() if ln.startswith("HALT")]
    assert halt_lines, f"expected a HALT line naming offenders, got: {captured!r}"
    assert "ccc" in halt_lines[0]
    assert "ddd" not in halt_lines[0]  # clean city is NOT named an offender
    # the YAML is still written (the stats are the evidence the halt cites)
    assert yaml.safe_load(out.read_text())["cities"]["ccc"]["frac_over_budget"] == 0.5


def test_cli_clean_corpus_exits_zero(sub_f_tree, tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    base = tmp_path / "clean_tree"
    _write_tile(base / _RELEASE / "ddd" / "tile=EPSG3414_i0_j0", [100, 200])
    monkeypatch.setattr(mod, "sub_f_region_dir", lambda release, region: base / release / region)
    monkeypatch.setattr(mod, "_git_sha", lambda: "clisha")
    out = tmp_path / "out.yaml"
    rc = mod.main(["--release", _RELEASE, "--regions", "ddd", "--out", str(out)])
    assert rc == 0
    assert yaml.safe_load(out.read_text())["cities"]["ddd"]["n_cells"] == 2
