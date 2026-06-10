"""§2 emergence floor-score guard in slice_eval (Phase-2 bake-off Task 2)."""

from __future__ import annotations

from cfm.eval.slice_metrics import EmergenceVerdict, emergence_verdict, slice_eval


def test_roads_only_run_is_floored_not_a_vacuous_pass() -> None:
    # ogc_valid_rate=1.0 but zero polygons across many cells -> ROADS_ONLY (floor), NOT a pass
    assert (
        emergence_verdict(n_polygons=0, n_cells=110, floor_per_cell=1.0)
        is EmergenceVerdict.ROADS_ONLY
    )


def test_run_clearing_the_floor_is_scoreable() -> None:
    assert (
        emergence_verdict(n_polygons=200, n_cells=110, floor_per_cell=1.0)
        is EmergenceVerdict.SCOREABLE
    )


def test_guard_distinguishes_regimes_floor_keys_on_density_not_validity() -> None:
    # Two runs with identical ogc_valid_rate=1.0 diverge ONLY on emergence density.
    assert (
        emergence_verdict(n_polygons=0, n_cells=110, floor_per_cell=1.0)
        is EmergenceVerdict.ROADS_ONLY
    )
    assert (
        emergence_verdict(n_polygons=300, n_cells=110, floor_per_cell=1.0)
        is EmergenceVerdict.SCOREABLE
    )


def test_slice_eval_flags_roads_only_when_below_floor() -> None:
    # A roads-only decoded set (only a LineString, zero polygons) across 50 cells.
    blocks = [[1, 2, 3]]
    geoms = [{"type": "LineString", "coordinates": [[0, 0], [1, 0]]}]
    out = slice_eval(blocks, geoms, [0], n_cells=50, emergence_floor_per_cell=1.0)
    assert out["emergence_verdict"] == EmergenceVerdict.ROADS_ONLY.value
    assert out["building_metrics_floored"] is True  # ogc_valid_rate must NOT read as a good score


def test_slice_eval_verdict_absent_when_no_emergence_inputs() -> None:
    # Backward-compatible: callers that do not pass n_cells get no verdict (None).
    out = slice_eval([[1]], [{"type": "LineString", "coordinates": [[0, 0], [1, 0]]}], [0])
    assert out["emergence_verdict"] is None
    assert out["building_metrics_floored"] is False
