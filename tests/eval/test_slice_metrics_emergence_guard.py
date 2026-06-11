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


# --- Task 13 (readiness-closure, F13/F15): floor provenance travels with the metrics ---


def test_slice_eval_carries_floor_provenance_through() -> None:
    prov = {
        "region": "krakow",
        "floor": 2.0,
        "holdout_density": 8.0,
        "frac": 0.25,
        "derived_at": "abc123",
        "derivation_regime": {"cell_length": "full", "denominator": "all_nonempty_cells"},
    }
    blocks = [[1, 2, 3]]
    geoms = [{"type": "LineString", "coordinates": [[0, 0], [1, 0]]}]
    out = slice_eval(
        blocks,
        geoms,
        [0],
        n_cells=50,
        emergence_floor_per_cell=2.0,
        emergence_floor_provenance=prov,
    )
    assert out["emergence_floor_provenance"] == prov
    # the denominator convention is part of the recorded provenance, not folklore
    assert out["emergence_floor_provenance"]["derivation_regime"]["denominator"] == (
        "all_nonempty_cells"
    )


def test_slice_eval_provenance_key_present_but_none_when_not_given() -> None:
    # present-but-None (report-stable): every metrics dict has the key.
    out = slice_eval([[1]], [{"type": "LineString", "coordinates": [[0, 0], [1, 0]]}], [0])
    assert "emergence_floor_provenance" in out
    assert out["emergence_floor_provenance"] is None
