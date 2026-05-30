"""Sub-F empirical gate (T14) — per-type RETENTION vs the Halt-4 locked floors.

T14 is the COMPLEMENT of T13, not a competitor. T13 verified round-trip
(position p99.9 / angle p95) + the BP7 cross-tile composite on real Singapore.
The one locked surface T13 does not touch is per-type RETENTION against the
Halt-4 floors (sequence_length_analysis.yaml lock.retention_floors_per_type:
roads 0.9936 / buildings 0.9889 / pois 0.9027 / base 0.9992). T14 fills that
gap and adds the meta-check that the gate reads the OPERATIVE lock, not stale
historical defaults.

DISCIPLINE: every assertion reads the locked value from the operative YAML
source-of-truth (run_empirical_gate.load_locked_*); NO locked number is
hardcoded in a test. The plan's stale hardcoded floors (0.999 / 0.99 — the
PRE-Halt-4 §7.5 defaults, which survive in the YAML as
retention_defaults_per_spec_7_5 for audit) would have created self-perpetuating
staleness: every re-lock would mean hunting down hardcoded test constants.
Reading from the lock means the gate stays current as the source moves (same
shape as content-anchored cites surviving line-number drift).

Retention is computed from sub-C features + the chunked encoder cost + the §7.2
formula stage-4 (sub-E ABSENT) — NOT from sub-E boundary data — so the real-data
gate runs against the present sub-C Singapore cache (test_empirical_gate_slow).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from shapely.geometry import LineString, Point, Polygon

from scripts.sub_f.run_empirical_gate import (
    evaluate_retention,
    load_locked_floors,
)
from tests.data.sub_f.test_pipeline import _feature_row, _write_sub_c_features

# Feature-class ints (sub-C): 0 roads, 1 buildings, 2 pois, 3 base. Match
# run_empirical_gate / sequence_length_analysis.yaml retention_floors_per_type keys.
_FC_ROADS, _FC_BUILDINGS, _FC_POIS, _FC_BASE = 0, 1, 2, 3

# A LineString whose chunked token cost vastly exceeds any plausible budget
# (200 km / 32 m ≈ 6250 dir/mag pairs ≈ 12,500 tokens >> 6016). Not coupled to
# the exact locked budget — just "definitely over".
_OVERSIZED_ROAD = LineString([(0.0, 0.0), (200_000.0, 0.0)])
_SMALL_BUILDING = Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)])
_SMALL_POI = Point(30, 30)
_SMALL_BASE = Polygon([(40, 40), (50, 40), (50, 50), (40, 50), (40, 40)])


def _build_region(tmp_path, rows) -> object:
    region = tmp_path / "sub_c" / "singapore"
    _write_sub_c_features(region / "tile=EPSG3414_i0_j0" / "features.parquet", rows)
    return region


def _all_small_rows() -> list[dict]:
    """One small feature of each type, each in its own under-budget cell."""
    return [
        _feature_row(0, 0, _FC_ROADS, LineString([(10, 10), (30, 10)]), "residential", "r0"),
        _feature_row(1, 1, _FC_BUILDINGS, _SMALL_BUILDING, "yes", "b0"),
        _feature_row(2, 2, _FC_POIS, _SMALL_POI, "cafe", "p0"),
        _feature_row(3, 3, _FC_BASE, _SMALL_BASE, "water", "g0"),
    ]


def test_retention_gate_fails_when_a_type_is_below_its_floor(tmp_path):
    """ADVERSARIAL (built first): a region where ROADS retention collapses to 0
    (the sole road sits alone in an over-budget cell → its cell is alpha-dropped)
    must FAIL the gate, with roads flagged."""
    rows = [
        _feature_row(0, 0, _FC_ROADS, _OVERSIZED_ROAD, "motorway", "huge"),
        _feature_row(1, 1, _FC_BUILDINGS, _SMALL_BUILDING, "yes", "b0"),
        _feature_row(2, 2, _FC_POIS, _SMALL_POI, "cafe", "p0"),
        _feature_row(3, 3, _FC_BASE, _SMALL_BASE, "water", "g0"),
    ]
    region = _build_region(tmp_path, rows)

    result = evaluate_retention(region)

    assert result["all_pass"] is False
    assert result["per_type"][_FC_ROADS]["pass"] is False
    assert result["per_type"][_FC_ROADS]["measured_retention"] == 0.0
    # Other types unaffected (retained → pass), so the failure is roads-specific.
    assert result["per_type"][_FC_BUILDINGS]["pass"] is True


def test_retention_gate_leg_neuter_roads_floor_isolates_the_rule(tmp_path):
    """Rule isolation (T9/T10/T13 standard): no-op ONLY the roads floor → the
    gate now passes → proves the roads-floor check is what fired above, not an
    incidental error. Floors read from the lock; only roads overridden to 0."""
    rows = [
        _feature_row(0, 0, _FC_ROADS, _OVERSIZED_ROAD, "motorway", "huge"),
        _feature_row(1, 1, _FC_BUILDINGS, _SMALL_BUILDING, "yes", "b0"),
        _feature_row(2, 2, _FC_POIS, _SMALL_POI, "cafe", "p0"),
        _feature_row(3, 3, _FC_BASE, _SMALL_BASE, "water", "g0"),
    ]
    region = _build_region(tmp_path, rows)

    real = {fc: d["floor"] for fc, d in load_locked_floors().items()}
    neutered = {**real, _FC_ROADS: 0.0}  # disable ONLY the roads floor
    result = evaluate_retention(region, floors=neutered)

    assert result["all_pass"] is True
    assert result["per_type"][_FC_ROADS]["pass"] is True


def test_retention_gate_passes_when_all_above_floor(tmp_path):
    region = _build_region(tmp_path, _all_small_rows())
    result = evaluate_retention(region)
    assert result["all_pass"] is True
    for fc in (_FC_ROADS, _FC_BUILDINGS, _FC_POIS, _FC_BASE):
        assert result["per_type"][fc]["measured_retention"] == 1.0


def test_gate_uses_operative_floors_not_spec_7_5_defaults():
    """Catch-1 regression guard: the gate must load lock.retention_floors_per_type
    (revisit_2026_05_29), NOT retention_defaults_per_spec_7_5 (historical audit
    trail). The POI floor is the discriminator — operative 0.9027 vs the §7.5
    default 0.99."""
    import yaml

    from scripts.sub_f.run_empirical_gate import _SEQ_LEN_YAML  # path constant

    floors = load_locked_floors()
    sla = yaml.safe_load(_SEQ_LEN_YAML.read_text(encoding="utf-8"))
    operative = sla["lock"]["retention_floors_per_type"]
    historical_defaults = sla["retention_defaults_per_spec_7_5"]

    # The gate's floors are exactly the operative lock block.
    for fc, d in floors.items():
        assert d["floor"] == operative[str(fc)]["floor"]
    # And materially differ from the §7.5 defaults the plan would have hardcoded
    # (POIs: 0.9027 operative vs 0.99 default → > 0.05 apart).
    assert abs(floors[_FC_POIS]["floor"] - historical_defaults["1"]) > 0.05


def test_long_cell_diagnostic_is_two_padding_blocks_below_padded_budget():
    """Lock coherence (no hardcoded value): the long-cell threshold tracks the
    padded budget by the locked action-contract rule (2 padding blocks = 256
    below). Catches a budget move that forgets to re-anchor the diagnostic."""
    import yaml

    from scripts.sub_f.run_empirical_gate import _SEQ_LEN_YAML

    lock = yaml.safe_load(_SEQ_LEN_YAML.read_text(encoding="utf-8"))["lock"]
    padded = lock["elbow_budget_padded_tokens"]
    long_cell = lock["long_cell_diagnostic"]["threshold_tokens"]
    assert long_cell == padded - 256


# ---------------------------------------------------------------------------
# @slow real-data gate: per-type retention on the present sub-C Singapore cache.
# Retention needs sub-C (present) + the §7.2 formula stage-4 — NOT sub-E — so
# unlike T13's tests this one RUNS today and passes (regression-guards the
# floors against encoder/budget drift). Fail-loud if sub-C is absent (T13
# pattern). HALT, never weaken, if any measured retention is below its floor.
# ---------------------------------------------------------------------------

_SUB_C_SINGAPORE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "sub_c"
    / "2026-04-15.0"
    / "singapore"
)


@pytest.mark.slow
def test_retention_floors_hold_on_real_singapore(tmp_path):
    if not (_SUB_C_SINGAPORE / "_SUCCESS").exists():
        pytest.fail(
            f"sub-C Singapore cache missing at {_SUB_C_SINGAPORE} — regenerate sub-C "
            f"(release 2026-04-15.0) before running the T14 retention gate. Fail-loud: "
            f"a silently-skipped retention gate proves nothing."
        )

    result = evaluate_retention(_SUB_C_SINGAPORE)

    # Write the (formula-stage-4) summary locally for inspection; NOT committed.
    # The real golden — recomputed with EMPIRICAL stage-4 — is a close-checklist
    # deliverable for when the sub-E cache regenerates.
    (tmp_path / "layer_singapore_retention_summary.yaml").write_text(
        yaml.safe_dump(result, sort_keys=True), encoding="utf-8"
    )

    for fc, d in sorted(result["per_type"].items()):
        assert d["pass"], (
            f"{d['role']} (fc={fc}) measured retention {d['measured_retention']:.5f} < locked "
            f"floor {d['locked_floor']:.4f} — HALT. This is the recurring pattern's next "
            f"instance, NOT 'close enough': either the floor lock is wrong (a third "
            f"sequence_length re-lock — surface loudly) or the encoder/budget/stage-4 "
            f"drifted (Step-0 catch on the measurement path). Do NOT weaken this assertion."
        )
    assert result["all_pass"]
