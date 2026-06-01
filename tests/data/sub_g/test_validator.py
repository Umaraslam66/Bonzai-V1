from __future__ import annotations

from types import SimpleNamespace

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_g.diagnostics import Diagnostic
from cfm.data.sub_g.validator import _macro_targets, finalize, validate_tile

_VOLATILE = {"run_timestamp": "T", "host": "h", "run_uuid": "u", "sub_g_commit_sha": "s"}


def _macro_row(**kw):
    return SimpleNamespace(
        slot_kind=kw["slot_kind"],
        cell_i=kw.get("cell_i"),
        cell_j=kw.get("cell_j"),
        lower_cell_i=kw.get("lower_cell_i"),
        lower_cell_j=kw.get("lower_cell_j"),
        axis=kw.get("axis"),
        cell_density_bucket=kw.get("cell_density_bucket"),
        road_skeleton_class=kw.get("road_skeleton_class"),
        zoning_class=kw.get("zoning_class"),
    )


def test_macro_targets_splits_cell_and_edge_slots():
    rows = [
        _macro_row(slot_kind=SlotKind.CELL, cell_i=0, cell_j=0, cell_density_bucket=2),
        _macro_row(
            slot_kind=SlotKind.INTERNAL_EDGE,
            lower_cell_i=0,
            lower_cell_j=0,
            axis=0,
            road_skeleton_class=1,
        ),
    ]
    density, skeleton = _macro_targets(rows)
    assert density == {(0, 0): 2}
    assert skeleton == {(0, 0, 0): 1}


def test_validate_tile_clean_empty_tile():
    diags, errors, _nbc = validate_tile(
        tile_id="tile=i0_j0",
        features_by_cell={(0, 0): []},
        area_by_cell={(0, 0): 1000.0},
        crossings=[],
        density_by_cell={(0, 0): 0},  # empty cell -> ratio 0 -> bucket 0
        skeleton_by_edge={},
        cell_contracts={(0, 0): {"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"}},
        tokens_by_cell={(0, 0): []},
    )
    assert diags == []
    assert errors == []


def test_validate_tile_seeded_density_mismatch():
    diags, _errors, _nbc = validate_tile(
        tile_id="tile=i0_j0",
        features_by_cell={(0, 0): []},
        area_by_cell={(0, 0): 1000.0},
        crossings=[],
        density_by_cell={(0, 0): 3},  # claims dense; cell is empty -> mismatch
        skeleton_by_edge={},
        cell_contracts={(0, 0): {}},
        tokens_by_cell={(0, 0): []},
    )
    assert any(d.invariant_name == "density_bucket_matches_footprint" for d in diags)


def _err(core, full=None, angle=1.0):
    return {
        "position_core_m": core,
        "position_full_m": full if full is not None else core,
        "angle_core_deg": angle,
    }


def test_finalize_writes_marker_when_clean(tmp_path):
    res = finalize(
        region="singapore",
        release="2026-04-15.0",
        all_diags=[],
        all_errors=[_err(1.0)],
        output_dir=tmp_path,
        volatile=_VOLATILE,
    )
    assert res.passed is True
    assert res.marker_written is True
    assert (tmp_path / "_PHASE1_VALIDATED").exists()
    assert (tmp_path / "quarantine_report.yaml").exists()
    assert (tmp_path / "_PHASE1_ACCURACY_BASELINE.yaml").exists()
    assert "groups: []" in (tmp_path / "quarantine_report.yaml").read_text()


def test_finalize_withholds_marker_on_quarantine(tmp_path):
    d = Diagnostic(
        "tile=i0_j0", "density_bucket_matches_footprint", "l", 0, "r", 3, "rel", "cite", "sig"
    )
    res = finalize("singapore", "2026-04-15.0", [d], [_err(1.0)], tmp_path, _VOLATILE)
    assert res.passed is False
    assert not (tmp_path / "_PHASE1_VALIDATED").exists()
    # report always written, even on failure (diffable iterations)
    assert (tmp_path / "quarantine_report.yaml").exists()


def test_finalize_sanity_floor_breach_blocks_marker(tmp_path):
    # CORE position p99.9 > 50m -> sanity breach -> no marker even with zero diags.
    res = finalize("singapore", "2026-04-15.0", [], [_err(60.0)], tmp_path, _VOLATILE)
    assert res.sanity_floor_violated is True
    assert res.passed is False
    assert not (tmp_path / "_PHASE1_VALIDATED").exists()


def test_finalize_gates_on_core_not_full(tmp_path):
    """Reviewer 2026-06-01: the floor gates on CORE. A huge FULL (the v1-unencoded
    bref crossing residual) with a small CORE must NOT trip the floor -- else the
    PRD §11 gate could never pass on a real region with crossing roads."""
    res = finalize(
        "singapore", "2026-04-15.0", [], [_err(core=3.0, full=300.0)], tmp_path, _VOLATILE
    )
    assert res.sanity_floor_violated is False
    assert res.passed is True
    assert (tmp_path / "_PHASE1_VALIDATED").exists()
    baseline = (tmp_path / "_PHASE1_ACCURACY_BASELINE.yaml").read_text()
    assert "position_full_p99_9: 300.0" in baseline  # full still reported + visible
    assert "core_excludes" in baseline


def test_finalize_angle_core_breach_blocks_marker(tmp_path):
    res = finalize(
        "singapore", "2026-04-15.0", [], [_err(core=1.0, angle=30.0)], tmp_path, _VOLATILE
    )
    assert res.sanity_floor_violated is True
    assert res.passed is False


def test_finalize_reports_bref_collapse_count_without_blocking(tmp_path):
    """sub-G T11 H3: the OGC-validity gate excludes the v1 outbound-bref placeholder
    collapse by construction identity. The excluded COUNT must be reported in the
    baseline (same crossing roads as the position_full residual) AND must NOT block
    the _PHASE1_VALIDATED marker -- consistent with H1's report-not-gate call."""
    res = finalize(
        "singapore",
        "2026-04-15.0",
        [],
        [_err(1.0)],
        tmp_path,
        _VOLATILE,
        bref_collapse_count=27958,
    )
    assert res.passed is True  # the excluded family does not block the marker
    baseline = (tmp_path / "_PHASE1_ACCURACY_BASELINE.yaml").read_text()
    assert "ogc_bref_collapse_excluded_from_gate: 27958" in baseline
