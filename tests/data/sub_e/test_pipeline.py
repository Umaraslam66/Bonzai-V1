from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_c.enums import FEATURE_CLASS, encode_enum
from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.io import SubCCrossingRow, SubCFeatureRow, SubDMacroCoreRow
from cfm.data.sub_e.pipeline import (
    PipelineConfig,
    _derive_tile_rows,
    derive_region,
)
from tests.data.sub_e._fixtures import _build_synthetic_sub_d_and_sub_c

_ROAD = encode_enum(FEATURE_CLASS, "road")
_BASE = encode_enum(FEATURE_CLASS, "base")
_BUILDING = encode_enum(FEATURE_CLASS, "building")


@pytest.fixture
def synthetic_sub_d_region(tmp_path: Path) -> Path:
    """Build a tiny sub-D-shaped region (2 tiles, valid macro_core) plus sub-C
    crossings/features so sub-E has consistent inputs to read. The fixture
    constructs minimal but valid sub-D output via writers that mirror sub-D's
    schema; sub-E reads it like a real sub-D region.
    """
    return _build_synthetic_sub_d_and_sub_c(tmp_path)


def test_pipeline_happy_path_writes_success_marker(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    derive_region(cfg)
    assert (out_root / "_SUCCESS").exists()
    assert (out_root / "manifest.yaml").exists()


def test_pipeline_aborts_when_sub_d_success_missing(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    (synthetic_sub_d_region / "_SUCCESS").unlink()
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_halts_on_inline_validator_failure(
    tmp_path: Path, synthetic_sub_d_region: Path, monkeypatch
) -> None:
    """Monkey-patch the derivation function to emit a violating row; assert
    pipeline raises and does NOT write _SUCCESS.
    """
    from cfm.data.sub_e import pipeline as pipeline_mod
    from cfm.data.sub_e.validator_inline import InlineValidationError

    def _bad_derive(*args, **kwargs):
        raise InlineValidationError("synthetic violation")

    monkeypatch.setattr(pipeline_mod, "_validate_or_raise", _bad_derive)

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(InlineValidationError):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_lever_3_collapse_uniformly_null_boundary_class(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    import pyarrow.parquet as pq

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=True,
    )
    derive_region(cfg)
    # All on-disk boundary_class_enum values should be null in lever-3 mode.
    for tile_dir in (out_root).glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        values = tbl.column("boundary_class_enum").to_pylist()
        assert all(v is None for v in values), f"non-null in lever-3 at {tile_dir}"


# --- Cycle-2 lock-and-guards: non-road crossings excluded from the vote (§5.1) ---
# These pin spec §5.1:274 ("an edge with only water or rail crossings becomes
# NONE") at the _derive_tile_rows level — the level where the cycle-2 None-overload
# bug lived. derive_boundary_class is correct and untouched; the bug was that the
# pipeline fabricated a None vote for every non-road crossing. The water-only /
# building-only cases FAIL in the pre-fix regime (they derived MINOR_ROAD) and
# PASS post-fix (NONE) — a regime-distinguishing guard, not a vacuous one.


def _active_internal_edge(li: int = 0, lj: int = 0, axis: int = 0) -> SubDMacroCoreRow:
    """One active (scope=0) internal-edge (slot_kind=1) macro_core row — the
    minimum `_derive_tile_rows` needs to emit a single derived edge row."""
    return SubDMacroCoreRow(
        slot_kind=1,
        slot_index=0,
        cell_i=None,
        cell_j=None,
        lower_cell_i=li,
        lower_cell_j=lj,
        axis=axis,
        scope=0,
        zoning_class=None,
        cell_density_bucket=None,
        road_skeleton_class=0,
    )


def _edge_enum(rows: list, li: int = 0, lj: int = 0, axis: int = 0) -> int | None:
    [edge] = [r for r in rows if r.lower_cell_i == li and r.lower_cell_j == lj and r.axis == axis]
    return edge.boundary_class_enum


def test_derive_tile_rows_water_only_edge_derives_none() -> None:
    """§5.1:274 — an active internal edge whose only crossing is a non-road
    (water `base`) feature derives NONE, not MINOR_ROAD. Exact regime the cycle-2
    None-overload bug mis-derived as MINOR_ROAD (pre-fix this asserted RED)."""
    rows = _derive_tile_rows(
        macro_core=[_active_internal_edge()],
        crossings=[SubCCrossingRow(0, 0, 0, "water-1")],
        features=[SubCFeatureRow("water-1", _BASE, "water")],
        lever_3_collapse=False,
    )
    assert _edge_enum(rows) == int(BoundaryClass.NONE)


def test_derive_tile_rows_building_only_edge_derives_none() -> None:
    """Building co-linear crossing (the dominant real subset: 1478 of 2016
    cycle-2 coverage failures) → NONE, per §5.1 non-road exclusion."""
    rows = _derive_tile_rows(
        macro_core=[_active_internal_edge()],
        crossings=[SubCCrossingRow(0, 0, 0, "bldg-1")],
        features=[SubCFeatureRow("bldg-1", _BUILDING, None)],
        lever_3_collapse=False,
    )
    assert _edge_enum(rows) == int(BoundaryClass.NONE)


def test_derive_tile_rows_road_crossing_still_votes() -> None:
    """Regression guard for over-exclusion: a road crossing still derives its
    class — the fix excludes only non-road, never roads."""
    rows = _derive_tile_rows(
        macro_core=[_active_internal_edge()],
        crossings=[SubCCrossingRow(0, 0, 0, "road-1")],
        features=[SubCFeatureRow("road-1", _ROAD, "primary")],
        lever_3_collapse=False,
    )
    assert _edge_enum(rows) == int(BoundaryClass.MAJOR_ROAD)


def test_derive_tile_rows_road_with_null_class_still_minor() -> None:
    """The [None] semantics the fix PRESERVES via membership-test (not .get()):
    a ROAD with null/unknown class_raw still votes → MINOR_ROAD default. None now
    means only 'road, unknown class', never 'non-road'."""
    rows = _derive_tile_rows(
        macro_core=[_active_internal_edge()],
        crossings=[SubCCrossingRow(0, 0, 0, "road-2")],
        features=[SubCFeatureRow("road-2", _ROAD, None)],
        lever_3_collapse=False,
    )
    assert _edge_enum(rows) == int(BoundaryClass.MINOR_ROAD)


def test_derive_tile_rows_absent_feature_crossing_excluded_and_warns(caplog) -> None:
    """Condition #3: a crossing whose source_feature_id is absent from the tile's
    features.parquet is excluded from the vote (edge → NONE) AND surfaced loudly
    via log.warning — not silently dropped (which would be a new vacuous pass)."""
    import logging

    with caplog.at_level(logging.WARNING):
        rows = _derive_tile_rows(
            macro_core=[_active_internal_edge()],
            crossings=[SubCCrossingRow(0, 0, 0, "ghost-1")],
            features=[],
            lever_3_collapse=False,
        )
    assert _edge_enum(rows) == int(BoundaryClass.NONE)
    assert "absent from tile features.parquet" in caplog.text
