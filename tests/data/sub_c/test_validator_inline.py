"""Task 11 tests: validate_tile_inline — 10 named invariants per spec §12.1.

Named tests per plan Task 11:
- test_inline_validator_passes_on_clean_torture_tile_output (Layer-2, deferred to Task 16)
- test_inline_invariant_1_schema_correctness_diagnostic
- test_inline_invariant_2_bbox_matches_wkb_diagnostic
- test_inline_invariant_3_geometry_type_matches_wkb_diagnostic
- test_inline_invariant_4_crossings_features_source_feature_id_diagnostic
- test_inline_invariant_5_water_fraction_bounds_combined_with_nan_check_single_pass
- test_inline_invariant_6_kept_cell_rule_diagnostic
- test_inline_invariant_7_kept_features_count_matches_features_parquet_row_count
- test_inline_invariant_8_mean_water_fraction_area_weighted_formula_match
- test_inline_invariant_9_cell_area_admin_clipped_alpha_structural_boundary
- test_inline_invariant_10_nan_free_on_non_water_fraction_numeric_columns

Plus baseline: test_validate_tile_inline_passes_on_clean_synthetic_tile
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from shapely.geometry import LineString, Point, Polygon

from cfm.data.sub_c.errors import TileValidationError
from cfm.data.sub_c.geom import CrossingRecord
from cfm.data.sub_c.io import (
    _CROSSINGS_SCHEMA,
    CellAggregate,
    FeatureRow,
    TileMeta,
    canonicalize_yaml,
    write_cells_parquet,
    write_crossings_parquet,
    write_features_parquet,
    write_meta_yaml,
)
from cfm.data.sub_c.validator_inline import validate_tile_inline

# ---------------------------------------------------------------------------
# Helper: read parquet without Hive partition column inference
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pa.Table:
    """Read a parquet file without Hive partition column inference.

    pq.read_table on a path inside a directory named 'tile=X' would inject a
    spurious 'tile' column. Use ParquetFile.read() instead.
    """
    return pq.ParquetFile(path).read()


# ---------------------------------------------------------------------------
# Clean synthetic tile builder
# ---------------------------------------------------------------------------

# Feature A: a LineString crossing two cells (cell 0,0 and cell 0,1)
# Feature B: a Point entirely in cell 0,0
# Feature C: a Polygon entirely in cell 0,0

_FEAT_A_GEOM = LineString([(10.0, 10.0), (310.0, 50.0)])  # crosses x=250 boundary
_FEAT_B_GEOM = Point((50.0, 50.0))
_FEAT_C_GEOM = Polygon([(60.0, 60.0), (90.0, 60.0), (90.0, 90.0), (60.0, 60.0)])


def _make_clean_tile_dir(tmp_path: Path) -> Path:
    """Build a minimal valid tile directory with:
    - cells.parquet: 2 cells (i=0,j=0) and (i=0,j=1)
    - features.parquet: 4 rows (feat_A in both cells, feat_B in cell 0,0, feat_C in cell 0,0)
    - crossings.parquet: feat_A crosses x-boundary between cell j=0 and j=1 (2 records)
    - meta.yaml: correct area-weighted mean_water_fraction + mean_sea_water_fraction
    """
    tile_dir = tmp_path / "tile=EPSG3414_i0_j0"
    tile_dir.mkdir()

    # Cell aggregates
    cell_00 = CellAggregate(
        cell_i=0,
        cell_j=0,
        water_fraction=0.2,
        sea_water_fraction=0.1,
        cell_area_admin_clipped_m2=62500.0,
        kept_features_count=3,  # feat_A(cell00 fragment), feat_B, feat_C
    )
    cell_01 = CellAggregate(
        cell_i=0,
        cell_j=1,
        water_fraction=0.3,
        sea_water_fraction=0.2,
        cell_area_admin_clipped_m2=50000.0,
        kept_features_count=1,  # feat_A(cell01 fragment)
    )
    write_cells_parquet([cell_00, cell_01], tile_dir / "cells.parquet")

    # Feature rows — feat_A appears as two rows (one per cell it spans)
    feat_a_cell00 = _make_feature_row(
        cell_i=0,
        cell_j=0,
        feature_class=0,  # road
        source_feature_id="feat_A",
        geometry=LineString([(10.0, 10.0), (250.0, 34.0)]),  # cell-00 fragment
        geometry_type=1,  # LineString
    )
    feat_a_cell01 = _make_feature_row(
        cell_i=0,
        cell_j=1,
        feature_class=0,  # road
        source_feature_id="feat_A",
        geometry=LineString([(250.0, 34.0), (310.0, 50.0)]),  # cell-01 fragment
        geometry_type=1,  # LineString
    )
    feat_b_cell00 = _make_feature_row(
        cell_i=0,
        cell_j=0,
        feature_class=2,  # poi
        source_feature_id="feat_B",
        geometry=_FEAT_B_GEOM,
        geometry_type=0,  # Point
    )
    feat_c_cell00 = _make_feature_row(
        cell_i=0,
        cell_j=0,
        feature_class=1,  # building
        source_feature_id="feat_C",
        geometry=_FEAT_C_GEOM,
        geometry_type=2,  # Polygon
    )
    write_features_parquet(
        [feat_a_cell00, feat_a_cell01, feat_b_cell00, feat_c_cell00],
        tile_dir / "features.parquet",
    )

    # Crossings — feat_A crosses x-axis boundary between cell j=0 and j=1
    crossing_enter = CrossingRecord(
        source_feature_id="feat_A",
        lower_cell_i=0,
        lower_cell_j=0,
        axis=0,  # x-axis
        ring_index=0,
        event_type=0,  # enter
        edge_position_m=125.0,
        edge_extent_length_m=0.0,
    )
    crossing_exit = CrossingRecord(
        source_feature_id="feat_A",
        lower_cell_i=0,
        lower_cell_j=0,
        axis=0,  # x-axis
        ring_index=0,
        event_type=1,  # exit
        edge_position_m=200.0,
        edge_extent_length_m=0.0,
    )
    write_crossings_parquet([crossing_enter, crossing_exit], tile_dir / "crossings.parquet")

    # Area-weighted mean_water_fraction:
    # = (0.2 * 62500 + 0.3 * 50000) / (62500 + 50000) = 27500/112500
    mean_wf = (0.2 * 62500.0 + 0.3 * 50000.0) / (62500.0 + 50000.0)
    mean_sea_wf = (0.1 * 62500.0 + 0.2 * 50000.0) / (62500.0 + 50000.0)

    meta = TileMeta(
        schema_version="1.1",
        tile_i=0,
        tile_j=0,
        aggregates={
            "kept_cell_count": 2,
            "sea_mask_drop_count": 0,
            "mean_water_fraction": mean_wf,
            "mean_sea_water_fraction": mean_sea_wf,
            "feature_count_by_class": {"road": 2, "building": 1, "poi": 1, "base": 0},
            "crossing_count": 2,
        },
        config={"sliver_drop_rule": "drop iff area < 0.01 m²"},
        conditioning_per_tile={
            "admin_region": "Central",
            "morphology_class": "Asian-megacity",
            "era_class": "contemporary",
            "coastal_inland_river": 1,
            "population_density_bucket": None,
            "population_density_bucket_owner": "sub-D",
        },
    )
    write_meta_yaml(meta, tile_dir / "meta.yaml")

    return tile_dir


def _make_feature_row(
    cell_i: int = 0,
    cell_j: int = 0,
    feature_class: int = 0,
    source_feature_id: str = "feat_X",
    geometry: object | None = None,
    geometry_type: int = 1,
    class_raw: str | None = "primary",
    subtype_raw: str | None = None,
    categories_primary: str | None = None,
    categories_alternate: list[str] | None = None,
    sea_overlap_fraction: float = 0.0,
) -> FeatureRow:
    if geometry is None:
        geometry = LineString([(0.0, 0.0), (100.0, 50.0)])
    min_x, min_y, max_x, max_y = geometry.bounds
    return FeatureRow(
        cell_i=cell_i,
        cell_j=cell_j,
        feature_class=feature_class,
        source_feature_id=source_feature_id,
        geometry=geometry,
        geometry_type=geometry_type,
        bbox_min_x=min_x,
        bbox_min_y=min_y,
        bbox_max_x=max_x,
        bbox_max_y=max_y,
        class_raw=class_raw,
        subtype_raw=subtype_raw,
        categories_primary=categories_primary,
        categories_alternate=categories_alternate,
        sea_overlap_fraction=sea_overlap_fraction,
    )


# ---------------------------------------------------------------------------
# Baseline test — clean tile passes all 10 invariants
# ---------------------------------------------------------------------------


def test_validate_tile_inline_passes_on_clean_synthetic_tile(tmp_path: Path) -> None:
    """A well-formed synthetic tile raises no TileValidationError."""
    tile_dir = _make_clean_tile_dir(tmp_path)
    # Must not raise
    validate_tile_inline(tile_dir)


# ---------------------------------------------------------------------------
# Layer-2 torture tile — wired up in Task 16 via session-scoped fixture
# (conftest.py re-exports torture_tile_output from test_fixture_builders).
# ---------------------------------------------------------------------------


def test_inline_validator_passes_on_clean_torture_tile_output(
    torture_tile_output: Path,
) -> None:
    """validate_tile_inline on the torture tile output must not raise.

    Equivalent to test_torture_tile_inline_validator_passes_on_clean_output in
    test_fixture_builders.py but kept here so the spec §13.2 named-test list
    has a one-to-one mapping with the suite.  Both tests share the same
    session-scoped fixture and add no measurable wall-clock cost.

    Per feedback_test_weakening_to_pass.md: if this raises, STOP and escalate;
    do NOT weaken the assertion.
    """
    from tests.fixtures.sub_c.build_torture_tile import (
        TORTURE_TILE_I,
        TORTURE_TILE_J,
    )

    tile_dir = torture_tile_output / f"tile=EPSG3414_i{TORTURE_TILE_I}_j{TORTURE_TILE_J}"
    validate_tile_inline(tile_dir)


# ---------------------------------------------------------------------------
# Invariant 1: schema correctness
# ---------------------------------------------------------------------------


def test_inline_invariant_1_schema_correctness_diagnostic(tmp_path: Path) -> None:
    """Corrupt: drop 'geometry_type' column from features.parquet.

    Validator must raise TileValidationError(invariant='schema_correctness')
    with detail including the name of the mismatched/missing column context.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    features_path = tile_dir / "features.parquet"

    table = _read_parquet(features_path)
    # Drop a column so schema no longer matches _FEATURES_SCHEMA
    idx = table.schema.get_field_index("geometry_type")
    table = table.remove_column(idx)
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "schema_correctness"
    assert err.tile == "tile=EPSG3414_i0_j0"
    # detail should mention which parquet had the mismatch
    assert err.detail


# ---------------------------------------------------------------------------
# Invariant 2: bbox vs WKB consistency
# ---------------------------------------------------------------------------


def test_inline_invariant_2_bbox_matches_wkb_diagnostic(tmp_path: Path) -> None:
    """Corrupt: bbox_min_x off by 100m on row 0.

    Validator must raise TileValidationError(invariant='bbox_matches_wkb')
    with failed_row containing row_index=0 and detail containing stored_bbox/wkb_bbox.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    features_path = tile_dir / "features.parquet"

    table = _read_parquet(features_path)
    bbox_col = table.column("bbox_min_x").to_pylist()
    bbox_col[0] += 100.0  # corrupt row 0
    table = table.set_column(
        table.schema.get_field_index("bbox_min_x"),
        "bbox_min_x",
        pa.array(bbox_col, type=pa.float64()),
    )
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "bbox_matches_wkb"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert err.failed_row.get("row_index") == 0
    assert "stored_bbox" in err.detail
    assert "wkb_bbox" in err.detail


# ---------------------------------------------------------------------------
# Invariant 3: geometry_type vs WKB header
# ---------------------------------------------------------------------------


def test_inline_invariant_3_geometry_type_matches_wkb_diagnostic(tmp_path: Path) -> None:
    """Corrupt: set geometry_type=2 (Polygon) on a row whose WKB is a LineString.

    Validator must raise TileValidationError(invariant='geometry_type_matches_wkb')
    with failed_row containing row_index and detail containing stored_type/wkb_type.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    features_path = tile_dir / "features.parquet"

    table = _read_parquet(features_path)
    # Find first LineString row (geometry_type==1) and corrupt its geometry_type to 2
    gt_col = table.column("geometry_type").to_pylist()
    target_row = next(i for i, v in enumerate(gt_col) if v == 1)
    gt_col[target_row] = 2  # claim it's a Polygon when it's a LineString
    table = table.set_column(
        table.schema.get_field_index("geometry_type"),
        "geometry_type",
        pa.array(gt_col, type=pa.int8()),
    )
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "geometry_type_matches_wkb"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert err.failed_row.get("row_index") == target_row
    assert "stored_type" in err.detail
    assert "wkb_type" in err.detail


# ---------------------------------------------------------------------------
# Invariant 4: crossings vs features source_feature_id consistency
# ---------------------------------------------------------------------------


def test_inline_invariant_4_crossings_features_source_feature_id_diagnostic(
    tmp_path: Path,
) -> None:
    """Corrupt: add a crossings row with source_feature_id='phantom_X' (appears on 0 cells).

    Validator must raise TileValidationError(invariant='crossings_features_source_id_consistency')
    with failed_row containing source_feature_id='phantom_X'.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    crossings_path = tile_dir / "crossings.parquet"

    table = _read_parquet(crossings_path)
    # Append a crossing row referencing an ID that doesn't appear in features at all
    extra = {
        "source_feature_id": pa.array(["phantom_X"], type=pa.string()),
        "lower_cell_i": pa.array([0], type=pa.int8()),
        "lower_cell_j": pa.array([0], type=pa.int8()),
        "axis": pa.array([0], type=pa.int8()),
        "ring_index": pa.array([0], type=pa.int16()),
        "event_type": pa.array([0], type=pa.int8()),
        "edge_position_m": pa.array([50.0], type=pa.float64()),
        "edge_extent_length_m": pa.array([0.0], type=pa.float64()),
    }
    extra_table = pa.table(extra, schema=_CROSSINGS_SCHEMA)
    combined = pa.concat_tables([table, extra_table])
    pq.write_table(combined, crossings_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "crossings_features_source_id_consistency"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert err.failed_row.get("source_feature_id") == "phantom_X"


# ---------------------------------------------------------------------------
# Invariant 5: water fraction bounds combined with NaN check
# ---------------------------------------------------------------------------


def test_inline_invariant_5_water_fraction_bounds_combined_with_nan_check_single_pass(
    tmp_path: Path,
) -> None:
    """Corrupt: water_fraction=1.5 AND sea_water_fraction=NaN on the same cell.

    Validator must raise TileValidationError(invariant='water_fraction_bounds')
    with failed_row containing cell_i, cell_j and detail including the bad value.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    cells_path = tile_dir / "cells.parquet"

    table = _read_parquet(cells_path)
    # Corrupt first cell: water_fraction > 1 + EPS_RATIO
    wf = table.column("water_fraction").to_pylist()
    sea_wf = table.column("sea_water_fraction").to_pylist()
    wf[0] = 1.5
    sea_wf[0] = float("nan")
    table = table.set_column(
        table.schema.get_field_index("water_fraction"),
        "water_fraction",
        pa.array(wf, type=pa.float64()),
    )
    table = table.set_column(
        table.schema.get_field_index("sea_water_fraction"),
        "sea_water_fraction",
        pa.array(sea_wf, type=pa.float64()),
    )
    pq.write_table(table, cells_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "water_fraction_bounds"
    assert err.tile == "tile=EPSG3414_i0_j0"
    # failed_row must identify the offending cell
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row


# ---------------------------------------------------------------------------
# Invariant 6: kept-cell rule consistency
# ---------------------------------------------------------------------------


def test_inline_invariant_6_kept_cell_rule_diagnostic(tmp_path: Path) -> None:
    """Corrupt: set sea_water_fraction=1.0, water_fraction=1.0 (bounds pass), AND
    kept_features_count=0 on one cell.

    Validator must raise TileValidationError(invariant='kept_cell_rule')
    with failed_row containing cell_i, cell_j and detail including both fractions.

    Note: per critical-constraint §2, if this fires on REAL data -> STOP + escalate.
    Here we deliberately corrupt a synthetic cell to test the assertion fires correctly.

    The corruption sets both wf and sea to 1.0 so invariant #5 (bounds check) passes
    (sea <= wf = 1.0, both within [0-EPS, 1+EPS]), and only #6 fires.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    cells_path = tile_dir / "cells.parquet"

    table = _read_parquet(cells_path)
    # Corrupt first cell: wf=1.0, sea=1.0 (bounds OK for #5), kfc=0 (violates #6)
    wf = table.column("water_fraction").to_pylist()
    sea_wf = table.column("sea_water_fraction").to_pylist()
    kfc = table.column("kept_features_count").to_pylist()
    area = table.column("cell_area_admin_clipped_m2").to_pylist()
    wf[0] = 1.0
    sea_wf[0] = 1.0
    kfc[0] = 0
    table = table.set_column(
        table.schema.get_field_index("water_fraction"),
        "water_fraction",
        pa.array(wf, type=pa.float64()),
    )
    table = table.set_column(
        table.schema.get_field_index("sea_water_fraction"),
        "sea_water_fraction",
        pa.array(sea_wf, type=pa.float64()),
    )
    table = table.set_column(
        table.schema.get_field_index("kept_features_count"),
        "kept_features_count",
        pa.array(kfc, type=pa.int32()),
    )
    pq.write_table(table, cells_path)

    # Also update meta.yaml so invariant #8 does not fire first
    # (area-weighted mean recalculated with wf[0]=1.0, sea[0]=1.0)
    meta_path = tile_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    total_area = sum(a for a in area)
    new_wf_mean = sum(w * a for w, a in zip(wf, area, strict=True)) / total_area
    new_sea_mean = sum(s * a for s, a in zip(sea_wf, area, strict=True)) / total_area
    meta["aggregates"]["mean_water_fraction"] = new_wf_mean
    meta["aggregates"]["mean_sea_water_fraction"] = new_sea_mean
    meta_path.write_text(canonicalize_yaml(meta), encoding="utf-8")

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "kept_cell_rule"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert "sea_water_fraction" in err.detail
    assert "kept_features_count" in err.detail


# ---------------------------------------------------------------------------
# Invariant 7: kept_features_count matches features.parquet row count per cell
# ---------------------------------------------------------------------------


def test_inline_invariant_7_kept_features_count_matches_features_parquet_row_count(
    tmp_path: Path,
) -> None:
    """Corrupt: set kept_features_count=99 for cell (0,0) which has 3 actual rows.

    Validator must raise TileValidationError(invariant='kept_features_count_matches')
    with failed_row containing cell_i, cell_j and detail with stored/actual counts.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    cells_path = tile_dir / "cells.parquet"

    table = _read_parquet(cells_path)
    kfc = table.column("kept_features_count").to_pylist()
    # Find cell (0,0) and corrupt its count
    ci = table.column("cell_i").to_pylist()
    cj = table.column("cell_j").to_pylist()
    for idx, (i, j) in enumerate(zip(ci, cj, strict=True)):
        if i == 0 and j == 0:
            kfc[idx] = 99
            break
    table = table.set_column(
        table.schema.get_field_index("kept_features_count"),
        "kept_features_count",
        pa.array(kfc, type=pa.int32()),
    )
    pq.write_table(table, cells_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "kept_features_count_matches"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert "stored" in err.detail
    assert "actual" in err.detail


# ---------------------------------------------------------------------------
# Invariant 8: meta.yaml mean_water_fraction matches area-weighted formula
# ---------------------------------------------------------------------------


def test_inline_invariant_8_mean_water_fraction_area_weighted_formula_match(
    tmp_path: Path,
) -> None:
    """Corrupt: set meta.yaml mean_water_fraction to a wrong value (0.999).

    Validator must raise TileValidationError(invariant='mean_water_fraction_matches')
    with detail containing stored and computed values.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    meta_path = tile_dir / "meta.yaml"

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    meta["aggregates"]["mean_water_fraction"] = 0.999  # wrong
    meta_path.write_text(canonicalize_yaml(meta), encoding="utf-8")

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "mean_water_fraction_matches"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert "stored" in err.detail
    assert "computed" in err.detail


# ---------------------------------------------------------------------------
# Invariant 9: cell_area_admin_clipped_m2 > EPS_AREA_M2 (alpha structural boundary)
# ---------------------------------------------------------------------------


def test_inline_invariant_9_cell_area_admin_clipped_alpha_structural_boundary(
    tmp_path: Path,
) -> None:
    """Corrupt: set cell_area_admin_clipped_m2=0.0 on one cell.

    Validator must raise TileValidationError(invariant='cell_area_positive')
    with failed_row containing cell_i, cell_j and detail with actual area.

    The corruption also updates meta.yaml so invariant #8 does not fire first
    (area-weighted mean recalculated with the zeroed area).
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    cells_path = tile_dir / "cells.parquet"

    table = _read_parquet(cells_path)
    wf_col = table.column("water_fraction").to_pylist()
    sea_col = table.column("sea_water_fraction").to_pylist()
    area_col = table.column("cell_area_admin_clipped_m2").to_pylist()
    area_col[0] = 0.0  # zero area violates > EPS_AREA_M2
    table = table.set_column(
        table.schema.get_field_index("cell_area_admin_clipped_m2"),
        "cell_area_admin_clipped_m2",
        pa.array(area_col, type=pa.float64()),
    )
    pq.write_table(table, cells_path)

    # Update meta.yaml so invariant #8 does not fire before #9
    meta_path = tile_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    total_area = sum(a for a in area_col)
    if total_area > 0:
        new_wf_mean = sum(w * a for w, a in zip(wf_col, area_col, strict=True)) / total_area
        new_sea_mean = sum(s * a for s, a in zip(sea_col, area_col, strict=True)) / total_area
    else:
        new_wf_mean = 0.0
        new_sea_mean = 0.0
    meta["aggregates"]["mean_water_fraction"] = new_wf_mean
    meta["aggregates"]["mean_sea_water_fraction"] = new_sea_mean
    meta_path.write_text(canonicalize_yaml(meta), encoding="utf-8")

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "cell_area_positive"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert "area" in err.detail


# ---------------------------------------------------------------------------
# Invariant 10: NaN-free on non-water-fraction numeric columns
# ---------------------------------------------------------------------------


def test_inline_invariant_10_nan_free_on_non_water_fraction_numeric_columns(
    tmp_path: Path,
) -> None:
    """Corrupt: set edge_position_m=NaN on a crossings row.

    Validator must raise TileValidationError(invariant='nan_free_numeric_columns')
    with failed_row containing row_index and detail containing column name.
    """
    tile_dir = _make_clean_tile_dir(tmp_path)
    crossings_path = tile_dir / "crossings.parquet"

    table = _read_parquet(crossings_path)
    epm = table.column("edge_position_m").to_pylist()
    epm[0] = float("nan")
    table = table.set_column(
        table.schema.get_field_index("edge_position_m"),
        "edge_position_m",
        pa.array(epm, type=pa.float64()),
    )
    pq.write_table(table, crossings_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "nan_free_numeric_columns"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert err.failed_row.get("row_index") == 0
    assert "column" in err.detail
