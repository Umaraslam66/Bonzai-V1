"""Per-tile inline validator. Runs after parquet + meta.yaml writes, BEFORE
provenance.yaml. Failure halts (no provenance.yaml = tile not complete).

Per spec §12.1 + §12.4: 10 named invariants; structured TileValidationError
payload (tile, invariant, failed_row, detail). Diagnostic-payload determinism
is verified end-to-end by test_validator_diagnostic_payloads_byte_deterministic
in Task 16.

Failures are bugs-in-sub-C, NOT data-issues — per auto-memory
feedback_test_weakening_to_pass.md. If an invariant fires on real Singapore
data, STOP and escalate.

Invariant IDs (spec §12.1):
  "schema_correctness"
  "bbox_matches_wkb"
  "geometry_type_matches_wkb"
  "crossings_features_source_id_consistency"
  "water_fraction_bounds"
  "kept_cell_rule"
  "kept_features_count_matches"
  "mean_water_fraction_matches"
  "cell_area_positive"
  "nan_free_numeric_columns"
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from shapely import wkb as shapely_wkb

from cfm.data.sub_c.epsilon import EPS_AREA_M2, EPS_COORD_M, EPS_RATIO
from cfm.data.sub_c.errors import TileValidationError
from cfm.data.sub_c.io import _CELLS_SCHEMA, _CROSSINGS_SCHEMA, _FEATURES_SCHEMA

# WKB geometry type codes (ISO WKB / little-endian NDR)
# Our int8 enum: 0=Point, 1=LineString, 2=Polygon, 3=MultiPoint, 4=MultiLineString, 5=MultiPolygon
# WKB type bytes: 1=Point, 2=LineString, 3=Polygon, 4=MultiPoint, 5=MultiLineString, 6=MultiPolygon
_ENUM_TO_WKB_TYPE: dict[int, int] = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def _read_parquet(path: Path) -> pa.Table:
    """Read a single parquet file without Hive partition inference.

    Using pq.ParquetFile instead of pq.read_table avoids pyarrow inferring
    Hive partition columns from parent directory names (e.g. tile=EPSG3414_i0_j0
    would otherwise inject a spurious 'tile' column into the schema).
    """
    return pq.ParquetFile(path).read()


def validate_tile_inline(tile_dir: Path) -> None:
    """Read all three parquets + meta.yaml from tile_dir. Run all 10 invariants.

    Raises TileValidationError(tile=..., invariant=..., failed_row=..., detail=...)
    on the first failure. Subsequent invariants are only run if earlier ones pass
    (single-pass; fast-fail).

    Per spec §12.1: each invariant has a named id used in TileValidationError.invariant:
      "schema_correctness", "bbox_matches_wkb", "geometry_type_matches_wkb",
      "crossings_features_source_id_consistency", "water_fraction_bounds",
      "kept_cell_rule", "kept_features_count_matches", "mean_water_fraction_matches",
      "cell_area_positive", "nan_free_numeric_columns"
    """
    tile_name = tile_dir.name

    cells = _read_parquet(tile_dir / "cells.parquet")
    features = _read_parquet(tile_dir / "features.parquet")
    crossings = _read_parquet(tile_dir / "crossings.parquet")
    meta = yaml.safe_load((tile_dir / "meta.yaml").read_text(encoding="utf-8"))

    # Invariants run in spec §12.1 order. Order is load-bearing: corruption
    # tests rely on fast-fail behaviour to isolate the target invariant.
    # Moving #6 (kept_cell_rule) before #5 (water_fraction_bounds) or moving
    # #9 (cell_area_positive) before #8 (mean_water_fraction_matches) will
    # silently change which invariant fires for a given bad tile.
    _check_schema_correctness(tile_name, cells, features, crossings)
    _check_bbox_matches_wkb(tile_name, features)
    _check_geometry_type_matches_wkb(tile_name, features)
    _check_crossings_features_source_id_consistency(tile_name, features, crossings)
    _check_water_fraction_bounds(tile_name, cells)
    _check_kept_cell_rule(tile_name, cells)
    _check_kept_features_count_matches(tile_name, cells, features)
    _check_mean_water_fraction_matches(tile_name, cells, meta)
    _check_cell_area_positive(tile_name, cells)
    _check_nan_free_numeric_columns(tile_name, cells, features, crossings)


# ---------------------------------------------------------------------------
# Invariant 1: Schema correctness
# ---------------------------------------------------------------------------


def _check_schema_correctness(
    tile_name: str,
    cells: pa.Table,
    features: pa.Table,
    crossings: pa.Table,
) -> None:
    """Every expected column present with right type; sort key canonical.

    Per spec §12.1 #1: compare actual pyarrow schema to each pinned schema.
    Uses pa.Schema.equals() which handles type comparisons cleanly.

    Sort-key check: verify the rows are sorted in canonical order.
    - features: (cell_i, cell_j, feature_class, source_feature_id)
    - cells: (cell_i, cell_j)
    - crossings: (lower_cell_i, lower_cell_j, axis, source_feature_id,
                  ring_index, edge_position_m, event_type)
    """
    for parquet_name, table, expected_schema in [
        ("features.parquet", features, _FEATURES_SCHEMA),
        ("cells.parquet", cells, _CELLS_SCHEMA),
        ("crossings.parquet", crossings, _CROSSINGS_SCHEMA),
    ]:
        if not table.schema.equals(expected_schema):
            raise TileValidationError(
                tile=tile_name,
                invariant="schema_correctness",
                failed_row={},
                detail={
                    "file": parquet_name,
                    "schema": "actual schema does not match expected schema",
                    "actual": str(table.schema),
                    "expected": str(expected_schema),
                },
            )

    # Sort-key check: features
    _assert_sort_key(
        tile_name=tile_name,
        table=features,
        parquet_name="features.parquet",
        key_cols=["cell_i", "cell_j", "feature_class", "source_feature_id"],
    )
    # Sort-key check: cells
    _assert_sort_key(
        tile_name=tile_name,
        table=cells,
        parquet_name="cells.parquet",
        key_cols=["cell_i", "cell_j"],
    )
    # Sort-key check: crossings
    _assert_sort_key(
        tile_name=tile_name,
        table=crossings,
        parquet_name="crossings.parquet",
        key_cols=[
            "lower_cell_i",
            "lower_cell_j",
            "axis",
            "source_feature_id",
            "ring_index",
            "edge_position_m",
            "event_type",
        ],
    )


def _assert_sort_key(
    tile_name: str,
    table: pa.Table,
    parquet_name: str,
    key_cols: list[str],
) -> None:
    """Raise TileValidationError(invariant='schema_correctness') if table rows are
    not sorted in ascending order by key_cols (stable lexicographic).
    """
    if table.num_rows < 2:
        return

    # Extract key tuples row-by-row
    columns = {col: table.column(col).to_pylist() for col in key_cols}
    n = table.num_rows
    for row_idx in range(1, n):
        prev_key = tuple(columns[c][row_idx - 1] for c in key_cols)
        curr_key = tuple(columns[c][row_idx] for c in key_cols)
        if curr_key < prev_key:
            raise TileValidationError(
                tile=tile_name,
                invariant="schema_correctness",
                failed_row={"row_index": row_idx},
                detail={
                    "file": parquet_name,
                    "schema": "sort key violated",
                    "key_cols": key_cols,
                    "prev_key": list(prev_key),
                    "curr_key": list(curr_key),
                },
            )


# ---------------------------------------------------------------------------
# Invariant 2: bbox_* vs WKB consistency
# ---------------------------------------------------------------------------


def _check_bbox_matches_wkb(tile_name: str, features: pa.Table) -> None:
    """For every feature row, bbox_* columns match WKB-derived bbox within EPS_COORD_M.

    Per spec §12.1 #2: EPS_COORD_M = 1e-6 (alpha structural-boundary EPSILON).
    """
    geom_col = features.column("geometry").to_pylist()
    bbox_min_x = features.column("bbox_min_x").to_pylist()
    bbox_min_y = features.column("bbox_min_y").to_pylist()
    bbox_max_x = features.column("bbox_max_x").to_pylist()
    bbox_max_y = features.column("bbox_max_y").to_pylist()
    source_ids = features.column("source_feature_id").to_pylist()

    for idx in range(features.num_rows):
        geom = shapely_wkb.loads(geom_col[idx])
        wkb_bounds = geom.bounds  # (min_x, min_y, max_x, max_y)
        stored = (bbox_min_x[idx], bbox_min_y[idx], bbox_max_x[idx], bbox_max_y[idx])

        if any(abs(a - b) > EPS_COORD_M for a, b in zip(wkb_bounds, stored, strict=True)):
            raise TileValidationError(
                tile=tile_name,
                invariant="bbox_matches_wkb",
                failed_row={
                    "source_feature_id": source_ids[idx],
                    "row_index": idx,
                },
                detail={
                    "stored_bbox": list(stored),
                    "wkb_bbox": list(wkb_bounds),
                },
            )


# ---------------------------------------------------------------------------
# Invariant 3: geometry_type vs WKB header
# ---------------------------------------------------------------------------


def _check_geometry_type_matches_wkb(tile_name: str, features: pa.Table) -> None:
    """For every feature row, geometry_type int8 enum matches WKB header type.

    Per spec §12.1 #3:
    - Our int8 enum: 0=Point, 1=LineString, 2=Polygon
    - WKB type codes (NDR little-endian): 1=Point, 2=LineString, 3=Polygon
    - WKB byte 0 = byte_order (0=XDR/big-endian, 1=NDR/little-endian)
    - WKB bytes 1-4 = geometry type (uint32 in stated byte order)

    We write NDR (little-endian) exclusively per spec §14.3 + dump_wkb.
    """
    geom_col = features.column("geometry").to_pylist()
    gt_col = features.column("geometry_type").to_pylist()
    source_ids = features.column("source_feature_id").to_pylist()

    for idx in range(features.num_rows):
        wkb_bytes = geom_col[idx]
        byte_order = wkb_bytes[0]  # 0=XDR (big-endian), 1=NDR (little-endian)
        endianness = "little" if byte_order == 1 else "big"
        wkb_type_code = int.from_bytes(wkb_bytes[1:5], byteorder=endianness)

        stored_enum = gt_col[idx]
        expected_wkb_type = _ENUM_TO_WKB_TYPE.get(stored_enum)

        if expected_wkb_type != wkb_type_code:
            raise TileValidationError(
                tile=tile_name,
                invariant="geometry_type_matches_wkb",
                failed_row={
                    "source_feature_id": source_ids[idx],
                    "row_index": idx,
                },
                detail={
                    "stored_type": stored_enum,
                    "wkb_type": wkb_type_code,
                },
            )


# ---------------------------------------------------------------------------
# Invariant 4: crossings vs features source_feature_id consistency
# ---------------------------------------------------------------------------


def _check_crossings_features_source_id_consistency(
    tile_name: str,
    features: pa.Table,
    crossings: pa.Table,
) -> None:
    """Every source_feature_id in crossings.parquet appears on >= 2 distinct cells
    in features.parquet (i.e., was actually cut).

    Per spec §12.1 #4. Build a dict of source_feature_id -> set of (cell_i, cell_j)
    from features. Then for each crossing source_feature_id, check len(cells) >= 2.
    """
    # Build: source_feature_id -> set of (cell_i, cell_j)
    feat_ci = features.column("cell_i").to_pylist()
    feat_cj = features.column("cell_j").to_pylist()
    feat_ids = features.column("source_feature_id").to_pylist()

    id_to_cells: dict[str, set[tuple[int, int]]] = {}
    for fid, ci, cj in zip(feat_ids, feat_ci, feat_cj, strict=True):
        id_to_cells.setdefault(fid, set()).add((ci, cj))

    # Check each crossing source_feature_id
    cross_ids = crossings.column("source_feature_id").to_pylist()
    # Deduplicate while preserving first occurrence for deterministic failure reporting
    seen: set[str] = set()
    for idx, fid in enumerate(cross_ids):
        if fid in seen:
            continue
        seen.add(fid)
        cells_for_id = id_to_cells.get(fid, set())
        if len(cells_for_id) < 2:
            raise TileValidationError(
                tile=tile_name,
                invariant="crossings_features_source_id_consistency",
                failed_row={
                    "source_feature_id": fid,
                    "crossing_row_index": idx,
                },
                detail={
                    "distinct_cells_in_features": [list(c) for c in sorted(cells_for_id)],
                    "required_min_distinct_cells": 2,
                },
            )


# ---------------------------------------------------------------------------
# Invariant 5: water fraction bounds combined with NaN check (single pass)
# ---------------------------------------------------------------------------


def _check_water_fraction_bounds(tile_name: str, cells: pa.Table) -> None:
    """Single-pass check: for every cell row:
    0 - EPS_RATIO <= sea_water_fraction <= water_fraction <= 1 + EPS_RATIO
    AND not isnan(water_fraction) AND not isnan(sea_water_fraction).

    Per spec §12.1 #5 + §14.3 efficiency lock (single column traversal).
    EPS_RATIO = 1e-9 (alpha structural-boundary comparison against 0 and 1).
    """
    ci_col = cells.column("cell_i").to_pylist()
    cj_col = cells.column("cell_j").to_pylist()
    wf_col = cells.column("water_fraction").to_pylist()
    sea_col = cells.column("sea_water_fraction").to_pylist()

    lo = 0.0 - EPS_RATIO
    hi = 1.0 + EPS_RATIO

    for idx in range(cells.num_rows):
        wf = wf_col[idx]
        sea = sea_col[idx]

        # NaN check (pyarrow nulls come through as None, not NaN)
        if (wf is not None and math.isnan(wf)) or (sea is not None and math.isnan(sea)):
            raise TileValidationError(
                tile=tile_name,
                invariant="water_fraction_bounds",
                failed_row={"cell_i": ci_col[idx], "cell_j": cj_col[idx], "row_index": idx},
                detail={
                    "water_fraction": wf,
                    "sea_water_fraction": sea,
                    "reason": "NaN value",
                },
            )

        # Bounds check: sea <= wf, both in [0-EPS, 1+EPS]
        wf_val = 0.0 if wf is None else wf
        sea_val = 0.0 if sea is None else sea
        if not (lo <= sea_val <= wf_val <= hi):
            raise TileValidationError(
                tile=tile_name,
                invariant="water_fraction_bounds",
                failed_row={"cell_i": ci_col[idx], "cell_j": cj_col[idx], "row_index": idx},
                detail={
                    "water_fraction": wf,
                    "sea_water_fraction": sea,
                    "reason": f"bounds violated: expected {lo} <= sea <= wf <= {hi}",
                },
            )


# ---------------------------------------------------------------------------
# Invariant 6: kept-cell rule consistency
# ---------------------------------------------------------------------------


def _check_kept_cell_rule(tile_name: str, cells: pa.Table) -> None:
    """For every cell in cells.parquet:
    NOT (sea_water_fraction >= 1.0 - EPS_RATIO AND kept_features_count == 0).

    Per spec §12.1 #6. Catches "drop rule wasn't applied" bugs.
    Per critical-constraint §2: if this fires on real data -> STOP and escalate.
    EPS_RATIO = 1e-9 (alpha structural-boundary comparison against 1.0).
    """
    ci_col = cells.column("cell_i").to_pylist()
    cj_col = cells.column("cell_j").to_pylist()
    sea_col = cells.column("sea_water_fraction").to_pylist()
    kfc_col = cells.column("kept_features_count").to_pylist()

    threshold = 1.0 - EPS_RATIO

    for idx in range(cells.num_rows):
        sea = sea_col[idx]
        kfc = kfc_col[idx]
        sea_val = 0.0 if sea is None else sea

        if sea_val >= threshold and kfc == 0:
            raise TileValidationError(
                tile=tile_name,
                invariant="kept_cell_rule",
                failed_row={"cell_i": ci_col[idx], "cell_j": cj_col[idx], "row_index": idx},
                detail={
                    "sea_water_fraction": sea,
                    "kept_features_count": kfc,
                    "reason": (
                        "cell with sea_water_fraction >= 1.0 - EPS_RATIO should have been dropped"
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Invariant 7: kept_features_count matches features.parquet row count per cell
# ---------------------------------------------------------------------------


def _check_kept_features_count_matches(
    tile_name: str,
    cells: pa.Table,
    features: pa.Table,
) -> None:
    """For each (cell_i, cell_j) in cells.parquet, kept_features_count must
    equal the number of rows in features.parquet with that (cell_i, cell_j).

    Per spec §12.1 #7.
    """
    # Count features per (cell_i, cell_j)
    feat_ci = features.column("cell_i").to_pylist()
    feat_cj = features.column("cell_j").to_pylist()

    actual_counts: dict[tuple[int, int], int] = {}
    for ci, cj in zip(feat_ci, feat_cj, strict=True):
        key = (ci, cj)
        actual_counts[key] = actual_counts.get(key, 0) + 1

    ci_col = cells.column("cell_i").to_pylist()
    cj_col = cells.column("cell_j").to_pylist()
    kfc_col = cells.column("kept_features_count").to_pylist()

    for idx in range(cells.num_rows):
        ci = ci_col[idx]
        cj = cj_col[idx]
        stored = kfc_col[idx]
        actual = actual_counts.get((ci, cj), 0)

        if stored != actual:
            raise TileValidationError(
                tile=tile_name,
                invariant="kept_features_count_matches",
                failed_row={"cell_i": ci, "cell_j": cj, "row_index": idx},
                detail={
                    "stored": stored,
                    "actual": actual,
                },
            )


# ---------------------------------------------------------------------------
# Invariant 8: meta.yaml.mean_water_fraction matches area-weighted formula
# ---------------------------------------------------------------------------


def _check_mean_water_fraction_matches(
    tile_name: str,
    cells: pa.Table,
    meta: dict,
) -> None:
    """meta.yaml mean_water_fraction and mean_sea_water_fraction must match
    the area-weighted formula recomputed from cells.parquet.

    Formula: sum(fraction * cell_area) / sum(cell_area)
    |stored - computed| < EPS_RATIO for both fractions.

    Per spec §12.1 #8. EPS_RATIO = 1e-9 (alpha structural-boundary EPSILON).
    """
    wf_col = cells.column("water_fraction").to_pylist()
    sea_col = cells.column("sea_water_fraction").to_pylist()
    area_col = cells.column("cell_area_admin_clipped_m2").to_pylist()

    total_area = sum(a for a in area_col if a is not None)
    if total_area == 0.0:
        # Degenerate case: no area; skip formula check (division by zero)
        return

    computed_wf = (
        sum((wf or 0.0) * (a or 0.0) for wf, a in zip(wf_col, area_col, strict=True)) / total_area
    )
    computed_sea = (
        sum((sea or 0.0) * (a or 0.0) for sea, a in zip(sea_col, area_col, strict=True))
        / total_area
    )

    aggregates = meta.get("aggregates", {})
    stored_wf = aggregates.get("mean_water_fraction")
    stored_sea = aggregates.get("mean_sea_water_fraction")

    if stored_wf is None or abs(stored_wf - computed_wf) >= EPS_RATIO:
        raise TileValidationError(
            tile=tile_name,
            invariant="mean_water_fraction_matches",
            failed_row={},
            detail={
                "field": "mean_water_fraction",
                "stored": stored_wf,
                "computed": computed_wf,
            },
        )

    if stored_sea is None or abs(stored_sea - computed_sea) >= EPS_RATIO:
        raise TileValidationError(
            tile=tile_name,
            invariant="mean_water_fraction_matches",
            failed_row={},
            detail={
                "field": "mean_sea_water_fraction",
                "stored": stored_sea,
                "computed": computed_sea,
            },
        )


# ---------------------------------------------------------------------------
# Invariant 9: cell_area_admin_clipped_m2 > EPS_AREA_M2 (alpha structural boundary)
# ---------------------------------------------------------------------------


def _check_cell_area_positive(tile_name: str, cells: pa.Table) -> None:
    """Every kept cell's area must be > EPS_AREA_M2 (structurally non-zero).

    Per spec §12.1 #9 + §4.3 alpha (structural-boundary comparison).
    EPS_AREA_M2 = 1e-6 m².
    """
    ci_col = cells.column("cell_i").to_pylist()
    cj_col = cells.column("cell_j").to_pylist()
    area_col = cells.column("cell_area_admin_clipped_m2").to_pylist()

    for idx in range(cells.num_rows):
        area = area_col[idx]
        area_val = 0.0 if area is None else area

        if area_val <= EPS_AREA_M2:
            raise TileValidationError(
                tile=tile_name,
                invariant="cell_area_positive",
                failed_row={"cell_i": ci_col[idx], "cell_j": cj_col[idx], "row_index": idx},
                detail={
                    "area": area,
                    "required": f"> EPS_AREA_M2 ({EPS_AREA_M2})",
                },
            )


# ---------------------------------------------------------------------------
# Invariant 10: NaN-free on every numeric column (standalone for non-water columns)
# ---------------------------------------------------------------------------


def _check_nan_free_numeric_columns(
    tile_name: str,
    cells: pa.Table,
    features: pa.Table,
    crossings: pa.Table,
) -> None:
    """NaN-free check on every numeric column NOT already covered by invariant #5.

    Per spec §12.1 #10 + §14.3 NaN policy. Water-fraction columns
    (water_fraction, sea_water_fraction) are folded into invariant #5's single
    combined pass; do NOT double-scan them here.

    Checked columns:
    - cells.parquet: cell_area_admin_clipped_m2
    - features.parquet: bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, sea_overlap_fraction
    - crossings.parquet: edge_position_m, edge_extent_length_m

    Note: pyarrow nulls come through as None (not NaN); only check NaN if value is not None.
    """
    checks: list[tuple[str, pa.Table, list[str]]] = [
        ("cells.parquet", cells, ["cell_area_admin_clipped_m2"]),
        (
            "features.parquet",
            features,
            ["bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y", "sea_overlap_fraction"],
        ),
        ("crossings.parquet", crossings, ["edge_position_m", "edge_extent_length_m"]),
    ]

    for parquet_name, table, columns in checks:
        # Pre-extract columns to avoid re-reading per row
        col_data: dict[str, list] = {col: table.column(col).to_pylist() for col in columns}

        for row_idx in range(table.num_rows):
            for col_name in columns:
                val = col_data[col_name][row_idx]
                if val is not None and math.isnan(val):
                    raise TileValidationError(
                        tile=tile_name,
                        invariant="nan_free_numeric_columns",
                        failed_row={"row_index": row_idx},
                        detail={
                            "file": parquet_name,
                            "column": col_name,
                            "value": val,
                        },
                    )
