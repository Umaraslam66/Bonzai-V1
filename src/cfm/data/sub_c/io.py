"""Write helpers and encoding-determinism primitives for sub-C tile extraction.

Per spec 14.3:
- _PARQUET_WRITE_KWARGS: pinned writer args for byte-deterministic parquet output.
- dump_wkb: explicit little-endian (NDR) WKB serialisation.
- canonicalize_yaml: byte-deterministic YAML serialisation (sorted keys, block style).

Per spec 11.2-11.6 (Task 9):
- CellAggregate, TileMeta, TileProvenance: dataclasses for structured tile data.
- write_features_parquet, write_cells_parquet, write_crossings_parquet: parquet writers.
- write_meta_yaml, write_provenance_yaml: YAML writers.

All parquet writers pin column types via pa.schema() to guarantee deterministic
bytes across pyarrow versions. Rows are sorted by canonical key BEFORE writing.
mean_water_fraction and mean_sea_water_fraction in meta.yaml are taken
PRE-COMPUTED by the caller (area-weighted per spec 11.5:
  sum(water_fraction * cell_area_admin_clipped_m2) / sum(cell_area_admin_clipped_m2)).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from shapely import wkb
from shapely.geometry.base import BaseGeometry

# PARQUET_WRITE_KWARGS re-export is DELIBERATE: the spec-§14.3 determinism pin
# test imports it from HERE (sub-C's write surface), not from cfm.data.io.
from cfm.data.io import (
    PARQUET_WRITE_KWARGS as _PARQUET_WRITE_KWARGS,  # noqa: F401
)
from cfm.data.io import (
    canonicalize_yaml,
    write_parquet,
)

# ---------------------------------------------------------------------------
# Pinned pyarrow schemas (spec §11.2, §11.3, §8.2)
# ---------------------------------------------------------------------------

#: features.parquet — spec §11.2 (15 columns)
_FEATURES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("feature_class", pa.int8()),
        pa.field("source_feature_id", pa.string()),
        pa.field("geometry", pa.binary()),
        pa.field("geometry_type", pa.int8()),
        pa.field("bbox_min_x", pa.float64()),
        pa.field("bbox_min_y", pa.float64()),
        pa.field("bbox_max_x", pa.float64()),
        pa.field("bbox_max_y", pa.float64()),
        pa.field("class_raw", pa.string()),  # nullable string (None allowed)
        pa.field("subtype_raw", pa.string()),  # nullable string
        pa.field("categories_primary", pa.string()),  # nullable string
        pa.field("categories_alternate", pa.list_(pa.string())),  # nullable list<string>
        pa.field("sea_overlap_fraction", pa.float64()),
    ]
)

#: cells.parquet — spec §11.3 (6 columns)
_CELLS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("water_fraction", pa.float64()),
        pa.field("sea_water_fraction", pa.float64()),
        pa.field("cell_area_admin_clipped_m2", pa.float64()),
        pa.field("kept_features_count", pa.int32()),
    ]
)

#: crossings.parquet — spec §8.2 (8 columns)
_CROSSINGS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("source_feature_id", pa.string()),
        pa.field("lower_cell_i", pa.int8()),
        pa.field("lower_cell_j", pa.int8()),
        pa.field("axis", pa.int8()),
        pa.field("ring_index", pa.int16()),
        pa.field("event_type", pa.int8()),
        pa.field("edge_position_m", pa.float64()),
        pa.field("edge_extent_length_m", pa.float64()),
    ]
)


# ---------------------------------------------------------------------------
# Dataclasses for structured tile data (Task 9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellAggregate:
    """Per-cell row for cells.parquet (spec §11.3).

    cell_area_admin_clipped_m2 must be > EPS_AREA_M2 for kept cells.
    Cells dropped by §9.2 sea-mask rule are NOT included.
    """

    cell_i: int
    cell_j: int
    water_fraction: float
    sea_water_fraction: float
    cell_area_admin_clipped_m2: float
    kept_features_count: int


@dataclass(frozen=True)
class TileMeta:
    """Contents of meta.yaml for one tile (spec §11.5).

    aggregates keys:
      kept_cell_count, sea_mask_drop_count, mean_water_fraction,
      mean_sea_water_fraction, feature_count_by_class (keys: road/building/poi/base),
      crossing_count.

    config keys:
      sliver_drop_rule (str).

    conditioning_per_tile keys:
      admin_region, morphology_class, era_class, coastal_inland_river (int8 enum),
      population_density_bucket (null at sub-C; sub-D owner),
      population_density_bucket_owner.

    mean_water_fraction and mean_sea_water_fraction must be pre-computed
    area-weighted by the caller: sum(w * area) / sum(area).
    """

    schema_version: str
    tile_i: int
    tile_j: int
    aggregates: dict  # kept_cell_count, sea_mask_drop_count, mean_water_fraction, ...
    config: dict  # sliver_drop_rule, ...
    conditioning_per_tile: dict


@dataclass(frozen=True)
class TileProvenance:
    """Contents of provenance.yaml for one tile (spec §11.6).

    extraction keys: commit_sha, extracted_utc, rerun_count, rerun_reason.
    inputs keys: release, admin_polygon_sha256, policy_yaml_sha256, vocab_yaml_sha256.
    outputs keys: cells_parquet_sha256, features_parquet_sha256,
                  crossings_parquet_sha256, meta_yaml_sha256.

    extracted_utc is EXCLUDED from sha computation per spec §14.6.
    rerun_reason IS included in sha (F2 spec fix - audit-trail purpose).
    """

    schema_version: str
    tile_i: int
    tile_j: int
    crs: str
    extraction: dict  # commit_sha, extracted_utc, rerun_count, rerun_reason
    inputs: dict  # release, admin_polygon_sha256, policy_yaml_sha256, vocab_yaml_sha256
    outputs: dict  # cells_parquet_sha256, features_parquet_sha256,
    # crossings_parquet_sha256, meta_yaml_sha256


# ---------------------------------------------------------------------------
# Row dataclass for features.parquet (used by write_features_parquet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureRow:
    """One row for features.parquet (spec §11.2, 15 columns).

    geometry is a shapely BaseGeometry in cell-local SVY21 coordinates.
    geometry_type and bbox_* must be PRE-COMPUTED by the caller and supplied on
    the dataclass; the writer reads them directly without re-deriving from geometry.
    feature_class must be the int8 enum code (0=road, 1=building, 2=poi, 3=base).
    geometry_type must be the int8 enum code (0=Point, 1=LineString, 2=Polygon,
    3=MultiPoint, 4=MultiLineString, 5=MultiPolygon).
    """

    cell_i: int
    cell_j: int
    feature_class: int  # int8 enum: 0=road, 1=building, 2=poi, 3=base
    source_feature_id: str
    geometry: BaseGeometry  # shapely; serialised to WKB by writer
    # int8 enum: 0=Point, 1=LineString, 2=Polygon, 3=MultiPoint,
    # 4=MultiLineString, 5=MultiPolygon
    geometry_type: int
    bbox_min_x: float
    bbox_min_y: float
    bbox_max_x: float
    bbox_max_y: float
    class_raw: str | None  # nullable; road/building/base; null for poi
    subtype_raw: str | None  # nullable; building/base; null for road/poi
    categories_primary: str | None  # nullable; poi only
    categories_alternate: list[str] | None  # nullable; poi only
    sea_overlap_fraction: float


# ---------------------------------------------------------------------------
# Public API — primitive helpers (Task 8)
# ---------------------------------------------------------------------------


def dump_wkb(geom: BaseGeometry) -> bytes:
    """Serialise *geom* to WKB bytes with explicit little-endian (NDR) byte order.

    Per spec §14.3: byte_order=1 forces NDR regardless of platform default.
    The first byte of the result is always 0x01.
    """
    return wkb.dumps(geom, hex=False, byte_order=1)


# ---------------------------------------------------------------------------
# Public API - per-tile writers (Task 9, spec 11.2-11.6)
# ---------------------------------------------------------------------------


def write_features_parquet(features: list[FeatureRow], path: Path) -> None:
    """Write features.parquet with pinned schema and canonical sort key.

    Sort key (spec §11.2): (cell_i, cell_j, feature_class, source_feature_id).
    Column types are pinned via _FEATURES_SCHEMA; no pyarrow type inference.
    geometry is serialised to WKB little-endian bytes via dump_wkb.
    """
    sorted_rows = sorted(
        features,
        key=lambda r: (r.cell_i, r.cell_j, r.feature_class, r.source_feature_id),
    )

    columns: dict[str, list] = {
        "cell_i": [r.cell_i for r in sorted_rows],
        "cell_j": [r.cell_j for r in sorted_rows],
        "feature_class": [r.feature_class for r in sorted_rows],
        "source_feature_id": [r.source_feature_id for r in sorted_rows],
        "geometry": [dump_wkb(r.geometry) for r in sorted_rows],
        "geometry_type": [r.geometry_type for r in sorted_rows],
        "bbox_min_x": [r.bbox_min_x for r in sorted_rows],
        "bbox_min_y": [r.bbox_min_y for r in sorted_rows],
        "bbox_max_x": [r.bbox_max_x for r in sorted_rows],
        "bbox_max_y": [r.bbox_max_y for r in sorted_rows],
        "class_raw": [r.class_raw for r in sorted_rows],
        "subtype_raw": [r.subtype_raw for r in sorted_rows],
        "categories_primary": [r.categories_primary for r in sorted_rows],
        "categories_alternate": [r.categories_alternate for r in sorted_rows],
        "sea_overlap_fraction": [r.sea_overlap_fraction for r in sorted_rows],
    }

    table = pa.Table.from_pydict(columns, schema=_FEATURES_SCHEMA)
    write_parquet(table, path)


def write_cells_parquet(cells: list[CellAggregate], path: Path) -> None:
    """Write cells.parquet with pinned schema and canonical sort key.

    Sort key (spec §11.3): (cell_i, cell_j).
    Column types are pinned via _CELLS_SCHEMA; no pyarrow type inference.
    """
    sorted_rows = sorted(cells, key=lambda c: (c.cell_i, c.cell_j))

    columns: dict[str, list] = {
        "cell_i": [r.cell_i for r in sorted_rows],
        "cell_j": [r.cell_j for r in sorted_rows],
        "water_fraction": [r.water_fraction for r in sorted_rows],
        "sea_water_fraction": [r.sea_water_fraction for r in sorted_rows],
        "cell_area_admin_clipped_m2": [r.cell_area_admin_clipped_m2 for r in sorted_rows],
        "kept_features_count": [r.kept_features_count for r in sorted_rows],
    }

    table = pa.Table.from_pydict(columns, schema=_CELLS_SCHEMA)
    write_parquet(table, path)


def write_crossings_parquet(crossings: list, path: Path) -> None:
    """Write crossings.parquet with pinned schema and canonical sort key.

    Accepts CrossingRecord instances (from cfm.data.sub_c.geom) or any object
    with the 8 attributes matching spec §8.2.

    Sort key (spec §8.2): (lower_cell_i, lower_cell_j, axis, source_feature_id,
                           ring_index, edge_position_m, event_type).
    Column types are pinned via _CROSSINGS_SCHEMA; no pyarrow type inference.
    """
    sorted_rows = sorted(
        crossings,
        key=lambda c: (
            c.lower_cell_i,
            c.lower_cell_j,
            c.axis,
            c.source_feature_id,
            c.ring_index,
            c.edge_position_m,
            c.event_type,
        ),
    )

    columns: dict[str, list] = {
        "source_feature_id": [r.source_feature_id for r in sorted_rows],
        "lower_cell_i": [r.lower_cell_i for r in sorted_rows],
        "lower_cell_j": [r.lower_cell_j for r in sorted_rows],
        "axis": [r.axis for r in sorted_rows],
        "ring_index": [r.ring_index for r in sorted_rows],
        "event_type": [r.event_type for r in sorted_rows],
        "edge_position_m": [r.edge_position_m for r in sorted_rows],
        "edge_extent_length_m": [r.edge_extent_length_m for r in sorted_rows],
    }

    table = pa.Table.from_pydict(columns, schema=_CROSSINGS_SCHEMA)
    write_parquet(table, path)


def write_meta_yaml(meta: TileMeta, path: Path) -> None:
    """Write meta.yaml for one tile (spec §11.5).

    Serialised via canonicalize_yaml for byte-deterministic output.
    feature_count_by_class keys are stored as string labels (road/building/poi/base)
    so the YAML is human-readable; canonicalize_yaml sorts keys alphabetically.
    """
    data = {
        "schema_version": meta.schema_version,
        "tile_i": meta.tile_i,
        "tile_j": meta.tile_j,
        "aggregates": meta.aggregates,
        "config": meta.config,
        "conditioning_per_tile": meta.conditioning_per_tile,
    }
    path.write_text(canonicalize_yaml(data), encoding="utf-8")


def write_provenance_yaml(provenance: TileProvenance, path: Path) -> None:
    """Write provenance.yaml for one tile (spec §11.6).

    Serialised via canonicalize_yaml for byte-deterministic output.
    extracted_utc is EXCLUDED from sha computation per EXCLUDED_FROM_SHA
    (but it IS written to the file — it is just ignored when hashing).
    """
    data = {
        "schema_version": provenance.schema_version,
        "tile_i": provenance.tile_i,
        "tile_j": provenance.tile_j,
        "crs": provenance.crs,
        "extraction": provenance.extraction,
        "inputs": provenance.inputs,
        "outputs": provenance.outputs,
    }
    path.write_text(canonicalize_yaml(data), encoding="utf-8")
