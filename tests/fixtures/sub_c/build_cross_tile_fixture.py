"""Cross-tile micro-fixture for sub-C validator failure-mode tests.

Builds a minimal 2-tile synthetic region used ONLY by the cross-tile validator
failure-mode tests in Task 16 (test_cross_tile_validator.py).

Design (spec §13.2 P5 cross-tile micro-fixture):
  - admin polygon spans exactly 2 adjacent tiles: (13, 14) and (14, 14)
  - each tile has exactly 1 cell with 1 feature
  - simple, deterministic: used to construct corruption scenarios

Tile layout:
  Tile (13,14): SVY21 box (26000, 28000) to (28000, 30000); 1 building in cell (0,0)
  Tile (14,14): SVY21 box (28000, 28000) to (30000, 30000); 1 building in cell (0,0)

The admin polygon in 4326 is the SVY21 union box (26000, 28000) to (30000, 30000)
reprojected to EPSG:4326.  After the pipeline reprojects and partitions, exactly
two tiles are produced.

Corruption scenarios for Task 16's named tests:
  - test_cross_tile_validator_detects_orphan_tile_dir:
      caller deletes one tile's provenance.yaml after extraction but leaves the
      tile dir; then registers an extra dir in manifest; validator should flag it.
  - test_cross_tile_validator_detects_missing_tile_dir:
      caller deletes an entire tile dir that the manifest references.
  - test_cross_tile_validator_detects_provenance_sha256_mismatch:
      caller mutates a tile's cells.parquet byte; provenance sha mismatch fires.
  - test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun:
      caller re-runs extract_region for one tile dir (manually) but does not
      update manifest.yaml; cross-tile validator detects stale manifest entry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyproj
from shapely import wkb as shapely_wkb
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

# ---------------------------------------------------------------------------
# CRS helpers (same pattern as build_torture_tile.py)
# ---------------------------------------------------------------------------

_TRANSFORMER_SVY21_TO_4326 = pyproj.Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)


def _svy21_to_4326(geom: BaseGeometry) -> BaseGeometry:
    return transform(_TRANSFORMER_SVY21_TO_4326.transform, geom)


def _wkb(geom_4326: BaseGeometry) -> bytes:
    return shapely_wkb.dumps(geom_4326, hex=False, byte_order=1)


# ---------------------------------------------------------------------------
# Tile constants
# ---------------------------------------------------------------------------

# Two adjacent tiles sharing the same j=14 row.
CROSS_TILE_LEFT_I: int = 13
CROSS_TILE_RIGHT_I: int = 14
CROSS_TILE_J: int = 14
TILE_SIZE_M: float = 2000.0

# Absolute SVY21 origins.
#   Left tile (13,14): (26000, 28000) to (28000, 30000)
#   Right tile (14,14): (28000, 28000) to (30000, 30000)
LEFT_ORIGIN_X: float = CROSS_TILE_LEFT_I * TILE_SIZE_M  # 26000
RIGHT_ORIGIN_X: float = CROSS_TILE_RIGHT_I * TILE_SIZE_M  # 28000
ORIGIN_Y: float = CROSS_TILE_J * TILE_SIZE_M  # 28000

# DECISION: admin polygon is inset 1 m from exact tile boundaries.
# Reprojecting SVY21 → 4326 → SVY21 introduces FP drift (~1e-6 m) that shifts
# the bounding box into adjacent tile(s).  A 1 m inset keeps the round-trip
# bounds safely within tiles (13,14) and (14,14) without spilling into (12,14).
_ADMIN_INSET_M: float = 1.0


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _make_buildings_table(id_: str, point_svy21: Point) -> pa.Table:
    """Single-row buildings table for one tile."""
    geom_4326 = _svy21_to_4326(point_svy21)
    return pa.table(
        {
            "id": [id_],
            "class": ["residential"],
            "subtype": ["residential"],
            "geometry": [_wkb(geom_4326)],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("subtype", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _empty_transportation() -> pa.Table:
    return pa.table(
        {"id": [], "class": [], "geometry": []},
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _empty_places() -> pa.Table:
    cats_type = pa.struct(
        [
            pa.field("primary", pa.string()),
            pa.field("alternate", pa.list_(pa.string())),
        ]
    )
    return pa.table(
        {"id": [], "categories": [], "geometry": []},
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("categories", cats_type),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _empty_base() -> pa.Table:
    return pa.table(
        {"id": [], "class": [], "subtype": [], "geometry": []},
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("subtype", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Region builder
# ---------------------------------------------------------------------------


def build_cross_tile_micro_region() -> SimpleNamespace:
    """Build a synthetic _RegionLike object spanning 2 adjacent tiles.

    Returns a SimpleNamespace satisfying the _RegionLike Protocol:
      .name          : str
      .themes        : dict[str, pa.Table]
      .admin_polygon : shapely BaseGeometry in EPSG:4326

    After extract_region runs this region, the manifest will contain exactly
    2 tile dirs:
      tile=EPSG3414_i13_j14
      tile=EPSG3414_i14_j14

    This is the starting point for all cross-tile-validator failure-mode tests.
    Each test copies the output to a fresh tmp_path and applies a specific
    corruption before calling the cross-tile validator.
    """
    # Admin polygon spans both tiles in SVY21; reprojected to 4326.
    # DECISION: inset 1 m from the exact left/bottom boundaries so FP round-trip
    # (SVY21→4326→SVY21) does not produce a bbox that extends left into tile (12,14).
    # The right boundary is 29999 (inset from 30000) for the same reason.
    combined_svy21 = shapely_box(
        LEFT_ORIGIN_X + _ADMIN_INSET_M,  # 26001 (keeps left boundary in tile 13)
        ORIGIN_Y + _ADMIN_INSET_M,  # 28001
        RIGHT_ORIGIN_X + TILE_SIZE_M - _ADMIN_INSET_M,  # 29999
        ORIGIN_Y + TILE_SIZE_M - _ADMIN_INSET_M,  # 29999
    )
    admin_polygon_4326 = _svy21_to_4326(combined_svy21)

    # One simple point feature per tile (a building in cell (0,0) of each tile).
    # Cell (0,0) of tile (13,14): absolute SVY21 (26000..26250) x (28000..28250)
    # place feature at (26100, 28100)
    left_bldg = _make_buildings_table(
        id_="cross_tile_left_bldg",
        point_svy21=Point(26100, 28100),
    )

    # Cell (0,0) of tile (14,14): absolute SVY21 (28000..28250) x (28000..28250)
    # place feature at (28100, 28100)
    right_bldg = _make_buildings_table(
        id_="cross_tile_right_bldg",
        point_svy21=Point(28100, 28100),
    )

    # Merge both buildings into a single table (both tiles share the same theme
    # in the region's themes dict; the orchestrator filters per-tile at extract time).
    all_buildings = pa.concat_tables([left_bldg, right_bldg])

    themes = {
        "buildings": all_buildings,
        "transportation": _empty_transportation(),
        "places": _empty_places(),
        "base": _empty_base(),
    }

    return SimpleNamespace(
        name="cross_tile_micro_synthetic",
        themes=themes,
        admin_polygon=admin_polygon_4326,
        projected_crs="EPSG:3414",
    )
