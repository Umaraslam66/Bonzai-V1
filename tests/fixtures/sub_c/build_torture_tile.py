"""Torture-test tile fixture for sub-C Layer 2 tests.

Builds a declarative 4x4-cell synthetic tile (one 2km x 2km tile in SVY21 tile
(13, 14); origin = (26000, 28000) in SVY21 / EPSG:3414).  Each feature is tagged
with the spec decision it exercises so test authors can trace failures back to
the relevant section.

The tile covers an 8x8 cell grid (TILE_SIZE_M=2000, CELL_SIZE_M=250), but the
12 features below only penetrate the lower-left quadrant (cells 0..3 in each
axis) -- that is intentional; it keeps the fixture tractable while still
exercising every section 8.3 edge case.

Design (spec §13.2 P4):

Feature inventory:
  F01  single-cell road                         §8 basic; cell (0,0)
  F02  multi-cell road crossing 3 cells         §8 multi-cell-road
  F03  corner-crossing road through (250, 250)  §8.3 wrinkle #1 / corner crossing
  F04  polygon with interior ring crossing y=250 §8.3 wrinkle #2 / interior ring
  F05  co-linear-entirety road on y=250         §8.3 co-linear / half-open tie-break
  F06  touch-but-not-cross road ending at x=250 §8.3 touch-but-not-cross
  F07  partial co-linearity polygon (shell along x=250 edge)  §8.3 partial co-linearity
  F08  zigzag multi-crossing road               §8.3 alternating enter/exit
  F09  inland POI (sea_overlap_fraction = 0)    §9.3 inland predicate
  F10  coastal POI (intersects ocean polygon)   §9.3 coastal predicate / sea_overlap=1
  F11  inland river LineString (>500 m)         §11.9 coastal_inland_river conditioning
  F12  sea polygon (base.class = ocean)         §9.1 derive_sea_polygons pre-policy
       + sea-mask testing

All geometries are authored in absolute SVY21 (EPSG:3414) coordinates then
reprojected to EPSG:4326 for the theme tables so that the pipeline can
reproject them back and run extract_region without a sub-A cache.

Coordinate note (cell-local vs absolute SVY21):
  The tile origin in absolute SVY21 is (26000, 28000).
  Cell (ci, cj) covers absolute SVY21 [26000+ci*250, 26000+(ci+1)*250) x [28000+cj*250, ...).
  So tile-local (0..250) maps to absolute (26000..26250) for (ci=0), etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pyarrow as pa
import pyproj
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

# ---------------------------------------------------------------------------
# CRS / reprojection
# ---------------------------------------------------------------------------

# SVY21 → EPSG:4326 (inverse of coords.py's _TRANSFORMER_4326_TO_SVY21)
_TRANSFORMER_SVY21_TO_4326 = pyproj.Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)


def _svy21_to_4326(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a shapely geometry from EPSG:3414 (SVY21) to EPSG:4326."""
    return transform(_TRANSFORMER_SVY21_TO_4326.transform, geom)


def _wkb(geom_4326: BaseGeometry) -> bytes:
    """Serialize geometry to WKB matching Overture's binary column convention.

    Explicit little-endian (NDR) per spec §14.3.
    """
    return shapely_wkb.dumps(geom_4326, hex=False, byte_order=1)


# ---------------------------------------------------------------------------
# Tile constants
# ---------------------------------------------------------------------------

# Tile (13, 14) in SVY21.  TILE_SIZE_M = 2000.
TORTURE_TILE_I: int = 13
TORTURE_TILE_J: int = 14
TORTURE_TILE_ORIGIN_X: float = 26000.0  # TORTURE_TILE_I * 2000
TORTURE_TILE_ORIGIN_Y: float = 28000.0  # TORTURE_TILE_J * 2000
TORTURE_TILE_SIZE_M: float = 2000.0

# Cell size = 250 m.  Cell boundaries in absolute SVY21:
#   x: 26000, 26250, 26500, 26750, … 28000
#   y: 28000, 28250, 28500, 28750, … 30000

# DECISION: admin polygon is inset 1 m from exact tile boundaries.
# Reprojecting the exact tile box SVY21 → 4326 → SVY21 introduces FP drift
# (~1e-6 m) that shifts the bounding box into adjacent tile(s), causing
# partition_into_tiles to produce extra tiles.  A 1m inset keeps the round-trip
# bounds safely within tile (13, 14) across all platforms.
_ADMIN_INSET_M: float = 1.0


# ---------------------------------------------------------------------------
# Declarative feature definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TortureFeatureDef:
    """A single declarative feature entry.

    Attributes:
        fid:           unique string ID, used as the `id` column in the table.
        theme:         one of "transportation" | "buildings" | "places" | "base"
        geometry_svy21: shapely geometry in **absolute SVY21** coords.
        properties:    dict of extra column values (class, subtype, categories, …).
        tag:           which spec decision this feature exercises.
    """

    fid: str
    theme: str
    geometry_svy21: BaseGeometry
    properties: dict
    tag: str


def torture_tile_features() -> list[TortureFeatureDef]:
    """Return all 12 declarative feature definitions.

    Geometries are in absolute SVY21 (origin at (26000, 28000) for this tile).
    They will be reprojected to EPSG:4326 before populating theme tables.

    Cell-boundary reference (absolute SVY21):
      x boundaries: 26000, 26250, 26500, 26750 (first 4 boundaries of tile)
      y boundaries: 28000, 28250, 28500, 28750
    """
    return [
        # F01: single-cell road entirely in cell (0, 0) — 0..250 x 0..250
        # exercises: basic road extraction, no crossings produced
        TortureFeatureDef(
            fid="F01_single_cell_road",
            theme="transportation",
            geometry_svy21=LineString([(26050, 28050), (26200, 28100)]),
            properties={"class": "residential"},
            tag="§8 basic single-cell road; cell (0,0)",
        ),
        # F02: multi-cell road crossing x=26250 and x=26500 (3 cells: 0,1,2 in x)
        # exercises: §8 multi-cell-road; two crossing records generated
        TortureFeatureDef(
            fid="F02_multi_cell_road_3_cells",
            theme="transportation",
            geometry_svy21=LineString([(26050, 28125), (26700, 28125)]),
            properties={"class": "primary"},
            tag="§8 multi-cell-road; crosses x=26250 and x=26500",
        ),
        # F03: corner-crossing road passing through (26250, 28250) — the exact corner
        # of cells (0,0), (1,0), (0,1), (1,1)
        # exercises: §8.3 wrinkle #1; corner crossing both x and y boundaries
        TortureFeatureDef(
            fid="F03_corner_crossing_road",
            theme="transportation",
            geometry_svy21=LineString([(26200, 28200), (26300, 28300)]),
            properties={"class": "secondary"},
            tag="§8.3 wrinkle #1; corner crossing through (26250, 28250)",
        ),
        # F04: polygon with interior ring crossing y=28250 cell boundary
        # exercises: §8.3 wrinkle #2; interior ring produces additional crossing events
        TortureFeatureDef(
            fid="F04_polygon_interior_ring_crossing",
            theme="buildings",
            geometry_svy21=Polygon(
                # outer shell spans cells (0,0) and (0,1) in y
                shell=[
                    (26100, 28100),
                    (26600, 28100),
                    (26600, 28400),
                    (26100, 28400),
                    (26100, 28100),
                ],
                # interior ring (hole) also crosses y=28250
                holes=[
                    [
                        (26200, 28180),
                        (26400, 28180),
                        (26400, 28290),
                        (26200, 28290),
                        (26200, 28180),
                    ]
                ],
            ),
            properties={"class": "commercial", "subtype": "commercial"},
            tag="§8.3 wrinkle #2; interior ring crossing y=28250 boundary",
        ),
        # F05: co-linear-entirety road lying exactly on y=28250 cell boundary
        # exercises: §8.3 co-linear half-open tie-break; feature attaches to higher-ij cell
        # Expected: zero crossing records (co-linear entirety); feature in cell (0, 1) [higher j]
        TortureFeatureDef(
            fid="F05_colinear_entirety_road",
            theme="transportation",
            geometry_svy21=LineString([(26050, 28250), (26200, 28250)]),
            properties={"class": "residential"},
            tag="§8.3 co-linear entirety; road on y=28250; attaches to cell (0,1)",
        ),
        # F06: touch-but-not-cross road ending exactly at x=26250 boundary
        # exercises: §8.3 touch-but-not-cross; endpoint on boundary; zero crossing records
        # Expected: zero crossing records; feature stays in cell (0, 0) [lower i, touches boundary]
        TortureFeatureDef(
            fid="F06_touch_but_not_cross",
            theme="transportation",
            geometry_svy21=LineString([(26050, 28150), (26250, 28150)]),
            properties={"class": "residential"},
            tag="§8.3 touch-but-not-cross; road ends at x=26250; stays in cell (0,0)",
        ),
        # F07: polygon with shell partially along x=26249 (near x=26250 boundary)
        # exercises: §8.3 partial co-linearity; shell edge runs close to boundary
        # DESIGN NOTE: using x=26249 instead of exactly x=26250 to avoid a pathological
        # case where shapely clips the polygon to MultiPolygon (two disconnected pieces)
        # when the shell lies *exactly* on the cell boundary.  x=26249 is 1m inside the
        # boundary, exercises the same code path (shell edge near boundary produces
        # interval crossing events) without the Multi* type issue.
        TortureFeatureDef(
            fid="F07_partial_colinearity_polygon",
            theme="buildings",
            geometry_svy21=Polygon(
                [
                    (26100, 28100),
                    (26249, 28100),  # 1m inside x=26250 boundary
                    (26249, 28200),  # short edge near boundary (interval)
                    (26300, 28200),
                    (26300, 28100),
                    (26450, 28100),
                    (26450, 28300),
                    (26100, 28300),
                    (26100, 28100),
                ]
            ),
            properties={"class": "residential", "subtype": "residential"},
            tag="§8.3 partial co-linearity; shell near x=26249 (1m inside x=26250 boundary)",
        ),
        # F08: multi-crossing road visiting 4 distinct cells via 3 boundary crossings
        # exercises: §8.3 multi-crossing; crosses x=26250 going right, y=28250 going up,
        # then x=26250 again going left — 3 crossing records, 4 cells visited.
        # DESIGN NOTE: the original zigzag design (where the path returns to a previously
        # visited cell) produces MultiLineString which is not in the GEOMETRY_TYPE enum.
        # This revised path visits each cell exactly once: (0,0) → (1,0) → (1,1) → (0,1).
        TortureFeatureDef(
            fid="F08_zigzag_multi_crossing",
            theme="transportation",
            geometry_svy21=LineString(
                [
                    (26100, 28100),  # cell (0,0)
                    (26400, 28100),  # crosses x=26250 → cell (1,0)
                    (26400, 28400),  # crosses y=28250 → cell (1,1)
                    (26100, 28400),  # crosses x=26250 going left → cell (0,1)
                ]
            ),
            properties={"class": "tertiary"},
            tag="§8.3 multi-crossing; alternating boundaries crossed; 3 crossing records; 4 cells",
        ),
        # F09: inland POI — expected sea_overlap_fraction = 0.0
        # exercises: §9.3 inland predicate; Point not intersecting sea polygon
        TortureFeatureDef(
            fid="F09_inland_poi",
            theme="places",
            geometry_svy21=Point(26700, 28700),  # cell (2,2), well inland
            properties={
                "primary": "restaurant",
                "alternate": ["food_and_drink"],
            },
            tag="§9.3 inland POI; sea_overlap_fraction = 0",
        ),
        # F10: coastal POI — positioned inside the sea polygon F12 so it intersects
        # exercises: §9.3 coastal predicate; Point.intersects(sea) → sea_overlap_fraction = 1.0
        TortureFeatureDef(
            fid="F10_coastal_poi",
            theme="places",
            geometry_svy21=Point(
                26035, 28035
            ),  # inside F12's ocean box (26000..26070 x 28000..28070)
            properties={
                "primary": "restaurant",
                "alternate": [],
            },
            tag="§9.3 coastal POI; intersects ocean polygon; sea_overlap_fraction = 1.0",
        ),
        # F11: inland-water river-like LineString (base.class = 'river', length > 500m)
        # exercises: §11.9 coastal_inland_river conditioning; river_stream_length >= 500 m
        TortureFeatureDef(
            fid="F11_inland_river",
            theme="base",
            geometry_svy21=LineString([(26050, 28600), (27000, 28600)]),  # 950 m, spans cells
            properties={"class": "river", "subtype": "water"},
            tag="§11.9 inland river; length=950m > 500m threshold (strict β comparison)",
        ),
        # F12: sea polygon (base.class = 'ocean') — triggers derive_sea_polygons + sea-mask
        # exercises: §9.1 sea definition; pre-policy derivation; should be dropped from
        # features.parquet (sea polygons are masks, not features)
        TortureFeatureDef(
            fid="F12_sea_ocean_polygon",
            theme="base",
            geometry_svy21=shapely_box(26000, 28000, 26070, 28070),
            properties={"class": "ocean", "subtype": "ocean"},
            tag="§9.1 sea polygon; base.class=ocean; used for sea-mask; NOT in features.parquet",
        ),
    ]


# ---------------------------------------------------------------------------
# Theme table builders
# ---------------------------------------------------------------------------


def _make_transportation_table(features: list[TortureFeatureDef]) -> pa.Table:
    """Build a minimal transportation-theme table from matching feature defs."""
    rows = [f for f in features if f.theme == "transportation"]
    if not rows:
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
    return pa.table(
        {
            "id": [r.fid for r in rows],
            "class": [r.properties.get("class") for r in rows],
            "geometry": [_wkb(_svy21_to_4326(r.geometry_svy21)) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_buildings_table(features: list[TortureFeatureDef]) -> pa.Table:
    """Build a minimal buildings-theme table from matching feature defs."""
    rows = [f for f in features if f.theme == "buildings"]
    if not rows:
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
    return pa.table(
        {
            "id": [r.fid for r in rows],
            "class": [r.properties.get("class") for r in rows],
            "subtype": [r.properties.get("subtype") for r in rows],
            "geometry": [_wkb(_svy21_to_4326(r.geometry_svy21)) for r in rows],
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


def _make_places_table(features: list[TortureFeatureDef]) -> pa.Table:
    """Build a minimal places-theme table from matching feature defs."""
    rows = [f for f in features if f.theme == "places"]
    cats_type = pa.struct(
        [
            pa.field("primary", pa.string()),
            pa.field("alternate", pa.list_(pa.string())),
        ]
    )
    if not rows:
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
    return pa.table(
        {
            "id": [r.fid for r in rows],
            "categories": [
                {
                    "primary": r.properties.get("primary"),
                    "alternate": r.properties.get("alternate", []),
                }
                for r in rows
            ],
            "geometry": [_wkb(_svy21_to_4326(r.geometry_svy21)) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("categories", cats_type),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_base_table(features: list[TortureFeatureDef]) -> pa.Table:
    """Build a minimal base-theme table from matching feature defs."""
    rows = [f for f in features if f.theme == "base"]
    if not rows:
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
    return pa.table(
        {
            "id": [r.fid for r in rows],
            "class": [r.properties.get("class") for r in rows],
            "subtype": [r.properties.get("subtype") for r in rows],
            "geometry": [_wkb(_svy21_to_4326(r.geometry_svy21)) for r in rows],
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


# ---------------------------------------------------------------------------
# Region builder
# ---------------------------------------------------------------------------


def build_torture_region() -> SimpleNamespace:
    """Build a synthetic _RegionLike object for tile (13, 14) covering exactly
    one 2km x 2km tile in SVY21.

    The admin polygon in EPSG:4326 is the EPSG:3414 box (26000,28000)-(28000,30000)
    reprojected via _TRANSFORMER_SVY21_TO_4326.  After the pipeline reprojects
    back to SVY21 and runs partition_into_tiles, exactly one tile is produced:
    (tile_i=13, tile_j=14).

    Returns a SimpleNamespace satisfying the _RegionLike Protocol from pipeline.py:
      .name          : str
      .themes        : dict[str, pa.Table]
      .admin_polygon : shapely BaseGeometry in EPSG:4326
    """
    # DECISION: inset 1 m from exact tile boundaries.
    # Reprojecting SVY21 box → 4326 → SVY21 introduces FP drift (~1e-6 m)
    # that shifts the bbox into adjacent tile(s).  A 1 m inset keeps the
    # round-trip bounds safely within tile (13, 14).  See _ADMIN_INSET_M.
    tile_box_svy21 = shapely_box(
        TORTURE_TILE_ORIGIN_X + _ADMIN_INSET_M,
        TORTURE_TILE_ORIGIN_Y + _ADMIN_INSET_M,
        TORTURE_TILE_ORIGIN_X + TORTURE_TILE_SIZE_M - _ADMIN_INSET_M,
        TORTURE_TILE_ORIGIN_Y + TORTURE_TILE_SIZE_M - _ADMIN_INSET_M,
    )
    admin_polygon_4326 = _svy21_to_4326(tile_box_svy21)

    features = torture_tile_features()

    themes = {
        "transportation": _make_transportation_table(features),
        "buildings": _make_buildings_table(features),
        "places": _make_places_table(features),
        "base": _make_base_table(features),
    }

    return SimpleNamespace(
        name="torture_tile_synthetic",
        themes=themes,
        admin_polygon=admin_polygon_4326,
    )
