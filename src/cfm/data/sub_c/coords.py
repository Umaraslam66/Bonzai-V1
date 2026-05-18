"""Coordinate handling for sub-C: reprojection, tile/cell partitioning, densification, clipping.

Per spec §7:
- §7.1 EPSG:3414 (SVY21) for Singapore; polymorphic per region.
- §7.2 CRS-origin-aligned 2km tile grid; half-open intervals.
- §7.3 Reproject everything to SVY21 first, then clip in SVY21.
- §7.4 Polygon densification: no-op for Singapore (max edge 775m < cell size); function
  signature locked for Sweden enrollment to pass max_edge_length_m without re-opening this code.
"""

from __future__ import annotations

import math

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

SVY21_EPSG_CODE: int = 3414  # Singapore SVY21 / Singapore TM
TILE_SIZE_M: int = 2000  # per spec §7.2
CELL_SIZE_M: int = 250  # per spec §7.2 (8x8 grid per tile)


# Reusable transformer; constructed once at module import for determinism.
_TRANSFORMER_4326_TO_SVY21 = pyproj.Transformer.from_crs(
    "EPSG:4326", f"EPSG:{SVY21_EPSG_CODE}", always_xy=True
)


def reproject_lonlat_to_svy21(lon: float, lat: float) -> tuple[float, float]:
    """Project (lon, lat) in EPSG:4326 to (easting, northing) in EPSG:3414 SVY21.

    Determinism: pyproj.Transformer is constructed once per module load; the
    transformation is a Transverse Mercator formula (no datum grid for SVY21);
    output is byte-deterministic given fixed input. See spec §14.1.
    """
    x, y = _TRANSFORMER_4326_TO_SVY21.transform(lon, lat)
    return float(x), float(y)


def reproject_geometry_to_svy21(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a shapely geometry from EPSG:4326 to EPSG:3414 SVY21.

    Z coordinates, if present, are dropped; output is always 2D.
    """
    return shapely_transform(_TRANSFORMER_4326_TO_SVY21.transform, geom)


def tile_id_from_svy21(x: float, y: float) -> tuple[int, int]:
    """Map an SVY21 (easting, northing) point to its (tile_i, tile_j).

    Half-open convention per spec §7.2: tile (i, j) covers
    [i*TILE_SIZE_M, (i+1)*TILE_SIZE_M) x [j*TILE_SIZE_M, (j+1)*TILE_SIZE_M).
    A point at exactly x = i*TILE_SIZE_M lands in tile i (not i-1).
    """
    i = math.floor(x / TILE_SIZE_M)
    j = math.floor(y / TILE_SIZE_M)
    return int(i), int(j)


def cell_id_within_tile(x_in_tile: float, y_in_tile: float) -> tuple[int, int]:
    """Map an in-tile metric coordinate (0 <= x_in_tile < TILE_SIZE_M)
    to its (cell_i, cell_j) within the 8x8 grid. Half-open at cell boundaries
    per spec §7.2; a point at x = c*CELL_SIZE_M lands in cell c (not c-1).
    """
    ci = math.floor(x_in_tile / CELL_SIZE_M)
    cj = math.floor(y_in_tile / CELL_SIZE_M)
    return ci, cj


def densify_polygon(
    polygon: BaseGeometry,
    max_edge_length_m: float | None,
) -> BaseGeometry:
    """If max_edge_length_m is None, return polygon unchanged (Singapore no-op
    per spec §7.4 — max edge 775m < cell quantization scale).

    Otherwise insert vertices on every edge longer than max_edge_length_m
    so the densified polygon has no edge exceeding the threshold. Sweden
    enrollment passes a real value without re-opening sub-C code.
    """
    if max_edge_length_m is None:
        return polygon

    return polygon.segmentize(max_segment_length=max_edge_length_m)


def clip_to_admin_polygon(
    features: list[BaseGeometry],
    admin_polygon: BaseGeometry,
) -> list[BaseGeometry]:
    """Intersect each feature with admin_polygon (both in SVY21 per spec §7.3).

    Returns the kept sub-geometries (in input order). Empty intersections
    are dropped. Order is preserved so callers can re-associate with
    feature attributes by index.
    """
    out: list[BaseGeometry] = []
    for f in features:
        clipped = f.intersection(admin_polygon)
        if not clipped.is_empty:
            out.append(clipped)
    return out


def partition_into_tiles(
    admin_polygon: BaseGeometry,
) -> dict[tuple[int, int], BaseGeometry]:
    """For each 2km x 2km tile that intersects admin_polygon, return the
    intersection of the tile box with the admin polygon as the tile's
    admin-clipped footprint. Result is sorted by (tile_i, tile_j) for
    byte-determinism (spec §11.7 manifest tiles[] sort).
    """
    from shapely.geometry import box as shapely_box

    min_x, min_y, max_x, max_y = admin_polygon.bounds
    min_i = math.floor(min_x / TILE_SIZE_M)
    min_j = math.floor(min_y / TILE_SIZE_M)
    max_i = math.floor((max_x - 1e-9) / TILE_SIZE_M)
    max_j = math.floor((max_y - 1e-9) / TILE_SIZE_M)

    inventory: dict[tuple[int, int], BaseGeometry] = {}
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            tile_box = shapely_box(
                i * TILE_SIZE_M,
                j * TILE_SIZE_M,
                (i + 1) * TILE_SIZE_M,
                (j + 1) * TILE_SIZE_M,
            )
            intersection = tile_box.intersection(admin_polygon)
            if not intersection.is_empty:
                inventory[(i, j)] = intersection

    return dict(sorted(inventory.items()))
