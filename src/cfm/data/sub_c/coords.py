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
