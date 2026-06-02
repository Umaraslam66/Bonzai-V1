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
from functools import cache

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

SVY21_EPSG_CODE: int = 3414  # Singapore SVY21 / Singapore TM
TILE_SIZE_M: int = 2000  # per spec §7.2
CELL_SIZE_M: int = 250  # per spec §7.2 (8x8 grid per tile)


@cache
def _transformer_for_crs(crs: str) -> pyproj.Transformer:
    """One pyproj transformer per target CRS, constructed exactly once.

    Cached so every caller for a given CRS shares this exact object — the single
    coordinate authority for that CRS. Determinism: pyproj.Transformer for a
    conformal (Transverse Mercator / UTM) projection with no datum grid is a
    closed-form formula; output is byte-deterministic given fixed input
    (spec §14.1). Construct-once also preserves the original module-load
    determinism guarantee.
    """
    return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)


class RegionCoords:
    """Region-bound EPSG:4326 -> projected-CRS reprojection.

    Built once per city from its ``projected_crs`` (multi-region policy: one
    conformal UTM zone per city, see ``utm_epsg_for_centroid``). All of a city's
    reprojection routes through this one object. The grid/tile/cell math
    (``tile_id_from_svy21``, ``cell_id_within_tile``, ``partition_into_tiles``)
    is CRS-agnostic metre arithmetic and stays as module functions — only the
    4326->projected transformer is CRS-bound.
    """

    def __init__(self, crs: str) -> None:
        self.crs = crs
        self._transformer = _transformer_for_crs(crs)

    def reproject_lonlat(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self._transformer.transform(lon, lat)
        return float(x), float(y)

    def reproject_geometry(self, geom: BaseGeometry) -> BaseGeometry:
        """Reproject a shapely geometry; Z is dropped (output always 2D)."""
        return shapely_transform(self._transformer.transform, geom)


@cache
def region_coords(crs: str) -> RegionCoords:
    """The shared ``RegionCoords`` for ``crs`` (one instance per CRS).

    This is the single entry point both the pipeline (``region_coords(
    region.projected_crs)``) and the backward-compat SVY21 delegates below use,
    so there is never a parallel coordinate path that can drift.
    """
    return RegionCoords(crs)


def reproject_lonlat_to_svy21(lon: float, lat: float) -> tuple[float, float]:
    """[backward-compat] EPSG:4326 -> EPSG:3414 SVY21 (easting, northing).

    Thin delegate to the shared ``region_coords("EPSG:3414")`` path — NOT a
    retained private transformer — so it can never drift from the region-bound
    code. Singapore's projected_crs is EPSG:3414, so the pipeline and this helper
    are byte-identical on Singapore. See spec §14.1.
    """
    return region_coords(f"EPSG:{SVY21_EPSG_CODE}").reproject_lonlat(lon, lat)


def reproject_geometry_to_svy21(geom: BaseGeometry) -> BaseGeometry:
    """[backward-compat] Reproject a shapely geometry EPSG:4326 -> EPSG:3414.

    Thin delegate to the shared ``region_coords("EPSG:3414")`` path (one
    authority). Z coordinates, if present, are dropped; output is always 2D.
    """
    return region_coords(f"EPSG:{SVY21_EPSG_CODE}").reproject_geometry(geom)


# ---- multi-region: centroid -> conformal UTM zone selection ---------------
# ETRS89 / UTM-North EPSG codes are 25800 + zone, defined for European zones.
_ETRS89_UTM_MIN_ZONE: int = 28
_ETRS89_UTM_MAX_ZONE: int = 38
_ETRS89_UTM_NORTH_EPSG_BASE: int = 25800


def epsg_label_from_crs(crs: str) -> str:
    """Derive the tile-dir label from a CRS string: ``'EPSG:25833' -> 'EPSG25833'``.

    The label is the EPSG prefix in ``tile=EPSG{code}_i{i}_j{j}``. It is
    load-bearing, not cosmetic: the prefix disambiguates tile indices across
    cities in different UTM zones (Berlin ``i5_j3`` vs Madrid ``i5_j3``). This is
    the single source so no module re-hardcodes the literal. Backward-compat:
    ``'EPSG:3414' -> 'EPSG3414'`` (the locked Singapore tile-dir label).
    """
    if not crs.startswith("EPSG:"):
        raise ValueError(f"epsg_label_from_crs: expected 'EPSG:<code>', got {crs!r}")
    code = crs[len("EPSG:") :]
    if not code.isdigit():
        raise ValueError(f"epsg_label_from_crs: non-numeric EPSG code in {crs!r}")
    return f"EPSG{code}"


def utm_epsg_for_centroid(lon: float, lat: float) -> str:
    """Return the ETRS89/UTM-North EPSG string for a city *centroid*.

    Multi-region policy (PI lock 2026-06-02): one conformal UTM zone PER CITY,
    chosen from the centroid and applied to ALL of that city's tiles (even tiles
    spilling past the 6° zone edge stay in the centroid's zone). Conformal
    (angle-preserving) is required because the shape model is scored on
    rectilinearity; an equal-area projection would shear right angles.

    Zone = ``int((lon + 180) / 6) + 1`` (half-open lower bound, like the tile
    grid: a lon exactly on a 6° multiple maps to the higher zone). ETRS89/UTM
    -North codes are ``25800 + zone``, defined for European zones 28-38.

    Raises ``ValueError`` for coordinates outside the ETRS89 European range
    (southern hemisphere, or a zone outside 28-38) rather than silently emitting
    an undefined code. Singapore uses EPSG:3414 and never calls this; this helper
    only picks the pinned ``projected_crs`` value at region-enrollment time.
    """
    if lat < 0:
        raise ValueError(
            f"utm_epsg_for_centroid: lat={lat} is southern hemisphere; "
            "ETRS89/UTM-North policy covers Europe (northern hemisphere) only"
        )
    zone = int((lon + 180.0) / 6.0) + 1
    if not (_ETRS89_UTM_MIN_ZONE <= zone <= _ETRS89_UTM_MAX_ZONE):
        raise ValueError(
            f"utm_epsg_for_centroid: lon={lon} -> UTM zone {zone}, outside the "
            f"ETRS89 European range [{_ETRS89_UTM_MIN_ZONE}, {_ETRS89_UTM_MAX_ZONE}]"
        )
    return f"EPSG:{_ETRS89_UTM_NORTH_EPSG_BASE + zone}"


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
    # Subtract epsilon so a bbox max that lands exactly on a tile boundary
    # (e.g. max_x = 2*TILE_SIZE_M = 4000) maps to the last real tile (i=1),
    # not to an empty degenerate tile one step beyond (i=2).
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
