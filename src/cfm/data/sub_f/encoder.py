"""Sub-F per-feature encoder.

This module implements:

1. Coordinate / direction / magnitude helpers (BP2 lock per spec §3.4-§3.6).
2. canonicalize_geometry helper (BP5 §5.6 — 3 pure-redundancy DOFs;
   open-polyline direction PRESERVED per Halt 5 same-day follow-up).
3. Per-feature 4-case encoder (§3.2 A/B/C/D) — added in T8.4.
4. Per-cell aggregator (§3.3) — added in T8.6.

Routing pre-flight (per Assertion 2 in task-8-writer plan):
- canonicalize_geometry dispatches strictly on ``geom.geom_type`` (Shapely
  classification). NEVER use ``geom.is_ring`` for routing — it returns True
  for closed LineStrings (roundabouts) AND closed Polygon rings; routing
  closed LineString to the ring path destroys oneway semantics.
"""

from __future__ import annotations

from typing import Final

from shapely.geometry import MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import LinearRing

# BP2 Halt 2 locked values — read here as module-level constants for fast
# access in encoder hot paths. Source of truth is configs/sub_f/encoding_primitives.yaml.
DEFAULT_DIRECTION_COUNT: Final[int] = 48
DEFAULT_MAGNITUDE_QUANTUM_M: Final[float] = 0.5
DEFAULT_ANCHOR_SCHEME: Final[str] = "hierarchical"
DEFAULT_N_ANCHOR_TOKENS: Final[int] = 4  # hierarchical scheme -> 4 tokens
DEFAULT_CHUNK_THRESHOLD_M: Final[float] = 32.0


# ---- helpers ---------------------------------------------------------------


def quantize_coord_m(coord_m: float, quantum_m: float = DEFAULT_MAGNITUDE_QUANTUM_M) -> int:
    """Quantize a coordinate (meters) to integer quantum count.

    Per BP5 §5.2 lock (Halt 5 ratification 2026-05-28):
    ``int(round(coord_m / quantum))`` with Python ``round()`` banker's
    tie-breaking (PEP 3141 round-half-to-even). Rationale: determinism
    requires one rule pinned; banker's is bias-free for coordinate snapping.
    """
    return round(coord_m / quantum_m)


def direction_bin(angle_deg: float, direction_count: int = DEFAULT_DIRECTION_COUNT) -> int:
    """Map angle (degrees, any sign) to direction bin index ``[0, direction_count)``.

    Per BP5 §5.2 lock: tie-break to LOWER bin index at exact bin boundaries.
    Implementation: floor division rounds toward lower index.
    """
    bin_width = 360.0 / direction_count
    angle_norm = angle_deg % 360.0
    return int(angle_norm // bin_width) % direction_count


def _vertex_count(geom: BaseGeometry) -> int:
    """Total vertex count for any supported geometry type.

    Used by tests (token-count invariance) and by the per-cell aggregator
    (feature_count column). Matches the count computed by
    ``scripts/sub_f/analyze_stage_1_2_joint.py`` (BP3 measurement basis).
    """
    gt = geom.geom_type
    if gt == "LineString":
        return len(geom.coords)
    if gt == "Polygon":
        return len(geom.exterior.coords)
    if gt == "Point":
        return 1
    if gt == "MultiPoint":
        return sum(1 for _ in geom.geoms)
    if gt == "MultiLineString":
        return sum(len(part.coords) for part in geom.geoms)
    if gt == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0


# ---- canonicalize_geometry (BP5 §5.6 contract) ----------------------------


def _signed_area_xy(coords: list[tuple[float, float]]) -> float:
    """Shoelace signed area. Positive = CCW (RFC 7946 exterior); negative = CW."""
    n = len(coords) - 1  # exclude closing duplicate
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        s += (x2 - x1) * (y2 + y1)
    return -s / 2.0  # negate so positive = CCW per RFC 7946 convention


def _canonicalize_ring(ring: LinearRing, *, is_exterior: bool) -> LinearRing:
    """Apply §5.6 rules (i) + (ii) to a single Polygon ring.

    (i) Rotate to start at lex-min vertex (exclude duplicate closing vertex).
    (ii) Enforce RFC 7946 winding: exterior CCW (positive signed area),
         interior holes CW (negative signed area).
    """
    coords = list(ring.coords)
    if len(coords) < 4:  # ring needs >=3 unique + 1 closing
        return ring

    # Strip closing vertex for rotation/winding logic; re-add at the end.
    unique = coords[:-1]
    n = len(unique)

    # (i) Rotate so coords[0] is lex-min (tuple comparison handles ties via
    # subsequent coords automatically).
    lex_min_idx = min(range(n), key=lambda i: unique[i])
    rotated = unique[lex_min_idx:] + unique[:lex_min_idx]

    # (ii) Winding correction.
    closed = [*rotated, rotated[0]]
    area = _signed_area_xy(closed)
    if is_exterior and area < 0:
        # Currently CW; reverse to CCW. Keep rotated[0] fixed so the first
        # vertex remains the lex-min; reverse the remaining vertices.
        rotated = [rotated[0], *reversed(rotated[1:])]
    elif (not is_exterior) and area > 0:
        rotated = [rotated[0], *reversed(rotated[1:])]

    return LinearRing([*rotated, rotated[0]])


def _canonicalize_polygon(poly: Polygon) -> Polygon:
    """§5.6 rules (i) + (ii) applied to exterior + each interior hole."""
    canon_exterior = _canonicalize_ring(poly.exterior, is_exterior=True)
    canon_holes = [_canonicalize_ring(hole, is_exterior=False) for hole in poly.interiors]
    return Polygon(canon_exterior, canon_holes)


def _first_vertex_key(geom: BaseGeometry) -> tuple:
    """Sort key for §5.6 rule (iv) multi-part order. For internally-canonical
    parts the first vertex equals the lex-min vertex by construction.

    Tiebreak: next vertex in (rotated for Polygon / source for LineString) order;
    if still tied, fall through to part vertex count (smaller first).
    """
    gt = geom.geom_type
    if gt == "Polygon":
        coords = list(geom.exterior.coords)
        # Strip closing vertex; use full coord list as tiebreak chain.
        return (tuple(coords[:-1]), len(coords) - 1)
    if gt == "LineString":
        coords = list(geom.coords)
        return (tuple(coords), len(coords))
    # Fallback for Point parts.
    return ((next(iter(geom.coords)),), 1)


def canonicalize_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Apply spec §5.6 canonical form to *geom*.

    Dispatch is STRICTLY on ``geom.geom_type`` per Assertion 2 — closed
    LineStrings (start == end, e.g., roundabouts) route to the
    LineString-preserve path, NOT to the polygon-ring canonicalizer. Using
    ``geom.is_ring`` for routing is the bug this design forbids.

    Idempotent: ``canonicalize(canonicalize(g)) == canonicalize(g)`` for every
    supported geom_type.
    """
    gt = geom.geom_type

    if gt == "LineString":
        # Rule (i'): PRESERVE source direction. Open or closed — both kept
        # as-is. See spec §5.6 "Open-polyline direction preservation"
        # evidence note for the BP1-grep rationale (no oneway, waterway-flow,
        # cycleway:left/right tokens in BP1 vocab; canonicalizing direction
        # would silently destroy OSM semantics).
        return geom

    if gt == "Polygon":
        return _canonicalize_polygon(geom)

    if gt == "MultiLineString":
        # Each part: preserve direction (rule i' per part). Then sort parts.
        parts = list(geom.geoms)
        parts_sorted = sorted(parts, key=_first_vertex_key)
        return MultiLineString(parts_sorted)

    if gt == "MultiPolygon":
        # Each part: canonicalize internally (rules i + ii). Then sort parts.
        parts_canon = [_canonicalize_polygon(p) for p in geom.geoms]
        parts_sorted = sorted(parts_canon, key=_first_vertex_key)
        return MultiPolygon(parts_sorted)

    if gt in ("Point", "MultiPoint"):
        # Trivially canonical (no traversal DOF on points).
        return geom

    raise ValueError(f"canonicalize_geometry: unsupported geom_type {gt!r}")
