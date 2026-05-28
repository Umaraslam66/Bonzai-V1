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

import math
from dataclasses import dataclass
from typing import Final, Literal

from shapely.geometry import MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import LinearRing

from cfm.data.sub_f.vocab import vocab_tag_to_id

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


# ---- 4-case grammar (spec section 3.2) ------------------------------------

Case = Literal["A", "B", "C", "D"]

# Structural sentinels (family="structural", consumed from BP2
# reserved_v2_headroom front per sentinel-inventory fix 2026-05-28).
# See pre-flight Assertion 4 + spec section 13.1 "T8 plan-write -> BP2
# inventory" row.
_FEATURE_TOKEN_ID = 509
_FEATURE_END_TOKEN_ID = 510

# BP2 anchor sub-block is 300-395 (full 96 slots usable as anchor coords).
# Hierarchical anchor scheme uses 4 tokens per anchor; with BLOCK=23 below,
# 23*4 = 92 slots are addressable (positions 300-391), leaving 4 spare
# (392-395) per anchor sub-block reserve.
_HIERARCHICAL_ANCHOR_BASE = 300

# Sub-block bases for direction/magnitude per sentinel_inventory.yaml.
_DIRECTION_BASE = 396
_MAGNITUDE_BASE = 444

# Hierarchical anchor block width: 23 slots per sub-axis (hi/lo for x/y).
# Cell extent 250m at 0.5m quantum -> 500 quantum-cells per axis;
# 23 chosen so 23*22 >= 500 with one spare slot per sub-axis.
_HIERARCHICAL_ANCHOR_BLOCK: Final[int] = 23

# Sub-C sentinel patterns that map to BP4 <unknown_*> family per spec
# section 3.3 + cascade #7.
_SUB_C_UNKNOWN_PATTERNS = ("__UNK__", "unknown")
_SUB_C_BUILDING_SENTINEL_PREFIX = "B_"  # B__UNK__ is the primary sentinel


@dataclass(frozen=True)
class EncodedFeature:
    """Per-feature encoded token sequence."""

    case: Case
    semantic_tag: str
    tokens: list[int]


def _resolve_semantic_tag_to_token_id(semantic_tag: str) -> int:
    """Map a raw OSM tag (key=value) to its token id.

    Per spec section 3.3 + cascade #7: sub-C unknown sentinels (``B__UNK__``,
    ``highway=unknown``, etc.) are NOT first-class BP1 slots - they map to
    the BP4 ``<unknown_KEY>`` family for the parent key.
    """
    tag_to_id = vocab_tag_to_id()

    if semantic_tag in tag_to_id:
        return tag_to_id[semantic_tag]

    # Fall through to BP4. Extract the key from "key=value" form.
    if "=" not in semantic_tag:
        raise ValueError(f"semantic_tag must be 'key=value' form; got {semantic_tag!r}")
    key, value = semantic_tag.split("=", 1)

    is_sub_c_sentinel = (
        any(pat in value for pat in _SUB_C_UNKNOWN_PATTERNS)
        or value.startswith(_SUB_C_BUILDING_SENTINEL_PREFIX)
        or value == ""
    )
    unknown_tag = f"<unknown_{key}>"
    if is_sub_c_sentinel and unknown_tag in tag_to_id:
        return tag_to_id[unknown_tag]

    # Non-sentinel value missing from BP1: also bucket via BP4 per cascade #7.
    if unknown_tag in tag_to_id:
        return tag_to_id[unknown_tag]

    raise KeyError(f"no BP1 or BP4 slot for semantic_tag {semantic_tag!r}")


def _hierarchical_anchor_tokens(x_m: float, y_m: float) -> list[int]:
    """Per spec section 3.6 hierarchical anchor scheme: 4 tokens encoding
    (x_hi, x_lo, y_hi, y_lo).

    Cell extent 250m, magnitude_quantum 0.5m -> 500 quantum-cells per axis.
    Hierarchical split: hi block = floor(coord_q / 23), lo block = coord_q % 23,
    where 23 chosen so 23*22 >= 500. Hi range [0, 22] = 23 slots; lo range
    [0, 22] = 23 slots; total per axis = 46; both axes = 92 (fits in the
    96-slot BP2 anchor sub-region 300..395 with 4 reserved spares).

    Layout within the BP2 anchor sub-block (300..395):
      300..322  x_hi  (23 slots)
      323..345  x_lo  (23 slots)
      346..368  y_hi  (23 slots)
      369..391  y_lo  (23 slots)
      392..395  spare (4 slots, reserved within anchor sub-block)
    """
    block = _HIERARCHICAL_ANCHOR_BLOCK
    x_q = quantize_coord_m(x_m)
    y_q = quantize_coord_m(y_m)
    x_hi, x_lo = divmod(x_q, block)
    y_hi, y_lo = divmod(y_q, block)
    return [
        _HIERARCHICAL_ANCHOR_BASE + 0 * block + x_hi,
        _HIERARCHICAL_ANCHOR_BASE + 1 * block + x_lo,
        _HIERARCHICAL_ANCHOR_BASE + 2 * block + y_hi,
        _HIERARCHICAL_ANCHOR_BASE + 3 * block + y_lo,
    ]


def _direction_magnitude_pair(dx: float, dy: float) -> list[int]:
    """One (direction, magnitude) pair token list for a segment.

    Direction sub-block: ids 396..443 (48 slots).
    Magnitude sub-block: ids 444..508 (65 slots; 0.5m * (1..64) plus a single
    overflow marker at 444+64 = 508).
    """
    angle_deg = math.degrees(math.atan2(dy, dx))
    direction = direction_bin(angle_deg)
    distance_m = math.hypot(dx, dy)
    magnitude_q = max(1, min(64, quantize_coord_m(distance_m)))
    return [_DIRECTION_BASE + direction, _MAGNITUDE_BASE + (magnitude_q - 1)]


def _vertex_pairs_dir_mag(coords: list[tuple[float, float]]) -> list[int]:
    """For V vertices, emit 2*(V-1) tokens - one (dir, mag) pair per segment."""
    out: list[int] = []
    for i in range(1, len(coords)):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        out.extend(_direction_magnitude_pair(x2 - x1, y2 - y1))
    return out


def _extract_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Get the encoder's input coord list for a feature.

    For Polygons we encode the exterior ring; multi-geometries are encoded
    per-part by the caller (encode_cell in T8.6 splits them).
    """
    gt = geom.geom_type
    if gt == "LineString":
        return list(geom.coords)
    if gt == "Polygon":
        return list(geom.exterior.coords)
    if gt == "Point":
        x, y = geom.x, geom.y
        return [(x, y)]
    raise ValueError(
        f"_extract_coords: encode multi-part geometries per part, not as a whole ({gt})"
    )


def encode_feature(
    geom: BaseGeometry,
    *,
    semantic_tag: str,
    inbound_bref: str | None = None,
    outbound_bref: str | None = None,
) -> EncodedFeature:
    """Encode one feature per spec section 3.2 four-case grammar.

    Caller responsibilities:
      - Pass the canonical geometry (use canonicalize_geometry first).
      - For multi-part geometries, split into parts and call once per part.
      - Set inbound_bref / outbound_bref from sub-E boundary contract per
        cell x edge (T8.5 provides the resolver).

    Routing:
      - inbound_bref=None, outbound_bref=None -> Case A
      - inbound_bref=None, outbound_bref=set  -> Case B
      - inbound_bref=set,  outbound_bref=None -> Case C
      - inbound_bref=set,  outbound_bref=set  -> Case D

    BP7 emission - UNVERIFIED against real sub-E parquet; see close-checklist +
    project_sub_e_cache_absent_t3c_code_inferred memory. inbound_bref /
    outbound_bref values originate at T8.5's sub-E reader against the
    documented schema.
    """
    tag_to_id = vocab_tag_to_id()
    coords = _extract_coords(geom)
    n_vertices = len(coords)
    if n_vertices < 1:
        raise ValueError("encode_feature: empty coord list")

    semantic_id = _resolve_semantic_tag_to_token_id(semantic_tag)

    case: Case
    if inbound_bref is None and outbound_bref is None:
        case = "A"
    elif inbound_bref is None and outbound_bref is not None:
        case = "B"
    elif inbound_bref is not None and outbound_bref is None:
        case = "C"
    else:
        case = "D"

    tokens: list[int] = [_FEATURE_TOKEN_ID, semantic_id]

    if case in ("C", "D"):
        # Inbound bref prepended; per spec section 3.2, the anchor IS the
        # entry vertex which IS coords[0] (canonical convention).
        assert inbound_bref is not None  # narrows for type-checker
        tokens.append(tag_to_id[inbound_bref])  # BP7 emission - UNVERIFIED

    anchor_x, anchor_y = coords[0]
    tokens.extend(_hierarchical_anchor_tokens(anchor_x, anchor_y))

    # Inner pairs: Case A/C emit (V-1) pairs (reach vertices 2..V);
    # Case B/D emit (V-2) pairs (final vertex replaced by outbound bref).
    inner_pairs_to = n_vertices if case in ("A", "C") else n_vertices - 1
    tokens.extend(_vertex_pairs_dir_mag(coords[:inner_pairs_to]))

    if case in ("B", "D"):
        assert outbound_bref is not None  # narrows for type-checker
        tokens.append(tag_to_id[outbound_bref])  # BP7 emission - UNVERIFIED

    tokens.append(_FEATURE_END_TOKEN_ID)
    return EncodedFeature(case=case, semantic_tag=semantic_tag, tokens=tokens)
