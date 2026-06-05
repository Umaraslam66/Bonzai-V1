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

from cfm.data.sub_f.boundary_contract import resolve_bref_tag
from cfm.data.sub_f.vocab import ROAD_L1_KEY, semantic_tag_to_l1_key, vocab_tag_to_id

# BP2 Halt 2 locked values — read here as module-level constants for fast
# access in encoder hot paths. Source of truth is configs/sub_f/encoding_primitives.yaml.
DEFAULT_DIRECTION_COUNT: Final[int] = 360  # Halt-2 revisit 2026-05-29 (was 48); 1° bins
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
# Halt-2 revisit 2026-05-29: direction widened 48->360 (1° bins) for round-trip
# fidelity on long/wiggly real features. 360 contiguous slots do not fit at the
# old 396 base (magnitude 444-508 + structural 509-510 sit immediately after), so
# the direction sub-block RELOCATES append-safely into reserved_v2_headroom at
# 511-870; magnitude STAYS at 444 (unchanged); old 396-443 retired to reserved.
# See sentinel_inventory.yaml direction + direction_v1_deprecated blocks.
_DIRECTION_BASE = 511
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


_MAX_MAGNITUDE_Q: Final[int] = 64  # 64 * 0.5m = 32m per spec §3.5 chunk threshold


def _direction_magnitude_pair(dx: float, dy: float) -> list[int]:
    """(direction, magnitude) tokens for a segment, CHUNKED per spec §3.5.

    Per spec §3.5: "Magnitudes beyond 32m broken into multiple direction +
    magnitude pairs at the same direction (e.g., a 50m straight stretch
    becomes `<direction_0> <magnitude_64>` followed by `<direction_0>
    <magnitude_36>`)." Encoder MUST emit chunked pairs to avoid silent
    geometry loss on long segments.

    Direction sub-block: ids 511..870 (360 slots, 1° bins; relocated from 396..443
    at the Halt-2 revisit 2026-05-29 — see _DIRECTION_BASE).
    Magnitude sub-block: ids 444..508 (65 slots; 0.5m * (1..64), max 32m per pair).

    Returns a list of `2 * ceil(distance_m / 32)` tokens for `distance_m > 0`,
    grouped as (dir, mag) pairs all carrying the same direction. The final
    pair carries the remainder. Zero-length segments emit a single
    (dir, mag=1) pair as a minimum to preserve vertex count (matches the
    pre-chunking magnitude_q = max(1, ...) floor).
    """
    angle_deg = math.degrees(math.atan2(dy, dx))
    direction = direction_bin(angle_deg)
    distance_m = math.hypot(dx, dy)

    # Quantize the total distance; the first ceil(total/64) - 1 chunks emit
    # max magnitude (64 = 32m); the final chunk carries the remainder.
    total_q = max(1, quantize_coord_m(distance_m))
    out: list[int] = []
    remaining = total_q
    direction_token = _DIRECTION_BASE + direction
    while remaining > 0:
        chunk = min(_MAX_MAGNITUDE_Q, remaining)
        out.append(direction_token)
        out.append(_MAGNITUDE_BASE + (chunk - 1))
        remaining -= chunk
    return out


def _vertex_pairs_dir_mag(coords: list[tuple[float, float]]) -> list[int]:
    """Emit (dir, mag) token pairs for V vertices, with chunking per §3.5.

    Pre-§3.5-chunking: emitted exactly `2 * (V - 1)` tokens.
    Post-§3.5-chunking: emits `2 * sum_segments(ceil(distance_m_i / 32))` tokens,
    which is >= `2 * (V - 1)` and equal when all segments are ≤ 32m.
    """
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

    BP7 emission: inbound_bref / outbound_bref values originate at T8.5's
    sub-E reader (`src/cfm/data/sub_f/boundary_contract.py`) against the
    source-derived contract (T8.5, commit efa6786, sub-E sources cited in
    `_SUB_E_CONTRACT`). Residual debt: empirical T3c stage-4 ratio +
    first real-data end-to-end flow, per
    `reports/2026-05-23-phase-1-sub-F-close-checklist.md`.
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
        tokens.append(tag_to_id[inbound_bref])  # BP7: source-derived per T8.5

    anchor_x, anchor_y = coords[0]
    tokens.extend(_hierarchical_anchor_tokens(anchor_x, anchor_y))

    # Inner pairs: Case A/C emit (V-1) pairs (reach vertices 2..V);
    # Case B/D emit (V-2) pairs (final vertex replaced by outbound bref).
    inner_pairs_to = n_vertices if case in ("A", "C") else n_vertices - 1
    tokens.extend(_vertex_pairs_dir_mag(coords[:inner_pairs_to]))

    if case in ("B", "D"):
        assert outbound_bref is not None  # narrows for type-checker
        tokens.append(tag_to_id[outbound_bref])  # BP7: source-derived per T8.5

    tokens.append(_FEATURE_END_TOKEN_ID)
    return EncodedFeature(case=case, semantic_tag=semantic_tag, tokens=tokens)


# ---- per-cell aggregator (§3.3, §4.4) -------------------------------------


@dataclass(frozen=True)
class EncodedCell:
    """Per-cell encoded sequence."""

    tokens: list[int]
    feature_count: int  # number of features encoded (matches cells.parquet col)


# Float-exact on-boundary tolerance for endpoint->edge classification. sub-C's
# clip snaps a crossing endpoint ONTO the cell edge, so "on edge" is float-exact,
# NOT a metric band. SHARED AUTHORITY: sub-G's seam-2 bijection
# (seam_contract_tokens._EDGE_TOL_M) imports THIS constant so the two never drift
# to different on-edge definitions (sub-G T11 H2, 2026-06-01; the old sub-G 0.5m
# misattributed near-corner endpoints -> 1,649 false bref-bijection mismatches).
ON_EDGE_EPS_M = 1e-6


def endpoint_edge_direction(
    x: float,
    y: float,
    cell_origin: tuple[float, float] = (0.0, 0.0),
    cell_extent_m: float = 250.0,
    edge_eps_m: float = ON_EDGE_EPS_M,
) -> str | None:
    """Return the cell-edge direction {N,E,S,W} an endpoint lies on, or None.

    SINGLE AUTHORITY for endpoint->edge classification. The encoder's bref
    emission gate and the sub-F cross-tile validator's road-edge-presence signal
    BOTH route through this one function so they can never drift on the N/S
    convention (the sub-G T11 flip lesson; see
    reports/2026-05-31-sub-G-T11-symmetry-root-cause.md).

    Direction names follow the BP7 AUTHORITY (sub_e.rotation.cell_to_edge_ids,
    which the locked configs/sub_f/boundary_reference_vocab.yaml defers to): a
    cell's NORTH edge is the one shared with (i, j-1) = the LOW-y edge
    (cell-local y=0); SOUTH = the high-y edge (cell-local y=extent). This is
    geographically inverted (recorded v2 convention debt) but MUST match
    cell_to_edge_ids so the contract class looked up by direction is the edge
    the endpoint physically lies on. Pinned by
    tests/data/sub_f/test_direction_authority.py — do NOT "fix" to geographic
    (y=extent->N) without re-deriving sub-F and updating that authority gate.
    """
    ox, oy = cell_origin
    x_rel = x - ox
    y_rel = y - oy
    if abs(x_rel) <= edge_eps_m:
        return "W"
    if abs(x_rel - cell_extent_m) <= edge_eps_m:
        return "E"
    if abs(y_rel) <= edge_eps_m:
        return "N"
    if abs(y_rel - cell_extent_m) <= edge_eps_m:
        return "S"
    return None


def _classify_feature_for_bref(
    geom: BaseGeometry,
    cell_edges: dict[str, str],
    cell_origin: tuple[float, float] = (0.0, 0.0),
    cell_extent_m: float = 250.0,
    edge_eps_m: float = ON_EDGE_EPS_M,
) -> tuple[str | None, str | None]:
    """Determine inbound / outbound boundary-ref tags for one feature.

    Compares geometry endpoints against cell edges; if an endpoint lies on
    an active boundary edge (MAJOR or MINOR per cell_edges), emit the
    matching <bref_DIR_CLASS> tag.

    LineStrings only emit brefs - Polygons (buildings) and Points (POIs) do
    not cross cell boundaries in token-layer per spec section 1.4 (non-road
    cross-cell features clipped at geometry layer).

    BP7 emission: cell_edges supplied here originates from the T8.5
    source-derived sub-E boundary-contract reader; resolve_bref_tag emits
    only for MAJOR_ROAD / MINOR_ROAD classes per spec section 3.7 BP7 lock.
    """
    gt = geom.geom_type
    if gt not in ("LineString", "MultiLineString"):
        return None, None

    if gt == "MultiLineString":
        # Multi-part is split by encode_cell before reaching here.
        return None, None

    coords = list(geom.coords)
    if len(coords) < 2:
        return None, None

    in_dir = endpoint_edge_direction(*coords[0], cell_origin, cell_extent_m, edge_eps_m)
    out_dir = endpoint_edge_direction(*coords[-1], cell_origin, cell_extent_m, edge_eps_m)

    in_class = cell_edges.get(in_dir) if in_dir else None
    out_class = cell_edges.get(out_dir) if out_dir else None

    inbound_bref = resolve_bref_tag(in_dir, in_class) if in_dir and in_class else None
    outbound_bref = resolve_bref_tag(out_dir, out_class) if out_dir and out_class else None
    return inbound_bref, outbound_bref


def encode_cell(
    features: list[tuple[BaseGeometry, str]],
    cell_edges: dict[str, str],
    cell_origin: tuple[float, float] = (0.0, 0.0),
) -> EncodedCell:
    """Encode one cell to a flat token sequence.

    Args:
      features: list of (geom, semantic_tag) tuples in sub-C row order
                (caller does NOT re-sort).
      cell_edges: per-cell boundary-class map from
                  ``boundary_contract.load_boundary_contract``. Pass empty
                  dict for empty / no-edge cells.
      cell_origin: cell SW corner in projected meters (default (0,0) for
                   cell-local coords).

    Per spec sections 3.3 + 4.4:
      - Empty cells emit tokens = [] (not null).
      - Per-feature output is concatenated with no <cell_start>/<cell_end>
        sentinel on-disk (cell boundary is the parquet row structure).
      - Each feature is canonicalized internally before encoding (BP5
        contract).
    """
    if not features:
        return EncodedCell(tokens=[], feature_count=0)

    tokens: list[int] = []
    feature_count = 0
    for geom, semantic_tag in features:
        canon = canonicalize_geometry(geom)

        # §1.4: only road (highway-keyed) features emit boundary-ref tokens.
        # Resolve the L1 key via the shared vocab authority so <unknown_highway>
        # still counts as road, while <natural=*> and other non-road LineStrings
        # do not (sub-G T11 cycle-4: a natural LineString clipped to a road edge
        # emitted a spurious <bref>). Same authority the validator's non-road leg
        # uses, so the two never re-determine road-ness with a local parse.
        is_road = semantic_tag_to_l1_key(semantic_tag) == ROAD_L1_KEY

        # Multi-part: encode each part separately per spec section 3.2 implicit
        # multi-part handling (one EncodedFeature per part).
        gt = canon.geom_type
        if gt in ("MultiLineString", "MultiPolygon"):
            for part in canon.geoms:
                inbound, outbound = (
                    _classify_feature_for_bref(part, cell_edges, cell_origin)
                    if is_road
                    else (None, None)
                )
                ef = encode_feature(
                    part,
                    semantic_tag=semantic_tag,
                    inbound_bref=inbound,
                    outbound_bref=outbound,
                )
                tokens.extend(ef.tokens)
                feature_count += 1
        elif gt == "MultiPoint":
            # MultiPoint: encode each Point as a separate Case A feature.
            for part in canon.geoms:
                ef = encode_feature(part, semantic_tag=semantic_tag)
                tokens.extend(ef.tokens)
                feature_count += 1
        else:
            inbound, outbound = (
                _classify_feature_for_bref(canon, cell_edges, cell_origin)
                if is_road
                else (None, None)
            )
            ef = encode_feature(
                canon,
                semantic_tag=semantic_tag,
                inbound_bref=inbound,
                outbound_bref=outbound,
            )
            tokens.extend(ef.tokens)
            feature_count += 1

    return EncodedCell(tokens=tokens, feature_count=feature_count)
