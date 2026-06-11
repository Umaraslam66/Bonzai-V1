"""Sub-F per-feature decoder: token sequence -> GeoJSON geometry dict.

Inverse of `encoder.encode_feature` per spec §3.2 four-case grammar. Output
geometry serializes to canonical GeoJSON via `serialize_geojson` (sort_keys,
no indent, ASCII) per spec §5.3 for byte-identity comparisons.

The decoder reconstructs vertices from the encoder's hierarchical anchor +
(direction, magnitude) pair stream and returns Point or LineString ONLY,
never Polygon: building closed-rings come back as a closed LineString
(coords[0] == coords[-1]) by contract, and consumers apply
`cfm.eval.geometry.promote_building_rings` for construction-identity
promotion to Polygon. Multi-part
geometries are split into separate features at encode_cell (T8.6); each
encode_feature call corresponds to one decode_feature call.

CASES B/C/D BREF VERTEX (per spec §1.4 scope lock #1 + §13.1 ledger):
Cases B/C/D drop the crossing-position vertex by design — the `<bref>`
token carries only direction + class, not position. The decoder emits a
last-known-vertex (the inner anchor + dir/mag extrapolation reaching the
edge) for Case B/D outbound and a first-vertex-on-edge stub for Case C/D
inbound. The bref edge vertex's geometric error is UNBOUNDED-BY-TEST in
sub-F-v1 (bounded ABOVE by cell_extent/2 = 125m); class agreement is the
v1 BP7 gate per §8.1, not L_inf. v2-scoped per §1.4. Every bref vertex
emission below carries an explicit # v2-scoped comment so a future
contributor cannot silently re-add a position assertion.
"""

from __future__ import annotations

import json
import math
from typing import Any

from cfm.data.sub_f.encoder import (
    _FEATURE_END_TOKEN_ID,
    _FEATURE_TOKEN_ID,
    _HIERARCHICAL_ANCHOR_BASE,
    DEFAULT_DIRECTION_COUNT,
    DEFAULT_MAGNITUDE_QUANTUM_M,
)

_BLOCK = 23  # hierarchical anchor block size; must match encoder
# Halt-2 revisit 2026-05-29: direction relocated 396->511 (widened 48->360, 1° bins);
# magnitude unchanged at 444. Must match encoder._DIRECTION_BASE / _MAGNITUDE_BASE.
_DIRECTION_BASE = 511
_MAGNITUDE_BASE = 444


def _decode_anchor(tokens: list[int], offset: int) -> tuple[tuple[float, float], int]:
    """Read 4 hierarchical anchor tokens starting at `offset`; return ((x,y), new_offset)."""
    x_hi_t, x_lo_t, y_hi_t, y_lo_t = tokens[offset : offset + 4]
    x_hi = x_hi_t - (_HIERARCHICAL_ANCHOR_BASE + 0 * _BLOCK)
    x_lo = x_lo_t - (_HIERARCHICAL_ANCHOR_BASE + 1 * _BLOCK)
    y_hi = y_hi_t - (_HIERARCHICAL_ANCHOR_BASE + 2 * _BLOCK)
    y_lo = y_lo_t - (_HIERARCHICAL_ANCHOR_BASE + 3 * _BLOCK)
    x_q = x_hi * _BLOCK + x_lo
    y_q = y_hi * _BLOCK + y_lo
    return (
        (x_q * DEFAULT_MAGNITUDE_QUANTUM_M, y_q * DEFAULT_MAGNITUDE_QUANTUM_M),
        offset + 4,
    )


def _decode_dir_mag(d_token: int, m_token: int) -> tuple[float, float]:
    """Inverse of _direction_magnitude_pair: tokens -> (dx, dy) in meters."""
    direction = d_token - _DIRECTION_BASE
    magnitude_q = (m_token - _MAGNITUDE_BASE) + 1
    bin_width_deg = 360.0 / DEFAULT_DIRECTION_COUNT
    angle_rad = math.radians(direction * bin_width_deg)
    distance_m = magnitude_q * DEFAULT_MAGNITUDE_QUANTUM_M
    return distance_m * math.cos(angle_rad), distance_m * math.sin(angle_rad)


def _is_bref_token(token_id: int) -> bool:
    """BP7 boundary-reference token IDs are 1500..1507."""
    return 1500 <= token_id <= 1507


def decode_feature(tokens: list[int]) -> dict[str, Any]:
    """Decode a per-feature token sequence to a GeoJSON-shape dict.

    Returns one of (Point or LineString ONLY — this function never returns
    Polygon):
      - {"type": "LineString", "coordinates": [[x, y], ...]}  (open or closed)
      - {"type": "Point", "coordinates": [x, y]}              (single vertex)

    Closure: building closed-rings come back as a closed LineString
    (coords[0] == coords[-1]) by contract — the token stream cannot
    distinguish a Polygon ring from a closed-LineString roundabout.
    Consumers that need Polygon geometry must apply
    `cfm.eval.geometry.promote_building_rings` (construction-identity
    promotion). Features carrying bref tokens (Case B/C/D) are always
    LineString (roads).

    For Cases B/C/D: the inbound bref token signals "first vertex is on
    the entry edge" (position class-only per §1.4 deferral; decoder uses
    the anchor as the entry vertex). The outbound bref token signals
    "last vertex is on the exit edge" (decoder appends a last-known-vertex
    derived from the prior inner vertex). Bref vertex positions are NOT
    asserted to round-trip — see module docstring.
    """
    if tokens[0] != _FEATURE_TOKEN_ID or tokens[-1] != _FEATURE_END_TOKEN_ID:
        raise ValueError("decode_feature: missing <feature>/<feature_end> markers")

    body = tokens[1:-1]
    # body[0] is semantic_tag id (skip — geometry decode doesn't need it).
    offset = 1

    # Optional inbound bref (Cases C/D): present immediately after semantic tag.
    has_inbound = offset < len(body) and _is_bref_token(body[offset])
    if has_inbound:
        offset += 1  # inbound bref consumed; anchor that follows IS the entry vertex

    # Anchor (4 hierarchical tokens) — always present.
    (x, y), offset = _decode_anchor(body, offset)
    coords: list[tuple[float, float]] = [(x, y)]

    # Walk (dir, mag) pairs until <feature_end> or outbound bref.
    while offset < len(body):
        if _is_bref_token(body[offset]):
            break  # outbound bref; no more inner pairs
        dx, dy = _decode_dir_mag(body[offset], body[offset + 1])
        nx, ny = coords[-1][0] + dx, coords[-1][1] + dy
        coords.append((nx, ny))
        offset += 2

    # Outbound bref present (Cases B/D)? Emit a last-known-vertex.
    # v2-scoped per spec §1.4 scope lock #1 + §13.1: bref vertex position is
    # NOT carried by the token; the decoder emits the previous interior
    # vertex unchanged as a stand-in. The bref edge vertex's geometric
    # error is unbounded-by-test in v1, bounded above by cell_extent/2.
    # DO NOT add an L_inf assertion against the source's edge vertex here
    # without first updating §1.4 + §13.1 + a fresh threshold lock.
    has_outbound = offset < len(body) and _is_bref_token(body[offset])
    if has_outbound:
        # v2-scoped: emit previous vertex as a stand-in for the edge crossing.
        # See module docstring + spec §1.4 + §13.1 ledger entry.
        if coords:
            coords.append(coords[-1])  # bref vertex placeholder
        offset += 1

    # Determine output GeoJSON shape from coord properties.
    is_closed = len(coords) >= 4 and coords[0] == coords[-1]
    has_brefs = has_inbound or has_outbound

    if len(coords) == 1:
        # Point feature.
        return {"type": "Point", "coordinates": [coords[0][0], coords[0][1]]}

    if is_closed and not has_brefs:
        # Closed sequence with no brefs: ambiguous between Polygon and
        # closed-LineString (roundabout). The encoder cannot distinguish
        # them in the token stream. Caller (with knowledge of original
        # geom_type) reconstructs the right Shapely shape; we return
        # LineString shape by default and tests promote to Polygon as
        # needed via `Polygon(decoded_coords)`. This matches the empirical
        # round-trip tests' usage pattern.
        return {
            "type": "LineString",
            "coordinates": [list(p) for p in coords],
        }

    # Open LineString (Case A open polyline) OR Case B/C/D road (always LineString).
    return {
        "type": "LineString",
        "coordinates": [list(p) for p in coords],
    }


def serialize_geojson(geom: dict) -> str:
    """Canonical GeoJSON serialization per spec §5.3.

    Use this for byte-identity comparisons across encode/decode runs.
    """
    return json.dumps(geom, sort_keys=True, indent=None, ensure_ascii=True)
