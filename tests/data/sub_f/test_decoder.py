"""Sub-F decoder + canonical GeoJSON round-trip gate (Case A scope).

The round-trip thresholds (BP2 Halt 2, RE-LOCKED at the 2026-05-29 Halt-2
revisit, configs/sub_f/encoding_primitives.yaml) are the empirical justification
for the hierarchical-anchor + 360-direction + 0.5m-magnitude lock:

- POSITION: L_inf <= 4.8m. Derivation/enforcement statistic is now p99.9 on real
  data (measured 3.7m at N=360; the original 4.8m was a thin sample-p95 — see the
  §13.1 Halt-2 revisit). The synthetic battery below gates the *max* over its
  fixtures (which sits well under 4.8m at N=360); the real-data p99.9 gate is the
  documented lock basis (reports/sub_f_halt2_*_scoping.*).
- ANGLE: right-angle-corner post-deviation <= 4.0deg p95. RE-DERIVED FRESH at
  N=360 (1deg bins; measured real-data p95 = 3.0deg). The old 7.5deg was the N=48
  bin width — a gate nothing could fail. The angle gate is NOT unit-tested here:
  its failure mode is data-dependent (near-but-not-exact right angles + drift),
  and a uniformly-rotated synthetic rectangle preserves its exact-90deg corners at
  EVERY N (both 7.5 and 1 divide 90), so a synthetic angle test gives false
  assurance. The angle gate binds via the real-data measurement + the
  encoding_primitives.yaml value (asserted in test_encoder) + the close-checklist
  re-measure obligation — the same way the original 7.5deg was established. See
  reports/sub_f_halt2_*_scoping and §13.1.

Relaxing either threshold retroactively invalidates the Halt-2 lock; HALT if any
test forces relaxation.

The round-trip metric is VERTEX-COUNT-AWARE: §3.5 chunking inserts collinear
vertices on segments > 32m, so the decoded coord list is longer than the source.
Each SOURCE vertex is mapped to its decoded counterpart via cumulative
`chunked_segment_pairs` (the inserted collinear vertices are admitted per
spec §3.8). A naive 1:1 zip would crash on any chunked feature.

Cases B/C/D bref vertex position is NOT round-trip-asserted per spec §1.4 scope
lock #1 + §13.1 v2-deferral ledger entry. The NEGATIVE test below asserts the
absence of a position gate so a future contributor cannot silently re-add one.
"""

from __future__ import annotations

import math
import random

from shapely.geometry import LineString, Polygon

_L_INF_THRESHOLD_M = 4.8  # BP2 position lock; do NOT relax without re-running Halt 2.


def _source_vertex_l_inf(
    source_coords: list[tuple[float, float]], decoded_coords: list[tuple[float, float]]
) -> float:
    """Vertex-count-aware round-trip L_inf.

    Maps each SOURCE vertex to its decoded counterpart via cumulative
    `chunked_segment_pairs` (§3.5 chunking inserts collinear decoded vertices on
    long segments; those are admitted per §3.8 and skipped here). For features
    with all segments <= 32m this reduces to a 1:1 comparison.
    """
    from cfm.data.sub_f.token_cost import chunked_segment_pairs

    cum = 0
    max_linf = max(
        abs(source_coords[0][0] - decoded_coords[0][0]),
        abs(source_coords[0][1] - decoded_coords[0][1]),
    )
    for k in range(1, len(source_coords)):
        seg = math.hypot(
            source_coords[k][0] - source_coords[k - 1][0],
            source_coords[k][1] - source_coords[k - 1][1],
        )
        cum += chunked_segment_pairs(seg)
        d = decoded_coords[cum]
        max_linf = max(max_linf, abs(source_coords[k][0] - d[0]), abs(source_coords[k][1] - d[1]))
    return max_linf


def test_canonical_geojson_byte_stable_across_key_order():
    """Per spec §5.3: sort_keys=True, indent=None, ensure_ascii=True."""
    from cfm.data.sub_f.decoder import serialize_geojson

    geom1 = {"type": "Point", "coordinates": [1.0, 2.0]}
    geom2 = {"coordinates": [1.0, 2.0], "type": "Point"}
    assert serialize_geojson(geom1) == serialize_geojson(geom2)


def test_round_trip_open_linestring_case_a_within_threshold():
    """Case A round-trip on an OPEN LineString. Direction is preserved (§5.6)."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    source = LineString([(10.0, 20.0), (15.0, 25.0), (20.0, 30.0)])
    canonical = canonicalize_geometry(source)
    assert list(canonical.coords) == list(source.coords), (
        "open LineString direction must be preserved by canonicalize (§5.6)"
    )
    encoded = encode_feature(canonical, semantic_tag="highway=residential")
    decoded = decode_feature(encoded.tokens)
    decoded_coords = [tuple(p) for p in decoded["coordinates"]]
    l_inf = _source_vertex_l_inf(list(source.coords), decoded_coords)
    assert l_inf <= _L_INF_THRESHOLD_M, (
        f"open LineString L_inf {l_inf:.4f}m exceeds threshold {_L_INF_THRESHOLD_M}m"
    )


def test_round_trip_closed_linestring_roundabout_case_a_within_threshold():
    """Case A round-trip on a CLOSED LineString (roundabout). Direction preserved."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    source = LineString([(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0), (2.0, 0.0)])
    canonical = canonicalize_geometry(source)
    assert list(canonical.coords) == list(source.coords), (
        "closed LineString direction must be preserved by canonicalize"
    )
    encoded = encode_feature(canonical, semantic_tag="highway=residential")
    decoded = decode_feature(encoded.tokens)
    decoded_coords = [tuple(p) for p in decoded["coordinates"]]
    l_inf = _source_vertex_l_inf(list(source.coords), decoded_coords)
    assert l_inf <= _L_INF_THRESHOLD_M, (
        f"closed LineString (roundabout) L_inf {l_inf:.4f}m exceeds {_L_INF_THRESHOLD_M}m"
    )


def test_canonicalize_preserves_polygon_vertex_set_zero_tolerance():
    """Independent invariant: canonicalize preserves the polygon's vertex set exactly."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    source = Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])  # CCW already
    canonical = canonicalize_geometry(source)
    src_unique = {tuple(p) for p in source.exterior.coords[:-1]}
    canon_unique = {tuple(p) for p in canonical.exterior.coords[:-1]}
    assert src_unique == canon_unique, (
        f"canonicalize altered polygon vertex set: source={src_unique} canonical={canon_unique}"
    )


def test_round_trip_polygon_case_a_within_threshold():
    """Case A round-trip on a Polygon (compared against CANONICAL coords)."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    source = Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])
    canonical = canonicalize_geometry(source)
    canonical_coords = [tuple(p) for p in canonical.exterior.coords]
    encoded = encode_feature(canonical, semantic_tag="building=residential")
    decoded = decode_feature(encoded.tokens)
    decoded_coords = [tuple(p) for p in decoded["coordinates"]]
    l_inf = _source_vertex_l_inf(canonical_coords, decoded_coords)
    assert l_inf <= _L_INF_THRESHOLD_M, (
        f"Polygon round-trip L_inf {l_inf:.4f}m exceeds {_L_INF_THRESHOLD_M}m"
    )


def test_round_trip_aggregate_linf_across_synthetic_fixtures():
    """Aggregate stress: max L_inf across 30 synthetic fixtures (incl. long,
    chunked segments) must stay <= 4.8m. The vertex-count-aware metric handles
    the inserted collinear vertices. Deterministic seed for reproducibility.
    """
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    rng = random.Random(20260529)
    cell_extent = 250.0
    max_l_inf = 0.0
    for _ in range(30):
        n_vertices = rng.randint(2, 6)
        coords = [
            (rng.uniform(10, cell_extent - 10), rng.uniform(10, cell_extent - 10))
            for _ in range(n_vertices)
        ]
        canonical = canonicalize_geometry(LineString(coords))
        encoded = encode_feature(canonical, semantic_tag="highway=residential")
        decoded = decode_feature(encoded.tokens)
        decoded_coords = [tuple(p) for p in decoded["coordinates"]]
        max_l_inf = max(max_l_inf, _source_vertex_l_inf(list(canonical.coords), decoded_coords))
    assert max_l_inf <= _L_INF_THRESHOLD_M, (
        f"aggregate Case A max L_inf {max_l_inf:.4f}m across 30 fixtures "
        f"exceeds threshold {_L_INF_THRESHOLD_M}m"
    )


def test_case_b_d_bref_vertex_position_is_NOT_round_trip_asserted():
    """NEGATIVE TEST — protects the v2-scoped deferral.

    Per spec §1.4 scope lock #1 + §13.1 "T8.7 plan-write -> BP7 bref vertex
    position: NO v1 round-trip gate": Cases B/C/D drop the crossing-position
    vertex by design (class-only `<bref>` tokens); its geometric error is
    UNBOUNDED-BY-TEST in v1 (bounded above only by cell_extent/2 = 125m). v1 BP7
    coverage is class-agreement via the §8.1 four-test composite (T8.5 + T10),
    NOT round-trip L_inf.

    This test EXISTS so a future contributor cannot silently re-add a bref-vertex
    round-trip assertion thinking it was forgotten. If you are reading this and
    wondering "shouldn't we round-trip the bref vertex?", the answer is NO — read
    §1.4 + §13.1 first. v2-scoped, with a threshold derived from sub-E rotation
    precision when sub-F-v2 ships exact crossing positions.
    """
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import encode_feature

    source = LineString([(10.0, 100.0), (50.0, 100.0), (250.0, 100.0)])  # exits E edge
    encoded = encode_feature(source, semantic_tag="highway=primary", outbound_bref="<bref_E_MAJOR>")
    assert encoded.case == "B"

    decoded = decode_feature(encoded.tokens)  # must NOT crash on Case B tokens
    assert "coordinates" in decoded
    assert len(decoded["coordinates"]) >= 1, (
        "decoder must emit at least the inner vertices; bref-edge vertex may be "
        "approximate per the v1 deferral"
    )

    # NEGATIVE: explicitly DO NOT compare the final decoded vertex against
    # source[-1] for L_inf. If you are about to add such an assertion, STOP and
    # read spec §1.4 scope lock #1 + §13.1 first. The deferral is v2-scoped.
