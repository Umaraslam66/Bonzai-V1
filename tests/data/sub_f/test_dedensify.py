"""Sub-F #19 quantum-inflation fix: de-densify sub-quantum vertices before encoding.

ROOT CAUSE (encoder.py `_direction_magnitude_pair`): every segment's distance is
quantized with ``total_q = max(1, quantize_coord_m(distance_m))``. A 0.04 m segment
quantizes to 0 → ``max(1, 0) = 1`` → emits magnitude 1 = 0.5 m. An over-densified
ring of N sub-quantum segments therefore decodes ~``0.5 / seg_len`` x longer (a
0.0735 m segmentation → 6.8x, rotterdam's degraded ~0.04 m → ~13x). The inflated
decoded ring's vertices walk far enough from the anchor to trip sub-G's
``decoded_vertex_within_cell_bound`` (max |coord| > 300 m = 250 m cell + 50 m margin).

THE FIX (teeth-proof below, red-before/green-after):
  (a) inflation-kill — an over-densified building decodes back to ~1.0x source
      path-length and stays within the 300 m cell bound;
  (b) over-simplify GUARD — a legitimately-detailed building (every segment already
      >= quantum) is preserved vertex-for-vertex (fixing inflation by destroying real
      geometry is the INVERSE failure), with a discrimination teeth test proving the
      preservation assertion is not vacuous (a large tolerance DOES drop vertices);
  (c) degenerate-extent edge — a sub-quantum-EXTENT ring (< 0.5 m across) falls back
      to its original coords rather than collapsing to an invalid ring.

These synthetic fixtures exercise the EXACT failing code path deterministically. Real
over-densified data confirms the fix end-to-end at the eindhoven re-derive gate
(Phase 2); the rotterdam/warsaw degraded-source recovery verdict is a real-data check
on Leonardo (Phase 3). See docs/known_issues.md #19.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from cfm.data.sub_f.encoder import DEFAULT_MAGNITUDE_QUANTUM_M

_BOUND_M = 300.0  # mirrors sub_g.seam_decodability._VERTEX_BOUND_M (250 cell + 50 margin)


def _path_length(coords: list[tuple[float, float]]) -> float:
    """Frame-invariant total polyline length (sum of segment lengths)."""
    return sum(
        math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
        for i in range(1, len(coords))
    )


def _overdensified_square(side_m: float, origin: tuple[float, float], seg_len_m: float) -> Polygon:
    """A square building whose edges are subdivided into `seg_len_m` micro-segments.

    With seg_len_m < quantum, every micro-segment hits the encoder's max(1, ...) floor,
    so the decoded ring inflates ~quantum/seg_len_m x.
    """
    ox, oy = origin
    corners = [(ox, oy), (ox + side_m, oy), (ox + side_m, oy + side_m), (ox, oy + side_m)]
    ring: list[tuple[float, float]] = []
    for k in range(4):
        x0, y0 = corners[k]
        x1, y1 = corners[(k + 1) % 4]
        n = max(1, round(math.hypot(x1 - x0, y1 - y0) / seg_len_m))
        for s in range(n):  # exclude the endpoint; next edge contributes it
            t = s / n
            ring.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    ring.append(corners[0])  # close
    return Polygon(ring)


def _decoded_ring(poly: Polygon) -> list[tuple[float, float]]:
    """Canonicalize → encode → decode → return the decoded coord ring."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    canon = canonicalize_geometry(poly)
    encoded = encode_feature(canon, semantic_tag="building=residential")
    decoded = decode_feature(encoded.tokens)
    return [(x, y) for x, y in decoded["coordinates"]]


# --- (a) inflation-kill: red-before / green-after -------------------------------------


def test_dedensify_kills_inflation_on_overdensified_building():
    """An over-densified building decodes back to ~1.0x source path-length.

    RED before the fix: the ~0.0735 m segmentation inflates ~6.8x (the ratio the PI
    named). GREEN after: de-densify drops the sub-quantum vertices, ratio ~1.0.
    """
    target_ratio = 6.8
    seg_len = DEFAULT_MAGNITUDE_QUANTUM_M / target_ratio  # ~0.0735 m
    poly = _overdensified_square(40.0, (100.0, 100.0), seg_len)
    source = list(poly.exterior.coords)

    decoded = _decoded_ring(poly)
    ratio = _path_length(decoded) / _path_length(source)
    assert ratio <= 1.1, f"decoded path-length inflated {ratio:.2f}x (expected ~1.0x)"


def test_dedensify_keeps_decoded_within_cell_bound():
    """Mirrors sub_g's decoded_vertex_within_cell_bound check directly.

    RED before: the inflated 40 m building anchored cell-locally decodes past 300 m.
    GREEN after: decoded max |coord| stays within the bound.
    """
    seg_len = DEFAULT_MAGNITUDE_QUANTUM_M / 6.8
    poly = _overdensified_square(40.0, (100.0, 100.0), seg_len)
    decoded = _decoded_ring(poly)
    max_abs = max(max(abs(x), abs(y)) for x, y in decoded)
    assert max_abs <= _BOUND_M, f"decoded vertex {max_abs:.1f} m exceeds {_BOUND_M} m bound"


# --- (b) over-simplify guard + discrimination teeth -----------------------------------


def _legit_detailed_ring() -> list[tuple[float, float]]:
    """An L-shaped building, every segment >= quantum (2-10 m edges, all representable)."""
    return [
        (10.0, 10.0),
        (22.0, 10.0),
        (22.0, 13.0),
        (16.0, 13.0),
        (16.0, 24.0),
        (10.0, 24.0),
        (10.0, 10.0),
    ]


def test_dedensify_preserves_legitimately_detailed_building():
    """Over-simplify GUARD: a building with all segments >= quantum is untouched.

    Every kept-vertex test asserts de-densify drops NOTHING here and preserves
    path-length exactly — fixing inflation must not destroy representable geometry.
    """
    from cfm.data.sub_f.encoder import dedensify_coords

    coords = _legit_detailed_ring()
    out = dedensify_coords(coords)
    assert out == coords, "de-densify must not drop any >= quantum-spaced vertex"
    assert math.isclose(_path_length(out), _path_length(coords), rel_tol=1e-12)


def test_dedensify_teeth_large_tolerance_does_oversimplify():
    """Discrimination teeth: the SAME detailed building IS simplified at a large
    tolerance. Proves the preservation assertion above is not vacuous — the function
    is genuinely capable of dropping vertices, it just does not at the quantum.
    """
    from cfm.data.sub_f.encoder import dedensify_coords

    coords = _legit_detailed_ring()
    out = dedensify_coords(coords, min_seg_m=5.0)
    assert len(out) < len(coords), "large tolerance must drop vertices (teeth)"


# --- (c) degenerate sub-quantum-extent edge -------------------------------------------


def test_dedensify_degenerate_subquantum_extent_ring_falls_back():
    """A ring smaller than the quantum across must not collapse to an invalid ring.

    de-densify would merge every interior vertex (all within 0.5 m of the anchor),
    leaving < 4 coords; the guard falls back to the original coords (bounded,
    sub-resolution — documented v1 edge, known_issue #19 addendum).
    """
    from cfm.data.sub_f.encoder import dedensify_coords, encode_feature

    tiny = _overdensified_square(0.2, (50.0, 50.0), 0.01)
    coords = list(tiny.exterior.coords)
    out = dedensify_coords(coords)
    assert out == coords, "degenerate ring must fall back to original (no collapse)"
    # And it must still encode without raising.
    encode_feature(tiny, semantic_tag="building=residential")
