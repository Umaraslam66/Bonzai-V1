"""Per-cell slice eval (spec §8) -- the one real per-cell number for the closed loop.

REPORTED-NOT-GATED: there is no pass/fail threshold here. The slice answers
"does the micro generator emit decodable, locally-valid cells?" -- NOT "does the
tile generator work." Tile cell-to-cell coherence, boundary-contract stitching and
macro-planner conditioning are explicitly UNSCORED-in-slice (named in ``scope``).

Three construction-identity contracts (one source, never reimplemented):
  * the bref-placeholder collapse RATE is the shared D3 instrument
    ``bref_placeholder_rate`` (identity-locked ``_bref_rate_fn``);
  * per-item bref classification reuses sub-G's construction-identity predicate
    ``_is_bref_placeholder_collapse`` (protocol v2 §9: import by reference);
  * decodability + OGC-validity mirror sub-G ``check_decodability`` (decode failure
    => undecodable; bref-collapse => excluded from the validity denominator, reported).
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import shape

# sub-G's construction-identity predicate, imported BY REFERENCE (protocol v2 §9).
from cfm.data.sub_g.seam_decodability import _is_bref_placeholder_collapse as _is_bref_collapse
from cfm.eval.holdout.bref_rate import bref_placeholder_rate

#: identity-locked shared D3 instrument (the test asserts `is` identity).
_bref_rate_fn = bref_placeholder_rate

#: A polygon corner counts as a right angle if within this many degrees of 90.
#: DECISION (slice v1, tier-2 / reported-not-gated): a 10-degree band tolerates
#: sub-F coordinate quantisation noise while still distinguishing right-angled
#: footprints from arbitrary ones. Revisit when the metric becomes a gate.
_RIGHT_ANGLE_TOL_DEG = 10.0


def _is_ogc_valid(geom: dict[str, Any]) -> bool:
    try:
        return bool(shape(geom).is_valid)
    except Exception:
        return False


def _polygon_rings(geoms: list[dict[str, Any]]) -> list[list[list[float]]]:
    return [g["coordinates"][0] for g in geoms if g.get("type") == "Polygon"]


def _corner_angles_deg(ring: list[list[float]]) -> list[float]:
    """Interior turn angles at each distinct corner of a (closed) polygon ring."""
    pts = ring[:-1] if len(ring) >= 2 and ring[0] == ring[-1] else ring
    n = len(pts)
    angles: list[float] = []
    for k in range(n):
        a, b, c = pts[k - 1], pts[k], pts[(k + 1) % n]
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        m1 = math.hypot(*v1)
        m2 = math.hypot(*v2)
        if m1 == 0 or m2 == 0:
            continue
        cos_v = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)))
        angles.append(math.degrees(math.acos(cos_v)))
    return angles


def _right_angle_stats(geoms: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Return (right_angle_rate, n_polygons, n_corners). The rate is the fraction of
    polygon corners within tolerance of 90 degrees, or 0.0 when there are no polygon
    corners — so a 0.0 with n_polygons==0 means 'no polygons emitted' (not 'polygons
    with no right angles'). Reporting the counts makes that distinction explicit."""
    rings = _polygon_rings(geoms)
    total = 0
    right = 0
    for ring in rings:
        for ang in _corner_angles_deg(ring):
            total += 1
            if abs(ang - 90.0) <= _RIGHT_ANGLE_TOL_DEG:
                right += 1
    rate = right / total if total else 0.0
    return rate, len(rings), total


def slice_eval(
    blocks: list[list[int]],
    geoms: list[dict[str, Any]],
    strata: list[int],
    *,
    n_attempted_blocks: int | None = None,
) -> dict[str, Any]:
    """Per-cell metrics over the DECODED (block, geom) pairs.

    ``blocks``/``geoms``/``strata`` are aligned: ``blocks[i]`` decoded to ``geoms[i]``
    in stratum ``strata[i]`` (the decoded subset; undecodable blocks are dropped
    upstream). ``n_attempted_blocks`` is the count BEFORE dropping failures so
    decodability is decoded/attempted; it defaults to ``len(blocks)`` (all decoded).
    """
    n_decoded = len(geoms)
    attempted = n_attempted_blocks if n_attempted_blocks is not None else len(blocks)

    # OGC validity over the NON-bref-collapse decoded geoms (structural exclusion):
    # bref-collapse is a known v1 limitation, removed from the denominator and
    # reported separately. A non-bref invalid geom IS counted as invalid.
    gated_valid = 0
    gated_total = 0
    for block, geom in zip(blocks, geoms, strict=True):
        if _is_bref_collapse(block, geom):
            continue  # known v1 outbound-bref placeholder collapse -> not gated
        gated_total += 1
        if _is_ogc_valid(geom):
            gated_valid += 1

    bref = _bref_rate_fn(blocks, geoms, strata)  # shared instrument; reported-not-gated
    right_angle_rate, n_polygons, n_corners = _right_angle_stats(geoms)

    return {
        "decodability_rate": n_decoded / attempted if attempted else 0.0,
        "ogc_valid_rate": gated_valid / gated_total if gated_total else 0.0,
        "right_angle_rate": right_angle_rate,
        "bref_collapse_rate": bref.overall_rate,  # REPORTED, never gates pass/fail
        "n_decoded": n_decoded,
        "n_attempted": attempted,
        "n_polygons": n_polygons,  # disambiguates right_angle_rate==0.0
        "n_corners": n_corners,
        "scope": "per-cell; tile-coherence UNSCORED",
    }
