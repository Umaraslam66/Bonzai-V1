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
from enum import Enum
from typing import Any

from shapely.geometry import shape

# sub-G's construction-identity predicate, imported BY REFERENCE (protocol v2 §9).
from cfm.data.sub_g.seam_decodability import _is_bref_placeholder_collapse as _is_bref_collapse
from cfm.eval.emergence import buildings_emerged  # one source for the emergence floor
from cfm.eval.geometry import promote_building_rings  # one source for building-ring promotion
from cfm.eval.holdout.bref_rate import bref_placeholder_rate


class EmergenceVerdict(Enum):
    """§2 paired structural check: did buildings emerge densely enough to score?

    ROADS_ONLY means the building-geometry metrics are FLOORED (a failure to produce
    buildings), never a vacuous pass like ``ogc_valid_rate=1.0`` over zero polygons.

    INCOMMENSURATE (F15) means the §2 verdict was REFUSED before it ever ran: the
    generation's length cap is below the floor's derivation regime (e.g. a 512-cap
    generation scored against a full-cell locked-budget-regime floor), so neither SCOREABLE
    nor ROADS_ONLY would mean anything. Loud and distinct from the legacy None
    (= "no floor inputs given, verdict never requested").
    """

    SCOREABLE = "scoreable"
    ROADS_ONLY = "roads_only"
    INCOMMENSURATE = "INCOMMENSURATE"


def emergence_verdict(*, n_polygons: int, n_cells: int, floor_per_cell: float) -> EmergenceVerdict:
    """SCOREABLE iff the run clears the holdout-density-tied floor (one source: emergence)."""
    if buildings_emerged(n_polygons=n_polygons, n_cells=n_cells, floor_per_cell=floor_per_cell):
        return EmergenceVerdict.SCOREABLE
    return EmergenceVerdict.ROADS_ONLY


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
    n_cells: int | None = None,
    emergence_floor_per_cell: float | None = None,
    emergence_floor_provenance: dict[str, Any] | None = None,
    generated_length_cap: int | None = None,
    floor_regime_cell_length: int | None = None,
) -> dict[str, Any]:
    """Per-cell metrics over the DECODED (block, geom) pairs.

    ``blocks``/``geoms``/``strata`` are aligned: ``blocks[i]`` decoded to ``geoms[i]``
    in stratum ``strata[i]`` (the decoded subset; undecodable blocks are dropped
    upstream). ``n_attempted_blocks`` is the count BEFORE dropping failures so
    decodability is decoded/attempted; it defaults to ``len(blocks)`` (all decoded).

    When both ``n_cells`` and ``emergence_floor_per_cell`` are given, the §2 emergence
    guard runs: a run below the holdout-density floor gets ``emergence_verdict ==
    ROADS_ONLY`` and ``building_metrics_floored == True`` so the curve never reads its
    ``ogc_valid_rate``/``right_angle_rate`` over ~zero polygons as a good score.

    ``emergence_floor_provenance`` (Task 13, F13/F15) is the floor's provenance record
    (region, holdout_density, frac, derived_at, derivation_regime incl. the denominator
    convention) and is carried through VERBATIM into the metrics dict so every report
    states where its floor came from. The key is present-but-None when not given
    (report-stable shape).

    Commensurability (Task 14, F15): ``generated_length_cap`` is the per-cell token
    budget the generation actually had; ``floor_regime_cell_length`` is the cell-token
    regime the floor was DERIVED under. When the verdict would run and both lengths
    are given but ``generated_length_cap < floor_regime_cell_length``, the §2 verdict
    is REFUSED (``INCOMMENSURATE``): a capped generation never got the budget to emit
    what the floor counts, so neither SCOREABLE nor ROADS_ONLY is meaningful.
    ``building_metrics_floored`` stays False — the metrics are not floored, they are
    REFUSED (flooring asserts "compared against the floor and failed"; refusal asserts
    "the comparison itself is invalid"). Legacy callers passing no lengths are
    unchanged (None-tolerant).
    """
    n_decoded = len(geoms)
    attempted = n_attempted_blocks if n_attempted_blocks is not None else len(blocks)

    # Promote building closed-ring LineStrings to polygons (Task 1.5) BEFORE any
    # building-geometry metric: the sealed decoder returns building rings as LineString by
    # contract, so without this n_polygons / right-angle / OGC-on-polygons read a vacuous 0.
    geoms = promote_building_rings(blocks, geoms)

    # OGC validity over the NON-bref-collapse decoded geoms (structural exclusion):
    # bref-collapse is a known v1 limitation, removed from the denominator and
    # reported separately. A non-bref invalid geom IS counted as invalid.
    # POINT SEMANTICS PINNED (Task 26 (f)): a decoded (Multi)Point is trivially
    # OGC-valid, so leaving it in the denominator inflates ogc_valid_rate over
    # geometry that has no validity question to answer. Points are EXCLUDED
    # from the denominator and reported as ``n_points`` — visible, never
    # silently scored.
    gated_valid = 0
    gated_total = 0
    n_points = 0
    for block, geom in zip(blocks, geoms, strict=True):
        if _is_bref_collapse(block, geom):
            continue  # known v1 outbound-bref placeholder collapse -> not gated
        if geom.get("type") in ("Point", "MultiPoint"):
            n_points += 1
            continue  # trivially valid; excluded from the denominator, counted
        gated_total += 1
        if _is_ogc_valid(geom):
            gated_valid += 1

    bref = _bref_rate_fn(blocks, geoms, strata)  # shared instrument; reported-not-gated
    right_angle_rate, n_polygons, n_corners = _right_angle_stats(geoms)

    verdict: EmergenceVerdict | None = None
    floored = False
    if n_cells is not None and emergence_floor_per_cell is not None:
        if (
            generated_length_cap is not None
            and floor_regime_cell_length is not None
            and generated_length_cap < floor_regime_cell_length
        ):
            # F15 refuse rule: the pure ROADS_ONLY/SCOREABLE function is NOT consulted;
            # floored stays False (refused, not floored — see the docstring).
            verdict = EmergenceVerdict.INCOMMENSURATE
        else:
            verdict = emergence_verdict(
                n_polygons=n_polygons, n_cells=n_cells, floor_per_cell=emergence_floor_per_cell
            )
            floored = verdict is EmergenceVerdict.ROADS_ONLY

    # the floor's denominator convention, quoted from provenance when present
    denominator = None
    if emergence_floor_provenance is not None:
        denominator = (emergence_floor_provenance.get("derivation_regime") or {}).get("denominator")

    return {
        "decodability_rate": n_decoded / attempted if attempted else 0.0,
        "ogc_valid_rate": gated_valid / gated_total if gated_total else 0.0,
        "right_angle_rate": right_angle_rate,
        "bref_collapse_rate": bref.overall_rate,  # REPORTED, never gates pass/fail
        "n_decoded": n_decoded,
        "n_attempted": attempted,
        "n_polygons": n_polygons,  # disambiguates right_angle_rate==0.0
        "n_corners": n_corners,
        "n_points": n_points,  # (f) pin: Points excluded from the OGC denominator
        "emergence_verdict": verdict.value if verdict is not None else None,
        "building_metrics_floored": floored,
        "emergence_floor_provenance": emergence_floor_provenance,
        # F15 commensurability record (present-but-None when absent, report-stable):
        "generated_length_cap": generated_length_cap,
        "floor_regime_cell_length": floor_regime_cell_length,
        "floor_denominator_convention": denominator,
        "scope": "per-cell; tile-coherence UNSCORED",
    }
