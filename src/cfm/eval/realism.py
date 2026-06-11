"""Per-feature geometry-realism KS distance (Phase-2 bake-off Task 2; spec §7).

The bake-off's decision-axis y-value: how far a backbone's generated feature
distribution is from the real holdout's, per feature kind (building areas, road
lengths). Lower = more realistic. Architecture-agnostic -- it scores decoded
OUTPUT, so AR and diffusion are measured on one ruler.

The two-sample Kolmogorov-Smirnov statistic is computed directly (no scipy):
``D = max_x |F_gen(x) - F_ref(x)|`` over the empirical CDFs. This matches the
existing codebase precedent (``holdout/sizing.py`` computes KS floors without
scipy) and avoids a new heavyweight dependency on Leonardo.
"""

from __future__ import annotations

import bisect
from enum import Enum

from shapely.geometry import shape

# the ONE building-ring promotion authority + its closed-ring predicate (imported
# by reference, protocol v2 §9 — same pattern as slice_metrics)
from cfm.eval.geometry import _is_closed_ring, promote_building_rings

_POLYGON_TYPES = ("Polygon", "MultiPolygon")
_LINE_TYPES = ("LineString", "MultiLineString")


class FeatureMetric(Enum):
    BUILDING_AREA = "building_area_m2"
    ROAD_LENGTH = "road_length_m"


def feature_samples(
    geoms: list[dict], *, metric: FeatureMetric, blocks: list[list[int]] | None = None
) -> list[float]:
    """Extract the per-feature scalar for ``metric`` from decoded GeoJSON geoms.

    Building areas come from (Multi)Polygon geoms; road lengths from
    (Multi)LineString geoms. Geoms of the other kind are skipped.

    PROMOTION IS INTERNAL (readiness Task 26 (d), resolving the F4-C1d fork
    structurally): the sealed sub-F decoder emits building closed rings as
    ``LineString`` BY CONTRACT, so a bare type-switch silently reads ZERO
    building areas on real decoded data. When ``blocks`` (the decoded token
    blocks aligned with ``geoms``) are given, building closed rings are
    promoted to Polygon FIRST via the one construction-identity authority
    (``cfm.eval.geometry.promote_building_rings``) — so the helper is safe for
    every future caller, not just callers that remembered to promote upstream.

    WITHOUT ``blocks`` a closed-ring LineString is AMBIGUOUS (unpromoted
    building ring vs closed-road roundabout): under BUILDING_AREA it would be
    silently skipped, under ROAD_LENGTH a building would be silently COUNTED
    as a road. Both are scoring corruption, so the regime is LOUD — pass the
    blocks (every decode path has them) to disambiguate by construction
    identity. Open lines and Polygons never need blocks.
    """
    if blocks is not None:
        geoms = promote_building_rings(blocks, geoms)
    out: list[float] = []
    for g in geoms:
        if blocks is None and g.get("type") == "LineString" and _is_closed_ring(g):
            raise ValueError(
                "feature_samples: closed-ring LineString with no `blocks` to "
                "disambiguate it (building ring vs road roundabout) — pass the "
                "decoded token blocks so promotion can key on construction "
                "identity; refusing a silent mis-sample."
            )
        geom = shape(g)
        if metric is FeatureMetric.BUILDING_AREA and geom.geom_type in _POLYGON_TYPES:
            out.append(float(geom.area))
        elif metric is FeatureMetric.ROAD_LENGTH and geom.geom_type in _LINE_TYPES:
            out.append(float(geom.length))
    return out


def ks_distance(generated: list[float], reference: list[float]) -> float:
    """Two-sample KS statistic in [0, 1]; 0 = identical distributions.

    No overlap to compare (either sample empty) -> 1.0 (maximally far): a backbone
    that emits no features of a kind is maximally unrealistic for that kind.
    """
    if not generated or not reference:
        return 1.0
    g = sorted(generated)
    r = sorted(reference)
    ng, nr = len(g), len(r)
    d = 0.0
    for x in sorted(set(g) | set(r)):
        f_gen = bisect.bisect_right(g, x) / ng
        f_ref = bisect.bisect_right(r, x) / nr
        d = max(d, abs(f_gen - f_ref))
    return d
