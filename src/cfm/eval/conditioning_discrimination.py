"""Task-9 conditioning-discrimination HALT-gate (gate input (i)).

Computed on CPU from REAL held-out tiles BEFORE any GPU pilot (model-independent).

The question: within the EXACT macro-plan stratum the model is conditioned on
(``(zoning, road_skeleton, cell_density_bucket, coastal_inland_river)``), do the
held-out cities' real feature distributions differ? If they do, the worst-case
"one city is the bar" assumption (T5) is contaminated by same-conditioning
cross-city variation and must reopen; if not, the bar is valid.

VERDICT semantics:
- ``PASS``        — no stratum/metric shows a BH-significant cross-city difference.
- ``FAIL``        — at least one BH-significant cross-city pair (T5 reopens).
- ``UNSUPPORTED`` — thin-n exclusion left zero qualifying comparisons; the held-out
  set cannot support the test at full granularity. REPORT, do NOT coarsen.

Multiple-comparison guard (REQUIRED): with dozens of per-pair KS tests, ~5% fire
by chance at alpha=0.05. Benjamini-Hochberg adjusts ALL per-pair p-values jointly,
so a single noise-tail outlier cannot reopen T5. The raw worst-KS never fires alone.

The pure stats + verdict are unit-tested locally; the IO extraction
(``extract_features_by_city_stratum_metric``) reads the real corpus on Leonardo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from shapely.geometry import shape

from cfm.data.sub_g.readers import read_sub_f_cells
from cfm.eval.geometry import promote_building_rings
from cfm.eval.holdout.labels import read_tile_labels
from cfm.eval.holdout.paths import (
    epsg_label_for_region,
    holdout_manifest_for_region,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
)
from cfm.eval.holdout.pipeline import _cell_density_by_cell
from cfm.eval.holdout.roundtrip import decode_region_blocks
from cfm.eval.realism import _LINE_TYPES, _POLYGON_TYPES, FeatureMetric, ks_distance

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: The 4 held-out EU cities (the multiregion manifest's held_out_cities).
DEFAULT_CITIES: tuple[str, ...] = ("eisenhuttenstadt", "glasgow", "krakow", "munich")


# --------------------------------------------------------------------------- #
# Pure stats helpers
# --------------------------------------------------------------------------- #


def noise_floor(n1: int, n2: int, *, c: float = 1.36) -> float:
    """alpha=0.05 two-sample KS critical value: ``c * sqrt((n1+n2)/(n1*n2))``.

    The per-comparison threshold PAIRED to the exact n that produced the KS
    distance: a smaller sample tolerates a larger KS before it counts as signal.
    """
    return c * math.sqrt((n1 + n2) / (n1 * n2))


def ks_pvalue(d: float, n1: int, n2: int) -> float:
    """Asymptotic two-sample KS p-value with the finite-sample lambda correction.

    ``ne = n1*n2/(n1+n2)``;
    ``lam = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * d``;
    ``Q(lam) = 2 * sum_{k=1..100} (-1)^(k-1) * exp(-2*k^2*lam^2)``, clamped to [0, 1].
    ``d == 0`` (or ``lam == 0``) -> 1.0 (no evidence of difference).
    """
    if d <= 0.0:
        return 1.0
    ne = n1 * n2 / (n1 + n2)
    sqrt_ne = math.sqrt(ne)
    lam = (sqrt_ne + 0.12 + 0.11 / sqrt_ne) * d
    if lam <= 0.0:
        return 1.0
    q = 0.0
    for k in range(1, 101):
        q += (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    q *= 2.0
    return min(1.0, max(0.0, q))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, returned IN THE INPUT ORDER.

    Sort ascending, ``adj_(i) = min(1, p_(i) * m / rank)``, enforce monotonicity
    from the largest rank down, then map back to original positions.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj_sorted: list[float] = [0.0] * m
    prev = 1.0
    # Walk from largest p (rank m) down to smallest (rank 1), enforcing monotonicity.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        raw = pvals[idx] * m / rank
        prev = min(prev, raw)
        adj_sorted[rank - 1] = min(1.0, prev)
    out: list[float] = [0.0] * m
    for rank in range(1, m + 1):
        out[order[rank - 1]] = adj_sorted[rank - 1]
    return out


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairResult:
    """One unordered city-pair KS comparison within a (metric, stratum)."""

    metric: str
    stratum: tuple
    city_a: str
    city_b: str
    n_a: int
    n_b: int
    ks: float
    floor: float
    p_raw: float
    p_bh: float  # filled after the GLOBAL BH correction across all pairs
    significant: bool


@dataclass(frozen=True)
class ConditioningDiscriminationResult:
    verdict: str  # "PASS" | "FAIL" | "UNSUPPORTED"
    per_metric_verdict: dict[str, str]
    pairs: list[PairResult]
    n_by_city_stratum_metric: dict[tuple[str, tuple, str], int]
    n_excluded_thin: int
    n_strata_too_few_cities: int
    n_qualifying_comparisons: int
    min_n: int
    alpha: float


# --------------------------------------------------------------------------- #
# Verdict (PURE)
# --------------------------------------------------------------------------- #


def conditioning_discrimination_verdict(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    min_n: int = 50,
    alpha: float = 0.05,
) -> ConditioningDiscriminationResult:
    """Decide PASS / FAIL / UNSUPPORTED from per-(city, stratum, metric) features.

    A (city, stratum, metric) cell QUALIFIES iff it has >= ``min_n`` samples.
    Within each (metric, stratum) with >= 2 qualified cities, every unordered
    city-pair becomes a comparison; the per-pair p-values are BH-corrected JOINTLY
    across both metrics and all strata. FAIL iff any pair is BH-significant.
    """
    # 1. Record every n (thin or not).
    n_by_key: dict[tuple[str, tuple, str], int] = {key: len(vals) for key, vals in features.items()}

    # 2. Qualified cells (>= min_n) and the thin-exclusion count.
    qualified: dict[tuple[str, tuple, str], list[float]] = {
        key: vals for key, vals in features.items() if len(vals) >= min_n
    }
    n_excluded_thin = len(features) - len(qualified)

    # 3. Group qualified cells by (metric, stratum) -> {city: samples}.
    by_metric_stratum: dict[tuple[str, tuple], dict[str, list[float]]] = {}
    for (city, stratum, metric), vals in qualified.items():
        by_metric_stratum.setdefault((metric, stratum), {})[city] = vals

    pairs: list[PairResult] = []
    n_strata_too_few_cities = 0
    for (metric, stratum), city_samples in by_metric_stratum.items():
        cities = sorted(city_samples)
        if len(cities) < 2:
            n_strata_too_few_cities += 1
            continue
        for ia in range(len(cities)):
            for ib in range(ia + 1, len(cities)):
                ca, cb = cities[ia], cities[ib]
                a, b = city_samples[ca], city_samples[cb]
                d = ks_distance(a, b)
                pairs.append(
                    PairResult(
                        metric=metric,
                        stratum=stratum,
                        city_a=ca,
                        city_b=cb,
                        n_a=len(a),
                        n_b=len(b),
                        ks=d,
                        floor=noise_floor(len(a), len(b)),
                        p_raw=ks_pvalue(d, len(a), len(b)),
                        p_bh=1.0,  # placeholder; filled by the global BH below
                        significant=False,
                    )
                )

    n_qualifying_comparisons = len(pairs)

    # 5. UNSUPPORTED: zero qualifying comparisons -> report, do NOT coarsen.
    if n_qualifying_comparisons == 0:
        return ConditioningDiscriminationResult(
            verdict="UNSUPPORTED",
            per_metric_verdict={},
            pairs=[],
            n_by_city_stratum_metric=n_by_key,
            n_excluded_thin=n_excluded_thin,
            n_strata_too_few_cities=n_strata_too_few_cities,
            n_qualifying_comparisons=0,
            min_n=min_n,
            alpha=alpha,
        )

    # 6. Global BH across ALL pairs' raw p-values.
    p_bh = benjamini_hochberg([p.p_raw for p in pairs])
    corrected: list[PairResult] = [
        PairResult(
            metric=p.metric,
            stratum=p.stratum,
            city_a=p.city_a,
            city_b=p.city_b,
            n_a=p.n_a,
            n_b=p.n_b,
            ks=p.ks,
            floor=p.floor,
            p_raw=p.p_raw,
            p_bh=adj,
            significant=adj < alpha,
        )
        for p, adj in zip(pairs, p_bh, strict=True)
    ]

    # 7. Per-metric verdict (FAIL if any pair of that metric is significant).
    metrics = sorted({p.metric for p in corrected})
    per_metric_verdict: dict[str, str] = {
        m: ("FAIL" if any(p.significant for p in corrected if p.metric == m) else "PASS")
        for m in metrics
    }
    verdict = "FAIL" if any(v == "FAIL" for v in per_metric_verdict.values()) else "PASS"

    return ConditioningDiscriminationResult(
        verdict=verdict,
        per_metric_verdict=per_metric_verdict,
        pairs=corrected,
        n_by_city_stratum_metric=n_by_key,
        n_excluded_thin=n_excluded_thin,
        n_strata_too_few_cities=n_strata_too_few_cities,
        n_qualifying_comparisons=n_qualifying_comparisons,
        min_n=min_n,
        alpha=alpha,
    )


# --------------------------------------------------------------------------- #
# IO extraction (Leonardo CPU; reuses sealed pieces)
# --------------------------------------------------------------------------- #


def _tile_features(
    blocks: list[list[int]],
    geoms: list[dict],
    density_strata: list[int],
) -> list[tuple[str, float, int]]:
    """Promote building closed-rings to Polygon (construction identity), THEN classify
    each feature as ``(metric, value, density)``.

    The sealed decoder returns building closed rings as ``LineString`` BY CONTRACT;
    without ``promote_building_rings`` every building is miscounted as a road
    (``building_area`` empty, ``road_length`` contaminated with building perimeters) —
    the same n_polygons=0 construction-identity trap caught in Phase 1. Roads (incl.
    closed roundabouts) are NOT promoted and stay road_length.
    """
    promoted = promote_building_rings(blocks, geoms)
    out: list[tuple[str, float, int]] = []
    for geom, density in zip(promoted, density_strata, strict=True):
        g = shape(geom)
        if g.geom_type in _POLYGON_TYPES:
            out.append((FeatureMetric.BUILDING_AREA.value, float(g.area), int(density)))
        elif g.geom_type in _LINE_TYPES:
            out.append((FeatureMetric.ROAD_LENGTH.value, float(g.length), int(density)))
    return out


def extract_features_by_city_stratum_metric(
    release: str,
    cities: Sequence[str],
) -> dict[tuple[str, tuple, str], list[float]]:
    """Accumulate per-(city, full-stratum, metric) real feature scalars.

    For each held-out city, for each held-out tile: read the tile-level conditioning
    labels (zoning / skeleton / coastal), decode the round-tripped-real geoms tagged
    with per-cell density, and bin each feature into
    ``(city, (zoning, skeleton, density, coastal), metric)``.

    Runs on Leonardo against the real corpus; there is no corpus locally.
    """
    features: dict[tuple[str, tuple, str], list[float]] = {}

    for city in cities:
        manifest = yaml.safe_load(holdout_manifest_for_region(release, city).read_text())
        tiles = manifest["regions"][city]["tiles"]
        epsg = epsg_label_for_region(city)
        sub_d = sub_d_region_dir(release, city)
        sub_f = sub_f_region_dir(release, city)

        for tile in tiles:
            ti, tj = int(tile["tile_i"]), int(tile["tile_j"])
            # CRITICAL (step-0 fix): construct the per-tile dir with the REGION's CRS
            # label, never the defaulted Singapore EPSG3414 dir (which reads nothing).
            dirname = tile_dirname(ti, tj, epsg)
            cells_path = sub_f / dirname / "cells.parquet"
            if not cells_path.exists():
                logger.warning(
                    "missing sub-F cells for %s tile (%d,%d): %s", city, ti, tj, cells_path
                )
                continue

            labels = read_tile_labels(sub_d / dirname, tile_i=ti, tile_j=tj)
            zoning = labels.morphology_stratum.dominant_zoning_class
            skeleton = labels.morphology_stratum.modal_road_skeleton_class
            coastal = labels.coastal_inland_river

            cdbc = _cell_density_by_cell(sub_d / dirname)
            tokens = read_sub_f_cells(cells_path)
            blocks, geoms, density_strata = decode_region_blocks(tokens, cdbc)
            for metric, value, density in _tile_features(blocks, geoms, density_strata):
                stratum = (zoning, skeleton, density, coastal)
                features.setdefault((city, stratum, metric), []).append(value)

    return features
