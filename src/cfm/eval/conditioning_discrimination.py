"""Task-9 conditioning-discrimination HALT-gate (gate input (i)).

Computed on CPU from REAL held-out tiles BEFORE any GPU pilot (model-independent).

The question: within the EXACT macro-plan stratum the model is conditioned on
(``(zoning, road_skeleton, cell_density_bucket, coastal_inland_river)``), do the
held-out cities' real feature distributions differ? If they do, the worst-case
"one city is the bar" assumption (T5) is contaminated by same-conditioning
cross-city variation and must reopen; if not, the bar is valid.

VERDICT semantics:
- ``PASS``        — no stratum/metric shows a BH-significant cross-city difference
  at or above the effect-size floor.
- ``FAIL``        — at least one cross-city pair is BH-significant AND its KS clears
  the effect-size floor (T5 reopens).
- ``UNSUPPORTED`` — thin-n exclusion left zero qualifying comparisons; the held-out
  set cannot support the test at full granularity. REPORT, do NOT coarsen.

Multiple-comparison guard (REQUIRED): with dozens of per-pair KS tests, ~5% fire
by chance at alpha=0.05. Benjamini-Hochberg adjusts ALL per-pair p-values jointly,
so a single noise-tail outlier cannot reopen T5. The raw worst-KS never fires alone.

Effect-size floor (PI-call #1, spec §4.3): at real per-cell sample sizes (thousands
of features) a MICROSCOPIC distribution shift is BH-significant — statistical
significance alone makes PASS structurally unreachable. A pair only counts as
``significant`` when it is BH-significant AND ``ks >= effect_size_floor`` (δ=0.15
default at the runner layer). Both counts are reported (``n_significant_raw_bh``
vs ``n_significant_effect``) so the δ's effect stays visible.

The pure stats + verdict are unit-tested locally; the IO extraction
(``extract_features_by_city_stratum_metric``) reads the real corpus on Leonardo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml
from shapely.geometry import shape

from cfm.data.sub_g.readers import read_sub_f_cells

# has_outbound_bref (public since W7; formerly a sanctioned private import) is the
# construction-identity AUTHORITY for the v1 outbound-bref artifact (token-structure
# fact, never an error-magnitude or zero-length test) — one source, not a copy.
from cfm.data.sub_g.seam_decodability import has_outbound_bref

# ONE source for the two-sample KS alpha=0.05 coefficient (Task 26 (g)): the
# EXACT 1.358 (~1.3581) from feature_resolution wins over this module's old
# rounded 1.36 literal; imported by reference so the modules cannot drift —
# the cross-guard test pins them equal. noise_floor is INFORMATIONAL here
# (significance is BH-based), so the only effect is reported floors shifting
# ~0.15%; the frozen floor artifact's 1.36-era values stay valid (nothing
# consumes that field on the verified-read path).
from cfm.eval.feature_resolution import KS_C_ALPHA_05
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


def noise_floor(n1: int, n2: int, *, c: float = KS_C_ALPHA_05) -> float:
    """alpha=0.05 two-sample KS critical value: ``c * sqrt((n1+n2)/(n1*n2))``.

    The per-comparison threshold PAIRED to the exact n that produced the KS
    distance: a smaller sample tolerates a larger KS before it counts as signal.
    ``c`` defaults to the ONE-SOURCED exact coefficient 1.358 (see the import
    note above); INFORMATIONAL — significance decisions are BH-p-value-based.
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
    significant: bool  # p_bh < alpha AND ks >= effect_size_floor (see module docstring)


@dataclass(frozen=True)
class TileCoverage:
    """Per-city held-out tile accounting for gate-(i) extraction (readiness F3).

    ``n_tiles_expected`` is the manifest's tile count; ``n_tiles_skipped`` counts
    tiles whose ``cells.parquet`` was absent (the former silent-shrinkage path).
    ``n_bref_excluded`` counts line-typed features (not only roads — e.g. an
    unpromoted open-ring building truncated by a bref) excluded from
    ``road_length_m`` by the outbound-bref construction identity — excluded,
    never silently dropped.
    """

    n_tiles_expected: int
    n_tiles_read: int
    n_tiles_skipped: int
    # DECISION: the feature-level bref-exclusion counter lives on the per-city
    # tile-accounting struct (the existing per-city accounting home the runner
    # already serializes) rather than a new parallel dict. Revisit if more
    # feature-level counters accumulate here.
    n_bref_excluded: int = 0


@dataclass(frozen=True)
class ExtractionResult:
    """Return of ``extract_features_by_city_stratum_metric``: features + coverage."""

    features: dict[tuple[str, tuple, str], list[float]]
    tile_coverage: dict[str, TileCoverage]


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
    effect_size_floor: float
    n_significant_raw_bh: int  # pairs with p_bh < alpha (pre-recalibration rule)
    n_significant_effect: int  # pairs with p_bh < alpha AND ks >= effect_size_floor
    # Per-city extraction coverage, threaded in by the RUNNER (dataclasses.replace);
    # the pure verdict fn stays coverage-agnostic, so the default keeps it green.
    tile_coverage: dict[str, TileCoverage] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Verdict (PURE)
# --------------------------------------------------------------------------- #


def conditioning_discrimination_verdict(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    min_n: int = 50,
    alpha: float = 0.05,
    effect_size_floor: float,
) -> ConditioningDiscriminationResult:
    """Decide PASS / FAIL / UNSUPPORTED from per-(city, stratum, metric) features.

    A (city, stratum, metric) cell QUALIFIES iff it has >= ``min_n`` samples.
    Within each (metric, stratum) with >= 2 qualified cities, every unordered
    city-pair becomes a comparison; the per-pair p-values are BH-corrected JOINTLY
    across both metrics and all strata. FAIL iff any pair is BH-significant AND its
    KS clears ``effect_size_floor`` (the δ recalibration, PI-call #1).

    ``effect_size_floor`` is deliberately a REQUIRED keyword: the δ=0.15 default
    lives at the runner layer, visible where the decision is made
    (diagnostic-threshold discipline), never buried here.
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
            effect_size_floor=effect_size_floor,
            n_significant_raw_bh=0,
            n_significant_effect=0,
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
            significant=(adj < alpha) and (p.ks >= effect_size_floor),
        )
        for p, adj in zip(pairs, p_bh, strict=True)
    ]
    n_significant_raw_bh = sum(1 for p in corrected if p.p_bh < alpha)
    n_significant_effect = sum(1 for p in corrected if p.significant)

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
        effect_size_floor=effect_size_floor,
        n_significant_raw_bh=n_significant_raw_bh,
        n_significant_effect=n_significant_effect,
    )


# --------------------------------------------------------------------------- #
# IO extraction (Leonardo CPU; reuses sealed pieces)
# --------------------------------------------------------------------------- #


def _tile_features(
    blocks: list[list[int]],
    geoms: list[dict],
    density_strata: list[int],
) -> tuple[list[tuple[str, float, int]], int]:
    """Promote building closed-rings to Polygon (construction identity), THEN classify
    each feature as ``(metric, value, density)``; returns ``(features, n_bref_excluded)``.

    The sealed decoder returns building closed rings as ``LineString`` BY CONTRACT;
    without ``promote_building_rings`` every building is miscounted as a road
    (``building_area`` empty, ``road_length`` contaminated with building perimeters) —
    the same n_polygons=0 construction-identity trap caught in Phase 1. Roads (incl.
    closed roundabouts) are NOT promoted and stay road_length.

    Outbound-bref roads are EXCLUDED from ``road_length_m`` BY CONSTRUCTION IDENTITY
    (``has_outbound_bref`` on the ORIGINAL token block, never a zero-length symptom):
    the v1 encoder replaces the last real vertex with a bref token and the decoder
    appends a placeholder, corrupting the decoded length. Excluded features are
    COUNTED, never silently dropped. Promoted building polygons are never bref-excluded.
    """
    promoted = promote_building_rings(blocks, geoms)
    out: list[tuple[str, float, int]] = []
    n_bref_excluded = 0
    for block, geom, density in zip(blocks, promoted, density_strata, strict=True):
        g = shape(geom)
        if g.geom_type in _POLYGON_TYPES:
            out.append((FeatureMetric.BUILDING_AREA.value, float(g.area), int(density)))
        elif g.geom_type in _LINE_TYPES:
            if has_outbound_bref(block):
                n_bref_excluded += 1
                continue
            out.append((FeatureMetric.ROAD_LENGTH.value, float(g.length), int(density)))
    return out, n_bref_excluded


#: Silent-shrinkage ceiling (readiness F3): the max tolerated fraction of a city's
#: manifest tiles missing ``cells.parquet`` before extraction HALTs (strict >).
#: "manifest", not "held-out": this extraction is SHARED — the conditioning-floor
#: runner walks TRAINING cities through the same loop (W6 wording fix).
_SHRINKAGE_CEILING: float = 0.1


def extract_features_by_city_stratum_metric(
    release: str,
    cities: Sequence[str],
    *,
    tiles_by_city: dict[str, list[dict]] | None = None,
) -> ExtractionResult:
    """Accumulate per-(city, full-stratum, metric) real feature scalars + coverage.

    For each city, for each of its tiles: read the tile-level conditioning
    labels (zoning / skeleton / coastal), decode the round-tripped-real geoms tagged
    with per-cell density, and bin each feature into
    ``(city, (zoning, skeleton, density, coastal), metric)``.

    Tile inventory (stage-2 integration fix, Slurm job 45835276): a city present
    in ``tiles_by_city`` uses its provided tile dicts (each carrying
    ``tile_i``/``tile_j``, like the manifest ``tiles[]`` entries) — the runner
    hands TRAINING cities in this way (sub-D validated inventory; a train city
    has no tile-level holdout, so ``holdout_manifest_for_region`` rightly raises
    for it — that fail-closed boundary is kept). A city ABSENT from the map keeps
    the EXACT holdout-manifest path. Default ``None`` keeps behavior BYTE-IDENTICAL
    to the pre-override extractor — the held-out bit-identity guarantee (family-1
    determinism anchor) rides on this default. Everything per-tile (labels,
    density, decode, bref exclusion, coverage counters, F3 halts) is shared
    regardless of inventory source.

    Per-city tile coverage is counted (``n_tiles_expected/read/skipped``) and gated
    at the END of extraction: any city with ``skipped/expected > 0.1`` raises — the
    silent-shrinkage ceiling; a partial city must be re-extracted or explicitly
    excluded, never quietly thinned (readiness F3). A zero-tile city is its own
    loud error (never a valid extraction target).

    Runs on Leonardo against the real corpus; there is no corpus locally.
    """
    features: dict[tuple[str, tuple, str], list[float]] = {}
    tile_coverage: dict[str, TileCoverage] = {}

    for city in cities:
        if tiles_by_city is not None and city in tiles_by_city:
            tiles = tiles_by_city[city]
        else:
            manifest = yaml.safe_load(holdout_manifest_for_region(release, city).read_text())
            tiles = manifest["regions"][city]["tiles"]
        epsg = epsg_label_for_region(city)
        sub_d = sub_d_region_dir(release, city)
        sub_f = sub_f_region_dir(release, city)
        n_read = 0
        n_skipped = 0
        n_bref_excluded = 0

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
                n_skipped += 1
                continue

            labels = read_tile_labels(sub_d / dirname, tile_i=ti, tile_j=tj)
            zoning = labels.morphology_stratum.dominant_zoning_class
            skeleton = labels.morphology_stratum.modal_road_skeleton_class
            coastal = labels.coastal_inland_river

            cdbc = _cell_density_by_cell(sub_d / dirname)
            tokens = read_sub_f_cells(cells_path)
            blocks, geoms, density_strata = decode_region_blocks(tokens, cdbc)
            tile_feats, n_excluded = _tile_features(blocks, geoms, density_strata)
            n_bref_excluded += n_excluded
            for metric, value, density in tile_feats:
                stratum = (zoning, skeleton, density, coastal)
                features.setdefault((city, stratum, metric), []).append(value)
            n_read += 1

        tile_coverage[city] = TileCoverage(
            n_tiles_expected=len(tiles),
            n_tiles_read=n_read,
            n_tiles_skipped=n_skipped,
            n_bref_excluded=n_bref_excluded,
        )

    # END-of-extraction HALT (readiness F3): a partial city can no longer quietly thin.
    for city, cov in tile_coverage.items():
        if cov.n_tiles_expected == 0:
            # DECISION: ValueError (invalid extraction target), distinct from the
            # RuntimeError shrinkage halt. Revisit if callers need to catch both as one.
            raise ValueError(
                f"gate-(i) extraction: city '{city}' has zero tiles in its holdout "
                "manifest — a zero-tile city is never a valid extraction target."
            )
        frac = cov.n_tiles_skipped / cov.n_tiles_expected
        if frac > _SHRINKAGE_CEILING:
            raise RuntimeError(
                f"gate-(i) extraction: city '{city}' skipped "
                f"{cov.n_tiles_skipped}/{cov.n_tiles_expected} manifest tiles "
                f"(missing cells.parquet; fraction {frac:.3f} > silent-shrinkage "
                f"ceiling {_SHRINKAGE_CEILING}). A partial city must be re-extracted "
                "or explicitly excluded, never quietly thinned (readiness F3)."
            )

    return ExtractionResult(features=features, tile_coverage=tile_coverage)
