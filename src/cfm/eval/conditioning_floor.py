"""Conditioning-floor artifact: pure logic + artifact IO (Task 25 step 1; spec §8).

The re-scoped Phase-2 eval claim: *given conditioning for geometry it did not
train on, the model produces plausible geometry matching that character — by
learned grammar, not memorization.* This module is its measurement instrument:

TWO SEPARATE BH FAMILIES (PI call 2026-06-11; supersedes the single joint
family — BH is family-size-dependent, so a joint family over held-out AND
training cities would shift the held-out p_bh whenever training cities are
extracted, breaking the stage-1 <-> stage-2 determinism check, while ~46k
T-vs-T pairs nothing consumes dilute BH power for Lane-M's D-vs-T strata):

- **Family 1 (D-D, "held-out pairwise")** — ``compute_pair_table`` over
  features RESTRICTED to held-out cities (the payload builder filters), per
  (metric, stratum) with >= 2 qualifying cities (the same ``min_n`` qualify
  rule as ``conditioning_discrimination_verdict``); its OWN BH. This family is
  literally the stage-1 computation — the bit-identity determinism anchor —
  and carries the delta ladder + the collapse/explosion halts.
- **Family 2 (D-T, "cross")** — ``compute_cross_pair_table``: ONLY (d, t)
  pairs with d held-out, t training, same qualify rule, its OWN BH over its
  own p_raws. No D-D, no T-T pair ever exists in this family. It feeds the
  Lane-M discriminating-strata selection and the floor_all tightening.
- **Floors (PI knob 1, STRICT min; two variants per row)** — per held-out
  city D and (metric, stratum): ``floor_heldout`` = min KS over D's family-1
  pairs (the stage-1 floor, reported context + determinism anchor) and
  ``floor_all`` = min KS over D's family-1 u family-2 pairs (KS only,
  BH-independent) — **Lane S scores floor_all** (PI rationale: closest real
  city, period; excluding training evidence would inflate the bar). With no
  qualifying cross pairs, ``floor_all == floor_heldout`` exactly. Context
  medians ride along for each variant, never as the floor.
- **Integrity halts (PI knob 4)** — ``FloorCollapseError`` if FAMILY 1's
  median KS < 0.049 (broken extraction: contradicts every prior run) and
  ``FloorExplosionError`` if it exceeds 0.5 (conditioning carries nothing).
  Both fire in the payload producer, BEFORE any artifact byte is written.
- **Lane S** — per qualifying (metric, stratum):
  ``excess = max(0, KS(gen_D, real_D) - floor_all_D)``; per-city aggregate is
  the median + p90 over strata (PI knob 3). Scoring REFUSES to run unless the
  floor artifact loads verified (Task-20 reader-side discipline).
- **Lane M** — the memorization discriminator: over strata where D is
  measured-distinct from training city T (CROSS-family KS >= delta AND
  cross-BH-significant, selected from REAL data only), require
  ``median KS(gen, real_D) < median KS(gen, real_T)`` strictly. The all-38
  orchestration is the Task-26 decision layer; the function + teeth live here.

NO-LEAKAGE BY SIGNATURE: ``select_discriminating_strata`` accepts ONLY the
real-real pair table (+ the delta knob) — generated data has no parameter to
arrive through, and the pin test asserts the signature.

DECISION (orchestrator-fixed): the artifact stores KS TABLES, never raw
samples — a 38-city raw-sample dump is a multi-hundred-MB YAML. Eval-time
Lane M live-reads the real samples on Leonardo instead.

Lock grammar mirrors the Task-20 / 24a-registry discipline: ``floor_sha256``
over the canonical YAML EXCLUDING itself + a ``_CONDITIONING_FLOOR_LOCKED``
marker beside the file; write-once; the reader refuses absent file / absent
marker / absent sha / sha mismatch / malformed YAML / schema-version skew
under one ``FloorArtifactError`` taxonomy.
"""

from __future__ import annotations

import copy
import logging
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml
from cfm.eval.conditioning_discrimination import (
    benjamini_hochberg,
    ks_pvalue,
    noise_floor,
)
from cfm.eval.realism import ks_distance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cfm.eval.conditioning_discrimination import TileCoverage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants (PI knob 4 thresholds are user thresholds: STRICT comparisons)
# --------------------------------------------------------------------------- #

#: Collapse halt: pair-table median KS strictly below this contradicts every
#: prior run (single-region KS floor 0.049) => broken extraction, not a result.
FLOOR_COLLAPSE_MEDIAN: float = 0.049

#: Explosion halt: pair-table median KS strictly above this means conditioning
#: carries nothing (cities are mutually alien within identical strata).
FLOOR_EXPLOSION_MEDIAN: float = 0.5

#: Reporting-ladder anchors over FAMILY 1 (delta=0.15 is ALSO the
#: discriminating-strata rule, which applies to the CROSS family).
DELTA_LADDER_ANCHORS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

#: 2.0 (PI call 2026-06-11): two-BH-family payload — ``pairs`` + ``cross_pairs``
#: + two-variant floor rows (``floor_heldout``/``floor_all``). 1.0 artifacts
#: (single joint family, single ``floor`` field) refuse to load — correct.
FLOOR_ARTIFACT_SCHEMA_VERSION: str = "2.0"
FLOOR_ARTIFACT_LOCK_NAME: str = "_CONDITIONING_FLOOR_LOCKED"


class FloorArtifactError(RuntimeError):
    """The floor artifact failed verification (absent / unsealed / tampered /
    version-skewed / malformed) or its content cannot serve the request."""


class FloorCollapseError(RuntimeError):
    """FAMILY-1 (held-out pairwise) median KS < 0.049: extraction is broken
    (contradicts every prior run); the artifact must not be written."""


class FloorExplosionError(RuntimeError):
    """FAMILY-1 (held-out pairwise) median KS > 0.5: conditioning carries
    nothing; the artifact must not be written."""


# --------------------------------------------------------------------------- #
# Pair table (real-real KS + BH significance)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FloorPair:
    """One unordered real-real city-pair KS within a (metric, stratum)."""

    metric: str
    stratum: tuple
    city_a: str
    city_b: str
    n_a: int
    n_b: int
    ks: float
    noise_floor: float
    p_raw: float
    p_bh: float  # BH-adjusted within THIS table's family only (never joint)


@dataclass(frozen=True)
class PairTable:
    """The real-real KS table the floors, halts, and strata all derive from."""

    pairs: tuple[FloorPair, ...]
    min_n: int
    alpha: float
    n_excluded_thin: int
    n_strata_too_few_cities: int

    def median_ks(self) -> float:
        """Median KS over every pair (the integrity-halt statistic)."""
        if not self.pairs:
            raise ValueError("median_ks over an empty pair table is undefined")
        return statistics.median(p.ks for p in self.pairs)


def _stratum_sort_key(stratum: tuple) -> tuple[str, ...]:
    return tuple(str(x) for x in stratum)


def _metric_stratum_key(ms: tuple[str, tuple]) -> tuple[str, tuple[str, ...]]:
    """The (metric, stratum) sort key shared by the table, the strata selection,
    the payload builder, and Lane S — named ONCE so the orders cannot drift."""
    return (ms[0], _stratum_sort_key(ms[1]))


def _qualified_by_metric_stratum(
    features: dict[tuple[str, tuple, str], list[float]],
    min_n: int,
) -> tuple[dict[tuple[str, tuple], dict[str, list[float]]], int]:
    """The SHARED qualify (>= ``min_n``, same rule as the verdict fn) ->
    group-by-(metric, stratum) step both BH families build on."""
    qualified = {key: vals for key, vals in features.items() if len(vals) >= min_n}
    n_excluded_thin = len(features) - len(qualified)
    by_metric_stratum: dict[tuple[str, tuple], dict[str, list[float]]] = {}
    for (city, stratum, metric), vals in qualified.items():
        by_metric_stratum.setdefault((metric, stratum), {})[city] = vals
    return by_metric_stratum, n_excluded_thin


def _ks_pair(
    metric: str, stratum: tuple, ca: str, cb: str, a: list[float], b: list[float]
) -> FloorPair:
    """One KS pair record with the BH placeholder (filled per-family later)."""
    d = ks_distance(a, b)
    return FloorPair(
        metric=metric,
        stratum=stratum,
        city_a=ca,
        city_b=cb,
        n_a=len(a),
        n_b=len(b),
        ks=d,
        noise_floor=noise_floor(len(a), len(b)),
        p_raw=ks_pvalue(d, len(a), len(b)),
        p_bh=1.0,  # placeholder; filled by the per-FAMILY BH
    )


def _with_family_bh(raw: list[FloorPair]) -> tuple[FloorPair, ...]:
    """BH-adjust over THIS family's p_raws only — each family is its own
    multiple-testing universe (the PI two-family decision)."""
    adjusted = benjamini_hochberg([p.p_raw for p in raw])
    return tuple(replace(p, p_bh=adj) for p, adj in zip(raw, adjusted, strict=True))


def compute_pair_table(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    min_n: int = 50,
    alpha: float = 0.05,
) -> PairTable:
    """FAMILY 1 when fed held-out-restricted features (the payload builder's
    job — this function pairs every city it sees): every unordered city-pair KS
    per (metric, stratum) with >= 2 qualifying cities; the qualify rule
    (>= ``min_n`` samples) is the SAME as ``conditioning_discrimination_verdict``.
    BH-adjusted p-values are computed across all of THIS table's pairs (both
    metrics, all strata) — its own family, never joint with the cross table.
    Iteration order is sorted, so the table — and the artifact sha downstream —
    is insertion-order independent (PYTHONHASHSEED-proof). Parity with the
    verdict fn's qualify -> group -> pair -> global-BH steps is PINNED by
    ``test_pair_table_parity_with_the_verdict_fn`` (external source of truth)."""
    by_metric_stratum, n_excluded_thin = _qualified_by_metric_stratum(features, min_n)

    raw: list[FloorPair] = []
    n_strata_too_few_cities = 0
    for metric, stratum in sorted(by_metric_stratum, key=_metric_stratum_key):
        city_samples = by_metric_stratum[(metric, stratum)]
        cities = sorted(city_samples)
        if len(cities) < 2:
            n_strata_too_few_cities += 1
            continue
        for ia in range(len(cities)):
            for ib in range(ia + 1, len(cities)):
                ca, cb = cities[ia], cities[ib]
                raw.append(_ks_pair(metric, stratum, ca, cb, city_samples[ca], city_samples[cb]))

    return PairTable(
        pairs=_with_family_bh(raw),
        min_n=min_n,
        alpha=alpha,
        n_excluded_thin=n_excluded_thin,
        n_strata_too_few_cities=n_strata_too_few_cities,
    )


def compute_cross_pair_table(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    held_out: Sequence[str],
    train: Sequence[str],
    min_n: int = 50,
    alpha: float = 0.05,
) -> PairTable:
    """FAMILY 2 (D-T, "cross"): per (metric, stratum), ONLY the (d, t) pairs
    with d held-out and t training where BOTH qualify (same ``min_n`` rule);
    BH over its OWN p_raws — no D-D, no T-T pair can exist here by
    construction. Pair records store ``(city_a, city_b)`` sorted (the same
    grammar as family 1), so ``select_discriminating_strata`` and the Lane-M
    ``tuple(sorted((d, t)))`` indexing work unchanged on this table.
    ``n_strata_too_few_cities`` counts strata where no cross pair is possible
    (no qualifying held-out city or no qualifying training city)."""
    held_set, train_set = set(held_out), set(train)
    overlap = sorted(held_set & train_set)
    if overlap:
        raise ValueError(
            f"cross pair table: cities in BOTH held_out and train: {overlap} — "
            "the D-vs-T boundary is broken; refusing."
        )
    by_metric_stratum, n_excluded_thin = _qualified_by_metric_stratum(features, min_n)

    raw: list[FloorPair] = []
    n_strata_no_cross_pair = 0
    for metric, stratum in sorted(by_metric_stratum, key=_metric_stratum_key):
        city_samples = by_metric_stratum[(metric, stratum)]
        ds = sorted(c for c in city_samples if c in held_set)
        ts = sorted(c for c in city_samples if c in train_set)
        if not ds or not ts:
            n_strata_no_cross_pair += 1
            continue
        for d_city in ds:
            for t_city in ts:
                ca, cb = sorted((d_city, t_city))
                raw.append(_ks_pair(metric, stratum, ca, cb, city_samples[ca], city_samples[cb]))

    return PairTable(
        pairs=_with_family_bh(raw),
        min_n=min_n,
        alpha=alpha,
        n_excluded_thin=n_excluded_thin,
        n_strata_too_few_cities=n_strata_no_cross_pair,
    )


# --------------------------------------------------------------------------- #
# Floors (PI knob 1: STRICT min over other cities; median is context ONLY)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FloorEntry:
    """Both floor variants per (city, metric, stratum) — each a STRICT min,
    each with its median as context ONLY:

    - ``floor_heldout``: min KS over D's FAMILY-1 (held-out) pairs — exactly
      the stage-1 floor; reported context AND the stage-1 bit-identity
      determinism anchor.
    - ``floor_all``: min KS over D's family-1 u family-2 pairs (KS only,
      BH-independent) — the value **Lane S scores** (PI call 2026-06-11:
      closest real city, period; excluding training evidence would inflate
      the bar). Equals ``floor_heldout`` when D has no qualifying cross pair.
    """

    floor_heldout: float
    floor_heldout_median_context: float
    floor_all: float
    floor_all_median_context: float
    n_heldout_pairs: int
    n_cross_pairs: int


def _ks_by_held_city_key(
    pairs: tuple[FloorPair, ...], held: set[str]
) -> dict[tuple[str, str, tuple], list[float]]:
    out: dict[tuple[str, str, tuple], list[float]] = {}
    for p in pairs:
        for d, _other in ((p.city_a, p.city_b), (p.city_b, p.city_a)):
            if d in held:
                out.setdefault((d, p.metric, p.stratum), []).append(p.ks)
    return out


def compute_floors(
    pair_table: PairTable,
    held_out_cities: Sequence[str],
    *,
    cross_table: PairTable | None = None,
) -> dict[str, dict[tuple[str, tuple], FloorEntry]]:
    """Per held-out city D, per (metric, stratum): ``floor_heldout`` from D's
    family-1 pairs and ``floor_all`` from family-1 u family-2 (``cross_table``)
    pairs, both strict mins with context medians.

    DECISION: floor rows are keyed by FAMILY-1 floors — a (city, metric,
    stratum) with cross pairs but ZERO family-1 pairs gets NO floor row
    (floor_heldout, the determinism anchor, would be undefined there); such
    strata still feed ``cross_pairs`` and the Lane-M strata selection. Revisit
    if a real run shows Lane-S-relevant strata floored only by training cities."""
    held = set(held_out_cities)
    heldout_ks = _ks_by_held_city_key(pair_table.pairs, held)
    cross_ks = _ks_by_held_city_key(cross_table.pairs if cross_table is not None else (), held)

    floors: dict[str, dict[tuple[str, tuple], FloorEntry]] = {}
    for (d, metric, stratum), ks_values in heldout_ks.items():
        cross_values = cross_ks.get((d, metric, stratum), [])
        all_values = ks_values + cross_values
        floors.setdefault(d, {})[(metric, stratum)] = FloorEntry(
            floor_heldout=min(ks_values),
            floor_heldout_median_context=statistics.median(ks_values),
            floor_all=min(all_values),
            floor_all_median_context=statistics.median(all_values),
            n_heldout_pairs=len(ks_values),
            n_cross_pairs=len(cross_values),
        )
    return floors


# --------------------------------------------------------------------------- #
# Integrity halts (PI knob 4; fire BEFORE any artifact byte)
# --------------------------------------------------------------------------- #


def assert_floor_sanity(pair_table: PairTable) -> None:
    """Raise the regime-matching halt; pass silently on a healthy table.

    FAMILY-1 ONLY (PI call 2026-06-11): these halts — and the delta ladder —
    live on the D-D held-out pairwise family, the floor-measurement core and
    the stage-1 determinism anchor. The cross (D-T) family never feeds them:
    a halt that moved when training cities were added would let the joint
    family blur the floor lineage.

    Zero qualifying pairs is UNSUPPORTED — loud, never a silent empty artifact
    (the Task-22 UNSUPPORTED contract: report, do NOT coarsen)."""
    if not pair_table.pairs:
        raise ValueError(
            "conditioning-floor: UNSUPPORTED — zero qualifying city-pairs at "
            f"min_n={pair_table.min_n} (thin-excluded cells: "
            f"{pair_table.n_excluded_thin}; strata with <2 cities: "
            f"{pair_table.n_strata_too_few_cities}). Refusing to freeze a "
            "silent empty artifact."
        )
    median = pair_table.median_ks()
    if median < FLOOR_COLLAPSE_MEDIAN:
        raise FloorCollapseError(
            f"conditioning-floor collapse: pair-table median KS {median:.4f} < "
            f"{FLOOR_COLLAPSE_MEDIAN} — contradicts every prior run (single-region "
            "floor 0.049); extraction is broken. No artifact written."
        )
    if median > FLOOR_EXPLOSION_MEDIAN:
        raise FloorExplosionError(
            f"conditioning-floor explosion: pair-table median KS {median:.4f} > "
            f"{FLOOR_EXPLOSION_MEDIAN} — conditioning carries nothing. "
            "No artifact written."
        )


# --------------------------------------------------------------------------- #
# Discriminating strata (NO-LEAKAGE: real-real pair table is the ONLY input)
# --------------------------------------------------------------------------- #


def select_discriminating_strata(
    pair_table: PairTable,
    *,
    delta: float = 0.15,
) -> dict[tuple[str, str], tuple[tuple[str, tuple], ...]]:
    """Per unordered city-pair: the (metric, stratum) keys where the REAL-REAL
    KS >= ``delta`` AND the pair is BH-significant (p_bh < the table's alpha).

    PRODUCTION INPUT IS THE CROSS TABLE (PI call 2026-06-11): the payload
    builder feeds family 2, so the selection is per (D, T) at cross-family BH
    significance — the Lane-M strata family. The signature stays
    pair-table-only on purpose; the function is family-agnostic.

    NO-LEAKAGE BY SIGNATURE: this takes ONLY a real-real pair table —
    generated data cannot reach the selection. Every pair present in the table
    gets an entry (possibly empty), so an absent pair is distinguishable from
    a pair with no discriminating strata. Lane-M consumers index the result by
    ``tuple(sorted((city_d, city_t)))``."""
    out: dict[tuple[str, str], list[tuple[str, tuple]]] = {}
    for p in pair_table.pairs:
        key = (p.city_a, p.city_b)  # already sorted by construction
        out.setdefault(key, [])
        if p.ks >= delta and p.p_bh < pair_table.alpha:
            out[key].append((p.metric, p.stratum))
    return {key: tuple(sorted(strata, key=_metric_stratum_key)) for key, strata in out.items()}


# --------------------------------------------------------------------------- #
# Artifact payload + freeze / verified load (Task-20 grammar)
# --------------------------------------------------------------------------- #


def _delta_ladder(pair_table: PairTable) -> list[dict]:
    return [
        {
            "delta": anchor,
            "n_pairs": sum(
                1 for p in pair_table.pairs if p.ks >= anchor and p.p_bh < pair_table.alpha
            ),
        }
        for anchor in DELTA_LADDER_ANCHORS
    ]


def _pair_records(table: PairTable) -> list[dict]:
    """The ONE pair-record grammar, shared by both families."""
    return [
        {
            "metric": p.metric,
            "stratum": list(p.stratum),
            "city_a": p.city_a,
            "city_b": p.city_b,
            "n_a": p.n_a,
            "n_b": p.n_b,
            "ks": float(p.ks),
            "noise_floor": float(p.noise_floor),
            "p_raw": float(p.p_raw),
            "p_bh": float(p.p_bh),
        }
        for p in table.pairs
    ]


def build_floor_artifact_payload(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    release: str,
    held_out_cities: Sequence[str],
    train_cities: Sequence[str] = (),
    min_n: int = 50,
    alpha: float = 0.05,
    delta: float = 0.15,
    tile_coverage: dict[str, TileCoverage] | None = None,
) -> dict:
    """The YAML-safe artifact payload from raw (city, stratum, metric) features.

    TWO BH FAMILIES (PI call 2026-06-11; supersedes the single joint family):
    family 1 is ``compute_pair_table`` over features RESTRICTED HERE to the
    held-out cities — so family 1 is literally the stage-1 computation (the
    bit-identity determinism anchor) regardless of whether training cities were
    extracted; the integrity halts and the delta ladder live on it. Family 2
    (``compute_cross_pair_table``, built only when training cities are present)
    pairs ONLY (d, t) with its OWN BH and feeds the Lane-M strata selection +
    the floor_all tightening. No T-T pair exists anywhere.

    Runs ``assert_floor_sanity`` (on FAMILY 1) FIRST so a collapse/explosion/
    UNSUPPORTED regime halts before any artifact content exists. Stores KS
    tables, never raw samples (see module docstring).

    MISSING-CITY HALT (Task-25 spec review #3): every held-out city must come
    out of ``compute_floors`` floored, or this raises BEFORE any artifact byte.
    Rationale: a held-out city silently absent from ``floors`` shrinks the
    worst-case max domain weeks later at Lane-S consumption against a
    write-once artifact — the aggregate-hides-subsets class, the same failure
    the global zero-pairs UNSUPPORTED halt already guards.

    UNKNOWN-CITY HALT (quality review, symmetric with the above): a city present
    in ``features`` but in NEITHER ``held_out_cities`` nor ``train_cities``
    would be silently dropped by the family filters below — unreachable via
    today's runner, reachable the moment extraction is cached/reused (the
    sample-regime-blind shape). Loud, naming the cities."""
    held_set = set(held_out_cities)
    unknown = sorted({key[0] for key in features} - held_set - set(train_cities))
    if unknown:
        raise ValueError(
            "conditioning-floor: cities present in features but in NEITHER "
            f"held_out_cities nor train_cities: {unknown} — the family filters "
            "would silently drop their samples (no pair, no floor, no trace); "
            "refusing."
        )
    family1_features = {key: vals for key, vals in features.items() if key[0] in held_set}
    pair_table = compute_pair_table(family1_features, min_n=min_n, alpha=alpha)
    assert_floor_sanity(pair_table)

    cross_table = (
        compute_cross_pair_table(
            features, held_out=held_out_cities, train=train_cities, min_n=min_n, alpha=alpha
        )
        if train_cities
        else None
    )

    floors = compute_floors(pair_table, held_out_cities, cross_table=cross_table)
    missing = sorted(set(held_out_cities) - set(floors))
    if missing:
        raise ValueError(
            "conditioning-floor: UNSUPPORTED — held-out cities with ZERO "
            f"qualifying pairs in the table (no floor derivable): {missing} "
            f"at min_n={pair_table.min_n}. A write-once artifact missing a "
            "held-out city silently shrinks the Lane-S worst-case max domain; "
            "refusing to freeze."
        )
    strata = (
        select_discriminating_strata(cross_table, delta=delta) if cross_table is not None else {}
    )

    floor_records = [
        {
            "city": city,
            "metric": metric,
            "stratum": list(stratum),
            "floor_heldout": float(entry.floor_heldout),
            "floor_heldout_median_context": float(entry.floor_heldout_median_context),
            "floor_all": float(entry.floor_all),
            "floor_all_median_context": float(entry.floor_all_median_context),
            "n_heldout_pairs": entry.n_heldout_pairs,
            "n_cross_pairs": entry.n_cross_pairs,
        }
        for city in sorted(floors)
        for (metric, stratum), entry in sorted(
            floors[city].items(), key=lambda kv: _metric_stratum_key(kv[0])
        )
    ]
    strata_records = [
        {
            "city_a": ca,
            "city_b": cb,
            "strata": [{"metric": m, "stratum": list(s)} for m, s in strata[(ca, cb)]],
        }
        for ca, cb in sorted(strata)
    ]
    coverage_records = {
        city: {
            "n_tiles_expected": cov.n_tiles_expected,
            "n_tiles_read": cov.n_tiles_read,
            "n_tiles_skipped": cov.n_tiles_skipped,
            "n_bref_excluded": cov.n_bref_excluded,
        }
        for city, cov in sorted((tile_coverage or {}).items())
    }
    return {
        "floor_schema_version": FLOOR_ARTIFACT_SCHEMA_VERSION,
        "release": release,
        "methodology": {
            "min_n": pair_table.min_n,
            "alpha": pair_table.alpha,
            "delta": delta,
            "delta_ladder_anchors": list(DELTA_LADDER_ANCHORS),
            "floor_collapse_median": FLOOR_COLLAPSE_MEDIAN,
            "floor_explosion_median": FLOOR_EXPLOSION_MEDIAN,
            # PI call 2026-06-11: two separate BH families, no T-T pairs ever
            "bh_families": "family1_heldout_dd_pairwise+family2_dt_cross_no_tt",
            # PI knob 1. REVERSE-LOCK 2026-06-11: renamed from
            # "strict_min_over_other_cities" — ambiguous across the two variants
            # (floor_heldout mins over held-out pairs only; floor_all over both).
            "floor_rule": "strict_min_per_variant",
            "floor_context_rule": "median_over_other_cities_context_only",
            "floor_scored": "floor_all",  # what Lane S consumes
            "floor_context": "floor_heldout",  # reported + determinism anchor
            "floor_scored_rationale": (
                "closest real city, period; excluding training evidence would inflate the bar"
            ),
            "lane_s_aggregate_rule": "median_p90_over_strata",  # PI knob 3
            "lane_m_scope": "all_training_cities",  # PI knob 2
        },
        "held_out_cities": sorted(held_out_cities),
        "train_cities": sorted(train_cities),
        "n_excluded_thin": pair_table.n_excluded_thin,
        "n_strata_too_few_cities": pair_table.n_strata_too_few_cities,
        "cross_n_excluded_thin": cross_table.n_excluded_thin if cross_table is not None else 0,
        "cross_n_strata_no_cross_pair": (
            cross_table.n_strata_too_few_cities if cross_table is not None else 0
        ),
        "pair_table_median_ks": float(pair_table.median_ks()),
        "cross_median_ks": (
            float(cross_table.median_ks())
            if cross_table is not None and cross_table.pairs
            else None
        ),
        "pairs": _pair_records(pair_table),
        "cross_pairs": _pair_records(cross_table) if cross_table is not None else [],
        "floors": floor_records,
        "discriminating_strata": strata_records,
        "delta_ladder": _delta_ladder(pair_table),
        "tile_coverage": coverage_records,
    }


def _pairs_from_records(records: list[dict]) -> tuple[FloorPair, ...]:
    return tuple(
        FloorPair(
            metric=rec["metric"],
            stratum=tuple(rec["stratum"]),
            city_a=rec["city_a"],
            city_b=rec["city_b"],
            n_a=int(rec["n_a"]),
            n_b=int(rec["n_b"]),
            ks=float(rec["ks"]),
            noise_floor=float(rec["noise_floor"]),
            p_raw=float(rec["p_raw"]),
            p_bh=float(rec["p_bh"]),
        )
        for rec in records
    )


def pair_table_from_payload(payload: dict) -> PairTable:
    """Rebuild the FAMILY-1 (held-out pairwise) table from an artifact payload."""
    return PairTable(
        pairs=_pairs_from_records(payload["pairs"]),
        min_n=int(payload["methodology"]["min_n"]),
        alpha=float(payload["methodology"]["alpha"]),
        n_excluded_thin=int(payload["n_excluded_thin"]),
        n_strata_too_few_cities=int(payload["n_strata_too_few_cities"]),
    )


def cross_pair_table_from_payload(payload: dict) -> PairTable:
    """Rebuild the FAMILY-2 (D-T cross) table from an artifact payload."""
    return PairTable(
        pairs=_pairs_from_records(payload["cross_pairs"]),
        min_n=int(payload["methodology"]["min_n"]),
        alpha=float(payload["methodology"]["alpha"]),
        n_excluded_thin=int(payload["cross_n_excluded_thin"]),
        n_strata_too_few_cities=int(payload["cross_n_strata_no_cross_pair"]),
    )


def floor_artifact_sha256(data: dict) -> str:
    """SHA over the canonical payload EXCLUDING the floor_sha256 field itself
    (the same ``*_sha256`` exclusion grammar as the holdout manifest freeze)."""
    payload = {k: v for k, v in data.items() if k != "floor_sha256"}
    return compute_sha256(canonicalize_yaml(payload).encode("utf-8"))


def freeze_floor_artifact(payload: dict, path: Path) -> None:
    """Stamp the sha, write ONCE, seal with the lock marker beside the file.

    Refuses to overwrite: re-measuring floors means deliberately deleting the
    old artifact first (the eval-set write-once discipline).

    DEDICATED-DIRECTORY rationale (Task-25 quality review #4): the
    ``_CONDITIONING_FLOOR_LOCKED`` marker is PER-DIRECTORY, so the artifact must
    live in a directory of its own (the runner defaults to
    ``reports/conditioning_floor/<release>/``). In a busy shared directory like
    ``reports/`` a marker left by an earlier freeze would sit beside any LATER
    hand-dropped artifact and make it look sealed — the stale-marker-seals-
    new-artifact hazard."""
    if "floor_schema_version" not in payload:
        raise FloorArtifactError(
            "refusing to freeze a floor payload without floor_schema_version — "
            "the verified reader would refuse it; build via "
            "build_floor_artifact_payload."
        )
    if path.exists():
        raise FileExistsError(
            f"conditioning-floor artifact already locked at {path}; it is "
            "write-once — delete deliberately only to re-measure the floors."
        )
    frozen = dict(payload)
    frozen["floor_sha256"] = floor_artifact_sha256(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(frozen), encoding="utf-8")
    (path.parent / FLOOR_ARTIFACT_LOCK_NAME).touch()


#: Module-private construction proof (Task-25 quality review #3): only
#: ``load_verified_floor`` holds a reference to this exact object, so only a
#: verified read can mint a VerifiedFloorArtifact. NOT exported; never pass it.
_CONSTRUCTION_TOKEN: object = object()


@dataclass(frozen=True)
class VerifiedFloorArtifact:
    """The PROOF-CARRYING load result: only ``load_verified_floor`` constructs
    one on a verified read; Lane S / strata accessors refuse anything else.

    ENFORCED: direct construction is refused — ``__post_init__`` checks the
    keyword-only ``_token`` against a module-private sentinel that only
    ``load_verified_floor`` passes, so ``VerifiedFloorArtifact(path, payload)``
    cannot forge the proof. The ``payload`` is DEEP-COPIED at construction, so
    no alias held by the constructor's caller can mutate it after the verify.

    NOT ENFORCED: in-place mutation of ``.payload`` itself (Python dicts cannot
    be frozen). A mutated copy never propagates anywhere: a re-load re-reads
    and re-verifies the sealed file from disk."""

    path: Path
    payload: dict
    _token: object = field(kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CONSTRUCTION_TOKEN:
            raise FloorArtifactError(
                "VerifiedFloorArtifact cannot be constructed directly — the proof "
                "token is module-private and only load_verified_floor (a verified "
                "read) mints one; refusing a forged artifact."
            )
        object.__setattr__(self, "payload", copy.deepcopy(self.payload))


def load_verified_floor(path: Path) -> VerifiedFloorArtifact:
    """Verified read: marker beside the file, stored sha == recomputed sha,
    schema version matched. Refuses absent file / absent marker / malformed
    YAML / absent sha / sha mismatch / version skew (one taxonomy)."""
    path = Path(path)
    if not path.exists():
        raise FloorArtifactError(
            f"conditioning-floor artifact {path} does not exist; Lane S cannot "
            "score without a frozen, verified floor artifact."
        )
    marker = path.parent / FLOOR_ARTIFACT_LOCK_NAME
    if not marker.exists():
        raise FloorArtifactError(
            f"no {FLOOR_ARTIFACT_LOCK_NAME} marker beside the floor artifact "
            f"(expected {marker}); refusing to read an unsealed artifact."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FloorArtifactError(
            f"malformed floor artifact at {path}: unparseable YAML ({exc}); refusing (fail-closed)."
        ) from exc
    if not isinstance(data, dict) or "floors" not in data:
        raise FloorArtifactError(
            f"malformed floor artifact at {path}: expected a YAML mapping with a "
            f"'floors' key (got {type(data).__name__}); refusing (fail-closed)."
        )
    stored = data.get("floor_sha256")
    if stored is None:
        raise FloorArtifactError(
            f"floor artifact {path} carries NO floor_sha256 field — an unstamped "
            "artifact is unverifiable; refusing (fail-closed)."
        )
    recomputed = floor_artifact_sha256(data)
    if stored != recomputed:
        raise FloorArtifactError(
            f"floor artifact sha mismatch at {path}: stored floor_sha256="
            f"{stored!r} but recomputed {recomputed!r} — the content was edited "
            "after the freeze; refusing (the floor is a locked instrument)."
        )
    version = data.get("floor_schema_version")
    if version != FLOOR_ARTIFACT_SCHEMA_VERSION:
        raise FloorArtifactError(
            f"floor artifact {path} declares floor_schema_version={version!r} "
            f"but this reader requires {FLOOR_ARTIFACT_SCHEMA_VERSION!r}; "
            "refusing a version-skewed artifact."
        )
    return VerifiedFloorArtifact(path=path, payload=data, _token=_CONSTRUCTION_TOKEN)


def discriminating_strata_from_artifact(
    artifact: VerifiedFloorArtifact,
    city_d: str,
    city_t: str,
) -> tuple[tuple[str, tuple], ...]:
    """The frozen discriminating strata for an unordered (D, T) pair — the
    Lane-M path derives strata EXCLUSIVELY from the verified artifact; a raw
    payload dict (anything that skipped ``load_verified_floor``) is refused,
    and a pair absent from the table is loud, never an empty selection."""
    if not isinstance(artifact, VerifiedFloorArtifact):
        raise FloorArtifactError(
            "discriminating strata require a VerifiedFloorArtifact (the "
            "load_verified_floor result); refusing an unverified "
            f"{type(artifact).__name__}."
        )
    ca, cb = sorted((city_d, city_t))
    for rec in artifact.payload["discriminating_strata"]:
        if rec["city_a"] == ca and rec["city_b"] == cb:
            return tuple((s["metric"], tuple(s["stratum"])) for s in rec["strata"])
    raise FloorArtifactError(
        f"floor artifact {artifact.path} holds no pair ({ca}, {cb}) — the pair "
        "never qualified in the real-real table; refusing to silently treat it "
        "as 'no discriminating strata'."
    )


# --------------------------------------------------------------------------- #
# Lane S — scored generalization (excess over the verified floor)
# --------------------------------------------------------------------------- #


def _resolve_min_n(
    explicit: int | None, verified: VerifiedFloorArtifact | None, *, lane: str
) -> int:
    """Lane S/M ``min_n`` resolution (Task-25 quality review #6): default is the
    verified artifact's frozen ``methodology.min_n`` (ONE qualify rule end-to-end);
    an explicit value that disagrees is honored but warned LOUDLY — never a
    silent rescore at a different min_n. Without an artifact in scope (Lane M
    called bare) the legacy default 50 applies."""
    if verified is None:
        return 50 if explicit is None else explicit
    artifact_min_n = int(verified.payload["methodology"]["min_n"])
    if explicit is None:
        return artifact_min_n
    if explicit != artifact_min_n:
        logger.warning(
            "%s: scoring at explicit min_n=%d but the floor artifact %s was derived "
            "at min_n=%d — a DIFFERENT qualify rule than the frozen floors "
            "(explicit override, never silent).",
            lane,
            explicit,
            verified.path,
            artifact_min_n,
        )
    return explicit


def _p90(values: list[float]) -> float:
    """p90 with linear interpolation between order statistics (matches the
    median convention: exact at the 0.9*(n-1) fractional rank)."""
    if not values:
        raise ValueError("p90 of an empty sequence is undefined")
    ordered = sorted(values)
    pos = 0.9 * (len(ordered) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(ordered):
        return ordered[lo]
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


@dataclass(frozen=True)
class LaneSResult:
    """Per-city Lane-S score: per-stratum excess + median/p90 aggregate."""

    city: str
    per_stratum_excess: dict[tuple[str, tuple], float]
    median_excess: float
    p90_excess: float
    n_qualifying: int
    n_skipped_thin: int


def lane_s_excess(
    gen_features: dict[tuple[str, tuple], list[float]],
    real_features: dict[tuple[str, tuple], list[float]],
    artifact: str | Path | VerifiedFloorArtifact,
    *,
    city: str,
    min_n: int | None = None,
) -> LaneSResult:
    """``excess = max(0, KS(gen, real_D) - floor_all_D)`` per qualifying
    (metric, stratum); aggregate = median + p90 over strata (PI knob 3).

    SCORED BAR = ``floor_all`` (PI call 2026-06-11): the floor is the closest
    real city, period — held-out or training; excluding training evidence
    would inflate the bar. ``floor_heldout`` stays in the artifact as reported
    context + the stage-1 determinism anchor, never the scored bar.

    REFUSAL TOOTH (Task-20 discipline): the artifact must load VERIFIED before
    any KS is computed — a str/Path is verified here; only the proof-carrying
    ``VerifiedFloorArtifact`` is accepted otherwise. Feature dicts are keyed
    ``(metric, stratum)`` for the single city ``city``. ``min_n`` defaults from
    the artifact's frozen ``methodology.min_n``; an explicit mismatch is warned,
    never silent (see ``_resolve_min_n``)."""
    if isinstance(artifact, (str, Path)):
        verified = load_verified_floor(Path(artifact))
    elif isinstance(artifact, VerifiedFloorArtifact):
        verified = artifact
    else:
        raise FloorArtifactError(
            "Lane S requires the floor-artifact PATH or a VerifiedFloorArtifact "
            f"(the load_verified_floor result); refusing an unverified "
            f"{type(artifact).__name__}."
        )
    min_n = _resolve_min_n(min_n, verified, lane="Lane S")

    floor_by_key = {
        (rec["metric"], tuple(rec["stratum"])): float(rec["floor_all"])  # the SCORED bar
        for rec in verified.payload["floors"]
        if rec["city"] == city
    }
    if not floor_by_key:
        raise ValueError(
            f"floor artifact {verified.path} holds no floors for city {city!r} — "
            "Lane S cannot score a city the artifact never floored."
        )

    per_stratum: dict[tuple[str, tuple], float] = {}
    n_skipped_thin = 0
    for key in sorted(floor_by_key, key=_metric_stratum_key):
        gen = gen_features.get(key, [])
        real = real_features.get(key, [])
        if len(gen) < min_n or len(real) < min_n:
            n_skipped_thin += 1
            continue
        per_stratum[key] = max(0.0, ks_distance(gen, real) - floor_by_key[key])

    if not per_stratum:
        raise ValueError(
            f"Lane S for city {city!r}: zero qualifying (metric, stratum) cells "
            f"at min_n={min_n} ({n_skipped_thin} floored cells were thin) — "
            "UNSUPPORTED, refusing a vacuous score."
        )
    excesses = list(per_stratum.values())
    return LaneSResult(
        city=city,
        per_stratum_excess=per_stratum,
        median_excess=statistics.median(excesses),
        p90_excess=_p90(excesses),
        n_qualifying=len(per_stratum),
        n_skipped_thin=n_skipped_thin,
    )


# --------------------------------------------------------------------------- #
# Lane M — memorization discriminator (one training city; Task 26 orchestrates)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LaneMResult:
    """One (D, T) discriminator verdict: PASS iff gen matches D strictly better
    (median KS(gen, D) < median KS(gen, T)); margin > 0 means PASS."""

    verdict: str  # "PASS" | "FAIL"
    median_ks_gen_d: float
    median_ks_gen_t: float
    margin: float  # median_ks_gen_t - median_ks_gen_d
    n_strata_scored: int
    n_strata_skipped_thin: int
    per_stratum: tuple[tuple[str, tuple, float, float], ...]  # (metric, stratum, ks_d, ks_t)


def lane_m_verdict(
    gen_d_features: dict[tuple[str, tuple], list[float]],
    real_d_features: dict[tuple[str, tuple], list[float]],
    real_t_features: dict[tuple[str, tuple], list[float]],
    discriminating_strata: Sequence[tuple[str, tuple]],
    *,
    min_n: int | None = None,
    artifact: VerifiedFloorArtifact | None = None,
) -> LaneMResult:
    """Over the REAL-data-selected discriminating strata for one (D, T): PASS
    iff ``median KS(gen, real_D) < median KS(gen, real_T)`` STRICTLY — a
    regurgitator of T's data sits at KS(gen, T) == 0 and cannot pass. Zero
    scoreable strata is loud (the verdict would be vacuous). The all-38
    training-city sweep (PI knob 2) is the Task-26 decision layer.

    ``artifact`` (optional, the ``load_verified_floor`` result the strata came
    from) sources the ``min_n`` default from the frozen ``methodology.min_n``;
    an explicit mismatch is warned, never silent. Bare calls (no artifact, no
    min_n) keep the legacy default 50."""
    if artifact is not None and not isinstance(artifact, VerifiedFloorArtifact):
        raise FloorArtifactError(
            "Lane M's artifact keyword takes a VerifiedFloorArtifact (the "
            f"load_verified_floor result); refusing an unverified "
            f"{type(artifact).__name__}."
        )
    min_n = _resolve_min_n(min_n, artifact, lane="Lane M")
    per_stratum: list[tuple[str, tuple, float, float]] = []
    n_skipped = 0
    for metric, stratum in discriminating_strata:
        key = (metric, stratum)
        gen = gen_d_features.get(key, [])
        real_d = real_d_features.get(key, [])
        real_t = real_t_features.get(key, [])
        if len(gen) < min_n or len(real_d) < min_n or len(real_t) < min_n:
            n_skipped += 1
            continue
        per_stratum.append((metric, stratum, ks_distance(gen, real_d), ks_distance(gen, real_t)))

    if not per_stratum:
        raise ValueError(
            f"Lane M: zero scoreable discriminating strata at min_n={min_n} "
            f"({n_skipped} skipped thin out of {len(list(discriminating_strata))}) "
            "— a vacuous memorization verdict guards nothing; refusing."
        )
    median_d = statistics.median(ks_d for _, _, ks_d, _ in per_stratum)
    median_t = statistics.median(ks_t for _, _, _, ks_t in per_stratum)
    return LaneMResult(
        verdict="PASS" if median_d < median_t else "FAIL",
        median_ks_gen_d=median_d,
        median_ks_gen_t=median_t,
        margin=median_t - median_d,
        n_strata_scored=len(per_stratum),
        n_strata_skipped_thin=n_skipped,
        per_stratum=tuple(per_stratum),
    )
