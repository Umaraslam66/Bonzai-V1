"""Conditioning-floor artifact: pure logic + artifact IO (Task 25 step 1; spec §8).

The re-scoped Phase-2 eval claim: *given conditioning for geometry it did not
train on, the model produces plausible geometry matching that character — by
learned grammar, not memorization.* This module is its measurement instrument:

- **Pair table** — per (metric, stratum) with >= 2 qualifying cities (the same
  ``min_n`` qualify rule as ``conditioning_discrimination_verdict``), every
  unordered real-real city-pair's KS distance with a globally BH-adjusted
  p-value riding along (the discriminating-strata rule needs significance).
- **Floors (PI knob 1, STRICT)** — per held-out city D and (metric, stratum):
  ``floor_D = min over OTHER cities T of KS(real_D, real_T)``; the median over
  T is reported as ``floor_median_context`` ONLY, never the floor.
- **Integrity halts (PI knob 4)** — ``FloorCollapseError`` if the pair table's
  median KS < 0.049 (broken extraction: contradicts every prior run) and
  ``FloorExplosionError`` if it exceeds 0.5 (conditioning carries nothing).
  Both fire in the payload producer, BEFORE any artifact byte is written.
- **Lane S** — per qualifying (metric, stratum):
  ``excess = max(0, KS(gen_D, real_D) - floor_D)``; per-city aggregate is the
  median + p90 over strata (PI knob 3). Scoring REFUSES to run unless the floor
  artifact loads verified (Task-20 reader-side discipline).
- **Lane M** — the memorization discriminator: over strata where D is
  measured-distinct from training city T (real-real KS >= delta AND
  BH-significant, selected from REAL data only), require
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
from dataclasses import dataclass, field
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

#: Reporting-ladder anchors (delta=0.15 is ALSO the discriminating-strata rule).
DELTA_LADDER_ANCHORS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

FLOOR_ARTIFACT_SCHEMA_VERSION: str = "1.0"
FLOOR_ARTIFACT_LOCK_NAME: str = "_CONDITIONING_FLOOR_LOCKED"


class FloorArtifactError(RuntimeError):
    """The floor artifact failed verification (absent / unsealed / tampered /
    version-skewed / malformed) or its content cannot serve the request."""


class FloorCollapseError(RuntimeError):
    """Pair-table median KS < 0.049: extraction is broken (contradicts every
    prior run); the artifact must not be written."""


class FloorExplosionError(RuntimeError):
    """Pair-table median KS > 0.5: conditioning carries nothing; the artifact
    must not be written."""


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
    p_bh: float  # globally BH-adjusted across ALL pairs in the table


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


def compute_pair_table(
    features: dict[tuple[str, tuple, str], list[float]],
    *,
    min_n: int = 50,
    alpha: float = 0.05,
) -> PairTable:
    """Every unordered city-pair KS per (metric, stratum) with >= 2 qualifying
    cities; the qualify rule (>= ``min_n`` samples) is the SAME as
    ``conditioning_discrimination_verdict``. BH-adjusted p-values are computed
    JOINTLY across all pairs (both metrics, all strata). Iteration order is
    sorted, so the table — and the artifact sha downstream — is insertion-order
    independent (PYTHONHASHSEED-proof). Parity with the verdict fn's qualify ->
    group -> pair -> global-BH steps is PINNED by
    ``test_pair_table_parity_with_the_verdict_fn`` (external source of truth)."""
    qualified = {key: vals for key, vals in features.items() if len(vals) >= min_n}
    n_excluded_thin = len(features) - len(qualified)

    by_metric_stratum: dict[tuple[str, tuple], dict[str, list[float]]] = {}
    for (city, stratum, metric), vals in qualified.items():
        by_metric_stratum.setdefault((metric, stratum), {})[city] = vals

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
                a, b = city_samples[ca], city_samples[cb]
                d = ks_distance(a, b)
                raw.append(
                    FloorPair(
                        metric=metric,
                        stratum=stratum,
                        city_a=ca,
                        city_b=cb,
                        n_a=len(a),
                        n_b=len(b),
                        ks=d,
                        noise_floor=noise_floor(len(a), len(b)),
                        p_raw=ks_pvalue(d, len(a), len(b)),
                        p_bh=1.0,  # placeholder; filled by the global BH below
                    )
                )

    p_bh = benjamini_hochberg([p.p_raw for p in raw])
    pairs = tuple(
        FloorPair(
            metric=p.metric,
            stratum=p.stratum,
            city_a=p.city_a,
            city_b=p.city_b,
            n_a=p.n_a,
            n_b=p.n_b,
            ks=p.ks,
            noise_floor=p.noise_floor,
            p_raw=p.p_raw,
            p_bh=adj,
        )
        for p, adj in zip(raw, p_bh, strict=True)
    )
    return PairTable(
        pairs=pairs,
        min_n=min_n,
        alpha=alpha,
        n_excluded_thin=n_excluded_thin,
        n_strata_too_few_cities=n_strata_too_few_cities,
    )


# --------------------------------------------------------------------------- #
# Floors (PI knob 1: STRICT min over other cities; median is context ONLY)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FloorEntry:
    """floor = STRICT min over other cities T of KS(D, T); median is context."""

    floor: float
    floor_median_context: float
    n_other_cities: int


def compute_floors(
    pair_table: PairTable,
    held_out_cities: Sequence[str],
) -> dict[str, dict[tuple[str, tuple], FloorEntry]]:
    """Per held-out city D, per (metric, stratum): the strict-min floor over
    every OTHER city D is paired with in the table — other held-out cities AND
    training cities alike, whenever their pairs exist."""
    held = set(held_out_cities)
    ks_by_city_key: dict[tuple[str, str, tuple], list[float]] = {}
    for p in pair_table.pairs:
        for d, _other in ((p.city_a, p.city_b), (p.city_b, p.city_a)):
            if d in held:
                ks_by_city_key.setdefault((d, p.metric, p.stratum), []).append(p.ks)

    floors: dict[str, dict[tuple[str, tuple], FloorEntry]] = {}
    for (d, metric, stratum), ks_values in ks_by_city_key.items():
        floors.setdefault(d, {})[(metric, stratum)] = FloorEntry(
            floor=min(ks_values),
            floor_median_context=statistics.median(ks_values),
            n_other_cities=len(ks_values),
        )
    return floors


# --------------------------------------------------------------------------- #
# Integrity halts (PI knob 4; fire BEFORE any artifact byte)
# --------------------------------------------------------------------------- #


def assert_floor_sanity(pair_table: PairTable) -> None:
    """Raise the regime-matching halt; pass silently on a healthy table.

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

    NO-LEAKAGE BY SIGNATURE: this takes ONLY the real-real pair table —
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


def build_floor_artifact_payload(
    pair_table: PairTable,
    *,
    release: str,
    held_out_cities: Sequence[str],
    train_cities: Sequence[str] = (),
    delta: float = 0.15,
    tile_coverage: dict[str, TileCoverage] | None = None,
) -> dict:
    """The YAML-safe artifact payload. Runs ``assert_floor_sanity`` FIRST so a
    collapse/explosion/UNSUPPORTED regime halts before any artifact content
    exists. Stores KS tables, never raw samples (see module docstring).

    MISSING-CITY HALT (Task-25 spec review #3): every held-out city must come
    out of ``compute_floors`` floored, or this raises BEFORE any artifact byte.
    Rationale: a held-out city silently absent from ``floors`` shrinks the
    worst-case max domain weeks later at Lane-S consumption against a
    write-once artifact — the aggregate-hides-subsets class, the same failure
    the global zero-pairs UNSUPPORTED halt already guards."""
    assert_floor_sanity(pair_table)
    floors = compute_floors(pair_table, held_out_cities)
    missing = sorted(set(held_out_cities) - set(floors))
    if missing:
        raise ValueError(
            "conditioning-floor: UNSUPPORTED — held-out cities with ZERO "
            f"qualifying pairs in the table (no floor derivable): {missing} "
            f"at min_n={pair_table.min_n}. A write-once artifact missing a "
            "held-out city silently shrinks the Lane-S worst-case max domain; "
            "refusing to freeze."
        )
    strata = select_discriminating_strata(pair_table, delta=delta)

    pair_records = [
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
        for p in pair_table.pairs
    ]
    floor_records = [
        {
            "city": city,
            "metric": metric,
            "stratum": list(stratum),
            "floor": float(entry.floor),
            "floor_median_context": float(entry.floor_median_context),
            "n_other_cities": entry.n_other_cities,
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
            "floor_rule": "strict_min_over_other_cities",  # PI knob 1
            "floor_context_rule": "median_over_other_cities_context_only",
            "lane_s_aggregate_rule": "median_p90_over_strata",  # PI knob 3
            "lane_m_scope": "all_training_cities",  # PI knob 2
        },
        "held_out_cities": sorted(held_out_cities),
        "train_cities": sorted(train_cities),
        "n_excluded_thin": pair_table.n_excluded_thin,
        "n_strata_too_few_cities": pair_table.n_strata_too_few_cities,
        "pair_table_median_ks": float(pair_table.median_ks()),
        "pair_table": pair_records,
        "floors": floor_records,
        "discriminating_strata": strata_records,
        "delta_ladder": _delta_ladder(pair_table),
        "tile_coverage": coverage_records,
    }


def pair_table_from_payload(payload: dict) -> PairTable:
    """Rebuild the PairTable from an artifact payload (strata back to tuples)."""
    pairs = tuple(
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
        for rec in payload["pair_table"]
    )
    return PairTable(
        pairs=pairs,
        min_n=int(payload["methodology"]["min_n"]),
        alpha=float(payload["methodology"]["alpha"]),
        n_excluded_thin=int(payload["n_excluded_thin"]),
        n_strata_too_few_cities=int(payload["n_strata_too_few_cities"]),
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
    """``excess = max(0, KS(gen, real_D) - floor_D)`` per qualifying (metric,
    stratum); aggregate = median + p90 over strata (PI knob 3).

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
        (rec["metric"], tuple(rec["stratum"])): float(rec["floor"])
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
