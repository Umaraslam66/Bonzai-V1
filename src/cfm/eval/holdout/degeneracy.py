"""D's known-limitation stance: per-instance exclude + distribution-level judge.

Per-instance (spec §D): the bref-placeholder shape is EXCLUDED via the §2 shared
predicate (construction identity) so a faithful model is not penalized. A
degenerate block WITHOUT an outbound bref is a genuine model defect -> MODEL_INVALID.

Distribution-level (spec §D = R2): REPORT the bref-placeholder RATE on model output
vs round-tripped-real; the excess is a reported model-degeneracy term. Per-instance
cannot separate "faithful reproduction" from "over-emission"; the rate can.

Guards are the §9 spine, regime-distinguishing:
- G-D1: re-proven on a MODEL-EMITTED fixture (test), not inherited from sub-G's
  real-data drill - coverage on real data does not carry to the model regime.
- G-D2: at-threshold (just-over trips, just-under passes) AND stratified (a global
  match with a per-stratum divergence must trip). Strata = cell_density_bucket.

delta is the single DELTA_BREF_REGIME imported from sizing.py (one number).

DEFERRED (spec §7): producing the model token stream (the tokenizer-on-MODEL side
of R2) needs a trained model - that runs in the eval-harness successor. This module
ships the classifier, the rate-comparison, and the guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cfm.eval.holdout.bref_rate import (
    BrefRateResult,
    _bref_predicate,
    bref_placeholder_rate,
)
from cfm.eval.holdout.sizing import DELTA_BREF_REGIME

# Re-export so callers read the rate through this module's R2 surface too.
__all__ = [
    "BrefRateResult",
    "RateVerdict",
    "Verdict",
    "bref_placeholder_rate",
    "classify_block",
    "over_emission_verdict",
    "stratified_over_emission",
]


class Verdict(Enum):
    EXCLUDED_BREF_PLACEHOLDER = 1  # faithful v1-limitation reproduction - not penalized
    MODEL_INVALID = 2  # genuine degeneracy (no outbound bref) - counts against
    VALID = 3


def classify_block(block: list[int], geom: dict) -> Verdict:
    """Per-instance verdict via the shared construction-identity predicate.

    Degeneracy is detected exactly as sub-G's gate does: an OGC-invalid
    LineString/Polygon (verified - a zero-length LineString is is_valid=False).
    """
    from shapely.geometry import shape

    if _bref_predicate(block, geom):
        return Verdict.EXCLUDED_BREF_PLACEHOLDER
    if geom.get("type") in ("LineString", "Polygon") and not shape(geom).is_valid:
        return Verdict.MODEL_INVALID
    return Verdict.VALID


class RateVerdict(Enum):
    WITHIN_TOLERANCE = 1
    OVER_EMITTING = 2


def over_emission_verdict(*, model_rate: float, faithful_rate: float) -> RateVerdict:
    """At-threshold rate judge: excess > delta => over-emitting (spec §D G-D2)."""
    return (
        RateVerdict.OVER_EMITTING
        if (model_rate - faithful_rate) > DELTA_BREF_REGIME
        else RateVerdict.WITHIN_TOLERANCE
    )


@dataclass(frozen=True)
class StratifiedOverEmissionReport:
    global_within_tolerance: bool
    over_emitting_strata: dict[int, float]  # stratum -> model rate excess
    per_stratum_verdict: dict[int, RateVerdict] = field(default_factory=dict)


def stratified_over_emission(
    model_rate: BrefRateResult, *, faithful_rate: dict[int, float]
) -> StratifiedOverEmissionReport:
    """Stratified rate judge: trips on ANY diverging stratum even if the global rate
    matches (the distributional vacuous pass). Strata = cell_density_bucket."""
    over: dict[int, float] = {}
    verdicts: dict[int, RateVerdict] = {}
    for stratum, sr in model_rate.per_stratum.items():
        r0 = faithful_rate.get(stratum, 0.0)
        v = over_emission_verdict(model_rate=sr.rate, faithful_rate=r0)
        verdicts[stratum] = v
        if v is RateVerdict.OVER_EMITTING:
            over[stratum] = sr.rate - r0

    denom = sum(sr.n_total for sr in model_rate.per_stratum.values())
    global_faithful = (
        sum(faithful_rate.get(s, 0.0) * sr.n_total for s, sr in model_rate.per_stratum.items())
        / denom
        if denom
        else 0.0
    )
    global_within = (model_rate.overall_rate - global_faithful) <= DELTA_BREF_REGIME

    return StratifiedOverEmissionReport(
        global_within_tolerance=global_within,
        over_emitting_strata=over,
        per_stratum_verdict=verdicts,
    )
