"""G's eval-set-size procedure: per-stratum floors -> (N, selection) -> degradation.

N is determined by floors and ceilings; the binding ones are PER-STRATUM, not
whole-set (spec §G). A whole-set power calc masks underpowered strata - the
vacuous pass at the sizing layer - so feasibility() always reports per-stratum
infeasibility even when the whole-set N looks sufficient (threshold-pairing,
protocol v2 §2).

The regime-distinguishing threshold is RELATIVE-to-base-rate with an absolute floor
(spec §6 + the 2026-06-01 δ review): a model is over-emitting iff its per-stratum
bref-rate exceeds the faithful rate by more than ``over_emission_threshold`` =
``max(RHO_BREF_REGIME * faithful_rate, DELTA_FLOOR_BREF)``. This is ONE policy (D's
faithful-vs-over-emitting boundary AND G's relaxation bound, spec §6 - do NOT carry
two). An ABSOLUTE δ was rejected: against the measured per-stratum faithful rates
(2.3-6.8%) an absolute 0.03 gave +44%..+129% relative tolerance, so the dense-bucket
guard was vacuous (a model could >2x the dense-bucket rate and pass). Relative rho
makes the discrimination uniform across strata.

Graceful degradation is ORDERED (spec §G): coarsen strata -> report UNDERPOWERED
-> relax rho ONLY within the regime-distinguishing bound. Relaxing past the bound is
weakening-the-assertion-to-pass (a halt-on-validator-fail violation in a sizing
costume) -> the honest output is UNDERPOWERED, not "passed at relaxed rho".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from cfm.eval.feature_resolution import KS_C_ALPHA_05

#: z for a two-sided 95% interval (one-source for both floors below).
_Z_0975: float = 1.95996

# DECISION: rho = 0.5 (relative over-emission boundary). A model is over-emitting iff
# its per-stratum bref-rate exceeds the faithful (round-tripped-real) rate by more
# than 50% relative. RELATIVE (not absolute) so the discrimination is UNIFORM across
# strata: the prior absolute δ=0.03 gave +44%/+79%/+105%/+129% tolerance for buckets
# 0/1/2/3 (faithful 6.79/3.82/2.85/2.33%) - the dense-bucket guard was vacuous (a >2x
# rate passed). rho=0.5 trips the dense-bucket DOUBLING that exposed the bug. Powered:
# detecting a 50% relative excess needs 53-646 features/stratum (z=1.96); the held-out
# set has tens of thousands per stratum (>100x margin), and rho=0.5 is ~12x the
# worst-case held-out sampling-SE so it never fires on noise. Because feature power is
# abundant, rho does NOT move N; it is tunable toward the data-supported ~0.25 once the
# model's natural over-emission variation is observed (model side deferred, spec §7).
RHO_BREF_REGIME: float = 0.5

# DECISION: δ_floor = 0.005 (0.5pp). Absolute backstop for GENUINELY near-zero strata
# only. Verified constraint: δ_floor < rho·faithful for every current bucket (min
# rho·faithful = 0.5*0.0233 = 0.01165 > 0.005), so the RELATIVE term governs exactly
# where it must - including the dense bucket - and δ_floor binds only when faithful <
# δ_floor/rho = 1%. 0.5pp is the rate-resolution floor: below it, per-stratum rate
# differences sit at/under sampling resolution for realistic held-out sizes.
DELTA_FLOOR_BREF: float = 0.005


def over_emission_threshold(faithful_rate: float) -> float:
    """The per-stratum rate-excess above which a model is over-emitting (spec §D).

    Relative-to-base-rate with an absolute floor: ``max(rho·faithful, δ_floor)``. The
    relative term gives a uniform discrimination across strata; the floor backstops
    near-zero strata so the guard is never absurdly tight there.
    """
    return max(RHO_BREF_REGIME * faithful_rate, DELTA_FLOOR_BREF)


class DegradationStep(Enum):
    COARSEN_STRATA = 1
    REPORT_UNDERPOWERED = 2
    RELAX_RHO_WITHIN_BOUND = 3


def rate_detection_floor(*, p: float, delta: float) -> int:
    """Min samples to detect a rate excess δ around base rate p: n ~ z^2 p(1-p)/δ^2."""
    if not (0.0 <= p <= 1.0) or delta <= 0.0:
        raise ValueError("require 0<=p<=1 and delta>0")
    return math.ceil((_Z_0975**2) * p * (1.0 - p) / (delta**2))


def ks_two_sample_floor(*, effect: float, alpha: float = 0.05) -> int:
    """v1 KS two-sample sample-size approximation (equal n).

    The alpha-critical statistic for equal n is c(alpha)*sqrt(2/n); c(0.05) is
    one-sourced at feature_resolution.KS_C_ALPHA_05;
    to resolve a true distributional gap ``effect``, n ~ ceil(2*(c(alpha)/effect)^2).
    This is a sizing FLOOR only; the KS/Wasserstein DISTANCE against model output is
    deferred (spec §7). v1 supports alpha=0.05 only (DECISION: revisit if another alpha
    is needed or if a stratum's effect-size assumption is contradicted by slow-run data).
    """
    if effect <= 0.0:
        raise ValueError("effect must be > 0")
    if abs(alpha - 0.05) > 1e-9:
        raise ValueError("v1 KS floor supports alpha=0.05 only")
    return math.ceil(2.0 * (KS_C_ALPHA_05 / effect) ** 2)


def relaxed_rho_is_legitimate(*, relaxed: float) -> bool:
    """A relaxed rho is legitimate iff it still separates faithful-from-over-emitting,
    i.e. it stays at or below the regime-distinguishing bound (spec §G option 3). A
    larger rho loosens the guard; relaxing past RHO_BREF_REGIME is weakening-to-pass."""
    return 0.0 < relaxed <= RHO_BREF_REGIME


@dataclass(frozen=True)
class FeasibilityReport:
    whole_set_ok: bool
    infeasible_strata: dict[int, int]  # stratum -> shortfall (floor - population)


def feasibility(populations: dict[int, int], floors: dict[int, int]) -> FeasibilityReport:
    """Per-stratum feasibility. whole_set_ok is reported ONLY to expose the masking
    risk; the verdict is the per-stratum infeasible set (threshold-pairing §2)."""
    whole_set_ok = sum(populations.values()) >= max(floors.values(), default=0)
    infeasible = {
        s: floors[s] - populations.get(s, 0) for s in floors if populations.get(s, 0) < floors[s]
    }
    return FeasibilityReport(whole_set_ok=whole_set_ok, infeasible_strata=infeasible)


@dataclass
class SizingResult:
    n: int
    per_stratum_floor: dict[int, int]
    per_stratum_population: dict[int, int]
    underpowered_strata: list[int] = field(default_factory=list)
    degradation_log: list[str] = field(default_factory=list)
