"""G's eval-set-size procedure: per-stratum floors -> (N, selection) -> degradation.

N is determined by floors and ceilings; the binding ones are PER-STRATUM, not
whole-set (spec §G). A whole-set power calc masks underpowered strata - the
vacuous pass at the sizing layer - so feasibility() always reports per-stratum
infeasibility even when the whole-set N looks sufficient (threshold-pairing,
protocol v2 §2).

δ (DELTA_BREF_REGIME) is ONE number: D's regime-distinguishing rate-excess AND
G's δ-relaxation bound (spec §6 - do NOT carry two). Defined here; imported by
degeneracy.py. The justification below is the load-bearing rationale.

Graceful degradation is ORDERED (spec §G): coarsen strata -> report UNDERPOWERED
-> relax δ ONLY within the regime-distinguishing bound. Relaxing past the bound is
weakening-the-assertion-to-pass (a halt-on-validator-fail violation in a sizing
costume) -> the honest output is UNDERPOWERED, not "passed at relaxed δ".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

#: z for a two-sided 95% interval (one-source for both floors below).
_Z_0975: float = 1.95996

# DECISION: δ = 0.03 (3 percentage-point rate-excess). Justification - the v1
# round-tripped-real bref-placeholder rate is small (single-digit %); a model that
# "learned the limitation" reproduces it within sampling noise, while a model
# "over-emitting" degenerate stubs pushes the rate materially above it. 3pp is
# chosen as the smallest excess reliably above the per-stratum sampling noise floor
# at the achievable per-stratum N (re-confirmed against the real measurement in the
# slow run, Task 11). NOT a round default (0.05/0.10 rejected as round; rough-
# numbers heuristic). Revisit if the slow-run per-stratum noise floor exceeds 3pp.
DELTA_BREF_REGIME: float = 0.03


class DegradationStep(Enum):
    COARSEN_STRATA = 1
    REPORT_UNDERPOWERED = 2
    RELAX_DELTA_WITHIN_BOUND = 3


def rate_detection_floor(*, p: float, delta: float) -> int:
    """Min samples to detect a rate excess δ around base rate p: n ~ z^2 p(1-p)/δ^2."""
    if not (0.0 <= p <= 1.0) or delta <= 0.0:
        raise ValueError("require 0<=p<=1 and delta>0")
    return math.ceil((_Z_0975**2) * p * (1.0 - p) / (delta**2))


def ks_two_sample_floor(*, effect: float, alpha: float = 0.05) -> int:
    """v1 KS two-sample sample-size approximation (equal n).

    The alpha-critical statistic for equal n is c(alpha)*sqrt(2/n) with c(0.05)=1.358;
    to resolve a true distributional gap ``effect``, n ~ ceil(2*(c(alpha)/effect)^2).
    This is a sizing FLOOR only; the KS/Wasserstein DISTANCE against model output is
    deferred (spec §7). v1 supports alpha=0.05 only (DECISION: revisit if another alpha
    is needed or if a stratum's effect-size assumption is contradicted by slow-run data).
    """
    if effect <= 0.0:
        raise ValueError("effect must be > 0")
    if abs(alpha - 0.05) > 1e-9:
        raise ValueError("v1 KS floor supports alpha=0.05 only")
    c_alpha = 1.358  # KS two-sample critical coefficient at alpha=0.05
    return math.ceil(2.0 * (c_alpha / effect) ** 2)


def relaxed_delta_is_legitimate(*, relaxed: float) -> bool:
    """A relaxed δ is legitimate iff it still separates faithful-from-over-emitting,
    i.e. it stays at or below the regime-distinguishing bound (spec §G option 3)."""
    return 0.0 < relaxed <= DELTA_BREF_REGIME


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
