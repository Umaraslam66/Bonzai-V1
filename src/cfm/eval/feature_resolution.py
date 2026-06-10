"""Per-feature resolution + the winner-vs-runner-up decision seam (Phase-2 bake-off Task 3).

Spec §8 / §10.3: the bake-off ranks backbones on a per-FEATURE geometry-realism KS
(building-area, road-length), so resolution must be re-derived in the per-FEATURE unit
on the real generated+holdout feature populations -- NOT the frozen per-CELL 0.076
(``cfm.eval.resolution.assert_resolution_sufficient`` stays the frozen-SET
representativeness seam, a different question; inheriting its marker here is the §8
unit trap). The seam fires on the WINNER-vs-RUNNER-UP gap only -- a tie for last place
is irrelevant to "pick the winner".

3-tier escalation (the generated eval is NOT write-once, unlike the eval-set):
  * RESOLVED            -- winner-runner-up gap clears the current per-feature binding gap.
  * GENERATE_MORE_CELLS -- gap below current binding but >= the floor: generate the
                           n_cells that resolves THIS gap ONCE (not a loop), reusing the
                           §4 locked eval content (same conditioning/seeds/holdout).
  * SECOND_REGION       -- gap below the floor: KS is two-sample and the holdout reference
                           is FIXED, so as generated->inf the gap asymptotes to
                           1.358/sqrt(n_ref); below that no generated cells help.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# KS two-sample critical coefficient at alpha=0.05 (matches holdout/sizing.py).
_KS_C_ALPHA_05 = 1.358


class DecisionUnresolvable(Exception):
    """The winner-vs-runner-up gap is finer than the per-feature binding resolution."""


class Escalation(Enum):
    RESOLVED = "resolved"
    GENERATE_MORE_CELLS = "generate_more_cells"
    SECOND_REGION = "second_region"


def per_feature_resolved_gap(*, n_features: int) -> float:
    """Finest KS gap resolvable with ``n_features`` per side (symmetric two-sample, alpha=0.05).

    Same formula family as the eval-set sizing, but in the per-FEATURE unit -- never the
    inherited per-cell 0.076.
    """
    if n_features <= 0:
        return float("inf")
    return _KS_C_ALPHA_05 * math.sqrt(2.0 / n_features)


def single_region_floor_gap(*, n_reference_features: int) -> float:
    """The finest gap achievable as the GENERATED side -> inf against a FIXED reference of
    ``n_reference_features`` (the two-sample asymptotic limit ``c/sqrt(n_ref)``).

    This is the tier-2/tier-3 boundary: below it, no number of generated cells helps,
    because the holdout reference cannot be enlarged.
    """
    if n_reference_features <= 0:
        return float("inf")
    return _KS_C_ALPHA_05 / math.sqrt(n_reference_features)


@dataclass(frozen=True)
class BindingResolution:
    binding_metric: str
    binding_gap: float
    per_metric_gap: dict[str, float]


def binding_resolution(n_features_by_metric: dict[str, int]) -> BindingResolution:
    """The WORST-resolved (coarsest-gap) feature distribution binds the seam.

    A pair resolvable on roads but not buildings must not be green-lit, so the seam
    checks against the coarsest per-feature resolution (§8 pin: worst-resolved feature).
    """
    per_metric = {
        m: per_feature_resolved_gap(n_features=n) for m, n in n_features_by_metric.items()
    }
    binding_metric = max(per_metric, key=lambda m: per_metric[m])
    return BindingResolution(binding_metric, per_metric[binding_metric], per_metric)


def _winner_runnerup_gap(ranked_scores: list[float]) -> float:
    """Gap between the best (lowest KS) and second-best. Inputs are KS distances
    (lower = better), so we sort ascending and take the first two. A single backbone
    has no runner-up -> infinite gap (trivially resolvable)."""
    if len(ranked_scores) < 2:
        return float("inf")
    s = sorted(ranked_scores)
    return s[1] - s[0]


def check_decision_resolvable(ranked_scores: list[float], binding_gap: float) -> None:
    """Raise ``DecisionUnresolvable`` iff the WINNER-vs-RUNNER-UP gap is below the
    per-feature binding resolution. Only the top-two pair gates the decision -- a tie
    for last place does not fire (it is irrelevant to "pick the winner")."""
    gap = _winner_runnerup_gap(ranked_scores)
    if gap < binding_gap:
        raise DecisionUnresolvable(
            f"winner-vs-runner-up KS gap {gap:.4f} < per-feature binding resolution "
            f"{binding_gap:.4f}: cannot rank the top two. See escalation_for() for the tier."
        )


def escalation_for(*, gap: float, binding_gap: float, floor_gap: float) -> Escalation:
    """Which tier a winner-vs-runner-up ``gap`` falls in (§8 3-tier escalation)."""
    if gap >= binding_gap:
        return Escalation.RESOLVED
    if gap >= floor_gap:
        return Escalation.GENERATE_MORE_CELLS
    return Escalation.SECOND_REGION


def cells_to_resolve(*, target_gap: float, features_per_cell: float) -> int:
    """The n_cells whose generated features resolve ``target_gap`` in ONE shot (not a loop).

    ``n_features = ceil(2 * (c / target_gap)^2)`` (invert the symmetric gap formula), then
    ``n_cells = ceil(n_features / features_per_cell)``. The tier-2 action generates exactly
    this many cells once, reusing the §4 locked eval content. Use ``escalation_for`` first:
    if the gap is below the floor this is unreachable and the answer is SECOND_REGION.
    """
    if target_gap <= 0 or features_per_cell <= 0:
        raise ValueError("target_gap and features_per_cell must be positive")
    n_features = math.ceil(2.0 * (_KS_C_ALPHA_05 / target_gap) ** 2)
    return math.ceil(n_features / features_per_cell)
