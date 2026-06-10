"""Pre-committed bake-off ladder + decision-basis rules (delta-spec §2; T3).

Rule 1 (`feasible_ladder`): scale N is on-frontier-feasible iff r*N <= train_tokens*E.
Failing scales are DROPPED, never run data-limited. Conservative boundary rounding uses
the UPPER r-CI bound (higher effective r -> fewer rungs).

Rule 2 (`decision_basis`): the number of feasible rungs selects the decision basis.
<3 points -> curve REPORTED, never decision-bearing (decide at the top feasible scale +
§13). >=3 points -> falsifiable scaling curve (>=1 degree of freedom).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Authoritative frozen EU train-token count (_EVAL_SET_LOCKED, release 2026-04-15.0).
TRAIN_TOKENS: int = 623_900_790
#: The PRD/baseline candidate scales, in params.
LADDER_SCALES_PARAMS: tuple[int, ...] = (
    30_000_000,
    100_000_000,
    300_000_000,
    1_000_000_000,
)


@dataclass(frozen=True)
class LadderDecision:
    """Outcome of the ladder feasibility rule (delta-spec §2).

    Note: ``escalate_more_data`` is equivalent to
    ``decision_basis(len(feasible)) is DecisionBasis.ESCALATE_MORE_DATA`` -- both derive from
    ``feasible`` being empty. A caller MUST NOT branch on ``escalate_more_data`` while computing
    the decision basis from a different count; read both from this same ``feasible`` tuple.
    """

    feasible: tuple[int, ...]
    dropped: tuple[int, ...]
    escalate_more_data: bool  # True iff feasible is empty (the empty-ladder case)


def feasible_ladder(
    r: float,
    *,
    epoch_factor: float = 1.0,
    train_tokens: int = TRAIN_TOKENS,
    scales: tuple[int, ...] = LADDER_SCALES_PARAMS,
) -> LadderDecision:
    """Rule 1 (delta-spec §2): a scale N is on-frontier-feasible iff r*N <= budget,
    where budget = train_tokens * epoch_factor. Scales that fail are dropped, never
    run data-limited; an empty feasible set sets ``escalate_more_data``.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    budget = train_tokens * epoch_factor
    feasible = tuple(n for n in scales if r * n <= budget)
    dropped = tuple(n for n in scales if n not in feasible)
    return LadderDecision(
        feasible=feasible,
        dropped=dropped,
        escalate_more_data=not feasible,
    )


def feasible_ladder_conservative(
    r_ci_high: float,
    *,
    epoch_factor: float = 1.0,
    train_tokens: int = TRAIN_TOKENS,
    scales: tuple[int, ...] = LADDER_SCALES_PARAMS,
) -> LadderDecision:
    """Boundary-straddle rule: size by the UPPER r-CI bound so we never add a rung the
    data can't clearly support. (Mirrors ``feasible_ladder``'s signature so the conservative
    entry point — the one Task 10 calls — keeps full kwarg type-checking.)"""
    return feasible_ladder(
        r_ci_high,
        epoch_factor=epoch_factor,
        train_tokens=train_tokens,
        scales=scales,
    )


class DecisionBasis(Enum):
    # 0 feasible points
    ESCALATE_MORE_DATA = "escalate_more_data"
    # 1-2 points: curve REPORTED, never decision-bearing
    FIXED_SCALE_PLUS_S13 = "fixed_scale_plus_s13"
    # 3+ points: falsifiable curve (>=1 DoF)
    SCALING_CURVE = "scaling_curve"


def decision_basis(n_feasible: int) -> DecisionBasis:
    """Rule 2 (delta-spec §2). <3 points -> curve reported, never decision-bearing:
    decide at the top feasible scale + §13. >=3 -> falsifiable curve (lever-arm
    sanity-checked)."""
    if n_feasible <= 0:
        return DecisionBasis.ESCALATE_MORE_DATA
    if n_feasible < 3:
        return DecisionBasis.FIXED_SCALE_PLUS_S13
    return DecisionBasis.SCALING_CURVE
