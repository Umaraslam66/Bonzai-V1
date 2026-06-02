"""Comparability-deviation log + the fails-to-train detector (Phase-2 bake-off Task 9).

The bake-off holds ONE optimizer recipe across all runs (§10). A per-architecture
deviation is allowed ONLY under three constraints, encoded here:

1. The trigger is a demonstrable TRAIN FAILURE (diverge / NaN / flatline-from-start),
   never "scores lower" -- you may fix a backbone that CAN'T train; you may not tune one
   that trains-but-loses (that is the confound the bake-off rejects).
2. The deviation is decided on the diagnostic, BEFORE the scored runs (recorded here so
   it is not a reaction to a ranking already seen).
3. It applies a NAMED principled rule uniformly (e.g. loss-scale-normalized lr), never a
   bespoke per-backbone number (a bespoke number is per-run-tuning in disguise).

``record`` enforces (1) and (3) structurally; (2) is enforced by WHEN it is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class DeviationError(Exception):
    """An attempted recipe deviation violates the principled-deviation constraints."""


def is_train_failure(loss_history: list[float]) -> bool:
    """True iff the loss history shows a demonstrable train FAILURE (not just a low rank).

    Failure = any NaN/Inf, OR the loss ended materially higher than it started
    (diverging), OR it never moved at all (flatline-from-start). A loss that descends
    and then plateaus is CONVERGED, not a failure.
    """
    if not loss_history:
        return False
    if any(math.isnan(x) or math.isinf(x) for x in loss_history):
        return True
    if len(loss_history) < 2:
        return False
    if loss_history[-1] > loss_history[0]:  # diverging
        return True
    if max(loss_history) - min(loss_history) == 0.0:  # flatline from the first step
        return True
    return False


@dataclass
class DeviationLog:
    """Append-only record of principled recipe deviations, written into the run report."""

    entries: list[dict] = field(default_factory=list)

    def record(self, *, backbone: str, scale: str, rule: str | None, trigger: str) -> None:
        if trigger == "scores-lower":
            raise DeviationError(
                "deviation trigger must be a demonstrable train FAILURE "
                "(diverge/NaN/flatline), never 'scores-lower' -- that is the per-run-tuning "
                "confound the bake-off rejects"
            )
        if not rule:
            raise DeviationError(
                "a deviation must name a principled rule applied uniformly across backbones "
                "(e.g. 'loss-scale-normalized-lr'), never a bespoke per-backbone number"
            )
        self.entries.append(
            {"backbone": backbone, "scale": scale, "rule": rule, "trigger": trigger}
        )
