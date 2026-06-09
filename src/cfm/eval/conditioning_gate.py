"""The §4 conditioning-discrimination gate (delta-spec T5).

Worst-case (or mean) per-city scoring is coherent IFF per-city KS tracks macro-plan
differences, not residual un-conditioned city-style. Operationalized on the Task-1
diagnostic: for tiles at the SAME macro-stratum, the cross-city feature-distribution KS
must sit within tolerance (the per-city KS noise floor). If it exceeds tolerance, a per-city
miss is ambiguous ("wasn't told the city") and T5 REOPENS before any scored run.

The gate carries NO baked threshold: ``tolerance`` is a required keyword-only parameter
with no default, injected by the Task-9 diagnostic (derived from the measured per-city n).
The module imports nothing beyond ``dataclass`` so no source constant is silently wired in
as the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passes: bool
    reason: str


def conditioning_discrimination_gate(
    same_stratum_cross_city_ks: dict[str, float],
    *,
    tolerance: float,
) -> GateResult:
    """Discharge the conditioning-discrimination gate.

    ``same_stratum_cross_city_ks`` maps a city-pair label to the cross-city
    feature-distribution KS of held-out tiles drawn from the SAME macro-stratum.
    ``tolerance`` is the per-city KS noise floor, injected by the Task-9 diagnostic.

    PASS iff the worst (largest) same-stratum cross-city KS is within tolerance:
    conditioning explains the per-city variation, so the worst-case bar is valid.
    FIRE/HALT (and T5 REOPENS) if it exceeds tolerance, or fail-closed if no KS supplied.
    """
    if not same_stratum_cross_city_ks:
        return GateResult(
            False,
            "no same-stratum cross-city KS supplied; cannot discharge gate (fail-closed)",
        )
    worst = max(same_stratum_cross_city_ks.values())
    if worst <= tolerance:
        return GateResult(
            True,
            f"same-stratum cross-city KS {worst:.3f} <= {tolerance}: "
            "conditioning explains per-city variation",
        )
    return GateResult(
        False,
        f"same-stratum cross-city KS {worst:.3f} > {tolerance}: "
        "residual city-style — T5 REOPENS before any scored run",
    )
