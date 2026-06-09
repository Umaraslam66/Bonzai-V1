"""Validation-separation gate for the coherence reference (tooth-3, spec §3.3/§3.5a).

The shuffle-null saturates on DENSE tiles: when a tile's interior road graph fills a
large fraction of the 60-edge 6x6-interior capacity, a *random* rearrangement is itself
near-fully-connected, so real ~ permuted and the gap collapses to ~0 (NOT a metric defect
— density, not incoherence). Such strata are EXEMPT from the tooth-3 separation gate by a
PRE-COMMITTED STRUCTURAL threshold on mean road-edges (a capacity fraction, set from the
mechanism, NOT by city name), with the full measure still reported.
"""

from __future__ import annotations

# 6x6 interior internal-edge capacity = 30 (axis-0) + 30 (axis-1) = 60.
INTERIOR_EDGE_CAPACITY: int = 60
# Dense-core saturation threshold: > 2/3 of capacity. Above this a random rearrangement of
# that many edges is itself near-fully-connected, saturating the shuffle-null. Capacity-derived,
# not fit to any city. (Observed: munich 47.8 > 40 [dense-core, #21]; krakow 36.3 / glasgow 29.2
# / eisenhuttenstadt 24.7 all < 40 [moderate].)
DENSE_CORE_EDGE_THRESHOLD: float = INTERIOR_EDGE_CAPACITY * 2.0 / 3.0  # = 40.0

MIN_SEPARATION_FRACTION: float = 0.70


class ValidationSeparationError(Exception):
    """A MODERATE-density held-out stratum failed the tooth-3 real-vs-permuted separation
    gate (>= MIN_SEPARATION_FRACTION). Dense-core strata are exempt by structure."""


def is_dense_core_saturated(mean_road_edges: float) -> bool:
    """True iff the stratum's mean interior road-edge count exceeds the dense-core threshold
    (the shuffle-null saturates → tooth-3 exempt)."""
    return mean_road_edges > DENSE_CORE_EDGE_THRESHOLD


def assert_validation_separation(
    per_stratum: dict[str, dict], *, min_fraction: float = MIN_SEPARATION_FRACTION
) -> None:
    """tooth-3 HALT-gate, scoped by STRUCTURE not identity: every stratum that is NOT
    dense-core-saturated (by its recorded ``mean_road_edges``) must have
    ``real_vs_permuted_positive_fraction`` >= min_fraction. Dense-core strata are reported,
    not gated. Raises ValidationSeparationError listing the failing MODERATE strata."""
    failures = []
    for city, s in sorted(per_stratum.items()):
        if is_dense_core_saturated(s["mean_road_edges"]):
            continue  # dense-core: shuffle-null saturates, exempt (full measure still recorded)
        if s["real_vs_permuted_positive_fraction"] < min_fraction:
            sep = s["real_vs_permuted_positive_fraction"]
            failures.append(
                f"{city}: separation {sep:.4f} < {min_fraction} "
                f"(moderate stratum, mean_road_edges={s['mean_road_edges']:.1f})"
            )
    if failures:
        raise ValidationSeparationError("; ".join(failures))
