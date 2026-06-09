"""Per-city worst-case aggregation + the #21 binding-city power gate (delta-spec §4).

Generalization is a worst-case property: the decision binds on the WORST held-out city,
NEVER a mean and NEVER a pooled (cell-count-weighted) reference. munich is INCLUDED (KS has
no null to saturate); the #21 gate only DEMOTES a binding city whose winner-vs-runner-up gap
is below that city's OWN C/sqrt(n) resolution floor (under-powered, cannot decide).
"""

from __future__ import annotations

from dataclasses import dataclass

from cfm.eval.feature_resolution import single_region_floor_gap


@dataclass(frozen=True)
class PerCityKS:
    city: str
    ks: float
    n_features: int  # reference feature count for this city's binding metric


@dataclass(frozen=True)
class BindingVerdict:
    binding_city: str
    winner: str
    runner_up: str
    gap: float
    city_floor: float
    demoted_from: tuple[str, ...]


def worst_case_city(per_city: list[PerCityKS]) -> PerCityKS:
    """The worst (highest-KS) held-out city. Generalization binds on the WORST city, never
    a mean and never a cell-count-weighted pool.

    Note: ties (equal-worst cities) are broken by input order (``max`` returns the first) and
    are decision-irrelevant -- equally-worst cities bind the same KS regardless of which is
    returned."""
    if not per_city:
        raise ValueError("no per-city KS supplied")
    return max(per_city, key=lambda c: c.ks)


def binding_city_verdict(
    per_backbone_per_city: dict[str, list[PerCityKS]],
) -> BindingVerdict:
    """Worst-case decision with the #21 power gate. Cities are considered worst-first; a city
    whose winner-vs-runner-up KS gap < its own C/sqrt(n) floor is DEMOTED (under-powered); the
    decision uses the first city that is both binding and resolved.

    Precondition: every backbone must cover the SAME set of held-out cities. A city present in
    one backbone but missing from another is rejected with a ``ValueError`` (not a bare
    ``StopIteration`` from the ``_ks``/``_n`` lookups, and never silently dropped — a silent drop
    of a city present only in a non-first backbone could skip the real worst city and defeat the
    worst-case bar). This also rejects an empty backbone mapping.
    """
    backbones = list(per_backbone_per_city)
    if not backbones:
        raise ValueError("binding_city_verdict: no backbones supplied")
    cities = {c.city for c in per_backbone_per_city[backbones[0]]}
    for b in backbones:
        bc = {c.city for c in per_backbone_per_city[b]}
        if bc != cities:
            raise ValueError(
                f"binding_city_verdict: backbone {b!r} covers cities {sorted(bc)}, "
                f"expected {sorted(cities)} (from {backbones[0]!r}); all backbones must "
                "cover the same held-out cities"
            )
    cities = [c.city for c in per_backbone_per_city[backbones[0]]]

    def city_mean(city: str) -> float:
        # mean KS per city across backbones, only to ORDER cities worst-first (never to decide).
        return sum(_ks(per_backbone_per_city[b], city) for b in backbones) / len(backbones)

    demoted: list[str] = []
    for city in sorted(cities, key=city_mean, reverse=True):
        ranked = sorted(backbones, key=lambda b, c=city: _ks(per_backbone_per_city[b], c))
        winner, runner_up = ranked[0], ranked[1]
        gap = _ks(per_backbone_per_city[runner_up], city) - _ks(per_backbone_per_city[winner], city)
        floor = single_region_floor_gap(
            n_reference_features=_n(per_backbone_per_city[winner], city)
        )
        if gap > floor:
            return BindingVerdict(city, winner, runner_up, gap, floor, tuple(demoted))
        demoted.append(city)
    raise ValueError(f"no resolved binding city; all under-powered: {demoted}")


def _ks(per_city: list[PerCityKS], city: str) -> float:
    return next(c.ks for c in per_city if c.city == city)


def _n(per_city: list[PerCityKS], city: str) -> int:
    return next(c.n_features for c in per_city if c.city == city)
