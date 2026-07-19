"""Per-city worst-case aggregation + the #21 binding-city power gate (delta-spec §4).

Generalization is a worst-case property: the decision binds on the WORST held-out city,
NEVER a mean and NEVER a pooled (cell-count-weighted) reference. munich is INCLUDED (KS has
no null to saturate); the #21 gate only DEMOTES a binding city whose winner-vs-runner-up gap
is below that city's OWN C/sqrt(n) resolution floor (under-powered, cannot decide).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfm.eval.feature_resolution import single_region_floor_gap
from cfm.eval.ladder import DecisionBasis


@dataclass(frozen=True)
class PerCityKS:
    city: str
    ks: float  # the point estimate (mean KS across seeds, for the scored 3-seed runs)
    n_features: int  # reference feature count for this city's binding metric
    #: seed-noise band = std/SEM of this backbone's per-seed KS at this city (3-seed runs).
    #: 0.0 (default) = single-seed / legacy: the effective floor collapses to C/sqrt(n).
    seed_sem: float = 0.0


@dataclass(frozen=True)
class BindingVerdict:
    binding_city: str
    winner: str
    runner_up: str
    gap: float
    #: the EFFECTIVE floor the gap cleared = max(resolution_floor, seed_noise_floor).
    city_floor: float
    demoted_from: tuple[str, ...]
    #: the two component floors (for reporting which bound; default 0.0 keeps legacy
    #: positional construction valid).
    resolution_floor: float = 0.0
    seed_noise_floor: float = 0.0


@dataclass(frozen=True)
class NoDecisiveWinner:
    """S13-family verdict: NO held-out city separated the backbones beyond BOTH the C/sqrt(n)
    resolution floor (statistical resolvability) AND the seed-noise band (run-to-run
    reproducibility). On param-matched near-ties this is the LIKELY outcome, not an edge case
    — so it is a named verdict, never a bare exception: the decision routes to the spec §13
    pre-committed simplest-backbone tie-break, never to improvisation in a later session.

    The per-city dicts record, for every (demoted) city, the winner-vs-runner-up ``gap`` and
    the two floors, so a reader can see WHICH floor bound: ``seed_noise_floor`` >
    ``resolution_floor`` => reproducibility blocked (LUCK); the reverse => resolvability
    blocked (MIDDLE)."""

    demoted: tuple[str, ...]  # all cities, worst-first (none decisive)
    gap: dict[str, float]
    resolution_floor: dict[str, float]
    seed_noise_floor: dict[str, float]
    basis: DecisionBasis = field(default=DecisionBasis.FIXED_SCALE_PLUS_S13)


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
) -> BindingVerdict | NoDecisiveWinner:
    """Worst-case decision with the #21 power gate + the two-floor seed→verdict rule. Cities are
    considered worst-first; a city is DECISIVE only if its winner-vs-runner-up KS gap clears
    ``effective_floor = max(C/sqrt(n) resolvability, seed-noise reproducibility)``. A non-decisive
    worst city is DEMOTED and the next-worst tried; if NO city is decisive the result is a
    ``NoDecisiveWinner`` (S13 family), never a raise.

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
    gaps: dict[str, float] = {}
    res_floors: dict[str, float] = {}
    seed_floors: dict[str, float] = {}
    for city in sorted(cities, key=city_mean, reverse=True):
        ranked = sorted(backbones, key=lambda b, c=city: _ks(per_backbone_per_city[b], c))
        winner, runner_up = ranked[0], ranked[1]
        gap = _ks(per_backbone_per_city[runner_up], city) - _ks(per_backbone_per_city[winner], city)
        # Two independent floors: C/sqrt(n) statistical resolvability, and the seed-noise band
        # (the larger of the winner's and runner-up's per-seed SEM at this city). The gap must
        # clear the MAX of the two — clearing only one (the likely near-tie MIDDLE band) is NOT
        # a win.
        resolution_floor = single_region_floor_gap(
            n_reference_features=_n(per_backbone_per_city[winner], city)
        )
        seed_noise_floor = max(
            _sem(per_backbone_per_city[winner], city),
            _sem(per_backbone_per_city[runner_up], city),
        )
        effective_floor = max(resolution_floor, seed_noise_floor)
        if gap > effective_floor:
            return BindingVerdict(
                city,
                winner,
                runner_up,
                gap,
                effective_floor,
                tuple(demoted),
                resolution_floor=resolution_floor,
                seed_noise_floor=seed_noise_floor,
            )
        demoted.append(city)
        gaps[city] = gap
        res_floors[city] = resolution_floor
        seed_floors[city] = seed_noise_floor
    # No city separated the backbones beyond BOTH floors -> S13 named verdict, not a raise.
    return NoDecisiveWinner(
        demoted=tuple(demoted),
        gap=gaps,
        resolution_floor=res_floors,
        seed_noise_floor=seed_floors,
    )


def _ks(per_city: list[PerCityKS], city: str) -> float:
    return next(c.ks for c in per_city if c.city == city)


def _n(per_city: list[PerCityKS], city: str) -> int:
    return next(c.n_features for c in per_city if c.city == city)


def _sem(per_city: list[PerCityKS], city: str) -> float:
    return next(c.seed_sem for c in per_city if c.city == city)
