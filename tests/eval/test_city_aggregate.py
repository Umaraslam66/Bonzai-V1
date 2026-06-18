"""Per-city worst-case aggregation + the #21 binding-city power gate (Phase-2 bake-off Task 4).

The gate teeth that matter: the C/sqrt(n) floor must DISCRIMINATE -- demote a binding city
when it is under-powered, but let it BIND when it is powered. The pair
test_underpowered_binding_city_is_demoted / test_powered_binding_city_decides_not_demoted
proves the floor comparison flips the outcome at the boundary (not "always demote" / "always
bind").
"""

from __future__ import annotations

import pytest

from cfm.eval.city_aggregate import (
    BindingVerdict,
    NoDecisiveWinner,
    PerCityKS,
    binding_city_verdict,
    worst_case_city,
)
from cfm.eval.ladder import DecisionBasis


def test_worst_case_binds_on_worst_city() -> None:
    cities = [PerCityKS("glasgow", 0.10, 5000), PerCityKS("munich", 0.42, 5000)]
    assert worst_case_city(cities).city == "munich"


def test_pooling_is_not_silently_reintroduced() -> None:
    # b is far worse per-city despite a 1000x smaller feature count; no cell-count weighting.
    cities = [PerCityKS("a", 0.1, 100_000), PerCityKS("b", 0.5, 100)]
    assert worst_case_city(cities).ks == 0.5
    import cfm.eval.city_aggregate as agg

    assert not any("pool" in name.lower() for name in dir(agg))


def test_underpowered_binding_city_is_demoted() -> None:
    # munich is worst but its winner-vs-runner-up gap (0.005) < its own floor (~0.1087, n=156)
    # -> demote to glasgow (gap 0.10 > glasgow floor ~0.0192).
    per_backbone = {
        "AR": [PerCityKS("munich", 0.42, 156), PerCityKS("glasgow", 0.30, 5000)],
        "diff": [PerCityKS("munich", 0.425, 156), PerCityKS("glasgow", 0.20, 5000)],
    }
    v = binding_city_verdict(per_backbone)
    assert "munich" in v.demoted_from
    assert v.binding_city == "glasgow"
    assert v.winner == "diff"


def test_powered_binding_city_decides_not_demoted() -> None:
    # munich worst AND gap (0.16) > its floor (~0.1087, n=156) -> munich binds, NOT demoted.
    per_backbone = {
        "AR": [PerCityKS("munich", 0.42, 156), PerCityKS("glasgow", 0.20, 5000)],
        "diff": [PerCityKS("munich", 0.58, 156), PerCityKS("glasgow", 0.30, 5000)],
    }
    v = binding_city_verdict(per_backbone)
    assert v.binding_city == "munich"  # powered -> munich itself decides
    assert v.demoted_from == ()  # nobody demoted
    assert v.winner == "AR"  # munich: 0.42 < 0.58 -> AR wins


# --- The two-floor / three-band seed→verdict rule (spec §9, plan T10 Step 3) --------------- #
# A winner is crowned at a city ONLY if the winner-vs-runner-up gap clears
# effective_floor = max(C/sqrt(n) resolvability, seed-noise reproducibility). The three bands:
#   DECISIVE (gap > both)        -> BindingVerdict (crown)
#   LUCK     (gap <= seed-noise) -> NoDecisiveWinner (reproducibility blocks)
#   MIDDLE   (clears one floor not the other) -> NoDecisiveWinner (the likely near-tie outcome)
# Each band below fires on a constructed single-city fixture (so the worst-first loop has one
# pass and the band logic is isolated).


def test_decisive_when_gap_clears_both_floors() -> None:
    # gap 0.30 >> max(resolution ~0.019 @ n=5000, seed-noise 0.02) -> crown.
    per_backbone = {
        "AR": [PerCityKS("munich", 0.20, 5000, seed_sem=0.02)],
        "diff": [PerCityKS("munich", 0.50, 5000, seed_sem=0.02)],
    }
    v = binding_city_verdict(per_backbone)
    assert isinstance(v, BindingVerdict)
    assert v.binding_city == "munich" and v.winner == "AR"
    assert v.gap > v.city_floor  # cleared the effective (max) floor


def test_luck_band_seed_noise_blocks_crown() -> None:
    # gap 0.05 clears the tiny resolution floor (n=100000 -> ~0.0043) but is INSIDE the
    # seed-noise band (0.10): reproducibility blocks -> NO_DECISIVE_WINNER.
    per_backbone = {
        "AR": [PerCityKS("munich", 0.40, 100_000, seed_sem=0.10)],
        "diff": [PerCityKS("munich", 0.45, 100_000, seed_sem=0.10)],
    }
    v = binding_city_verdict(per_backbone)
    assert isinstance(v, NoDecisiveWinner)
    assert v.basis is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert "munich" in v.demoted
    assert v.seed_noise_floor["munich"] > v.resolution_floor["munich"]  # seed-noise binds


def test_middle_band_resolution_blocks_crown() -> None:
    # gap 0.05 clears the tight seed-noise (0.01) but is INSIDE the resolution floor
    # (n=156 -> ~0.109): not statistically resolvable -> NO_DECISIVE_WINNER (the MIDDLE band).
    per_backbone = {
        "AR": [PerCityKS("munich", 0.40, 156, seed_sem=0.01)],
        "diff": [PerCityKS("munich", 0.45, 156, seed_sem=0.01)],
    }
    v = binding_city_verdict(per_backbone)
    assert isinstance(v, NoDecisiveWinner)
    assert v.basis is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert "munich" in v.demoted
    assert v.resolution_floor["munich"] > v.seed_noise_floor["munich"]  # resolvability binds


def test_seed_sem_defaults_to_zero_preserves_legacy_floor() -> None:
    # seed_sem unset -> effective floor == C/sqrt(n) (old behavior): powered munich still binds.
    per_backbone = {
        "AR": [PerCityKS("munich", 0.42, 156), PerCityKS("glasgow", 0.20, 5000)],
        "diff": [PerCityKS("munich", 0.58, 156), PerCityKS("glasgow", 0.30, 5000)],
    }
    v = binding_city_verdict(per_backbone)
    assert isinstance(v, BindingVerdict) and v.binding_city == "munich"


def test_binding_verdict_raises_on_city_missing_from_a_backbone() -> None:
    # ragged shape (a): a city in the first backbone is MISSING from another -> clear ValueError,
    # NOT a bare StopIteration.
    per_backbone = {
        "AR": [PerCityKS("munich", 0.42, 156), PerCityKS("glasgow", 0.30, 5000)],
        "diff": [PerCityKS("glasgow", 0.20, 5000)],  # munich missing
    }
    with pytest.raises(ValueError, match="same held-out cities"):
        binding_city_verdict(per_backbone)


def test_binding_verdict_raises_on_city_only_in_a_noninitial_backbone() -> None:
    # ragged shape (b): a city present ONLY in a non-first backbone -> must RAISE, not be
    # silently dropped (which could skip the real worst city).
    per_backbone = {
        "AR": [PerCityKS("glasgow", 0.30, 5000)],
        "diff": [
            PerCityKS("glasgow", 0.20, 5000),
            PerCityKS("munich", 0.42, 156),
        ],  # munich only here
    }
    with pytest.raises(ValueError, match="same held-out cities"):
        binding_city_verdict(per_backbone)
