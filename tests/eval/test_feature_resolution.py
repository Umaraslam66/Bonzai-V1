"""Per-feature resolution + winner-vs-runner-up seam (Phase-2 bake-off Task 3)."""

from __future__ import annotations

import math

import pytest

from cfm.eval.feature_resolution import (
    DecisionUnresolvable,
    Escalation,
    binding_resolution,
    cells_to_resolve,
    check_decision_resolvable,
    escalation_for,
    per_feature_resolved_gap,
    single_region_floor_gap,
)


def test_resolution_uses_feature_count_not_inherited_0076() -> None:
    g = per_feature_resolved_gap(n_features=400)
    assert math.isclose(g, 1.358 * math.sqrt(2 / 400), rel_tol=1e-9)
    assert g != 0.076  # NOT the inherited per-cell gap


def test_binding_resolution_is_the_worst_resolved_feature() -> None:
    # building-area has fewer samples (coarser/larger gap) than road-length -> it binds
    res = binding_resolution({"building_area_m2": 100, "road_length_m": 2000})
    assert res.binding_metric == "building_area_m2"
    assert res.binding_gap == per_feature_resolved_gap(n_features=100)


def test_seam_fires_when_winner_runnerup_gap_below_binding() -> None:
    # winner 0.20, runner-up 0.205 -> gap 0.005, below a 0.05 binding gap
    with pytest.raises(DecisionUnresolvable):
        check_decision_resolvable([0.20, 0.205, 0.40], binding_gap=0.05)


def test_seam_silent_when_winner_runnerup_gap_clears() -> None:
    check_decision_resolvable([0.20, 0.30, 0.40], binding_gap=0.05)  # no raise


def test_last_place_tie_does_not_fire_the_seam() -> None:
    # winner clearly separated; 2nd and 3rd tied -> irrelevant to the decision, no raise
    check_decision_resolvable([0.20, 0.39, 0.395], binding_gap=0.05)


def test_single_backbone_is_trivially_resolvable() -> None:
    check_decision_resolvable([0.20], binding_gap=0.05)  # no runner-up -> no raise


def test_three_tier_escalation_boundaries() -> None:
    binding, floor = 0.10, 0.04
    assert escalation_for(gap=0.12, binding_gap=binding, floor_gap=floor) is Escalation.RESOLVED
    assert (
        escalation_for(gap=0.07, binding_gap=binding, floor_gap=floor)
        is Escalation.GENERATE_MORE_CELLS
    )
    assert (
        escalation_for(gap=0.02, binding_gap=binding, floor_gap=floor) is Escalation.SECOND_REGION
    )


def test_floor_is_the_fixed_reference_asymptote() -> None:
    # as generated -> inf against n_ref fixed, gap -> 1.358/sqrt(n_ref)
    assert math.isclose(
        single_region_floor_gap(n_reference_features=900), 1.358 / 30.0, rel_tol=1e-9
    )
    # the floor is finer than any finite symmetric gap at the same n
    assert single_region_floor_gap(n_reference_features=900) < per_feature_resolved_gap(
        n_features=900
    )


def test_cells_to_resolve_is_one_shot_and_monotone() -> None:
    # a finer target needs more cells; the computed target resolves THIS gap in one shot
    coarse = cells_to_resolve(target_gap=0.10, features_per_cell=4.0)
    fine = cells_to_resolve(target_gap=0.05, features_per_cell=4.0)
    assert fine > coarse
    # the n_features implied by the returned n_cells actually resolves the target gap
    n_features = cells_to_resolve(target_gap=0.05, features_per_cell=1.0)
    assert per_feature_resolved_gap(n_features=n_features) <= 0.05
