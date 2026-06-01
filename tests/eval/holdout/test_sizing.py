from __future__ import annotations

import math

from cfm.eval.holdout import sizing


def test_rho_is_relative_with_a_small_absolute_floor():
    # spec §6 + δ review: the over-emission threshold is relative-to-base-rate with an
    # absolute floor. rho in (0,1); delta_floor strictly below rho (and below every
    # meaningful base rate) so the relative term governs where it must.
    assert 0.0 < sizing.RHO_BREF_REGIME < 1.0
    assert 0.0 < sizing.DELTA_FLOOR_BREF < sizing.RHO_BREF_REGIME


def test_floor_does_not_dominate_meaningful_base_rate_but_backstops_near_zero():
    # At the densest measured bucket (faithful 2.33%) the RELATIVE term must govern -
    # this is exactly where the absolute-δ form went vacuous. delta_floor only takes
    # over for genuinely near-zero strata.
    dense_faithful = 0.0233
    assert sizing.over_emission_threshold(dense_faithful) == sizing.RHO_BREF_REGIME * dense_faithful
    assert sizing.DELTA_FLOOR_BREF < sizing.RHO_BREF_REGIME * dense_faithful  # floor doesn't bind
    near_zero = 0.001
    assert sizing.over_emission_threshold(near_zero) == sizing.DELTA_FLOOR_BREF  # floor backstops


def test_rate_detection_floor_matches_hand_computed():
    # n ~ z^2 p(1-p)/delta^2  (spec §G: R2 rate-detection floor). z(0.975)=1.95996.
    p, delta = 0.10, 0.05
    expected = math.ceil((1.95996**2) * p * (1 - p) / (delta**2))
    assert sizing.rate_detection_floor(p=p, delta=delta) == expected
    assert sizing.rate_detection_floor(p=p, delta=delta) != 100  # rough, not round


def test_ks_power_floor_is_documented_inverse_square_in_effect():
    # Smaller effect => strictly larger floor (monotone), per the v1 KS approximation.
    assert sizing.ks_two_sample_floor(effect=0.2) > sizing.ks_two_sample_floor(effect=0.4)


def test_degradation_is_ordered_coarsen_then_underpowered_then_relax():
    order = [s.name for s in sizing.DegradationStep]
    assert order == ["COARSEN_STRATA", "REPORT_UNDERPOWERED", "RELAX_RHO_WITHIN_BOUND"]


def test_relax_rho_only_within_regime_bound():
    # A relaxed rho at or below the regime bound still separates faithful-from-over-emitting.
    assert sizing.relaxed_rho_is_legitimate(relaxed=sizing.RHO_BREF_REGIME)
    assert sizing.relaxed_rho_is_legitimate(relaxed=sizing.RHO_BREF_REGIME - 0.01)
    # Relaxing PAST the bound (looser) is weakening-to-pass => illegitimate.
    assert not sizing.relaxed_rho_is_legitimate(relaxed=sizing.RHO_BREF_REGIME + 0.01)


def test_STRUCTURAL_per_stratum_not_whole_set():
    """Threshold-pairing (protocol v2 §2): a whole-set-only sizing that masks an
    underpowered stratum must be caught. feasibility() returns infeasible strata
    even when the whole-set N looks sufficient."""
    populations = {0: 1000, 3: 4}  # stratum 3 is starved
    floors = {0: 50, 3: 50}
    report = sizing.feasibility(populations, floors)
    assert report.whole_set_ok is True  # 1004 >= 50 - the masking signal
    assert 3 in report.infeasible_strata  # but the per-stratum check still fires
