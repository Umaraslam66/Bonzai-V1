from __future__ import annotations

import math

from cfm.eval.holdout import sizing


def test_DELTA_is_a_single_justified_number_below_one():
    # spec §6 + §G option-3: δ is a chosen, justified rate-excess, not a round default.
    assert 0.0 < sizing.DELTA_BREF_REGIME < 1.0
    assert sizing.DELTA_BREF_REGIME != 0.5  # not a round default (rough-numbers heuristic)


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
    assert order == ["COARSEN_STRATA", "REPORT_UNDERPOWERED", "RELAX_DELTA_WITHIN_BOUND"]


def test_relax_delta_only_within_regime_bound():
    # A relaxed δ that still separates faithful-from-over-emitting (<= the bound) is OK.
    assert sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME)
    assert sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME - 0.001)
    # Relaxing PAST the regime-distinguishing bound is weakening-to-pass => illegitimate.
    assert not sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME + 0.001)


def test_STRUCTURAL_per_stratum_not_whole_set():
    """Threshold-pairing (protocol v2 §2): a whole-set-only sizing that masks an
    underpowered stratum must be caught. feasibility() returns infeasible strata
    even when the whole-set N looks sufficient."""
    populations = {0: 1000, 3: 4}  # stratum 3 is starved
    floors = {0: 50, 3: 50}
    report = sizing.feasibility(populations, floors)
    assert report.whole_set_ok is True  # 1004 >= 50 - the masking signal
    assert 3 in report.infeasible_strata  # but the per-stratum check still fires
