from __future__ import annotations

from cfm.eval.ladder import feasible_ladder

M30, M100, M300, B1 = 30_000_000, 100_000_000, 300_000_000, 1_000_000_000


def test_r20_unique_only_30M():
    d = feasible_ladder(r=20.0)
    assert d.feasible == (M30,)


def test_r10_unique_only_30M():
    assert feasible_ladder(r=10.0).feasible == (M30,)


def test_r5_unique_adds_100M():
    assert feasible_ladder(r=5.0).feasible == (M30, M100)


def test_r5_epoch4_adds_300M():
    assert feasible_ladder(r=5.0, epoch_factor=4.0).feasible == (M30, M100, M300)


def test_1B_dropped_even_at_low_r_and_E4():
    # 1B needs r <= 624M*E/1e9; at E=4 that is r<=2.496. r=2.5 must still drop it.
    assert B1 not in feasible_ladder(r=2.5, epoch_factor=4.0).feasible


def test_empty_ladder_sets_escalate():
    d = feasible_ladder(r=1000.0)  # nothing clears
    assert d.feasible == () and d.escalate_more_data is True


def test_conservative_rounding_uses_upper_r_ci():
    # CI [5,7] straddles the 100M boundary (6.24). Conservative => use 7 => drop 100M.
    from cfm.eval.ladder import feasible_ladder_conservative

    assert feasible_ladder_conservative(r_ci_high=7.0).feasible == (M30,)


def test_decision_basis_step_function():
    from cfm.eval.ladder import DecisionBasis, decision_basis

    assert decision_basis(0) is DecisionBasis.ESCALATE_MORE_DATA
    assert decision_basis(1) is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert decision_basis(2) is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert decision_basis(3) is DecisionBasis.SCALING_CURVE
    assert decision_basis(4) is DecisionBasis.SCALING_CURVE


def test_train_tokens_matches_the_frozen_multiregion_marker():
    """F9 TRAIN_TOKENS guard: the hard-coded ladder constant must equal the frozen
    multiregion _EVAL_SET_LOCKED's train_tokens (real artifact, tracked in git --
    same real-marker idiom as test_resolution_seam's
    test_reads_the_real_frozen_marker_fields). A re-lock that changes the corpus
    without updating the ladder constant fails HERE, not silently at the bake-off."""
    import yaml

    from cfm.eval import ladder
    from cfm.eval.holdout.paths import multiregion_eval_set_locked_marker

    marker = yaml.safe_load(
        multiregion_eval_set_locked_marker("2026-04-15.0").read_text(encoding="utf-8")
    )
    assert ladder.TRAIN_TOKENS == marker["train_tokens"]
