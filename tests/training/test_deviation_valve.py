"""Comparability-deviation valve (Phase-2 bake-off Task 9)."""

from __future__ import annotations

import math

import pytest

from cfm.training.deviation_log import DeviationError, DeviationLog, is_train_failure


def test_train_failure_is_diverged_nan_or_flatline_not_low_score() -> None:
    assert is_train_failure([5.0, math.nan]) is True  # NaN
    assert is_train_failure([5.0, 6.0, 8.0, 12.0]) is True  # diverging
    assert is_train_failure([5.0, 5.0, 5.0, 5.0]) is True  # flatline from step 0
    assert is_train_failure([5.0, 4.0, 3.5, 3.2]) is False  # trains fine (may lose later)
    assert is_train_failure([5.0, 4.0, 3.0, 3.0, 3.0]) is False  # converged plateau != failure


def test_inf_loss_is_failure() -> None:
    assert is_train_failure([5.0, math.inf]) is True


def test_deviation_must_be_logged_with_a_uniform_rule_not_a_bespoke_number() -> None:
    log = DeviationLog()
    log.record(
        backbone="discrete-diffusion",
        scale="100M",
        rule="loss-scale-normalized-lr",
        trigger="flatline",
    )
    assert len(log.entries) == 1

    # a bespoke per-backbone deviation with no named rule, on a "scores-lower" trigger, is rejected
    with pytest.raises(DeviationError):
        log.record(backbone="discrete-diffusion", scale="100M", rule=None, trigger="scores-lower")


def test_scores_lower_trigger_is_rejected_even_with_a_rule() -> None:
    log = DeviationLog()
    with pytest.raises(DeviationError):
        log.record(backbone="mamba-hybrid", scale="300M", rule="some-rule", trigger="scores-lower")


def test_missing_rule_is_rejected_even_on_a_real_failure_trigger() -> None:
    log = DeviationLog()
    with pytest.raises(DeviationError):
        log.record(backbone="mamba-hybrid", scale="300M", rule=None, trigger="diverged")
