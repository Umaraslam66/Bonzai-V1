from __future__ import annotations

from collections.abc import Sequence
from math import comb

import pytest

from cfm.eval.perplexity_gap import (
    GapResult,
    PerCellNLL,
    _binomial_sign_test_pvalue,
    compute_perplexity_gap,
)


def test_sign_test_no_overflow_at_full_run_n() -> None:
    """At n=8000 (full-run scale) the exact tail must not overflow.

    The hand-rolled ``comb(n, x) * 0.5**n`` raised OverflowError here (comb(8000, x)
    is too large to convert to float); the full 6-checkpoint matrix died on it.
    """
    p_half = _binomial_sign_test_pvalue(n=8000, k=4200)
    assert 0.0 <= p_half <= 1.0
    # ~13 sigma above half -> significant; ~0.2 sigma -> not.
    assert _binomial_sign_test_pvalue(n=8000, k=4600) < 0.01
    assert _binomial_sign_test_pvalue(n=8000, k=4010) > 0.01


def test_sign_test_matches_exact_for_small_n() -> None:
    """Where ``comb`` does NOT overflow, the (fixed) tail must match it to float precision."""
    for n, k in [(30, 30), (30, 0), (30, 20), (10, 7), (100, 60)]:
        exact = sum(comb(n, x) * 0.5**n for x in range(k, n + 1))
        assert _binomial_sign_test_pvalue(n=n, k=k) == pytest.approx(exact, abs=1e-9)


def _fake_model_forward(
    *, micro_tokens: Sequence[int], conditioning_prefix: Sequence[int]
) -> float:
    """Deterministic toy model: NLL is a function of conditioning agreement.

    If conditioning_prefix[0] equals micro_tokens[0], return low NLL (0.1);
    otherwise high (0.5). This simulates a model that uses conditioning.
    """
    if conditioning_prefix and micro_tokens and conditioning_prefix[0] == micro_tokens[0]:
        return 0.1
    return 0.5


def test_gap_calculation_on_toy_data_positive_when_conditioning_matters() -> None:
    cells = [
        PerCellNLL(
            cell_id=f"c{k}",
            micro_tokens=[k % 3] * 10,
            matched_conditioning_prefix=[k % 3],
            shuffled_conditioning_prefix=[(k + 1) % 3],
        )
        for k in range(30)
    ]
    result = compute_perplexity_gap(
        cells=cells, model_forward=_fake_model_forward, p_threshold=0.01
    )
    assert isinstance(result, GapResult)
    # All 30 cells: matched NLL = 0.1, shuffled NLL = 0.5 → gap = 0.4 nats/cell.
    # Per-token gap = 0.4 / 10 = 0.04 (just under §11.5 threshold; but signal
    # is monotonic so sign-test should be significant).
    assert result.gap_nats_per_token > 0
    assert result.fraction_positive == pytest.approx(1.0)
    assert result.sign_test_significant_at_p


def test_gap_calculation_zero_signal() -> None:
    """If the fake model returns identical NLL regardless of conditioning,
    gap should be ≈0 and sign test should NOT be significant.
    """

    def _flat(*, micro_tokens, conditioning_prefix) -> float:
        return 0.3

    cells = [
        PerCellNLL(
            cell_id=f"c{k}",
            micro_tokens=[k] * 10,
            matched_conditioning_prefix=[k],
            shuffled_conditioning_prefix=[k + 1],
        )
        for k in range(30)
    ]
    result = compute_perplexity_gap(cells=cells, model_forward=_flat, p_threshold=0.01)
    assert result.gap_nats_per_token == pytest.approx(0.0)
    assert not result.sign_test_significant_at_p
