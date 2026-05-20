from __future__ import annotations

from collections.abc import Sequence

import pytest

from cfm.eval.perplexity_gap import (
    GapResult,
    PerCellNLL,
    compute_perplexity_gap,
)


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
