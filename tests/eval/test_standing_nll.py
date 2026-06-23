"""Perplexity-gap NLL wrapper (spec §2, §9 tooth).

THE consistency tooth: the harness's per-cell NLL must equal an INDEPENDENT hand-computed
masked-body cross-entropy from the model's own logits (external source of truth) — proving
eval NLL uses the same prefix masking + nats/token reduction as training. Plus the gap loop
(reuses perplexity_gap.GapResult + sign test) computes shuffled - matched and a sign test.
"""

from __future__ import annotations

import torch

from cfm.eval.standing.nll import (
    GapCell,
    cell_nll,
    compute_gap,
    effective_macro_shuffle_fraction,
)
from cfm.models.micro_ar import MicroAR, MicroARConfig

N_VOCAB = 300
N_COND = 576
PREFIX = [*range(N_VOCAB, N_VOCAB + 9), 0]  # 9 conditioning ids + char placeholder
CHAR = [1.78, 1.19, 0.08, 1.43, 1.51, 1.0, 1.0]


def _tiny_model():
    torch.manual_seed(0)
    cfg = MicroARConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_subf_vocab=N_VOCAB,
        n_cond=N_COND,
        max_len=128,
        n_char_stats=7,
        char_position=9,
    )
    return MicroAR(cfg).eval()


def test_cell_nll_equals_hand_masked_body_ce():
    m = _tiny_model()
    body = [12, 45, 7, 200, 13, 99]
    got = cell_nll(m, body_tokens=body, conditioning_prefix=PREFIX, char_stats=CHAR, device="cpu")

    # INDEPENDENT reference: mean next-token CE over body-target positions only.
    ids = torch.tensor([PREFIX + body])
    cs = torch.tensor([CHAR], dtype=torch.float32)
    with torch.no_grad():
        logits = m(ids, char_stats=cs)[:, :-1]  # logits[:, i] predicts position i+1
    targets = ids[:, 1:]
    pl = len(PREFIX)
    hand = torch.nn.functional.cross_entropy(logits[0, pl - 1 :], targets[0, pl - 1 :]).item()
    assert abs(got - hand) < 1e-5


def test_cell_nll_changes_with_conditioning():
    """The gap's premise: a different conditioning prefix gives a different body NLL."""
    m = _tiny_model()
    body = [12, 45, 7, 200, 13, 99]
    matched = cell_nll(m, body_tokens=body, conditioning_prefix=PREFIX, char_stats=CHAR)
    other_prefix = [*range(N_VOCAB + 20, N_VOCAB + 29), 0]
    shuffled = cell_nll(m, body_tokens=body, conditioning_prefix=other_prefix, char_stats=CHAR)
    assert matched != shuffled


def test_compute_gap_signs_and_structure():
    m = _tiny_model()
    cells = [
        GapCell(
            cell_id=str(i),
            body_tokens=[10 + i, 20, 30, 40 + i],
            matched_prefix=PREFIX,
            shuffled_prefix=[*range(N_VOCAB + 9, N_VOCAB + 18), 0],  # different macro ids
            matched_char_stats=CHAR,
            shuffled_char_stats=CHAR,  # macro-only: char_stats matched
        )
        for i in range(5)
    ]
    r = compute_gap(m, cells, device="cpu", p_threshold=0.05)
    assert r.n_cells == 5
    assert 0.0 <= r.fraction_positive <= 1.0
    # gap is reported in nats/token (== mean per-cell gap by the model_forward contract)
    assert r.gap_nats_per_token == r.mean_gap_nats_per_cell


def test_effective_macro_shuffle_fraction_detects_noop():
    """A no-op shuffle (donor prefix == own prefix) must read 0.0; a real shuffle 1.0 —
    so a near-zero gap can be told apart from a near-zero effective shuffle."""
    noop = [GapCell(str(i), [1, 2], [9, 0], [9, 0], [0.0], [0.0]) for i in range(4)]
    real = [GapCell(str(i), [1, 2], [9, 0], [8, 0], [0.0], [0.0]) for i in range(4)]
    assert effective_macro_shuffle_fraction(noop) == 0.0
    assert effective_macro_shuffle_fraction(real) == 1.0
    half = noop[:2] + real[:2]
    assert effective_macro_shuffle_fraction(half) == 0.5
