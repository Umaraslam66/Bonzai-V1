"""Task 7 discrimination tests for the toy micro-generator.

Two locked invariants (spec §7):
  1. The conditioning prefix is GIVEN, never predicted -> its target positions are
     masked out of the loss. Supervised positions = all (T - prefix_len) cell-token
     next-token targets per example, INCLUDING the first-cell-token-from-conditioning
     prediction (target index prefix_len-1). (Plan's `-2` was a typo; correct count
     is (T - prefix_len) * batch.)
  2. The output head projects to the sub-F vocab range ONLY (n_subf_vocab). The model
     embeds conditioning ids (input-only) but NEVER predicts them.
"""

from __future__ import annotations

import torch

from cfm.models.micro_ar import MicroAR, MicroARConfig

# Real sub-F prediction range (sub-F ids span 0..1507 -> 1508). n_cond=8 conditioning
# id-block. Embedding table covers n_subf_vocab + n_cond = 1516; head = 1508.
_N_SUBF = 1508
_N_COND = 8


def _cfg(**kw) -> MicroARConfig:
    base = dict(
        d_model=64, n_layers=2, n_heads=2, n_subf_vocab=_N_SUBF, n_cond=_N_COND, max_len=128
    )
    base.update(kw)
    return MicroARConfig(**base)


def test_loss_ignores_conditioning_prefix():
    m = MicroAR(_cfg())
    T, B, prefix = 20, 2, 8
    tokens = torch.randint(0, _N_SUBF, (B, T))
    prefix_len = torch.tensor([prefix, prefix])
    out = m.training_loss(tokens, prefix_len=prefix_len)
    assert out.loss.requires_grad
    assert out.loss.ndim == 0
    # supervise every cell-token target; mask only the conditioning targets.
    assert out.n_supervised_positions == (T - prefix) * B  # == 24


def test_loss_mask_tracks_variable_prefix_len():
    """Regime-distinguishing: different prefix_len per example changes the count
    (proves the mask is per-example, not a hardcoded constant)."""
    m = MicroAR(_cfg())
    T = 20
    tokens = torch.randint(0, _N_SUBF, (2, T))
    out = m.training_loss(tokens, prefix_len=torch.tensor([8, 5]))
    assert out.n_supervised_positions == (T - 8) + (T - 5)  # 12 + 15 = 27


def test_loss_masks_right_padding_via_seq_len():
    """A padded batch: seq_len marks the real length; targets in the right-padding
    region are masked just like the conditioning prefix."""
    m = MicroAR(_cfg())
    T = 20
    tokens = torch.randint(0, _N_SUBF, (2, T))
    # ex0: prefix 8, real length 14 (6 padding); ex1: prefix 8, real length 20 (no pad)
    out = m.training_loss(tokens, prefix_len=torch.tensor([8, 8]), seq_len=torch.tensor([14, 20]))
    assert out.n_supervised_positions == (14 - 8) + (20 - 8)  # 6 + 12 = 18


def test_logits_cover_only_subf_predict_range():
    m = MicroAR(_cfg())
    logits = m(torch.randint(0, _N_SUBF, (1, 10)))
    assert logits.shape == (1, 10, _N_SUBF)  # predicts only sub-F vocab, never conditioning ids


def test_model_embeds_conditioning_ids_without_predicting_them():
    """The embedding table spans sub-F + conditioning so a prepended conditioning
    id (>= n_subf_vocab) is a valid INPUT; the head still never emits one."""
    m = MicroAR(_cfg())
    # a sequence whose prefix uses conditioning ids [1508..1515] then cell tokens
    cond = list(range(_N_SUBF, _N_SUBF + _N_COND))
    body = torch.randint(0, _N_SUBF, (10,)).tolist()
    ids = torch.tensor([cond + body])
    logits = m(ids)  # must not raise (embedding index in range)
    assert logits.shape[-1] == _N_SUBF
