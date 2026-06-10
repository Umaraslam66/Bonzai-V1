"""Absorbing-state masking for discrete diffusion (Phase-2 bake-off Task 8)."""

from __future__ import annotations

import torch

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import CONDITIONING_VALUE_BASE, conditioning_id_span
from cfm.models.diffusion.mask import MASK_ID, apply_absorbing_mask


def test_mask_id_is_above_vocab_and_conditioning_span() -> None:
    n_subf = max(vocab_tag_to_id().values()) + 1
    assert MASK_ID >= n_subf  # never a real token
    assert MASK_ID >= CONDITIONING_VALUE_BASE + conditioning_id_span()  # never a conditioning id


def test_absorbing_mask_replaces_a_fraction_with_MASK_ID() -> None:
    seq = torch.arange(100).reshape(1, 100)
    masked, target_mask = apply_absorbing_mask(seq, frac=0.5, seed=7)
    assert (masked == MASK_ID).sum() > 0
    assert bool((masked[target_mask] == MASK_ID).all())  # masked positions hold MASK_ID
    assert bool((masked[~target_mask] == seq[~target_mask]).all())  # others untouched


def test_absorbing_mask_is_seeded_reproducible() -> None:
    seq = torch.arange(100).reshape(1, 100)
    a, ma = apply_absorbing_mask(seq, frac=0.5, seed=7)
    b, mb = apply_absorbing_mask(seq, frac=0.5, seed=7)
    assert torch.equal(a, b) and torch.equal(ma, mb)


def test_absorbing_mask_differs_across_seeds() -> None:
    seq = torch.arange(100).reshape(1, 100)
    a, _ = apply_absorbing_mask(seq, frac=0.5, seed=7)
    b, _ = apply_absorbing_mask(seq, frac=0.5, seed=8)
    assert not torch.equal(a, b)
