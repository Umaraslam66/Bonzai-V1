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

import pytest
import torch

from cfm.data.training.conditioning import (
    CONDITIONING_ID_BASE,
    CONDITIONING_PREFIX_LEN,
    build_value_bearing_prefix,
    conditioning_id_span,
)
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


def test_prefix_mask_invariant_holds_at_live_n_cond_576():
    """F16 hygiene: the tests above prove the prefix-mask invariant at n_cond=8
    fixtures, but production (backbone.build_micro_ar) builds
    n_cond=conditioning_id_span()=576 — exercise the LIVE shape. Tiny dims keep
    it CPU-cheap; the input prefix is a REAL value-bearing prefix (ids in
    [CONDITIONING_ID_BASE, CONDITIONING_ID_BASE + 576)), so this also proves the
    embedding table spans the live conditioning block. (512 -> 576: Task 24a
    appended the city_identity field, 9 fields * 64 stride.)"""
    n_cond = conditioning_id_span()
    assert n_cond == 576  # the live span this test pins (Task 24a reverse-lock)
    m = MicroAR(
        MicroARConfig(
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_subf_vocab=CONDITIONING_ID_BASE,  # 1508: conditioning ids start right above
            n_cond=n_cond,
            max_len=32,
        )
    )
    prefix = build_value_bearing_prefix(
        population_density_bucket=0,
        zoning_class=1,
        road_skeleton_class=1,
        cell_density_bucket=2,
        region=None,
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=7,
        city_identity="singapore",  # Task 24a: 9th field
    )
    assert len(prefix) == 9
    assert all(CONDITIONING_ID_BASE <= i < CONDITIONING_ID_BASE + n_cond for i in prefix)
    T, B = 20, 2
    body = torch.randint(0, _N_SUBF, (B, T - len(prefix)))
    tokens = torch.cat([torch.tensor([prefix, prefix]), body], dim=1)
    out = m.training_loss(tokens, prefix_len=torch.tensor([9, 9]))
    assert out.loss.requires_grad
    # the load-bearing invariant at the live shape: prefix targets are masked
    assert out.n_supervised_positions == (T - 9) * B  # == 22


def test_generation_at_exact_positional_capacity_through_production_build():
    """Pins the F15 commensurability-note arithmetic (Task 14 follow-up) on the
    PRODUCTION build path: build_backbone sizes positions as
    cfg.max_len + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS (n_cond=576
    is embedding-table ROWS on the token-id axis, NOT positions). With max_len=32
    the capacity is 42 positions, so generate_cell_tokens(max_new=32) returns
    (9 ids + 1 character placeholder) + 32 generated = 42 — exactly fills capacity,
    zero headroom, by construction (the diagnostic sbatch's --max-len 2048 /
    --eval-max-new 2048 = 2058-position case in miniature; Task 24a moved the
    prefix from 8 to 9 id positions, Task 24b appended the continuous position).

    Non-vacuity at the boundary: max_new = max_len + 1 ALSO survives because the
    generation loop's final forward sees only prefix + max_new - 1 positions (the
    last sampled token is appended, never fed back); max_new = max_len + 2 is the
    first overflow and raises IndexError from the positional-embedding lookup."""
    from cfm.data.training.conditioning import CHARACTER_PREFIX_POSITIONS
    from cfm.data.training.datamodule import CHARACTER_PLACEHOLDER_ID
    from cfm.inference.generate import generate_cell_tokens
    from cfm.models.backbone import build_backbone
    from cfm.training.config import ScaffoldConfig

    cfg = ScaffoldConfig(
        region="singapore",
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_len=32,
        accelerator="cpu",
        devices=1,
    )
    model = build_backbone("transformer-ar", cfg)
    capacity = cfg.max_len + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS
    assert model.pos.num_embeddings == capacity == 42
    prefix = [
        *build_value_bearing_prefix(
            population_density_bucket=0,
            zoning_class=1,
            road_skeleton_class=1,
            cell_density_bucket=2,
            region=None,
            coastal_inland_river=0,
            sub_c_morphology_class="Asian-megacity",
            seed=7,
            city_identity="singapore",  # Task 24a: 9th field
        ),
        CHARACTER_PLACEHOLDER_ID,  # Task 24b: the continuous character position
    ]
    assert len(prefix) == CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS == 10
    stats = [0.9, 0.4, 0.3, 1.1, 0.7, 1.0, 1.0]

    # exact fit: 10-token prefix + 32 generated == 42 == positional capacity
    out = generate_cell_tokens(model, prefix=prefix, max_new=cfg.max_len, seed=0, char_stats=stats)
    assert len(out) == cfg.max_len

    # one past the loop's last in-capacity forward: positional lookup overflows
    with pytest.raises(IndexError):
        generate_cell_tokens(
            model, prefix=prefix, max_new=cfg.max_len + 2, seed=0, char_stats=stats
        )
