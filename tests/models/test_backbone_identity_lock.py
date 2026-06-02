"""Swappable-backbone identity-lock (Phase-2 bake-off Task 7; §9)."""

from __future__ import annotations

import pytest
import torch

from cfm.data.training.conditioning import build_value_bearing_prefix
from cfm.models.backbone import (
    BackboneNotYetBuilt,
    build_backbone,
    shared_conditioning_builder,
    subf_vocab_size,
)
from cfm.training.config import ScaffoldConfig


def _tiny_cfg() -> ScaffoldConfig:
    return ScaffoldConfig(
        backbone="transformer-ar", d_model=32, n_layers=2, n_heads=2, max_len=64, accelerator="cpu"
    )


def test_all_backbones_use_the_same_conditioning_builder_by_identity() -> None:
    # The §9 proof: the shared builder is the ONE object, not a fork that happens to match.
    assert shared_conditioning_builder() is build_value_bearing_prefix


def test_transformer_ar_head_is_the_sealed_subf_range() -> None:
    model = build_backbone("transformer-ar", _tiny_cfg())
    # 1508 = max sub-F id (1507) + 1; the sub-F vocab is non-contiguous (directions
    # relocated to 511-870), so the head size is max-id+1, NOT the 686 tag count.
    assert model.head.out_features == subf_vocab_size() == 1508


def test_embedding_covers_the_value_bearing_conditioning_id_span() -> None:
    # A value-bearing prefix id must index the embedding without going out of range
    # (Task 6 integration: embedding spans n_subf_vocab + conditioning_id_span()).
    model = build_backbone("transformer-ar", _tiny_cfg())
    prefix = build_value_bearing_prefix(
        population_density_bucket=5,
        zoning_class=3,
        road_skeleton_class=2,
        cell_density_bucket=5,
        region="singapore",
        coastal_inland_river=1,
        sub_c_morphology_class="Asian-megacity",
        seed=7,
    )
    ids = torch.tensor([prefix], dtype=torch.long)  # (1, prefix_len)
    out = model(ids)  # must not raise an index error
    assert out.shape == (1, len(prefix), subf_vocab_size())


def test_gated_backbones_raise_until_task5() -> None:
    for name in ("mamba-hybrid", "discrete-diffusion"):
        with pytest.raises(BackboneNotYetBuilt):
            build_backbone(name, _tiny_cfg())


def test_unknown_backbone_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_backbone("gpt5", _tiny_cfg())
