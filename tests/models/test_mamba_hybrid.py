"""Tests for the mamba-hybrid bake-off backbone (Task 4).

Module-gated on ``mamba_ssm``: the default repo env has no mamba, so the whole module
skips there. The construction/interleave-layout tests run on CPU (Mamba CONSTRUCTION is
CPU-safe). The forward-shape test is gated on CUDA because ``mamba_ssm`` 2.3.1's
``Mamba.forward`` requires CUDA — it is exercised for real at the Task-7 GPU smoke.
"""

import pytest

pytest.importorskip("mamba_ssm")

import torch

from cfm.models.mamba_hybrid import (
    MambaHybrid,
    MambaHybridConfig,
    _interleave_positions,
)


def _cfg() -> MambaHybridConfig:
    return MambaHybridConfig(
        d_model=128,
        n_layers=8,
        n_heads=4,
        n_subf_vocab=40,
        n_cond=16,
        max_len=64,
        n_char_stats=7,
        char_position=9,
        d_state=16,
        d_conv=4,
        expand=2,
    )


def test_mamba_hybrid_construction_and_interleave() -> None:
    m = MambaHybrid(_cfg())
    # 8 layers, 7:1 => 7 mamba + 1 transformer; >=1 transformer always.
    assert m.n_transformer_layers == 1
    assert m.n_mamba_layers == 7
    assert m.n_mamba_layers + m.n_transformer_layers == 8
    # _build_head() ran -> head exists and projects to the sub-F range.
    assert m.head.out_features == 40


def test_interleave_keeps_transformer_at_small_layer_count() -> None:
    # n_layers=3 with every=7: the periodic rule selects no transformer, so the
    # fallback must force the last layer to a transformer (spec §3.3, >=1 always).
    layout = _interleave_positions(3, 7)
    assert sum(layout) >= 1
    assert layout[-1] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm forward requires CUDA")
def test_mamba_hybrid_forward_shape_cuda() -> None:
    m = MambaHybrid(_cfg()).cuda()
    ids = torch.randint(0, 56, (2, 20), device="cuda")
    char = torch.randn(2, 7, device="cuda")
    out = m(ids, char_stats=char)
    assert out.shape == (2, 20, 40)  # head -> sub-F range
