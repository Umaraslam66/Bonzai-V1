"""Effective-batch lock via gradient accumulation (Phase-2 bake-off Task 9)."""

from __future__ import annotations

from cfm.training.config import ScaffoldConfig
from cfm.training.train import effective_batch_size


def test_effective_batch_constant_across_per_gpu_batch_via_grad_accum() -> None:
    # 30M fits batch 8; 1B fits batch 2 -> grad_accum 4 keeps the effective batch identical
    small = ScaffoldConfig(batch_size=8, grad_accum=1, devices=4)
    large = ScaffoldConfig(batch_size=2, grad_accum=4, devices=4)
    assert effective_batch_size(small) == effective_batch_size(large) == 32


def test_grad_accum_defaults_to_one() -> None:
    assert ScaffoldConfig().grad_accum == 1
