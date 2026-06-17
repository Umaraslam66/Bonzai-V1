"""Swappable-backbone abstraction (Phase-2 bake-off Task 7; §9).

The bake-off compares THREE backbones on one ruler. That only holds if they differ
ONLY in their sequence-mixing layers (+ diffusion's quarantined loss/generation/mask)
and SHARE the embedding / value-bearing conditioning builder / sub-F vocab head / eval
content -- by IDENTITY, not "equal output today". This module is the shared factory and
the identity anchors a Gate-6 test asserts against.

mamba-hybrid and discrete-diffusion are still gated: ``build_backbone`` raises
``BackboneNotYetBuilt`` because the backbone MODULES are not implemented yet. The
Task-5 mamba-ssm verify-before-lock SETTLED on 2026-06-17 (GPU fwd/bwd kernel
numerics PASS on A100; reports/2026-06-17-mamba-gpu-half-probe.json), so the mamba
stack is locked (env_lock.py); ``build_backbone`` now asserts that mamba env-lock at
the mamba-hybrid construction site (``assert_mamba_env_locked``) ahead of the gate,
so the lock is in place when MambaHybrid lands. discrete-diffusion has no mamba
dependency and gates directly. The transformer-ar backbone (``MicroAR``) exists today
and is sized here to cover the value-bearing conditioning id span (Task 6 integration:
the embedding must span ``n_subf_vocab + conditioning_id_span()``).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CHARACTER_STAT_CHANNELS,
    CONDITIONING_PREFIX_LEN,
    build_value_bearing_prefix,
    conditioning_id_span,
)
from cfm.models.micro_ar import MicroAR, MicroARConfig
from cfm.training.env_lock import assert_mamba_env_locked

_AR_FAMILY = ("transformer-ar",)
_MAMBA_FAMILY = ("mamba-hybrid",)
_DIFFUSION_FAMILY = ("discrete-diffusion",)
_GATED = _MAMBA_FAMILY + _DIFFUSION_FAMILY


class BackboneNotYetBuilt(NotImplementedError):
    """A backbone whose build is gated behind Task 5's mamba-ssm verify-before-lock."""


def shared_conditioning_builder() -> Callable[..., list[int]]:
    """The ONE value-bearing conditioning builder every backbone consumes (identity anchor)."""
    return build_value_bearing_prefix


def subf_vocab_size() -> int:
    """The sealed sub-F head dimension every backbone shares (one source)."""
    return max(vocab_tag_to_id().values()) + 1


def build_backbone(name: str, cfg: object) -> nn.Module:
    """Construct the backbone ``name`` from a ``ScaffoldConfig``-like ``cfg``.

    The embedding spans ``n_subf_vocab + conditioning_id_span()`` so value-bearing
    conditioning ids never index out of range; positions cover the cell-token budget plus
    the conditioning prefix (``CONDITIONING_PREFIX_LEN`` id positions +
    ``CHARACTER_PREFIX_POSITIONS`` continuous; Task 24b — POSITION axis, distinct from
    the embedding-ROW axis, which is unchanged). The head projects to the sub-F range
    only (shared across backbones).
    """
    if name in _AR_FAMILY:
        return MicroAR(
            MicroARConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                n_subf_vocab=subf_vocab_size(),
                n_cond=conditioning_id_span(),  # embedding rows above sub-F for value-bearing ids
                # positions: budget + 9 id positions + 1 continuous character position
                max_len=cfg.max_len + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS,
                # Task 24b: the continuous carrier (channels, not rows/positions) ...
                n_char_stats=CHARACTER_STAT_CHANNELS,
                # ... overwriting the placeholder at the position after the 9 id slots
                char_position=CONDITIONING_PREFIX_LEN,
            )
        )
    if name in _MAMBA_FAMILY:
        # GPU verify-before-lock PASSED 2026-06-17 (A100; fused kernel numerics vs the
        # pure-PyTorch reference, fwd+bwd; reports/2026-06-17-mamba-gpu-half-probe.json).
        # The mamba env-lock now guards mamba-backbone construction; the backbone MODULE
        # is the downstream build — this raise lifts when MambaHybrid lands.
        assert_mamba_env_locked()
        raise BackboneNotYetBuilt(
            f"{name!r}: mamba-ssm kernel lock VERIFIED on A100 (2026-06-17); the backbone "
            f"module is the downstream build (mamba env-lock enforced above)"
        )
    if name in _DIFFUSION_FAMILY:
        raise BackboneNotYetBuilt(
            f"{name!r} is built behind the Task-5 gate (no mamba dependency)"
        )
    raise ValueError(f"unknown backbone {name!r}; expected one of {_AR_FAMILY + _GATED}")
