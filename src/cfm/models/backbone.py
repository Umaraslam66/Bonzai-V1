"""Swappable-backbone abstraction (Phase-2 bake-off Task 7; §9).

The bake-off compares THREE backbones on one ruler. That only holds if they differ
ONLY in their sequence-mixing layers (+ diffusion's quarantined loss/generation/mask)
and SHARE the embedding / value-bearing conditioning builder / sub-F vocab head / eval
content -- by IDENTITY, not "equal output today". This module is the shared factory and
the identity anchors a Gate-6 test asserts against.

mamba-hybrid and discrete-diffusion are built BEHIND the Task-5 mamba-ssm
verify-before-lock gate: their imports may not survive the locked torch stack, so
``build_backbone`` raises ``BackboneNotYetBuilt`` for them until Task 5 settles. The
transformer-ar backbone (``MicroAR``) exists today and is sized here to cover the
value-bearing conditioning id span (Task 6 integration: the embedding must span
``n_subf_vocab + conditioning_id_span()``).
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

_AR_FAMILY = ("transformer-ar",)
_GATED = ("mamba-hybrid", "discrete-diffusion")


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
    if name in _GATED:
        raise BackboneNotYetBuilt(
            f"{name!r} is built behind the Task-5 mamba-ssm verify-before-lock gate "
            f"(its import may not survive the locked torch stack)"
        )
    raise ValueError(f"unknown backbone {name!r}; expected one of {_AR_FAMILY + _GATED}")
