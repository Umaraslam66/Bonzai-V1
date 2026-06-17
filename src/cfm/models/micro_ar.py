"""Toy decoder-only micro-generator (spec §7).

The simplest building block shared by bake-off candidates 1-3: a decoder-only
autoregressive transformer trained with next-token cross-entropy.

Two locked invariants:
  * The embedding table spans the sub-F vocab PLUS the appended conditioning
    id-block (``n_subf_vocab + n_cond``) so a prepended conditioning token is a
    valid INPUT id. The output head projects to ``n_subf_vocab`` ONLY -- the model
    never predicts a conditioning id (conditioning is given, not generated).
  * ``training_loss`` masks the conditioning-prefix target positions: the prefix is
    GIVEN. Supervised targets are every cell-token next-token target, which includes
    the first-cell-token-from-conditioning prediction (target index ``prefix_len-1``).

Task 24b adds ONE continuous prefix position (``n_char_stats``/``char_position``):
its input embedding is ``Linear(n_char_stats -> d_model)`` of the per-cell
``char_stats`` vector, overwriting the placeholder id's token embedding at that
position. POSITION axis only -- the token-id span (``n_cond``) is unchanged.

DECISION (slice v1, tier-2 / model-side encoding): the conditioning prefix is the
9 field-SLOT tokens ``[CONDITIONING_ID_BASE .. +9)`` -- value-agnostic (slot scheme
superseded by the 24a value-bearing path for production; retained for the legacy
builder). The locked
id-block is ``n_cond=conditioning_id_span()`` (= 576 embedding rows reserved above
the sub-F vocab; 9 fields x 64-id stride since Task 24a, wired in ``backbone.py``)
and the model
takes a single id sequence
with no value channel, so value-bearing conditioning is out of scope for the slice;
its informative encoding is a bake-off concern. The tier-1 conditioning VALUES
(``conditioning.conditioning_prefix_ids``) remain the schema artifact consumed by
trigger-2 identity + future conditioning-compliance scoring (out-of-slice).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Backward-compatible re-exports: the loss dataclass, the shared base, and the CE
# ignore_index all live in scaffold_backbone now (Task 3 extract-base refactor).
# Downstream code that did ``from cfm.models.micro_ar import LossOut`` (and the
# ``_IGNORE`` sentinel) keeps working unchanged.
from cfm.models.scaffold_backbone import _IGNORE, LossOut, ScaffoldBackbone

__all__ = ["_IGNORE", "LossOut", "MicroAR", "MicroARConfig", "ScaffoldBackbone"]


@dataclass(frozen=True)
class MicroARConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_subf_vocab: int  # prediction range (sealed sub-F vocab size = max sub-F id + 1)
    n_cond: int  # conditioning id-block size (input-only embedding rows)
    max_len: int
    dropout: float = 0.0
    #: Task 24b: width of the continuous character-stats vector (0 = no carrier; the
    #: hand-built fixture default). Production (backbone.build_backbone) passes
    #: CHARACTER_STAT_CHANNELS == 7. CHANNEL axis — not positions, not embedding rows.
    n_char_stats: int = 0
    #: Task 24b: the sequence POSITION whose token embedding is overwritten by
    #: Linear(n_char_stats -> d_model)(char_stats). Production passes
    #: CONDITIONING_PREFIX_LEN (9: the 10th position, after the 9 id positions).
    #: Required when n_char_stats > 0.
    char_position: int | None = None

    def __post_init__(self) -> None:
        # Cross-check both directions: char_position without a carrier would be
        # silently ignored (the carrier never builds) — loud, never silent.
        if self.n_char_stats == 0 and self.char_position is not None:
            raise ValueError(
                f"MicroARConfig: char_position={self.char_position} set while "
                f"n_char_stats=0 — the carrier never builds, so the position would "
                f"be silently ignored (mismatched wiring, refusing)"
            )


class MicroAR(ScaffoldBackbone):
    """The ``transformer-ar`` backbone: a decoder-only causal self-attention mixer on
    the shared scaffold. The shared parts (embedding = sub-F vocab + conditioning
    id-block, additive positions, Task-24b character carrier, sub-F-range head, AR
    loss) live in ``ScaffoldBackbone``; this subclass supplies only the mixer
    (``_mix``) — the one divergence point in the bake-off."""

    def __init__(self, cfg: MicroARConfig) -> None:
        super().__init__(
            d_model=cfg.d_model,
            n_subf_vocab=cfg.n_subf_vocab,
            n_cond=cfg.n_cond,
            max_len=cfg.max_len,
            n_char_stats=cfg.n_char_stats,
            char_position=cfg.char_position,
            dropout=cfg.dropout,
        )
        self.cfg = cfg
        # Mixer built AFTER the base's embed/pos/char_proj draws — then the head is
        # built LAST (below) to preserve the pre-refactor RNG draw order
        # embed -> pos -> char_proj -> mixer -> head (the bit-identity gate).
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        # norm_first (pre-norm) => the nested-tensor fast path is unused anyway.
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        # Head LAST — AFTER the mixer — to keep the RNG draw order identical to the
        # pre-refactor MicroAR (reordering this breaks the bit-identity golden gate).
        self._build_head()

    def _mix(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        return self.blocks(x, mask=causal, is_causal=True)
