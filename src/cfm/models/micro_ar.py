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

#: ignore_index for cross-entropy on masked (conditioning-prefix) target positions.
_IGNORE = -100


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


@dataclass
class LossOut:
    loss: torch.Tensor
    n_supervised_positions: int


class MicroAR(nn.Module):
    """Decoder-only AR transformer. Embedding = sub-F vocab + conditioning id-block;
    the head projects to ``n_subf_vocab`` only."""

    def __init__(self, cfg: MicroARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.n_subf_vocab + cfg.n_cond, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        # Task 24b: the continuous character carrier — ONE prefix position whose input
        # embedding is a Linear projection of char_stats, never a table lookup (the
        # token-id axis is untouched: no new vocabulary ids).
        if cfg.n_char_stats > 0:
            if cfg.char_position is None:
                raise ValueError(
                    "MicroARConfig: n_char_stats > 0 requires char_position (the prefix "
                    "position the projection overwrites)"
                )
            self.char_proj: nn.Linear | None = nn.Linear(cfg.n_char_stats, cfg.d_model)
        else:
            self.char_proj = None
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
        self.head = nn.Linear(cfg.d_model, cfg.n_subf_vocab)  # sub-F range only

    def forward(self, ids: torch.Tensor, char_stats: torch.Tensor | None = None) -> torch.Tensor:
        """``char_stats`` (Task 24b): float ``[B, n_char_stats]``. When the model
        carries the character channel (``n_char_stats > 0``) it is REQUIRED — a
        char-built model must never silently train on the placeholder id's token
        embedding; the mismatch is loud in BOTH directions. The projection
        OVERWRITES the token embedding at ``char_position`` (the placeholder id
        there is inert); the positional embedding stays additive."""
        t = ids.shape[1]
        emb = self.embed(ids)
        if self.char_proj is None:
            if char_stats is not None:
                raise ValueError(
                    "char_stats given but this model has no character carrier "
                    "(n_char_stats=0) — mismatched wiring, refusing"
                )
        else:
            if char_stats is None:
                raise ValueError(
                    f"this model carries a continuous character position "
                    f"(n_char_stats={self.cfg.n_char_stats}) but forward() got "
                    f"char_stats=None — refusing the silent placeholder-embedding regime"
                )
            p = self.cfg.char_position
            assert p is not None  # constructor invariant: n_char_stats > 0 => set
            if t <= p:
                raise ValueError(
                    f"sequence length {t} does not reach the character position {p} — "
                    f"the prefix must include the placeholder slot"
                )
            emb[:, p, :] = self.char_proj(char_stats.to(emb.dtype))
        positions = torch.arange(t, device=ids.device)
        x = emb + self.pos(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        h = self.blocks(x, mask=causal, is_causal=True)
        return self.head(h)

    def training_loss(
        self,
        ids: torch.Tensor,
        *,
        prefix_len: torch.Tensor,
        seq_len: torch.Tensor | None = None,
        char_stats: torch.Tensor | None = None,
    ) -> LossOut:
        """Next-token CE over the cell-token positions; conditioning prefix masked.

        ``ids`` is ``[conditioning prefix | cell tokens | (right-padding)]``.
        ``prefix_len[b]`` is the number of conditioning tokens for example ``b``
        (10 since Task 24b: 9 id positions + the continuous character position);
        targets ``< prefix_len-1`` (the conditioning tokens themselves) are masked,
        while target ``prefix_len-1`` (first cell token, predicted FROM conditioning)
        onward is supervised. ``seq_len[b]`` (optional) is the real length incl. the
        prefix; targets at index ``>= seq_len-1`` are right-padding and also masked.
        When ``seq_len`` is None, every example is assumed full-length (no padding).
        ``char_stats`` threads to ``forward`` (required iff the model carries it).
        """
        logits = self(ids, char_stats=char_stats)[:, :-1]  # logits[:, i] predicts pos i+1
        target = ids[:, 1:].clone()
        lens = seq_len.tolist() if seq_len is not None else [None] * ids.shape[0]
        for b, pl in enumerate(prefix_len.tolist()):
            target[b, : pl - 1] = _IGNORE  # mask conditioning-token targets
            if lens[b] is not None:
                target[b, lens[b] - 1 :] = _IGNORE  # mask right-padding targets
        n = int((target != _IGNORE).sum())
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=_IGNORE,
        )
        return LossOut(loss=loss, n_supervised_positions=n)
