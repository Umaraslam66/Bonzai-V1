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

DECISION (slice v1, tier-2 / model-side encoding): the conditioning prefix is the
8 field-SLOT tokens ``[CONDITIONING_ID_BASE .. +8)`` -- value-agnostic. The locked
id-block is one id per FIELD (``n_cond=8``) and the model takes a single id sequence
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        positions = torch.arange(t, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        h = self.blocks(x, mask=causal, is_causal=True)
        return self.head(h)

    def training_loss(
        self,
        ids: torch.Tensor,
        *,
        prefix_len: torch.Tensor,
        seq_len: torch.Tensor | None = None,
    ) -> LossOut:
        """Next-token CE over the cell-token positions; conditioning prefix masked.

        ``ids`` is ``[conditioning prefix | cell tokens | (right-padding)]``.
        ``prefix_len[b]`` is the number of conditioning tokens for example ``b``;
        targets ``< prefix_len-1`` (the conditioning tokens themselves) are masked,
        while target ``prefix_len-1`` (first cell token, predicted FROM conditioning)
        onward is supervised. ``seq_len[b]`` (optional) is the real length incl. the
        prefix; targets at index ``>= seq_len-1`` are right-padding and also masked.
        When ``seq_len`` is None, every example is assumed full-length (no padding).
        """
        logits = self(ids)[:, :-1]  # logits[:, i] predicts token at position i+1
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
