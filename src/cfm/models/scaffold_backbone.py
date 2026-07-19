"""Shared scaffold for the bake-off backbones (spec §3.2).

Every bake-off candidate (``transformer-ar`` = ``MicroAR``, ``mamba-hybrid``, ...)
must differ ONLY in its sequence-mixing layer and SHARE everything else: the
embedding table (sub-F vocab + conditioning id-block), the additive positional
embedding, the Task-24b continuous character carrier, the sub-F-range output head,
and the next-token AR loss. This module lifts that shared scaffold into a base
class whose one overridable seam is ``_mix`` — the per-backbone mixer.

RNG-ORDER CONTRACT (load-bearing — bit-identity gate, spec §3.2 / protocol Gate-6):
the pre-refactor ``MicroAR.__init__`` drew the seeded RNG in the order
``embed -> pos -> char_proj -> mixer -> head``. The head is built LAST, AFTER the
mixer. To preserve that order through the base/subclass split, the base builds
``embed -> pos -> char_proj`` in ``__init__`` and does NOT build the head; the
subclass builds its mixer and THEN calls ``self._build_head()``. Reordering those
two draws (e.g. a base that builds the head before the subclass's mixer) silently
changes the random init and breaks bit-identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#: ignore_index for cross-entropy on masked (conditioning-prefix) target positions.
_IGNORE = -100


@dataclass
class LossOut:
    loss: torch.Tensor
    n_supervised_positions: int


class ScaffoldBackbone(nn.Module):
    """Shared scaffold for the bake-off backbones (embedding + conditioning id-block,
    additive positions, Task-24b character carrier, sub-F-range head, AR loss). The
    sequence mixer is the single divergence point: subclasses override ``_mix``.

    Subclass contract: build the mixer in ``__init__`` AFTER calling
    ``super().__init__(...)``, then call ``self._build_head()`` LAST — this preserves
    the pre-refactor RNG draw order ``embed -> pos -> char_proj -> mixer -> head``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_subf_vocab: int,
        n_cond: int,
        max_len: int,
        n_char_stats: int = 0,
        char_position: int | None = None,
    ) -> None:
        # NOTE: dropout is intentionally NOT a base parameter — it is a mixer concern.
        # Each subclass threads its own dropout into its mixer layers (the scaffold has
        # no dropout-bearing layers). A future backbone owns dropout in its own __init__.
        super().__init__()
        # RNG-order contract: draw embed -> pos -> char_proj here; the head is built
        # LATER by the subclass (via _build_head) AFTER its mixer, so the head's init
        # draw stays after the mixer's exactly as in the pre-refactor MicroAR.
        self.embed = nn.Embedding(n_subf_vocab + n_cond, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        # Task 24b: the continuous character carrier — ONE prefix position whose input
        # embedding is a Linear projection of char_stats, never a table lookup (the
        # token-id axis is untouched: no new vocabulary ids).
        if n_char_stats > 0:
            if char_position is None:
                raise ValueError(
                    "ScaffoldBackbone: n_char_stats > 0 requires char_position (the prefix "
                    "position the projection overwrites)"
                )
            self.char_proj: nn.Linear | None = nn.Linear(n_char_stats, d_model)
        else:
            self.char_proj = None
        # Plain (non-module) attributes — these draw no RNG, so their placement is free.
        self._n_char_stats = n_char_stats
        self._char_position = char_position
        self._n_subf_vocab = n_subf_vocab
        self._d_model = d_model

    def _build_head(self) -> None:
        """Build the sub-F-range output head. The subclass calls this AFTER building
        its mixer, to preserve the pre-refactor RNG draw order (the head's random init
        must come after the mixer's — see this module's RNG-ORDER CONTRACT)."""
        self.head = nn.Linear(self._d_model, self._n_subf_vocab)  # sub-F range only

    def _mix(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        """The one per-backbone divergence point: mix the sequence. Subclasses override
        (transformer-ar = causal self-attention; mamba-hybrid = SSM scan; ...)."""
        raise NotImplementedError

    def _input_embeddings(self, ids: torch.Tensor, char_stats: torch.Tensor | None) -> torch.Tensor:
        """Token-embedding lookup + Task-24b character overwrite + additive positions.

        ``char_stats`` (Task 24b): float ``[B, n_char_stats]``. When the model carries
        the character channel (``n_char_stats > 0``) it is REQUIRED — a char-built model
        must never silently train on the placeholder id's token embedding; the mismatch
        is loud in BOTH directions. The projection OVERWRITES the token embedding at
        ``char_position`` (the placeholder id there is inert); the positional embedding
        stays additive."""
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
                    f"(n_char_stats={self._n_char_stats}) but forward() got "
                    f"char_stats=None — refusing the silent placeholder-embedding regime"
                )
            p = self._char_position
            assert p is not None  # constructor invariant: n_char_stats > 0 => set
            if t <= p:
                raise ValueError(
                    f"sequence length {t} does not reach the character position {p} — "
                    f"the prefix must include the placeholder slot"
                )
            emb[:, p, :] = self.char_proj(char_stats.to(emb.dtype))
        positions = torch.arange(t, device=ids.device)
        return emb + self.pos(positions)

    def forward(self, ids: torch.Tensor, char_stats: torch.Tensor | None = None) -> torch.Tensor:
        x = self._input_embeddings(ids, char_stats)
        t = ids.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        return self.head(self._mix(x, causal))

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
