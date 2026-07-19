"""The ``mamba-hybrid`` bake-off backbone: a 7:1 Jamba-style interleave (spec §3.3).

This is bake-off candidate 2. Like every candidate it diverges from the others ONLY
in its sequence mixer (``_mix``) and SHARES everything else with ``ScaffoldBackbone``:
the embedding table (sub-F vocab + conditioning id-block), the additive positional
embedding, the Task-24b continuous character carrier, the sub-F-range output head, and
the next-token AR loss.

The mixer here is an interleave of Mamba SSM blocks and a small number of causal
self-attention blocks, in the Jamba 7:1 ratio (one transformer per group of seven
mamba layers). Mamba is a linear-time state-space scan — think of it as a recurrence
that summarises the whole left context into a fixed-size running state, so it is causal
by construction (it can only look left). The interleaved transformer layers add the
occasional global-attention "look back at any earlier token directly" that a pure SSM
lacks. ``_interleave_positions`` guarantees ``>= 1`` transformer layer is ALWAYS present
(spec §3.3) so attention is never absent even at tiny layer counts.

The Mamba shape (``d_state=16``, ``d_conv=4``, ``expand=2``) is the (b)-verified shape
from the Phase-2 mamba-lock step — do not change it without re-running that gate.

CUDA-ONLY FORWARD (verified empirically against ``mamba_ssm`` 2.3.1): ``Mamba.forward``
requires CUDA and raises on CPU even with ``use_fast_path=False``. CONSTRUCTION works on
CPU (so the construction/interleave-layout test is CPU-safe), but any forward pass must
run on GPU; the forward-shape test is gated on ``torch.cuda.is_available()``.

RNG-ORDER CONTRACT: inherited from ``ScaffoldBackbone`` — ``super().__init__`` draws
``embed -> pos -> char_proj``; this subclass builds its interleaved mixer and THEN calls
``self._build_head()`` LAST so the head's init draw stays after the mixer's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cfm.models.scaffold_backbone import ScaffoldBackbone

__all__ = ["MambaHybrid", "MambaHybridConfig"]


@dataclass(frozen=True)
class MambaHybridConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_subf_vocab: int  # prediction range (sealed sub-F vocab size = max sub-F id + 1)
    n_cond: int  # conditioning id-block size (input-only embedding rows)
    max_len: int
    #: Task 24b: width of the continuous character-stats vector (0 = no carrier).
    n_char_stats: int = 0
    #: Task 24b: the sequence POSITION whose token embedding is overwritten by
    #: Linear(n_char_stats -> d_model)(char_stats). Required when n_char_stats > 0.
    char_position: int | None = None
    #: Mixer-owned dropout for the interleaved transformer layers. The base
    #: (ScaffoldBackbone) does NOT take dropout — it is a mixer concern.
    dropout: float = 0.0
    #: Mamba SSM shape — the (b)-verified mamba-lock shape (spec §3.3).
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    #: Jamba interleave ratio: one transformer layer per `transformer_every` mamba
    #: layers (7:1). `>= 1` transformer is always kept (see _interleave_positions).
    transformer_every: int = 7

    def __post_init__(self) -> None:
        # Cross-check the config-side direction (symmetric with MicroARConfig): a
        # char_position without a carrier would be silently ignored (the carrier never
        # builds). The OTHER direction — a carrier WITHOUT char_position — is enforced
        # in ScaffoldBackbone.__init__. Loud, never silent.
        if self.n_char_stats == 0 and self.char_position is not None:
            raise ValueError(
                f"MambaHybridConfig: char_position={self.char_position} set while "
                f"n_char_stats=0 — the carrier never builds, so the position would "
                f"be silently ignored (mismatched wiring, refusing)"
            )


def _interleave_positions(n_layers: int, every: int) -> list[bool]:
    """Return the per-layer mixer layout: ``True`` = transformer layer, ``False`` = mamba.

    A transformer is placed at the last slot of each group of ``every`` + 1 layers
    (the Jamba 7:1 pattern: with ``every=7``, slots 7, 15, 23, ... are transformers).
    At small layer counts the periodic rule may select no transformer at all; in that
    case the last layer is forced to a transformer so attention is never absent
    (spec §3.3 — ``>= 1`` transformer always)."""
    is_tf = [((i + 1) % (every + 1) == 0) for i in range(n_layers)]
    if not any(is_tf):
        is_tf[-1] = True
    return is_tf


class _MambaResidualBlock(nn.Module):
    """Pre-norm residual wrapper around a Mamba mixer.

    Raw ``mamba_ssm`` ``Mamba.forward`` returns ONLY the mixer output — no residual and
    no norm (mamba_ssm keeps the residual stream in its separate ``Block``). The
    interleave's transformer layers (``TransformerEncoderLayer(norm_first=True)``) already
    carry their own pre-norm + residual internally, so without this wrapper the mamba
    layers would be bare transforms (``x = mamba(x)``, no residual, no norm) — inconsistent
    with the transformer layers and poor for training, and NOT the Jamba structure (Jamba
    wraps every layer in residual + norm). This applies the SAME pre-norm residual form the
    transformer layers use (``x + mamba(norm(x))``) with LayerNorm, to match them. The
    LayerNorm draws no RNG at init (weight=1/bias=0), so it does not perturb construction
    order. (No final norm before the head — transformer-ar omits it too, so the backbones
    stay matched everywhere but the mixer.)"""

    def __init__(self, *, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        # Import inside __init__: mamba_ssm is env-gated (the default repo env has no
        # CUDA/mamba). Constructing a MambaHybrid is what requires the package present.
        from mamba_ssm.modules.mamba_simple import Mamba

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaHybrid(ScaffoldBackbone):
    """The ``mamba-hybrid`` backbone: a 7:1 Jamba interleave of Mamba SSM blocks and
    causal self-attention blocks on the shared scaffold. The shared parts (embedding =
    sub-F vocab + conditioning id-block, additive positions, Task-24b character carrier,
    sub-F-range head, AR loss) live in ``ScaffoldBackbone``; this subclass supplies only
    the mixer (``_mix``). The forward pass requires CUDA (Mamba kernels); construction is
    CPU-safe."""

    def __init__(self, cfg: MambaHybridConfig) -> None:
        super().__init__(
            d_model=cfg.d_model,
            n_subf_vocab=cfg.n_subf_vocab,
            n_cond=cfg.n_cond,
            max_len=cfg.max_len,
            n_char_stats=cfg.n_char_stats,
            char_position=cfg.char_position,
        )
        self.cfg = cfg
        # Mixer built AFTER the base's embed/pos/char_proj draws — then the head is
        # built LAST (below) to preserve the RNG draw order embed -> pos -> char_proj
        # -> mixer -> head (inherited bit-identity contract). Transformer layers carry
        # their own pre-norm+residual; mamba layers are wrapped in _MambaResidualBlock to
        # match (raw Mamba has neither) — so every sublayer is a residual block.
        layout = _interleave_positions(cfg.n_layers, cfg.transformer_every)
        self._is_tf = layout
        self.n_transformer_layers = sum(layout)
        self.n_mamba_layers = len(layout) - self.n_transformer_layers
        blocks: list[nn.Module] = []
        for is_tf in layout:
            if is_tf:
                blocks.append(
                    nn.TransformerEncoderLayer(
                        d_model=cfg.d_model,
                        nhead=cfg.n_heads,
                        dim_feedforward=4 * cfg.d_model,
                        dropout=cfg.dropout,
                        batch_first=True,
                        norm_first=True,
                    )
                )
            else:
                blocks.append(
                    _MambaResidualBlock(
                        d_model=cfg.d_model,
                        d_state=cfg.d_state,
                        d_conv=cfg.d_conv,
                        expand=cfg.expand,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        # Head LAST — AFTER the mixer — to keep the RNG draw order identical to the
        # scaffold contract (reordering breaks the bit-identity golden gate).
        self._build_head()

    def _mix(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        for blk, is_tf in zip(self.blocks, self._is_tf, strict=True):
            # The mamba block (_MambaResidualBlock) is causal by construction (left-to-right
            # scan) and takes only x; the interleaved transformer layer needs the explicit
            # causal mask to stay AR-correct.
            x = blk(x, src_mask=causal, is_causal=True) if is_tf else blk(x)
        return x
