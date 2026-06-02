"""Absorbing-state (MDLM-family) masking for discrete diffusion (bake-off Task 8).

Training corrupts a token sequence by replacing a fraction of positions with the
absorbing ``MASK_ID``; the model learns to fill them back. ``MASK_ID`` sits ABOVE the
sub-F vocab AND the value-bearing conditioning id span (append-only -- never collides
with a real token or conditioning id). Mask sampling is SEEDED: diffusion adds randomness
the AR backbones don't have, and it must be reproducible for DDP determinism + comparable
eval noise (§10).
"""

from __future__ import annotations

import torch

from cfm.data.training.conditioning import CONDITIONING_VALUE_BASE, conditioning_id_span

#: The absorbing mask token: strictly above the sub-F vocab and the conditioning id span,
#: so it never collides with a real token (< n_subf) or a conditioning id (< span end).
MASK_ID: int = CONDITIONING_VALUE_BASE + conditioning_id_span()


def apply_absorbing_mask(
    seq: torch.Tensor, *, frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a ``frac`` of positions with ``MASK_ID`` (seeded).

    Returns ``(masked_seq, target_mask)`` where ``target_mask`` is True at the positions
    that were masked (the denoising loss is computed only there). Deterministic under a
    fixed ``seed`` (verified by the round-trip determinism test).
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    gen = torch.Generator(device=seq.device).manual_seed(seed)
    target_mask = torch.rand(seq.shape, generator=gen, device=seq.device) < frac
    masked = seq.clone()
    masked[target_mask] = MASK_ID
    return masked, target_mask
