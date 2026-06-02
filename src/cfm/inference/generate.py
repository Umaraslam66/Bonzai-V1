"""Inference: generate cell tokens, decode via the SEALED sub-F decoder (§3, one source).

Generation is plain autoregressive multinomial sampling (seeded -> reproducible).
The model head emits only the sub-F vocab range, so every generated token is a
valid sub-F id; the conditioning prefix is fed as INPUT and stripped from the output.

Decoding reuses the sealed splitter + decoder by REFERENCE (never reimplemented):
``split_cell_into_features`` (sub-G) carves 509/510 feature blocks; ``decode_feature``
(sub-F) decodes one block. A block that fails to decode is skipped, mirroring sub-G
``check_decodability``'s per-block try/except -- so decodability is a measured RATE,
not an exception. The caller (slice eval) gets the attempted-block count from
``split_cell_into_features`` and the decoded subset from here.
"""

from __future__ import annotations

from typing import Any

import torch

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.models.micro_ar import MicroAR

__all__ = [
    "decode_cell_to_geojson",
    "decode_feature",
    "generate_cell_tokens",
    "split_cell_into_features",
    "try_decode_block",
]


@torch.no_grad()
def generate_cell_tokens(
    model: MicroAR, *, prefix: list[int], max_new: int, seed: int
) -> list[int]:
    """Autoregressively sample ``max_new`` cell tokens after the conditioning ``prefix``.

    Seeded via a dedicated ``torch.Generator`` so the same seed yields identical
    tokens (reproducibility / resume-safe). Returns only the generated tail (the
    conditioning prefix is stripped)."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([prefix], dtype=torch.long, device=device)
        for _ in range(max_new):
            logits = model(ids)[:, -1]  # (1, n_subf_vocab) -- sub-F range only
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
    finally:
        model.train(was_training)
    return ids[0, len(prefix) :].tolist()


def try_decode_block(block: list[int]) -> dict[str, Any] | None:
    """Decode one 509/510 feature block, or ``None`` if it fails to decode.

    The robust per-block primitive (one source for "decode-or-None"), mirroring
    sub-G ``check_decodability``'s try/except so a malformed block never raises here."""
    try:
        return decode_feature(block)
    except Exception:
        return None


def decode_cell_to_geojson(cell_tokens: list[int]) -> list[dict[str, Any]]:
    """Split a generated cell into feature blocks and decode each via the SEALED
    decoder, skipping blocks that fail to decode. Never raises on well-formed input."""
    geoms: list[dict[str, Any]] = []
    for block in split_cell_into_features(cell_tokens):
        geom = try_decode_block(block)
        if geom is not None:
            geoms.append(geom)
    return geoms
