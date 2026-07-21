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

from collections.abc import Sequence
from typing import Any

import torch

# try_decode_block MOVED to the torch-free decoder (Task-5 review 2026-07-20) and
# re-exported here unchanged so this module's existing importers keep working.
from cfm.data.sub_f.decoder import decode_feature, try_decode_block
from cfm.data.sub_f.vocab import CELL_END_TOKEN_ID
from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CONDITIONING_PREFIX_LEN,
)
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
    model: MicroAR,
    *,
    prefix: list[int],
    max_new: int,
    seed: int,
    char_stats: Sequence[float] | None = None,
) -> list[int]:
    """Autoregressively sample ``max_new`` cell tokens after the conditioning ``prefix``.

    Seeded via a dedicated ``torch.Generator`` so the same seed yields identical
    tokens (reproducibility / resume-safe). Returns only the generated tail (the
    conditioning prefix is stripped).

    ``char_stats`` (Task 24b): the per-cell continuous character vector; required by
    a char-built model (its forward refuses None — fail-loud), threaded into EVERY
    step's forward so the carrier conditions the whole generation. When given,
    ``prefix`` MUST be the 10-position Task-24b layout (9 value-bearing conditioning
    ids + the character placeholder slot) — a pre-24b 9-id prefix would put the
    projection overwrite on the first CELL token, so the layout is checked loudly."""
    if char_stats is not None:
        expected = CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS
        if len(prefix) != expected:
            raise ValueError(
                f"char_stats given but prefix has {len(prefix)} ids — a char-built "
                f"generation requires the {expected}-position Task-24b layout "
                f"({CONDITIONING_PREFIX_LEN} value-bearing conditioning ids + "
                f"{CHARACTER_PREFIX_POSITIONS} character placeholder slot); a pre-24b "
                f"prefix would silently hand the projection a cell-token position"
            )
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([prefix], dtype=torch.long, device=device)
        cs = (
            torch.tensor([list(char_stats)], dtype=torch.float32, device=device)
            if char_stats is not None
            else None
        )
        for _ in range(max_new):
            logits = model(ids, char_stats=cs)[:, -1]  # (1, n_subf_vocab) -- sub-F range only
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
            # cell-EOS: the model self-terminates by emitting <cell_end>=260. Stop there
            # instead of running to the max_new cap. The 260 is KEPT in the returned tail
            # (mirrors the training sequences `...510, 260`; split_cell_into_features drops
            # it after the last 510, so decode is unaffected).
            if int(nxt) == CELL_END_TOKEN_ID:
                break
    finally:
        model.train(was_training)
    return ids[0, len(prefix) :].tolist()


def decode_cell_to_geojson(cell_tokens: list[int]) -> list[dict[str, Any]]:
    """Split a generated cell into feature blocks and decode each via the SEALED
    decoder, skipping blocks that fail to decode. Never raises on well-formed input."""
    geoms: list[dict[str, Any]] = []
    for block in split_cell_into_features(cell_tokens):
        geom = try_decode_block(block)
        if geom is not None:
            geoms.append(geom)
    return geoms
