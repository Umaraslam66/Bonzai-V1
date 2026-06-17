"""Param-matched per-scale bake-off table (spec §3.3, §8 — a VERIFIED gate).

The bake-off's validity condition is that ARCHITECTURE is the only variable: at each scale
``transformer-ar`` and ``mamba-hybrid`` must land at the SAME parameter count (NOT the same
depth). A Jamba 7:1 interleave layer has different per-layer params than a pure-transformer
layer, so equal-depth would be unequal-capacity — confounding architecture with capacity.
This table param-MATCHES the two backbones at each scale within a 2% tolerance, counted on
the ACTUAL built models (``tests/models/test_bakeoff_param_match.py``).

The two backbones share ``d_model`` at every scale, so the shared scaffold (embedding +
conditioning id-block, positions, char-carrier, sub-F head) is IDENTICAL between them and
only the mixer stack differs: a Jamba layer ~= 0.5x a transformer layer in params, so
mamba-hybrid runs ~1.7x the depth at ``transformer_every=7`` to match. Derived by
``scripts/tune_bakeoff_scales.py`` on the unified Leonardo env (2026-06-17); measured
counts (transformer-ar vs mamba-hybrid, rel):

    30M : 30,732,260 vs 30,471,140 (0.85%)   d_model 512  / 7L  vs 12L (1 tf + 11 mamba)
    100M: 98,052,068 vs 98,849,252 (0.81%)   d_model 768  / 12L vs 21L (2 tf + 19 mamba)
    300M: 294,446,564 vs 294,436,324 (0.003%) d_model 1024 / 22L vs 38L (4 tf + 34 mamba)
    1B  : 1,041,823,204 vs 1,029,404,132 (1.19%) d_model 2048 / 20L vs 34L (4 tf + 30 mamba)
"""

from __future__ import annotations

from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CHARACTER_STAT_CHANNELS,
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.data.training.datamodule import DEFAULT_MAX_CELL_TOKENS
from cfm.models.backbone import subf_vocab_size
from cfm.models.mamba_hybrid import MambaHybridConfig
from cfm.models.micro_ar import MicroARConfig

#: The four bake-off scales (nominal labels; the actual counts are the table above).
BAKEOFF_SCALES = ("30M", "100M", "300M", "1B")

# Per-scale mixer knobs (the param-matched table — verified <=2% per scale). The shared
# scaffold fields (vocab, conditioning span, positional capacity, char-carrier) come from
# the authoritative constants below, NOT hardcoded, so the table stays faithful if they move.
_TRANSFORMER_AR: dict[str, dict[str, int]] = {
    "30M": {"d_model": 512, "n_layers": 7, "n_heads": 8},
    "100M": {"d_model": 768, "n_layers": 12, "n_heads": 12},
    "300M": {"d_model": 1024, "n_layers": 22, "n_heads": 16},
    "1B": {"d_model": 2048, "n_layers": 20, "n_heads": 32},
}
_MAMBA_HYBRID: dict[str, dict[str, int]] = {
    "30M": {"d_model": 512, "n_layers": 12, "n_heads": 8, "transformer_every": 7},
    "100M": {"d_model": 768, "n_layers": 21, "n_heads": 12, "transformer_every": 7},
    "300M": {"d_model": 1024, "n_layers": 38, "n_heads": 16, "transformer_every": 7},
    "1B": {"d_model": 2048, "n_layers": 34, "n_heads": 32, "transformer_every": 7},
}


def _shared_kwargs() -> dict[str, int]:
    """Scaffold fields identical for BOTH backbones (the bake-off holds them constant):
    sub-F head dim, conditioning id-block, positional capacity (cell budget + the 9 id
    positions + 1 continuous character position), and the char-carrier channels/position."""
    return {
        "n_subf_vocab": subf_vocab_size(),
        "n_cond": conditioning_id_span(),
        "max_len": DEFAULT_MAX_CELL_TOKENS + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS,
        "n_char_stats": CHARACTER_STAT_CHANNELS,
        "char_position": CONDITIONING_PREFIX_LEN,
    }


def build_pair_for_scale(scale: str) -> tuple[MicroARConfig, MambaHybridConfig]:
    """Return the param-matched ``(MicroARConfig, MambaHybridConfig)`` pair for ``scale``.

    Both configs carry the same shared scaffold fields and differ only in the mixer knobs
    (the table above). The two backbones land within 2% of each other in actual built
    parameter count (the verified gate).
    """
    if scale not in BAKEOFF_SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {BAKEOFF_SCALES}")
    shared = _shared_kwargs()
    tf = MicroARConfig(**_TRANSFORMER_AR[scale], **shared)
    mamba = MambaHybridConfig(**_MAMBA_HYBRID[scale], **shared)
    return tf, mamba
