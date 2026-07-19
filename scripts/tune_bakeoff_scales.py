"""Search the param-matched per-scale bake-off table (Task 5, spec §3.3).

For each nominal scale {30M, 100M, 300M, 1B} we fix a transformer-ar config near the
target, count its ACTUAL built parameters, then search the mamba-hybrid mixer knobs
(n_layers, transformer_every, d_model) for the config whose ACTUAL built parameter count
lands closest to the transformer's — within the 2% tolerance the bake-off validity
condition requires (param-matched, NOT layer-matched; equal-depth would be unequal
capacity and confound architecture with capacity).

Counting is on the REAL constructed module (never an eyeballed formula); the analytic
estimate only SEEDS the integer-n_layers search so we construct ~7 candidates per scale
instead of dozens (cheap on CPU — construction is CPU-safe; only Mamba.forward needs GPU).

Run on the Leonardo mamba env (probe venv + the 3 mamba import preconditions). Emits a
table to copy into src/cfm/models/bakeoff_scales.py.
"""

from __future__ import annotations

import gc

import torch

from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CHARACTER_STAT_CHANNELS,
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.models.backbone import subf_vocab_size
from cfm.models.mamba_hybrid import MambaHybrid, MambaHybridConfig, _interleave_positions
from cfm.models.micro_ar import MicroAR, MicroARConfig

# Shared scaffold dims (identical for both backbones at a given d_model).
N_VOCAB = subf_vocab_size()
N_COND = conditioning_id_span()
MAX_LEN = 13_312 + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS  # positional capacity
N_CHAR = CHARACTER_STAT_CHANNELS
CHAR_POS = CONDITIONING_PREFIX_LEN

# transformer-ar sizings near each nominal target (head_dim = d_model / n_heads = 64).
TF_CONFIGS = {
    "30M": dict(d_model=512, n_layers=7, n_heads=8),
    "100M": dict(d_model=768, n_layers=12, n_heads=12),
    "300M": dict(d_model=1024, n_layers=22, n_heads=16),
    "1B": dict(d_model=2048, n_layers=20, n_heads=32),
}

TOL = 0.02


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _tf_params(d_model: int, n_layers: int, n_heads: int) -> int:
    m = MicroAR(
        MicroARConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_subf_vocab=N_VOCAB,
            n_cond=N_COND,
            max_len=MAX_LEN,
            n_char_stats=N_CHAR,
            char_position=CHAR_POS,
        )
    )
    n = _params(m)
    del m
    gc.collect()
    return n


def _mamba_params(d_model: int, n_layers: int, n_heads: int, every: int) -> int:
    m = MambaHybrid(
        MambaHybridConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_subf_vocab=N_VOCAB,
            n_cond=N_COND,
            max_len=MAX_LEN,
            n_char_stats=N_CHAR,
            char_position=CHAR_POS,
            transformer_every=every,
        )
    )
    n = _params(m)
    del m
    gc.collect()
    return n


# Analytic per-layer / shared estimates (seed the search ONLY; real counts decide).
def _tf_layer(d: int) -> float:
    return 12.0 * d * d + 13 * d


def _mamba_layer(d: int) -> float:
    return 6.25 * d * d + 112 * d  # incl. the LayerNorm wrapper (+2d)


def _shared(d: int) -> float:
    return (N_VOCAB + N_COND) * d + MAX_LEN * d + (N_CHAR + 1) * d + d * N_VOCAB + N_VOCAB


def _seed_n_layers(p_tf: int, d_model: int, every: int) -> int:
    """Pick n so the analytic mamba total (shared + stack) is closest to p_tf."""
    best_n, best_err = 1, float("inf")
    for n in range(1, 400):
        layout = _interleave_positions(n, every)
        n_tf = sum(layout)
        stack = n_tf * _tf_layer(d_model) + (n - n_tf) * _mamba_layer(d_model)
        err = abs(_shared(d_model) + stack - p_tf)
        if err < best_err:
            best_err, best_n = err, n
    return best_n


def main() -> None:
    print(
        f"shared dims: N_VOCAB={N_VOCAB} N_COND={N_COND} MAX_LEN={MAX_LEN} "
        f"N_CHAR={N_CHAR} CHAR_POS={CHAR_POS}\n"
    )
    for scale, tf in TF_CONFIGS.items():
        d, n_heads = tf["d_model"], tf["n_heads"]
        p_tf = _tf_params(d, tf["n_layers"], n_heads)
        print(
            f"=== {scale}: transformer-ar d_model={d} n_layers={tf['n_layers']} "
            f"n_heads={n_heads} -> {p_tf:,} params ==="
        )
        best = None
        # Cleanest first: equal width, 7:1. Widen (every, d_model) only if no <=2% point.
        plans = [(d, 7)]
        plans += [(d, e) for e in (6, 8)]
        plans += [(d + dd, e) for dd in (-64, 64, -128, 128) for e in (6, 7, 8) if d + dd > 0]
        for d_m, every in plans:
            heads_m = d_m // 64
            seed = _seed_n_layers(p_tf, d_m, every)
            for n in range(max(1, seed - 3), seed + 4):
                p_m = _mamba_params(d_m, n, heads_m, every)
                rel = abs(p_m - p_tf) / p_tf
                if best is None or rel < best["rel"]:
                    best = dict(
                        d_model=d_m, n_layers=n, n_heads=heads_m, every=every, p_m=p_m, rel=rel
                    )
            if best["rel"] <= TOL and (d_m, every) == (d, 7):
                break  # accept the cleanest equal-width / 7:1 match
        layout = _interleave_positions(best["n_layers"], best["every"])
        print(
            f"    BEST mamba-hybrid: d_model={best['d_model']} n_layers={best['n_layers']} "
            f"n_heads={best['n_heads']} transformer_every={best['every']} "
            f"({sum(layout)} tf + {len(layout) - sum(layout)} mamba)"
        )
        print(
            f"    params={best['p_m']:,}  rel={best['rel']:.3%}  "
            f"{'PASS <=2%' if best['rel'] <= TOL else 'FAIL >2%'}\n"
        )


if __name__ == "__main__":
    main()
