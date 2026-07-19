"""Aggregator (spec §2): 6-way seed-noise ranking read — gap-diff vs seed-noise.

The ranking flag must read NO_DECISIVE when the transformer-vs-mamba mean-gap difference is
within seed-noise (std across seeds), and decisive only when it clears it.
"""

from __future__ import annotations

from cfm.eval.standing.harness import aggregate


def _per_ckpt(backbone, gaps):
    return [
        {
            "meta": {"backbone": backbone},
            "perplexity_gap": {"macro_only_primary": {"gap_nats_per_token": g}},
        }
        for g in gaps
    ]


def test_no_decisive_when_diff_within_seed_noise():
    per = _per_ckpt("transformer-ar", [0.20, 0.22, 0.18]) + _per_ckpt(
        "mamba-hybrid", [0.21, 0.19, 0.23]
    )
    out = aggregate(per)
    assert "NO_DECISIVE" in out


def test_decisive_when_diff_exceeds_seed_noise():
    # tight within-backbone seeds, large between-backbone difference
    per = _per_ckpt("transformer-ar", [0.50, 0.51, 0.49]) + _per_ckpt(
        "mamba-hybrid", [0.10, 0.11, 0.09]
    )
    out = aggregate(per)
    assert "NO_DECISIVE" not in out
    assert "exceeds seed-noise" in out
