"""Saturation metric — loss-vs-steps plateau classification (spec §3, §9 tooth).

The load-bearing tooth: the plateau threshold is DERIVED FROM the final-window noise,
so the SAME modest slope must read DESCENDING under low noise and PLATEAUED under high
noise. A magic absolute threshold would fail this (it can't distinguish the regimes).
"""

from __future__ import annotations

import random

from cfm.eval.standing.saturation import classify_saturation


def _series(
    slope_per_step: float, noise_sd: float, *, n: int = 240, step: int = 500, seed: int = 0
):
    rng = random.Random(seed)
    steps = [i * step for i in range(n)]
    losses = [3.0 + slope_per_step * s + rng.gauss(0.0, noise_sd) for s in steps]
    return steps, losses


def test_clear_descending_is_descending():
    steps, losses = _series(slope_per_step=-1e-5, noise_sd=0.005)
    r = classify_saturation(steps, losses, window_steps=10000)
    assert r.classification == "DESCENDING"
    assert r.final_window_slope < 0


def test_flat_with_noise_is_plateaued():
    steps, losses = _series(slope_per_step=0.0, noise_sd=0.02, seed=3)
    r = classify_saturation(steps, losses, window_steps=10000)
    assert r.classification == "PLATEAUED"


def test_threshold_is_noise_derived_distinguishes_regimes():
    """SAME slope; low noise -> DESCENDING, high noise -> PLATEAUED; threshold scales with noise."""
    lo = classify_saturation(
        *_series(slope_per_step=-2e-6, noise_sd=0.001, seed=1), window_steps=10000
    )
    hi = classify_saturation(
        *_series(slope_per_step=-2e-6, noise_sd=0.06, seed=1), window_steps=10000
    )
    assert lo.classification == "DESCENDING"
    assert hi.classification == "PLATEAUED"
    assert hi.plateau_threshold > lo.plateau_threshold  # derived from noise, not absolute


def test_reports_units_and_window_fields():
    steps, losses = _series(slope_per_step=-1e-5, noise_sd=0.005)
    r = classify_saturation(steps, losses, window_steps=10000)
    # slope reported in nats/token per 1000 steps (signed)
    assert abs(r.final_window_slope - (-1e-5 * 1000)) < 5e-3
    assert r.final_step == steps[-1]
    assert r.final_window_noise > 0
