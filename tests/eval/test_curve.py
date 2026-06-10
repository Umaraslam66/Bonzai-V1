"""Scaling-curve fit + extrapolation + §13 tie-break (Phase-2 bake-off Task 13)."""

from __future__ import annotations

from cfm.eval.curve import (
    TIEBREAK_BACKBONE,
    extrapolate,
    fit_scaling_curve,
    pick_winner,
    structural_check_ok,
)


def test_fit_returns_point_estimate_and_confidence_interval() -> None:
    pts = [(0.14, 0.40), (1.6, 0.30), (14.6, 0.22), (160.0, 0.18)]  # (node-h, KS) lower=better
    fit = fit_scaling_curve(pts, n_bootstrap=200)
    lo, hi = extrapolate(fit, target_node_h=500.0)
    assert lo < hi  # a confidence interval, not a bare point
    assert structural_check_ok(fit) is True  # improving curve


def test_structural_check_rejects_non_monotonic_fit() -> None:
    bad = [(0.14, 0.30), (1.6, 0.35), (14.6, 0.20), (160.0, 0.50)]  # non-improving / noisy
    assert structural_check_ok(fit_scaling_curve(bad, n_bootstrap=200)) is False


def test_fit_is_deterministic_across_calls() -> None:
    pts = [(0.14, 0.40), (1.6, 0.30), (14.6, 0.22), (160.0, 0.18)]
    a = extrapolate(fit_scaling_curve(pts, n_bootstrap=200), target_node_h=500.0)
    b = extrapolate(fit_scaling_curve(pts, n_bootstrap=200), target_node_h=500.0)
    assert a == b  # seeded bootstrap -> reproducible decision


def test_tiebreak_is_transformer_ar_when_extrapolated_cis_overlap() -> None:
    overlapping = {
        "transformer-ar": (0.17, 0.21),
        "mamba-hybrid": (0.18, 0.22),
        "discrete-diffusion": (0.30, 0.34),
    }
    assert pick_winner(overlapping) == TIEBREAK_BACKBONE == "transformer-ar"


def test_clear_winner_is_picked_when_cis_separate() -> None:
    separated = {
        "transformer-ar": (0.30, 0.33),
        "mamba-hybrid": (0.17, 0.20),
        "discrete-diffusion": (0.40, 0.44),
    }
    assert pick_winner(separated) == "mamba-hybrid"


def test_tiebreak_even_when_a_non_simplest_backbone_leads_but_overlaps() -> None:
    # mamba leads on midpoint but its CI overlaps transformer-ar's -> no separation -> tie-break
    overlapping = {"mamba-hybrid": (0.18, 0.24), "transformer-ar": (0.20, 0.26)}
    assert pick_winner(overlapping) == "transformer-ar"
