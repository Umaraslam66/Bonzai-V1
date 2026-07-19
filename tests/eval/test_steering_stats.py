"""Tests for the steering-probe scoring core (orchestrator-authored, spec-pinned).

Exercises the spec's corner cases, not just the happy path: exact sign-test values,
tie handling, dropped-pair accounting, duplicate-row fail-loud, the boundary p == ALPHA
non-pass, and the C4-invalid regime where the probe must yield NO conclusion (the gate
must be able to fail — and here it demonstrably can).
"""

from __future__ import annotations

import pytest

from cfm.eval.steering_stats import (
    ALPHA,
    PRIMARY,
    judge,
    paired_deltas,
    rank_biserial,
    sign_test_p,
)


# ---------------------------------------------------------------- sign_test_p (exact values)
def test_sign_test_exact_one_sided_sweep() -> None:
    # 10/0 split: 2 * P(X >= 10 | n=10) = 2 / 1024
    assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
    # 9/1: 2 * (C(10,9) + C(10,10)) / 1024 = 22/1024
    assert sign_test_p(9, 1) == pytest.approx(22 / 1024)


def test_sign_test_symmetric_and_capped() -> None:
    assert sign_test_p(3, 17) == sign_test_p(17, 3)
    assert sign_test_p(5, 5) == 1.0  # balanced caps at 1
    assert sign_test_p(0, 0) == 1.0  # vacuous


def test_sign_test_40_pairs_regimes() -> None:
    # The probe's n=40 arm size: a 30/10 split must clear ALPHA, 24/16 must not.
    assert sign_test_p(30, 10) < ALPHA
    assert sign_test_p(24, 16) > ALPHA


# ---------------------------------------------------------------- rank_biserial
def test_rank_biserial_extremes_and_balance() -> None:
    assert rank_biserial([1.0, 2.0, 0.5]) == 1.0
    assert rank_biserial([-1.0, -3.0]) == -1.0
    assert rank_biserial([]) == 0.0
    assert rank_biserial([0.0, 0.0]) == 0.0  # zeros excluded -> empty -> 0
    # perfectly mirrored magnitudes cancel
    assert rank_biserial([2.0, -2.0, 1.0, -1.0]) == 0.0


def test_rank_biserial_tied_magnitudes_average_ranks() -> None:
    # |d| = 1,1,2: tied pair gets rank 1.5 each; W+ = 1.5+3, W- = 1.5 -> (4.5-1.5)/6
    assert rank_biserial([1.0, -1.0, 2.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------- paired_deltas
def _row(ckpt: str, contrast: str, arm: str, seed: int, **metrics: float | None) -> dict:
    return {
        "ckpt_id": ckpt,
        "contrast": contrast,
        "arm": arm,
        "gen_seed": seed,
        "metrics": metrics,
    }


def test_paired_deltas_pairs_by_seed_and_drops_missing() -> None:
    rows = [
        _row("t7", "C1", "A", 2000, total_road_length=10.0),
        _row("t7", "C1", "B", 2000, total_road_length=14.0),
        _row("t7", "C1", "A", 2001, total_road_length=8.0),  # B missing -> dropped
        _row("t7", "C1", "B", 2002, total_road_length=9.0),  # A missing -> dropped
        _row("t7", "C1", "A", 2003, total_road_length=None),  # undecodable -> dropped
        _row("t7", "C1", "B", 2003, total_road_length=7.0),
        _row("t7", "C2", "A", 2000, total_road_length=99.0),  # other contrast: ignored
    ]
    deltas, dropped = paired_deltas(rows, contrast="C1", ckpt_id="t7", metric="total_road_length")
    assert deltas == [4.0]
    assert dropped == 3


def test_paired_deltas_duplicate_row_fails_loud() -> None:
    rows = [
        _row("t7", "C1", "A", 2000, total_road_length=1.0),
        _row("t7", "C1", "A", 2000, total_road_length=2.0),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        paired_deltas(rows, contrast="C1", ckpt_id="t7", metric="total_road_length")


# ---------------------------------------------------------------- judge (verdict rule)
def _contrast_rows(
    ckpt: str, contrast: str, metric: str, deltas: list[float], base: float = 100.0
) -> list[dict]:
    rows = []
    for i, d in enumerate(deltas):
        rows.append(_row(ckpt, contrast, "A", 2000 + i, **{metric: base}))
        rows.append(_row(ckpt, contrast, "B", 2000 + i, **{metric: base + d}))
    return rows


def _full_probe_rows(effects: dict[str, dict[str, list[float]]]) -> list[dict]:
    """effects: contrast -> ckpt -> per-seed deltas on that contrast's primary metric."""
    rows: list[dict] = []
    for contrast, per_ckpt in effects.items():
        metric = PRIMARY[contrast][0]
        for ckpt, deltas in per_ckpt.items():
            rows.extend(_contrast_rows(ckpt, contrast, metric, deltas))
    return rows


STRONG_UP = [1.0] * 18 + [-1.0] * 2  # 18/2 -> p ~ 2e-4, majority +
FLAT = [1.0] * 10 + [-1.0] * 10  # p = 1.0
STRONG_DOWN = [-1.0] * 18 + [1.0] * 2  # significant but WRONG direction


def test_judge_steers_requires_two_of_three_checkpoints() -> None:
    ckpts = ["t7", "t13", "m7"]
    effects = {
        "C4": {c: STRONG_UP for c in ckpts},  # control healthy -> probe valid
        "C1": {"t7": STRONG_UP, "t13": STRONG_UP, "m7": FLAT},  # 2/3 -> steers
        "C2": {"t7": STRONG_UP, "t13": FLAT, "m7": FLAT},  # 1/3 -> not
        "C3": {c: FLAT for c in ckpts},
        "C5": {c: STRONG_UP for c in ckpts},
    }
    v = judge(_full_probe_rows(effects))
    assert v["probe_valid"] is True
    assert v["contrasts"]["C1"]["steers"] is True
    assert v["contrasts"]["C2"]["steers"] is False
    assert v["macro_steers"] is True
    assert v["product_steers"] is True


def test_judge_significant_wrong_direction_does_not_pass() -> None:
    ckpts = ["t7", "t13", "m7"]
    effects = {
        "C4": {c: STRONG_UP for c in ckpts},
        "C1": {c: STRONG_DOWN for c in ckpts},  # p tiny but direction wrong
        "C2": {c: FLAT for c in ckpts},
        "C3": {c: FLAT for c in ckpts},
        "C5": {c: FLAT for c in ckpts},
    }
    v = judge(_full_probe_rows(effects))
    assert v["contrasts"]["C1"]["n_pass"] == 0
    assert v["macro_steers"] is False


def test_judge_invalid_control_yields_no_conclusion() -> None:
    ckpts = ["t7", "t13", "m7"]
    effects = {
        "C4": {c: FLAT for c in ckpts},  # control fails -> probe UNRELIABLE
        "C1": {c: STRONG_UP for c in ckpts},  # even a strong C1 must NOT be concluded
        "C2": {c: FLAT for c in ckpts},
        "C3": {c: FLAT for c in ckpts},
        "C5": {c: STRONG_UP for c in ckpts},
    }
    v = judge(_full_probe_rows(effects))
    assert v["probe_valid"] is False
    assert v["macro_steers"] is None  # no conclusion, not "no"
    assert v["product_steers"] is None
