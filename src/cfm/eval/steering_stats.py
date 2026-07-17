"""Steering-probe SCORING CORE (spec: docs/superpowers/specs/2026-07-17-steering-probe.md).

Orchestrator-authored (not subagent-authored): paired per-seed deltas, exact two-sided
binomial sign test, matched-pairs rank-biserial effect size, and the PRE-REGISTERED verdict
rule. Torch-free pure Python so the verdict layer is auditable in isolation.

Input rows (produced by ``scripts/steering_probe_analyze.py``):
``{ckpt_id, contrast, arm ('A'|'B'), gen_seed, metrics: {name: float | None}}``.
A ``None`` metric (e.g. undecodable cell) drops that PAIR (counted, never silently).

Verdict rule (pre-registered in the spec — do not tune post hoc):
a contrast STEERS iff, on >= MIN_CKPTS of the checkpoints present, the exact sign-test
p < ALPHA AND the majority direction of (arm B - arm A) matches the registered
expectation. The probe as a whole is VALID iff the positive control C4 STEERS; with an
invalid probe no macro conclusion is drawn (a gate must be able to fail).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from statistics import median
from typing import Any

#: Pre-registered primary metric and expected sign of (arm B - arm A) per contrast.
PRIMARY: dict[str, tuple[str, int]] = {
    "C1": ("total_road_length", +1),
    "C2": ("n_buildings", +1),
    "C3": ("n_features", +1),
    "C4": ("n_buildings", +1),
    "C5": ("total_road_length", +1),
}

ALPHA = 0.01
MIN_CKPTS = 2  # "on >= 2 of 3 checkpoints" (spec); evaluated against ckpts present


def sign_test_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided binomial sign test at p=0.5; ties are excluded by the caller."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def rank_biserial(deltas: Sequence[float]) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-); zeros excluded, |d| ties
    receive average ranks. +1 = all-positive deltas, -1 = all-negative, 0 = balanced/empty."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0
    by_abs = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[by_abs[j + 1]]) == abs(nz[by_abs[i]]):
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[by_abs[k]] = avg_rank
        i = j + 1
    w_pos = sum(r for r, d in zip(ranks, nz, strict=True) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz, strict=True) if d < 0)
    return (w_pos - w_neg) / (w_pos + w_neg)


def paired_deltas(
    rows: list[dict[str, Any]], *, contrast: str, ckpt_id: str, metric: str
) -> tuple[list[float], int]:
    """(arm B - arm A) per gen_seed. Returns (deltas, n_dropped_pairs). A pair drops when
    either arm's metric is missing/None. Duplicate (arm, seed) rows are a data defect ->
    ValueError (fail loud, never average silently)."""
    arm_vals: dict[str, dict[int, float | None]] = {"A": {}, "B": {}}
    for r in rows:
        if r["contrast"] != contrast or r["ckpt_id"] != ckpt_id:
            continue
        seed, arm = r["gen_seed"], r["arm"]
        if seed in arm_vals[arm]:
            raise ValueError(f"duplicate row: {contrast}/{ckpt_id}/{arm}/seed={seed}")
        arm_vals[arm][seed] = r["metrics"].get(metric)
    deltas: list[float] = []
    dropped = 0
    for seed in sorted(set(arm_vals["A"]) | set(arm_vals["B"])):
        va = arm_vals["A"].get(seed)
        vb = arm_vals["B"].get(seed)
        if va is None or vb is None:
            dropped += 1
            continue
        deltas.append(vb - va)
    return deltas, dropped


def _ckpt_cell(deltas: list[float], dropped: int, expected_sign: int) -> dict[str, Any]:
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_tie = len(deltas) - n_pos - n_neg
    p = sign_test_p(n_pos, n_neg)
    majority = 1 if n_pos > n_neg else (-1 if n_neg > n_pos else 0)
    return {
        "n_pairs": len(deltas),
        "n_dropped_pairs": dropped,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tie": n_tie,
        "median_delta": median(deltas) if deltas else None,
        "p_sign": p,
        "rank_biserial": rank_biserial(deltas),
        "passes": p < ALPHA and majority == expected_sign,
    }


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered verdict rule to per-cell metric rows.

    Returns {contrasts: {cid: {metric, expected_sign, per_ckpt, n_pass, steers}},
    probe_valid, macro_steers, product_steers}. ``macro_steers`` = any of C1/C2/C3 STEERS;
    ``product_steers`` = C5 STEERS (macro moves generation without target-derived char).
    Both are None when the probe is invalid (C4 control failed): no conclusion, not "no"."""
    ckpt_ids = sorted({r["ckpt_id"] for r in rows})
    contrasts: dict[str, Any] = {}
    for cid, (metric, expected) in PRIMARY.items():
        per_ckpt = {}
        for ck in ckpt_ids:
            deltas, dropped = paired_deltas(rows, contrast=cid, ckpt_id=ck, metric=metric)
            if deltas or dropped:
                per_ckpt[ck] = _ckpt_cell(deltas, dropped, expected)
        n_pass = sum(1 for c in per_ckpt.values() if c["passes"])
        contrasts[cid] = {
            "metric": metric,
            "expected_sign": expected,
            "per_ckpt": per_ckpt,
            "n_pass": n_pass,
            "steers": n_pass >= MIN_CKPTS,
        }
    probe_valid = contrasts["C4"]["steers"]
    return {
        "contrasts": contrasts,
        "probe_valid": probe_valid,
        "macro_steers": (
            (contrasts["C1"]["steers"] or contrasts["C2"]["steers"] or contrasts["C3"]["steers"])
            if probe_valid
            else None
        ),
        "product_steers": contrasts["C5"]["steers"] if probe_valid else None,
    }
