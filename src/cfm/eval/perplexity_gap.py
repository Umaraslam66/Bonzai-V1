"""Conditional-perplexity gap calculation shell.

Computes gap = NLL_shuffled - NLL_matched on held-out micro tokens under
two conditioning prefixes, plus a per-cell sign test. The model is injected
as a callable; this module does NOT load weights. When the training scaffold
ships, callers wire `model_forward` to a real forward pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

ModelForward = Callable[..., float]
"""Injected model-forward contract for the gap calculation.

Signature: ``model_forward(*, micro_tokens, conditioning_prefix) -> float``,
where the returned float is the **NLL in nats per token**, already reduced
(averaged or otherwise aggregated) over the micro-token sequence by the
caller's model wrapper.

Design choice (ratified during Task 13 pre-dispatch audit): this module
is **tokenizer-free and torch-free by construction**. Callers wrap their
model to return scalar NLL; the eval shell never sees logits, never
imports torch, never reasons about token-level conditional probability
shape. NLL has subtleties (token vs sequence reduction, pad masking,
conditional baselines) that belong with the model — exposing logits
here would push those subtleties into the eval harness.

Spec §11.4 framing matches: "NLL on held-out micro tokens conditioned
on..." — NLL is the unit of measurement, not logits. The shipped
training scaffold's real wrapper will look like::

    def real_model_forward(*, micro_tokens, conditioning_prefix) -> float:
        logits = model(micro_tokens, conditioning_prefix)
        nll = compute_nll(logits, targets=micro_tokens)
        return float(nll.mean().item())
"""


@dataclass(frozen=True)
class PerCellNLL:
    cell_id: str
    micro_tokens: Sequence[int]
    matched_conditioning_prefix: Sequence[int]
    shuffled_conditioning_prefix: Sequence[int]


@dataclass(frozen=True)
class GapResult:
    n_cells: int
    n_positive: int  # cells where shuffled_nll > matched_nll
    fraction_positive: float
    mean_gap_nats_per_cell: float
    mean_tokens_per_cell: float
    gap_nats_per_token: float
    sign_test_significant_at_p: bool
    p_threshold: float


def _binomial_sign_test_pvalue(n: int, k: int) -> float:
    """One-sided sign-test p-value for H0: P(positive) = 0.5 vs H1: P > 0.5.

    Exact binomial; n cells, k positives.
    """
    from math import comb

    if n == 0:
        return 1.0
    p_tail = 0.0
    for x in range(k, n + 1):
        p_tail += comb(n, x) * 0.5**n
    return p_tail


def compute_perplexity_gap(
    *,
    cells: list[PerCellNLL],
    model_forward: ModelForward,
    p_threshold: float,
) -> GapResult:
    """Compute the gap and run a per-cell sign test against `p_threshold`."""
    if not cells:
        return GapResult(
            n_cells=0,
            n_positive=0,
            fraction_positive=0.0,
            mean_gap_nats_per_cell=0.0,
            mean_tokens_per_cell=0.0,
            gap_nats_per_token=0.0,
            sign_test_significant_at_p=False,
            p_threshold=p_threshold,
        )

    matched_nlls: list[float] = []
    shuffled_nlls: list[float] = []
    token_counts: list[int] = []

    for c in cells:
        matched_nlls.append(
            model_forward(
                micro_tokens=c.micro_tokens,
                conditioning_prefix=c.matched_conditioning_prefix,
            )
        )
        shuffled_nlls.append(
            model_forward(
                micro_tokens=c.micro_tokens,
                conditioning_prefix=c.shuffled_conditioning_prefix,
            )
        )
        token_counts.append(len(c.micro_tokens))

    gaps = [s - m for s, m in zip(shuffled_nlls, matched_nlls, strict=True)]
    n = len(gaps)
    n_positive = sum(1 for g in gaps if g > 0)
    mean_gap = sum(gaps) / n
    mean_tokens = sum(token_counts) / n
    # NLL is already per-token from the model_forward contract (see
    # ModelForward docstring) — no scaling by mean_tokens needed. Earlier
    # draft had `mean_gap if mean_tokens == 0 else mean_gap`, a dead-code
    # ternary that would tempt a future maintainer to "fix" it by adding
    # a division (which would double-divide and silently halve the gap).
    # mean_tokens is still computed for GapResult.mean_tokens_per_cell —
    # informational only, not load-bearing for the gap value.
    gap_per_token = mean_gap

    p_value = _binomial_sign_test_pvalue(n=n, k=n_positive)

    return GapResult(
        n_cells=n,
        n_positive=n_positive,
        fraction_positive=n_positive / n,
        mean_gap_nats_per_cell=mean_gap,
        mean_tokens_per_cell=mean_tokens,
        gap_nats_per_token=gap_per_token,
        sign_test_significant_at_p=p_value < p_threshold,
        p_threshold=p_threshold,
    )
