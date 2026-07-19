"""Perplexity-gap NLL wrapper + gap loop (spec §2).

``cell_nll`` runs ONE teacher-forced forward of ``[prefix + body]`` through the model and
returns the mean NLL in nats/token over the BODY positions, reusing the training masked-CE
path (``ScaffoldBackbone.training_loss``) so eval NLL == training NLL by construction.

``compute_gap`` reuses ``perplexity_gap.GapResult`` + the exact binomial sign test, but
threads ``char_stats`` per cell (the original ``perplexity_gap.ModelForward`` signature
predates the Task-24b character carrier, so it cannot carry char_stats — we keep its result
type + sign test and supply the per-cell forward here). For the MACRO-ONLY (primary) gap,
``matched_char_stats == shuffled_char_stats`` (the cell's own), so char_stats cancels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cfm.eval.perplexity_gap import GapResult, _binomial_sign_test_pvalue


def cell_nll(
    model: torch.nn.Module,
    *,
    body_tokens: list[int],
    conditioning_prefix: list[int],
    char_stats: list[float] | None,
    device: str | torch.device = "cpu",
) -> float:
    """Mean NLL (nats/token) of ``body_tokens`` given ``conditioning_prefix``, teacher-forced.

    Builds the batch exactly as training does (ids = prefix+body, prefix_len = len(prefix),
    no padding) and returns ``training_loss(...).loss`` — the body-only masked-CE mean.
    """
    ids = torch.tensor(
        [list(conditioning_prefix) + list(body_tokens)], dtype=torch.long, device=device
    )
    prefix_len = torch.tensor([len(conditioning_prefix)], dtype=torch.long, device=device)
    cs = (
        torch.tensor([list(char_stats)], dtype=torch.float32, device=device)
        if char_stats is not None
        else None
    )
    with torch.no_grad():
        out = model.training_loss(ids, prefix_len=prefix_len, seq_len=None, char_stats=cs)
    return float(out.loss)


@dataclass(frozen=True)
class GapCell:
    cell_id: str
    body_tokens: list[int]
    matched_prefix: list[int]
    shuffled_prefix: list[int]
    matched_char_stats: list[float] | None
    shuffled_char_stats: list[float] | None


def effective_macro_shuffle_fraction(cells: list[GapCell]) -> float:
    """Fraction of cells whose shuffled prefix actually differs from the matched prefix.

    The macro-only gap is only meaningful when the shuffle is NOT a no-op. A near-zero gap
    with a low effective fraction is a vacuous shuffle, not a real null effect — the harness
    flags the gap UNRELIABLE below a floor (spec §2 (ii) guard)."""
    if not cells:
        return 0.0
    changed = sum(1 for c in cells if list(c.shuffled_prefix) != list(c.matched_prefix))
    return changed / len(cells)


def compute_gap(
    model: torch.nn.Module,
    cells: list[GapCell],
    *,
    device: str | torch.device = "cpu",
    p_threshold: float = 0.05,
) -> GapResult:
    """gap = NLL_shuffled - NLL_matched per cell; reuses GapResult + the binomial sign test."""
    if not cells:
        return GapResult(0, 0, 0.0, 0.0, 0.0, 0.0, False, p_threshold)
    gaps: list[float] = []
    token_counts: list[int] = []
    for c in cells:
        matched = cell_nll(
            model,
            body_tokens=c.body_tokens,
            conditioning_prefix=c.matched_prefix,
            char_stats=c.matched_char_stats,
            device=device,
        )
        shuffled = cell_nll(
            model,
            body_tokens=c.body_tokens,
            conditioning_prefix=c.shuffled_prefix,
            char_stats=c.shuffled_char_stats,
            device=device,
        )
        gaps.append(shuffled - matched)
        token_counts.append(len(c.body_tokens))
    n = len(gaps)
    n_positive = sum(1 for g in gaps if g > 0)
    mean_gap = sum(gaps) / n
    p_value = _binomial_sign_test_pvalue(n=n, k=n_positive)
    return GapResult(
        n_cells=n,
        n_positive=n_positive,
        fraction_positive=n_positive / n,
        mean_gap_nats_per_cell=mean_gap,
        mean_tokens_per_cell=sum(token_counts) / n,
        gap_nats_per_token=mean_gap,  # cell_nll is already per-token (training_loss mean)
        sign_test_significant_at_p=p_value < p_threshold,
        p_threshold=p_threshold,
    )
