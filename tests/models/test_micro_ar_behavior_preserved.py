"""Behavior-preservation gate for the ScaffoldBackbone refactor (spec §3.2, protocol Gate-6).

Captured against the CURRENT (pre-refactor) ``MicroAR``. After the Task-3 extract-base
refactor, the refactored ``MicroAR`` must reproduce these outputs BIT-IDENTICALLY
(``torch.equal``, not ``allclose``). This is the external-source-of-truth check: the new
abstraction is cross-referenced against the OLD module's real output, not its description.

Bit-identity is RNG-order sensitive: the refactored ``__init__`` must consume the seeded
RNG in the SAME order as the pre-refactor one (``embed -> pos -> char_proj -> mixer ->
head``). A reorder (e.g. building the head before the mixer) silently changes the random
init and this test goes red — which is exactly the regression it exists to catch.

The golden is captured once on the locked Leonardo CPU env (torch 2.5.1, Linux x86_64) and
committed; the T3 verification runs on that same env so float arithmetic is identical.
Bit-identity is therefore DEFINED only on that env: other BLAS backends (e.g. macOS Apple
Accelerate) round matmuls differently (~1e-7 uniform drift, loss unaffected). Off the locked
env this test asserts allclose instead — still red on real logic regressions (structured /
large drift), while the bit-exact gate continues to run where it is defined.
"""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from cfm.models.micro_ar import MicroAR, MicroARConfig

_GOLDEN = Path(__file__).resolve().parent / "_golden" / "micro_ar_v1.pt"

#: The env the golden was captured on (docstring): bit-identity holds there and only there.
_ON_CAPTURE_ENV = platform.system() == "Linux" and platform.machine() == "x86_64"


def _fixed_model_and_input():
    torch.manual_seed(1234)
    cfg = MicroARConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_subf_vocab=40,
        n_cond=16,
        max_len=24,
        n_char_stats=7,
        char_position=9,
    )
    m = MicroAR(cfg).eval()
    # A VALID example respects the model's locked contract: the head predicts sub-F ids
    # ONLY (< n_subf_vocab), so every SUPERVISED (cell-token) target must be sub-F. The
    # prefix id slots carry conditioning ids (the id-block ABOVE the sub-F range) — valid
    # INPUTS whose targets are masked. Drawing cell targets from the conditioning range
    # would hand cross_entropy an out-of-range class (the model never predicts conditioning):
    # a malformed fixture, not a model defect.
    ids = torch.randint(0, 40, (2, 12))  # cell tokens in the sub-F range -> always valid targets
    ids[:, :9] = torch.randint(
        40, 40 + 16, (2, 9)
    )  # 9 conditioning-id prefix slots (masked targets)
    char = torch.randn(2, 7)
    prefix_len = torch.tensor([10, 10])
    seq_len = torch.tensor([12, 11])
    return m, ids, char, prefix_len, seq_len


def test_micro_ar_forward_and_loss_match_golden() -> None:
    m, ids, char, pl, sl = _fixed_model_and_input()
    with torch.no_grad():
        logits = m(ids, char_stats=char)
        loss = m.training_loss(ids, prefix_len=pl, seq_len=sl, char_stats=char).loss
    if not _GOLDEN.exists():  # one-time capture, then committed + frozen
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": logits, "loss": loss}, _GOLDEN)
    g = torch.load(_GOLDEN, weights_only=True)  # our own trusted fixture of plain tensors
    if _ON_CAPTURE_ENV:
        assert torch.equal(logits, g["logits"]), (
            "forward logits drifted from the pre-refactor golden"
        )
        assert torch.equal(loss, g["loss"]), "training loss drifted from the pre-refactor golden"
    else:
        # Off the capture env: BLAS rounding differs at ~1e-7; assert closeness, not bits.
        assert torch.allclose(logits, g["logits"], rtol=1e-4, atol=1e-6), (
            "forward logits drifted from the pre-refactor golden beyond BLAS rounding"
        )
        assert torch.allclose(loss, g["loss"], rtol=1e-4, atol=1e-6), (
            "training loss drifted from the pre-refactor golden beyond BLAS rounding"
        )
