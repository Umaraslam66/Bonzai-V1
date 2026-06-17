"""Task 8 discrimination tests: the LightningModule wires the REAL vocab sizes and
the loss masks both prefix and padding; the Trainer's checkpoint monitors val_loss,
never the holdout (selection-loop leak guard, Task 10's eval-protocol lock)."""

from __future__ import annotations

import torch

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit


def _cfg(**kw) -> ScaffoldConfig:
    # region is REQUIRED (no default); SG keeps these vocab/loss tests region-agnostic
    base = dict(
        region="singapore",
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_len=128,
        accelerator="cpu",
        devices=1,
    )
    base.update(kw)
    return ScaffoldConfig(**base)


def _batch(ids: torch.Tensor, prefix_len: list[int], seq_len: list[int]) -> dict:
    # Task 24b: production batches always carry char_stats (the char-built model
    # REFUSES a batch without them — fail-loud, never silent placeholder training).
    return {
        "ids": ids,
        "prefix_len": torch.tensor(prefix_len),
        "seq_len": torch.tensor(seq_len),
        "char_stats": torch.zeros(ids.shape[0], 7),
    }


def test_lit_builds_model_from_real_vocab_sizes():
    lit = ScaffoldLit(_cfg())
    assert lit.model.cfg.n_subf_vocab == max(vocab_tag_to_id().values()) + 1  # 1508
    # n_cond = the VALUE-BEARING conditioning embedding span (Task 6/7), not the 9-field
    # count: the embedding must cover every value-bearing id above the sub-F vocab.
    assert lit.model.cfg.n_cond == conditioning_id_span()  # 9 fields * 64 stride = 576 (Task 24a)
    # positional capacity covers the conditioning prefix (9 id positions + the Task-24b
    # continuous character position) PLUS the cell budget — positions, NOT embedding rows
    assert lit.model.cfg.max_len == 128 + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS


def test_training_step_returns_scalar_loss():
    lit = ScaffoldLit(_cfg())
    ids = torch.randint(0, 1508, (2, 32))
    loss = lit.training_step(_batch(ids, [10, 10], [32, 32]), 0)
    assert loss.ndim == 0 and loss.requires_grad


def test_validation_step_runs_without_grad_error():
    lit = ScaffoldLit(_cfg())
    ids = torch.randint(0, 1508, (2, 24))
    lit.eval()
    out = lit.validation_step(_batch(ids, [10, 10], [20, 24]), 0)  # must not raise
    assert out is None or out.ndim == 0


def test_init_is_seed_reproducible_across_instances():
    """Weight init must be reproducible across separate constructions: same seed ->
    identical initial weights; different seed -> different. This is what makes a
    4->4 resume comparable to an uninterrupted reference (init must not vary per
    process). Regression guard for the cross-process divergence found on Leonardo."""
    a = ScaffoldLit(_cfg(seed=7))
    b = ScaffoldLit(_cfg(seed=7))
    c = ScaffoldLit(_cfg(seed=8))
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # same seed -> identical init
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)  # different seed -> differs


def test_checkpoint_hparams_record_the_scheme():
    # F16: save_hyperparameters(cfg.model_dump()) puts the scheme tag into every
    # checkpoint's hparams, making a scheme mismatch detectable at load.
    # Task 24b reverse-lock (knob B): "value" -> "value-char-v1" (one bump covering
    # 24a+24b; no checkpoint was blessed under the interim 24a layout).
    lit = ScaffoldLit(_cfg())
    assert lit.hparams["conditioning_scheme"] == "value-char-v1"


def test_configure_optimizers_returns_adamw_and_step_cosine():
    lit = ScaffoldLit(_cfg(max_steps=100))
    cfg = lit.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    assert cfg["lr_scheduler"]["interval"] == "step"
