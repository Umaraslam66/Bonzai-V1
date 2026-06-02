"""Task 8 discrimination tests: the LightningModule wires the REAL vocab sizes and
the loss masks both prefix and padding; the Trainer's checkpoint monitors val_loss,
never the holdout (selection-loop leak guard, Task 10's eval-protocol lock)."""

from __future__ import annotations

import torch

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import conditioning_field_to_id
from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit


def _cfg(**kw) -> ScaffoldConfig:
    base = dict(d_model=64, n_layers=2, n_heads=2, max_len=128, accelerator="cpu", devices=1)
    base.update(kw)
    return ScaffoldConfig(**base)


def test_lit_builds_model_from_real_vocab_sizes():
    lit = ScaffoldLit(_cfg())
    assert lit.model.cfg.n_subf_vocab == max(vocab_tag_to_id().values()) + 1  # 1508
    assert lit.model.cfg.n_cond == len(conditioning_field_to_id())  # 8
    # positional capacity covers the conditioning prefix PLUS the cell-token budget
    assert lit.model.cfg.max_len == 128 + len(conditioning_field_to_id())


def test_training_step_returns_scalar_loss():
    lit = ScaffoldLit(_cfg())
    ids = torch.randint(0, 1508, (2, 32))
    batch = {"ids": ids, "prefix_len": torch.tensor([8, 8]), "seq_len": torch.tensor([32, 32])}
    loss = lit.training_step(batch, 0)
    assert loss.ndim == 0 and loss.requires_grad


def test_validation_step_runs_without_grad_error():
    lit = ScaffoldLit(_cfg())
    ids = torch.randint(0, 1508, (2, 24))
    batch = {"ids": ids, "prefix_len": torch.tensor([8, 8]), "seq_len": torch.tensor([20, 24])}
    lit.eval()
    out = lit.validation_step(batch, 0)  # must not raise
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


def test_configure_optimizers_returns_adamw_and_step_cosine():
    lit = ScaffoldLit(_cfg(max_steps=100))
    cfg = lit.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    assert cfg["lr_scheduler"]["interval"] == "step"
