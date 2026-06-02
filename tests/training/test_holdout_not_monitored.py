"""Task 10 eval-protocol lock (Lightning-dependent half): the frozen holdout eval
is NEVER a monitored Lightning val metric, so it can't drive checkpoint selection.
Only the internal val split's val_loss may. (Selection-loop leak guard, spec §8.)"""

from __future__ import annotations

from cfm.training.config import ScaffoldConfig
from cfm.training.train import build_trainer


def test_holdout_eval_is_not_a_lightning_val_metric():
    trainer = build_trainer(ScaffoldConfig(devices=1, accelerator="cpu"))
    monitored = {c.monitor for c in trainer.callbacks if getattr(c, "monitor", None)}
    assert all("holdout" not in (m or "") for m in monitored)  # holdout never monitored
    assert "val_loss" in monitored  # the internal val split IS the selection signal


def test_checkpoint_callback_has_30min_interval_and_save_last():
    import datetime

    from lightning.pytorch.callbacks import ModelCheckpoint

    trainer = build_trainer(ScaffoldConfig(devices=1, accelerator="cpu"))
    ckpts = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    assert ckpts, "a ModelCheckpoint callback must be configured (mandatory cadence)"
    ck = ckpts[0]
    assert ck._train_time_interval == datetime.timedelta(minutes=30)
    assert ck.save_last is True
