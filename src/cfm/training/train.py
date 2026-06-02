"""Trainer wiring (spec §7): 4xA100 DDP, bf16, deterministic, 30-min checkpointing,
loss curves to tensorboard. The comparability lock (torch/lightning/pydantic) is
asserted separately at the run entrypoint (``scripts/train_scaffold.py``).

Checkpoint selection monitors ``val_loss`` (the internal val split) ONLY — never the
frozen holdout. The DataModule supplies its own seeded DistributedSampler, so
Lightning's automatic sampler replacement is disabled (one source of data order).
"""

from __future__ import annotations

import datetime
import logging

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger

from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit

logger = logging.getLogger(__name__)

#: mandatory checkpoint cadence (CLAUDE.md: every 30 minutes, resumable).
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=30)


def _build_logger() -> Logger:
    """Prefer TensorBoard (spec §7 loss curves) but fall back to the dependency-free
    CSVLogger when tensorboard is absent. tensorboard is NOT in the comparability
    lock (it is logging-only, not numerics) and is currently installed on neither the
    Mac dev env nor the Leonardo freeze, so CSV is the actual default; adding
    tensorboard later is a Task-0 dependency decision, never an ad-hoc add."""
    import importlib.util

    if importlib.util.find_spec("tensorboard") is not None:
        from lightning.pytorch.loggers import TensorBoardLogger

        return TensorBoardLogger("reports/tb", name="training-scaffold")
    logger.warning("tensorboard not installed; logging loss curves to CSVLogger")
    return CSVLogger("reports/logs", name="training-scaffold")


def build_trainer(cfg: ScaffoldConfig, *, fast_dev_run: bool = False) -> L.Trainer:
    """Construct the Trainer. ``accelerator="cpu"`` (tests / login node) downgrades
    precision to fp32 and strategy to single-process; the real run uses cfg's
    accelerator="gpu", devices=4, ddp, bf16 (the comparability lock)."""
    L.seed_everything(cfg.seed, workers=True)

    on_gpu = cfg.accelerator == "gpu"
    precision = cfg.precision if on_gpu else "32-true"
    strategy = "ddp" if cfg.devices > 1 else "auto"

    checkpoint = ModelCheckpoint(
        train_time_interval=CHECKPOINT_INTERVAL,
        save_last=True,
        monitor="val_loss",  # internal val split ONLY; never the holdout
    )
    return L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=strategy,
        precision=precision,
        max_steps=cfg.max_steps,
        fast_dev_run=fast_dev_run,
        deterministic=True,
        use_distributed_sampler=False,  # the DataModule owns the seeded sampler
        callbacks=[checkpoint],
        logger=_build_logger(),
    )


def maybe_compile(lit: ScaffoldLit, cfg: ScaffoldConfig) -> ScaffoldLit:
    """torch.compile the model when cfg.compile (default-on, CLAUDE.md); disable on
    error rather than fight the compiler."""
    if not cfg.compile:
        return lit
    try:
        import torch

        lit.model = torch.compile(lit.model)
    except Exception as exc:
        logger.warning("torch.compile disabled (%s: %s)", type(exc).__name__, exc)
    return lit
