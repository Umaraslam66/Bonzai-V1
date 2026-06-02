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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger

from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit

logger = logging.getLogger(__name__)

#: mandatory checkpoint cadence (CLAUDE.md: every 30 minutes, resumable).
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=30)


class WorldSizeGuard(Callback):
    """Fail loud if the run did not actually launch on ``expected`` ranks.

    The handoff's non-vacuous-DDP rule: a 4->4 resume or all-ranks-halt test that
    silently ran on 1 rank is a vacuous pass. This asserts ``trainer.world_size ==
    expected`` at fit start, so a misconfigured single-rank launch fails instead of
    quietly "passing"."""

    def __init__(self, expected: int) -> None:
        self.expected = expected

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.world_size != self.expected:
            raise RuntimeError(
                f"WorldSizeGuard: launched on world_size={trainer.world_size}, "
                f"expected {self.expected} — a DDP check on the wrong rank count is vacuous."
            )


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


def build_trainer(
    cfg: ScaffoldConfig,
    *,
    fast_dev_run: bool = False,
    max_epochs: int | None = None,
    limit_train_batches: int | float | None = None,
    default_root_dir: str | None = None,
    max_time: str | None = None,
) -> L.Trainer:
    """Construct the Trainer. ``accelerator="cpu"`` (tests / login node) downgrades
    precision to fp32 and strategy to single-process; the real run uses cfg's
    accelerator="gpu", devices=4, ddp, bf16 (the comparability lock).

    ``max_epochs`` (+ optional ``limit_train_batches``) switches from the default
    step-budget to an epoch budget — used by the bit-identical resume check, which
    resumes at an epoch boundary. ``max_time`` ("DD:HH:MM:SS") wall-clock-bounds the
    run regardless of step count — the scale-up probe uses it so it always yields a
    per-step cost number rather than timing out mid-step."""
    L.seed_everything(cfg.seed, workers=True)

    on_gpu = cfg.accelerator == "gpu"
    precision = cfg.precision if on_gpu else "32-true"
    strategy = "ddp" if cfg.devices > 1 else "auto"

    checkpoint = ModelCheckpoint(
        train_time_interval=CHECKPOINT_INTERVAL,
        save_last=True,
        monitor="val_loss",  # internal val split ONLY; never the holdout
    )
    callbacks: list[Callback] = [checkpoint]
    if cfg.devices > 1:
        callbacks.append(WorldSizeGuard(cfg.devices))  # non-vacuous DDP guard

    epoch_mode = max_epochs is not None
    return L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=strategy,
        precision=precision,
        max_time=max_time,
        max_steps=-1 if epoch_mode else cfg.max_steps,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=cfg.grad_accum,  # holds effective batch constant across scales
        fast_dev_run=fast_dev_run,
        deterministic=True,
        use_distributed_sampler=False,  # the DataModule owns the seeded sampler
        callbacks=callbacks,
        logger=_build_logger(),
        default_root_dir=default_root_dir,
    )


def effective_batch_size(cfg: ScaffoldConfig) -> int:
    """The comparability-relevant batch: per-GPU batch * devices * grad accumulation.

    Held constant across scales (§10) so a memory-forced smaller per-GPU batch at 300M/1B
    is compensated by grad_accum rather than silently changing the optimization regime.
    """
    return cfg.batch_size * cfg.devices * cfg.grad_accum


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
