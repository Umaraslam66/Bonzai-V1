"""Across-job checkpoint resume on $WORK (Phase-2 bake-off Task 10; §10).

The 1B runs are long, Slurm jobs have hard wall-clock limits, so a 1B run SPANS
MULTIPLE jobs. Checkpoints land on ``$WORK`` (allocation-independent storage that
survives allocation expiry / the renewal gap), and a relaunched job resumes from the
latest checkpoint there rather than starting over. This module is the resume-decision
logic; ``scripts/bakeoff_run.sbatch`` is the relaunch-on-timeout wrapper, and the real
multi-job relaunch is verified on Leonardo (the unit logic is verified here).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_STEP_RE = re.compile(r"step=(\d+)")


def work_checkpoint_dir(
    backbone: str,
    scale: str,
    *,
    region: str,
    seed: int,
    work_root: str | os.PathLike[str] | None = None,
) -> Path:
    """The per-run checkpoint directory on allocation-independent ``$WORK`` storage.

    ``$WORK`` survives allocation expiry, so a couple-day renewal gap means "resume when
    the new allocation lands", not "start over". Falls back to a local ``checkpoints/``
    root when ``$WORK`` is unset (tests / dev).

    EVERY run-key axis (backbone, scale, region, seed) is in the path, not just
    backbone+scale: ``resume_ckpt_path`` is wired into the training entrypoint (Task 18),
    so two runs that share a dir would SILENTLY resume each other's ``last.ckpt`` — a
    wrong-resume that no error surfaces.
    Nested ``{backbone}-{scale}/{region}-seed{seed}`` keeps the bake-off matrix grouping
    readable on $WORK.
    """
    # DECISION: chose to leave train_set OUT of the key — union runs are distinguished
    # by their region tag today, so adding it would only deepen the tree. Revisit if
    # two train_sets ever train at the same (backbone, scale, region, seed).
    root = Path(work_root) if work_root is not None else Path(os.environ.get("WORK", "checkpoints"))
    return (
        root
        / "Bonzai-OSM"
        / "checkpoints"
        / "bakeoff"
        / f"{backbone}-{scale}"
        / f"{region}-seed{seed}"
    )


def _step_of(ckpt: Path) -> int:
    m = _STEP_RE.search(ckpt.name)
    return int(m.group(1)) if m else -1


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """The checkpoint to resume from: Lightning's ``last.ckpt`` if present, else the
    highest-step ``*.ckpt``. ``None`` if the directory is absent or empty."""
    if not ckpt_dir.exists():
        return None
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=_step_of)
    return ckpts[-1] if ckpts else None


def resume_ckpt_path(ckpt_dir: Path) -> Path | None:
    """``ckpt_path`` to hand ``trainer.fit`` so a relaunched job continues.

    ``None`` on a fresh run (empty/absent dir) so no false resume is attempted; the
    latest checkpoint otherwise, so a relaunch continues from the last step, not step 0.
    """
    return latest_checkpoint(ckpt_dir)
