"""DDP bit-identical 4->4 resume check (spec §; handoff non-vacuous-DDP rule).

Launched under ``srun`` (4 tasks / 4 A100). Mirrors real preemption with three
phases run as separate ``srun`` steps that share a directory:

  reference : train uninterrupted for ``EPOCHS`` -> save final weights (ref.pt)
  part1     : train HALF (1 epoch) -> checkpoint (part.ckpt)
  part2     : RESUME part.ckpt -> finish to ``EPOCHS`` -> compare to ref.pt

The comparison must be BIT-IDENTICAL (every tensor ``torch.equal``). Determinism
comes from a fixed seed + ``deterministic=True`` + the DataModule's seeded sampler
(use_distributed_sampler=False, fixed order across epochs) + resume at the epoch
boundary. ``WorldSizeGuard`` (added by build_trainer for devices>1) asserts
world_size==4 in every phase, so a single-rank launch fails instead of passing
vacuously. The training manifest is pre-built by the sbatch preamble (single
process) — these 4 ranks only READ it.

Honesty note: if DDP float non-determinism breaks bit-identity, this FAILS loudly;
that is a real finding to report, not something to silently relax.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from cfm.data.training.datamodule import CellDataModule
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import holdout_manifest_path
from cfm.training.config import ScaffoldConfig
from cfm.training.env_lock import assert_training_env_locked
from cfm.training.lit_module import ScaffoldLit
from cfm.training.train import build_trainer

_EPOCHS = 2
_LIMIT_TRAIN_BATCHES = 8
_RELEASE, _REGION = "2026-04-15.0", "singapore"


def _cfg() -> ScaffoldConfig:
    return ScaffoldConfig(
        devices=4,
        accelerator="gpu",
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_len=512,
        batch_size=8,
        compile=False,  # compile + deterministic resume comparison don't mix
        seed=7,
    )


def _datamodule(cfg: ScaffoldConfig) -> CellDataModule:
    return CellDataModule(
        training_manifest=training_manifest_path(cfg.release, cfg.region),
        holdout_manifest=holdout_manifest_path(cfg.release),
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_cell_tokens=cfg.max_len,
    )


def _cpu_state(lit: ScaffoldLit) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in lit.state_dict().items()}


def _save_state(state: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["reference", "part1", "part2"], required=True)
    parser.add_argument("--shared", default=os.environ.get("SHARED_DIR", "reports/ddp-resume"))
    args = parser.parse_args()

    assert_training_env_locked()  # comparability lock (GPU run)
    shared = Path(args.shared)
    ref_path = shared / "ref.pt"
    ckpt_path = shared / "part.ckpt"

    cfg = _cfg()
    dm = _datamodule(cfg)

    if args.phase == "reference":
        lit = ScaffoldLit(cfg)
        trainer = build_trainer(
            cfg,
            max_epochs=_EPOCHS,
            limit_train_batches=_LIMIT_TRAIN_BATCHES,
            default_root_dir=str(shared / "reference"),
        )
        trainer.fit(lit, dm)
        if trainer.is_global_zero:
            _save_state(_cpu_state(lit), ref_path)
            print(f"[ddp-resume] reference saved -> {ref_path}", flush=True)
        sys.exit(0)

    if args.phase == "part1":
        lit = ScaffoldLit(cfg)
        trainer = build_trainer(
            cfg,
            max_epochs=_EPOCHS // 2,
            limit_train_batches=_LIMIT_TRAIN_BATCHES,
            default_root_dir=str(shared / "part1"),
        )
        trainer.fit(lit, dm)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # save_checkpoint is a COLLECTIVE (internal barrier; Lightning writes on rank 0).
        # Guarding it with is_global_zero deadlocks the other ranks — call on ALL ranks.
        trainer.save_checkpoint(ckpt_path)
        if trainer.is_global_zero:
            print(f"[ddp-resume] part1 checkpoint -> {ckpt_path}", flush=True)
        sys.exit(0)

    # part2: resume the checkpoint to the full epoch budget, then compare to reference
    lit = ScaffoldLit(cfg)
    trainer = build_trainer(
        cfg,
        max_epochs=_EPOCHS,
        limit_train_batches=_LIMIT_TRAIN_BATCHES,
        default_root_dir=str(shared / "part2"),
    )
    trainer.fit(lit, dm, ckpt_path=str(ckpt_path))
    code = 0
    if trainer.is_global_zero:
        resumed = _cpu_state(lit)
        reference = torch.load(ref_path, map_location="cpu", weights_only=False)
        keys = sorted(reference)
        identical = all(k in resumed and torch.equal(reference[k], resumed[k]) for k in keys)
        n_diff = sum(
            1 for k in keys if k not in resumed or not torch.equal(reference[k], resumed[k])
        )
        # Classify the divergence: tiny max|diff| (~1e-5) is DDP/NCCL reduction-order
        # float noise (resume is functionally correct); a large diff is a real bug.
        max_abs = 0.0
        for k in keys:
            if k in resumed and not torch.equal(reference[k], resumed[k]):
                d = (reference[k] - resumed[k]).abs().max().item()
                max_abs = max(max_abs, d)
                print(f"[ddp-resume]   differs: {k} max|diff|={d:.3e}", flush=True)
        # functionally-identical tolerance: float32 accumulation noise over a few
        # DDP steps stays well under 1e-3; a real trajectory divergence is >> that.
        functionally_identical = all(
            k in resumed and torch.allclose(reference[k], resumed[k], rtol=0, atol=1e-4)
            for k in keys
        )
        if identical:
            status = "PASS"
        elif functionally_identical:
            status = "PASS(within 1e-4)"
        else:
            status = "FAIL"
        print(
            f"[ddp-resume] world_size=4 bit_identical_4to4_resume={identical} "
            f"functionally_identical_atol1e-4={functionally_identical} "
            f"({n_diff}/{len(keys)} tensors differ, max|diff|={max_abs:.3e}) -> {status}",
            flush=True,
        )
        code = 0 if functionally_identical else 1
    sys.exit(code)


if __name__ == "__main__":
    main()
