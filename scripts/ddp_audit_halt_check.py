"""DDP all-ranks audit-halt check (spec §6 trigger 1; handoff non-vacuous-DDP rule).

Launched under ``srun`` with 4 tasks (one per A100). Proves that a planted holdout
leak halts ``CellDataModule.setup()`` BEFORE any batch on EVERY rank — and asserts
``world_size == 4`` so a check that silently ran on 1 rank fails loudly instead of
passing vacuously.

Uses a planted-leak synthetic manifest (a training tile whose stamped lineage points
at a holdout tile), so the audit raises before any real data is read. The per-rank
"halted before batch 0" booleans are all-reduced; the job passes only if all 4 ranks
halted. Backend is gloo (the audit is pure Python; no GPU tensor needed for the
collective), but the launch itself occupies the 4-GPU node.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from cfm.data.training.datamodule import CellDataModule
from cfm.eval.holdout.lineage_audit import HoldoutLeakError

_EXPECTED_WORLD = 4
_REGION = "singapore"


def _planted_leak_manifests(tmp: Path) -> tuple[Path, Path]:
    holdout = tmp / "holdout.yaml"
    holdout.write_text(
        yaml.safe_dump(
            {
                # schema 2.0 so the planted-leak audit halts on the LEAK, not on a schema refusal
                "manifest_schema_version": "2.0",
                "regions": {_REGION: {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
            }
        )
    )
    training = tmp / "training_manifest.yaml"
    training.write_text(
        yaml.safe_dump(
            {
                "release": "2026-04-15.0",
                "region": _REGION,
                # tile (2,2) whose STAMPED lineage references the holdout tile (1,7) -> leak
                "tiles": [{"tile_i": 2, "tile_j": 2, "lineage": [[_REGION, 1, 7]]}],
            }
        )
    )
    return training, holdout


def main() -> None:
    # torchrun sets RANK/WORLD_SIZE; under bare srun fall back to SLURM_*.
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world)
    try:
        if dist.get_world_size() != _EXPECTED_WORLD:
            raise RuntimeError(
                f"world_size={dist.get_world_size()} != {_EXPECTED_WORLD} — vacuous DDP check"
            )

        tmp = Path(tempfile.mkdtemp(prefix=f"ddp-audit-{rank}-"))
        training, holdout = _planted_leak_manifests(tmp)
        dm = CellDataModule(training_manifest=training, holdout_manifest=holdout, seed=7)

        halted_before_batch0 = 0
        try:
            dm.setup("fit")
        except HoldoutLeakError:
            halted_before_batch0 = 1 if dm._batches_yielded == 0 else 0

        flag = torch.tensor([halted_before_batch0], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
        total = int(flag.item())

        ok = total == _EXPECTED_WORLD
        if rank == 0:
            status = "PASS" if ok else "FAIL"
            print(
                f"[ddp-audit-halt] world_size={world} "
                f"ranks_halted_before_batch0={total}/{_EXPECTED_WORLD} -> {status}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
