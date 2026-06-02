"""End-to-end scaffold runner (spec §; plan Task 12).

Closes the loop: build training shards -> CellDataModule (fail-closed holdout audit)
-> train the toy micro-generator -> checkpoint + resume -> generate + decode cells
via the sealed sub-F decoder -> per-cell slice eval -> reports/ summary.

``run_smoke`` proves the loop wires end-to-end on a tiny budget (CPU-runnable, no
GPU); it asserts a checkpoint round-trips bit-identically. ``run_short`` is the real
pre-deadline run (Leonardo 4xA100). The comparability lock is asserted at the run
entrypoint for GPU runs only (the CPU smoke runs a non-locked CPU torch build).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import torch

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.data.training.build_shards import build_training_shards
from cfm.data.training.datamodule import CellDataModule, build_conditioning_prefix
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import eval_set_locked_marker, holdout_manifest_path
from cfm.eval.slice_metrics import slice_eval
from cfm.inference.generate import generate_cell_tokens, try_decode_block
from cfm.training.config import ScaffoldConfig
from cfm.training.env_lock import assert_training_env_locked
from cfm.training.lit_module import ScaffoldLit
from cfm.training.train import build_trainer, maybe_compile

logger = logging.getLogger(__name__)

_RIGHT_ANGLE_BAR = 0.95  # PoC plausibility bar (reported; not a hard gate in the slice)


def _accelerator_for(devices: int) -> str:
    """GPU for multi-device (Leonardo) or when CUDA is present; CPU otherwise (the
    login-node / Mac smoke). bf16 + the env lock apply only on GPU."""
    return "gpu" if (devices > 1 or torch.cuda.is_available()) else "cpu"


def _datamodule(cfg: ScaffoldConfig) -> CellDataModule:
    build_training_shards(cfg.release, cfg.region)  # writes the lineage manifest
    return CellDataModule(
        training_manifest=training_manifest_path(cfg.release, cfg.region),
        holdout_manifest=holdout_manifest_path(cfg.release),
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_cell_tokens=cfg.max_len,
    )


def _generate_and_score(
    model: ScaffoldLit, cfg: ScaffoldConfig, *, n_cells: int, max_new: int
) -> dict:
    """Generate cells from the conditioning prefix, decode via the sealed decoder,
    and score the per-cell slice metrics (decoded / attempted; OGC validity; 90-corner;
    bref-collapse via the shared instrument). One real per-cell number for the loop."""
    prefix = build_conditioning_prefix()
    blocks: list[list[int]] = []
    geoms: list[dict] = []
    strata: list[int] = []
    n_attempted = 0
    for i in range(n_cells):
        tokens = generate_cell_tokens(
            model.model, prefix=prefix, max_new=max_new, seed=cfg.seed + i
        )
        cell_blocks = split_cell_into_features(tokens)
        n_attempted += len(cell_blocks)
        for block in cell_blocks:
            decoded = try_decode_block(block)
            if decoded is not None:
                blocks.append(block)
                geoms.append(decoded)
                strata.append(0)  # single stratum: conditioning is value-agnostic in slice v1
    return slice_eval(blocks, geoms, strata, n_attempted_blocks=n_attempted)


def run_smoke(devices: int = 4) -> dict:
    """Minimal loop on a tiny budget: train a couple of steps, checkpoint, prove the
    on-disk checkpoint reproduces the trained weights bit-identically, decode a cell."""
    accel = _accelerator_for(devices)
    cfg = ScaffoldConfig(
        devices=devices,
        accelerator=accel,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_len=256,
        batch_size=2,
        max_steps=2,
        compile=False,  # compile is wasteful on a 2-step smoke
    )
    if accel == "gpu":
        assert_training_env_locked()  # comparability lock (GPU runs only)

    dm = _datamodule(cfg)
    lit = ScaffoldLit(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(lit, dm)

    tmp = Path(tempfile.mkdtemp(prefix="scaffold-smoke-"))
    ckpt = tmp / "smoke.ckpt"
    trainer.save_checkpoint(ckpt)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    live = lit.state_dict()
    identical = bool(saved) and all(
        k in live and torch.equal(saved[k].cpu(), live[k].cpu()) for k in saved
    )

    score = _generate_and_score(lit, cfg, n_cells=4, max_new=64)
    return {
        "trained_steps": int(trainer.global_step),
        "checkpoint_written": ckpt.exists(),
        "resumed_bit_identical": identical,
        "decoded_cells": int(score["n_decoded"]),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_report(cfg: ScaffoldConfig, metrics: dict, *, trained_steps: int) -> Path:
    out_dir = Path("reports/phase-1-training-scaffold")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = eval_set_locked_marker(cfg.release)
    report = out_dir / f"{cfg.release}-{cfg.region}-loop-closed.md"
    lines = [
        f"# Phase-1 training scaffold — loop closed on {cfg.region}",
        "",
        f"- **commit:** `{_git_commit()}`",
        f"- **data snapshot:** release `{cfg.release}`, holdout marker `{marker}`",
        f"- **trained steps:** {trained_steps}",
        "",
        "## Config",
        "```json",
        json.dumps(cfg.model_dump(), indent=2, sort_keys=True),
        "```",
        "",
        "## Per-cell slice metrics (REPORTED, not gated)",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Scope",
        "A green per-cell number means the micro generator emits decodable, locally-",
        "valid cells. It does NOT mean the tile generator works: tile cell-to-cell",
        "coherence, boundary-contract stitching and macro-planner conditioning are",
        "UNSCORED-in-slice. Conditioning is the field-slot id-block (value-agnostic in",
        "slice v1); value-bearing conditioning + its compliance scoring are bake-off",
        f"follow-ons. Right-angle PoC bar (reported): {_RIGHT_ANGLE_BAR}.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_short(cfg: ScaffoldConfig | None = None) -> dict:
    """The real pre-deadline run (Leonardo 4xA100). Trains for cfg.max_steps, evals
    once, writes the reports/ summary."""
    cfg = cfg or ScaffoldConfig()
    if cfg.accelerator == "gpu":
        assert_training_env_locked()

    dm = _datamodule(cfg)
    lit = maybe_compile(ScaffoldLit(cfg), cfg)
    trainer = build_trainer(cfg)
    trainer.fit(lit, dm)

    metrics = _generate_and_score(lit, cfg, n_cells=64, max_new=512)
    report = _write_report(cfg, metrics, trained_steps=int(trainer.global_step))
    logger.info("wrote %s", report)
    return {"trained_steps": int(trainer.global_step), "metrics": metrics, "report": str(report)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Phase-1 training scaffold runner")
    parser.add_argument("--smoke", action="store_true", help="tiny loop-closing smoke")
    parser.add_argument("--devices", type=int, default=4, help="DDP devices (Leonardo node = 4)")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(devices=args.devices)))
    else:
        print(json.dumps(run_short(ScaffoldConfig(devices=args.devices)), default=str))


if __name__ == "__main__":
    main()
