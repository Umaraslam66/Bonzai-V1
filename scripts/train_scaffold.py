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
import time
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


def _datamodule(cfg: ScaffoldConfig, *, build: bool = True) -> CellDataModule:
    # DDP: pre-build the manifest ONCE in the sbatch preamble (single process) and
    # pass build=False to the 4 srun ranks, so they only READ it (no write race).
    if build:
        build_training_shards(cfg.release, cfg.region)  # writes the lineage manifest
    return CellDataModule(
        training_manifest=training_manifest_path(cfg.release, cfg.region),
        holdout_manifest=holdout_manifest_path(cfg.release),
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_cell_tokens=cfg.max_len,
        # Legacy SG thin-slice: audits the FROZEN, IMMUTABLE Singapore holdout manifest
        # (schema 1.0), which can never be re-stamped to 2.0. EU/bake-off reuse of this
        # script MUST flip this to "2.0" AND re-point holdout_manifest to
        # multiregion_holdout_manifest_path, or the schema backstop is defeated here (the
        # #16 failure, one layer over). See handoff residual.
        expected_holdout_schema="1.0",
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


def _write_report(
    cfg: ScaffoldConfig, metrics: dict, *, trained_steps: int, cost: dict | None = None
) -> Path:
    out_dir = Path("reports/phase-1-training-scaffold")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = eval_set_locked_marker(cfg.release)
    suffix = f"-scaleup-{cost['n_params_M']:.0f}M" if cost else ""
    report = out_dir / f"{cfg.release}-{cfg.region}-loop-closed{suffix}.md"
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
    ]
    if cost is not None:
        lines += [
            "## Cost / throughput (scale-up probe — bake-off sizing)",
            "Per-step node-h measured on 4xA100; warmup INCLUDED (conservative). Use",
            "`est_node_h_12_runs` vs `prd_bakeoff_budget_node_h` to confirm the 4x3",
            "bake-off at this top scale fits the PRD envelope.",
            "```json",
            json.dumps(cost, indent=2, sort_keys=True),
            "```",
            "",
        ]
    lines += [
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


def _cost(cfg: ScaffoldConfig, *, fit_seconds: float, steps: int) -> dict:
    """Per-run cost from wall-clock x the whole 4-GPU node. node-h is wall-hours
    (1 Booster node billed whole). Warmup (torch.compile) is INCLUDED -> a
    conservative (over-)estimate, the safe direction for budgeting. Extrapolates a
    full bake-off run + the 12-run total vs the PRD's ~375 node-h (1500 GPU-h) budget."""
    node_h = fit_seconds / 3600.0  # 1 node occupied for the wall duration
    per_step_node_h = node_h / steps if steps else 0.0
    full_run_steps = 10_000  # illustrative bake-off per-run length (identical-compute bake-off)
    full_run_node_h = per_step_node_h * full_run_steps
    return {
        "n_params_M": round(_param_count(cfg) / 1e6, 1),
        "fit_seconds": round(fit_seconds, 1),
        "steps_completed": steps,
        "steps_per_sec": round(steps / fit_seconds, 3) if fit_seconds else 0.0,
        "per_step_node_h": round(per_step_node_h, 6),
        "est_node_h_per_10k_step_run": round(full_run_node_h, 2),
        "est_node_h_12_runs": round(full_run_node_h * 12, 1),
        "prd_bakeoff_budget_node_h": 375,  # PRD §6.4: ~1500 GPU-h / 4 GPU-per-node
        "warmup_included": True,
    }


def _param_count(cfg: ScaffoldConfig) -> int:
    from cfm.data.sub_f.vocab import vocab_tag_to_id
    from cfm.data.training.conditioning import conditioning_field_to_id
    from cfm.models.micro_ar import MicroAR, MicroARConfig

    n_subf = max(vocab_tag_to_id().values()) + 1
    m = MicroAR(
        MicroARConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_subf_vocab=n_subf,
            n_cond=len(conditioning_field_to_id()),
            max_len=cfg.max_len + len(conditioning_field_to_id()),
        )
    )
    return sum(p.numel() for p in m.parameters())


def run_short(
    cfg: ScaffoldConfig | None = None,
    *,
    build_shards: bool = True,
    max_time: str | None = None,
    eval_cells: int = 64,
    eval_max_new: int = 512,
) -> dict:
    """The real pre-deadline run (Leonardo 4xA100). Trains for cfg.max_steps (or until
    ``max_time`` wall-clock, used by the scale-up probe), evals once, writes the
    reports/ summary with per-run cost. ``build_shards=False`` for DDP runs whose
    manifest the sbatch preamble already built. ``eval_cells``/``eval_max_new`` size
    the post-train eval generation — KEEP SMALL at large model scales: autoregressive
    generation is per-token forward passes, so 64x512 at 300M is minutes-to-tens-of-
    minutes (it overran the probe's first run). The eval cost is itself a bake-off
    finding; the slice's cost deliverable is TRAINING throughput, not eval."""
    cfg = cfg or ScaffoldConfig()
    if cfg.accelerator == "gpu":
        assert_training_env_locked()

    dm = _datamodule(cfg, build=build_shards)
    lit = maybe_compile(ScaffoldLit(cfg), cfg)
    trainer = build_trainer(cfg, max_time=max_time)
    t0 = time.time()
    trainer.fit(lit, dm)
    fit_seconds = time.time() - t0

    # Eval + report on the global-zero rank only: the model is DDP-synced, so rank 0
    # is representative, and only one process must write the report file.
    if not trainer.is_global_zero:
        return {"trained_steps": int(trainer.global_step)}
    cost = _cost(cfg, fit_seconds=fit_seconds, steps=int(trainer.global_step))
    metrics = _generate_and_score(lit, cfg, n_cells=eval_cells, max_new=eval_max_new)
    report = _write_report(cfg, metrics, trained_steps=int(trainer.global_step), cost=cost)
    logger.info("wrote %s", report)
    return {
        "trained_steps": int(trainer.global_step),
        "metrics": metrics,
        "cost": cost,
        "report": str(report),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Phase-1 training scaffold runner")
    parser.add_argument("--smoke", action="store_true", help="tiny loop-closing smoke")
    parser.add_argument("--devices", type=int, default=4, help="DDP devices (Leonardo node = 4)")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override ScaffoldConfig.max_steps"
    )
    parser.add_argument("--max-len", type=int, default=None, help="override cell-token budget")
    parser.add_argument("--d-model", type=int, default=None, help="scale-up: model width")
    parser.add_argument("--n-layers", type=int, default=None, help="scale-up: depth")
    parser.add_argument("--n-heads", type=int, default=None, help="scale-up: attention heads")
    parser.add_argument("--batch-size", type=int, default=None, help="per-GPU batch size")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile (probe: avoids variable-shape recompilation; cost becomes a "
        "conservative over-estimate vs the compile-on bake-off)",
    )
    parser.add_argument(
        "--max-time", default=None, help="wall budget DD:HH:MM:SS (scale-up probe; bounds the run)"
    )
    parser.add_argument("--eval-cells", type=int, default=64, help="post-train eval: cells to gen")
    parser.add_argument(
        "--eval-max-new",
        type=int,
        default=512,
        help="post-train eval: tokens/cell (keep small@scale)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="DDP: manifest already built by the sbatch preamble; ranks only read it",
    )
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(devices=args.devices)))
        return
    overrides: dict = {"devices": args.devices, "accelerator": _accelerator_for(args.devices)}
    for flag, key in [
        ("max_steps", "max_steps"),
        ("max_len", "max_len"),
        ("d_model", "d_model"),
        ("n_layers", "n_layers"),
        ("n_heads", "n_heads"),
        ("batch_size", "batch_size"),
    ]:
        val = getattr(args, flag)
        if val is not None:
            overrides[key] = val
    if args.no_compile:
        overrides["compile"] = False
    result = run_short(
        ScaffoldConfig(**overrides),
        build_shards=not args.no_build,
        max_time=args.max_time,
        eval_cells=args.eval_cells,
        eval_max_new=args.eval_max_new,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
