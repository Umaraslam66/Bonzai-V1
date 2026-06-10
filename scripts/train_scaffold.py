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
import random
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import yaml

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.data.training.build_shards import (
    DEFAULT_G4_ROLLUP,
    build_training_shards,
    train_cities,
)
from cfm.data.training.datamodule import CellDataModule
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import (
    eval_set_locked_marker,
    expected_holdout_schema_for_region,
    holdout_manifest_for_region,
    multiregion_holdout_manifest_path,
)
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


def _g4_rollup_path() -> Path:
    """The production G4 corpus-DoD roll-up — derived from the ONE-source
    ``DEFAULT_G4_ROLLUP`` constant (shared with the Task-8 driver and the
    bakeoff_run.sbatch union preamble, so the preamble verifies the SAME city
    set training consumes). Module-level so tests can monkeypatch it onto a
    synthetic fixture."""
    return Path(DEFAULT_G4_ROLLUP)


def _union_datamodule(cfg: ScaffoldConfig) -> CellDataModule:
    """eu-train-union: the Task-8 per-city manifests get their consumer (F7).

    Resolves the train cities (G4-validated MINUS held-out) and hands their per-city
    manifest paths to CellDataModule's union mode. The multiregion holdout manifest +
    schema "2.0" are pinned DIRECTLY (not via holdout_manifest_for_region, which
    RAISES for train-city regions). This path NEVER rebuilds shards — they are built
    by the dedicated Task-8 sbatch (build_multiregion_train_shards.sbatch); a missing
    manifest surfaces loudly in CellDataModule.setup() (FileNotFoundError) and in the
    bakeoff_run.sbatch preamble existence check."""
    holdout_path = multiregion_holdout_manifest_path(cfg.release)
    holdout = yaml.safe_load(holdout_path.read_text(encoding="utf-8"))
    # STRICT read (fail-closed caller pattern, never .get): train_cities itself reads
    # held_out_cities with .get(..., []) — a manifest missing the key would silently
    # exclude NOTHING and leak the held-out cities into training. Raise here instead.
    if "held_out_cities" not in holdout:
        raise ValueError(
            f"multiregion holdout manifest {holdout_path} has no 'held_out_cities' key; "
            "refusing to build a training union with an empty exclusion set"
        )
    cities = train_cities(cfg.release, g4_rollup=_g4_rollup_path(), holdout_manifest=holdout)
    return CellDataModule(
        training_manifests=[training_manifest_path(cfg.release, c) for c in cities],
        holdout_manifest=holdout_path,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_cell_tokens=cfg.max_len,
        expected_holdout_schema="2.0",
    )


def _datamodule(cfg: ScaffoldConfig, *, build: bool = True) -> CellDataModule:
    # DDP: pre-build the manifest ONCE in the sbatch preamble (single process) and
    # pass build=False to the 4 srun ranks, so they only READ it (no write race).
    # The union path ignores ``build`` entirely: it never rebuilds (see _union_datamodule);
    # calling build_training_shards for a train city would RAISE (the I1 boundary).
    if cfg.train_set == "eu-train-union":
        return _union_datamodule(cfg)
    if build:
        build_training_shards(cfg.release, cfg.region)  # writes the lineage manifest
    return CellDataModule(
        training_manifest=training_manifest_path(cfg.release, cfg.region),
        # REGION-AWARE (obligation (a), delta-spec §3 CORRECTION): manifest AND schema are
        # derived from cfg.region and TRAVEL TOGETHER, so the local SG smoke path stays
        # 1.0/SG (behavior unchanged) while an EU run picks the 2.0 multiregion manifest.
        # The old hazard (a "1.0" pin silently auditing the EU corpus against the wrong
        # holdout — the #16 failure) is now structurally impossible: region selects both.
        holdout_manifest=holdout_manifest_for_region(cfg.release, cfg.region),
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        max_cell_tokens=cfg.max_len,
        expected_holdout_schema=expected_holdout_schema_for_region(cfg.region),
    )


def _generate_and_score(
    model: ScaffoldLit,
    dm: CellDataModule,
    cfg: ScaffoldConfig,
    *,
    n_cells: int,
    max_new: int,
    emergence_floor_per_cell: float | None = None,
) -> dict:
    """Generate cells under MATCHED conditioning, decode via the sealed decoder,
    and score the per-cell slice metrics (decoded / attempted; OGC validity; 90-corner;
    bref-collapse via the shared instrument). One real per-cell number for the loop.

    Matched conditioning (F6, generation side): each generated cell is conditioned on
    a REAL val example's value-bearing prefix (``CellExample.prefix_ids``) and scored
    in that example's stratum (``CellExample.stratum``: density bucket, -1 unknown) —
    no new IO; the datamodule's loaded val examples already carry both. The n_cells
    contexts are sampled from ``dm.val_cells`` deterministically (seeded shuffle,
    cycling when val is smaller than n_cells).

    Diagnostic instrumentation (bake-off Task 4): counts building-token presence in the
    GENERATED streams (the §5 stage-1 truncation discriminator -- did the model emit
    building-class tokens at all, vs they didn't close into polygons), and passes
    ``n_cells`` + ``emergence_floor_per_cell`` so slice_eval emits the §2 emergence verdict.
    """
    from cfm.eval.emergence import sequence_has_building_tokens

    val = dm.val_cells
    if not val:
        raise ValueError(
            "matched-conditioning eval needs >=1 val example to sample conditioning "
            "contexts from; dm.val_cells is empty"
        )
    # DECISION: sample WITH a seeded shuffle (not val order) so a small --eval-cells
    # doesn't always score the lexicographically-first tiles; cycle deterministically
    # when n_cells > len(val). Revisit if eval ever needs stratified sampling.
    order = list(range(len(val)))
    random.Random(cfg.seed).shuffle(order)
    sampled = [val[order[i % len(order)]] for i in range(n_cells)]

    blocks: list[list[int]] = []
    geoms: list[dict] = []
    strata: list[int] = []
    n_attempted = 0
    n_cells_with_building_tokens = 0
    for i, example in enumerate(sampled):
        tokens = generate_cell_tokens(
            model.model, prefix=list(example.prefix_ids), max_new=max_new, seed=cfg.seed + i
        )
        if sequence_has_building_tokens(tokens):
            n_cells_with_building_tokens += 1
        cell_blocks = split_cell_into_features(tokens)
        n_attempted += len(cell_blocks)
        for block in cell_blocks:
            decoded = try_decode_block(block)
            if decoded is not None:
                blocks.append(block)
                geoms.append(decoded)
                strata.append(example.stratum)  # real stratum: density bucket (-1 unknown)
    metrics = slice_eval(
        blocks,
        geoms,
        strata,
        n_attempted_blocks=n_attempted,
        n_cells=n_cells,
        emergence_floor_per_cell=emergence_floor_per_cell,
    )
    # stage-1 truncation discriminator: NO building tokens => never tried / truncated;
    # building tokens present but n_polygons low => didn't close (a different cause).
    metrics["n_cells_generated"] = n_cells
    metrics["n_cells_with_building_tokens"] = n_cells_with_building_tokens
    return metrics


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

    score = _generate_and_score(lit, dm, cfg, n_cells=4, max_new=64)
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
        "UNSCORED-in-slice. Conditioning is MATCHED (F6): each generated cell uses a",
        "real val example's value-bearing prefix and is scored in that example's",
        "density stratum (-1 = unknown); conditioning COMPLIANCE scoring is still a",
        f"bake-off follow-on. Right-angle PoC bar (reported): {_RIGHT_ANGLE_BAR}.",
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
    # Use the ONE backbone factory so the count matches the real model (value-bearing
    # embedding span = conditioning_id_span(), not the 8-field count -- the build_backbone
    # sizing, Task 7). Reconstructing MicroAR by hand here would drift.
    from cfm.models.backbone import build_backbone

    m = build_backbone(cfg.backbone, cfg)
    return sum(p.numel() for p in m.parameters())


def run_short(
    cfg: ScaffoldConfig | None = None,
    *,
    build_shards: bool = True,
    max_time: str | None = None,
    eval_cells: int = 64,
    eval_max_new: int = 512,
    emergence_floor_per_cell: float | None = None,
    ckpt_every_n_steps: int | None = None,
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
    trainer = build_trainer(cfg, max_time=max_time, ckpt_every_n_steps=ckpt_every_n_steps)
    t0 = time.time()
    trainer.fit(lit, dm)
    fit_seconds = time.time() - t0

    # Eval + report on the global-zero rank only: the model is DDP-synced, so rank 0
    # is representative, and only one process must write the report file.
    if not trainer.is_global_zero:
        return {"trained_steps": int(trainer.global_step)}
    cost = _cost(cfg, fit_seconds=fit_seconds, steps=int(trainer.global_step))
    # Time the eval EXPLICITLY: autoregressive generation is the binding bake-off cost,
    # so price it rather than assume it is free (the eval-cost reframe).
    e0 = time.time()
    metrics = _generate_and_score(
        lit,
        dm,
        cfg,
        n_cells=eval_cells,
        max_new=eval_max_new,
        emergence_floor_per_cell=emergence_floor_per_cell,
    )
    eval_seconds = time.time() - e0
    cost["eval_seconds"] = round(eval_seconds, 1)
    cost["eval_node_h"] = round(eval_seconds / 3600.0, 4)
    cost["eval_node_h_per_cell"] = round(eval_seconds / 3600.0 / max(1, eval_cells), 6)
    report = _write_report(cfg, metrics, trained_steps=int(trainer.global_step), cost=cost)
    logger.info("wrote %s", report)
    return {
        "trained_steps": int(trainer.global_step),
        "metrics": metrics,
        "cost": cost,
        "report": str(report),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-1 training scaffold runner")
    parser.add_argument("--smoke", action="store_true", help="tiny loop-closing smoke")
    parser.add_argument("--devices", type=int, default=4, help="DDP devices (Leonardo node = 4)")
    parser.add_argument(
        "--backbone",
        default=None,
        help="bake-off backbone: transformer-ar (default) | mamba-hybrid | discrete-diffusion",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override ScaffoldConfig.max_steps"
    )
    parser.add_argument("--max-len", type=int, default=None, help="override cell-token budget")
    parser.add_argument("--d-model", type=int, default=None, help="scale-up: model width")
    parser.add_argument("--n-layers", type=int, default=None, help="scale-up: depth")
    parser.add_argument("--n-heads", type=int, default=None, help="scale-up: attention heads")
    parser.add_argument("--batch-size", type=int, default=None, help="per-GPU batch size")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="gradient accumulation; holds effective batch constant across scales (§10)",
    )
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
    parser.add_argument(
        "--emergence-floor",
        type=float,
        default=None,
        help="emergence floor (polys/active-cell) for the §2 verdict; diagnostic: 1.96 "
        "(0.25x the real holdout density 7.85)",
    )
    parser.add_argument(
        "--ckpt-every-n-steps",
        type=int,
        default=None,
        help="save step-interval checkpoints (diagnostic: for the emergence-vs-step trajectory)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="override ScaffoldConfig.region (e.g. krakow); default keeps the config value",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="override ScaffoldConfig.release (e.g. 2026-04-15.0); default keeps the config value",
    )
    parser.add_argument(
        "--train-set",
        default=None,
        choices=["single", "eu-train-union"],
        help="training corpus: single (per-region manifest, default) | eu-train-union "
        "(the Task-8 per-city manifest union; never rebuilds)",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> ScaffoldConfig:
    """Pure args->config mapping (no side effects beyond accelerator probing)."""
    overrides: dict = {"devices": args.devices, "accelerator": _accelerator_for(args.devices)}
    for flag, key in [
        ("backbone", "backbone"),
        ("max_steps", "max_steps"),
        ("max_len", "max_len"),
        ("d_model", "d_model"),
        ("n_layers", "n_layers"),
        ("n_heads", "n_heads"),
        ("batch_size", "batch_size"),
        ("grad_accum", "grad_accum"),
        ("region", "region"),
        ("release", "release"),
        ("train_set", "train_set"),
    ]:
        val = getattr(args, flag)
        if val is not None:
            overrides[key] = val
    if args.no_compile:
        overrides["compile"] = False
    return ScaffoldConfig(**overrides)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(devices=args.devices)))
        return
    result = run_short(
        build_config_from_args(args),
        build_shards=not args.no_build,
        max_time=args.max_time,
        eval_cells=args.eval_cells,
        eval_max_new=args.eval_max_new,
        emergence_floor_per_cell=args.emergence_floor,
        ckpt_every_n_steps=args.ckpt_every_n_steps,
    )
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
