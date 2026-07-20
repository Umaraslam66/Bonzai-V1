"""Scored realism-eval generation driver (Task 3) — runs on Leonardo under torchrun.

Joins the sealed Lane-S manifest to the checkpoint's training-style conditioning
(``conditioning.build_conditioned_cells``), generates every cell through the
golden-verified sharded path (``driver.run_generation`` over
``cfm.eval.shard.sharded_eval``), and writes ONE write-once gen artifact
(``driver.write_gen_artifact``) on rank 0. The scored run is a full 4-GPU node
job; ``--dry-run`` (with ``--stratum``/``--limit-cells``) is a single-process
smoke affordance ONLY.

Torch discipline: every torch-touching import (``torch``, ``torch.distributed``,
``load_model``, the generate/decode primitives) is LAZY — inside ``main()`` /
the ``gen_fn`` closure — so ``build_arg_parser``, ``filter_cells`` and
``resolve_ablation`` (the unit-tested surface) import and run WITHOUT a GPU or
torch (mirrors ``scripts/steering_probe_gen.py`` :261-265).

Run (ops; NOT this session):
    torchrun --standalone --nproc_per_node=4 scripts/realism_eval_gen.py \\
        --ckpt <path> --manifest <lane_s_manifest> --out <artifact.json>

A single-process smoke:
    torchrun --standalone --nproc_per_node=1 scripts/realism_eval_gen.py ... --dry-run
    (or plain ``python`` — the dry-run inits a loopback gloo group itself).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from cfm.eval.realism_driver.conditioning import (
    ConditionedCell,
    build_conditioned_cells,
    load_verified_manifest_or_raise,
)
from cfm.eval.realism_driver.driver import (
    read_gen_artifact,
    run_generation,
    write_gen_artifact,
)

logger = logging.getLogger(__name__)

#: Sentinel default for ``--ablation``: read the ablation the checkpoint was
#: trained under and use it (A3). A concrete ``--ablation`` value instead ASSERTS
#: equality with the checkpoint's scheme (a mismatch is a hard ``SystemExit``).
READ_FROM_CKPT = "read-from-ckpt"

#: Overture release the sealed manifest's tiles belong to (join must rebuild the
#: SAME release's shards, else the conditioning join drifts).
DEFAULT_RELEASE = "2026-04-15.0"

#: Base generation seed. Cell ``i`` is generated with ``base_seed + i`` (global-index
#: keying — rank-independent; ``driver.run_generation`` guarantees this).
DEFAULT_BASE_SEED = 20260720

# DECISION (orchestrator review 2026-07-20): --max-new default is 4096, NOT the 13312
# context cap. Measured probe median 394-400 tok/cell, mean ~486, 1.6% hit the probe's
# 1536 cap. 4096 (~8x median) bounds a pathological non-terminating tail to ~+12%
# budget worst-case, while 13312 could add ~40%+. It is a CLI flag — the PI can override
# at submit without a code change; flagged at the budget checkpoint. Revisit if the
# dry-run shows >2% of cells at cap.
DEFAULT_MAX_NEW = 4096

#: Ablation the join was performed under is stamped into the artifact ``meta``.
_SPEC = "realism-eval-gen-v1"

#: rank-0 end-state sentinel (printed to stdout, deliberately, AFTER the artifact is
#: re-read and its count/order/holes verified — no marker without end-state verify).
SENTINEL = "REALISM_EVAL_GEN_DONE"


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI. Kept torch-free so it is unit-testable without a GPU."""
    ap = argparse.ArgumentParser(description="Scored realism-eval generation driver (Task 3).")
    ap.add_argument("--ckpt", required=True, help="bake-off checkpoint (.ckpt) to generate from")
    ap.add_argument("--manifest", required=True, help="sealed Lane-S sampler manifest JSON")
    ap.add_argument("--out", required=True, help="output gen artifact JSON (write-once)")
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="Overture release for the join")
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="seed base (base+i)")
    ap.add_argument(
        "--max-new", type=int, default=DEFAULT_MAX_NEW, help="max generated tokens/cell"
    )
    ap.add_argument(
        "--ablation",
        default=READ_FROM_CKPT,
        help="conditioning ablation; default reads it from the checkpoint (A3). A concrete "
        "value must MATCH the checkpoint's scheme or the run aborts.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="single-process smoke (allows WORLD_SIZE<2 and the --stratum/--limit-cells filters)",
    )
    # --stratum / --limit-cells are DRY-RUN-ONLY affordances. A full scored run uses NO
    # filters (every manifest cell is generated in manifest order).
    ap.add_argument(
        "--stratum",
        default=None,
        help="dry-run only: restrict to cells whose density_bucket == this value",
    )
    ap.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="dry-run only: generate at most this many cells (prefix, in manifest order)",
    )
    return ap


def filter_cells(
    cells: Sequence[ConditionedCell],
    *,
    stratum: str | None = None,
    limit_cells: int | None = None,
) -> list[ConditionedCell]:
    """Apply the dry-run cell filters, preserving manifest order.

    DECISION: ``--stratum`` matches on ``density_bucket`` — the one stratum dimension
    that survives the conditioning join into ``ConditionedCell`` (the floor 4-tuple's
    zoning/skeleton/coastal are dropped in the join; density_bucket is *the* conditioned
    stratum dim per lane_s_sampler). Matched by string so ``--stratum 3`` works as
    given. Revisit if the join is extended to carry the full 4-tuple.
    """
    out = list(cells)
    if stratum is not None:
        out = [c for c in out if str(c.density_bucket) == stratum]
    if limit_cells is not None:
        out = out[:limit_cells]
    return out


def resolve_ablation(requested: str, ckpt_ablation: str) -> str:
    """Resolve the ablation to generate under (A3).

    ``requested == READ_FROM_CKPT`` -> use the checkpoint's own scheme. Any concrete
    value MUST equal the checkpoint's ablation, else :func:`SystemExit` (a mismatch
    would silently condition on a scheme the model never trained under — a hard abort,
    never a warning)."""
    if requested == READ_FROM_CKPT:
        return ckpt_ablation
    if requested != ckpt_ablation:
        raise SystemExit(
            f"--ablation={requested!r} does not match the checkpoint's trained "
            f"conditioning_ablation={ckpt_ablation!r}; refusing to generate under a "
            "scheme the model never saw. Pass the matching ablation or omit --ablation."
        )
    return requested


def _free_tcp_port() -> int:
    """A free loopback TCP port for the dry-run single-process rendezvous."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _init_distributed(*, dry_run: bool):
    """Initialise the process group and pick this rank's device.

    A scored run is launched by ``torchrun`` (WORLD_SIZE>=2, one rank per GPU) and uses
    NCCL. ``sharded_eval`` calls ``all_gather_object`` UNCONDITIONALLY (no group-less
    path), so even a ``--dry-run`` single process needs an initialised group — a loopback
    gloo group of size 1. WORLD_SIZE<2 without ``--dry-run`` is refused (mirrors
    ``eval_sharding_golden.py``: a scored run must saturate the full 4-GPU node).
    """
    import os

    import torch
    import torch.distributed as dist

    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_env < 2 and not dry_run:
        raise SystemExit(
            "realism_eval_gen: refusing WORLD_SIZE<2 for a scored run. Launch via "
            "torchrun --standalone --nproc_per_node=4 (full node), or pass --dry-run for a "
            "single-process smoke."
        )
    if world_env >= 2:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        port = _free_tcp_port()
        dist.init_process_group(
            backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, device


def _make_gen_fn(model, device, *, max_new: int):
    """Build the per-cell generate+decode closure the driver dispatches.

    ``driver.run_generation`` needs a dict with ``tokens``/``blocks``/``geoms``.
    ``scripts.train_scaffold.score_cell`` returns ``blocks``/``geoms`` but only
    ``n_tokens`` (not the raw token tail the driver needs for ``self_terminated``), so
    this closure calls the SAME underlying primitives ``score_cell`` wraps — a single
    ``generate_cell_tokens`` then the identical ``split_cell_into_features`` /
    ``try_decode_block`` decode — and additionally surfaces ``tokens``. One generation
    per cell; no parallel decode implementation."""
    import torch

    from cfm.data.sub_g.seam_decodability import split_cell_into_features
    from cfm.inference.generate import generate_cell_tokens, try_decode_block

    def gen_fn(cell: ConditionedCell, seed: int) -> dict:
        char_stats = list(cell.char_stats) if cell.char_stats else None
        with torch.no_grad():
            tokens = generate_cell_tokens(
                model,
                prefix=list(cell.prefix_ids),
                max_new=max_new,
                seed=seed,
                char_stats=char_stats,
            )
        blocks = split_cell_into_features(tokens)
        decoded = [(b, try_decode_block(b)) for b in blocks]
        return {
            "tokens": tokens,
            "blocks": [b for b, d in decoded if d is not None],
            "geoms": [d for b, d in decoded if d is not None],
        }

    return gen_fn


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)

    if (args.stratum is not None or args.limit_cells is not None) and not args.dry_run:
        raise SystemExit(
            "realism_eval_gen: --stratum/--limit-cells are dry-run-only affordances; a "
            "scored run must generate every manifest cell. Pass --dry-run to subset."
        )

    # Torch-touching imports are LAZY (below): arg-parse + filter stay GPU-free.
    from cfm.eval.standing.harness import load_model

    rank, world, device = _init_distributed(dry_run=args.dry_run)
    try:
        import torch.distributed as dist

        model, model_meta = load_model(args.ckpt, device)
        ckpt_ablation = model_meta["conditioning_ablation"]
        ablation = resolve_ablation(args.ablation, ckpt_ablation)
        ckpt_id = f"{model_meta['backbone']}-seed{model_meta['seed']}"
        logger.info(
            "rank %d/%d loaded %s (ablation=%s) on %s", rank, world, ckpt_id, ablation, device
        )

        manifest = load_verified_manifest_or_raise(Path(args.manifest))
        cells = build_conditioned_cells(manifest, release=args.release, ablation=ablation)
        work = filter_cells(cells, stratum=args.stratum, limit_cells=args.limit_cells)
        logger.info(
            "rank %d: %d conditioned cells (%d after dry-run filter)", rank, len(cells), len(work)
        )
        if not work:
            raise SystemExit("realism_eval_gen: no cells to generate after filtering; aborting.")

        gen_fn = _make_gen_fn(model, device, max_new=args.max_new)
        records = run_generation(
            work, gen_fn, base_seed=args.base_seed, rank=rank, world_size=world
        )

        if rank == 0:
            meta = {
                **model_meta,
                "spec": _SPEC,
                "ckpt_id": ckpt_id,
                "ckpt_path": str(args.ckpt),
                "release": args.release,
                "ablation": ablation,
                "base_seed": args.base_seed,
                "max_new": args.max_new,
                "n_cells": len(work),
                "manifest_path": str(args.manifest),
                "manifest_floor_sha256": manifest["floor_sha256"],
                "manifest_census_sha256": manifest["census_sha256"],
                "world_size": world,
                "dry_run": args.dry_run,
                "stratum_filter": args.stratum,
                "limit_cells": args.limit_cells,
            }
            write_gen_artifact(records, args.out, meta=meta)
            _verify_end_state(args.out, expected=work)
            n_terminated = sum(1 for r in records if r.self_terminated)
            logger.info(
                "wrote %d records -> %s (%d self-terminated)",
                len(records),
                args.out,
                n_terminated,
            )
            # Deliberate stdout sentinel (matches steering_probe_gen's shard-done marker) —
            # emitted ONLY after the artifact was re-read and verified above.
            print(SENTINEL, flush=True)

        dist.barrier()  # keep the group alive until rank 0's write+verify completes
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _verify_end_state(out: str, *, expected: Sequence[ConditionedCell]) -> None:
    """Re-read the just-written artifact and prove it is complete BEFORE the sentinel.

    Asserts the record count equals the expected cell count, the cell_keys are in the
    exact expected (manifest) order, and no slot is a hole (``None``). A false DONE
    poisons every downstream scoring step, so the marker is earned by disk state, never
    by control flow reaching the end (F8)."""
    _meta, records = read_gen_artifact(out)
    if len(records) != len(expected):
        raise SystemExit(
            f"end-state verify FAILED: artifact has {len(records)} records, expected "
            f"{len(expected)} (count not conserved)."
        )
    holes = [i for i, r in enumerate(records) if r is None]
    if holes:
        raise SystemExit(f"end-state verify FAILED: {len(holes)} hole(s), e.g. {holes[:8]}.")
    got = [r.cell_key for r in records]
    want = [c.cell_key for c in expected]
    if got != want:
        first = next(i for i in range(len(want)) if got[i] != want[i])
        raise SystemExit(
            f"end-state verify FAILED: cell order diverges at index {first} "
            f"(got {got[first]}, expected {want[first]})."
        )


if __name__ == "__main__":
    main()
