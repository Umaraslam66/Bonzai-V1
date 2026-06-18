"""EXECUTION-DEFERRED — eval-sharding GPU equivalence golden (plan Task 11, Step 4).

⛔ DO NOT run during the CINECA ``$WORK`` outage, and do NOT run on a login node. This script
constructs a (tiny) model and runs distributed NCCL generation on 4 GPUs — it is launched ONLY
by ``scripts/eval_sharding_golden.sbatch`` (torchrun, 4 ranks, the unified mamba env) once
``$WORK`` heavy I/O has recovered, bundled with the 50M param-match verify. Run nothing here by
hand; if invoked outside a ≥2-rank distributed launch it refuses (exit 2).

It proves the 4-GPU sharded eval is EQUIVALENT to the rank-0-only baseline before any scored
GPU-hour relies on sharding — two teeth, plus determinism:

  TOOTH 1 (bit-identity): the per-cell generated token sequence from the 4-GPU sharded path is
          identical, cell-for-cell, to the rank-0 serial baseline on the SAME model. (Generation
          is keyed on the GLOBAL cell index — seed = BASE + i — so a cell's output must not
          depend on which rank computed it; this verifies that empirically on real hardware.)
  TOOTH 2 (paired count-conservation): on a RAGGED city count (523, NOT divisible by 4) every
          held-out cell is scored exactly once — no boundary cell dropped or double-counted.
          Aggregate equality alone is insufficient, so the count is checked structurally.
  DETERMINISM: re-running the sharded path yields a byte-identical sequence (so the downstream
          worst-case-city verdict, built on canonical-ordered results + max(), is reproducible).

A green run writes the marker ``reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS`` (rank 0 only),
AFTER both teeth pass — never on mere control-flow reaching the end (F8 end-state discipline).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- hard guard: this is a distributed GPU golden, never a plain script run ---------------- #
_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
if _WORLD < 2:
    sys.stderr.write(
        "eval_sharding_golden: EXECUTION-DEFERRED. Launch via scripts/eval_sharding_golden.sbatch "
        "(torchrun --nproc_per_node=4) on $WORK after recovery — not standalone, not on login.\n"
    )
    raise SystemExit(2)

import torch  # noqa: E402  (imported only inside the distributed launch)
import torch.distributed as dist  # noqa: E402

from cfm.data.training.conditioning import (  # noqa: E402
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.eval.shard import partition_indices, sharded_eval  # noqa: E402
from cfm.inference.generate import generate_cell_tokens  # noqa: E402
from cfm.models.backbone import subf_vocab_size  # noqa: E402
from cfm.models.micro_ar import MicroAR, MicroARConfig  # noqa: E402

# --- fixed, deterministic golden inputs ---------------------------------------------------- #
INIT_SEED = 20260618  # identical model init on every rank -> identical weights
GEN_SEED_BASE = 7_000  # per-cell generation seed = GEN_SEED_BASE + global_index (rank-agnostic)
N_RAGGED = 523  # a held-out city count NOT divisible by 4 -> exercises the ragged boundary
MAX_NEW = 32  # short generations; the golden tests equivalence, not realism
# A fixed, no-char-carrier conditioning prefix (n_char_stats=0 path: char_stats=None, 9 ids).
PREFIX = list(range(CONDITIONING_PREFIX_LEN))


def _build_identical_model(device: torch.device) -> MicroAR:
    """A tiny transformer-ar built with the SAME seed on every rank, so weights are bit-identical
    across ranks (the premise of the bit-identity tooth). No char-carrier (keeps the prefix at the
    9 conditioning ids + char_stats=None)."""
    torch.manual_seed(INIT_SEED)
    cfg = MicroARConfig(
        d_model=128,
        n_layers=2,
        n_heads=2,
        n_subf_vocab=subf_vocab_size(),
        n_cond=conditioning_id_span(),
        max_len=512,
        n_char_stats=0,
        char_position=None,
    )
    model = MicroAR(cfg).to(device)
    model.eval()
    return model


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = _build_identical_model(device)

    def score_one(i: int) -> tuple[int, ...]:
        # Per-cell generated token sequence, keyed on the GLOBAL index i (NOT the rank).
        with torch.no_grad():
            toks = generate_cell_tokens(
                model, prefix=PREFIX, max_new=MAX_NEW, seed=GEN_SEED_BASE + i, char_stats=None
            )
        return tuple(toks)

    # TOOTH 1 — sharded path vs rank-0 serial baseline, bit-identical per cell.
    sharded = sharded_eval(N_RAGGED, score_one, rank=rank, world_size=world)
    if rank == 0:
        baseline = [score_one(i) for i in range(N_RAGGED)]  # rank-0 serial, all cells
        mism = [i for i in range(N_RAGGED) if sharded[i] != baseline[i]]
        if mism:
            raise AssertionError(
                f"TOOTH 1 FAILED: {len(mism)} cell(s) differ between 4-GPU sharded and rank-0 "
                f"baseline (first: cell {mism[0]}) — sharding is NOT score-equivalent."
            )

    # TOOTH 2 — count-conservation on the ragged partition (structural, on the REAL run).
    from cfm.eval.shard import assert_conservation

    assert_conservation(partition_indices(N_RAGGED, world), N_RAGGED)
    if len(sharded) != N_RAGGED or any(x is None for x in sharded):
        raise AssertionError(
            f"TOOTH 2 FAILED: gathered {len(sharded)} cells (expected {N_RAGGED}) or a hole — "
            "a cell was dropped or double-counted across the ragged shards."
        )

    # DETERMINISM — re-run the sharded path; the canonical-ordered sequence must be byte-identical.
    sharded2 = sharded_eval(N_RAGGED, score_one, rank=rank, world_size=world)
    if sharded2 != sharded:
        raise AssertionError("DETERMINISM FAILED: sharded eval is not reproducible across runs.")

    dist.barrier()
    if rank == 0:
        marker = Path("reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"eval-sharding golden PASS: world_size={world}, N_ragged={N_RAGGED}, "
            f"bit-identity vs rank-0 + ragged count-conservation + determinism all green.\n",
            encoding="utf-8",
        )
        print("SHARDING_GOLDEN_PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
