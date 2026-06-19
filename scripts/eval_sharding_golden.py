"""EXECUTION-DEFERRED — eval-sharding GPU equivalence golden (plan Task 11, Step 4).

⛔ DO NOT run on a login node. This constructs models and runs distributed NCCL generation on
4 GPUs — launched ONLY by ``scripts/eval_sharding_golden.sbatch`` (torchrun, 4 ranks, the unified
mamba env). If invoked outside a ≥2-rank distributed launch it refuses (exit 2).

It proves the 4-GPU sharded eval is EQUIVALENT to the rank-0-only baseline before any scored
GPU-hour relies on sharding. It runs the **mamba-hybrid backbone FIRST** — the stateful,
CUDA-kernel-sensitive backbone whose per-cell state-independence under sharding is an ASSUMPTION
to verify on real hardware, not assume — and then transformer-ar, so the golden is never
transformer-only. For each backbone, two teeth + determinism:

  TOOTH 1 (bit-identity): the per-cell WIRED payload (generate+decode via the SAME score_cell
          run_short uses — decoded blocks/geoms/flags, gen_seconds excluded) from the 4-GPU
          sharded path is identical, cell-for-cell, to the rank-0 serial baseline on the SAME
          model. (Keyed on the GLOBAL cell index — seed = BASE + i — so a cell's output must not
          depend on which rank/GPU computed it; for mamba this stresses cross-GPU kernel
          determinism a pure-transformer run would never exercise. This exercises the wired eval
          path, not just the isolated sharded_eval primitive.)
  TOOTH 2 (paired count-conservation): on a RAGGED city count (523, NOT divisible by 4) every
          held-out cell is scored exactly once — no boundary cell dropped or double-counted.
          Aggregate equality alone is insufficient, so the count is checked structurally.
  DETERMINISM: re-running the sharded path yields a byte-identical sequence (so the downstream
          worst-case-city verdict, built on canonical-ordered results + max(), is reproducible).

A green run writes ``reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS`` (rank 0 only), AFTER both
teeth pass for BOTH backbones — never on mere control-flow reaching the end (F8 end-state
discipline). The marker records the per-backbone tooth measurements so the pass is verifiable
from disk, not from the exit code.
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
        "(torchrun --nproc_per_node=4) on $WORK — not standalone, not on login.\n"
    )
    raise SystemExit(2)

import torch  # noqa: E402  (imported only inside the distributed launch)
import torch.distributed as dist  # noqa: E402

from cfm.data.training.conditioning import (  # noqa: E402
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.eval.shard import (  # noqa: E402
    assert_conservation,
    partition_indices,
    sharded_eval,
)
from cfm.models.backbone import subf_vocab_size  # noqa: E402
from cfm.models.mamba_hybrid import MambaHybrid, MambaHybridConfig  # noqa: E402
from cfm.models.micro_ar import MicroAR, MicroARConfig  # noqa: E402

# Import the WIRED per-cell path (generate + decode) the scored eval uses, so this golden
# verifies THAT function across GPUs — not a parallel copy. repo root on sys.path for scripts.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_scaffold import score_cell  # noqa: E402

# --- fixed, deterministic golden inputs ---------------------------------------------------- #
INIT_SEED = 20260618  # identical model init on every rank -> identical weights
GEN_SEED_BASE = 7_000  # per-cell generation seed = GEN_SEED_BASE + global_index (rank-agnostic)
N_RAGGED = 523  # a held-out city count NOT divisible by 4 -> exercises the ragged boundary
MAX_NEW = 32  # short generations; the golden tests equivalence, not realism
# A fixed, no-char-carrier conditioning prefix (n_char_stats=0 path: char_stats=None, 9 ids).
PREFIX = list(range(CONDITIONING_PREFIX_LEN))


def _shared_kwargs() -> dict[str, object]:
    return dict(
        n_subf_vocab=subf_vocab_size(),
        n_cond=conditioning_id_span(),
        max_len=512,
        n_char_stats=0,
        char_position=None,
    )


def _build_mamba_hybrid(device: torch.device) -> MambaHybrid:
    """The stateful, kernel-sensitive backbone the golden exists to verify. Built on CPU
    (construction is CPU-safe) under a FIXED seed so weights are bit-identical on every rank,
    then moved to the rank's GPU. d128/8L/e7 = 1 tf + 7 mamba (real mamba SSM layers)."""
    torch.manual_seed(INIT_SEED)
    cfg = MambaHybridConfig(
        d_model=128, n_layers=8, n_heads=2, transformer_every=7, **_shared_kwargs()
    )
    return MambaHybrid(cfg).to(device).eval()


def _build_transformer_ar(device: torch.device) -> MicroAR:
    """The trivially per-cell-independent backbone (no cross-cell state); the sanity baseline."""
    torch.manual_seed(INIT_SEED)
    cfg = MicroARConfig(d_model=128, n_layers=2, n_heads=2, **_shared_kwargs())
    return MicroAR(cfg).to(device).eval()


def _sync_weights(model: torch.nn.Module) -> None:
    """Broadcast rank-0's weights to every rank so TOOTH 1 isolates sharding/kernel determinism
    (does the same cell on a different GPU produce the same tokens?) and is NOT confounded by any
    construction-RNG difference across ranks."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)


# mamba-hybrid FIRST (the backbone whose sharding equivalence is the real question).
BACKBONES = (("mamba-hybrid", _build_mamba_hybrid), ("transformer-ar", _build_transformer_ar))


def _run_teeth(model: torch.nn.Module, rank: int, world: int) -> dict[str, object] | None:
    """Run both teeth + determinism for one backbone. All ranks participate in the two collective
    sharded passes (matched order); rank 0 additionally computes the serial baseline and returns
    the measurements. Non-zero ranks return None."""

    def score_one(i: int) -> dict:
        # The WIRED per-cell generate+decode (score_cell), keyed on the GLOBAL index i (NOT rank).
        with torch.no_grad():
            return score_cell(
                model, prefix_ids=PREFIX, char_stats=None, max_new=MAX_NEW, seed=GEN_SEED_BASE + i
            )

    def _det(p: dict) -> dict:
        # The DETERMINISTIC per-cell score: decoded blocks/geoms/flags/n_tokens. gen_seconds is
        # wall-clock (differs run-to-run), so it is excluded from every bit-identity comparison.
        return {k: v for k, v in p.items() if k != "gen_seconds"}

    sharded = sharded_eval(N_RAGGED, score_one, rank=rank, world_size=world)  # collective 1
    # TOOTH 2 structural shape — every rank can assert the ragged partition conserves all cells.
    assert_conservation(partition_indices(N_RAGGED, world), N_RAGGED)
    sharded2 = sharded_eval(N_RAGGED, score_one, rank=rank, world_size=world)  # collective 2
    if rank != 0:
        return None
    # TOOTH 1 bit-identity of the WIRED payload vs the rank-0 serial baseline (all cells on one
    # rank): equal decoded output across GPUs is the wired-path equivalence claim (not just tokens).
    baseline = [score_one(i) for i in range(N_RAGGED)]
    mism = [i for i in range(N_RAGGED) if _det(sharded[i]) != _det(baseline[i])]
    return dict(
        tooth1_mismatches=len(mism),
        tooth1_first=(mism[0] if mism else -1),
        tooth2_gathered=len(sharded),
        tooth2_expected=N_RAGGED,
        tooth2_holes=sum(1 for x in sharded if x is None),
        determinism_ok=bool([_det(x) for x in sharded2] == [_det(x) for x in sharded]),
    )


def _passed(r: dict[str, object]) -> bool:
    return (
        r["tooth1_mismatches"] == 0
        and r["tooth2_gathered"] == r["tooth2_expected"]
        and r["tooth2_holes"] == 0
        and r["determinism_ok"] is True
    )


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    results: list[tuple[str, dict[str, object]]] = []
    for name, build in BACKBONES:
        model = build(device)
        _sync_weights(model)  # identical weights on every rank (collective; matched order)
        res = _run_teeth(model, rank, world)
        dist.barrier()  # collective 3 — all ranks, before the next backbone
        if rank == 0 and res is not None:
            results.append((name, res))
            print(
                f"[{name}] TOOTH1 bit-identity: {res['tooth1_mismatches']} mismatch(es) "
                f"(first cell {res['tooth1_first']}; 0=pass) | "
                f"TOOTH2 conservation: gathered {res['tooth2_gathered']}/{res['tooth2_expected']}, "
                f"holes {res['tooth2_holes']} (gathered==expected & 0 holes = pass) | "
                f"determinism: {res['determinism_ok']} | "
                f"{'PASS' if _passed(res) else 'FAIL'}",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    if rank == 0:
        if not results or not all(_passed(r) for _, r in results):
            raise AssertionError(f"SHARDING GOLDEN FAILED — per-backbone measurements: {results}")
        marker = Path("reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS")
        marker.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"eval-sharding golden PASS: world_size={world}, N_ragged={N_RAGGED}"]
        for name, r in results:
            lines.append(
                f"  [{name}] tooth1_mismatches={r['tooth1_mismatches']} | "
                f"tooth2 gathered={r['tooth2_gathered']}/{r['tooth2_expected']} "
                f"holes={r['tooth2_holes']} | determinism={r['determinism_ok']}"
            )
        marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("SHARDING_GOLDEN_PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
