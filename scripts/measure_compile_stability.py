"""Compile-stability probe (spec §5): does ``torch.compile`` recompile-storm on the
bake-off's variable-length cell batches, or does automatic-dynamic-shape bucketing
converge to a small bounded graph count?

``collate_cells`` right-pads each batch to its OWN max length (per-batch dynamic, NOT a
fixed global max — see ``datamodule.py``), so a compiled model sees a NEW input shape
whenever the batch-max length changes. ``torch.compile`` recompiles per new shape until
automatic-dynamic marks the length dim dynamic and stops. This probe runs a compiled
``MambaHybrid`` over a window (>= ~200 steps) of REAL variable-length training batches and
emits the §5 verdict: keep compile ON for scored runs iff BOTH

  1. recompiles PLATEAU — bounded (target <= ~10) AND none in the final half of the window,
  2. compile OVERHEAD < 10% of the window's wall-clock,

else scored runs go ``--no-compile`` (finding recorded; compile demoted to a later optim).

Detection is TIMING-based and so robust to torch-internal API churn: a (re)compilation is
an observably slow step (kernel codegen), steady-state steps are ~constant. We classify a
step as a compile step when it exceeds a multiple of the steady-state median; recompiles =
the count of such steps, overhead = their excess time over steady. ``torch._dynamo``
counters are logged best-effort as a cross-check, never as the load-bearing signal.

Runs on ONE GPU (mamba kernels require CUDA); shape-driven recompiles are per-process, so a
single-rank probe is representative of the DDP scored regime. Invoked by
``scripts/mamba_smoke.sbatch``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))  # iCloud/spack-safe: don't rely on an editable install
sys.path.insert(0, str(_REPO / "scripts"))  # sibling import of train_scaffold helpers

import torch  # noqa: E402
from train_scaffold import _datamodule  # noqa: E402

from cfm.models.bakeoff_scales import build_pair_for_scale  # noqa: E402
from cfm.models.mamba_hybrid import MambaHybrid  # noqa: E402
from cfm.training.config import ScaffoldConfig  # noqa: E402
from cfm.training.env_lock import assert_mamba_env_locked  # noqa: E402

#: a step slower than this multiple of the steady-state median is a (re)compilation event.
_COMPILE_STEP_FACTOR = 3.0
#: §5 thresholds.
_MAX_RECOMPILES = 10
_MAX_OVERHEAD_FRAC = 0.10


def _dynamo_recompile_count() -> int | None:
    """Best-effort cross-check from torch._dynamo counters; None if the API differs."""
    try:
        import torch._dynamo as dynamo

        stats = dynamo.utils.counters.get("stats", {})
        return int(stats.get("unique_graphs", 0))
    except Exception:
        return None


def measure(
    *,
    region: str,
    release: str,
    train_set: str,
    shard_cache: str | None,
    steps: int,
    batch_size: int,
    scale: str,
) -> dict:
    assert_mamba_env_locked()
    if not torch.cuda.is_available():  # the probe is meaningless without the real kernels
        raise RuntimeError("measure_compile_stability requires CUDA (mamba kernels)")
    device = torch.device("cuda")

    # train_set="eu-train-union" loads the sealed EU cache (the real variable-length cell
    # distribution the scored runs see); the probe needs that distribution, not singapore.
    cfg = ScaffoldConfig(
        backbone="mamba-hybrid",
        region=region,
        release=release,
        train_set=train_set,
        shard_cache=shard_cache,
        batch_size=batch_size,
        accelerator="gpu",
        devices=1,
    )
    dm = _datamodule(cfg, build=False)
    dm.setup("fit")

    _, mcfg = build_pair_for_scale(scale)
    model = MambaHybrid(mcfg).to(device).train()
    compiled = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    times: list[float] = []
    loader = dm.train_dataloader()
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ids = batch["ids"].to(device)
        char = batch["char_stats"].to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = compiled(ids, char_stats=char)
        # COMPILE probe, not a learning run (training quality is the smoke's separate
        # train_scaffold run): a scalar of the logits exercises forward+backward codegen
        # over the real shape without replicating the masked CE.
        loss = logits.float().mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return _verdict(times, n_steps=steps)


def _verdict(times: list[float], *, n_steps: int) -> dict:
    # steady-state per-step time from the tail (post-convergence); guard tiny windows.
    tail = times[-50:] if len(times) >= 50 else times[len(times) // 2 :] or times
    steady = statistics.median(tail)
    threshold = _COMPILE_STEP_FACTOR * steady
    compile_steps = [i for i, t in enumerate(times) if t > threshold]
    recompiles = len(compile_steps)
    total = sum(times)
    overhead = sum(times[i] - steady for i in compile_steps)
    overhead_frac = overhead / total if total > 0 else 0.0
    half = n_steps // 2
    none_in_second_half = all(i < half for i in compile_steps)
    plateaued = none_in_second_half and recompiles <= _MAX_RECOMPILES
    compile_kept = bool(plateaued and overhead_frac < _MAX_OVERHEAD_FRAC)
    return {
        "n_steps": n_steps,
        "recompiles": recompiles,
        "compile_step_indices": compile_steps,
        "steady_step_seconds": round(steady, 5),
        "compile_overhead_frac": round(overhead_frac, 4),
        "recompiles_plateaued": plateaued,
        "none_in_second_half": none_in_second_half,
        "dynamo_unique_graphs": _dynamo_recompile_count(),
        "thresholds": {"max_recompiles": _MAX_RECOMPILES, "max_overhead_frac": _MAX_OVERHEAD_FRAC},
        # THE §5 verdict: keep compile ON for scored runs iff both teeth pass.
        "compile_kept_for_scored_runs": compile_kept,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile-stability probe (spec §5).")
    ap.add_argument("--region", default="krakow")  # a held-out EU city (light); NOT singapore
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--train-set", dest="train_set", default="eu-train-union")
    ap.add_argument("--shard-cache", default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--scale", default="30M")
    ap.add_argument("--out", required=True, help="JSON verdict path")
    args = ap.parse_args()

    verdict = measure(
        region=args.region,
        release=args.release,
        train_set=args.train_set,
        shard_cache=args.shard_cache,
        steps=args.steps,
        batch_size=args.batch_size,
        scale=args.scale,
    )
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"COMPILE_KEPT={verdict['compile_kept_for_scored_runs']} -> {args.out}")


if __name__ == "__main__":
    main()
