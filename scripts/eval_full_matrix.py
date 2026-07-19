"""Standing eval — FULL 6-checkpoint matrix with a shared held-out cache.

Builds + VERIFIES the held-out cache once (cached≡uncached gate), then evals all 6
checkpoints (transformer-ar-53M, mamba-hybrid-54M x seed 7/13/23) reusing that cache,
and writes the aggregate T-vs-M table with seed-noise + the effective-shuffle fraction
and UNRELIABLE flags. Resumable: a checkpoint whose JSON already exists is skipped.

  uv run python scripts/eval_full_matrix.py   # on a GPU node (see .sbatch)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfm.eval.standing.harness import HELD_OUT_CITIES, aggregate, eval_checkpoint
from cfm.eval.standing.heldout_cells import build_and_verify_cache

# the 6 bake-off checkpoints (backbone dir, seed); dir name carries the rounded param tag.
MATRIX = [("transformer-ar-53M", s) for s in (7, 13, 23)] + [
    ("mamba-hybrid-54M", s) for s in (7, 13, 23)
]


def _expected_id(backbone_dir: str, seed: int) -> str:
    return f"{backbone_dir.rsplit('-', 1)[0]}-seed{seed}"  # strip the -53M/-54M tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt-root",
        default="/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff",
    )
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--logs-dir", default="reports/logs/training-scaffold")
    ap.add_argument("--out", default="reports/_standing_eval")
    ap.add_argument("--n-per-city", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "heldout_cache.json"

    # ── GATE: build + verify the cache (cached≡uncached) before any checkpoint runs ──
    if cache.exists():
        print(f"[matrix] reusing existing cache {cache}", flush=True)
    else:
        n = build_and_verify_cache(
            args.release, HELD_OUT_CITIES, n_per_city=args.n_per_city, cache_path=cache
        )
        print(f"CACHE_VERIFIED n_cells={n} (read-back byte-identical to fresh load)", flush=True)

    # ── eval the 6 (skip any already done) ──
    for backbone_dir, seed in MATRIX:
        ckpt_id = _expected_id(backbone_dir, seed)
        if (out / f"{ckpt_id}.json").exists():
            print(f"[matrix] skip {ckpt_id} (already done)", flush=True)
            continue
        ckpt = Path(args.ckpt_root) / backbone_dir / f"krakow-seed{seed}" / "last.ckpt"
        print(f"[matrix] eval {ckpt_id} <- {ckpt}", flush=True)
        eval_checkpoint(
            str(ckpt),
            release=args.release,
            logs_dir=Path(args.logs_dir),
            out_dir=out,
            n_per_city=args.n_per_city,
            heldout_cache=cache,
        )
        print(f"[matrix] done {ckpt_id}", flush=True)

    # ── aggregate ──
    per_ckpt = [json.loads((out / f"{_expected_id(b, s)}.json").read_text()) for b, s in MATRIX]
    table = aggregate(per_ckpt)
    (out / "aggregate.md").write_text(table)
    print("\n" + table, flush=True)
    print("FULL_MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
