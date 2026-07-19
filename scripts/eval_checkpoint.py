"""Standing methodology-eval — ONE checkpoint -> JSON + markdown table.

Three echo-immune metrics (spec docs/superpowers/specs/2026-06-23-standing-eval-harness.md):
perplexity-gap (macro-only PRIMARY + full), saturation, geometry-validity. NOT the Lane-S
crown (no decide/floor/echo). GPU for metrics 1+3; CPU for metric 2.

  uv run python scripts/eval_checkpoint.py --ckpt <last.ckpt> [--smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfm.eval.standing.harness import eval_checkpoint, render_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--logs-dir", default="reports/logs/training-scaffold")
    ap.add_argument("--out", default="reports/_standing_eval")
    ap.add_argument("--n-per-city", type=int, default=2000)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--cells-per-context", type=int, default=7)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="fast end-to-end check: 30 cells/city over <=8 tiles/city, 2 probe cells/context",
    )
    args = ap.parse_args()

    res = eval_checkpoint(
        args.ckpt,
        release=args.release,
        logs_dir=Path(args.logs_dir),
        out_dir=Path(args.out),
        n_per_city=30 if args.smoke else args.n_per_city,
        max_new=args.max_new,
        cells_per_context=2 if args.smoke else args.cells_per_context,
        max_tiles_per_city=8 if args.smoke else None,
    )
    print(render_table(res))
    print("STANDING_EVAL_DONE")


if __name__ == "__main__":
    main()
