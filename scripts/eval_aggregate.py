"""Standing eval aggregator: combine per-checkpoint JSONs -> 6-way table with seed-noise.

uv run python scripts/eval_aggregate.py --in reports/_standing_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfm.eval.standing.harness import aggregate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="reports/_standing_eval")
    args = ap.parse_args()
    per_ckpt = [
        json.loads(p.read_text())
        for p in sorted(Path(args.inp).glob("*.json"))
        if p.name != "aggregate.json"
    ]
    if not per_ckpt:
        raise SystemExit(f"no per-checkpoint JSONs under {args.inp}")
    table = aggregate(per_ckpt)
    (Path(args.inp) / "aggregate.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
