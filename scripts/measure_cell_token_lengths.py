#!/usr/bin/env python3
"""scripts/measure_cell_token_lengths.py — EU cell-token-length stats (Task 15, F13).

Scans each region's sub-F ``tile=*/cells.parquet`` token lengths (NON-EMPTY cells
only) and writes per-city stats — p50 / p99 / p99.9 / max and the fraction of cells
STRICTLY over the cell-token budget (``len > budget``, matching
``flatten_shards_to_cells``'s ``n > max_cell_tokens`` drop) — to a deterministic
YAML artifact (``canonicalize_yaml``; ``derived_at`` = git sha, never a timestamp).

F13 action contract travels with the tool: any city whose ``frac_over_budget``
exceeds ``--halt-threshold`` (default 0.005 — 5x the SG P99.9 design point behind
DEFAULT_MAX_CELL_TOKENS, the same rate at which ``CellDataModule.setup`` raises
DropRateExceeded) makes the CLI exit nonzero, naming the offending cities. The
escalation is the F13 one: raise DEFAULT_MAX_CELL_TOKENS via a recorded decision
or re-chunk.

A region with NO sub-F tile data fails LOUD (names the dir) — never a silent skip
that under-reports the corpus. Needs the real sub-F tile data (Leonardo ``$WORK``)
for the 38-city run; tests inject a synthetic ``sub_f_root_fn``.

    uv run python scripts/measure_cell_token_lengths.py \\
        --release 2026-04-15.0 --regions krakow valencia \\
        --out reports/cell_token_lengths.yaml
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

# iCloud-safe sys.path inject — mirrors scripts/measure_emergence_floor.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.data.sub_g.readers import read_sub_f_cells  # noqa: E402
from cfm.data.training.datamodule import (  # noqa: E402
    DEFAULT_MAX_CELL_TOKENS,
    MAX_TOO_LONG_DROP_RATE,
)
from cfm.eval.holdout.paths import sub_f_region_dir  # noqa: E402

_LOG = logging.getLogger("measure_cell_token_lengths")

_HEADER = (
    "# Per-city sub-F cell-token-length stats (readiness-closure Task 15, F13).\n"
    "# Lengths are NON-EMPTY cells only; frac_over_budget uses len > budget\n"
    "# (STRICT, matching flatten_shards_to_cells's drop). Percentiles are numpy\n"
    "# linear interpolation. Written by scripts/measure_cell_token_lengths.py\n"
    "# (derived_at = git sha; deliberately no timestamp for byte-determinism).\n"
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - mirrors measure_emergence_floor._git_sha
        return "unknown"


def measure_city(
    release: str,
    region: str,
    *,
    sub_f_root_fn=sub_f_region_dir,
    budget: int = DEFAULT_MAX_CELL_TOKENS,
) -> dict:
    """Token-length stats over every ``tile=*/cells.parquet`` under the region's
    sub-F dir (NON-EMPTY cells only).

    Percentiles use numpy's default LINEAR interpolation between order statistics
    (``np.percentile(..., method='linear')``) — documented so a future numpy-method
    change is a recorded decision, not silent drift. ``frac_over_budget`` counts
    ``len > budget`` (strict, the flatten drop predicate). Fail-closed: a region
    dir with no tiles or no non-empty cells raises, naming the dir.
    """
    region_dir = Path(sub_f_root_fn(release, region))
    tile_dirs = sorted(region_dir.glob("tile=*"))
    if not tile_dirs:
        raise FileNotFoundError(
            f"measure_city: no sub-F tile dirs (tile=*) under {region_dir} for "
            f"region {region!r} — refusing to silently report an empty city"
        )
    lengths: list[int] = []
    for tile_dir in tile_dirs:
        cells = read_sub_f_cells(tile_dir / "cells.parquet")
        lengths.extend(len(tokens) for tokens in cells.values() if tokens)
    if not lengths:
        raise ValueError(
            f"measure_city: {len(tile_dirs)} tiles under {region_dir} but zero "
            f"non-empty cells for region {region!r}"
        )
    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "n_cells": int(arr.size),
        "p50": float(np.percentile(arr, 50.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "max": int(arr.max()),
        "frac_over_budget": float(np.count_nonzero(arr > budget) / arr.size),
        "budget": int(budget),
    }


def measure_and_write(
    release: str,
    regions: list[str],
    *,
    out_path: Path,
    sub_f_root_fn=sub_f_region_dir,
    budget: int = DEFAULT_MAX_CELL_TOKENS,
    git_sha: str | None = None,
) -> dict:
    """Measure every region, write the per-city YAML artifact deterministically.

    Output is ``canonicalize_yaml`` (sorted keys, block style) under a fixed header
    — same inputs + same git sha => byte-identical file. Returns the written dict.
    """
    out_path = Path(out_path)
    cities = {
        region: measure_city(release, region, sub_f_root_fn=sub_f_root_fn, budget=budget)
        for region in regions
    }
    data = {
        "schema_version": "1.0",
        "release": release,
        "budget": int(budget),
        "derived_at": git_sha if git_sha is not None else _git_sha(),
        "cities": cities,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_HEADER + canonicalize_yaml(data), encoding="utf-8")
    for region, stats in cities.items():
        _LOG.info(
            "%s: n_cells=%d p50=%.1f p99=%.1f p99.9=%.1f max=%d frac_over_budget=%.6f",
            region,
            stats["n_cells"],
            stats["p50"],
            stats["p99"],
            stats["p99_9"],
            stats["max"],
            stats["frac_over_budget"],
        )
    return data


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--regions", required=True, nargs="+", help="region names to scan")
    parser.add_argument("--out", type=Path, required=True, help="output YAML path")
    parser.add_argument(
        "--halt-threshold",
        type=float,
        default=MAX_TOO_LONG_DROP_RATE,
        help=(
            "exit nonzero if any city's frac_over_budget exceeds this (F13 action "
            "contract; default matches CellDataModule's MAX_TOO_LONG_DROP_RATE)"
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_MAX_CELL_TOKENS,
        help=(
            "cell-token budget the frac_over/drop is measured against (default: the "
            "locked DEFAULT_MAX_CELL_TOKENS); explicit so a recorded-tail-drop "
            "measurement at a candidate number never requires editing a constant"
        ),
    )
    args = parser.parse_args(argv)

    # sub_f_root_fn looked up via the MODULE global (not the def-time default) so
    # tests can monkeypatch the layout without real sub-F tile data.
    data = measure_and_write(
        args.release,
        args.regions,
        out_path=args.out,
        sub_f_root_fn=sub_f_region_dir,
        budget=args.budget,
        git_sha=_git_sha(),
    )
    offenders = sorted(
        (region, stats["frac_over_budget"])
        for region, stats in data["cities"].items()
        if stats["frac_over_budget"] > args.halt_threshold
    )
    if offenders:
        named = ", ".join(f"{region}={frac:.6f}" for region, frac in offenders)
        print(
            f"HALT: frac_over_budget > {args.halt_threshold} for: {named} — "
            f"raise DEFAULT_MAX_CELL_TOKENS via a recorded decision or re-chunk; "
            f"see readiness F13 (stats: {args.out})"
        )
        return 1
    print(f"OK: {len(data['cities'])} cities within halt threshold -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
