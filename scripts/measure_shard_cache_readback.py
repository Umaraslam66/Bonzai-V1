"""The W3 payoff measurement: a VERIFIED cache read-back of the full union, timed.

This is exactly what every training-job start will do instead of the ~40-min
features walk: recompute the live key inputs (per-city training-manifest shas),
load through ``load_verified_shard_cache`` (sealed manifest, component
staleness, tier-(b') source verification, full cache-parquet integrity), and
reconstruct every shard. The wall-clock here IS the new per-start cost.

    python scripts/measure_shard_cache_readback.py --release 2026-04-15.0 \
        --out reports/2026-06-12-shard-cache-readback.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.training.build_shards import (
    DEFAULT_G4_ROLLUP,
    verify_union_manifests,
)
from cfm.data.training.paths import training_manifest_path
from cfm.data.training.shard_cache import (
    default_cache_data_root,
    default_cache_root,
    iter_verified_shard_cache,
)
from cfm.eval.holdout.paths import multiregion_holdout_manifest_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    t0 = time.time()
    cities = verify_union_manifests(
        args.release,
        g4_rollup=Path(DEFAULT_G4_ROLLUP),
        holdout_manifest=multiregion_holdout_manifest_path(args.release),
    )
    shas = {c: compute_sha256(training_manifest_path(args.release, c).read_bytes()) for c in cities}
    t_inputs = time.time() - t0

    t1 = time.time()
    # STREAMED city-by-city (the datamodule's real consumption pattern; the
    # whole-union dict peaks >25 GB in Python objects — jobs 46065304/46068302)
    n_cities = n_tiles = n_cells = 0
    for _city, shards in iter_verified_shard_cache(
        default_cache_root(),
        release=args.release,
        cities=cities,
        training_manifest_sha_by_city=shas,
        data_root=default_cache_data_root(),
    ):
        n_cities += 1
        n_tiles += len(shards)
        n_cells += sum(len(s.cells) for s in shards)
    t_load = time.time() - t1

    payload = {
        "measurement": "W3 payoff: verified shard-cache read-back of the full union (streamed)",
        "release": args.release,
        "n_cities": n_cities,
        "n_tiles": n_tiles,
        "n_cells": n_cells,
        "live_key_inputs_seconds": round(t_inputs, 1),
        "verified_load_seconds": round(t_load, 1),
        "total_seconds": round(t_inputs + t_load, 1),
        "replaces": "the ~40-min build_shards_in_memory features walk per job start",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(payload, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
