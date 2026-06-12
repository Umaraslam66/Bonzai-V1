"""Build the sealed shard-derivation cache (W3; standalone, lrd_all_serial).

One-time CPU job: derives every train city's shards (the ~40-min features walk,
paid HERE instead of at every training-job start), serializes them per city,
verifies by re-read + seeded from-source re-derivation, and seals the union
cache manifest (locked_yaml grammar) + ``_SHARD_CACHE_VALID``.

Usage (Leonardo, from the repo root):
    python scripts/build_shard_cache.py --release 2026-04-15.0
    # cities default to the G4-rollup union minus held-out (the Task-8 set)

Deliberately STANDALONE (PI call #3): the sealed Task-8 manifest builder's scope
is manifests, not payloads — this script never writes a training manifest.
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
    build_shard_cache,
    default_cache_data_root,
    default_cache_root,
)
from cfm.eval.holdout.paths import multiregion_holdout_manifest_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_shard_cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="override the city set (default: G4-rollup union minus held-out)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="cache root (default: data/processed/training_cache — $WORK, never $SCRATCH)",
    )
    parser.add_argument(
        "--sample-cells-per-city",
        type=int,
        default=8,
        help="seeded from-source re-derivation sample per city (the seal gate)",
    )
    args = parser.parse_args(argv)

    if args.cities is not None:
        cities = sorted(args.cities)
    else:
        cities = verify_union_manifests(
            args.release,
            g4_rollup=Path(DEFAULT_G4_ROLLUP),
            holdout_manifest=multiregion_holdout_manifest_path(args.release),
        )
    logger.info("building shard cache for %d cities", len(cities))

    tile_ids_by_city: dict[str, list[tuple[int, int]]] = {}
    manifest_sha_by_city: dict[str, str] = {}
    for city in cities:
        p = training_manifest_path(args.release, city)
        raw = p.read_bytes()
        manifest_sha_by_city[city] = compute_sha256(raw)
        manifest = yaml.safe_load(raw.decode("utf-8"))
        tile_ids_by_city[city] = [
            (int(t["tile_i"]), int(t["tile_j"])) for t in manifest.get("tiles", [])
        ]

    t0 = time.time()
    manifest_path = build_shard_cache(
        args.release,
        cities,
        tile_ids_by_city=tile_ids_by_city,
        cache_root=args.cache_root or default_cache_root(),
        manifest_sha_by_city=manifest_sha_by_city,
        sample_cells_per_city=args.sample_cells_per_city,
        data_root=default_cache_data_root(),
    )
    logger.info(
        "SEALED %s in %.1f min (%d cities, %d tiles)",
        manifest_path,
        (time.time() - t0) / 60.0,
        len(cities),
        sum(len(v) for v in tile_ids_by_city.values()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
