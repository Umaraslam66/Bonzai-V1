"""One-off tier-(a) cost measurement for the shard-cache read verification (W3, PI call #1).

Tier (a) = full re-hash of EVERY keyed source file at every cache read; tier (b)
= full re-hash small + size-check all + seeded-sample re-hash big. Tier (b) is
the provisional implementation; THIS measurement (the real union, on Leonardo)
decides whether tier (a) is cheap enough to upgrade to. Read-only; writes one
small YAML report.

    python scripts/measure_shard_cache_verify_cost.py --release 2026-04-15.0 \
        --out reports/2026-06-12-shard-cache-tier-a-cost.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from cfm.data.training.build_shards import (
    DEFAULT_G4_ROLLUP,
    verify_union_manifests,
)
from cfm.data.training.paths import training_manifest_path
from cfm.data.training.shard_cache import (
    SMALL_SOURCE_FILE_BYTES,
    default_cache_data_root,
    source_file_records,
)
from cfm.eval.holdout.paths import multiregion_holdout_manifest_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measure_shard_cache_verify_cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stat-only",
        action="store_true",
        help="measure the stat-walk variant (existence+size on every keyed file, "
        "NO hashing) — the tier-(b') candidate's all-files component",
    )
    args = parser.parse_args(argv)

    cities = verify_union_manifests(
        args.release,
        g4_rollup=Path(DEFAULT_G4_ROLLUP),
        holdout_manifest=multiregion_holdout_manifest_path(args.release),
    )
    per_city = {}
    tot_files = tot_bytes = tot_small = 0
    t0 = time.time()
    for city in cities:
        manifest = yaml.safe_load(training_manifest_path(args.release, city).read_text("utf-8"))
        tile_ids = [(int(t["tile_i"]), int(t["tile_j"])) for t in manifest.get("tiles", [])]
        c0 = time.time()
        if args.stat_only:
            # tier-(b') all-files component: stat every keyed file (no read/hash)
            import cfm.data.training.shard_cache as sc

            roots = {
                "sub_d": sc.sub_d_region_dir(args.release, city),
                "sub_f": sc.sub_f_region_dir(args.release, city),
                "sub_c": sc.sub_c_region_dir(args.release, city),
            }
            epsg = sc.epsg_label_for_region(city)
            records = []
            for ti, tj in sorted(tile_ids):
                dirname = sc.tile_dirname(ti, tj, epsg)
                for sub, fname in sc._TILE_SOURCE_FILES:
                    st = (roots[sub] / dirname / fname).stat()
                    records.append({"size": st.st_size})
        else:
            # source_file_records IS tier (a): it reads + hashes every keyed file
            records = source_file_records(
                args.release, city, tile_ids, data_root=default_cache_data_root()
            )
        seconds = time.time() - c0
        n_bytes = sum(r["size"] for r in records)
        n_small = sum(1 for r in records if r["size"] <= SMALL_SOURCE_FILE_BYTES)
        per_city[city] = {
            "files": len(records),
            "bytes": n_bytes,
            "small_files": n_small,
            "seconds": round(seconds, 2),
        }
        tot_files += len(records)
        tot_bytes += n_bytes
        tot_small += n_small
        logger.info("%s: %d files, %.1f MB, %.1fs", city, len(records), n_bytes / 2**20, seconds)
    total_s = time.time() - t0
    payload = {
        "measurement": (
            "shard-cache tier-(b') stat-walk cost (stat ALL keyed files, no hashing)"
            if args.stat_only
            else "shard-cache tier-(a) read-verification cost (full re-hash, real union)"
        ),
        "release": args.release,
        "n_cities": len(cities),
        "total_files": tot_files,
        "total_small_files_le_64k": tot_small,
        "total_gib": round(tot_bytes / 2**30, 2),
        "total_seconds": round(total_s, 1),
        "total_minutes": round(total_s / 60.0, 2),
        "per_city": per_city,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({k: v for k, v in payload.items() if k != "per_city"}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
