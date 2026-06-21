"""Build + seal the Lane-S cell sampler manifest (CPU-only; read-only inputs).

Inputs: the locked conditioning floor (verified-load for lineage) + the per-stratum cell
census parquet (Task 5 emit). Output: a sha-locked sampler-manifest.yaml + marker. Prints the
§6 cost re-derivation. NO generation, NO GPU. Gated on PI word + Leonardo redeploy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cfm.eval.conditioning_floor import load_verified_floor
from cfm.eval.lane_s_sampler import (
    EXPECTED_FLOOR_SHA256,
    build_manifest,
    floored_targets,
    load_verified_manifest,
    read_cell_census,
    seal_manifest,
)

PER_CELL_GPU_H_TRANSFORMER = 0.0045  # GROUND_TRUTH §3 (~600-tok self-terminated, 4-GPU-sharded)
MATRIX_RUNS = 6  # 2 backbones x 3 seeds (one manifest consumed by all)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Build + seal the Lane-S cell sampler manifest.")
    ap.add_argument("--floor", required=True, type=Path, help="Path to conditioning-floor.yaml")
    ap.add_argument(
        "--census", required=True, type=Path, help="Path to heldout-cell-census.parquet"
    )
    ap.add_argument("--out", required=True, type=Path, help="Output path for sampler-manifest.yaml")
    ap.add_argument("--release", required=True, help="Release tag (e.g. 2026-04-15.0)")
    ap.add_argument("--seed", type=int, required=True, help="Blake2b rank seed")
    ap.add_argument(
        "--target-features",
        type=int,
        default=50,
        help="Target gen features per scored stratum (default: 50)",
    )
    ap.add_argument(
        "--headroom",
        type=float,
        default=2.0,
        help="Sizing headroom multiplier (default: 2.0, spec Gate 5 + R3)",
    )
    args = ap.parse_args()

    # Verified-load the floor (raises if internal sha broken — lineage chain)
    verified = load_verified_floor(args.floor)
    floor_payload = verified.payload
    floor_sha = str(floor_payload["floor_sha256"])

    # LOCK-AND-GUARDS-TRAVEL-TOGETHER: the sampler is locked to the 95abb88 floor.
    # A re-derived floor that changes n_a/n_b must update EXPECTED_FLOOR_SHA256 + the
    # Task-4 SoT test in the SAME commit. The build CLI is the hard runtime guard.
    if floor_sha != EXPECTED_FLOOR_SHA256:
        raise SystemExit(
            f"floor sha {floor_sha!r} != pinned {EXPECTED_FLOOR_SHA256!r} — the sampler is "
            "locked to the 95abb88 floor (floor_n READ, never recomputed). A re-derived floor "
            "must update EXPECTED_FLOOR_SHA256 + the Task-4 SoT test in the SAME commit "
            "(lock-and-guards-travel-together); refusing to build against an unverified floor "
            "lineage."
        )

    pool = read_cell_census(args.census)

    payload = build_manifest(
        floor_payload=floor_payload,
        floor_sha256=floor_sha,
        cell_pool=pool,
        release=args.release,
        seed=args.seed,
        target_features=args.target_features,
        headroom=args.headroom,
    )
    seal_manifest(payload, args.out)

    # Round-trip: prove the manifest reads back verified before printing the summary
    loaded = load_verified_manifest(args.out)

    total = sum(s["n_cells_selected"] for s in loaded["strata"])
    ceil_bound = sum(1 for s in loaded["strata"] if s["ceiling_bound"])
    gpu_h = total * MATRIX_RUNS * PER_CELL_GPU_H_TRANSFORMER
    expected = len(floored_targets(verified.payload))
    built = len(loaded["strata"])
    print(f"=== Lane-S sampler built: {args.out} ===")
    print(f"strata={len(loaded['strata'])}  cells_selected={total}  ceiling_bound={ceil_bound}")
    print(f"floored strata: built={built} / expected={expected}")
    if built < expected:
        print(
            f"WARNING: {expected - built} floored strata had NO census cells — "
            "census/floor lineage gap; cost is understated. Investigate before any scored run."
        )
    print(f"generations={total * MATRIX_RUNS} (x{MATRIX_RUNS} runs)")
    print(
        f"est transformer GPU-h={gpu_h:.1f}  (~{100 * gpu_h / 5000:.1f}% of 5,000 GPU-h grant); "
        "MAMBA RATE UNVERIFIED — measure at next GPU smoke"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
