"""scripts/eval/measure_coherence_reference.py — per-stratum held-out coherence REFERENCE.

Spec §3.1/§3.5/§10.2. This measures the per-stratum held-out-REAL coherence gap on
the held-out **usable** tiles of the frozen multi-region (EU) holdout. The output is a
*reference input* to the §7 first-model power gate: this script MEASURES and RECORDS
ONLY — it never gates / pass-fails on the reference values (no assertion that a gap is
above/below anything). The §7 gate (at first model) sets the model-vs-real threshold
RELATIVE TO this reference; that is not this script's job.

Threshold-after-measuring: we record the locked numbers AND their measurement regime
(``n_shuffle`` + the per-tile standard deviation of each gap per stratum), because a
locked number is meaningless without the noise floor the gate is built on.

Tooth-3 (real-vs-permuted separation): per stratum we also record the FRACTION of
held-out usable tiles whose ``fragmentation_gap > 0`` (real beats permuted). The GATE
asserting that fraction is >= 0.7 lives in the test, not here — the script measures.

This is a read-only measurement harness; it never writes into the corpus. Run on
Leonardo against the real sub-D corpus (the controller runs the measurement):

    uv run python scripts/eval/measure_coherence_reference.py \
        --release 2026-04-15.0 \
        --n-shuffle 200 \
        --seed 0 \
        --out reports/2026-06-08-coherence-reference.yaml

Output YAML:

    release: "2026-04-15.0"
    n_shuffle: 200
    seed: 0
    per_stratum:
      eisenhuttenstadt: {continuity_gap: <f>, fragmentation_gap: <f>,
                         continuity_gap_sd: <f>, fragmentation_gap_sd: <f>,
                         n_usable_tiles: <int>, real_vs_permuted_positive_fraction: <f>}
      glasgow: {...}
      krakow: {...}
      munich: {...}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# iCloud-safe sys.path inject — mirrors scripts/eval/measure_usable_tiles.py
# (parents[2] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_d.io import read_macro_core_parquet  # noqa: E402
from cfm.eval.holdout.coherence import coherence_gap  # noqa: E402
from cfm.eval.holdout.paths import (  # noqa: E402
    epsg_label_for_region,
    multiregion_holdout_manifest_path,
    sub_d_region_dir,
    tile_dirname,
)
from cfm.eval.usable_tiles import tile_is_usable  # noqa: E402

logger = logging.getLogger("measure_coherence_reference")

#: Per-tile sub-D core artifact filename (spec §11.2 / sub_d/pipeline.py).
MACRO_CORE_FILENAME = "macro_core.parquet"


def _enumerated_tiles(manifest: dict, city: str) -> list[tuple[int, int]]:
    """Held-out ``(tile_i, tile_j)`` keys for ``city`` from the frozen manifest,
    in deterministic sorted (tile_i, tile_j) order (spec §2.1 enumeration)."""
    region = manifest["regions"][city]
    keys = [(int(t["tile_i"]), int(t["tile_j"])) for t in region["tiles"]]
    return sorted(keys)


def measure_stratum(
    manifest: dict,
    release: str,
    city: str,
    *,
    seed: int,
    n_shuffle: int,
) -> dict[str, float | int]:
    """Per-stratum held-out-REAL coherence reference for one held-out city.

    Iterates the city's enumerated tiles in sorted (tile_i, tile_j) order, keeps
    only the usable tiles (``tile_is_usable``), and on each usable tile measures the
    real-vs-permuted coherence gaps with a deterministic per-tile RNG seeded by
    ``seed + k`` where ``k`` is the tile's enumerate index across the full sorted
    stratum list (so the RNG stream is reproducible and independent of which tiles
    turn out usable).

    Returns a dict with the per-stratum aggregates (means, standard deviations,
    usable count, tooth-3 positive fraction). This MEASURES + RECORDS only — it does
    NOT gate / pass-fail on the reference values.
    """
    epsg_label = epsg_label_for_region(city)
    region_dir = sub_d_region_dir(release, city)

    continuity_gaps: list[float] = []
    fragmentation_gaps: list[float] = []
    n_usable = 0

    for k, (ti, tj) in enumerate(_enumerated_tiles(manifest, city)):
        macro_core = region_dir / tile_dirname(ti, tj, epsg_label) / MACRO_CORE_FILENAME
        rows = read_macro_core_parquet(macro_core)
        if not tile_is_usable(rows):
            continue
        n_usable += 1
        # Deterministic per-tile RNG: seed + k over the FULL sorted stratum index.
        gaps = coherence_gap(rows, rng=np.random.default_rng(seed + k), n_shuffle=n_shuffle)
        if gaps["continuity_gap"] is not None:
            continuity_gaps.append(float(gaps["continuity_gap"]))
        if gaps["fragmentation_gap"] is not None:
            fragmentation_gaps.append(float(gaps["fragmentation_gap"]))

    # ddof=0 (population std over the held-out usable tiles): the held-out set is the
    # full population being characterised, not a sample drawn from a larger universe.
    continuity_gap = float(np.mean(continuity_gaps)) if continuity_gaps else float("nan")
    fragmentation_gap = float(np.mean(fragmentation_gaps)) if fragmentation_gaps else float("nan")
    continuity_gap_sd = float(np.std(continuity_gaps)) if continuity_gaps else float("nan")
    fragmentation_gap_sd = float(np.std(fragmentation_gaps)) if fragmentation_gaps else float("nan")

    # Tooth-3: fraction of usable tiles with fragmentation_gap > 0 (real beats permuted).
    # Denominator is the usable tiles that produced a fragmentation gap (the scored set).
    positive_fraction = (
        sum(1 for g in fragmentation_gaps if g > 0.0) / len(fragmentation_gaps)
        if fragmentation_gaps
        else float("nan")
    )

    logger.info(
        "stratum %s: n_usable=%d continuity_gap=%.6f (sd %.6f) "
        "fragmentation_gap=%.6f (sd %.6f) real_vs_permuted_positive_fraction=%.4f "
        "[n_shuffle=%d]",
        city,
        n_usable,
        continuity_gap,
        continuity_gap_sd,
        fragmentation_gap,
        fragmentation_gap_sd,
        positive_fraction,
        n_shuffle,
    )

    return {
        "continuity_gap": continuity_gap,
        "fragmentation_gap": fragmentation_gap,
        "continuity_gap_sd": continuity_gap_sd,
        "fragmentation_gap_sd": fragmentation_gap_sd,
        "n_usable_tiles": n_usable,
        "real_vs_permuted_positive_fraction": positive_fraction,
    }


def measure(release: str, *, seed: int, n_shuffle: int) -> dict[str, object]:
    """Build the full per-stratum reference payload from the frozen manifest."""
    manifest_path = multiregion_holdout_manifest_path(release)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    # Strata = the manifest's held-out cities (sorted, deterministic).
    cities = sorted(manifest["held_out_cities"])
    per_stratum = {
        city: measure_stratum(manifest, release, city, seed=seed, n_shuffle=n_shuffle)
        for city in cities
    }
    return {
        "release": release,
        "n_shuffle": n_shuffle,
        "seed": seed,
        "per_stratum": per_stratum,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", default="2026-04-15.0", help="sub-D release id (default 2026-04-15.0)"
    )
    parser.add_argument(
        "--n-shuffle",
        type=int,
        default=200,
        help="interior-permutation null draws per tile (default 200, PI-locked)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="base seed for the per-tile RNG (default 0)"
    )
    parser.add_argument("--out", required=True, type=Path, help="output YAML path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)
    payload = measure(args.release, seed=args.seed, n_shuffle=args.n_shuffle)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
