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

We record BOTH measures per stratum (PI ruling): the shuffle-GAP (arrangement vs the
interior-permuted null) AND the absolute coherence BAND (the real arrangement scores
themselves), plus the structural ``mean_road_edges`` and a ``dense_core_saturated``
flag. Dense-core strata (#21 inner-core, e.g. munich) saturate the shuffle-null and are
exempt from the tooth-3 separation gate by a structural mean-road-edge threshold (set
from the 60-edge interior capacity, NOT by city name); the full measure is still recorded.

Output YAML:

    release: "2026-04-15.0"
    n_shuffle: 200
    seed: 0
    dense_core_edge_threshold: 40.0
    interior_edge_capacity: 60
    notes: "Dense-core strata saturate the shuffle-null; exempt from tooth-3 ..."
    per_stratum:
      eisenhuttenstadt: {continuity_gap: <f>, fragmentation_gap: <f>,
                         continuity_gap_sd: <f>, fragmentation_gap_sd: <f>,
                         continuity_real: <f>, giant_real: <f>, zoning_real: <f>,
                         mean_road_edges: <f>, dense_core_saturated: <bool>,
                         n_usable_tiles: <int>, n_missing: <int>, n_unreadable: <int>,
                         real_vs_permuted_positive_fraction: <f>}
      glasgow: {...}
      krakow: {...}
      munich: {...}
"""

from __future__ import annotations

import argparse
import logging
import math
import os
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
from cfm.eval.holdout.coherence_reference import (  # noqa: E402
    DENSE_CORE_EDGE_THRESHOLD,
    INTERIOR_EDGE_CAPACITY,
    is_dense_core_saturated,
)
from cfm.eval.holdout.macro_graph import interior_road_graph  # noqa: E402
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

    Missing / unreadable tiles are skipped-and-logged (mirrors
    ``scripts/eval/measure_usable_tiles.py``): they are counted in ``n_missing`` /
    ``n_unreadable`` and the loop continues rather than aborting the whole
    unattended run. Crucially the deterministic ``k`` (the RNG seed index) is bound
    from ``enumerate`` over the FULL sorted stratum enumeration BEFORE any skip, so
    skips never shift the per-tile RNG stream.
    """
    epsg_label = epsg_label_for_region(city)
    region_dir = sub_d_region_dir(release, city)

    continuity_gaps: list[float] = []
    fragmentation_gaps: list[float] = []
    # Absolute coherence band (the real arrangement score, not the shuffle-gap) + the
    # structural road-edge count. Recorded alongside the gap per the PI ruling so the
    # §7 gate sees BOTH the band and the gap, and so the tooth-3 dense-core exemption
    # keys on a structural ``mean_road_edges`` rather than on a city name.
    continuity_reals: list[float] = []
    giant_reals: list[float] = []
    zoning_reals: list[float] = []
    road_edge_counts: list[int] = []
    n_usable = 0
    n_missing = 0
    n_unreadable = 0

    enumerated = list(enumerate(_enumerated_tiles(manifest, city)))
    n_total = len(enumerated)
    for k, (ti, tj) in enumerated:
        if k % 100 == 0:
            logger.info("city %s: %d/%d tiles processed", city, k, n_total)
        tile_dir = region_dir / tile_dirname(ti, tj, epsg_label)
        macro_core = tile_dir / MACRO_CORE_FILENAME
        if not macro_core.exists():
            logger.warning("city %s: %s has no %s; skipping", city, tile_dir, MACRO_CORE_FILENAME)
            n_missing += 1
            continue
        try:
            rows = read_macro_core_parquet(macro_core)
        except Exception as exc:
            logger.warning("city %s: could not read %s: %s", city, tile_dir, exc)
            n_unreadable += 1
            continue
        if not tile_is_usable(rows):
            continue
        n_usable += 1
        # Structural road-edge count: computed ONCE per tile from the shared
        # ``interior_road_graph`` builder (the same source coherence_gap reads), then
        # reused — we never recompute the gap. This is the per-tile interior road-edge
        # count whose stratum mean drives the dense-core saturation exemption.
        road_edge_counts.append(len(interior_road_graph(rows)))
        # Deterministic per-tile RNG: seed + k over the FULL sorted stratum index.
        gaps = coherence_gap(rows, rng=np.random.default_rng(seed + k), n_shuffle=n_shuffle)
        if gaps["continuity_gap"] is not None:
            continuity_gaps.append(float(gaps["continuity_gap"]))
        if gaps["fragmentation_gap"] is not None:
            fragmentation_gaps.append(float(gaps["fragmentation_gap"]))
        # Absolute band (real arrangement scores) from the SAME gap return; skip None
        # (no active edges/cells) so the mean is over the tiles that scored each term.
        if gaps["continuity_real"] is not None:
            continuity_reals.append(float(gaps["continuity_real"]))
        if gaps["giant_real"] is not None:
            giant_reals.append(float(gaps["giant_real"]))
        if gaps["zoning_real"] is not None:
            zoning_reals.append(float(gaps["zoning_real"]))

    # ddof=0 (population std over the held-out usable tiles): the held-out set is the
    # full population being characterised, not a sample drawn from a larger universe.
    continuity_gap = float(np.mean(continuity_gaps)) if continuity_gaps else float("nan")
    fragmentation_gap = float(np.mean(fragmentation_gaps)) if fragmentation_gaps else float("nan")
    continuity_gap_sd = float(np.std(continuity_gaps)) if continuity_gaps else float("nan")
    fragmentation_gap_sd = float(np.std(fragmentation_gaps)) if fragmentation_gaps else float("nan")

    # Absolute coherence band (means of the real arrangement scores).
    continuity_real = float(np.mean(continuity_reals)) if continuity_reals else float("nan")
    giant_real = float(np.mean(giant_reals)) if giant_reals else float("nan")
    zoning_real = float(np.mean(zoning_reals)) if zoning_reals else float("nan")

    # Structural mean interior road-edge count → drives the dense-core saturation flag.
    mean_road_edges = float(np.mean(road_edge_counts)) if road_edge_counts else float("nan")
    dense_core_saturated = is_dense_core_saturated(mean_road_edges) if road_edge_counts else False

    # Tooth-3: fraction of usable tiles with fragmentation_gap > 0 (real beats permuted).
    # Denominator is the usable tiles that produced a fragmentation gap (the scored set).
    positive_fraction = (
        sum(1 for g in fragmentation_gaps if g > 0.0) / len(fragmentation_gaps)
        if fragmentation_gaps
        else float("nan")
    )

    logger.info(
        "stratum %s: n_usable=%d (n_missing=%d n_unreadable=%d) "
        "continuity_gap=%.6f (sd %.6f) "
        "fragmentation_gap=%.6f (sd %.6f) real_vs_permuted_positive_fraction=%.4f "
        "continuity_real=%.6f giant_real=%.6f zoning_real=%.6f "
        "mean_road_edges=%.1f dense_core_saturated=%s [n_shuffle=%d]",
        city,
        n_usable,
        n_missing,
        n_unreadable,
        continuity_gap,
        continuity_gap_sd,
        fragmentation_gap,
        fragmentation_gap_sd,
        positive_fraction,
        continuity_real,
        giant_real,
        zoning_real,
        mean_road_edges,
        dense_core_saturated,
        n_shuffle,
    )

    return {
        # Shuffle-gap (arrangement vs interior-permuted null).
        "continuity_gap": continuity_gap,
        "fragmentation_gap": fragmentation_gap,
        "continuity_gap_sd": continuity_gap_sd,
        "fragmentation_gap_sd": fragmentation_gap_sd,
        # Absolute coherence band (the real arrangement scores themselves).
        "continuity_real": continuity_real,
        "giant_real": giant_real,
        "zoning_real": zoning_real,
        # Structural fields driving the tooth-3 dense-core exemption.
        "mean_road_edges": mean_road_edges,
        "dense_core_saturated": dense_core_saturated,
        # Counts + tooth-3 separation fraction.
        "n_usable_tiles": n_usable,
        "n_missing": n_missing,
        "n_unreadable": n_unreadable,
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
        # Structural threshold metadata: the tooth-3 dense-core exemption keys on
        # mean_road_edges > dense_core_edge_threshold (= 2/3 of the 60-edge interior
        # capacity), a capacity fraction set from the mechanism, NOT by city name.
        "dense_core_edge_threshold": DENSE_CORE_EDGE_THRESHOLD,
        "interior_edge_capacity": INTERIOR_EDGE_CAPACITY,
        "notes": (
            "Dense-core strata (#21 inner-core bbox, e.g. munich) fill a large fraction "
            "of the 60-edge interior capacity, so a random interior rearrangement is itself "
            "near-fully-connected and the shuffle-null SATURATES (real ~ permuted). Their "
            "tooth-3 real-vs-permuted separation collapses toward 0 — a VALIDATION "
            "limitation of the shuffle-null on dense tiles, NOT a power failure (the §7 gate "
            "consumes resolution's KS number + the first-model effect, not this shuffle-gap). "
            "Such strata are recorded in full but EXEMPT from the tooth-3 separation gate by "
            "the structural mean_road_edges > dense_core_edge_threshold rule."
        ),
        "per_stratum": per_stratum,
    }


#: Per-stratum float fields that must be finite before the reference can be locked.
#: Covers the shuffle-gap, the absolute coherence band, and the structural
#: mean_road_edges (the dense-core exemption must not key off a nan). The bool
#: ``dense_core_saturated`` is intentionally absent — the nan check below only
#: inspects float fields, so a bool is naturally skipped.
_NAN_GUARDED_FIELDS = (
    "continuity_gap",
    "fragmentation_gap",
    "continuity_gap_sd",
    "fragmentation_gap_sd",
    "real_vs_permuted_positive_fraction",
    "continuity_real",
    "giant_real",
    "zoning_real",
    "mean_road_edges",
)


def assert_no_nan_reference(per_stratum: dict[str, dict]) -> None:
    """Refuse to lock a nan / empty coherence reference (plan Step-3, in-script).

    Scans every stratum's gap fields for nan and rejects any stratum that produced
    no scored usable tiles (``n_usable_tiles == 0``). Raises ``SystemExit`` so the
    process dies BEFORE any file write — the §7 gate must never calibrate against a
    nan reference, and there must be no partial/nan artifact left on disk.
    """
    bad = {
        c: s
        for c, s in per_stratum.items()
        if s["n_usable_tiles"] == 0
        or any(isinstance(s[k], float) and math.isnan(s[k]) for k in _NAN_GUARDED_FIELDS)
    }
    if bad:
        raise SystemExit(
            f"refusing to lock a nan/empty coherence reference for strata: {sorted(bad)} "
            "(a stratum produced no scored usable tiles; the §7 gate must not calibrate "
            "against nan)"
        )


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

    # Pre-write nan/empty-stratum guard: must fire BEFORE any file is touched so a
    # killed/garbage run never leaves a nan reference on disk for the §7 gate.
    assert_no_nan_reference(payload["per_stratum"])

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: render to a sibling temp path then os.replace, so a killed
    # process never leaves a truncated locked file.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(tmp, out_path)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
