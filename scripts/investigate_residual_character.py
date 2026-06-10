#!/usr/bin/env python3
"""scripts/investigate_residual_character.py — PI-ordered reconnaissance (2026-06-11).

PURE INVESTIGATION of the V4-residual finding: after the localization diagnostic
eliminated every cheap single conditioning dimension (V1b zoning, V2 quantization,
V3 sea/untestable, V4 building-size median), cross-city discrimination persists at
~37% of qualifying pairs at delta=0.15. This script characterizes WHAT survives —
it changes nothing, gates nothing, and recommends nothing.

Anchoring (verified-end-state): before reporting anything, the script recomputes
V0 and V4 through the SAME harness (`collect_tile_records` + `variant_features` +
`conditioning_discrimination_verdict`) and HARD-ASSERTS the verified run-3 numbers
(V0: 321 pairs / 141 sig; V4: 1056 / 392). A mismatch aborts the report — the
analysis pipeline is only trusted while it reproduces the blessed run.

Sections of the output YAML:
  delta_sweep        — V0/V4 n_sig + rate at floors 0.15..0.50 (where does it die?)
  pair_structure     — V4@0.15 significant pairs by unordered city pair x metric
  concentration      — KS quantiles + >=0.2/0.3/0.4 counts among V4 sig pairs
  shape_vs_location  — per V4 sig pair: raw KS vs median-normalized KS (samples
                       divided by their own median); if normalization kills a pair
                       the residual there is LOCATION at finer-than-bucket
                       resolution; if KS survives it is SHAPE beyond location
  richer_dims_bound  — in-memory recon variants stacked on V4 (IQR bucket,
                       p90/p50 bucket, count bucket, kitchen-sink) -> rate@0.15;
                       an upper-bound probe on "would a richer single-dim carrier
                       help" (recon-only bucketing, NOT a product proposal)
  road_probe         — the never-probed road residual: sig-pair strata profile,
                       per-city zero-length share (artifact probe), pooled city
                       medians, shape-vs-location for road pairs
  power_probe        — min(n) quantiles among sig vs non-sig pairs; munich
                       involvement (its 171-tile holdout is the power floor)

Zero GPU; held-out tiles only; runs on Leonardo CPU:

    .venv/bin/python scripts/investigate_residual_character.py --release 2026-04-15.0
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import statistics
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path

# iCloud-safe sys.path inject — mirrors the diagnostic script.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from shapely import wkb as shapely_wkb  # noqa: E402

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.data.sub_d.enums import FeatureClass  # noqa: E402
from cfm.eval.conditioning_discrimination import (  # noqa: E402
    DEFAULT_CITIES,
    conditioning_discrimination_verdict,
)
from cfm.eval.holdout.paths import (  # noqa: E402
    epsg_label_for_region,
    holdout_manifest_for_region,
    sub_c_region_dir,
    tile_dirname,
)
from cfm.eval.realism import ks_distance  # noqa: E402

logger = logging.getLogger(__name__)

# Import the diagnostic harness as a module (same pattern as its test suite).
_spec = importlib.util.spec_from_file_location(
    "locdiag", _REPO / "scripts" / "run_localization_diagnostic.py"
)
locdiag = importlib.util.module_from_spec(_spec)
sys.modules["locdiag"] = locdiag
_spec.loader.exec_module(locdiag)

#: Verified run-3 anchors (reports/2026-06-10-localization-diagnostic.yaml @ abde43f).
_ANCHORS = {"V0": (321, 141), "V4": (1056, 392)}
_FLOORS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
_BASE_FLOOR = 0.15


# --------------------------------------------------------------------------- #
# Recon-only bucketing for the richer-dims upper-bound probe.
# DECISION (recon-only, NOT product): IQR reuses the V4 log2 edges (same m^2
# range); p90/p50 ratio uses edges [1.5, 2, 3, 5, 8]; count uses log2 doublings.
# These exist to ask "does ANY single richer summary kill the rate", not to
# propose a scheme.
# --------------------------------------------------------------------------- #


def _iqr_bucket(areas: list[float]) -> int:
    if len(areas) < 2:
        return 0
    qs = statistics.quantiles(areas, n=4)
    return locdiag.building_size_bucket(max(qs[2] - qs[0], 1e-9))


def _p90p50_bucket(areas: list[float]) -> int:
    if len(areas) < 2:
        return 0
    qs = statistics.quantiles(areas, n=10)
    p90, p50 = qs[8], statistics.median(areas)
    if p50 <= 0:
        return 0
    return 1 + bisect_left([1.5, 2.0, 3.0, 5.0, 8.0], p90 / p50)


def _count_bucket(areas: list[float]) -> int:
    c = len(areas)
    return 0 if c == 0 else 1 + min(8, int(math.log2(c)))


def _read_cell_building_areas(path: Path, cells: set[tuple[int, int]]) -> dict:
    """Per-cell building-area LISTS (recon needs full distributions, not medians).

    Same 4-column projected read + BUILDING-filter-before-WKB as the diagnostic's
    `_read_cell_building_sizes`; keyed over the sub-C cells universe.
    """
    table = pq.ParquetFile(path).read(columns=["cell_i", "cell_j", "feature_class", "geometry"])
    pooled: dict[tuple[int, int], list[float]] = {}
    fi = table["cell_i"].to_pylist()
    fj = table["cell_j"].to_pylist()
    fc = table["feature_class"].to_pylist()
    fg = table["geometry"].to_pylist()
    for i, j, c, g in zip(fi, fj, fc, fg, strict=True):
        if int(c) != int(FeatureClass.BUILDING):
            continue
        pooled.setdefault((int(i), int(j)), []).append(float(shapely_wkb.loads(bytes(g)).area))
    return {cell: pooled.get(cell, []) for cell in cells}


def _collect_area_lists(release: str, cities: list[str]) -> list[tuple[str, dict]]:
    """Per-tile (city, {cell: [areas]}) in the SAME walk order as
    ``collect_tile_records`` (cities order x manifest tile order).

    TileRecord carries no tile coordinates, so the richer-dims probe joins
    positionally; ``_richer_dims_bound`` hard-guards the alignment per record
    (city match + cell-universe equality vs the record's sea layer, which is
    keyed over the identical cells.parquet universe). Valid only at 0 skipped
    tiles — guarded by the caller against coverage.
    """
    out: list[tuple[str, dict]] = []
    for city in cities:
        man = yaml.safe_load(holdout_manifest_for_region(release, city).read_text())
        epsg = epsg_label_for_region(city)
        base = sub_c_region_dir(release, city)
        for t in man["regions"][city]["tiles"]:
            dn = tile_dirname(int(t["tile_i"]), int(t["tile_j"]), epsg)
            cells_tab = pq.ParquetFile(base / dn / "cells.parquet").read(
                columns=["cell_i", "cell_j"]
            )
            ci = cells_tab["cell_i"].to_pylist()
            cj = cells_tab["cell_j"].to_pylist()
            universe = {(int(i), int(j)) for i, j in zip(ci, cj, strict=True)}
            out.append((city, _read_cell_building_areas(base / dn / "features.parquet", universe)))
    return out


# --------------------------------------------------------------------------- #
# Analyses (pure; operate on the harness's own outputs)
# --------------------------------------------------------------------------- #


def _verdict(features: dict, floor: float):
    return conditioning_discrimination_verdict(
        features, min_n=50, alpha=0.05, effect_size_floor=floor
    )


def _rate(sig: int, pairs: int) -> float | None:
    return None if pairs == 0 else sig / pairs


def _delta_sweep(feats_by_variant: dict) -> dict:
    out: dict = {}
    for name, feats in feats_by_variant.items():
        rows = {}
        for f in _FLOORS:
            r = _verdict(feats, f)
            rows[f"{f:.2f}"] = {
                "n_pairs": r.n_qualifying_comparisons,
                "n_significant_effect": r.n_significant_effect,
                "rate": _rate(r.n_significant_effect, r.n_qualifying_comparisons),
            }
        out[name] = rows
    return out


def _pair_structure(result) -> dict:
    sig = Counter()
    tot = Counter()
    for p in result.pairs:
        key = f"{p.city_a}|{p.city_b}|{p.metric}"
        tot[key] += 1
        if p.significant:
            sig[key] += 1
    return {
        k: {"n_pairs": tot[k], "n_significant": sig.get(k, 0), "rate": _rate(sig.get(k, 0), tot[k])}
        for k in sorted(tot)
    }


def _concentration(result) -> dict:
    out = {}
    for metric in sorted({p.metric for p in result.pairs}):
        kss = sorted(p.ks for p in result.pairs if p.significant and p.metric == metric)
        if not kss:
            out[metric] = {"n": 0}
            continue
        out[metric] = {
            "n": len(kss),
            "ks_p25": kss[len(kss) // 4],
            "ks_median": kss[len(kss) // 2],
            "ks_p90": kss[int(len(kss) * 0.9)],
            "ks_max": kss[-1],
            "n_ge_0.20": sum(k >= 0.20 for k in kss),
            "n_ge_0.30": sum(k >= 0.30 for k in kss),
            "n_ge_0.40": sum(k >= 0.40 for k in kss),
        }
    return out


def _shape_vs_location(result, feats: dict) -> dict:
    """Median-normalize each significant pair's samples and re-measure KS.

    samples / their own median -> location removed; surviving KS == shape.
    """
    out = {}
    for metric in sorted({p.metric for p in result.pairs}):
        rows = []
        n_skipped = 0
        for p in result.pairs:
            if not p.significant or p.metric != metric:
                continue
            a = feats[(p.city_a, p.stratum, metric)]
            b = feats[(p.city_b, p.stratum, metric)]
            ma, mb = statistics.median(a), statistics.median(b)
            if ma <= 0 or mb <= 0:
                n_skipped += 1
                continue
            ks_norm = ks_distance([x / ma for x in a], [x / mb for x in b])
            rows.append((p.ks, ks_norm))
        if not rows:
            out[metric] = {"n": 0, "n_skipped_zero_median": n_skipped}
            continue
        norm = sorted(k for _, k in rows)
        out[metric] = {
            "n": len(rows),
            "n_skipped_zero_median": n_skipped,
            "ks_norm_median": norm[len(norm) // 2],
            "ks_norm_p90": norm[int(len(norm) * 0.9)],
            "n_norm_below_floor": sum(k < _BASE_FLOOR for k in norm),
            "n_norm_at_or_above_floor": sum(k >= _BASE_FLOOR for k in norm),
            "frac_location_dominated": sum(k < _BASE_FLOOR for k in norm) / len(norm),
        }
    return out


def _richer_dims_bound(records, area_lists: list[tuple[str, dict]], v4_rate: float) -> dict:
    """Stack recon buckets on V4's stratum; rate@0.15 per stacked variant."""
    if len(records) != len(area_lists):
        raise RuntimeError(
            f"alignment broken: {len(records)} records vs {len(area_lists)} tile walks"
        )
    for rec, (city, areas_by_cell) in zip(records, area_lists, strict=True):
        if rec.city != city or set(rec.cell_sea) != set(areas_by_cell):
            raise RuntimeError(
                f"alignment broken at a {rec.city}/{city} tile: cell universes differ"
            )

    def stacked_features(extra) -> dict:
        feats: dict = {}
        for rec, (_city, areas_by_cell) in zip(records, area_lists, strict=True):
            for metric, value, cell in rec.features:
                density = int(rec.cell_density[cell])
                areas = areas_by_cell.get(cell, [])
                stratum = (
                    rec.tile_zoning,
                    rec.tile_skeleton,
                    density,
                    rec.tile_coastal,
                    locdiag.building_size_bucket(statistics.median(areas) if areas else None),
                    *extra(areas),
                )
                feats.setdefault((rec.city, stratum, metric), []).append(value)
        return feats

    # Self-anchor: restacking V4 through THIS path (no extra dims) must reproduce
    # the verified 1056/392 — proves the parallel walk's medians match the
    # diagnostic's reader before any richer-dim number is trusted.
    base = _verdict(stacked_features(lambda a: ()), _BASE_FLOOR)
    if (base.n_qualifying_comparisons, base.n_significant_effect) != _ANCHORS["V4"]:
        raise RuntimeError(
            "restacked-V4 anchor mismatch: "
            f"({base.n_qualifying_comparisons}, {base.n_significant_effect}) != "
            f"{_ANCHORS['V4']} — area-list walk NOT trusted; aborting."
        )

    probes = {
        "V4_plus_iqr": lambda a: (_iqr_bucket(a),),
        "V4_plus_p90p50": lambda a: (_p90p50_bucket(a),),
        "V4_plus_count": lambda a: (_count_bucket(a),),
        "V4_kitchen_sink": lambda a: (_iqr_bucket(a), _p90p50_bucket(a)),
    }
    out = {"V4_baseline_rate": v4_rate, "V4_restacked_anchor": "reproduced"}
    for name, extra in probes.items():
        r = _verdict(stacked_features(extra), _BASE_FLOOR)
        out[name] = {
            "n_pairs": r.n_qualifying_comparisons,
            "n_significant_effect": r.n_significant_effect,
            "rate": _rate(r.n_significant_effect, r.n_qualifying_comparisons),
        }
    return out


def _road_probe(records, result, feats: dict) -> dict:
    zero_share = {}
    medians = {}
    pooled: dict[str, list[float]] = {}
    for rec in records:
        for metric, value, _cell in rec.features:
            if metric == "road_length_m":
                pooled.setdefault(rec.city, []).append(value)
    for city, vals in sorted(pooled.items()):
        zero_share[city] = sum(v == 0.0 for v in vals) / len(vals)
        medians[city] = statistics.median(vals)
    sig_road = [p for p in result.pairs if p.significant and p.metric == "road_length_m"]
    by_density = Counter(str(p.stratum[2]) for p in sig_road)
    by_skeleton = Counter(str(p.stratum[1]) for p in sig_road)
    return {
        "zero_length_share_by_city": zero_share,
        "pooled_median_m_by_city": medians,
        "sig_pairs_by_density_bucket": dict(sorted(by_density.items())),
        "sig_pairs_by_skeleton_class": dict(sorted(by_skeleton.items())),
        "shape_vs_location": _shape_vs_location(result, feats).get("road_length_m"),
    }


def _power_probe(result) -> dict:
    def quantiles(ns: list[int]) -> dict:
        ns = sorted(ns)
        if not ns:
            return {"n": 0}
        return {
            "n": len(ns),
            "min_n_p25": ns[len(ns) // 4],
            "min_n_median": ns[len(ns) // 2],
            "min_n_p90": ns[int(len(ns) * 0.9)],
        }

    sig = [min(p.n_a, p.n_b) for p in result.pairs if p.significant]
    non = [min(p.n_a, p.n_b) for p in result.pairs if not p.significant]
    mun_all = sum(1 for p in result.pairs if "munich" in (p.city_a, p.city_b))
    mun_sig = sum(1 for p in result.pairs if p.significant and "munich" in (p.city_a, p.city_b))
    return {
        "sig_pairs_min_n": quantiles(sig),
        "nonsig_pairs_min_n": quantiles(non),
        "munich_share_of_pairs": _rate(mun_all, len(result.pairs)),
        "munich_share_of_sig": _rate(mun_sig, len(sig)) if sig else None,
    }


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="V4-residual reconnaissance (read-only)")
    parser.add_argument("--release", default="2026-04-15.0")
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--report-out", default="reports/2026-06-11-residual-character-recon.yaml")
    args = parser.parse_args(argv)

    logger.info("collecting tile records (same harness as the diagnostic)")
    records, coverage = locdiag.collect_tile_records(args.release, args.cities)

    feats = {name: locdiag.variant_features(records, name) for name in ("V0", "V4")}
    results = {name: _verdict(f, _BASE_FLOOR) for name, f in feats.items()}

    # HARD ANCHOR: refuse to report from an unverified pipeline.
    for name, (want_pairs, want_sig) in _ANCHORS.items():
        got = (results[name].n_qualifying_comparisons, results[name].n_significant_effect)
        if got != (want_pairs, want_sig):
            raise RuntimeError(
                f"anchor mismatch for {name}: got {got}, verified run-3 says "
                f"({want_pairs}, {want_sig}) — analysis pipeline NOT trusted; aborting."
            )
    logger.info("run-3 anchors reproduced (V0 321/141, V4 1056/392) — proceeding")

    logger.info("reading per-cell building-area lists for the richer-dims probe")
    area_lists = _collect_area_lists(args.release, args.cities)

    v4 = results["V4"]
    report = {
        "anchors_reproduced": {k: list(v) for k, v in _ANCHORS.items()},
        "delta_sweep": _delta_sweep(feats),
        "pair_structure_v4": _pair_structure(v4),
        "concentration_v4": _concentration(v4),
        "shape_vs_location_v4": _shape_vs_location(v4, feats["V4"]),
        "richer_dims_bound": _richer_dims_bound(
            records, area_lists, _rate(v4.n_significant_effect, v4.n_qualifying_comparisons)
        ),
        "road_probe": _road_probe(records, v4, feats["V4"]),
        "power_probe_v4": _power_probe(v4),
        "tile_coverage": {
            c: {"expected": cov.n_tiles_expected, "read": cov.n_tiles_read}
            for c, cov in sorted(coverage.items())
        },
    }

    out = (
        _REPO / args.report_out
        if not Path(args.report_out).is_absolute()
        else Path(args.report_out)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonicalize_yaml(report), encoding="utf-8")
    print(f"recon report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
