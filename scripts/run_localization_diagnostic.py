#!/usr/bin/env python3
"""scripts/run_localization_diagnostic.py — coarseness-localization diagnostic (F5; spec §4.2).

WHERE does city character live in the conditioning stratum? This script recomputes
the recalibrated gate-(i) verdict (``conditioning_discrimination_verdict``, δ effect
floor + BH guard) under one-layer-at-a-time stratum VARIANTS on the held-out EU
cities — the variant that kills the most discrimination signal localizes the layer:

  V0     baseline — the EXACT gate-(i) stratum
         (dominant_zoning, modal_skeleton, per-cell 4-bucket density, coastal);
         must reproduce the gate-(i) extraction's feature pool byte-for-byte.
  V1     un-collapse — (per_cell_zoning, per_cell_density) REPLACE the tile-level
         dims (each feature is assigned its OWN cell's stratum).
  V1b    attribution: zoning swap, dims KEPT — V0 with only the zoning slot
         un-collapsed (per_cell_zoning, modal_skeleton, per-cell density, coastal).
  V1d    attribution: dims DROPPED, zoning kept — (tile_zoning, per-cell density);
         V0 with the skeleton+coastal dims pooled away. Together V1b/V1d decompose
         V1's two simultaneous changes (Task-23 step-6 PI request). The PI-requested
         "V1c = per-cell-density-only swap" is identical to V0 by construction
         (V0's density slot is ALREADY per-cell) — see the methodology note.
  V2_8   un-quantize — V0's 4-bucket density slot replaced by an 8-bucket
  V2_16  (resp. 16-bucket) equal-width index over the raw building_footprint_ratio
         (derivation_evidence.parquet).
  V3     candidate dim — V0's tuple PLUS an appended per-cell sea_water_fraction
         bucket (sub-C cells.parquet).

Uniformity contracts (the variant comparison is otherwise confounded):
  - The outbound-bref construction-identity exclusion (gate-(i) Task 22) is applied
    ONCE at feature classification, so every variant sees the SAME feature pool.
  - Missing-artifact policy: a tile missing sub-F ``cells.parquet`` is skipped+counted
    (the F3 coverage counters, mirroring the reference extraction); but a tile that
    HAS sub-F cells and is missing ``derivation_evidence.parquet`` or the sub-C
    ``cells.parquet`` is a LOUD FileNotFoundError — variants must never silently see
    different tile sets (denominator integrity).

The script REPORTS (YAML + stdout table); it does not verdict-gate. Exit code 0.
The PI reads the table at the step-6 halt-gate; the Task-24 character feature is
chosen BY this data (characterize-before-recommend).

Runs on Leonardo CPU against the real corpus (step 5); locally it is exercised
against synthetic fixtures in tests/scripts/test_localization_diagnostic.py.

    uv run python scripts/run_localization_diagnostic.py \
        --release 2026-04-15.0 \
        --cities eisenhuttenstadt glasgow krakow munich
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/run_gate_i_conditioning_discrimination.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.data.sub_c.epsilon import EPS_RATIO  # noqa: E402
from cfm.data.sub_d.enums import SlotKind  # noqa: E402
from cfm.data.sub_d.io import (  # noqa: E402
    read_derivation_evidence_parquet,
    read_macro_core_parquet,
)
from cfm.data.sub_d.lattice import CELL_GRID_SIZE  # noqa: E402
from cfm.data.sub_f.decoder import decode_feature  # noqa: E402
from cfm.data.sub_g.readers import read_sub_f_cells  # noqa: E402

# Private imports sanctioned by the Task-22/23 plan: _has_outbound_bref is the
# construction-identity AUTHORITY for the v1 outbound-bref artifact, and
# _SHRINKAGE_CEILING / _cell_density_by_cell keep this walk one-sourced with the
# gate-(i) reference extraction it must reproduce at V0.
from cfm.data.sub_g.seam_decodability import (  # noqa: E402
    _has_outbound_bref,
    split_cell_into_features,
)
from cfm.eval.conditioning_discrimination import (  # noqa: E402
    _SHRINKAGE_CEILING,
    DEFAULT_CITIES,
    ConditioningDiscriminationResult,
    TileCoverage,
    conditioning_discrimination_verdict,
)
from cfm.eval.geometry import promote_building_rings  # noqa: E402
from cfm.eval.holdout.labels import read_tile_labels  # noqa: E402
from cfm.eval.holdout.paths import (  # noqa: E402
    epsg_label_for_region,
    holdout_manifest_for_region,
    sub_c_region_dir,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
)
from cfm.eval.holdout.pipeline import _cell_density_by_cell  # noqa: E402
from cfm.eval.realism import _LINE_TYPES, _POLYGON_TYPES, FeatureMetric  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_RELEASE = "2026-04-15.0"
_DEFAULT_REPORT_OUT = "reports/2026-06-10-localization-diagnostic.yaml"

#: The locked variant set (plan Task 23 + step-6 V1b/V1d attribution decomposition).
#: Order is the report/table order.
VARIANTS: tuple[str, ...] = ("V0", "V1", "V1b", "V1d", "V2_8", "V2_16", "V3")

_METRICS: tuple[str, ...] = tuple(m.value for m in FeatureMetric)

#: The raw density metric in derivation_evidence.parquet (sub_d/evidence.py).
_RATIO_METRIC = "building_footprint_ratio"


# --------------------------------------------------------------------------- #
# Per-tile record (pure-function input; IO-free past collect_tile_records)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TileRecord:
    """One held-out tile's labels, per-cell layers, and classified features.

    ``features`` are ``(metric, value, cell)`` triples — the SAME bref-excluded
    pool every variant strata over (uniformity contract, module docstring).
    """

    city: str
    tile_zoning: int | None
    tile_skeleton: int | None
    tile_coastal: int | None
    cell_density: dict[tuple[int, int], int]
    cell_zoning: dict[tuple[int, int], int | None]
    cell_ratio: dict[tuple[int, int], float]
    cell_sea: dict[tuple[int, int], float]
    features: list[tuple[str, float, tuple[int, int]]]


# --------------------------------------------------------------------------- #
# Bucketing (PURE)
# --------------------------------------------------------------------------- #


def ratio_bucket(ratio: float, n_buckets: int) -> int:
    """N-bucket index for a raw building_footprint_ratio (V2 un-quantize).

    # DECISION: equal-width buckets over [0, 1] with the top edge inclusive;
    # ratios above 1 are pathological-only and clip into the last bucket.
    # Chose equal-width over quantile edges because the variant compares
    # QUANTIZATION granularity, not edge placement, and equal-width is
    # data-independent (deterministic across cities). Revisit if the real
    # corpus shows heavy mass piling into one bucket at V2_16.
    """
    if ratio < 0.0:
        raise ValueError(f"building_footprint_ratio < 0 is impossible by construction: {ratio}")
    return min(int(ratio * n_buckets), n_buckets - 1)


def sea_bucket(sea_fraction: float) -> int:
    """3-bucket index for per-cell sea_water_fraction (V3 candidate dim).

    # DECISION: {<= EPS_RATIO -> 0, (EPS_RATIO, 0.5] -> 1, > 0.5 -> 2}. The
    # 0-boundary is structural and gets the EPS_RATIO (1e-9) treatment sub-C's
    # validator applies at the same boundary (_check_water_fraction_bounds);
    # 0.5 is a chosen edge -> strict comparison (epsilon-at-structural-
    # boundaries convention). Revisit if the halt-gate wants a finer coastal
    # gradient than land/partial/majority-sea.
    """
    if sea_fraction <= EPS_RATIO:
        return 0
    if sea_fraction <= 0.5:
        return 1
    return 2


# --------------------------------------------------------------------------- #
# Variant stratification (PURE)
# --------------------------------------------------------------------------- #


def _require(
    layer: dict[tuple[int, int], object], cell: tuple[int, int], what: str, city: str
) -> object:
    """Loud lookup: a feature's cell missing from a per-cell layer invalidates the
    variant comparison (denominator integrity) — never a silent skip.

    Returns ``object`` (the layers are heterogeneous); call sites cast (int/float)."""
    if cell not in layer:
        raise KeyError(
            f"localization diagnostic: city '{city}' cell {cell} has features but no "
            f"{what} entry — the variant comparison is invalid if variants silently "
            "see different feature sets."
        )
    return layer[cell]


def variant_features(
    records: list[TileRecord], variant: str
) -> dict[tuple[str, tuple, str], list[float]]:
    """Strata the SAME feature pool under one variant's stratum definition.

    Returns the ``features`` mapping ``conditioning_discrimination_verdict`` takes:
    ``(city, stratum_tuple, metric) -> list[float]``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    features: dict[tuple[str, tuple, str], list[float]] = {}
    for rec in records:
        for metric, value, cell in rec.features:
            density = int(_require(rec.cell_density, cell, "cell_density_bucket", rec.city))
            if variant == "V0":
                stratum = (rec.tile_zoning, rec.tile_skeleton, density, rec.tile_coastal)
            elif variant == "V1":
                # Un-collapse: the cell's OWN zoning + density REPLACE the tile dims.
                zoning = _require(rec.cell_zoning, cell, "per-cell zoning_class", rec.city)
                stratum = (zoning, density)
            elif variant == "V1b":
                # Attribution (zoning swap, dims KEPT): V0 with only the zoning
                # slot un-collapsed to the cell's own value.
                zoning = _require(rec.cell_zoning, cell, "per-cell zoning_class", rec.city)
                stratum = (zoning, rec.tile_skeleton, density, rec.tile_coastal)
            elif variant == "V1d":
                # Attribution (dims DROPPED, zoning kept): V0 with skeleton+coastal
                # pooled away; tile-dominant zoning retained.
                stratum = (rec.tile_zoning, density)
            elif variant in ("V2_8", "V2_16"):
                n = 8 if variant == "V2_8" else 16
                ratio = float(_require(rec.cell_ratio, cell, _RATIO_METRIC, rec.city))
                stratum = (
                    rec.tile_zoning,
                    rec.tile_skeleton,
                    ratio_bucket(ratio, n),
                    rec.tile_coastal,
                )
            else:  # V3
                sea = float(_require(rec.cell_sea, cell, "sea_water_fraction", rec.city))
                stratum = (
                    rec.tile_zoning,
                    rec.tile_skeleton,
                    density,
                    rec.tile_coastal,
                    sea_bucket(sea),
                )
            features.setdefault((rec.city, stratum, metric), []).append(value)
    return features


def diagnose(
    records: list[TileRecord],
    *,
    min_n: int,
    alpha: float,
    effect_size_floor: float,
) -> dict[str, dict]:
    """Per-variant recalibrated verdict + the per-metric localization table.

    Per variant: ``verdict``, raw-BH vs effect significance counts, per metric
    ``n_pairs`` / ``n_significant_effect`` / ``median_ks`` (None if no pairs), and
    the per-city TOTAL feature n (the step-5 ±20%-of-V0 denominator sanity check).
    """
    out: dict[str, dict] = {}
    for variant in VARIANTS:
        feats = variant_features(records, variant)
        result = conditioning_discrimination_verdict(
            feats, min_n=min_n, alpha=alpha, effect_size_floor=effect_size_floor
        )
        n_by_city: dict[str, int] = {}
        for (city, _stratum, _metric), vals in feats.items():
            n_by_city[city] = n_by_city.get(city, 0) + len(vals)
        out[variant] = {
            "verdict": result.verdict,
            "n_qualifying_comparisons": result.n_qualifying_comparisons,
            "n_significant_raw_bh": result.n_significant_raw_bh,
            "n_significant_effect": result.n_significant_effect,
            "per_metric": _per_metric_summary(result),
            "n_features_by_city": dict(sorted(n_by_city.items())),
        }
    return out


def _per_metric_summary(result: ConditioningDiscriminationResult) -> dict[str, dict]:
    per_metric: dict[str, dict] = {}
    for metric in _METRICS:
        pairs = [p for p in result.pairs if p.metric == metric]
        per_metric[metric] = {
            "n_pairs": len(pairs),
            "n_significant_effect": sum(1 for p in pairs if p.significant),
            "median_ks": (float(statistics.median(p.ks for p in pairs)) if pairs else None),
        }
    return per_metric


# --------------------------------------------------------------------------- #
# Per-cell feature classification (mirrors decode_region_blocks + the gate-(i)
# _tile_features, but KEEPS the cell key — decode_region_blocks collapses it)
# --------------------------------------------------------------------------- #


def _tile_cell_features(
    tokens_by_cell: dict[tuple[int, int], list[int]],
    cell_density_by_cell: dict[tuple[int, int], int],
) -> tuple[list[tuple[str, float, tuple[int, int]]], int]:
    """Classify one tile's decoded features as ``(metric, value, cell)`` triples.

    Mirrors ``decode_region_blocks`` (sorted cell walk; a cell with no recorded
    cell_density_bucket is SKIPPED, not bucketed as 0 — so V0 reproduces the
    gate-(i) feature counts) and the gate-(i) ``_tile_features`` classification:
    building closed rings promoted to Polygon (construction identity) before
    typing; outbound-bref line features EXCLUDED by ``_has_outbound_bref`` on the
    ORIGINAL token block and COUNTED, never silently dropped.
    """
    blocks: list[list[int]] = []
    geoms: list[dict] = []
    cells: list[tuple[int, int]] = []
    for cell, token_sequence in sorted(tokens_by_cell.items()):
        if cell_density_by_cell.get(cell) is None:
            continue
        for block in split_cell_into_features(token_sequence):
            blocks.append(block)
            geoms.append(decode_feature(block))
            cells.append(cell)

    promoted = promote_building_rings(blocks, geoms)
    out: list[tuple[str, float, tuple[int, int]]] = []
    n_bref_excluded = 0
    for block, geom, cell in zip(blocks, promoted, cells, strict=True):
        g = shape(geom)
        if g.geom_type in _POLYGON_TYPES:
            out.append((FeatureMetric.BUILDING_AREA.value, float(g.area), cell))
        elif g.geom_type in _LINE_TYPES:
            if _has_outbound_bref(block):
                n_bref_excluded += 1
                continue
            out.append((FeatureMetric.ROAD_LENGTH.value, float(g.length), cell))
    return out, n_bref_excluded


# --------------------------------------------------------------------------- #
# IO walk (thin; mirrors the gate-(i) reference extraction)
# --------------------------------------------------------------------------- #


def _read_cell_sea_fractions(path: Path) -> dict[tuple[int, int], float]:
    """Per-cell sea_water_fraction from a sub-C cells.parquet.

    ``pq.ParquetFile(path).read()`` — NEVER bare ``pq.read_table``: Hive partition
    inference on ``tile=...`` dirs injects a spurious ``tile`` column (established
    project correction).
    """
    table = pq.ParquetFile(path).read()
    ci = table.column("cell_i").to_pylist()
    cj = table.column("cell_j").to_pylist()
    sea = table.column("sea_water_fraction").to_pylist()
    return {(int(i), int(j)): float(s) for i, j, s in zip(ci, cj, sea, strict=True)}


def _read_cell_ratios(path: Path) -> dict[tuple[int, int], float]:
    """Per-cell raw building_footprint_ratio from derivation_evidence.parquet.

    CELL rows store ``slot_index = cell_i * CELL_GRID_SIZE + cell_j``
    (sub_d/evidence.py ``derive_density_evidence``) — recovered by divmod.
    """
    rows = read_derivation_evidence_parquet(path)
    return {
        divmod(int(r.slot_index), CELL_GRID_SIZE): float(r.value)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.metric_name == _RATIO_METRIC
    }


def collect_tile_records(
    release: str, cities: list[str]
) -> tuple[list[TileRecord], dict[str, TileCoverage]]:
    """Read every held-out tile's layers + features into ``TileRecord``s.

    Mirrors ``extract_features_by_city_stratum_metric``: per-region CRS label on
    every ``tile_dirname`` call; a tile missing sub-F ``cells.parquet`` is
    skipped+counted; the end-of-walk F3 checks (zero-tile city, silent-shrinkage
    ceiling) are identical. A present sub-F tile missing the diagnostic's extra
    inputs (derivation_evidence / sub-C cells) raises FileNotFoundError (LOUD;
    module docstring policy).
    """
    records: list[TileRecord] = []
    tile_coverage: dict[str, TileCoverage] = {}

    for city in cities:
        manifest = yaml.safe_load(holdout_manifest_for_region(release, city).read_text())
        tiles = manifest["regions"][city]["tiles"]
        epsg = epsg_label_for_region(city)
        sub_c = sub_c_region_dir(release, city)
        sub_d = sub_d_region_dir(release, city)
        sub_f = sub_f_region_dir(release, city)
        n_read = 0
        n_skipped = 0
        n_bref_excluded = 0

        for tile in tiles:
            ti, tj = int(tile["tile_i"]), int(tile["tile_j"])
            dirname = tile_dirname(ti, tj, epsg)
            cells_path = sub_f / dirname / "cells.parquet"
            if not cells_path.exists():
                logger.warning(
                    "missing sub-F cells for %s tile (%d,%d): %s", city, ti, tj, cells_path
                )
                n_skipped += 1
                continue

            evidence_path = sub_d / dirname / "derivation_evidence.parquet"
            sub_c_cells_path = sub_c / dirname / "cells.parquet"
            for required in (evidence_path, sub_c_cells_path):
                if not required.exists():
                    raise FileNotFoundError(
                        f"localization diagnostic: {city} tile ({ti},{tj}) HAS sub-F "
                        f"cells but is missing {required} — the variant comparison is "
                        "invalid if variants silently see different tile sets "
                        "(denominator integrity; module docstring policy)."
                    )

            labels = read_tile_labels(sub_d / dirname, tile_i=ti, tile_j=tj)
            cdbc = _cell_density_by_cell(sub_d / dirname)
            # Per-cell zoning over the SAME density-bucketed cells the feature walk
            # keeps (mirrors _cell_density_by_cell's None-filter).
            macro_rows = read_macro_core_parquet(sub_d / dirname / "macro_core.parquet")
            cell_zoning = {
                (int(r.cell_i), int(r.cell_j)): (
                    int(r.zoning_class) if r.zoning_class is not None else None
                )
                for r in macro_rows
                if r.slot_kind == SlotKind.CELL and r.cell_density_bucket is not None
            }

            tokens = read_sub_f_cells(cells_path)
            features, n_excluded = _tile_cell_features(tokens, cdbc)
            n_bref_excluded += n_excluded
            records.append(
                TileRecord(
                    city=city,
                    tile_zoning=labels.morphology_stratum.dominant_zoning_class,
                    tile_skeleton=labels.morphology_stratum.modal_road_skeleton_class,
                    tile_coastal=labels.coastal_inland_river,
                    cell_density=cdbc,
                    cell_zoning=cell_zoning,
                    cell_ratio=_read_cell_ratios(evidence_path),
                    cell_sea=_read_cell_sea_fractions(sub_c_cells_path),
                    features=features,
                )
            )
            n_read += 1

        tile_coverage[city] = TileCoverage(
            n_tiles_expected=len(tiles),
            n_tiles_read=n_read,
            n_tiles_skipped=n_skipped,
            n_bref_excluded=n_bref_excluded,
        )

    # End-of-walk F3 checks: identical to the gate-(i) reference extraction.
    for city, cov in tile_coverage.items():
        if cov.n_tiles_expected == 0:
            raise ValueError(
                f"localization diagnostic: city '{city}' has zero tiles in its holdout "
                "manifest — a zero-tile city is never a valid extraction target."
            )
        frac = cov.n_tiles_skipped / cov.n_tiles_expected
        if frac > _SHRINKAGE_CEILING:
            raise RuntimeError(
                f"localization diagnostic: city '{city}' skipped "
                f"{cov.n_tiles_skipped}/{cov.n_tiles_expected} held-out tiles "
                f"(missing cells.parquet; fraction {frac:.3f} > silent-shrinkage "
                f"ceiling {_SHRINKAGE_CEILING}). A partial city must be re-extracted "
                "or explicitly excluded, never quietly thinned (readiness F3)."
            )

    return records, tile_coverage


# --------------------------------------------------------------------------- #
# Report (YAML + stdout table)
# --------------------------------------------------------------------------- #


def _methodology(
    *, release: str, cities: list[str], min_n: int, alpha: float, effect_size_floor: float
) -> dict:
    """The methodology block the PI reads at the halt-gate — every scheme serialized."""
    variants: dict[str, dict] = {
        "V0": {
            "description": "baseline: the exact gate-(i) stratum",
            "stratum": [
                "dominant_zoning_class",
                "modal_road_skeleton_class",
                "cell_density_bucket",
                "coastal_inland_river",
            ],
        },
        "V1": {
            "description": (
                "un-collapse: per-cell zoning + per-cell density REPLACE the "
                "tile-level dims (each feature gets its own cell's stratum)"
            ),
            "stratum": ["per_cell_zoning_class", "per_cell_density_bucket"],
        },
        "V1b": {
            "description": (
                "attribution (zoning swap, dims KEPT): V0 with only the zoning "
                "slot un-collapsed to the cell's own zoning_class — isolates V1's "
                "tile->cell zoning change"
            ),
            "stratum": [
                "per_cell_zoning_class",
                "modal_road_skeleton_class",
                "cell_density_bucket",
                "coastal_inland_river",
            ],
        },
        "V1d": {
            "description": (
                "attribution (dims DROPPED, zoning kept): V0 with the "
                "skeleton+coastal dims pooled away, tile-dominant zoning retained "
                "— isolates V1's dim-drop change"
            ),
            "stratum": ["dominant_zoning_class", "cell_density_bucket"],
        },
    }
    # V2_8 / V2_16 differ ONLY in bucket count: one definition, two instantiations.
    for n_buckets in (8, 16):
        article = "an" if n_buckets == 8 else "a"
        variants[f"V2_{n_buckets}"] = {
            "description": (
                f"un-quantize: V0 with the 4-bucket density slot replaced by {article} "
                f"{n_buckets}-bucket index over raw building_footprint_ratio"
            ),
            "density_bucket_scheme": {
                "scheme": "equal_width",
                "range": [0.0, 1.0],
                "n_buckets": n_buckets,
                "top_edge": "inclusive",
                "overflow": "ratios > 1 (pathological) clip into the last bucket",
            },
        }
    variants["V3"] = {
        "description": (
            "candidate dim: V0 PLUS an appended per-cell sea_water_fraction "
            "bucket (sub-C cells.parquet)"
        ),
        "sea_bucket_scheme": {
            "bucket_0": f"sea_water_fraction <= EPS_RATIO ({EPS_RATIO})",
            "bucket_1": "(EPS_RATIO, 0.5]",
            "bucket_2": "> 0.5",
        },
    }
    return {
        "release": release,
        "cities": list(cities),
        "min_n": min_n,
        "alpha": alpha,
        "effect_size_floor": effect_size_floor,
        "bref_exclusion": (
            "outbound-bref line features excluded by construction identity "
            "(_has_outbound_bref on the ORIGINAL token block), counted per city, "
            "applied ONCE so every variant sees the SAME feature pool"
        ),
        "v1c_note": (
            "the PI-requested 'V1c = per-cell-density-only swap' is IDENTICAL to "
            "V0 by construction and is therefore not run: the gate's density slot "
            "is already per-cell (each feature's own cell_density_bucket via "
            "_cell_density_by_cell / macro_core cell_density_bucket), so swapping "
            "it tile->cell changes nothing. The correct decomposition of V1 is "
            "the 2x2 over {tile vs cell zoning} x {dims kept vs dropped}: "
            "V0 (tile zoning, dims kept), V1b (cell zoning, dims kept), "
            "V1d (tile zoning, dims dropped), V1 (cell zoning, dims dropped)."
        ),
        "variants": variants,
    }


def _coverage_dict(tile_coverage: dict[str, TileCoverage]) -> dict:
    return {
        city: {
            "n_tiles_expected": cov.n_tiles_expected,
            "n_tiles_read": cov.n_tiles_read,
            "n_tiles_skipped": cov.n_tiles_skipped,
            "n_bref_excluded": cov.n_bref_excluded,
        }
        for city, cov in sorted(tile_coverage.items())
    }


def _print_summary(
    variants: dict[str, dict], tile_coverage: dict[str, TileCoverage], report_path: Path
) -> None:
    print("=" * 78)
    print("Coarseness-localization diagnostic (F5): recalibrated verdict per variant")
    print("=" * 78)
    header = f"  {'variant':<7} {'metric':<18} {'n_pairs':>7} {'n_sig_eff':>9} {'median_ks':>10}"
    print(header)
    for variant in VARIANTS:
        v = variants[variant]
        for metric in _METRICS:
            cell = v["per_metric"][metric]
            mks = "null" if cell["median_ks"] is None else f"{cell['median_ks']:.4f}"
            print(
                f"  {variant:<7} {metric:<18} {cell['n_pairs']:>7} "
                f"{cell['n_significant_effect']:>9} {mks:>10}"
            )
        print(
            f"  {variant:<7} verdict={v['verdict']} raw_bh={v['n_significant_raw_bh']} "
            f"effect={v['n_significant_effect']} "
            f"n_features_by_city={v['n_features_by_city']}"
        )
    for city, cov in sorted(tile_coverage.items()):
        print(
            f"  tiles {city:<18}: expected={cov.n_tiles_expected} "
            f"read={cov.n_tiles_read} skipped={cov.n_tiles_skipped} "
            f"bref_excluded={cov.n_bref_excluded}"
        )
    print(f"  report: {report_path}")
    print("  the variant that kills the most discrimination signal localizes the layer")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Task-23 coarseness-localization diagnostic (F5; spec §4.2)"
    )
    parser.add_argument("--release", default=_DEFAULT_RELEASE)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    # δ=0.15 (PI-call #1, spec §4.3): identical to the gate-(i) runner — the
    # diagnostic recomputes the RECALIBRATED verdict under each variant.
    parser.add_argument("--effect-size-floor", type=float, default=0.15)
    parser.add_argument("--report-out", default=_DEFAULT_REPORT_OUT)
    args = parser.parse_args(argv)

    logger.info("collecting tile records for cities=%s release=%s", args.cities, args.release)
    records, tile_coverage = collect_tile_records(args.release, args.cities)
    logger.info("collected %d tile records", len(records))

    variants = diagnose(
        records,
        min_n=args.min_n,
        alpha=args.alpha,
        effect_size_floor=args.effect_size_floor,
    )

    report = {
        "methodology": _methodology(
            release=args.release,
            cities=args.cities,
            min_n=args.min_n,
            alpha=args.alpha,
            effect_size_floor=args.effect_size_floor,
        ),
        "variants": variants,
        "tile_coverage": _coverage_dict(tile_coverage),
    }

    report_path = (
        (_REPO / args.report_out)
        if not Path(args.report_out).is_absolute()
        else Path(args.report_out)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonicalize_yaml(report), encoding="utf-8")

    _print_summary(variants, tile_coverage, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
