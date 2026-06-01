"""Sub-G per-region validator orchestration (spec Decisions 5+6).

Runs the three seam checks over every tile, ACCUMULATES all diagnostics (no
halt-on-first; spec Decision 6), aggregates seam-3 accuracy across the region,
then `finalize` groups by signature, writes `quarantine_report.yaml` +
`_PHASE1_ACCURACY_BASELINE.yaml` EVERY run, applies the sanity floor, and writes
`_PHASE1_VALIDATED` iff the quarantine is empty AND the sanity floor holds.

The `<100`-tile scope-stop is enforced PRE-CHAIN (T11 Step-0b), NOT here — a
region reaching the validator has already cleared the pre-chain gate.

Decomposition: `validate_tile` is pure (in-memory dicts) and unit-tested;
`validate_region` is the thin parquet-loading loop, exercised by the Singapore
integration run (T11).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import read_macro_core_parquet
from cfm.data.sub_g.diagnostics import Diagnostic, group_by_signature, render_quarantine_report
from cfm.data.sub_g.readers import (
    read_sub_c_features_by_cell,
    read_sub_e_contract_rows,
    read_sub_f_cells,
)
from cfm.data.sub_g.seam_contract_tokens import (
    build_cell_contracts,
    check_cell_bijection,
    parse_actual_brefs_per_cell,
    predict_expected_brefs_per_cell,
)
from cfm.data.sub_g.seam_decodability import check_decodability
from cfm.data.sub_g.seam_macro_geometry import check_density, check_road_skeleton
from cfm.data.sub_g.versions import (
    VALIDATOR_VERSION,
    _percentile,
    render_accuracy_baseline,
    render_validated_marker,
)

_log = logging.getLogger(__name__)

# Ratified sanity-floor cliffs (sub-G design §3c; reviewer-set 2026-05-31).
_SANITY_POS_P999_M = 50.0
_SANITY_ANGLE_P95_DEG = 20.0


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    n_quarantine_groups: int
    sanity_floor_violated: bool
    n_diagnostics: int
    marker_written: bool


def _macro_targets(
    macro_rows: list,
) -> tuple[dict[tuple[int, int], int | None], dict[tuple[int, int, int], int | None]]:
    """Split macro_core rows into per-cell density + per-internal-edge skeleton.

    zoning is intentionally NOT extracted (SI-3 deferred; see seam_macro_geometry).
    """
    density_by_cell: dict[tuple[int, int], int | None] = {}
    skeleton_by_edge: dict[tuple[int, int, int], int | None] = {}
    for r in macro_rows:
        if r.slot_kind == SlotKind.CELL and r.cell_i is not None:
            density_by_cell[(int(r.cell_i), int(r.cell_j))] = r.cell_density_bucket
        elif r.slot_kind == SlotKind.INTERNAL_EDGE and r.lower_cell_i is not None:
            skeleton_by_edge[(int(r.lower_cell_i), int(r.lower_cell_j), int(r.axis))] = (
                r.road_skeleton_class
            )
    return density_by_cell, skeleton_by_edge


def validate_tile(
    tile_id: str,
    features_by_cell: dict[tuple[int, int], list[dict]],
    area_by_cell: dict[tuple[int, int], float],
    crossings: list[dict],
    density_by_cell: dict[tuple[int, int], int | None],
    skeleton_by_edge: dict[tuple[int, int, int], int | None],
    cell_contracts: dict[tuple[int, int], dict[str, str]],
    tokens_by_cell: dict[tuple[int, int], list[int]],
) -> tuple[list[Diagnostic], list[dict], int]:
    """Run all three seams on one tile's in-memory artifacts. Pure (no I/O).

    Third return is ``n_bref_collapse``: decoded blocks excluded from the
    OGC-validity gate as the v1-by-design outbound-bref placeholder collapse
    (sub-G T11 H3); reported in the accuracy baseline, not gated.
    """
    diags: list[Diagnostic] = []
    errors: list[dict] = []
    n_bref_collapse = 0

    # Seam 1 (macro <-> geometry).
    diags += check_density(tile_id, features_by_cell, area_by_cell, density_by_cell)
    features_flat = [f for cell_feats in features_by_cell.values() for f in cell_feats]
    diags += check_road_skeleton(tile_id, features_flat, crossings, skeleton_by_edge)

    # Seams 2 + 3, driven by sub-F's cells (authoritative 64 per tile).
    for cell, tokens in tokens_by_cell.items():
        cell_features = features_by_cell.get(cell, [])
        expected = predict_expected_brefs_per_cell(cell_features, cell_contracts.get(cell, {}))
        actual = parse_actual_brefs_per_cell(tokens)
        diags += check_cell_bijection(tile_id, cell, expected, actual)

        d, e, nbc = check_decodability(tile_id, cell, tokens, cell_features)
        diags += d
        errors += e
        n_bref_collapse += nbc
    return diags, errors, n_bref_collapse


def finalize(
    region: str,
    release: str,
    all_diags: list[Diagnostic],
    all_errors: list[dict],
    output_dir: Path,
    volatile: dict[str, str],
    bref_collapse_count: int = 0,
) -> ValidationResult:
    """Group + write reports (every run) + apply sanity floor + gate.

    The sanity floor gates on the CORE accuracy distribution (excludes the
    v1-unencoded outbound bref vertex by construction identity; reviewer 2026-06-01)
    so it means "broken encode/decode," not the designed crossing-road info-loss.
    The FULL distribution is still reported (render_accuracy_baseline) so the bref
    residual stays visible. Angle is defined only on count-matched features.
    """
    pos_core = [e["position_core_m"] for e in all_errors]
    pos_full = [e["position_full_m"] for e in all_errors]
    ang_core = [e["angle_core_deg"] for e in all_errors if e.get("angle_core_deg") is not None]
    pos_core_p999 = _percentile(pos_core, 99.9)
    ang_core_p95 = _percentile(ang_core, 95.0)
    sanity_violated = pos_core_p999 > _SANITY_POS_P999_M or ang_core_p95 > _SANITY_ANGLE_P95_DEG
    structural_breaches = sum(
        1 for d in all_diags if d.invariant_name == "decoded_vertex_within_cell_bound"
    )

    diags = list(all_diags)
    if sanity_violated:
        reasons = []
        if pos_core_p999 > _SANITY_POS_P999_M:
            reasons.append(f"position_core p99.9 {pos_core_p999:.1f}m > {_SANITY_POS_P999_M}m")
        if ang_core_p95 > _SANITY_ANGLE_P95_DEG:
            reasons.append(f"angle_core p95 {ang_core_p95:.1f}deg > {_SANITY_ANGLE_P95_DEG}deg")
        diags.append(
            Diagnostic(
                tile_id=f"region={region}",
                invariant_name="accuracy_sanity_floor",
                artifact_left="seam3 core accuracy (region p-values)",
                observed_left={
                    "position_core_p99_9": pos_core_p999,
                    "angle_core_p95": ang_core_p95,
                },
                artifact_right="sanity floor",
                observed_right={"position": _SANITY_POS_P999_M, "angle": _SANITY_ANGLE_P95_DEG},
                expected_relationship="region core accuracy within sanity cliffs",
                spec_clause_citation="sub-G design §3c sanity floor",
                signature="; ".join(reasons),
            )
        )

    groups = group_by_signature(diags)
    quarantine_yaml = render_quarantine_report(groups, region, release, VALIDATOR_VERSION)
    baseline_yaml = render_accuracy_baseline(
        pos_core, pos_full, ang_core, region, release, structural_breaches, bref_collapse_count
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quarantine_report.yaml").write_text(quarantine_yaml, encoding="utf-8")
    (output_dir / "_PHASE1_ACCURACY_BASELINE.yaml").write_text(baseline_yaml, encoding="utf-8")

    passed = len(groups) == 0
    if passed:
        content_digest = hashlib.sha256(
            (quarantine_yaml + baseline_yaml).encode("utf-8")
        ).hexdigest()
        marker_yaml = render_validated_marker(region, release, content_digest, volatile)
        (output_dir / "_PHASE1_VALIDATED").write_text(marker_yaml, encoding="utf-8")

    return ValidationResult(
        passed=passed,
        n_quarantine_groups=len(groups),
        sanity_floor_violated=sanity_violated,
        n_diagnostics=len(diags),
        marker_written=passed,
    )


def _read_cell_areas(sub_c_cells_parquet: Path) -> dict[tuple[int, int], float]:
    tbl = pq.ParquetFile(sub_c_cells_parquet).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return {
        (int(cols["cell_i"][i]), int(cols["cell_j"][i])): float(
            cols["cell_area_admin_clipped_m2"][i]
        )
        for i in range(tbl.num_rows)
    }


def _read_crossings(sub_c_crossings_parquet: Path) -> list[dict]:
    tbl = pq.ParquetFile(sub_c_crossings_parquet).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return [
        {
            "source_feature_id": cols["source_feature_id"][i],
            "lower_cell_i": int(cols["lower_cell_i"][i]),
            "lower_cell_j": int(cols["lower_cell_j"][i]),
            "axis": int(cols["axis"][i]),
        }
        for i in range(tbl.num_rows)
    ]


def validate_region(
    sub_c_region_dir: Path,
    sub_d_region_dir: Path,
    sub_e_region_dir: Path,
    sub_f_region_dir: Path,
    region: str,
    release: str,
    output_dir: Path,
    volatile: dict[str, str],
) -> ValidationResult:
    """Thin parquet-loading loop over sub-F's tiles (exercised by T11).

    The tile set is sub-F's tiles (= sub-D's, per the sub-C/sub-D gap disposition).
    """
    all_diags: list[Diagnostic] = []
    all_errors: list[dict] = []
    all_bref_collapse = 0
    tile_dirs = sorted(sub_f_region_dir.glob("tile=*"))
    _log.info("sub-G validating %d tiles in %s", len(tile_dirs), sub_f_region_dir)

    for sub_f_tile in tile_dirs:
        tile = sub_f_tile.name  # "tile=EPSG3414_iX_jY"
        features_by_cell = read_sub_c_features_by_cell(sub_c_region_dir / tile / "features.parquet")
        area_by_cell = _read_cell_areas(sub_c_region_dir / tile / "cells.parquet")
        crossings = _read_crossings(sub_c_region_dir / tile / "crossings.parquet")
        macro_rows = read_macro_core_parquet(sub_d_region_dir / tile / "macro_core.parquet")
        density_by_cell, skeleton_by_edge = _macro_targets(macro_rows)
        contract_rows = read_sub_e_contract_rows(
            sub_e_region_dir / tile / "boundary_contract.parquet"
        )
        cell_contracts = build_cell_contracts(contract_rows)
        tokens_by_cell = read_sub_f_cells(sub_f_tile / "cells.parquet")

        d, e, nbc = validate_tile(
            tile,
            features_by_cell,
            area_by_cell,
            crossings,
            density_by_cell,
            skeleton_by_edge,
            cell_contracts,
            tokens_by_cell,
        )
        all_diags += d
        all_errors += e
        all_bref_collapse += nbc

    return finalize(region, release, all_diags, all_errors, output_dir, volatile, all_bref_collapse)
