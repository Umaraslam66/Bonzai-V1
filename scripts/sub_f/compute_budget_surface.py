"""Compute BP3 budget surface for Halt 4.

Joint 4D distribution per feature type -> surface over (quantile x
data_loss_per_type x sequence_length x padding_overhead). Per spec §7.4:
NO autonomous P100 default; reviewer picks elbow.

Stage 4 (cross-cell overhead) per spec §7.2 BP7 -> BP3 correction:
  outbound bref: net 0 tokens (replaces tail direction+magnitude)
  inbound bref: +1 token per crossing
  approx 0.7 tokens/cell on Singapore rough estimate

Sub-E-absent adaptation (2026-05-28; see
`reports/2026-05-23-phase-1-sub-F-close-checklist.md` line 12 +
`project_sub_e_cache_absent_t3c_code_inferred` memory):

  - `--sub-e-region-dir` is OPTIONAL.
  - When sub-E parquet cache is missing or not provided, apply the §7.2
    formula uniformly: every non-empty cell gets +0.7 tokens stage-4
    overhead; empty cells get 0.
  - Output YAML carries `stage_4_provenance` so the Halt 4 reviewer can
    see the source of the stage-4 figure prominently.
  - Per user directive: formula-derived stages cut CONSERVATIVE
    (lower confidence, cheap-to-keep applies).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]

# Per-type retention defaults per spec §7.5.
DEFAULT_RETENTION = {
    0: 0.999,  # roads — AV use case
    1: 0.99,  # buildings
    2: 0.99,  # POIs
    3: 0.99,  # base / landuse-class
}

# Stage-4 formula estimate per spec §7.2 (Singapore rough estimate).
STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL = 0.7


def case_a_tokens(v: int, n_anchor: int) -> int:
    """Stage-3 Case A: 3 + N_anchor + 2*(V-1), V >= 1.

    V = 0 guard returns 2 (matches T3b).
    """
    return 3 + n_anchor + 2 * (v - 1) if v >= 1 else 2


def _vertex_count(geom) -> int:
    """Match T3a vertex-count semantics, including Multi* geometries."""
    gt = geom.geom_type
    if gt == "LineString":
        return len(geom.coords)
    if gt == "Polygon":
        return len(geom.exterior.coords)
    if gt == "Point":
        return 1
    if gt == "MultiPoint":
        return sum(1 for _ in geom.geoms)
    if gt == "MultiLineString":
        return sum(len(part.coords) for part in geom.geoms)
    if gt == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0


def _quantile(values: list[int | float], q: float) -> float:
    """Quantile of values at percentile q in [0, 100]. q=100 returns max."""
    if not values:
        return 0.0
    if q >= 100:
        return float(max(values))
    # statistics.quantiles n=10000 yields 9999 cut points spaced 0.01% apart.
    qs = quantiles(values, n=10000)
    idx = round(q * 100) - 1
    idx = max(0, min(idx, len(qs) - 1))
    return float(qs[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument(
        "--sub-e-region-dir",
        required=False,
        default=None,
        type=Path,
        help=(
            "Optional. If absent or directory does not exist, stage-4 overhead "
            "is estimated from spec §7.2 formula (0.7 tokens/non-empty cell)."
        ),
    )
    args = parser.parse_args()

    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    lock = (
        primitives.get("lock_metadata", {}).get("approved_lock_values")
        or primitives.get("proposed_lock")
        or primitives
    )
    anchor_scheme = lock["anchor_scheme"]
    n_anchor = 2 if anchor_scheme == "flat" else 4

    # --- Aggregate per-cell stage-3 length and per-type feature counts -------
    # Cell key: (tile_i, tile_j, cell_i, cell_j).
    per_cell_stage3: dict[tuple[int, int, int, int], int] = defaultdict(int)
    per_cell_features_by_type: dict[tuple[int, int, int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    tile_keys: set[tuple[int, int]] = set()

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    print(f"[budget surface] reading {len(tile_features)} sub-C tile features", flush=True)
    for path in tile_features:
        tile_name = path.parent.name
        parts = tile_name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        tile_keys.add((tile_i, tile_j))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            geom = wkb_loads(r["geometry"])
            v = _vertex_count(geom)
            key = (tile_i, tile_j, int(r["cell_i"]), int(r["cell_j"]))
            per_cell_stage3[key] += case_a_tokens(v, n_anchor)
            per_cell_features_by_type[key][int(r["feature_class"])] += 1

    # Enumerate all 8x8 grid cells per observed tile (empty cells included).
    for ti, tj in tile_keys:
        for ci in range(8):
            for cj in range(8):
                _ = per_cell_stage3[(ti, tj, ci, cj)]  # defaultdict creates 0 if missing

    # --- Stage-4 overhead --------------------------------------------------
    sub_e_present = (
        args.sub_e_region_dir is not None
        and args.sub_e_region_dir.exists()
        and any(args.sub_e_region_dir.glob("tile=*/boundary_contract.parquet"))
    )

    stage_4_per_cell: dict[tuple[int, int, int, int], float] = defaultdict(float)

    if sub_e_present:
        # Measured path — uniform attribution per BP7 §7.2 estimate (kept rough
        # because the plan's earlier rotation-aware version is sub-E-API
        # specific; a full Task 7-style emit would belong in T3c-v2).
        # Each MAJOR/MINOR boundary edge contributes ~0.5 tokens to cells facing it.
        tile_contracts = sorted(args.sub_e_region_dir.glob("tile=*/boundary_contract.parquet"))
        for path in tile_contracts:
            tile_name = path.parent.name
            parts = tile_name.replace("tile=", "").split("_")
            tile_i = int(parts[1].lstrip("i"))
            tile_j = int(parts[2].lstrip("j"))
            table = pq.ParquetFile(path).read()
            for r in table.to_pylist():
                bclass = r["boundary_class_enum"]
                if bclass in (2, 3):  # MAJOR / MINOR per sub-E derivation.py:19-23
                    # Uniform distribute over 64 cells (rough; reviewer sees provenance).
                    for cell_idx in range(64):
                        ci, cj = divmod(cell_idx, 8)
                        stage_4_per_cell[(tile_i, tile_j, ci, cj)] += 0.5 / 8.0
        stage_4_provenance = "measured_from_sub_e"
    else:
        # Formula fallback: 0.7 tokens per non-empty cell.
        for key, base_len in per_cell_stage3.items():
            if base_len > 0:
                stage_4_per_cell[key] = STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL
        stage_4_provenance = "formula_derived_per_spec_7_2_no_sub_e_cache"

    # --- Compose per-cell total ---------------------------------------------
    total_lengths: list[int] = []
    per_cell_total_for_key: dict[tuple[int, int, int, int], int] = {}
    for key, base_len in per_cell_stage3.items():
        total = round(base_len + stage_4_per_cell.get(key, 0.0))
        total_lengths.append(total)
        per_cell_total_for_key[key] = total

    n_cells = len(total_lengths)
    n_empty = sum(1 for v in total_lengths if v == 0)
    if not total_lengths:
        print("[budget surface] no per-cell lengths computed", file=sys.stderr)
        return 1

    # --- Budget surface (5 quantiles) ---------------------------------------
    quantile_targets = [99.0, 99.5, 99.9, 99.99, 100.0]
    surface = []
    for q in quantile_targets:
        l_value = _quantile(total_lengths, q)
        padded = ((int(l_value) + 127) // 128) * 128
        surface.append(
            {
                "quantile": float(q),
                "sequence_length_tokens": int(l_value),
                "padded_length_tokens": int(padded),
                "padding_overhead_pct": (
                    float((padded - int(l_value)) / int(l_value) * 100.0) if l_value else 0.0
                ),
            }
        )

    # --- Per-type retention at each quantile --------------------------------
    # Retention: feature is retained iff its cell's total length <= budget.
    # (Cells whose total exceeds budget lose ALL features in cell per (alpha)
    # truncation strategy default; if (beta) is chosen later, retention table
    # shifts within-cell rather than per-cell.)
    retention_by_quantile: dict[float, dict[int, float]] = {}
    for q in quantile_targets:
        budget = _quantile(total_lengths, q)
        per_type_retained: dict[int, int] = defaultdict(int)
        per_type_total: dict[int, int] = defaultdict(int)
        for key, fc_counts in per_cell_features_by_type.items():
            cell_len = per_cell_total_for_key.get(key, 0)
            for fc, count in fc_counts.items():
                per_type_total[fc] += count
                if cell_len <= budget:
                    per_type_retained[fc] += count
        retention_by_quantile[q] = {
            int(fc): (per_type_retained[fc] / per_type_total[fc]) if per_type_total[fc] else 1.0
            for fc in sorted(per_type_total)
        }

    # --- Stage breakdown (measured vs formula) ------------------------------
    stage_breakdown = {
        "stage_1": {
            "what": "Per-cell feature counts by type",
            "provenance": "MEASURED from sub-C Singapore parquet",
            "source": str(args.sub_c_region_dir),
        },
        "stage_2": {
            "what": "Per-feature vertex counts by type",
            "provenance": "MEASURED from sub-C Singapore parquet (geometry WKB vertex count)",
            "source": str(args.sub_c_region_dir),
        },
        "stage_3": {
            "what": "Tokens per geometry element",
            "provenance": (
                "DERIVED from spec §7.2 encoder formula "
                f"(Case A; n_anchor={n_anchor}, anchor={anchor_scheme}, "
                "per BP2 Halt 2 lock)"
            ),
            "source": "configs/sub_f/encoding_primitives.yaml",
        },
        "stage_4": {
            "what": "Cross-cell coordination overhead per cell",
            "provenance": stage_4_provenance,
            "source": (
                str(args.sub_e_region_dir)
                if sub_e_present
                else "spec §7.2 formula (sub-E parquet cache absent)"
            ),
            "formula_tokens_per_nonempty_cell": (
                STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL if not sub_e_present else None
            ),
        },
    }

    # --- Scaling projection at 1% and 5% Singapore-share -------------------
    # PRD does NOT contain an explicit "10K global sequence-length budget"
    # cite. The only "10,000" reference at PRD line 61 is the BP1 frequency
    # floor (categories with fewer than 10,000 instances bucket up). The
    # 10K-global-token-budget assumption is reviewer-pre-loaded per the
    # implementer prompt; surface this gap explicitly so the reviewer can
    # confirm or correct.
    PRD_GLOBAL_BUDGET_TOKENS = 10000  # ASSUMED per implementer-prompt directive
    SINGAPORE_CELL_COUNT = n_cells

    scaling_projection: dict = {
        "prd_global_budget_tokens_assumed": PRD_GLOBAL_BUDGET_TOKENS,
        "prd_cite_status": (
            "needs_reviewer_cite_confirmation — PRD.md grep for '10K' / '10,000' / "
            "'sequence length' did not return a global token-budget cite; the "
            "10,000 at PRD line 61 is the BP1 frequency floor (categories with "
            "fewer than 10,000 global instances bucket up), not a sequence-length "
            "ceiling. The 10K projection here uses the implementer-prompt "
            "assumption pending reviewer correction."
        ),
        "singapore_cell_count": SINGAPORE_CELL_COUNT,
        "by_share": {},
    }
    for share in (0.01, 0.05):
        share_rows = []
        for q in quantile_targets:
            per_cell_tokens = _quantile(total_lengths, q)
            projected = share * SINGAPORE_CELL_COUNT * per_cell_tokens
            share_rows.append(
                {
                    "quantile": q,
                    "per_cell_tokens": float(per_cell_tokens),
                    "projected_total_tokens": float(projected),
                    "fraction_of_10K_budget": float(projected / PRD_GLOBAL_BUDGET_TOKENS),
                }
            )
        scaling_projection["by_share"][f"singapore_share_{int(share * 100)}pct"] = share_rows

    # --- Compose output -----------------------------------------------------
    output = {
        "_status": "PROPOSED — pending Halt 4 reviewer approval per spec §10.3.",
        "stage_4_provenance": stage_4_provenance,
        "n_cells_analyzed": n_cells,
        "n_empty_cells": n_empty,
        "anchor_scheme_used": anchor_scheme,
        "n_anchor": n_anchor,
        "budget_surface": surface,
        "retention_by_quantile_by_type": {
            float(q): {str(fc): float(rate) for fc, rate in row.items()}
            for q, row in retention_by_quantile.items()
        },
        "retention_defaults_per_spec_7_5": {
            str(fc): rate for fc, rate in DEFAULT_RETENTION.items()
        },
        "stage_breakdown": stage_breakdown,
        "scaling_projection": scaling_projection,
        "proposed_truncation_strategy": "alpha",  # spec §7.9 default alpha tail-cell rejection
        "proposed_truncation_strategy_note": (
            "alpha (tail-cell rejection) recommended. Bias conservative because "
            "stage-4 is formula-derived (sub-E cache absent). When sub-E is "
            "regenerated, re-run T3c against measured stage-4 and revisit the "
            "elbow — see close-checklist line 12."
            if not sub_e_present
            else "alpha (tail-cell rejection) recommended per §7.9 default."
        ),
        "proposed_long_cell_diagnostic_pp": 0.5,
        "proposed_long_cell_diagnostic_note": (
            "Long-cell diagnostic fires at the per-cell-token threshold corresponding "
            "to (chosen_quantile - 0.5pp) per spec §7.7. Reviewer chooses the quantile; "
            "the threshold value computed once the quantile is selected."
        ),
    }

    out = ROOT / "configs" / "sub_f" / "sequence_length_analysis.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[budget surface] wrote {out}")
    print(f"[budget surface] stage_4_provenance: {stage_4_provenance}")
    print(f"[budget surface] n_cells={n_cells} n_empty={n_empty}")
    for row in surface:
        print(
            f"  P{row['quantile']:>5}: seq={row['sequence_length_tokens']:>5} "
            f"padded={row['padded_length_tokens']:>5} "
            f"pad_overhead={row['padding_overhead_pct']:.1f}%"
        )
    print("[budget surface] per-type retention (quantile -> {type: rate}):")
    for q, row in retention_by_quantile.items():
        print(f"  P{q}: " + " ".join(f"fc{fc}={rate * 100:.3f}%" for fc, rate in row.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
