"""Compute alpha (tail-cell rejection) drop report at the chosen budget.

Per Halt 4 reviewer pre-load (2026-05-28): alpha should not silently truncate.
Emit count of cells dropped at the chosen elbow, their per-type feature
composition, and the % of total per-type observations the drop set captures.
This makes the β-upgrade decision data-driven once sub-E lands and stage-4
is measured.

Per-cell length uses the CHUNKED, encoder-faithful Case-A token cost from
`cfm.data.sub_f.token_cost.feature_token_cost` — the same budget-accounting
twin (pinned against the encoder) used by the chunked-budget audit. This
replaces the previous UNCHUNKED `3 + n_anchor + 2*(v-1)` formula, which
under-counted long road segments (it ignored the §3.5 32m chunking) and
mis-modelled Multi* features (it concatenated parts rather than splitting them
per-feature as `encode_cell` does). Grading the drop set against the chunked
budget with an unchunked length would compare apples to oranges and produce
wrong drop counts; see the Halt 4 revisit (2026-05-29).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.token_cost import feature_token_cost

ROOT = Path(__file__).resolve().parents[2]

# Spec §7.2 stage-4 estimate (sub-E absent); matches audit_chunked_budget.py.
STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL = 0.7


def compute_alpha_drop_report(
    sub_c_region_dir: Path,
    budget_raw: int,
    budget_padded: int,
) -> dict:
    """Compute the alpha (tail-cell rejection) drop report and return it.

    Pure computation: reads the sub-C region's per-tile features.parquet, grades
    each cell against the chunked encoder-faithful Case-A token cost, and returns
    the report dict (drop counts, per-type composition, length-tail stats). Does
    NOT write any file or print — see ``main`` / ``run_alpha_drop_report`` for
    the I/O wrappers. Extracted as a thin importable entrypoint so the Task 11
    pipeline orchestrator can emit the warning-band composition into the region
    run report without shelling out (close-checklist obligation).
    """
    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    lock = primitives.get("lock_metadata", primitives).get("approved_lock_values", primitives)
    anchor_scheme = lock.get("anchor_scheme", "hierarchical")
    n_anchor = 2 if anchor_scheme == "flat" else 4

    # Per-cell: (tile_i, tile_j, cell_i, cell_j) -> {"length": int, "by_type": {fc: count}}
    per_cell: dict[tuple[int, int, int, int], dict] = defaultdict(
        lambda: {"length": 0, "by_type": defaultdict(int)}
    )

    tile_features = sorted(sub_c_region_dir.glob("tile=*/features.parquet"))
    print(f"[alpha drop report] {len(tile_features)} tiles", flush=True)
    tile_keys: set[tuple[int, int]] = set()
    total_by_type: dict[int, int] = defaultdict(int)

    for path in tile_features:
        parts = path.parent.name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        tile_keys.add((tile_i, tile_j))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            geom = wkb_loads(r["geometry"])
            fc = int(r["feature_class"])
            cell_key = (tile_i, tile_j, int(r["cell_i"]), int(r["cell_j"]))
            per_cell[cell_key]["length"] += feature_token_cost(geom, n_anchor)
            per_cell[cell_key]["by_type"][fc] += 1
            total_by_type[fc] += 1

    # Add empty cells per spec §7.8.
    for ti, tj in tile_keys:
        for ci in range(8):
            for cj in range(8):
                _ = per_cell[(ti, tj, ci, cj)]

    # Stage-4 formula adder: 0.7 tokens per non-empty cell (sub-E absent).
    for _key, data in per_cell.items():
        if data["by_type"]:
            data["length"] += STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL

    # Apply alpha at budget_raw (cells whose length exceeds raw quantile are dropped).
    n_cells_total = len(per_cell)
    dropped_cells = [(key, data) for key, data in per_cell.items() if data["length"] > budget_raw]
    retained_cells = [(key, data) for key, data in per_cell.items() if data["length"] <= budget_raw]

    # Per-type drop composition.
    dropped_by_type: dict[int, int] = defaultdict(int)
    for _key, data in dropped_cells:
        for fc, count in data["by_type"].items():
            dropped_by_type[fc] += count

    # Dropped-cell length stats (head of the tail).
    dropped_lengths = sorted([d["length"] for _k, d in dropped_cells], reverse=True)

    output = {
        "_status": "PROPOSED - companion to sequence_length_analysis.yaml",
        "anchor_scheme_used": anchor_scheme,
        "n_anchor": n_anchor,
        "per_feature_cost_provenance": (
            "chunked_encoder_faithful_case_a "
            "(cfm.data.sub_f.token_cost.feature_token_cost; pinned against encoder; "
            "Multi* split per-part per encode_cell)"
        ),
        "stage_4_provenance": "formula_derived_per_spec_7_2_no_sub_e_cache",
        "budget_raw": budget_raw,
        "budget_padded": budget_padded,
        "n_cells_total": n_cells_total,
        "n_cells_dropped": len(dropped_cells),
        "n_cells_retained": len(retained_cells),
        "drop_fraction_pct": float(len(dropped_cells) / n_cells_total * 100.0)
        if n_cells_total
        else 0.0,
        "drop_set_by_type": {
            int(fc): {
                "n_features_dropped": int(dropped_by_type.get(fc, 0)),
                "n_features_total": int(total_by_type[fc]),
                "fraction_of_type_dropped_pct": (
                    float(dropped_by_type.get(fc, 0) / total_by_type[fc] * 100.0)
                    if total_by_type[fc]
                    else 0.0
                ),
            }
            for fc in sorted(total_by_type)
        },
        "dropped_cell_length_head_top10": [int(x) for x in dropped_lengths[:10]],
        "dropped_cell_length_min": int(dropped_lengths[-1]) if dropped_lengths else 0,
        "dropped_cell_length_max": int(dropped_lengths[0]) if dropped_lengths else 0,
        "dropped_cell_length_median": int(
            sorted([d["length"] for _k, d in dropped_cells])[len(dropped_cells) // 2]
        )
        if dropped_cells
        else 0,
    }

    return output


def run_alpha_drop_report(
    sub_c_region_dir: Path,
    budget_raw: int,
    budget_padded: int,
    label: str = "alpha_drop_at_chosen_elbow",
) -> dict:
    """Compute the report, write it to reports/, log a one-line summary, return it.

    Thin importable entrypoint (also called by ``main``). The Task 11 pipeline
    orchestrator calls this after a successful region derive to emit the
    warning-band composition; it returns the report dict so the caller can log
    the warning-band count + per-type composition into the region run report.
    """
    output = compute_alpha_drop_report(sub_c_region_dir, budget_raw, budget_padded)
    out = ROOT / "reports" / f"sub_f_task_3c_{label}.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[alpha drop report] wrote {out}")
    print(
        f"[alpha drop report] dropped {output['n_cells_dropped']}/"
        f"{output['n_cells_total']} cells "
        f"({output['drop_fraction_pct']:.3f}%); "
        f"per-type drop: "
        + ", ".join(
            f"fc={fc}: {v['n_features_dropped']}/{v['n_features_total']} "
            f"({v['fraction_of_type_dropped_pct']:.3f}%)"
            for fc, v in output["drop_set_by_type"].items()
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument(
        "--budget-raw",
        type=int,
        required=True,
        help="Raw budget at chosen quantile (e.g. 5899 for chunked P99.9).",
    )
    parser.add_argument(
        "--budget-padded",
        type=int,
        required=True,
        help="Padded budget at chosen quantile (e.g. 6016 for chunked P99.9 with 128 padding).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="alpha_drop_at_chosen_elbow",
        help="Label for output file naming.",
    )
    args = parser.parse_args()
    run_alpha_drop_report(
        sub_c_region_dir=args.sub_c_region_dir,
        budget_raw=args.budget_raw,
        budget_padded=args.budget_padded,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
