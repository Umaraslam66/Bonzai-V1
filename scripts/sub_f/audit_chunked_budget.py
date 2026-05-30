"""Audit BP3 budget against the chunked encoder formula.

Halt 4 BP3 budget (P99.9 = 5,888 padded) was measured using the
non-chunked Case A formula `tokens = 5 + 2V` per feature. The encoder
chunking fix (commit 86f0c99) makes per-feature token count depend on
actual segment lengths, not just vertex count:

  tokens_per_feature = 3 + N_anchor + 2 * sum_segments(ceil(distance_i / 32))

This script re-aggregates per-cell budgets against cached Singapore
sub-C features using the CHUNKED formula and compares to the locked
P99.9 = 5,888 padded / 5,792 raw budget.

Stage-4 cross-cell overhead retains the spec §7.2 formula (0.7
tokens/non-empty cell) since sub-E parquet is absent locally; this is
consistent with the existing budget surface's stage-4 provenance.

Outputs `reports/sub_f_chunked_budget_audit.yaml` with:
- new budget surface (P99, P99.5, P99.9, P99.99, P100)
- delta vs locked P99.9 = 5,792 raw / 5,888 padded
- per-type chunked-vs-unchunked token-count comparison at v_mean
- recommendation: Halt 4 holds vs revisit
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]

# Encoder constants (mirrors src/cfm/data/sub_f/encoder.py).
CHUNK_THRESHOLD_M = 32.0  # 64 quanta x 0.5m, per spec section 3.5
QUANTUM_M = 0.5

# Locked Halt 4 budget values (configs/sub_f/sequence_length_analysis.yaml).
LOCKED_BUDGET_RAW = 5792
LOCKED_BUDGET_PADDED = 5888
LOCKED_PADDING_SLACK = LOCKED_BUDGET_PADDED - LOCKED_BUDGET_RAW  # 96 tokens

# Spec section 7.2 stage-4 estimate (sub-E absent).
STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL = 0.7


def _chunked_segment_pairs(distance_m: float) -> int:
    """Number of (dir, mag) pairs the encoder emits for a segment of length
    distance_m, per spec section 3.5 chunking rule. Matches
    `_direction_magnitude_pair` in encoder.py.
    """
    if distance_m <= 0:
        return 1  # zero-length floor; matches encoder's max(1, ...) semantic
    total_q = max(1, round(distance_m / QUANTUM_M))
    return math.ceil(total_q / 64)


def _per_feature_chunked_tokens(coords: list[tuple[float, float]], n_anchor: int) -> int:
    """Case A token count using actual segment chunking.

    Token shape: <feature> <semantic_tag> anchor (V-1 segments, chunked) <feature_end>
    = 1 + 1 + N_anchor + 2 * sum(ceil(seg_distance / 32m)) + 1
    = 3 + N_anchor + 2 * sum_chunked_pairs

    For V < 2 (Point or degenerate): emit just structural + anchor = 3 + N_anchor.
    """
    n_vertices = len(coords)
    if n_vertices < 2:
        return 3 + n_anchor
    total_pairs = 0
    for i in range(1, n_vertices):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        dist = math.hypot(x2 - x1, y2 - y1)
        total_pairs += _chunked_segment_pairs(dist)
    return 3 + n_anchor + 2 * total_pairs


def _per_feature_unchunked_tokens(n_vertices: int, n_anchor: int) -> int:
    """Original non-chunked formula: 3 + N_anchor + 2*(V-1). Used for delta."""
    if n_vertices < 2:
        return 3 + n_anchor
    return 3 + n_anchor + 2 * (n_vertices - 1)


def _vertex_count_and_coords(geom):
    """Extract vertex count + coord list for token counting."""
    gt = geom.geom_type
    if gt == "LineString":
        return len(geom.coords), list(geom.coords)
    if gt == "Polygon":
        return len(geom.exterior.coords), list(geom.exterior.coords)
    if gt == "Point":
        return 1, [(geom.x, geom.y)]
    if gt == "MultiLineString":
        # encode_cell splits multi-parts into separate features; sum chunks per part.
        total_v = 0
        total_coords: list[tuple[float, float]] = []
        for part in geom.geoms:
            total_v += len(part.coords)
            total_coords.extend(list(part.coords))
        return total_v, total_coords
    if gt == "MultiPolygon":
        total_v = 0
        total_coords: list[tuple[float, float]] = []
        for part in geom.geoms:
            total_v += len(part.exterior.coords)
            total_coords.extend(list(part.exterior.coords))
        return total_v, total_coords
    if gt == "MultiPoint":
        return sum(1 for _ in geom.geoms), [(p.x, p.y) for p in geom.geoms]
    return 0, []


def main() -> int:
    primitives_path = ROOT / "configs" / "sub_f" / "encoding_primitives.yaml"
    primitives = yaml.safe_load(primitives_path.read_text(encoding="utf-8"))
    lock = primitives.get("lock_metadata", primitives).get("approved_lock_values", primitives)
    anchor_scheme = lock.get("anchor_scheme", "hierarchical")
    n_anchor = 2 if anchor_scheme == "flat" else 4

    sub_c_region = ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
    if not sub_c_region.exists():
        print(f"sub-C Singapore cache absent at {sub_c_region}", file=sys.stderr)
        return 1

    tile_features = sorted(sub_c_region.glob("tile=*/features.parquet"))
    print(f"[chunked budget audit] {len(tile_features)} tiles", flush=True)

    # Per-cell aggregation: chunked tokens and unchunked tokens (for delta), per-type counts.
    per_cell_chunked: dict[tuple[int, int, int, int], int] = defaultdict(int)
    per_cell_unchunked: dict[tuple[int, int, int, int], int] = defaultdict(int)
    per_cell_nonempty: set[tuple[int, int, int, int]] = set()
    per_type_chunked_sum: dict[int, int] = defaultdict(int)
    per_type_unchunked_sum: dict[int, int] = defaultdict(int)
    per_type_n_obs: dict[int, int] = defaultdict(int)
    tile_keys: set[tuple[int, int]] = set()

    for path in tile_features:
        parts = path.parent.name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        tile_keys.add((tile_i, tile_j))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            geom = wkb_loads(r["geometry"])
            n_v, coords = _vertex_count_and_coords(geom)
            chunked = _per_feature_chunked_tokens(coords, n_anchor)
            unchunked = _per_feature_unchunked_tokens(n_v, n_anchor)
            cell_key = (tile_i, tile_j, int(r["cell_i"]), int(r["cell_j"]))
            per_cell_chunked[cell_key] += chunked
            per_cell_unchunked[cell_key] += unchunked
            per_cell_nonempty.add(cell_key)
            fc = int(r["feature_class"])
            per_type_chunked_sum[fc] += chunked
            per_type_unchunked_sum[fc] += unchunked
            per_type_n_obs[fc] += 1

    # Add empty cells per spec section 7.8.
    for ti, tj in tile_keys:
        for ci in range(8):
            for cj in range(8):
                _ = per_cell_chunked[(ti, tj, ci, cj)]
                _ = per_cell_unchunked[(ti, tj, ci, cj)]

    # Stage-4 formula adder: 0.7 tokens/non-empty cell.
    for key in per_cell_nonempty:
        per_cell_chunked[key] += STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL
        per_cell_unchunked[key] += STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL

    chunked_lengths = sorted(per_cell_chunked.values())
    unchunked_lengths = sorted(per_cell_unchunked.values())
    n_cells = len(chunked_lengths)
    n_empty = sum(1 for v in chunked_lengths if v < 1.0)

    def quantile(values: list[float], q_pct: float) -> float:
        if q_pct == 100:
            return float(max(values))
        idx = int(q_pct * 100) - 1
        return float(quantiles(values, n=10000)[idx])

    quantile_targets = [99.0, 99.5, 99.9, 99.99, 100.0]
    surface = []
    for q in quantile_targets:
        chunked_q = quantile(chunked_lengths, q)
        unchunked_q = quantile(unchunked_lengths, q)
        chunked_padded = ((int(chunked_q) + 127) // 128) * 128
        surface.append(
            {
                "quantile": q,
                "chunked_sequence_length_tokens": int(chunked_q),
                "chunked_padded_length_tokens": int(chunked_padded),
                "unchunked_sequence_length_tokens": int(unchunked_q),
                "delta_chunked_minus_unchunked": int(chunked_q - unchunked_q),
            }
        )

    # Per-type delta at the budget-relevant scale.
    per_type_delta_pct = {}
    for fc in sorted(per_type_n_obs):
        if per_type_unchunked_sum[fc] == 0:
            continue
        delta_pct = (
            (per_type_chunked_sum[fc] - per_type_unchunked_sum[fc])
            / per_type_unchunked_sum[fc]
            * 100.0
        )
        per_type_delta_pct[fc] = round(delta_pct, 2)

    # P99.9 delta vs locked Halt 4 budget.
    p999_chunked_raw = int(quantile(chunked_lengths, 99.9))
    p999_chunked_padded = ((p999_chunked_raw + 127) // 128) * 128
    locked_p999_raw = LOCKED_BUDGET_RAW  # 5792
    locked_p999_padded = LOCKED_BUDGET_PADDED  # 5888
    p999_delta_raw = p999_chunked_raw - locked_p999_raw
    p999_delta_padded = p999_chunked_padded - locked_p999_padded
    within_slack = p999_delta_raw <= LOCKED_PADDING_SLACK
    same_padded_block = p999_chunked_padded == locked_p999_padded

    if same_padded_block:
        recommendation = (
            "Halt 4 HOLDS. Chunked P99.9 lands in the same padding block "
            f"({locked_p999_padded}) as the locked budget. Padding slack "
            f"({LOCKED_PADDING_SLACK} tokens) absorbs the chunking delta "
            f"({p999_delta_raw:+d} tokens raw)."
        )
    elif within_slack:
        recommendation = (
            "Halt 4 HOLDS marginally. Chunked P99.9 raw delta "
            f"({p999_delta_raw:+d}) within padding slack "
            f"({LOCKED_PADDING_SLACK}), but the padded budget shifts "
            f"({locked_p999_padded} -> {p999_chunked_padded}). Reviewer "
            "may choose to re-lock at the new padded value."
        )
    else:
        recommendation = (
            "Halt 4 REVISIT required. Chunked P99.9 raw delta "
            f"({p999_delta_raw:+d}) exceeds padding slack "
            f"({LOCKED_PADDING_SLACK}). New padded budget "
            f"{p999_chunked_padded} vs locked {locked_p999_padded} "
            f"(+{p999_delta_padded}). Elbow choice or per-type floors "
            "may need re-calibration; alpha drop report would also shift."
        )

    output = {
        "anchor_scheme_used": anchor_scheme,
        "n_anchor": n_anchor,
        "n_cells_analyzed": n_cells,
        "n_empty_cells": n_empty,
        "stage_4_provenance": "formula_derived_per_spec_7_2_no_sub_e_cache",
        "chunked_vs_unchunked_budget_surface": surface,
        "per_type_chunked_token_inflation_pct": {
            str(fc): pct for fc, pct in per_type_delta_pct.items()
        },
        "per_type_n_observations": {str(fc): per_type_n_obs[fc] for fc in sorted(per_type_n_obs)},
        "halt_4_p999_lock": {
            "raw_tokens": locked_p999_raw,
            "padded_tokens": locked_p999_padded,
            "padding_slack_tokens": LOCKED_PADDING_SLACK,
        },
        "chunked_p999_audit": {
            "raw_tokens": p999_chunked_raw,
            "padded_tokens": p999_chunked_padded,
            "delta_raw_vs_locked": p999_delta_raw,
            "delta_padded_vs_locked": p999_delta_padded,
            "within_padding_slack": within_slack,
            "same_padded_block": same_padded_block,
        },
        "recommendation": recommendation,
    }
    out = ROOT / "reports" / "sub_f_chunked_budget_audit.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[chunked budget audit] wrote {out}")
    print(f"  Chunked P99.9 raw={p999_chunked_raw} padded={p999_chunked_padded}")
    print(f"  Locked  P99.9 raw={locked_p999_raw} padded={locked_p999_padded}")
    print(f"  Delta raw={p999_delta_raw:+d} padded={p999_delta_padded:+d}")
    print(f"  Per-type chunked inflation (%): {dict(per_type_delta_pct)}")
    print(f"  Recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
