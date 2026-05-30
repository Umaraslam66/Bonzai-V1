"""Verify the locked chunked P99.9 budget survives encoder-faithful costing.

The chunked-budget audit (``scripts/sub_f/audit_chunked_budget.py``, commit
2562c89) costed multi-part geometries by CONCATENATING their part coords into a
single coord list, which introduces a phantom inter-part segment and a single
``<feature>..<feature_end>`` wrapper. The real encoder (``encode_cell``) SPLITS
Multi* per-part: one wrapper per part, no phantom segment. The two agree exactly
for every single-part geometry (LineString / Polygon / Point); they can only
differ on cells containing Multi* features.

This script recomputes the per-cell length two ways — the audit's CONCAT method
and the encoder-faithful SPLIT method (via ``token_cost.feature_token_cost``,
pinned against the encoder) — over the cached Singapore sub-C features, using
the same stage-4 adder, empty-cell handling, and quantile method as the audit.
It prints P99.9 raw/padded for both and the Multi* prevalence, so the Halt 4
re-lock target (5,899 raw / 6,016 padded) can be confirmed against the corrected
emitter-faithful accounting before it is locked (verify-before-lock).

Read-only: writes nothing. Run:

    uv run python scripts/sub_f/verify_chunked_budget_p999.py \
        --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
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

STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL = 0.7  # spec §7.2, sub-E absent

# Mirror the audit's CONCAT costing (scripts/sub_f/audit_chunked_budget.py).
from cfm.data.sub_f.token_cost import chunked_per_feature_tokens, feature_token_cost  # noqa: E402


def _concat_coords(geom) -> list[tuple[float, float]]:
    """Audit method: flatten all part coords into one list (phantom segments)."""
    gt = geom.geom_type
    if gt == "LineString":
        return list(geom.coords)
    if gt == "Polygon":
        return list(geom.exterior.coords)
    if gt == "Point":
        return [(geom.x, geom.y)]
    if gt == "MultiLineString":
        out: list[tuple[float, float]] = []
        for part in geom.geoms:
            out.extend(list(part.coords))
        return out
    if gt == "MultiPolygon":
        out = []
        for part in geom.geoms:
            out.extend(list(part.exterior.coords))
        return out
    if gt == "MultiPoint":
        return [(p.x, p.y) for p in geom.geoms]
    return []


def _quantile(values: list[float], q_pct: float) -> float:
    """Match audit_chunked_budget.py:quantile exactly."""
    if q_pct == 100:
        return float(max(values))
    idx = int(q_pct * 100) - 1
    return float(quantiles(values, n=10000)[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    lock = primitives.get("lock_metadata", primitives).get("approved_lock_values", primitives)
    anchor_scheme = lock.get("anchor_scheme", "hierarchical")
    n_anchor = 2 if anchor_scheme == "flat" else 4

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    if not tile_features:
        print(f"no tiles under {args.sub_c_region_dir}", file=sys.stderr)
        return 1
    print(f"[verify] {len(tile_features)} tiles, n_anchor={n_anchor}", flush=True)

    per_cell_concat: dict[tuple, float] = defaultdict(float)
    per_cell_split: dict[tuple, float] = defaultdict(float)
    nonempty: set[tuple] = set()
    tile_keys: set[tuple[int, int]] = set()
    multi_counts: dict[str, int] = defaultdict(int)
    n_features = 0
    max_abs_feature_delta = 0
    n_features_with_delta = 0

    for path in tile_features:
        parts = path.parent.name.replace("tile=", "").split("_")
        ti = int(parts[1].lstrip("i"))
        tj = int(parts[2].lstrip("j"))
        tile_keys.add((ti, tj))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            geom = wkb_loads(r["geometry"])
            n_features += 1
            gt = geom.geom_type
            if gt.startswith("Multi"):
                multi_counts[gt] += 1
            concat = chunked_per_feature_tokens(_concat_coords(geom), n_anchor)
            split = feature_token_cost(geom, n_anchor)
            if concat != split:
                n_features_with_delta += 1
                max_abs_feature_delta = max(max_abs_feature_delta, abs(split - concat))
            key = (ti, tj, int(r["cell_i"]), int(r["cell_j"]))
            per_cell_concat[key] += concat
            per_cell_split[key] += split
            nonempty.add(key)

    for ti, tj in tile_keys:
        for ci in range(8):
            for cj in range(8):
                _ = per_cell_concat[(ti, tj, ci, cj)]
                _ = per_cell_split[(ti, tj, ci, cj)]
    for key in nonempty:
        per_cell_concat[key] += STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL
        per_cell_split[key] += STAGE_4_FORMULA_TOKENS_PER_NONEMPTY_CELL

    concat_lengths = sorted(per_cell_concat.values())
    split_lengths = sorted(per_cell_split.values())

    def pad(x: float) -> int:
        return ((int(x) + 127) // 128) * 128

    concat_p999 = int(_quantile(concat_lengths, 99.9))
    split_p999 = int(_quantile(split_lengths, 99.9))

    print(f"[verify] features={n_features} cells={len(concat_lengths)}")
    print(f"[verify] Multi* features: {dict(multi_counts)}")
    print(
        f"[verify] features where split != concat: {n_features_with_delta} "
        f"(max abs per-feature token delta = {max_abs_feature_delta})"
    )
    print(f"[verify] CONCAT (audit method)  P99.9 raw={concat_p999} padded={pad(concat_p999)}")
    print(f"[verify] SPLIT  (encoder-faithful) P99.9 raw={split_p999} padded={pad(split_p999)}")
    print("[verify] locked re-lock target     P99.9 raw=5899 padded=6016")
    for q in (99.0, 99.5, 99.9, 99.99, 100.0):
        c = int(_quantile(concat_lengths, q))
        s = int(_quantile(split_lengths, q))
        print(f"  q={q:<6} concat raw={c} padded={pad(c)} | split raw={s} padded={pad(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
