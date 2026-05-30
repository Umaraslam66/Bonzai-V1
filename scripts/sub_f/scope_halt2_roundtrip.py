"""Scoping audit for the Halt 2 round-trip-threshold revisit (NO lock changes).

T8.7 surfaced that the BP2 Halt 2 lock (round-trip L_inf <= 4.8m) does not hold
for long straight segments: 48-direction quantization (7.5 deg bins) drifts an
endpoint by ~L*sin(3.75deg), linear in segment length, so any Case-A segment
longer than ~73m reconstructs > 4.8m off. The Halt 2 sample never exercised this
(recorded `chunking_is_no_op_on_test_sample`); the chunking fix (86f0c99) made
long segments faithfully representable and unmasked the gap.

This script produces the NUMBERS for a re-lock proposal. It changes no locks.

It reports, anchored to REAL Singapore sub-C data (not synthetic happy-path):

1. HONEST round-trip L_inf distribution per feature type at the CURRENT 48
   directions, using a vertex-count-aware metric (each source vertex mapped to
   its decoded counterpart via cumulative chunked_segment_pairs). This is the
   fix to the plan's buggy 1:1-zip metric, re-measured.
2. Real segment-length distribution per type (P50/P95/P99/P99.9/max) — the
   design anchor for the direction-count curve.
3. Direction-count curve: for a range of N, the max segment length supportable
   at 4.8m (= 4.8 / sin(180/N deg)) and the reconstruction error at the real
   P99.9 segment length; the min N that holds 4.8m at P99.9.
4. The synthetic aggregate-test L_inf (deterministic seed 20260529) under the
   honest metric, to confirm/replace the subagent's 21.57m.

Geometry-only: uses the encoder's actual quantization helpers
(_hierarchical_anchor_tokens, _vertex_pairs_dir_mag) so the round-trip is
faithful to what ships, without the semantic/vocab overhead.

Run:
    uv run python scripts/sub_f/scope_halt2_roundtrip.py \
        --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.encoder import (
    _DIRECTION_BASE,
    _HIERARCHICAL_ANCHOR_BASE,
    _MAGNITUDE_BASE,
    _hierarchical_anchor_tokens,
    _vertex_pairs_dir_mag,
)
from cfm.data.sub_f.encoder import (
    _HIERARCHICAL_ANCHOR_BLOCK as BLK,
)
from cfm.data.sub_f.token_cost import chunked_segment_pairs

ROOT = Path(__file__).resolve().parents[2]

QUANTUM_M = 0.5
DIR_COUNT = 48


def _decode_geom_tokens(
    anchor_tokens: list[int], pair_tokens: list[int]
) -> list[tuple[float, float]]:
    """Inverse of anchor + (dir,mag) pairs -> coord list. Mirrors decoder.py."""
    xh, xl, yh, yl = anchor_tokens
    xq = (xh - _HIERARCHICAL_ANCHOR_BASE) * BLK + (xl - (_HIERARCHICAL_ANCHOR_BASE + BLK))
    yq = (yh - (_HIERARCHICAL_ANCHOR_BASE + 2 * BLK)) * BLK + (
        yl - (_HIERARCHICAL_ANCHOR_BASE + 3 * BLK)
    )
    pts = [(xq * QUANTUM_M, yq * QUANTUM_M)]
    for i in range(0, len(pair_tokens), 2):
        d = pair_tokens[i] - _DIRECTION_BASE
        m = (pair_tokens[i + 1] - _MAGNITUDE_BASE) + 1
        ang = math.radians(d * (360.0 / DIR_COUNT))
        dist = m * QUANTUM_M
        pts.append((pts[-1][0] + dist * math.cos(ang), pts[-1][1] + dist * math.sin(ang)))
    return pts


def _honest_part_l_inf(coords: list[tuple[float, float]]) -> float:
    """Vertex-count-aware round-trip L_inf for one coord list (Case A).

    Encodes anchor + chunked (dir,mag) pairs, decodes, then maps each SOURCE
    vertex k to its decoded counterpart at cumulative chunk-pair index, and
    returns max per-source-vertex L_inf. Chunking inserts collinear decoded
    vertices between source vertices; this metric ignores those (they are
    admitted per spec §3.8) and compares only the source vertices.
    """
    if len(coords) < 2:
        return 0.0
    anchor = _hierarchical_anchor_tokens(coords[0][0], coords[0][1])
    pairs = _vertex_pairs_dir_mag(coords)
    decoded = _decode_geom_tokens(anchor, pairs)

    # Cumulative decoded index for each source vertex.
    cum = 0
    max_linf = 0.0
    # source vertex 0 -> decoded[0]
    sx, sy = coords[0]
    dx, dy = decoded[0]
    max_linf = max(max_linf, abs(sx - dx), abs(sy - dy))
    for k in range(1, len(coords)):
        seg = math.hypot(coords[k][0] - coords[k - 1][0], coords[k][1] - coords[k - 1][1])
        cum += chunked_segment_pairs(seg)
        sx, sy = coords[k]
        dxk, dyk = decoded[cum]
        max_linf = max(max_linf, abs(sx - dxk), abs(sy - dyk))
    return max_linf


def _parts(geom) -> list[list[tuple[float, float]]]:
    gt = geom.geom_type
    if gt == "LineString":
        return [list(geom.coords)]
    if gt == "Polygon":
        return [list(geom.exterior.coords)]
    if gt == "Point":
        return [[(geom.x, geom.y)]]
    if gt == "MultiLineString":
        return [list(p.coords) for p in geom.geoms]
    if gt == "MultiPolygon":
        return [list(p.exterior.coords) for p in geom.geoms]
    if gt == "MultiPoint":
        return [[(p.x, p.y)] for p in geom.geoms]
    return []


def _segment_lengths(coords: list[tuple[float, float]]) -> list[float]:
    return [
        math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
        for i in range(1, len(coords))
    ]


def _pctile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q >= 100:
        return sorted_vals[-1]
    idx = min(len(sorted_vals) - 1, int(q / 100 * len(sorted_vals)))
    return sorted_vals[idx]


def _dist_summary(vals: list[float]) -> dict:
    s = sorted(vals)
    return {
        "n": len(s),
        "p50": round(_pctile(s, 50), 3),
        "p95": round(_pctile(s, 95), 3),
        "p99": round(_pctile(s, 99), 3),
        "p99_9": round(_pctile(s, 99.9), 3),
        "max": round(s[-1], 3) if s else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    role = {0: "roads", 1: "buildings", 2: "pois", 3: "base"}
    seg_lengths_by_type: dict[int, list[float]] = defaultdict(list)
    all_seg_lengths: list[float] = []
    linf_by_type: dict[int, list[float]] = defaultdict(list)
    all_linf: list[float] = []

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    if not tile_features:
        print(f"no tiles under {args.sub_c_region_dir}", file=sys.stderr)
        return 1
    print(f"[scope] {len(tile_features)} tiles", flush=True)

    for path in tile_features:
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            fc = int(r["feature_class"])
            geom = wkb_loads(r["geometry"])
            feat_linf = 0.0
            for coords in _parts(geom):
                segs = _segment_lengths(coords)
                seg_lengths_by_type[fc].extend(segs)
                all_seg_lengths.extend(segs)
                feat_linf = max(feat_linf, _honest_part_l_inf(coords))
            linf_by_type[fc].append(feat_linf)
            all_linf.append(feat_linf)

    # Direction-count curve, anchored to real P99.9 segment length.
    all_seg_sorted = sorted(all_seg_lengths)
    seg_p999 = _pctile(all_seg_sorted, 99.9)
    seg_max = all_seg_sorted[-1] if all_seg_sorted else 0.0
    n_curve = []
    for N in (48, 64, 72, 96, 128, 144, 192, 256, 360):
        half_bin_rad = math.radians(180.0 / N)
        max_len_at_4_8 = 4.8 / math.sin(half_bin_rad) if half_bin_rad > 0 else float("inf")
        err_at_p999 = seg_p999 * math.sin(half_bin_rad)
        err_at_max = seg_max * math.sin(half_bin_rad)
        n_curve.append(
            {
                "N_directions": N,
                "bin_width_deg": round(360.0 / N, 4),
                "max_segment_len_within_4_8m": round(max_len_at_4_8, 1),
                "recon_err_at_real_p99_9_segment_m": round(err_at_p999, 3),
                "recon_err_at_real_max_segment_m": round(err_at_max, 3),
                "holds_4_8m_at_p99_9": err_at_p999 <= 4.8,
            }
        )
    min_N_for_p999 = next(
        (row["N_directions"] for row in n_curve if row["holds_4_8m_at_p99_9"]), None
    )

    # Synthetic aggregate fixtures (plan's deterministic seed) under honest metric.
    rng = random.Random(20260529)
    cell_extent = 250.0
    synth_max = 0.0
    for _ in range(30):
        n_v = rng.randint(2, 6)
        coords = [
            (rng.uniform(10, cell_extent - 10), rng.uniform(10, cell_extent - 10))
            for _ in range(n_v)
        ]
        # canonicalize is identity for open LineString direction; use coords as-is.
        synth_max = max(synth_max, _honest_part_l_inf(coords))

    out = {
        "_status": "SCOPING - Halt 2 round-trip revisit; NO lock changes. Re-lock proposal input.",
        "current_direction_count": DIR_COUNT,
        "current_l_inf_threshold_m": 4.8,
        "honest_metric": (
            "per-source-vertex L_inf via cumulative chunked_segment_pairs mapping "
            "(admits collinear chunk-inserted vertices per spec 3.8)"
        ),
        "real_roundtrip_l_inf_at_48_dir": {
            "ALL": _dist_summary(all_linf),
            **{role[fc]: _dist_summary(linf_by_type[fc]) for fc in sorted(linf_by_type)},
        },
        "real_segment_length_m": {
            "ALL": _dist_summary(all_seg_lengths),
            **{
                role[fc]: _dist_summary(seg_lengths_by_type[fc])
                for fc in sorted(seg_lengths_by_type)
            },
        },
        "direction_count_curve": n_curve,
        "min_N_holding_4_8m_at_real_p99_9_segment": min_N_for_p999,
        "real_p99_9_segment_len_m": round(seg_p999, 3),
        "real_max_segment_len_m": round(seg_max, 3),
        "synthetic_aggregate_honest_l_inf_max_m": round(synth_max, 3),
    }
    out_path = ROOT / "reports" / "sub_f_halt2_roundtrip_scoping.yaml"
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"[scope] wrote {out_path}")
    print(
        f"[scope] real round-trip L_inf @48dir ALL: {out['real_roundtrip_l_inf_at_48_dir']['ALL']}"
    )
    for fc in sorted(linf_by_type):
        print(f"  {role[fc]:<10} L_inf: {_dist_summary(linf_by_type[fc])}")
    print(f"[scope] real segment len ALL: {out['real_segment_length_m']['ALL']}")
    print(f"[scope] real P99.9 segment = {seg_p999:.1f}m, max = {seg_max:.1f}m")
    print(f"[scope] min N holding 4.8m at P99.9 segment = {min_N_for_p999}")
    print(f"[scope] synthetic aggregate honest L_inf max = {synth_max:.2f}m (was 21.57m est)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
