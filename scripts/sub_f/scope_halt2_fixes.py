"""Compare candidate fixes for the Halt 2 round-trip gap (NO lock changes).

Measures honest per-source-vertex round-trip L_inf on real Singapore sub-C data
under three encoder variants, to make the re-lock proposal concrete:

  baseline_48   : current encoder (open-loop: every chunk of a segment uses the
                  segment's single quantized direction). Reproduces the gap.
  feedback_48   : SAME 48 directions, SAME chunk count (so token count and the
                  BP3 budget are UNCHANGED), but each chunk re-aims its direction
                  from the running DECODED position toward the segment's true
                  endpoint (error-feedback / Bresenham-style). Decoder is
                  unchanged. Tests whether drift can be bounded at zero lock cost.
  openloop_144  : open-loop with 144 directions (option A representative; the
                  single-segment curve said 144 holds 4.8m at the P99.9 segment).

For each variant: L_inf distribution (p50/p95/p99/p99.9/max) by type + overall,
and a token-count parity check (feedback must equal baseline pair count, else it
would move the just-re-locked BP3 budget).

Run:
    uv run python scripts/sub_f/scope_halt2_fixes.py \
        --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/ [--stride N]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]

QUANTUM_M = 0.5
MAX_MAG_Q = 64  # 32m per chunk


def _dir_bin(angle_deg: float, n: int) -> int:
    bw = 360.0 / n
    return int((angle_deg % 360.0) // bw) % n  # tie-to-lower, matches encoder.direction_bin


def _qcoord(v: float) -> int:
    return round(v / QUANTUM_M)


def _seg_chunks(total_q: int) -> list[int]:
    """Chunk a quantized distance into <=64q pieces (same count as encoder)."""
    total_q = max(1, total_q)
    out = []
    rem = total_q
    while rem > 0:
        c = min(MAX_MAG_Q, rem)
        out.append(c)
        rem -= c
    return out


def _roundtrip_linf(coords, n_dir: int, feedback: bool) -> tuple[float, int]:
    """Return (per-source-vertex max L_inf, n_pairs) for one coord list."""
    if len(coords) < 2:
        return 0.0, 0
    # decoded anchor = hierarchical-quantized first vertex
    dx0, dy0 = _qcoord(coords[0][0]) * QUANTUM_M, _qcoord(coords[0][1]) * QUANTUM_M
    cur = (dx0, dy0)
    max_linf = max(abs(coords[0][0] - dx0), abs(coords[0][1] - dy0))
    n_pairs = 0
    for k in range(1, len(coords)):
        a = coords[k - 1]
        b = coords[k]
        total_q = max(1, _qcoord(math.hypot(b[0] - a[0], b[1] - a[1])))
        chunks = _seg_chunks(total_q)
        if not feedback:
            # open-loop: one quantized direction for the whole segment
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            dbin = _dir_bin(ang, n_dir)
            arad = math.radians(dbin * (360.0 / n_dir))
            for cq in chunks:
                dist = cq * QUANTUM_M
                cur = (cur[0] + dist * math.cos(arad), cur[1] + dist * math.sin(arad))
                n_pairs += 1
        else:
            # feedback: re-aim each chunk from running decoded pos toward true b
            for cq in chunks:
                ang = math.degrees(math.atan2(b[1] - cur[1], b[0] - cur[0]))
                dbin = _dir_bin(ang, n_dir)
                arad = math.radians(dbin * (360.0 / n_dir))
                dist = cq * QUANTUM_M
                cur = (cur[0] + dist * math.cos(arad), cur[1] + dist * math.sin(arad))
                n_pairs += 1
        max_linf = max(max_linf, abs(b[0] - cur[0]), abs(b[1] - cur[1]))
    return max_linf, n_pairs


def _parts(geom):
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


def _summ(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return {"n": 0}

    def pc(q):
        return round(s[min(n - 1, int(q / 100 * n))], 3)

    return {
        "n": n,
        "p50": pc(50),
        "p95": pc(95),
        "p99": pc(99),
        "p99_9": pc(99.9),
        "max": round(s[-1], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-c-region-dir", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=1, help="sample 1-in-stride features for speed")
    args = ap.parse_args()

    role = {0: "roads", 1: "buildings", 2: "pois", 3: "base"}
    variants = [("baseline_48", 48, False), ("feedback_48", 48, True), ("openloop_144", 144, False)]
    linf = {v[0]: defaultdict(list) for v in variants}
    linf_all = {v[0]: [] for v in variants}
    pair_mismatch = 0  # feedback vs baseline pair-count parity
    i = 0

    for path in sorted(args.sub_c_region_dir.glob("tile=*/features.parquet")):
        for r in pq.ParquetFile(path).read().to_pylist():
            i += 1
            if i % args.stride:
                continue
            fc = int(r["feature_class"])
            geom = wkb_loads(r["geometry"])
            per_variant = {v[0]: 0.0 for v in variants}
            base_pairs = fb_pairs = 0
            for coords in _parts(geom):
                for name, ndir, fb in variants:
                    li, np_ = _roundtrip_linf(coords, ndir, fb)
                    per_variant[name] = max(per_variant[name], li)
                    if name == "baseline_48":
                        base_pairs += np_
                    if name == "feedback_48":
                        fb_pairs += np_
            if base_pairs != fb_pairs:
                pair_mismatch += 1
            for name in per_variant:
                linf[name][fc].append(per_variant[name])
                linf_all[name].append(per_variant[name])

    out = {
        "_status": "SCOPING - Halt 2 fix comparison; NO lock changes.",
        "stride": args.stride,
        "n_features_measured": len(linf_all["baseline_48"]),
        "feedback_token_count_parity": (
            "IDENTICAL to baseline (budget unaffected)"
            if pair_mismatch == 0
            else f"MISMATCH on {pair_mismatch} features (budget WOULD shift)"
        ),
        "roundtrip_l_inf_by_variant": {
            name: {
                "ALL": _summ(linf_all[name]),
                **{role[fc]: _summ(linf[name][fc]) for fc in sorted(linf[name])},
            }
            for name, _n, _f in variants
        },
    }
    out_path = ROOT / "reports" / "sub_f_halt2_fix_comparison.yaml"
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"[fixes] wrote {out_path}")
    print(f"[fixes] features measured: {out['n_features_measured']} (stride {args.stride})")
    print(f"[fixes] feedback token parity: {out['feedback_token_count_parity']}")
    for name, _n, _f in variants:
        a = out["roundtrip_l_inf_by_variant"][name]["ALL"]
        rd = out["roundtrip_l_inf_by_variant"][name]["roads"]
        print(
            f"  {name:<14} ALL p95={a['p95']} p99={a['p99']} p99.9={a['p99_9']} max={a['max']} "
            f"| roads p95={rd['p95']} p99={rd['p99']} max={rd['max']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
