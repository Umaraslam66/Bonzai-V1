"""Scope re-anchoring as the Halt 2 fix (option 3) — NO lock changes.

Re-anchor = emit a fresh ABSOLUTE anchor (4 tokens, reusing the existing anchor
sub-block 300..395 — no new sentinel) whenever cumulative path since the last
anchor exceeds threshold T; the decoder snaps to it, resetting accumulated drift.
Re-anchoring a long (multi-chunk) segment REPLACES its chunked dir/mag pairs
(4 tokens vs 2*ceil(L/32)) -> token savings on the long-segment-heavy cells that
drive the budget tail; re-anchoring within wiggly short-segment runs costs +2 per
trigger. Net budget effect is uncertain -> measured here.

Deliverables (reviewer-specified):
1. BP3 token cost: re-anchor trigger count + net token delta at T in {73,60,40,30};
   per-cell sequence length recomputed -> P99.9 and MAX vs the locked 6016 padded.
2. Angle preservation: right-angle-corner post-deviation under re-anchor vs baseline
   (re-anchor uses open-loop directions + exact re-anchored vertices -> expected
   same-or-better).
3. Determinism: trigger is `cum_path > T` (a float compare); count segments whose
   cumulative path lands within float-noise of T (trigger-flip fragility).
4. Position bound: real-data position L_inf (p95/p99.9/MAX) at each T.
5. Grammar/sentinel cascade: re-anchor needs a §3.2 grammar change (mid-feature
   anchor) but NO new vocab/sentinel (reuses anchor block); self-delimiting holds
   (disjoint ID ranges). Reported as a constant, not measured.

Run:
    uv run python scripts/sub_f/scope_halt2_reanchor.py \
        --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.encoder import direction_bin, quantize_coord_m

ROOT = Path(__file__).resolve().parents[2]
QUANTUM_M = 0.5
MAX_MAG_Q = 64
N_DIR = 48
BIN_DEG = 360.0 / N_DIR
STAGE_4 = 0.7
PADDED_BUDGET = 6016
THRESHOLDS = (73.0, 60.0, 40.0, 30.0)
TIE_EPS_M = 1e-6  # trigger-flip fragility window around T
N_ANCHOR = 4
STRUCT = 3  # <feature> + semantic + <feature_end>


def _chunks(total_q: int) -> int:
    return max(1, math.ceil(max(1, total_q) / MAX_MAG_Q))


def _reanchor_part(coords, T: float):
    """Encode one part with path-threshold re-anchoring.

    Returns (n_pair_dirmag_tokens, n_reanchors, position_l_inf, n_trigger_ties).
    Token count for the part = STRUCT + N_ANCHOR(initial) + 2*pairs + N_ANCHOR*reanchors.
    """
    if len(coords) < 2:
        return STRUCT + N_ANCHOR, 0, 0.0, 0
    q = lambda v: quantize_coord_m(v) * QUANTUM_M  # noqa: E731
    cur = (q(coords[0][0]), q(coords[0][1]))
    cum = 0.0
    pairs = 0
    reanchors = 0
    ties = 0
    max_linf = max(abs(coords[0][0] - cur[0]), abs(coords[0][1] - cur[1]))
    for k in range(1, len(coords)):
        a, b = coords[k - 1], coords[k]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if abs((cum + seg) - T) < TIE_EPS_M:
            ties += 1
        if cum + seg > T:
            # re-anchor at b: absolute, decoder snaps to quantized b
            cur = (q(b[0]), q(b[1]))
            reanchors += 1
            cum = 0.0
        else:
            total_q = max(1, quantize_coord_m(seg))
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            arad = math.radians(direction_bin(ang) * BIN_DEG)
            # reconstruct cumulative (chunks share one open-loop direction)
            dist = total_q * QUANTUM_M
            cur = (cur[0] + dist * math.cos(arad), cur[1] + dist * math.sin(arad))
            pairs += _chunks(total_q)
            cum += seg
        max_linf = max(max_linf, abs(b[0] - cur[0]), abs(b[1] - cur[1]))
    n_tokens = STRUCT + N_ANCHOR + 2 * pairs + N_ANCHOR * reanchors
    return n_tokens, reanchors, max_linf, ties


def _corner_angle_deg(prev, cur, nxt):
    ax, ay = prev[0] - cur[0], prev[1] - cur[1]
    bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))))


def _reanchor_decode_ring(coords, T: float):
    """Decode a closed ring under re-anchoring; return (decoded, original->decoded map).

    T = inf reproduces the open-loop baseline (no re-anchor fires).
    """
    uniq = coords[:-1] if (len(coords) > 1 and coords[0] == coords[-1]) else coords
    if len(uniq) < 2:
        return None
    q = lambda v: quantize_coord_m(v) * QUANTUM_M  # noqa: E731
    cur = (q(uniq[0][0]), q(uniq[0][1]))
    decoded = [cur]
    mapping = [0]
    cum = 0.0
    for k in range(1, len(uniq) + 1):
        a, b = uniq[k - 1], uniq[k % len(uniq)]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if cum + seg > T:
            cur = (q(b[0]), q(b[1]))
            cum = 0.0
        else:
            total_q = max(1, quantize_coord_m(seg))
            arad = math.radians(
                direction_bin(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) * BIN_DEG
            )
            cur = (
                cur[0] + total_q * QUANTUM_M * math.cos(arad),
                cur[1] + total_q * QUANTUM_M * math.sin(arad),
            )
            cum += seg
        decoded.append(cur)
        if k < len(uniq):
            mapping.append(len(decoded) - 1)
    return decoded, mapping, uniq


def _baseline_part_tokens(coords) -> int:
    if len(coords) < 2:
        return STRUCT + N_ANCHOR
    pairs = sum(
        _chunks(
            max(
                1,
                quantize_coord_m(
                    math.hypot(coords[k][0] - coords[k - 1][0], coords[k][1] - coords[k - 1][1])
                ),
            )
        )
        for k in range(1, len(coords))
    )
    return STRUCT + N_ANCHOR + 2 * pairs


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


def _q(values, qp):
    if qp >= 100:
        return float(max(values))
    return float(quantiles(values, n=10000)[int(qp * 100) - 1])


def _summ(vals):
    s = sorted(vals)
    n = len(s)
    pc = lambda qp: round(s[min(n - 1, int(qp / 100 * n))], 3)  # noqa: E731
    return {"p95": pc(95), "p99": pc(99), "p99_9": pc(99.9), "max": round(s[-1], 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = ap.parse_args()

    pos = {T: [] for T in THRESHOLDS}
    reanchor_total = {T: 0 for T in THRESHOLDS}
    token_delta_total = {T: 0 for T in THRESHOLDS}
    ties_total = {T: 0 for T in THRESHOLDS}
    per_cell = {T: defaultdict(float) for T in THRESHOLDS}
    nonempty = set()
    tile_keys = set()
    baseline_total = 0
    n_features = 0
    # Angle (deliverable 2): right-angle building-corner post-deviation under
    # baseline (open-loop, T=inf) vs re-anchor at two thresholds.
    ANGLE_T = {"baseline_openloop": math.inf, "reanchor_73m": 73.0, "reanchor_40m": 40.0}
    angle_dev = {k: [] for k in ANGLE_T}
    angle_cat = {k: 0 for k in ANGLE_T}
    angle_corners = 0

    for path in sorted(args.sub_c_region_dir.glob("tile=*/features.parquet")):
        name = path.parent.name.replace("tile=", "").split("_")
        ti, tj = int(name[1].lstrip("i")), int(name[2].lstrip("j"))
        tile_keys.add((ti, tj))
        for r in pq.ParquetFile(path).read().to_pylist():
            n_features += 1
            geom = wkb_loads(r["geometry"])
            cell = (ti, tj, int(r["cell_i"]), int(r["cell_j"]))
            nonempty.add(cell)
            parts = _parts(geom)
            base_tok = sum(_baseline_part_tokens(c) for c in parts)
            baseline_total += base_tok
            feat_pos = {T: 0.0 for T in THRESHOLDS}
            for T in THRESHOLDS:
                tok = 0
                for c in parts:
                    nt, nr, li, ti_ = _reanchor_part(c, T)
                    tok += nt
                    reanchor_total[T] += nr
                    ties_total[T] += ti_
                    feat_pos[T] = max(feat_pos[T], li)
                per_cell[T][cell] += tok
                token_delta_total[T] += tok - base_tok
                pos[T].append(feat_pos[T])

            # angle: building right-angle corners
            if int(r["feature_class"]) == 1 and geom.geom_type == "Polygon":
                ring = list(geom.exterior.coords)
                uq = ring[:-1] if (len(ring) > 1 and ring[0] == ring[-1]) else ring
                if len(uq) < 4:
                    continue
                ra = [
                    k
                    for k in range(len(uq))
                    if (ca := _corner_angle_deg(uq[k - 1], uq[k], uq[(k + 1) % len(uq)]))
                    is not None
                    and abs(ca - 90.0) <= 5.0
                ]
                if not ra:
                    continue
                for label, T in ANGLE_T.items():
                    dr = _reanchor_decode_ring(ring, T)
                    if dr is None:
                        continue
                    dec, mp, uqd = dr
                    for k in ra:
                        da = _corner_angle_deg(
                            dec[mp[(k - 1) % len(uqd)]], dec[mp[k]], dec[mp[(k + 1) % len(uqd)]]
                        )
                        if da is None:
                            continue
                        if label == "baseline_openloop":
                            angle_corners += 1
                        pd = abs(da - 90.0)
                        if pd > 45.0:
                            angle_cat[label] += 1
                        else:
                            angle_dev[label].append(pd)

    # add empty cells + stage-4
    for T in THRESHOLDS:
        for ti, tj in tile_keys:
            for ci in range(8):
                for cj in range(8):
                    _ = per_cell[T][(ti, tj, ci, cj)]
        for cell in nonempty:
            per_cell[T][cell] += STAGE_4

    rows = []
    for T in THRESHOLDS:
        lengths = sorted(per_cell[T].values())
        p999 = int(_q(lengths, 99.9))
        cmax = int(max(lengths))
        rows.append(
            {
                "threshold_m": T,
                "position_l_inf_m": _summ(pos[T]),
                "reanchor_triggers": reanchor_total[T],
                "net_token_delta_vs_baseline": token_delta_total[T],
                "net_token_delta_pct": round(100 * token_delta_total[T] / baseline_total, 3),
                "cell_p99_9_tokens": p999,
                "cell_max_tokens": cmax,
                "cell_p99_9_padded": ((p999 + 127) // 128) * 128,
                "exceeds_locked_6016_padded": ((p999 + 127) // 128) * 128 > PADDED_BUDGET,
                "trigger_flip_ties_within_1e-6m": ties_total[T],
            }
        )

    out = {
        "_status": "SCOPING - Halt 2 re-anchor option; NO lock changes.",
        "n_features": n_features,
        "baseline_total_tokens": baseline_total,
        "locked_padded_budget": PADDED_BUDGET,
        "grammar_sentinel_cost": (
            "Re-anchor needs a §3.2 GRAMMAR change (mid-feature absolute anchor); the "
            "decoder snaps on anchor-range tokens (300..395). NO new vocab/sentinel — "
            "reuses the existing anchor sub-block. Self-delimiting preserved (disjoint "
            "ID ranges). token_cost.py budget twin would need a re-anchor-aware update."
        ),
        "thresholds": rows,
        "angle_right_angle_corners": {
            "n_corners": angle_corners,
            "note": (
                "open-loop directions, no dithering; preserves angle p95 but the "
                "absolute snap (exact vertex mixed with drifted neighbours) worsens "
                "the angle TAIL (p99/catastrophic) vs baseline."
            ),
            **{
                label: {
                    "non_catastrophic_deg": _summ(angle_dev[label]),
                    "catastrophic_gt45": angle_cat[label],
                }
                for label in ANGLE_T
            },
        },
    }
    (ROOT / "reports" / "sub_f_halt2_reanchor_scoping.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
    )
    print("[reanchor] wrote reports/sub_f_halt2_reanchor_scoping.yaml")
    print(f"[reanchor] baseline total tokens={baseline_total} budget(padded)=6016")
    for row in rows:
        p = row["position_l_inf_m"]
        print(
            f"  T={row['threshold_m']:>5}m: pos p95={p['p95']} p99.9={p['p99_9']} max={p['max']} | "
            f"triggers={row['reanchor_triggers']} tok_delta={row['net_token_delta_pct']}% | "
            f"cell P99.9={row['cell_p99_9_tokens']} max={row['cell_max_tokens']} "
            f"padded={row['cell_p99_9_padded']} over6016={row['exceeds_locked_6016_padded']} | "
            f"ties={row['trigger_flip_ties_within_1e-6m']}"
        )
    av = out["angle_right_angle_corners"]
    print(f"[reanchor] angle (n={av['n_corners']} right-angle corners):")
    for label in ANGLE_T:
        nc = av[label]["non_catastrophic_deg"]
        print(f"   {label:<18} non-cat={nc} cat={av[label]['catastrophic_gt45']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
