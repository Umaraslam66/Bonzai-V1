"""Open-loop direction-count sweep: position AND angle round-trip (NO lock changes).

The error-feedback option is disqualified (it fixes position but regresses the
right-angle-corner round-trip: non-catastrophic p95 7.5deg -> 22.5deg, catastrophic
3,495 -> 33,735). So the comparison reopens to the clean fix: more directions
(open-loop, no dithering -> improves BOTH position and angle). This measures, on
real Singapore data, the position L_inf distribution AND the right-angle-corner
post-deviation, for N in {48, 144, 192, 256, 360}, so the reviewer can pick N
against both Halt 2 thresholds at once.

Run:
    uv run python scripts/sub_f/scope_halt2_direction_sweep.py \
        --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.encoder import quantize_coord_m

ROOT = Path(__file__).resolve().parents[2]
QUANTUM_M = 0.5
MAX_MAG_Q = 64
N_VALUES = (48, 144, 192, 256, 360)


def _dbin(angle_deg: float, n: int) -> int:
    return int((angle_deg % 360.0) // (360.0 / n)) % n


def _corner_angle_deg(prev, cur, nxt) -> float | None:
    ax, ay = prev[0] - cur[0], prev[1] - cur[1]
    bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cosv))


def _seg_chunks(total_q: int) -> list[int]:
    total_q = max(1, total_q)
    out, rem = [], total_q
    while rem > 0:
        c = min(MAX_MAG_Q, rem)
        out.append(c)
        rem -= c
    return out


def _encode_decode_openloop(coords, n: int, closed: bool):
    original = coords[:-1] if (closed and coords[0] == coords[-1]) else coords
    if len(original) < 2:
        return None
    dx0 = quantize_coord_m(original[0][0]) * QUANTUM_M
    dy0 = quantize_coord_m(original[0][1]) * QUANTUM_M
    decoded = [(dx0, dy0)]
    mapping = [0]
    seg_count = len(original) if closed else len(original) - 1
    for idx in range(seg_count):
        a = original[idx]
        b = original[(idx + 1) % len(original)]
        total_q = max(1, quantize_coord_m(math.hypot(b[0] - a[0], b[1] - a[1])))
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        arad = math.radians(_dbin(ang, n) * (360.0 / n))
        for cq in _seg_chunks(total_q):
            cur = decoded[-1]
            decoded.append(
                (cur[0] + cq * QUANTUM_M * math.cos(arad), cur[1] + cq * QUANTUM_M * math.sin(arad))
            )
        if not closed or idx < len(original) - 1:
            mapping.append(len(decoded) - 1)
    if closed:
        decoded.append(decoded[0])
    return decoded, mapping


def _pos_linf(coords, n, closed):
    r = _encode_decode_openloop(coords, n, closed)
    if r is None:
        return 0.0
    decoded, mapping = r
    original = coords[:-1] if (closed and coords[0] == coords[-1]) else coords
    return max(
        max(abs(p[0] - decoded[mapping[i]][0]), abs(p[1] - decoded[mapping[i]][1]))
        for i, p in enumerate(original)
    )


def _summ(vals):
    s = sorted(vals)
    nn = len(s)
    if not nn:
        return {"n": 0}

    def pc(q):
        return round(s[min(nn - 1, int(q / 100 * nn))], 3)

    return {"p95": pc(95), "p99": pc(99), "p99_9": pc(99.9), "max": round(s[-1], 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = ap.parse_args()

    pos = {n: [] for n in N_VALUES}
    ang_noncat = {n: [] for n in N_VALUES}
    ang_cat = {n: 0 for n in N_VALUES}
    n_corners = 0

    for path in sorted(args.sub_c_region_dir.glob("tile=*/features.parquet")):
        for r in pq.ParquetFile(path).read().to_pylist():
            fc = int(r["feature_class"])
            geom = wkb_loads(r["geometry"])
            gt = geom.geom_type
            if gt == "LineString":
                parts = [(list(geom.coords), False)]
            elif gt == "Polygon":
                parts = [(list(geom.exterior.coords), True)]
            elif gt == "MultiLineString":
                parts = [(list(p.coords), False) for p in geom.geoms]
            elif gt == "MultiPolygon":
                parts = [(list(p.exterior.coords), True) for p in geom.geoms]
            else:
                continue
            for coords, closed in parts:
                if len(coords) < 2:
                    continue
                for n in N_VALUES:
                    pos[n].append(_pos_linf(coords, n, closed))

            if fc == 1 and gt == "Polygon":
                ring = list(geom.exterior.coords)
                uniq = ring[:-1] if ring[0] == ring[-1] else ring
                if len(uniq) < 4:
                    continue
                ra = [
                    k
                    for k in range(len(uniq))
                    if (
                        (ca := _corner_angle_deg(uniq[k - 1], uniq[k], uniq[(k + 1) % len(uniq)]))
                        is not None
                        and abs(ca - 90.0) <= 5.0
                    )
                ]
                if not ra:
                    continue
                for n in N_VALUES:
                    rr = _encode_decode_openloop(ring, n, True)
                    if rr is None:
                        continue
                    decoded, mapping = rr
                    for k in ra:
                        da = _corner_angle_deg(
                            decoded[mapping[(k - 1) % len(uniq)]],
                            decoded[mapping[k]],
                            decoded[mapping[(k + 1) % len(uniq)]],
                        )
                        if da is None:
                            continue
                        if n == N_VALUES[0]:
                            n_corners += 1
                        pd = abs(da - 90.0)
                        if pd > 45.0:
                            ang_cat[n] += 1
                        else:
                            ang_noncat[n].append(pd)

    rows = []
    for n in N_VALUES:
        rows.append(
            {
                "N_directions": n,
                "bin_width_deg": round(360.0 / n, 3),
                "position_l_inf_m": _summ(pos[n]),
                "angle_noncatastrophic_deg": _summ(ang_noncat[n]),
                "angle_catastrophic_gt45_count": ang_cat[n],
                "position_p99_9_holds_4_8m": _summ(pos[n]).get("p99_9", 99) <= 4.8,
                "angle_p95_holds_7_5deg": _summ(ang_noncat[n]).get("p95", 99) <= 7.5,
            }
        )
    out = {
        "_status": "SCOPING - open-loop direction sweep (position + angle); NO lock changes.",
        "n_right_angle_corners": n_corners,
        "note": (
            "open-loop (no dithering) so angle is preserved/improved unlike error-feedback. "
            "Pick N holding BOTH position AND angle at the lock's derivation statistic."
        ),
        "sweep": rows,
    }
    (ROOT / "reports" / "sub_f_halt2_direction_sweep.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
    )
    print("[sweep] wrote reports/sub_f_halt2_direction_sweep.yaml")
    print(f"[sweep] right-angle corners: {n_corners}")
    for row in rows:
        p = row["position_l_inf_m"]
        a = row["angle_noncatastrophic_deg"]
        pos_ok = "OK" if row["position_p99_9_holds_4_8m"] else "no"
        ang_ok = "OK" if row["angle_p95_holds_7_5deg"] else "no"
        print(
            f"  N={row['N_directions']:>3} ({row['bin_width_deg']:>5}deg): "
            f"pos p99.9={p['p99_9']:>6} max={p['max']:>6} [{pos_ok}] | "
            f"angle p95={a['p95']:>5} cat={row['angle_catastrophic_gt45_count']:>5} [{ang_ok}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
