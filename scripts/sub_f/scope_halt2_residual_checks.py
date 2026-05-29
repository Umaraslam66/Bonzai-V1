"""Pre-lock residual checks for the Halt 2 error-feedback proposal (NO lock changes).

Three checks the reviewer requires BEFORE the lock proposal is finalised:

1. TAIL — identify the feature(s) with the largest feedback round-trip L_inf
   (the ~9.66m max) and describe their geometry, so the gate posture
   (p99.9 <= 4.8m + documented tail) can be ratified knowing what the tail IS.
2. ANGLE — the OTHER Halt 2 threshold (right-angle-corner post-deviation, derived
   as the non-catastrophic p95 = 7.5deg). Measure it under BASELINE vs FEEDBACK on
   real building corners. Error-feedback must be same-or-better, not a regression.
3. DETERMINISM — feedback's direction depends on an ACCUMULATING decoded position
   (a determinism-risk shape). Verify within-process byte-identity of the feedback
   token stream, and quantify cross-runtime exposure (chunk directions landing
   within float-noise of a 7.5deg bin boundary, where a cross-libm 1-ULP atan2
   difference could flip the bin) vs baseline.

Geometry-only + faithful corner-angle helper imported from the Halt 2 analyzer.

Run:
    uv run python scripts/sub_f/scope_halt2_residual_checks.py \
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

from cfm.data.sub_f.encoder import direction_bin, quantize_coord_m

ROOT = Path(__file__).resolve().parents[2]
QUANTUM_M = 0.5
MAX_MAG_Q = 64
BIN_DEG_48 = 360.0 / 48
TIE_EPS_DEG = 1e-7  # cross-libm atan2 noise proxy near a bin boundary


def _corner_angle_deg(prev, cur, nxt) -> float | None:
    ax, ay = prev[0] - cur[0], prev[1] - cur[1]
    bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
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


def _encode_decode(coords, *, feedback: bool, closed: bool):
    """Return (decoded_coords, original_to_decoded, dir_tokens, tie_count).

    dir_tokens: the sequence of direction-bin indices emitted (for determinism /
    tie analysis). tie_count: chunk directions whose true angle is within
    TIE_EPS_DEG of a 48-bin boundary.
    """
    original = coords[:-1] if (closed and len(coords) > 1 and coords[0] == coords[-1]) else coords
    if len(original) < 2:
        return None
    dx0 = quantize_coord_m(original[0][0]) * QUANTUM_M
    dy0 = quantize_coord_m(original[0][1]) * QUANTUM_M
    decoded = [(dx0, dy0)]
    mapping = [0]
    dir_tokens: list[int] = []
    ties = 0
    seg_count = len(original) if closed else len(original) - 1
    for idx in range(seg_count):
        a = original[idx]
        b = original[(idx + 1) % len(original)]
        total_q = max(1, quantize_coord_m(math.hypot(b[0] - a[0], b[1] - a[1])))
        if not feedback:
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            dbin = direction_bin(ang)
            arad = math.radians(dbin * BIN_DEG_48)
            frac = ang % BIN_DEG_48
            if min(frac, BIN_DEG_48 - frac) < TIE_EPS_DEG:
                ties += 1
            for cq in _seg_chunks(total_q):
                cur = decoded[-1]
                decoded.append(
                    (
                        cur[0] + cq * QUANTUM_M * math.cos(arad),
                        cur[1] + cq * QUANTUM_M * math.sin(arad),
                    )
                )
                dir_tokens.append(dbin)
        else:
            for cq in _seg_chunks(total_q):
                cur = decoded[-1]
                ang = math.degrees(math.atan2(b[1] - cur[1], b[0] - cur[0]))
                dbin = direction_bin(ang)
                frac = ang % BIN_DEG_48
                if min(frac, BIN_DEG_48 - frac) < TIE_EPS_DEG:
                    ties += 1
                arad = math.radians(dbin * BIN_DEG_48)
                decoded.append(
                    (
                        cur[0] + cq * QUANTUM_M * math.cos(arad),
                        cur[1] + cq * QUANTUM_M * math.sin(arad),
                    )
                )
                dir_tokens.append(dbin)
        if not closed or idx < len(original) - 1:
            mapping.append(len(decoded) - 1)
    if closed:
        decoded.append(decoded[0])
    return decoded, mapping, dir_tokens, ties


def _feature_l_inf(coords, *, feedback: bool, closed: bool) -> float:
    r = _encode_decode(coords, feedback=feedback, closed=closed)
    if r is None:
        return 0.0
    decoded, mapping, _dt, _t = r
    original = coords[:-1] if (closed and coords[0] == coords[-1]) else coords
    m = 0.0
    for i, p in enumerate(original):
        dp = decoded[mapping[i]]
        m = max(m, abs(p[0] - dp[0]), abs(p[1] - dp[1]))
    return m


def _summ(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return {"n": 0}
    pc = lambda q: round(s[min(n - 1, int(q / 100 * n))], 4)  # noqa: E731
    return {"n": n, "p50": pc(50), "p95": pc(95), "p99": pc(99), "max": round(s[-1], 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = ap.parse_args()

    role = {0: "roads", 1: "buildings", 2: "pois", 3: "base"}
    top_tail: list[tuple] = []  # (linf, descriptor)
    angle_post_dev = {"baseline": [], "feedback": []}
    angle_catastrophic = {"baseline": 0, "feedback": 0}
    angle_total_corners = 0
    det_mismatch = 0
    ties = {"baseline": 0, "feedback": 0}
    total_chunks = 0
    i = 0

    for path in sorted(args.sub_c_region_dir.glob("tile=*/features.parquet")):
        for r in pq.ParquetFile(path).read().to_pylist():
            i += 1
            fc = int(r["feature_class"])
            geom = wkb_loads(r["geometry"])
            gt = geom.geom_type
            # iterate single parts
            if gt == "LineString":
                partlist = [(list(geom.coords), False)]
            elif gt == "Polygon":
                partlist = [(list(geom.exterior.coords), True)]
            elif gt == "MultiLineString":
                partlist = [(list(p.coords), False) for p in geom.geoms]
            elif gt == "MultiPolygon":
                partlist = [(list(p.exterior.coords), True) for p in geom.geoms]
            else:
                continue

            for coords, closed in partlist:
                if len(coords) < 2:
                    continue
                # CHECK 1: tail
                li = _feature_l_inf(coords, feedback=True, closed=closed)
                if len(top_tail) < 8 or li > top_tail[-1][0]:
                    open_coords = coords[:-1] if (closed and coords[0] == coords[-1]) else coords
                    segs = [
                        math.hypot(
                            open_coords[k][0] - open_coords[k - 1][0],
                            open_coords[k][1] - open_coords[k - 1][1],
                        )
                        for k in range(1, len(open_coords))
                    ]
                    # max turn angle
                    max_turn = 0.0
                    for k in range(1, len(open_coords) - 1):
                        ca = _corner_angle_deg(
                            open_coords[k - 1], open_coords[k], open_coords[k + 1]
                        )
                        if ca is not None:
                            max_turn = max(max_turn, abs(180.0 - ca))
                    desc = {
                        "l_inf_m": round(li, 3),
                        "type": role[fc],
                        "geom_type": gt,
                        "n_vertices": len(open_coords),
                        "max_segment_m": round(max(segs), 1) if segs else 0.0,
                        "total_len_m": round(sum(segs), 1),
                        "max_turn_deg": round(max_turn, 1),
                        "shapely_is_simple": bool(geom.is_simple),
                    }
                    top_tail.append((li, desc))
                    top_tail.sort(key=lambda x: -x[0])
                    top_tail[:] = top_tail[:8]

                # CHECK 3: determinism within-process + tie exposure (buildings/roads only, sampled)
                if i % 5 == 0:
                    r1 = _encode_decode(coords, feedback=True, closed=closed)
                    r2 = _encode_decode(coords, feedback=True, closed=closed)
                    if r1 and r2 and r1[2] != r2[2]:
                        det_mismatch += 1
                    if r1:
                        ties["feedback"] += r1[3]
                        total_chunks += len(r1[2])
                    rb = _encode_decode(coords, feedback=False, closed=closed)
                    if rb:
                        ties["baseline"] += rb[3]

            # CHECK 2: angle round-trip on building right-angle corners
            if fc == 1 and gt == "Polygon":
                ring = list(geom.exterior.coords)
                uniq = ring[:-1] if ring[0] == ring[-1] else ring
                if len(uniq) < 4:
                    continue
                ra_idx, ra_in = [], []
                for k in range(len(uniq)):
                    ca = _corner_angle_deg(uniq[k - 1], uniq[k], uniq[(k + 1) % len(uniq)])
                    if ca is not None and abs(ca - 90.0) <= 5.0:
                        ra_idx.append(k)
                        ra_in.append(ca)
                if not ra_idx:
                    continue
                for variant in ("baseline", "feedback"):
                    rr = _encode_decode(ring, feedback=(variant == "feedback"), closed=True)
                    if rr is None:
                        continue
                    decoded, mapping, _dt, _t = rr
                    for k in ra_idx:
                        da = _corner_angle_deg(
                            decoded[mapping[(k - 1) % len(uniq)]],
                            decoded[mapping[k]],
                            decoded[mapping[(k + 1) % len(uniq)]],
                        )
                        if da is None:
                            continue
                        pd = abs(da - 90.0)
                        if variant == "baseline":
                            angle_total_corners += 1
                        if pd > 45.0:
                            angle_catastrophic[variant] += 1
                        else:
                            angle_post_dev[variant].append(pd)

    out = {
        "_status": "SCOPING - Halt 2 pre-lock residual checks; NO lock changes.",
        "check_1_feedback_tail_top8": [d for _li, d in top_tail],
        "check_2_angle_roundtrip_right_angle_corners": {
            "metric": (
                "post-roundtrip |corner_angle - 90deg| on building corners with input "
                "within 5deg of 90; non-catastrophic p95 is the Halt 2 basis (7.5deg)"
            ),
            "total_right_angle_corners_measured": angle_total_corners,
            "baseline_non_catastrophic_deg": _summ(angle_post_dev["baseline"]),
            "feedback_non_catastrophic_deg": _summ(angle_post_dev["feedback"]),
            "baseline_catastrophic_gt45_count": angle_catastrophic["baseline"],
            "feedback_catastrophic_gt45_count": angle_catastrophic["feedback"],
        },
        "check_3_determinism": {
            "within_process_token_stream_mismatches": det_mismatch,
            "within_process_verdict": "DETERMINISTIC" if det_mismatch == 0 else "NON-DETERMINISTIC",
            "feedback_chunk_dirs_within_1e-7deg_of_bin_boundary": ties["feedback"],
            "baseline_chunk_dirs_within_1e-7deg_of_bin_boundary": ties["baseline"],
            "total_feedback_chunks_sampled": total_chunks,
            "cross_runtime_note": (
                "Within-env determinism is what BP5 locks; cross-env (dev darwin/arm64 "
                "vs Leonardo linux/x86_64) is the existing end-of-Phase-1 deferral (spec "
                "1.4 #4). Tie exposure above estimates how many feedback direction "
                "decisions a cross-libm atan2 1-ULP difference could flip; ~0 means feedback "
                "does not materially raise the deferred cross-env risk."
            ),
        },
    }
    out_path = ROOT / "reports" / "sub_f_halt2_residual_checks.yaml"
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"[residual] wrote {out_path}")
    print("[residual] CHECK 1 tail top-3:")
    for _li, d in top_tail[:3]:
        print(f"   {d}")
    a = out["check_2_angle_roundtrip_right_angle_corners"]
    print(f"[residual] CHECK 2 angle (n={a['total_right_angle_corners_measured']} corners):")
    bnc, bcat = a["baseline_non_catastrophic_deg"], a["baseline_catastrophic_gt45_count"]
    fnc, fcat = a["feedback_non_catastrophic_deg"], a["feedback_catastrophic_gt45_count"]
    print(f"   baseline non-catastrophic: {bnc} catastrophic={bcat}")
    print(f"   feedback non-catastrophic: {fnc} catastrophic={fcat}")
    c = out["check_3_determinism"]
    print(
        f"[residual] CHECK 3 determinism: {c['within_process_verdict']} "
        f"(mismatches={c['within_process_token_stream_mismatches']})"
    )
    fb_tie = c["feedback_chunk_dirs_within_1e-7deg_of_bin_boundary"]
    bl_tie = c["baseline_chunk_dirs_within_1e-7deg_of_bin_boundary"]
    print(
        f"   tie exposure feedback={fb_tie} baseline={bl_tie} "
        f"of {c['total_feedback_chunks_sampled']} chunks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
