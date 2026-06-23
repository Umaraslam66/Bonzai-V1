"""Read-only road-connectivity diagnostic on the 21 eyeball-probe cells.

NO generation, NO scoring, NO fix. Answers: once near-closed buildings are removed
from the road set (by construction identity, the viz reclassifier), do the model's
TRUE road segments form a connected network within each cell?

Method (per cell, coordinates are local cell units; quantum = 0.5):
  - true roads = road-class LineStrings (>=2 vertices). Building blocks and degenerate
    1-vertex roads / road Points are excluded from the segment graph (counted separately).
  - cluster segment endpoints within tolerance tau (union-find). A cluster with >=2
    endpoints is a junction; degree = #endpoints in it. Singleton clusters are dangling ends.
  - segments sharing a cluster are connected -> connected components.
  - secondary: endpoints landing within tau of ANOTHER segment's interior vertex (T-junctions
    the endpoint-only graph misses).

Run:  uv run python scripts/_road_connectivity_diag.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.eval.geometry import _is_building_block, _is_closed_ring
from cfm.inference.generate import try_decode_block

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "reports" / "_eyeball_probe" / "gen_tokens.json"

TAU_PRIMARY = 1.0  # cell units (~1 m = 2 quanta); junction snap tolerance
TAU_SWEEP = [0.5, 1.0, 2.0, 4.0]


class UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def cell_roads(tokens: list[int]) -> tuple[list[list[list[float]]], int, int, int]:
    """Return (road segments [>=2 pts], n_unsealed_building_lines, n_degenerate_roads, n_road_points).

    Only UNSEALED building lines masquerade as ``road`` in the raw GeoJSON: a building
    block whose ring closes exactly was promoted to a ``Polygon`` (kind=building), so it is
    NOT a raw "road". Unclosed building rings stay ``LineString`` -> kind=road in raw.
    """
    segments: list[list[list[float]]] = []
    n_unsealed_bld = 0  # building LineStrings that are kind=road in raw GeoJSON
    n_degenerate = 0  # road-class LineString with <2 distinct pts
    n_points = 0  # road-class Point
    for block in split_cell_into_features(tokens):
        g = try_decode_block(block)
        if g is None:
            continue
        is_bld = _is_building_block(block)
        t = g["type"]
        if t == "LineString":
            if is_bld:
                if not _is_closed_ring(g):  # sealed (closed) ones are raw Polygons, not "roads"
                    n_unsealed_bld += 1
                continue
            pts = [[float(x), float(y)] for x, y in g["coordinates"]]
            # drop consecutive dups
            dedup = [pts[0]] + [p for prev, p in zip(pts, pts[1:]) if p != prev]
            if len(dedup) >= 2:
                segments.append(dedup)
            else:
                n_degenerate += 1
        elif t == "Point" and not is_bld:
            n_points += 1
    return segments, n_unsealed_bld, n_degenerate, n_points


def connectivity(segments: list[list[list[float]]], tau: float) -> dict:
    """Endpoint-clustered connectivity metrics for one cell's road segments."""
    n = len(segments)
    if n == 0:
        return {"n_segments": 0}
    # endpoint index space: endpoint 2*i = start of seg i, 2*i+1 = end of seg i
    eps = []
    for s in segments:
        eps.append(tuple(s[0]))
        eps.append(tuple(s[-1]))
    uf = UF(2 * n)
    for a in range(2 * n):
        for b in range(a + 1, 2 * n):
            if math.dist(eps[a], eps[b]) <= tau:
                uf.union(a, b)
    # clusters of endpoints
    clusters: dict[int, list[int]] = {}
    for i in range(2 * n):
        clusters.setdefault(uf.find(i), []).append(i)
    junctions = {r: ep for r, ep in clusters.items() if len(ep) >= 2}
    dangling = sum(1 for ep in clusters.values() if len(ep) == 1)
    joined = 2 * n - dangling
    # segment graph: union segments that share an endpoint-cluster
    seg_uf = UF(n)
    for ep in clusters.values():
        segs = {e // 2 for e in ep}
        segs = list(segs)
        for k in range(1, len(segs)):
            seg_uf.union(segs[0], segs[k])
    comp_sizes = Counter(seg_uf.find(i) for i in range(n))
    n_comp = len(comp_sizes)
    largest = max(comp_sizes.values())
    degdist = Counter(min(len(ep), 5) for ep in junctions.values())  # cap label at 5+
    # secondary: endpoints near ANOTHER segment's interior vertex (T-junctions)
    t_touch = 0
    for i, s in enumerate(segments):
        for endp in (s[0], s[-1]):
            for j, s2 in enumerate(segments):
                if j == i:
                    continue
                interior = s2[1:-1]
                if any(math.dist(endp, v) <= tau for v in interior):
                    t_touch += 1
                    break
    return {
        "n_segments": n,
        "n_components": n_comp,
        "largest_cc": largest,
        "largest_cc_frac": largest / n,
        "comp_per_seg": n_comp / n,
        "n_endpoints": 2 * n,
        "dangling": dangling,
        "joined": joined,
        "dangling_frac": dangling / (2 * n),
        "n_junctions": len(junctions),
        "degdist": dict(degdist),
        "t_touch": t_touch,
    }


def main() -> None:
    data = json.loads(GEN.read_text())
    by_ctx: dict[str, list[dict]] = {}
    raw_bld_in_roads = 0
    raw_road_lines = (
        0  # unsealed-building lines + true road LineStrings = raw "road" LineString count
    )

    for rec in data["records"]:
        segs, n_unsealed_bld, n_degen, n_pts = cell_roads(rec["tokens"])
        m = connectivity(segs, TAU_PRIMARY)
        m["context"] = rec["context"]
        m["cell"] = rec["cell_index"]
        m["n_unsealed_bld"] = n_unsealed_bld
        m["n_degenerate"] = n_degen
        m["n_points"] = n_pts
        by_ctx.setdefault(rec["context"], []).append(m)
        raw_bld_in_roads += n_unsealed_bld
        raw_road_lines += n_unsealed_bld + len(segs) + n_degen

    print("=" * 78)
    print("RECLASSIFICATION (construction identity, before any connectivity)")
    print("=" * 78)
    true_roads = sum(len(cell_roads(r["tokens"])[0]) for r in data["records"])
    print(f"raw GeoJSON 'road' LineStrings: {raw_road_lines}")
    print(
        f"  ... of which were actually BUILDING footprints (unsealed): {raw_bld_in_roads} "
        f"({raw_bld_in_roads / raw_road_lines:.0%})"
    )
    print(f"  ... TRUE road segments (>=2 distinct pts): {true_roads}")
    degen_total = sum(c["n_degenerate"] for cells in by_ctx.values() for c in cells)
    pts_total = sum(c["n_points"] for cells in by_ctx.values() for c in cells)
    print(
        f"  (excluded from graph: {degen_total} degenerate 1-pt road lines, {pts_total} road Points)"
    )

    print(f"\nJunction tolerance tau = {TAU_PRIMARY} cell units (~1 m; quantum = 0.5).\n")

    print("=" * 78)
    print("PER-CONTEXT CONNECTIVITY (true roads only)")
    print("=" * 78)
    for ctx, cells in by_ctx.items():
        active = [c for c in cells if c["n_segments"] > 0]
        segs = [c["n_segments"] for c in active]
        comps = [c["n_components"] for c in active]
        cps = [c["comp_per_seg"] for c in active]
        lcf = [c["largest_cc_frac"] for c in active]
        dfr = [c["dangling_frac"] for c in active]
        pooled_deg: Counter = Counter()
        pooled_t = 0
        for c in active:
            for k, v in c["degdist"].items():
                pooled_deg[k] += v
            pooled_t += c["t_touch"]
        print(f"\n### {ctx}  ({len(active)}/{len(cells)} cells have road segments)")
        print(
            f"  segments/cell      median {statistics.median(segs):.0f}  range {min(segs)}-{max(segs)}"
        )
        print(
            f"  components/cell     median {statistics.median(comps):.0f}  range {min(comps)}-{max(comps)}"
        )
        print(
            f"  components/segment  median {statistics.median(cps):.2f}   "
            f"(1.00 = totally fragmented, ->0 = connected)"
        )
        print(
            f"  largest-CC fraction median {statistics.median(lcf):.2f}   "
            f"(1.00 = one network, ->1/n = all isolated)"
        )
        print(
            f"  dangling-endpoint frac median {statistics.median(dfr):.2f}  "
            f"(pooled {sum(c['dangling'] for c in active)}/{sum(c['n_endpoints'] for c in active)} "
            f"= {sum(c['dangling'] for c in active) / sum(c['n_endpoints'] for c in active):.2f})"
        )
        print(
            f"  junction degree dist (pooled): "
            f"{ {('5+' if k == 5 else k): pooled_deg[k] for k in sorted(pooled_deg)} }"
        )
        print(f"  endpoint->other-segment interior touches (T-junctions, pooled): {pooled_t}")

    print("\n" + "=" * 78)
    print("PER-CELL DETAIL")
    print("=" * 78)
    print(
        f"{'context':16s} {'cell':>4} {'segs':>4} {'comp':>4} {'lcc%':>5} {'dangl%':>6} {'junc':>4} {'Tj':>3}"
    )
    for ctx, cells in by_ctx.items():
        for c in cells:
            if c["n_segments"] == 0:
                print(f"{ctx:16s} {c['cell']:>4} {0:>4}  (no road segments)")
                continue
            print(
                f"{ctx:16s} {c['cell']:>4} {c['n_segments']:>4} {c['n_components']:>4} "
                f"{c['largest_cc_frac'] * 100:>4.0f}% {c['dangling_frac'] * 100:>5.0f}% "
                f"{c['n_junctions']:>4} {c['t_touch']:>3}"
            )

    print("\n" + "=" * 78)
    print("TOLERANCE SENSITIVITY (median components/segment per context)")
    print("=" * 78)
    print(f"{'context':16s} " + "  ".join(f"tau={t:>4}" for t in TAU_SWEEP))
    for ctx in by_ctx:
        recs = [r for r in data["records"] if r["context"] == ctx]
        row = []
        for tau in TAU_SWEEP:
            vals = []
            for r in recs:
                segs, *_ = cell_roads(r["tokens"])
                m = connectivity(segs, tau)
                if m["n_segments"]:
                    vals.append(m["comp_per_seg"])
            row.append(statistics.median(vals) if vals else float("nan"))
        print(f"{ctx:16s} " + "  ".join(f"{v:>8.2f}" for v in row))


if __name__ == "__main__":
    main()
