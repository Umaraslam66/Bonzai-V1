"""Geometry-validity metric (spec §4): echo-immune structural metrics per context.

Operates on a probe ``gen_tokens.json`` (records with tokens, context, self_terminated).
Classifies features by CONSTRUCTION IDENTITY (building-class tokens) so near-closed
buildings are NOT counted as roads, then reports the metrics char_stats does NOT hand the
model: building closure-gap distribution, road fragmentation (components/segments at
tolerance tau), self-term %, decode %. Counts are reported but are echo-tainted
(char_stats carries count/presence) — descriptive only.

Closure logic mirrors viz/build_data; connectivity mirrors
scripts/_road_connectivity_diag.py (this is its reusable home).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.eval.geometry import _is_building_block, _is_closed_ring
from cfm.inference.generate import try_decode_block


@dataclass(frozen=True)
class ContextGeometry:
    context: str
    n_cells: int
    self_term_frac: float
    decode_frac: float
    closure_gap_median: float
    closure_gap_p90: float
    closure_within_2pct: float
    closure_within_5pct: float
    median_components_per_segment: float
    median_largest_cc_frac: float
    dangling_endpoint_frac: float
    counts: dict[str, int]


@dataclass(frozen=True)
class GeometryValidityReport:
    per_context: dict[str, ContextGeometry]


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def _closure_gap(coords: list[list[float]]) -> float | None:
    if not coords or len(coords) < 4:
        return None
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    return math.hypot(x1 - x0, y1 - y0) / diag


def _road_connectivity(segments: list[list[list[float]]], tau: float) -> tuple[float, float, float]:
    """(components/segments, largest_cc_frac, dangling_endpoint_frac) for one cell."""
    n = len(segments)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    eps = []
    for s in segments:
        eps.append(tuple(s[0]))
        eps.append(tuple(s[-1]))
    uf = _UF(2 * n)
    for a in range(2 * n):
        for b in range(a + 1, 2 * n):
            if math.dist(eps[a], eps[b]) <= tau:
                uf.union(a, b)
    clusters: dict[int, list[int]] = {}
    for i in range(2 * n):
        clusters.setdefault(uf.find(i), []).append(i)
    dangling = sum(1 for ep in clusters.values() if len(ep) == 1)
    seg_uf = _UF(n)
    for ep in clusters.values():
        segs = list({e // 2 for e in ep})
        for k in range(1, len(segs)):
            seg_uf.union(segs[0], segs[k])
    comp_roots = {seg_uf.find(i) for i in range(n)}
    sizes: dict[int, int] = {}
    for i in range(n):
        r = seg_uf.find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return (len(comp_roots) / n, max(sizes.values()) / n, dangling / (2 * n))


def _classify_cell(
    tokens: list[int],
) -> tuple[dict[str, int], list[float], list[list[list[float]]]]:
    counts = {
        "building_sealed": 0,
        "building_unsealed": 0,
        "road": 0,
        "road_node": 0,
        "undecodable": 0,
    }
    closure_gaps: list[float] = []
    segments: list[list[list[float]]] = []
    for block in split_cell_into_features(tokens):
        g = try_decode_block(block)
        if g is None:
            counts["undecodable"] += 1
            continue
        t = g["type"]
        if _is_building_block(block):
            if t == "LineString" and _is_closed_ring(g):
                counts["building_sealed"] += 1
                closure_gaps.append(0.0)
            elif t in ("Polygon", "MultiPolygon"):
                counts["building_sealed"] += 1
                closure_gaps.append(0.0)
            else:
                counts["building_unsealed"] += 1
                if t == "LineString":
                    gap = _closure_gap([[float(x), float(y)] for x, y in g["coordinates"]])
                    if gap is not None:
                        closure_gaps.append(gap)
        elif t == "Point":
            counts["road_node"] += 1
        elif t in ("LineString", "MultiLineString"):
            counts["road"] += 1
            pts = [[float(x), float(y)] for x, y in g["coordinates"]]
            dedup = [pts[0]] + [p for prev, p in pairwise(pts) if p != prev]
            if len(dedup) >= 2:
                segments.append(dedup)
    return counts, closure_gaps, segments


def _frac(values: list[float], thresh: float) -> float:
    return sum(1 for v in values if v <= thresh) / len(values) if values else 0.0


def geometry_validity_report(gen_tokens_path: Path, *, tau: float = 1.0) -> GeometryValidityReport:
    data = json.loads(Path(gen_tokens_path).read_text())
    by_ctx: dict[str, list[dict]] = {}
    for rec in data["records"]:
        by_ctx.setdefault(rec["context"], []).append(rec)

    per_context: dict[str, ContextGeometry] = {}
    for ctx, recs in by_ctx.items():
        counts = {
            "building_sealed": 0,
            "building_unsealed": 0,
            "road": 0,
            "road_node": 0,
            "undecodable": 0,
        }
        all_gaps: list[float] = []
        cps: list[float] = []  # components/segment per cell
        lcc: list[float] = []
        dangling_num = 0
        dangling_den = 0
        n_blocks = 0
        n_self = 0
        for rec in recs:
            c, gaps, segs = _classify_cell(rec["tokens"])
            for k in counts:
                counts[k] += c[k]
            all_gaps.extend(gaps)
            n_blocks += sum(c.values())
            if rec.get("self_terminated"):
                n_self += 1
            if segs:
                r, lc, dfrac = _road_connectivity(segs, tau)
                cps.append(r)
                lcc.append(lc)
                dangling_num += round(dfrac * 2 * len(segs))
                dangling_den += 2 * len(segs)
        decoded = n_blocks - counts["undecodable"]
        per_context[ctx] = ContextGeometry(
            context=ctx,
            n_cells=len(recs),
            self_term_frac=n_self / len(recs),
            decode_frac=decoded / n_blocks if n_blocks else 0.0,
            closure_gap_median=statistics.median(all_gaps) if all_gaps else float("nan"),
            closure_gap_p90=(
                sorted(all_gaps)[int(0.9 * (len(all_gaps) - 1))] if all_gaps else float("nan")
            ),
            closure_within_2pct=_frac(all_gaps, 0.02),
            closure_within_5pct=_frac(all_gaps, 0.05),
            median_components_per_segment=statistics.median(cps) if cps else float("nan"),
            median_largest_cc_frac=statistics.median(lcc) if lcc else float("nan"),
            dangling_endpoint_frac=dangling_num / dangling_den if dangling_den else float("nan"),
            counts=counts,
        )
    return GeometryValidityReport(per_context=per_context)
