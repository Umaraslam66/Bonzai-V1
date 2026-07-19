"""Local decode + render of the eyeball generation probe (runs on the Mac, no GPU).

Reads the JSON of generated token sequences (pulled from Leonardo), decodes each cell via
the SEALED decoder (split_cell_into_features -> try_decode_block -> promote_building_rings),
writes a GeoJSON FeatureCollection + a PNG per cell, per-context montages, and prints a
directional summary (buildings/roads per cell by context). NO scoring — eyeball only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.eval.geometry import promote_building_rings
from cfm.inference.generate import try_decode_block


def decode_cell(tokens: list[int]) -> tuple[list[dict], int, int]:
    """tokens -> (promoted geojson geom dicts, n_blocks_attempted, n_decoded)."""
    blocks = split_cell_into_features(tokens)
    pairs = [(b, try_decode_block(b)) for b in blocks]
    ok_blocks = [b for b, d in pairs if d is not None]
    ok_geoms = [d for b, d in pairs if d is not None]
    promoted = promote_building_rings(ok_blocks, ok_geoms) if ok_geoms else []
    return promoted, len(blocks), len(ok_geoms)


def classify(geoms: list[dict]) -> tuple[list, list, list]:
    buildings = [g for g in geoms if g.get("type") in ("Polygon", "MultiPolygon")]
    roads = [g for g in geoms if g.get("type") in ("LineString", "MultiLineString")]
    points = [g for g in geoms if g.get("type") == "Point"]
    return buildings, roads, points


def render_png(geoms: list[dict], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    buildings, roads, points = classify(geoms)
    for g in roads:
        try:
            geom = shape(g)
        except Exception:
            continue
        for ls in getattr(geom, "geoms", [geom]):
            xs, ys = ls.xy
            ax.plot(xs, ys, color="#444", linewidth=0.8, zorder=1)
    for g in buildings:
        try:
            geom = shape(g)
        except Exception:
            continue
        for poly in getattr(geom, "geoms", [geom]):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, facecolor="#9ecae1", edgecolor="#08519c", linewidth=0.6, zorder=2)
    for g in points:
        x, y = g["coordinates"]
        ax.plot([x], [y], "o", color="#d62728", markersize=2, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="reports/_eyeball_probe")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    print(f"[render] meta={data['meta']}  max_new={data['max_new']}")
    out = Path(args.out)
    (out / "geojson").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)

    by_ctx: dict[str, list[dict]] = {}
    rows = []
    for rec in data["records"]:
        ctx, i = rec["context"], rec["cell_index"]
        geoms, n_blocks, n_dec = decode_cell(rec["tokens"])
        buildings, roads, points = classify(geoms)
        decodability = (n_dec / n_blocks) if n_blocks else 0.0
        stem = f"{ctx}_{i}"
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": g,
                 "properties": {"kind": "building" if g["type"].endswith("Polygon") else "road"}}
                for g in geoms
            ],
        }
        (out / "geojson" / f"{stem}.geojson").write_text(json.dumps(fc))
        title = (f"{ctx} #{i} | b={len(buildings)} r={len(roads)} "
                 f"dec={decodability:.0%} ntok={rec['n_tokens']}")
        render_png(geoms, title, out / "png" / f"{stem}.png")
        row = {"context": ctx, "i": i, "n_tokens": rec["n_tokens"],
               "n_blocks": n_blocks, "n_decoded": n_dec, "decodability": decodability,
               "n_buildings": len(buildings), "n_roads": len(roads), "n_points": len(points),
               "self_terminated": rec.get("self_terminated"), "hit_cap": rec.get("hit_cap")}
        rows.append(row)
        by_ctx.setdefault(ctx, []).append(geoms)

    # per-context montage
    for ctx, cells in by_ctx.items():
        n = len(cells)
        cols = min(n, 4)
        import math
        rws = math.ceil(n / cols)
        fig, axes = plt.subplots(rws, cols, figsize=(cols * 3, rws * 3))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, geoms in zip(axes, cells):
            b, r, _ = classify(geoms)
            for g in r:
                try:
                    geom = shape(g)
                except Exception:
                    continue
                for ls in getattr(geom, "geoms", [geom]):
                    xs, ys = ls.xy
                    ax.plot(xs, ys, color="#444", linewidth=0.7)
            for g in b:
                try:
                    geom = shape(g)
                except Exception:
                    continue
                for poly in getattr(geom, "geoms", [geom]):
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, facecolor="#9ecae1", edgecolor="#08519c", linewidth=0.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=4)
        for ax in axes[len(cells):]:
            ax.axis("off")
        fig.suptitle(f"context: {ctx}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"montage_{ctx}.png", dpi=110)
        plt.close(fig)

    # summary
    print("\n=== per-cell ===")
    print(f"{'context':18s} {'i':>2} {'ntok':>5} {'blk':>4} {'dec':>4} {'bld':>4} {'road':>4} {'pt':>3} selfterm")
    for r in rows:
        print(f"{r['context']:18s} {r['i']:>2} {r['n_tokens']:>5} {r['n_blocks']:>4} "
              f"{r['decodability']:>3.0%} {r['n_buildings']:>4} {r['n_roads']:>4} {r['n_points']:>3} "
              f"{r['self_terminated']}")
    print("\n=== per-context medians (DIRECTIONAL read) ===")
    print(f"{'context':18s} {'med_ntok':>8} {'med_dec':>7} {'med_bld':>7} {'med_road':>8}")
    for ctx in by_ctx:
        cr = [r for r in rows if r["context"] == ctx]
        med = lambda k: statistics.median([r[k] for r in cr])
        print(f"{ctx:18s} {int(med('n_tokens')):>8} {med('decodability'):>6.0%} "
              f"{med('n_buildings'):>7.1f} {med('n_roads'):>8.1f}")
    print(f"\n[render] wrote PNG+GeoJSON per cell + montages to {out}/")


if __name__ == "__main__":
    main()
