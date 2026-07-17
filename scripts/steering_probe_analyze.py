"""Thin analysis CLI for the generation-steering probe (SPEC 2026-07-17).

Loads the shard JSONs emitted by ``steering_probe_gen.py``, decodes each record's tokens to
features via the SAME sealed decode + classification route the gen_realism / eyeball lanes use
(``split_cell_into_features`` -> ``try_decode_block`` ->
``conditioning_discrimination._tile_features``, the ONE feature-classification rule), and dumps
a tidy PER-CELL metrics JSON.

It STOPS at per-cell metrics — it computes NO paired deltas, sign tests, effect sizes, or
verdicts. Those live in ``cfm.eval.steering_stats`` (ORCHESTRATOR-owned). If that module exists
at runtime this CLI hands it the per-cell metrics; otherwise it writes the metrics and prints a
clear note that verdict computation is orchestrator-owned.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.eval.conditioning_discrimination import _tile_features
from cfm.eval.realism import FeatureMetric
from cfm.inference.generate import try_decode_block

_BUILDING = FeatureMetric.BUILDING_AREA.value
_ROAD = FeatureMetric.ROAD_LENGTH.value


def decode_record_metrics(tokens: list[int]) -> dict[str, float | int]:
    """Decode ONE generated cell's tokens to the spec's per-cell outcome metrics.

    Route (identical to gen_realism / _eyeball_render): split into feature blocks, decode each
    (skip undecodable, mirroring sub-G ``check_decodability``), then classify the ALIGNED
    (block, geom) pairs with ``_tile_features`` (ring promotion + outbound-bref exclusion) — the
    density argument is irrelevant to these outcome metrics, so a constant 0 is passed.
    """
    blocks = split_cell_into_features(tokens)
    pairs = [(b, try_decode_block(b)) for b in blocks]
    ok_blocks = [b for b, g in pairs if g is not None]
    ok_geoms = [g for b, g in pairs if g is not None]
    feats, _n_bref = _tile_features(ok_blocks, ok_geoms, [0] * len(ok_blocks))

    road_lengths = [v for m, v, _d in feats if m == _ROAD]
    building_areas = [v for m, v, _d in feats if m == _BUILDING]
    building_areas_sorted = sorted(building_areas)
    if building_areas_sorted:
        mid = len(building_areas_sorted) // 2
        median_area = (
            building_areas_sorted[mid]
            if len(building_areas_sorted) % 2
            else (building_areas_sorted[mid - 1] + building_areas_sorted[mid]) / 2
        )
    else:
        median_area = 0.0
    return {
        "n_features": len(feats),
        "n_road_segments": len(road_lengths),
        "total_road_length": float(sum(road_lengths)),
        "n_buildings": len(building_areas),
        "total_building_area": float(sum(building_areas)),
        "median_building_area": float(median_area),
        "n_tokens": len(tokens),
        # Decode visibility: all-zero metrics from an UNDECODABLE cell must be distinguishable
        # from a genuinely empty generation (aggregate-signal-hides-subsets).
        "n_blocks": len(blocks),
        "n_decoded_blocks": len(ok_blocks),
    }


def load_shards(pattern: str) -> list[dict[str, Any]]:
    """Load + merge shard JSONs (glob), returning the flat list of records with each record's
    ckpt_id filled from its shard ``meta`` when absent."""
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f"no shard files matched {pattern!r}")
    records: list[dict[str, Any]] = []
    for p in paths:
        payload = json.loads(Path(p).read_text())
        meta = payload.get("meta", {})
        default_ckpt = f"{meta.get('backbone')}-seed{meta.get('seed')}"
        for r in payload["records"]:
            r.setdefault("ckpt_id", default_ckpt)
            records.append(r)
    return records


def per_cell_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tidy per-cell rows: {ckpt_id, contrast, arm, gen_seed, metrics:{...}} — the shape
    ``steering_stats`` consumes."""
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "ckpt_id": r["ckpt_id"],
                "contrast": r["contrast"],
                "arm": r["arm"],
                "gen_seed": r["gen_seed"],
                "swapped_field": r.get("swapped_field"),
                "stratum": r.get("stratum"),
                "hit_cap": r.get("hit_cap"),
                "self_terminated": r.get("self_terminated"),
                "metrics": decode_record_metrics(r["tokens"]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shards",
        default="reports/_steering/steering-*.json",
        help="glob for the gen shard JSONs",
    )
    ap.add_argument(
        "--out",
        default="reports/_steering/steering_per_cell_metrics.json",
        help="where to write the per-cell metrics JSON",
    )
    args = ap.parse_args()

    records = load_shards(args.shards)
    rows = per_cell_metrics(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"spec": "2026-07-17-steering-probe", "rows": rows}, indent=2))
    print(f"[analyze] wrote {len(rows)} per-cell metric rows -> {out}", flush=True)

    # Verdict computation is ORCHESTRATOR-owned (cfm.eval.steering_stats). Call it ONLY if it
    # exists at runtime; never author or inline the scoring core here.
    try:
        from cfm.eval import steering_stats
    except ImportError:
        print(
            "[analyze] cfm.eval.steering_stats not present — per-cell metrics written; "
            "verdict computation (paired deltas / sign test / effect size / verdict rule) is "
            "orchestrator-owned and not run here.",
            flush=True,
        )
        return

    verdict = steering_stats.judge(rows)
    verdict_path = out.with_name(out.stem + "_verdicts.json")
    verdict_path.write_text(json.dumps(verdict, indent=2))
    print(f"[analyze] steering_stats present -> wrote verdicts to {verdict_path}", flush=True)


if __name__ == "__main__":
    main()
