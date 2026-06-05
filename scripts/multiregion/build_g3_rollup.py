#!/usr/bin/env python
"""Phase-G G3: roll-up + advisory redundancy proxy + cost-model for the canary corpus.

Run ON Leonardo (the validated sub_f artifacts + the Singapore reference corpus
live there). Assembles the §6 roll-up (one CityRecord per validated canary city
with axis labels read from the ratified canary manifest — NOT re-guessed), runs
the §7 ADVISORY proxy (geometry-token redundancy vs the Singapore corpus; the
language r=20 anchor is recorded as DEFERRED because no language-token corpus is
pinned and the spec forbids inventing one — it is advisory and does not gate the
budget anyway), and prices a per-morphology cost-model. Writes a reproducible YAML
to reports/. Read-only over data/; writes only the report.

Carry-ins (PI, 2026-06-04):
  - tile counts on clipped fallback bboxes are LOWER BOUNDS, not morphology floors
    (barcelona dense-core landed below moderate munich) -> bias batch-2 sizing UP.
  - lrd_all_serial is 8-CPU serial-per-user; batch-2 throughput must weigh a
    parallel non-boost CPU partition (dcgp_usr_prod, billed) vs free-but-serial.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.multiregion import proxy, rollup, selection  # noqa: E402

RELEASE = "2026-04-15.0"
BASE_TILE_BUDGET = 20_600  # spec §7: 30M ceiling x r=20 = 600M tokens / 29,150 tok/tile
MEASURED_TOKENS_PER_TILE = 29_150  # spec §7 (Singapore-measured); EU re-checked below
TARGET_TOKENS = 600_000_000  # 30M-ceiling need at r=20

PROC = _REPO / "data" / "processed"
PREFETCH_LOG = _REPO / "logs" / "g2-prefetch.log"


def _city_token_stats(region: str) -> tuple[int, int, bytes]:
    """(tile_count, token_count, concatenated int16 token bytes) from sub_f cells."""
    city_dir = PROC / "sub_f" / RELEASE / region
    tile_files = sorted(glob.glob(str(city_dir / "tile=*" / "cells.parquet")))
    n_tokens = 0
    chunks: list[bytes] = []
    for f in tile_files:
        # Read the FILE directly (not the partitioned dir) to avoid pyarrow Hive
        # 'tile=' column inference. Only the token_sequence column is needed.
        tbl = pq.ParquetFile(f).read(columns=["token_sequence"])
        flat = tbl.column("token_sequence").combine_chunks().flatten()
        arr = flat.to_numpy(zero_copy_only=False).astype("<i2")
        n_tokens += int(arr.size)
        chunks.append(arr.tobytes())
    return len(tile_files), n_tokens, b"".join(chunks)


def _fetch_seconds() -> dict[str, float]:
    out: dict[str, float] = {}
    if not PREFETCH_LOG.exists():
        return out
    txt = PREFETCH_LOG.read_text(errors="ignore").replace("\r", "\n")
    for m in re.finditer(r"\[(\w+)\] OK rc=0 elapsed=(\d+)s", txt):
        out[m.group(1)] = float(m.group(2))
    return out


def _stage_shas(region: str) -> dict[str, str]:
    import json

    p = PROC / "multiregion" / "state" / f"{region}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {k: v["sha"][:12] for k, v in d.get("completions", {}).items()}


def main() -> int:
    manifest = selection.load_canary_manifest(_REPO / "configs" / "multiregion" / "canary_v1.yaml")
    fetch = _fetch_seconds()

    records: list[rollup.CityRecord] = []
    eu_chunks: list[bytes] = []
    per_city: list[dict] = []
    for c in manifest:
        name = c["name"]
        tiles, tokens, tbytes = _city_token_stats(name)
        eu_chunks.append(tbytes)
        validated = (PROC / "sub_g" / RELEASE / name / "_PHASE1_VALIDATED").exists()
        rec = rollup.CityRecord(
            name=name,
            morphology=c["morphology"],
            density=c["density"],
            geography=c["geography"],
            region_crs=c["projected_crs"],
            tile_count=tiles,
            fetch_seconds=fetch.get(name, 0.0),
            stage_shas=_stage_shas(name),
            release=RELEASE,
            validation_status=rollup.VALIDATED if validated else rollup.FAILED,
            token_count=tokens,
        )
        records.append(rec)
        per_city.append(
            {
                "name": name,
                "morphology": c["morphology"],
                "density": c["density"],
                "tiles": tiles,
                "tokens": tokens,
                "tokens_per_tile": round(tokens / tiles, 1) if tiles else None,
                "fetch_seconds": fetch.get(name, 0.0),
            }
        )

    r = rollup.RollUp(cities=records)

    # --- structural gate (raises if a failure or an uncovered axis label) ---
    gate_ok = True
    gate_err = ""
    try:
        rollup.assert_ready_for_next_batch(r)
    except RuntimeError as exc:
        gate_ok = False
        gate_err = str(exc)

    # --- §7 advisory proxy: geometry redundancy vs Singapore (language = deferred) ---
    eu_bytes = b"".join(eu_chunks)
    geom_red = proxy.compression_redundancy(eu_bytes)
    _, _, sg_bytes = _city_token_stats("singapore")
    sg_red = proxy.compression_redundancy(sg_bytes) if sg_bytes else None
    rel_sg = (geom_red - sg_red) / sg_red if sg_red else None
    proxy_block = {
        "geometry_redundancy": round(geom_red, 6),
        "singapore_redundancy": round(sg_red, 6) if sg_red is not None else None,
        "rel_singapore": round(rel_sg, 6) if rel_sg is not None else None,
        "language_baseline": None,
        "verdict": "language_anchor_DEFERRED (no language-token corpus pinned; "
        "spec §7 forbids inventing one; advisory only, does not gate budget)",
        "base_tile_budget": BASE_TILE_BUDGET,
        "recommended_tile_budget": round(BASE_TILE_BUDGET * (1.0 + proxy.Y_SIZE_UP)),
        "r_unresolved_until_bakeoff": True,  # ALWAYS set — bake-off is sole r authority
    }

    # --- cost-model (per-morphology; carry-ins applied) ---
    validated_tiles = rollup.total_validated_tiles(r)
    validated_tokens = rollup.total_validated_tokens(r)
    tiles_by_city = {rec.name: rec.tile_count for rec in records}
    by_morph: dict[str, dict] = {}
    for rec in records:
        b = by_morph.setdefault(
            rec.morphology, {"cities": [], "tiles": 0, "tokens": 0, "fetch_s": 0.0}
        )
        b["cities"].append(rec.name)
        b["tiles"] += rec.tile_count
        b["tokens"] += rec.token_count
        b["fetch_s"] += rec.fetch_seconds
    cost_model = {
        "validated_tiles": validated_tiles,
        "validated_tokens": validated_tokens,
        "eu_tokens_per_tile_actual": round(validated_tokens / validated_tiles, 1)
        if validated_tiles
        else None,
        "singapore_measured_tokens_per_tile": MEASURED_TOKENS_PER_TILE,
        "by_morphology": {
            k: {**v, "tokens_per_tile": round(v["tokens"] / v["tiles"], 1) if v["tiles"] else None}
            for k, v in by_morph.items()
        },
        "batch2_tiles_to_base": max(0, BASE_TILE_BUDGET - validated_tiles),
        "batch2_tiles_to_sized_up": max(
            0, proxy_block["recommended_tile_budget"] - validated_tiles
        ),
        "target_tokens": TARGET_TOKENS,
        "caveats": [
            f"Tile counts are LOWER BOUNDS (clipped fallback bboxes): barcelona "
            f"dense-core={tiles_by_city.get('barcelona')} landed BELOW moderate "
            f"munich={tiles_by_city.get('munich')} -> do NOT treat low counts as "
            f"morphology floors. Bias batch-2 UP; barcelona is an unreliable point.",
            "lrd_all_serial QoS = cpu=8 per USER -> per-city jobs run SERIAL (one at "
            "a time). A 44-city batch-2 fanout there ~= sum of per-city walls. Weigh "
            "dcgp_usr_prod (112-core, parallel, BILLS core-hours) vs free-but-serial.",
        ],
    }

    out = _REPO / "reports" / "2026-06-04-phase-2-g3-canary-rollup.yaml"
    rollup.write_rollup(
        r,
        out,
        extra={
            "gate_ready_for_next_batch": gate_ok,
            "gate_error": gate_err,
            "proxy": proxy_block,
            "cost_model": cost_model,
            "baseline_head_sha": "b98a20b",  # composition-valid HEAD (stages ran at 5bdcf05)
        },
    )

    # --- human summary ---
    print("=== G3 ROLL-UP (canary) ===")
    for pc in per_city:
        print(
            f"  {pc['name']:<14} {pc['morphology']:<17} {pc['density']:<10} "
            f"tiles={pc['tiles']:>4} tokens={pc['tokens']:>9} "
            f"tok/tile={pc['tokens_per_tile']} fetch={pc['fetch_seconds']:.0f}s"
        )
    n_val = sum(1 for x in records if x.validation_status == rollup.VALIDATED)
    print(f"\n  validated_cities={n_val}/5 tiles={validated_tiles} tokens={validated_tokens:,}")
    print(f"  axis_coverage={rollup.axis_coverage(r)}")
    print(f"  uncovered={rollup.uncovered_axis_labels(r)}")
    print(f"  GATE ready_for_next_batch={gate_ok} {gate_err}")
    print("\n=== §7 PROXY (advisory) ===")
    print(
        f"  geometry_redundancy={geom_red:.4f} singapore={sg_red and round(sg_red, 4)} "
        f"rel_sg={rel_sg and round(rel_sg, 4)}"
    )
    print(
        f"  language_baseline=DEFERRED  rec_tile_budget="
        f"{proxy_block['recommended_tile_budget']} (base {BASE_TILE_BUDGET} x1.5)  "
        f"r_unresolved=True"
    )
    print("\n=== COST MODEL ===")
    print(
        f"  EU actual tok/tile={cost_model['eu_tokens_per_tile_actual']} "
        f"(Singapore-measured {MEASURED_TOKENS_PER_TILE})"
    )
    print(
        f"  batch2 tiles to base={cost_model['batch2_tiles_to_base']} "
        f"to sized-up={cost_model['batch2_tiles_to_sized_up']}"
    )
    print(f"\n  wrote {out}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
