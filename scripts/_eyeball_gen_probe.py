"""Eyeball generation probe (PI-approved 2026-06-23) — NON-scored, methodology validation.

Loads ONE trained transformer-ar checkpoint and generates a handful of cells across
density-graded, HAND-BUILT conditioning contexts (dense / medium / sparse). The point is
to eyeball whether the tokenization+training methodology produces VALID, COHERENT geometry
that responds DIRECTIONALLY to conditioning — NOT to score, crown, or compute any KS/excess.

NO scoring, NO floor, NO manifest/sampler harness, NO decide(). Just: load -> build prefix
by hand -> generate tokens -> dump JSON. Decode + render happen locally off the dumped tokens.

Contexts are built from REAL in-distribution stratum tuples that exist in the locked floor
(zoning=1, coastal=2 held FIXED; density 3->2->0 and road_skeleton 2->1->0 stepped together),
paired with char_stats from the REAL character_stats_for_cell transform on many-small vs
few-large footprint lists. city_identity/region = a real TRAINING city (berlin); held-out
cities (glasgow/eisenhuttenstadt/krakow/munich) are deliberately avoided.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from cfm.data.training.build_shards import character_stats_for_cell
from cfm.data.training.conditioning import build_value_bearing_prefix
from cfm.inference.generate import generate_cell_tokens
from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig

# datamodule.py:74 — the char placeholder is PAD_ID == 0 (the 10th prefix position whose
# embedding is overwritten by the char_stats projection). Hardcoded to avoid importing the
# whole datamodule (which pulls shard-build guards we don't need here).
CHARACTER_PLACEHOLDER_ID = 0

# A real TRAINING city (configs/training/city_identity_registry.yaml) — NOT a held-out city.
FIXED_CITY = "berlin"

# Density-graded contexts. stratum = (zoning, skeleton, density, coastal); zoning=1 & coastal=2
# held fixed so the only varying signal is the density/skeleton gradient + char_stats.
CONTEXTS = [
    {
        "name": "dense_urban",
        "pop_density": 3,
        "zoning": 1,
        "skeleton": 2,  # densest road skeleton present in held-out strata
        "cell_density": 3,
        "coastal": 2,
        # many small footprints + many short road segments
        "areas": [45, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 48, 54, 57,
                  63, 66, 75, 80, 49, 53, 59, 61, 67, 71, 46, 64],
        "lengths": [22, 25, 28, 30, 32, 35, 38, 40, 24, 27, 31, 34, 37, 42, 29, 33],
    },
    {
        "name": "medium_mixed",
        "pop_density": 2,
        "zoning": 1,
        "skeleton": 1,
        "cell_density": 2,
        "coastal": 2,
        "areas": [95, 110, 125, 140, 155, 170, 130, 145, 120, 160],
        "lengths": [55, 65, 75, 85, 95, 70, 80],
    },
    {
        "name": "sparse_suburban",
        "pop_density": 1,
        "zoning": 1,
        "skeleton": 0,  # minimal road skeleton
        "cell_density": 0,
        "coastal": 2,
        # few large footprints + few long roads
        "areas": [300, 360, 440],
        "lengths": [185, 250],
    },
]


def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    cfg = ScaffoldConfig(**hp)
    model = build_backbone(cfg.backbone, cfg)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch — missing={list(missing)} unexpected={list(unexpected)}; "
            "refusing to generate from a partially-loaded model."
        )
    model.eval().to(device)
    return model, {
        "backbone": cfg.backbone,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "global_step": int(ck.get("global_step", -1)),
        "train_set": cfg.train_set,
        "conditioning_scheme": cfg.conditioning_scheme,
        "conditioning_ablation": cfg.conditioning_ablation,
    }


def build_prefix(ctx: dict) -> tuple[list[int], list[float]]:
    ids9 = build_value_bearing_prefix(
        population_density_bucket=ctx["pop_density"],
        zoning_class=ctx["zoning"],
        road_skeleton_class=ctx["skeleton"],
        cell_density_bucket=ctx["cell_density"],
        region=FIXED_CITY,
        coastal_inland_river=ctx["coastal"],
        sub_c_morphology_class=None,
        seed=7,  # conditioning-field seed: NOT embedded (constant bucket 0)
        city_identity=FIXED_CITY,
        ablation="full",
    )
    prefix = [*ids9, CHARACTER_PLACEHOLDER_ID]  # 10-position Task-24b layout
    char_stats = list(character_stats_for_cell(ctx["areas"], ctx["lengths"]))
    return prefix, char_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--cells-per-context", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}", flush=True)
    model, meta = load_model(args.ckpt, device)
    print(f"[probe] loaded: {meta}", flush=True)

    records = []
    for ctx in CONTEXTS:
        prefix, char_stats = build_prefix(ctx)
        print(f"\n[probe] context={ctx['name']} stratum(zoning,skel,dens,coast)="
              f"({ctx['zoning']},{ctx['skeleton']},{ctx['cell_density']},{ctx['coastal']}) "
              f"char_stats={[round(x, 3) for x in char_stats]}", flush=True)
        lengths = []
        for i in range(args.cells_per_context):
            gen_seed = 1000 + i
            toks = generate_cell_tokens(
                model, prefix=prefix, max_new=args.max_new, seed=gen_seed, char_stats=char_stats
            )
            hit_cap = len(toks) >= args.max_new
            lengths.append(len(toks))
            n_cell_end = sum(1 for t in toks if t == 260)
            records.append({
                "context": ctx["name"],
                "stratum": [ctx["zoning"], ctx["skeleton"], ctx["cell_density"], ctx["coastal"]],
                "pop_density": ctx["pop_density"],
                "cell_index": i,
                "gen_seed": gen_seed,
                "prefix": prefix,
                "char_stats": char_stats,
                "tokens": toks,
                "n_tokens": len(toks),
                "hit_cap": hit_cap,
                "self_terminated": n_cell_end > 0 and not hit_cap,
            })
            print(f"    cell {i}: n_tokens={len(toks):5d} hit_cap={hit_cap} "
                  f"self_terminated={n_cell_end > 0 and not hit_cap}", flush=True)
        print(f"  [{ctx['name']}] lengths min/med/max = "
              f"{min(lengths)}/{int(statistics.median(lengths))}/{max(lengths)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "max_new": args.max_new, "records": records}))
    alln = [r["n_tokens"] for r in records]
    n_cap = sum(1 for r in records if r["hit_cap"])
    print(f"\n[probe] DONE: {len(records)} cells -> {out}", flush=True)
    print(f"[probe] token-length over ALL cells: min/med/max = "
          f"{min(alln)}/{int(statistics.median(alln))}/{max(alln)} (cap={args.max_new}, "
          f"hard_cap=13312); hit_cap={n_cap}/{len(records)}", flush=True)
    print("EYEBALL_GEN_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
