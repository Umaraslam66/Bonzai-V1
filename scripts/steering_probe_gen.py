"""Generation-steering probe (SPEC 2026-07-17) — NON-scored methodology experiment.

Answers the open caveat of the 2026-06-26 macro diagnostic: NLL sensitivity != generation
steering. Builds 5 pre-registered CONTRASTS (C1..C5), each TWO arms differing in EXACTLY the
named conditioning field(s), with 40 PAIRED generation seeds (2000..2039) shared across the two
arms of a contrast, and generates cells for one checkpoint. Emits tokens ONLY; decode +
per-cell metrics + verdict happen off-GPU (``steering_probe_analyze.py`` -> ``steering_stats``).

Arm construction (``build_arms``) is PURE and unit-tested (arms differ ONLY at the swapped
prefix position(s); paired seeds identical) so the load-bearing "everything else identical"
guarantee is proven without a GPU.

Base context (spec table): city/region=berlin (a TRAINING city; held-out cities avoided),
zoning=1, coastal=2, pop_density=2, cell_density=2, road_skeleton=1, seed-field=7.
char_fixed = character_stats_for_cell of the eyeball ``medium_mixed`` areas/lengths.
char_mean  = the factorial diagnostic's dataset-mean char convention (over the held-out cache).

| id | swapped field(s)      | arm A -> arm B          | char regime          |
|----|-----------------------|-------------------------|----------------------|
| C1 | road_skeleton         | 0 -> 2                  | char_fixed           |
| C2 | cell_density          | 0 -> 3                  | char_fixed           |
| C3 | joint (pop,skel,dens) | (1,0,0) -> (3,2,3)      | char_fixed           |
| C4 | char_stats ONLY       | sparse chars -> dense   | macro fixed at base  |
| C5 | road_skeleton         | 0 -> 2                  | char_mean (ablated)  |
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from cfm.data.training.build_shards import character_stats_for_cell
from cfm.data.training.conditioning import build_value_bearing_prefix

# datamodule.py:74 — the char placeholder is PAD_ID == 0 (the 10th prefix position whose
# embedding is overwritten by the char_stats projection). Hardcoded to avoid importing the
# whole datamodule (mirrors _eyeball_gen_probe.CHARACTER_PLACEHOLDER_ID).
CHARACTER_PLACEHOLDER_ID = 0

# A real TRAINING city (configs/training/city_identity_registry.yaml) — NOT a held-out city;
# same fixed city the eyeball probe used.
FIXED_CITY = "berlin"
# Conditioning-field seed (spec base context): NOT value-embedded (constant bucket 0).
SEED_FIELD = 7

# Paired generation seeds (spec §Design): 40 seeds shared across both arms of every contrast.
GEN_SEEDS: tuple[int, ...] = tuple(range(2000, 2040))

# Base macro context (the fields NOT swapped by a given contrast stay at these values).
BASE_POP_DENSITY = 2
BASE_ZONING = 1
BASE_SKELETON = 1
BASE_CELL_DENSITY = 2
BASE_COASTAL = 2

_REPO = Path(__file__).resolve().parents[1]
_EYEBALL_PROBE = _REPO / "scripts" / "_eyeball_gen_probe.py"


def _load_eyeball_contexts() -> dict[str, dict]:
    """Load ``_eyeball_gen_probe.CONTEXTS`` by REFERENCE (spec: C4/char_fixed sample its
    areas/lengths) via importlib — the same file-load route the standing harness uses — so the
    char vectors are never a drifting copy of the eyeball probe's."""
    spec = importlib.util.spec_from_file_location("_eyeball_gen_probe", _EYEBALL_PROBE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load eyeball probe module at {_EYEBALL_PROBE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {c["name"]: c for c in mod.CONTEXTS}


def _char_for(context_name: str) -> list[float]:
    """char_stats via the REAL character_stats_for_cell transform on an eyeball context's
    (areas, lengths) — the one source the eyeball/geometry-validity lanes use."""
    ctx = _load_eyeball_contexts()[context_name]
    return list(character_stats_for_cell(ctx["areas"], ctx["lengths"]))


def _prefix(
    *,
    pop_density: int,
    zoning: int,
    skeleton: int,
    cell_density: int,
    coastal: int,
) -> list[int]:
    """10-position Task-24b prefix (9 value-bearing conditioning ids + char placeholder slot),
    built EXACTLY like ``_eyeball_gen_probe.build_prefix`` (ablation='full')."""
    ids9 = build_value_bearing_prefix(
        population_density_bucket=pop_density,
        zoning_class=zoning,
        road_skeleton_class=skeleton,
        cell_density_bucket=cell_density,
        region=FIXED_CITY,
        coastal_inland_river=coastal,
        sub_c_morphology_class=None,
        seed=SEED_FIELD,
        city_identity=FIXED_CITY,
        ablation="full",
    )
    return [*ids9, CHARACTER_PLACEHOLDER_ID]


def build_arms(ckpt_id: str, *, char_mean: list[float]) -> list[dict[str, Any]]:
    """PURE construction of the full flattened work-item list for ONE checkpoint.

    5 contrasts x 2 arms x 40 paired seeds = 400 work items. Each item::

        {ckpt_id, contrast, arm ('A'|'B'), gen_seed, prefix (10 ids), char_stats (list[float]),
         stratum (zoning, skeleton, cell_density, coastal), swapped_field}

    ``char_mean`` (spec: the factorial diagnostic's dataset-mean char) is injected by the caller
    (``--mean-char-json`` or computed from the held-out cache); every other char vector is
    derived here from the REAL transform so the whole list is reproducible from ``ckpt_id`` +
    ``char_mean`` alone.
    """
    char_fixed = _char_for("medium_mixed")
    char_dense = _char_for("dense_urban")
    char_sparse = _char_for("sparse_suburban")

    # Each contrast: two arms as (macro-override dict, char_stats). ``stratum`` = the floor's
    # (zoning, skeleton, cell_density, coastal) 4-tuple for off-manifold annotation downstream.
    def base_macro(**over: int) -> dict[str, int]:
        macro = {
            "pop_density": BASE_POP_DENSITY,
            "zoning": BASE_ZONING,
            "skeleton": BASE_SKELETON,
            "cell_density": BASE_CELL_DENSITY,
            "coastal": BASE_COASTAL,
        }
        macro.update(over)
        return macro

    # (contrast, swapped_field, arm_A(macro,char), arm_B(macro,char))
    contrasts: list[tuple[str, str, tuple[dict, list[float]], tuple[dict, list[float]]]] = [
        (
            "C1",
            "road_skeleton",
            (base_macro(skeleton=0), char_fixed),
            (base_macro(skeleton=2), char_fixed),
        ),
        (
            "C2",
            "cell_density",
            (base_macro(cell_density=0), char_fixed),
            (base_macro(cell_density=3), char_fixed),
        ),
        (
            "C3",
            "pop+skeleton+cell_density",
            (base_macro(pop_density=1, skeleton=0, cell_density=0), char_fixed),
            (base_macro(pop_density=3, skeleton=2, cell_density=3), char_fixed),
        ),
        (
            "C4",
            "char_stats",
            (base_macro(), char_sparse),
            (base_macro(), char_dense),
        ),
        (
            "C5",
            "road_skeleton",
            (base_macro(skeleton=0), char_mean),
            (base_macro(skeleton=2), char_mean),
        ),
    ]

    items: list[dict[str, Any]] = []
    for contrast, swapped_field, (macro_a, char_a), (macro_b, char_b) in contrasts:
        for arm, macro, char_stats in (("A", macro_a, char_a), ("B", macro_b, char_b)):
            prefix = _prefix(
                pop_density=macro["pop_density"],
                zoning=macro["zoning"],
                skeleton=macro["skeleton"],
                cell_density=macro["cell_density"],
                coastal=macro["coastal"],
            )
            stratum = [macro["zoning"], macro["skeleton"], macro["cell_density"], macro["coastal"]]
            for gen_seed in GEN_SEEDS:
                items.append(
                    {
                        "ckpt_id": ckpt_id,
                        "contrast": contrast,
                        "arm": arm,
                        "swapped_field": swapped_field,
                        "gen_seed": gen_seed,
                        "prefix": list(prefix),
                        "char_stats": list(char_stats),
                        "stratum": stratum,
                    }
                )
    return items


def _sort_key(item: dict[str, Any]) -> tuple[str, str, str, int]:
    return (item["ckpt_id"], item["contrast"], item["arm"], int(item["gen_seed"]))


def shard_items(items: list[dict[str, Any]], k: int, n: int) -> list[dict[str, Any]]:
    """Deterministically partition ``items`` into shard ``k`` of ``n`` by stable-sorted index
    modulo ``n``. Union of shards 0..n-1 reproduces the full list exactly; shards are disjoint."""
    if not 0 <= k < n:
        raise ValueError(f"shard index {k} out of range for {n} shards")
    ordered = sorted(items, key=_sort_key)
    return [it for i, it in enumerate(ordered) if i % n == k]


def _mean_char_from_cache(cache_path: Path) -> list[float]:
    """Dataset-mean char over the held-out cache — the SAME formula as
    ``_diag_conditioning_factorial._mean_char`` (mean of each ``own_char`` component)."""
    cells = json.loads(cache_path.read_text())
    if not cells:
        raise ValueError(f"held-out cache {cache_path} is empty")
    n = len(cells)
    dim = len(cells[0]["own_char"])
    return [sum(c["own_char"][i] for c in cells) / n for i in range(dim)]


def _parse_shard(spec: str) -> tuple[int, int]:
    k_str, _, n_str = spec.partition("/")
    if not n_str:
        raise argparse.ArgumentTypeError(f"--shard must be K/N, got {spec!r}")
    return int(k_str), int(n_str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--shard", type=_parse_shard, default=(0, 1), help="K/N (e.g. 0/4)")
    ap.add_argument(
        "--mean-char-json",
        default=None,
        help="path to a JSON list of char_mean; overrides --cache",
    )
    ap.add_argument("--cache", default="reports/_standing_eval/heldout_cache.json")
    args = ap.parse_args()

    # Torch-touching imports are LAZY so ``build_arms``/tests never pull GPU/model machinery.
    import torch

    from cfm.eval.standing.harness import load_model
    from cfm.inference.generate import generate_cell_tokens

    k, n = args.shard
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[steering] device={device} shard={k}/{n}", flush=True)

    if args.mean_char_json is not None:
        char_mean = list(json.loads(Path(args.mean_char_json).read_text()))
        print(f"[steering] char_mean from {args.mean_char_json}", flush=True)
    else:
        char_mean = _mean_char_from_cache(Path(args.cache))
        print(f"[steering] char_mean from cache {args.cache}", flush=True)
    print(f"[steering] char_mean={[round(x, 3) for x in char_mean]}", flush=True)

    model, meta = load_model(args.ckpt, device)
    ckpt_id = f"{meta['backbone']}-seed{meta['seed']}"
    print(f"[steering] loaded {ckpt_id}: {meta}", flush=True)

    full = build_arms(ckpt_id, char_mean=char_mean)
    work = shard_items(full, k, n)
    print(f"[steering] {len(work)}/{len(full)} items in this shard", flush=True)

    records: list[dict[str, Any]] = []
    for j, item in enumerate(work):
        toks = generate_cell_tokens(
            model,
            prefix=item["prefix"],
            max_new=args.max_new,
            seed=item["gen_seed"],
            char_stats=item["char_stats"],
        )
        hit_cap = len(toks) >= args.max_new
        self_terminated = (260 in toks) and not hit_cap
        records.append(
            {
                **item,
                "tokens": toks,
                "n_tokens": len(toks),
                "hit_cap": hit_cap,
                "self_terminated": self_terminated,
            }
        )
        if j % 20 == 0 or j == len(work) - 1:
            print(
                f"    [{j + 1:3d}/{len(work)}] {item['contrast']}-{item['arm']} "
                f"seed={item['gen_seed']} n_tokens={len(toks)} hit_cap={hit_cap} "
                f"self_terminated={self_terminated}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "spec": "2026-07-17-steering-probe",
        "shard": {"k": k, "n": n},
        "max_new": args.max_new,
        "char_mean": char_mean,
        "records": records,
    }
    out.write_text(json.dumps(payload))

    # NO-marker-without-endstate-verification: re-read + parse the file BEFORE the sentinel.
    reloaded = json.loads(out.read_text())
    assert len(reloaded["records"]) == len(records), "readback record count mismatch"
    n_cap = sum(1 for r in records if r["hit_cap"])
    n_self = sum(1 for r in records if r["self_terminated"])
    print(
        f"\n[steering] DONE: {len(records)} cells -> {out} "
        f"(hit_cap={n_cap}, self_terminated={n_self})",
        flush=True,
    )
    print("STEERING_PROBE_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
