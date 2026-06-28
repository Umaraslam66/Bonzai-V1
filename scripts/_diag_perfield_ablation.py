"""READ-ONLY diagnostic (NOT scored): per-macro-FIELD ablation.

Which macro field carries the within-city macro effect (~0.018, see _diag_conditioning_factorial)?
For v2: tells which conditioning fields are worth keeping / strengthening.

Per cell, teacher-forced NLL (cell_nll == training masked-CE). Baseline = (own macro, own char).
For each within-city-VARYING macro position p (derived from the data), swap ONLY position p:
  swap_donor_p  = own prefix with pos p <- random within-city donor's value (decomposes the ~0.007 cache gap)
  swap_maxd_p   = own prefix with pos p <- MAXDIST within-city donor's value (decomposes the ~0.018 gap)
char_stats held at own throughout (matches the macro-gap regime). The field with the largest
single-field gap is the carrier.

Field order (src/cfm/data/training/conditioning.py build_value_bearing_prefix), positions 0..8:
  0 pop_density  1 zoning  2 road_skeleton  3 cell_density  4 region
  5 coastal  6 sub_c_morph(inert)  7 seed(inert)  8 city_identity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cfm.eval.standing.harness import load_model
from cfm.eval.standing.nll import cell_nll

N_MACRO = 9
FIELD = {
    0: "pop_density", 1: "zoning", 2: "road_skeleton", 3: "cell_density", 4: "region",
    5: "coastal", 6: "sub_c_morph", 7: "seed_inert", 8: "city_identity",
}


def _varying_positions(cells: list[dict]) -> list[int]:
    """Positions that take >1 value WITHIN at least one city (the only ones a within-city swap moves)."""
    by_city: dict[str, list[dict]] = {}
    for c in cells:
        by_city.setdefault(c["region"], []).append(c)
    varying = []
    for p in range(N_MACRO):
        if any(len({c["own_prefix"][p] for c in pool}) > 1 for pool in by_city.values()):
            varying.append(p)
    return varying


def _maxdist_donor_prefix(cells: list[dict]) -> list[list[int]]:
    by_city: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        by_city.setdefault(c["region"], []).append(i)
    out: list[list[int]] = [None] * len(cells)  # type: ignore
    for idxs in by_city.values():
        macros = [cells[i]["own_prefix"][:N_MACRO] for i in idxs]
        for a, ia in enumerate(idxs):
            best_d, best_j, ma = -1, ia, macros[a]
            for b, ib in enumerate(idxs):
                if ib == ia:
                    continue
                d = sum(1 for p in range(N_MACRO) if ma[p] != macros[b][p])
                if d > best_d:
                    best_d, best_j = d, ib
            out[ia] = list(cells[best_j]["own_prefix"])
    return out


def _subsample(cells: list[dict], n_per_city: int | None, seed: int = 7) -> list[dict]:
    if n_per_city is None:
        return cells
    import random

    by_city: dict[str, list[dict]] = {}
    for c in cells:
        by_city.setdefault(c["region"], []).append(c)
    out = []
    for ci, city in enumerate(sorted(by_city)):
        pool = by_city[city]
        rng = random.Random(seed * 1000 + ci)
        out.extend(pool if n_per_city >= len(pool)
                   else [pool[i] for i in sorted(rng.sample(range(len(pool)), n_per_city))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="reports/_standing_eval/heldout_cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-city", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cells = _subsample(json.loads(Path(args.cache).read_text()), args.n_per_city)
    varying = _varying_positions(cells)
    maxd = {id(c): p for c, p in zip(cells, _maxdist_donor_prefix(cells))}
    print(f"cells={len(cells)} device={dev}")
    print(f"within-city-varying positions: {[(p, FIELD[p]) for p in varying]}")

    model, meta = load_model(args.ckpt, dev)
    ckpt_id = f"{meta['backbone']}-seed{meta['seed']}"

    def nll(pre, ch):
        return cell_nll(model, body_tokens=pre[1], conditioning_prefix=pre[0], char_stats=ch, device=dev)

    # accumulators
    g_donor = {p: [] for p in varying}
    g_maxd = {p: [] for p in varying}
    joint_donor, joint_maxd = [], []
    for c in cells:
        own, ch, body = c["own_prefix"], c["own_char"], c["body_tokens"]
        base = nll((own, body), ch)
        joint_donor.append(nll((c["donor_prefix"], body), ch) - base)
        joint_maxd.append(nll((maxd[id(c)], body), ch) - base)
        for p in varying:
            pd = list(own); pd[p] = c["donor_prefix"][p]
            g_donor[p].append(nll((pd, body), ch) - base)
            pm = list(own); pm[p] = maxd[id(c)][p]
            g_maxd[p].append(nll((pm, body), ch) - base)

    def stat(d):
        n = len(d)
        return {"mean": sum(d) / n, "frac_pos": sum(1 for x in d if x > 0) / n,
                "frac_changed": sum(1 for x in d if x != 0) / n, "n": n}

    print(f"\n=== {ckpt_id} :: per-field single-swap gap (nats/token, char held own) ===")
    print(f"{'field':>14} {'donor mean':>11} {'fracΔ':>6} {'maxdist mean':>13} {'fracΔ':>6}")
    res_fields = {}
    for p in varying:
        sd, sm = stat(g_donor[p]), stat(g_maxd[p])
        res_fields[FIELD[p]] = {"pos": p, "swap_donor": sd, "swap_maxdist": sm}
        print(f"{FIELD[p]:>14} {sd['mean']:>+11.4f} {sd['frac_changed']:>6.2f} "
              f"{sm['mean']:>+13.4f} {sm['frac_changed']:>6.2f}")
    jd, jm = stat(joint_donor), stat(joint_maxd)
    print(f"{'JOINT(all)':>14} {jd['mean']:>+11.4f} {jd['frac_changed']:>6.2f} "
          f"{jm['mean']:>+13.4f} {jm['frac_changed']:>6.2f}")
    print(f"\nsum-of-per-field maxdist = {sum(res_fields[f]['swap_maxdist']['mean'] for f in res_fields):+.4f} "
          f"vs JOINT maxdist {jm['mean']:+.4f} (additivity check)")

    res = {"ckpt_id": ckpt_id, "meta": meta, "n_cells": len(cells),
           "varying": [[p, FIELD[p]] for p in varying], "fields": res_fields,
           "joint_donor": jd, "joint_maxdist": jm}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
