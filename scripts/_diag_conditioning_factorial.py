"""READ-ONLY diagnostic (NOT scored): conditioning channel decomposition factorial.

WHY is the macro-only gap ~0.006 while the full gap ~0.45? Separate causes (1) char DOMINANCE /
explained-away vs (2) injection failure, on the checkpoints we already have. No training.

Per cell, teacher-forced NLL (reusing cell_nll == training masked-CE) under conditioning variants:
  base        = (own macro, own char)          -- lowest NLL
  noMACRO     = (CONST macro, own char)         -- removes ALL per-cell macro info, keeps char
  noCHAR      = (own macro, MEAN char)          -- removes per-cell char info, keeps macro
  noBOTH      = (CONST macro, MEAN char)        -- ~unconditional
  macro_cache = (donor macro, own char)         -- reproduce the headline within-city macro gap
  macro_maxd  = (maxdist donor macro, own char) -- within-city donor MAXIMALLY different (cause-1 artifact)
  macro_noCH  = (donor macro, MEAN char)        -- KEY: macro gap WITH char ablated

Reported deltas (mean over cells; gap>0 means that conditioning helps):
  char_marginal              = noCHAR  - base     (how much char helps, given macro)
  macro_marginal             = noMACRO - base     (how much macro helps, given char)  [expect tiny]
  macro_gap_cache            = macro_cache - base  (reproduce ~0.006)
  macro_gap_maxdist          = macro_maxd - base
  macro_gap_when_char_ablated= macro_noCH - noCHAR (THE TEST: does macro gap jump w/o char?)
  macro_marginal_no_char     = noBOTH  - noCHAR    (macro's value when char already absent)

Decision:
  macro_gap_when_char_ablated >> macro_gap_cache  -> char EXPLAINS AWAY macro (cause 1 dominance)
  macro_gap_when_char_ablated ~ 0                 -> model never uses macro (cause 2 injection)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from cfm.eval.standing.harness import load_model
from cfm.eval.standing.nll import cell_nll

N_MACRO = 9


def _const_macro(cells: list[dict]) -> list[int]:
    """Per-position mode of macro positions 0..8 over ALL cells; keep position 9 (inert) as own."""
    mode = []
    for p in range(N_MACRO):
        mode.append(Counter(c["own_prefix"][p] for c in cells).most_common(1)[0][0])
    return [*mode, 0]  # position 9 = CHARACTER_PLACEHOLDER_ID


def _mean_char(cells: list[dict]) -> list[float]:
    n = len(cells)
    k = len(cells[0]["own_char"])
    return [sum(c["own_char"][i] for c in cells) / n for i in range(k)]


def _maxdist_donor_prefix(cells: list[dict]) -> list[list[int]]:
    """For each cell, the within-city donor whose macro[0:9] is MAX Hamming distance away."""
    by_city: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        by_city.setdefault(c["region"], []).append(i)
    out: list[list[int]] = [None] * len(cells)  # type: ignore
    for _city, idxs in by_city.items():
        macros = [cells[i]["own_prefix"][:N_MACRO] for i in idxs]
        for a, ia in enumerate(idxs):
            best_d, best_j = -1, ia
            ma = macros[a]
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
        rng = random.Random(seed * 1000 + ci)  # deterministic, NOT hash()-based
        out.extend(pool if n_per_city >= len(pool) else [pool[i] for i in sorted(rng.sample(range(len(pool)), n_per_city))])
    return out


def _signtest_frac(deltas: list[float]) -> tuple[float, float]:
    pos = sum(1 for d in deltas if d > 0)
    return pos / len(deltas), float(sum(deltas) / len(deltas))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="reports/_standing_eval/heldout_cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-city", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_cells = json.loads(Path(args.cache).read_text())
    const_macro = _const_macro(all_cells)  # constants from the FULL set
    mean_char = _mean_char(all_cells)
    cells = _subsample(all_cells, args.n_per_city)
    maxd = {id(c): p for c, p in zip(cells, _maxdist_donor_prefix(cells))}
    print(f"cells={len(cells)} (of {len(all_cells)}) device={dev}")
    print(f"const_macro={const_macro}  mean_char={[round(x, 3) for x in mean_char]}")

    model, meta = load_model(args.ckpt, dev)
    ckpt_id = f"{meta['backbone']}-seed{meta['seed']}"

    # accumulate per-cell deltas
    d_char_marg, d_macro_marg = [], []
    d_macro_cache, d_macro_maxd, d_macro_noch, d_macro_marg_noch = [], [], [], []
    for c in cells:
        own_p, own_c = c["own_prefix"], c["own_char"]
        body = c["body_tokens"]
        f = lambda pre, ch: cell_nll(model, body_tokens=body, conditioning_prefix=pre, char_stats=ch, device=dev)
        base = f(own_p, own_c)
        noMACRO = f(const_macro, own_c)
        noCHAR = f(own_p, mean_char)
        noBOTH = f(const_macro, mean_char)
        macro_cache = f(c["donor_prefix"], own_c)
        macro_maxd = f(maxd[id(c)], own_c)
        macro_noCH = f(c["donor_prefix"], mean_char)
        d_char_marg.append(noCHAR - base)
        d_macro_marg.append(noMACRO - base)
        d_macro_cache.append(macro_cache - base)
        d_macro_maxd.append(macro_maxd - base)
        d_macro_noch.append(macro_noCH - noCHAR)
        d_macro_marg_noch.append(noBOTH - noCHAR)

    def summ(name, d):
        frac, mean = _signtest_frac(d)
        print(f"  {name:<28} mean={mean:+.4f}  frac_pos={frac:.2f}")
        return {"mean": mean, "frac_pos": frac, "n": len(d)}

    print(f"\n=== {ckpt_id} :: conditioning decomposition (nats/token) ===")
    res = {
        "ckpt_id": ckpt_id, "meta": meta, "n_cells": len(cells),
        "char_marginal": summ("char_marginal (noCHAR-base)", d_char_marg),
        "macro_marginal": summ("macro_marginal (noMACRO-base)", d_macro_marg),
        "macro_gap_cache": summ("macro_gap_cache", d_macro_cache),
        "macro_gap_maxdist": summ("macro_gap_maxdist", d_macro_maxd),
        "macro_gap_when_char_ablated": summ("macro_gap_when_char_ablated", d_macro_noch),
        "macro_marginal_no_char": summ("macro_marginal_no_char (noBOTH-noCHAR)", d_macro_marg_noch),
        "const_macro": const_macro, "mean_char": mean_char,
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
