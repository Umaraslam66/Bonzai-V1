"""READ-ONLY diagnostic (NOT scored, NOT committed-as-result): is the macro channel
redundant with char_stats at the DATA level?

Model-free. Loads the held-out cache (8000 cells) and measures, with NO model:
  (1) per-position distinct macro values overall + WITHIN-city (which macro fields actually
      vary within a city — the only ones the within-city deranged donor can perturb);
  (2) macro -> char_stats predictability (Ridge R^2): how much of the continuous char vector
      is recoverable from the discrete macro ids. char_stats IS a geometry summary computed
      from the cell, so high R^2 == macro already implies the cell's size signature;
  (3) char_stats -> within-city-varying macro predictability (logistic accuracy vs base rate):
      does char already determine the macro buckets the gap shuffles? If yes, conditional on
      char, macro is near-determined -> a small conditional macro gap is EXPECTED from the data,
      not a model failure.

Answers cause (3) [data/information] and informs the (1)/(3) entanglement. Says nothing about
whether the MODEL uses macro — that's the model-side factorial.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path("data/_diag/heldout_cache.json")
N_MACRO = 9  # positions 0..8 are value-bearing; position 9 is the inert char placeholder


def main() -> None:
    cells = json.loads(CACHE.read_text())
    regions = sorted({c["region"] for c in cells})
    macro = np.array([c["own_prefix"][:N_MACRO] for c in cells], dtype=np.int64)  # [N,9]
    char = np.array([c["own_char"] for c in cells], dtype=np.float64)  # [N,7]
    city = np.array([c["region"] for c in cells])
    n = len(cells)
    print(f"n={n} cells, regions={regions}\n")

    # (1) per-position distinct values overall + within-city ----------------------------------
    print("=== (1) macro per-position distinct values (which fields vary) ===")
    print(f"{'pos':>3} {'overall':>8} " + " ".join(f"{r[:6]:>7}" for r in regions))
    varying_positions: list[int] = []
    for p in range(N_MACRO):
        overall = len(set(macro[:, p].tolist()))
        per_city = [len(set(macro[city == r, p].tolist())) for r in regions]
        flag = " <- varies within-city" if max(per_city) > 1 else ""
        if max(per_city) > 1:
            varying_positions.append(p)
        print(f"{p:>3} {overall:>8} " + " ".join(f"{v:>7}" for v in per_city) + flag)
    print(f"\nwithin-city-VARYING macro positions (what the gap can perturb): {varying_positions}")
    print(f"city-CONSTANT macro positions (never perturbed within-city): "
          f"{[p for p in range(N_MACRO) if p not in varying_positions]}\n")

    # (2) macro -> char_stats: how much of char is recoverable from macro (Ridge R^2) ----------
    # one-hot the macro ids (per position), ridge-regress each char channel, report R^2.
    print("=== (2) macro -> char_stats recoverability (Ridge R^2, 5-fold-ish holdout) ===")
    onehot = _onehot(macro)
    r2_overall = _ridge_r2(onehot, char)
    print("char channel R^2 (macro predicts char): "
          + " ".join(f"c{i}={v:+.2f}" for i, v in enumerate(r2_overall)))
    print(f"mean char-R^2 (all): {np.mean(r2_overall):+.3f}   "
          f"(high => char is largely implied by macro => redundant)\n")

    # within-city (remove city as a predictor: regress char on macro WITH city fixed effects,
    # report the EXTRA R^2 from the within-city-varying macro beyond city alone)
    city_oh = _onehot(_city_codes(city)[:, None])
    r2_city = _ridge_r2(city_oh, char)
    r2_full = _ridge_r2(np.concatenate([city_oh, onehot], axis=1), char)
    print("within-city: char-R^2 from CITY alone vs CITY+macro (the macro increment is what "
          "within-city conditioning could add):")
    print("  city-only  R^2: " + " ".join(f"c{i}={v:+.2f}" for i, v in enumerate(r2_city)))
    print("  city+macro R^2: " + " ".join(f"c{i}={v:+.2f}" for i, v in enumerate(r2_full)))
    incr = np.mean(r2_full) - np.mean(r2_city)
    print(f"  mean increment from macro beyond city: {incr:+.3f}  "
          f"(small => within-city macro adds little char-info)\n")

    # (3) char -> within-city-varying macro: does char already determine the shuffled buckets? -
    print("=== (3) char_stats -> within-city-varying macro (nearest-centroid acc vs base rate) ===")
    for p in varying_positions:
        acc, base = _predict_macro_from_char(char, macro[:, p], city)
        verdict = "char SUBSUMES" if acc - base > 0.15 else "weak"
        print(f"  pos{p}: within-city acc={acc:.2f}  base(majority)={base:.2f}  "
              f"lift={acc - base:+.2f}  [{verdict}]")
    print("\n(high lift => conditional on char, the shuffled macro bucket is near-determined => "
          "a small conditional macro gap is EXPECTED from the data.)")


def _onehot(cols: np.ndarray) -> np.ndarray:
    """One-hot each integer column independently, concatenated. cols: [N, K]."""
    blocks = []
    for k in range(cols.shape[1]):
        vals = np.unique(cols[:, k])
        idx = {v: i for i, v in enumerate(vals.tolist())}
        oh = np.zeros((cols.shape[0], len(vals)))
        for i, v in enumerate(cols[:, k].tolist()):
            oh[i, idx[v]] = 1.0
        blocks.append(oh)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((cols.shape[0], 0))


def _city_codes(city: np.ndarray) -> np.ndarray:
    order = {r: i for i, r in enumerate(sorted(set(city.tolist())))}
    return np.array([order[c] for c in city.tolist()], dtype=np.int64)


def _ridge_r2(X: np.ndarray, Y: np.ndarray, lam: float = 1.0, folds: int = 5) -> np.ndarray:
    """Per-column out-of-fold R^2 of ridge X->Y. Standardize Y per column."""
    n = X.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    fold = np.array_split(perm, folds)
    preds = np.zeros_like(Y)
    Xa = np.concatenate([X, np.ones((n, 1))], axis=1)  # bias
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        A = Xa[tr]
        reg = lam * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(A.T @ A + reg, A.T @ Y[tr])
        preds[te] = Xa[te] @ W
    ss_res = ((Y - preds) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def _predict_macro_from_char(char: np.ndarray, macro_col: np.ndarray, city: np.ndarray) -> tuple:
    """Within-city nearest-centroid classification of a macro bucket from char (out-of-fold-ish).
    Returns (within-city accuracy, within-city majority base rate)."""
    accs, bases, weights = [], [], []
    for r in sorted(set(city.tolist())):
        m = city == r
        y = macro_col[m]
        X = char[m]
        classes = np.unique(y)
        if len(classes) < 2:
            continue
        # nearest class-centroid on standardized char (simple, no extra deps)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        cent = {c: Xs[y == c].mean(0) for c in classes.tolist()}
        pred = np.array([min(cent, key=lambda c: ((xs - cent[c]) ** 2).sum()) for xs in Xs])
        acc = (pred == y).mean()
        base = Counter(y.tolist()).most_common(1)[0][1] / len(y)
        accs.append(acc); bases.append(base); weights.append(len(y))
    w = np.array(weights, dtype=float); w /= w.sum()
    return float(np.dot(w, accs)), float(np.dot(w, bases))


if __name__ == "__main__":
    sys.exit(main())
