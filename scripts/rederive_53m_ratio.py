"""Re-derive the Phase-2 bake-off pair under a clean Jamba 1:7 ratio constraint (2026-06-18).

LOCKED RESULT (the "53M" rung in src/cfm/models/bakeoff_scales.py): shared d_model=512;
transformer-ar 14L/8H = 52,798,948; mamba-hybrid 24L/e7 = 53,733,348 (21 mamba + 3 tf, tf at
layers 8/16/24 = a clean 1:7); delta 1.77% <= 2%. It targets ~50M but a clean 1:7 within 2% is
only reachable at ~53M, so the rung is labelled 53M (NOT 50M).

The param-match-only search picked mamba d640/14L/e7 = 1 tf + 13 mamba = 13:1 (a long
trailing pure-mamba tail past the single 7:1 block) — sparser attention than Jamba's
validated ~1:7. This search instead requires a CLEAN ratio: n_layers a multiple of
(every+1) so there is no trailing partial block and the tf:mamba ratio is exact, in the
1:3..1:7 zone (prioritising 1:7). Both backbones share d_model; the transformer-ar partner
is param-matched (<=2%) to the mamba at that same d_model, targeting ~50M.

The analytic per-layer/shared formulas are EXACT here (verified: tf d640/8L=50,219,748 and
mamba d640/14L/e7=49,966,948 reproduce to the digit), so the search is analytic and the top
candidates are then ACTUALLY built to confirm count == analytic (never trust analytic to lock).
os._exit(0) dodges the GPU-less torch/mamba teardown hang. Writes NOTHING to bakeoff_scales.py.
"""

from __future__ import annotations

import os

from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CHARACTER_STAT_CHANNELS,
    CONDITIONING_PREFIX_LEN,
    conditioning_id_span,
)
from cfm.models.backbone import subf_vocab_size
from cfm.models.mamba_hybrid import MambaHybrid, MambaHybridConfig, _interleave_positions
from cfm.models.micro_ar import MicroAR, MicroARConfig

N_VOCAB = subf_vocab_size()
N_COND = conditioning_id_span()
MAX_LEN = 13_312 + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS
N_CHAR = CHARACTER_STAT_CHANNELS
CHAR_POS = CONDITIONING_PREFIX_LEN
SHARED = dict(
    n_subf_vocab=N_VOCAB,
    n_cond=N_COND,
    max_len=MAX_LEN,
    n_char_stats=N_CHAR,
    char_position=CHAR_POS,
)
TARGET = 50_000_000
TOL = 0.02


def _params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def tf_build(d: int, n: int, h: int) -> int:
    return _params(MicroAR(MicroARConfig(d_model=d, n_layers=n, n_heads=h, **SHARED)))


def mamba_build(d: int, n: int, h: int, e: int) -> int:
    cfg = MambaHybridConfig(d_model=d, n_layers=n, n_heads=h, transformer_every=e, **SHARED)
    return _params(MambaHybrid(cfg))


# EXACT analytic forms (calibrated; confirmed to the digit at d640).
def tf_layer(d: int) -> float:
    return 12.0 * d * d + 13 * d


def mamba_layer(d: int) -> float:
    return 6.25 * d * d + 112 * d


def shared(d: int) -> float:
    return (N_VOCAB + N_COND) * d + MAX_LEN * d + (N_CHAR + 1) * d + d * N_VOCAB + N_VOCAB


def mamba_est(d: int, n: int, e: int) -> float:
    layout = _interleave_positions(n, e)
    n_tf = sum(layout)
    return shared(d) + n_tf * tf_layer(d) + (n - n_tf) * mamba_layer(d)


def tf_est(d: int, n: int) -> float:
    return shared(d) + n * tf_layer(d)


# Clean-ratio mamba layouts: n a multiple of (every+1) -> no trailing partial block -> exact ratio.
ZONE = []
for every, ratio in [(7, "1:7"), (6, "1:6"), (5, "1:5"), (4, "1:4"), (3, "1:3")]:
    for k in (1, 2, 3, 4):
        n = k * (every + 1)
        ZONE.append((every, n, ratio))
DGRID = [448, 512, 576, 640, 704, 768, 832]  # head_dim = 64 (n_heads = d/64)
RATIO_RANK = {"1:7": 0, "1:6": 1, "1:5": 2, "1:4": 3, "1:3": 4}


def main() -> None:
    print(f"shared dims: N_VOCAB={N_VOCAB} N_COND={N_COND} MAX_LEN={MAX_LEN}  TARGET~{TARGET:,}\n")
    cands = []
    for every, n, ratio in ZONE:
        for d in DGRID:
            h = d // 64
            pm = mamba_est(d, n, every)
            ntf = max(1, round((pm - shared(d)) / tf_layer(d)))
            pt = tf_est(d, ntf)
            rel = abs(pt - pm) / pt
            total = (pt + pm) / 2
            if 42e6 <= total <= 60e6:
                lay = _interleave_positions(n, every)
                cands.append(
                    dict(every=every, n=n, ratio=ratio, d=d, h=h, ntf=ntf, pm=pm, pt=pt,
                         rel=rel, total=total, a=sum(lay), b=n - sum(lay))
                )
    cands.sort(key=lambda c: (RATIO_RANK[c["ratio"]], abs(c["total"] - TARGET), c["rel"]))
    print(f"=== analytic candidates (clean ratio, ~50M): {len(cands)} ; top 14 ===")
    for c in cands[:14]:
        ok = "<=2%" if c["rel"] <= TOL else " >2%"
        print(
            f"  {c['ratio']} d={c['d']}/{c['h']}h  mamba {c['n']}L/e{c['every']}"
            f"({c['b']}m+{c['a']}tf)~{c['pm']/1e6:.2f}M  vs tf {c['ntf']}L~{c['pt']/1e6:.2f}M"
            f"  rel~{c['rel']:.2%} {ok}  ~{c['total']/1e6:.1f}M"
        )

    # ACTUALLY build the best few <=2% candidates (analytic is exact, but never lock on analytic).
    passing = [c for c in cands if c["rel"] <= TOL]
    build_set = passing[:4] if passing else cands[:4]
    print("\n=== ACTUAL builds (confirm analytic==built; rel on built counts) ===")
    built = []
    for c in build_set:
        d, h, every, n, ntf = c["d"], c["h"], c["every"], c["n"], c["ntf"]
        pm = mamba_build(d, n, h, every)
        best = None
        for nt in (ntf - 1, ntf, ntf + 1):
            if nt < 1:
                continue
            pt = tf_build(d, nt, h)
            rel = abs(pt - pm) / pt
            if best is None or rel < best[2]:
                best = (nt, pt, rel)
        nt, pt, rel = best
        lay = _interleave_positions(n, every)
        a, b = sum(lay), n - sum(lay)
        built.append(dict(ratio=c["ratio"], d=d, h=h, every=every, n=n, a=a, b=b, nt=nt, pm=pm,
                          pt=pt, rel=rel, total=(pm + pt) / 2))
        analytic = round(mamba_est(d, n, every))
        exact = "exact==analytic" if pm == analytic else "DIFFERS"
        tag = "PASS<=2%" if rel <= TOL else "fail>2%"
        print(
            f"  {c['ratio']} d={d}/{h}h: mamba {n}L/e{every}({b}m+{a}tf)={pm:,} [{exact}]"
            f" | tf {nt}L={pt:,} | rel={rel:.3%} {tag} | ~{(pm + pt) / 2 / 1e6:.1f}M"
        )

    ok = [r for r in built if r["rel"] <= TOL]
    ok.sort(key=lambda r: (RATIO_RANK[r["ratio"]], abs(r["total"] - TARGET)))
    print("\n=== VERDICT ===")
    if not ok:
        print("NO clean-ratio config <=2% near 50M among built candidates — widen DGRID/zone.")
        print("DONE", flush=True)
        os._exit(0)
    w = ok[0]
    layout = _interleave_positions(w["n"], w["every"])
    layout_str = "".join("T" if x else "m" for x in layout)
    tf_pos = [i + 1 for i, x in enumerate(layout) if x]
    print(
        f"WINNER ratio {w['ratio']}  shared d_model={w['d']} (heads={w['h']}, head_dim=64)\n"
        f"  transformer-ar: d{w['d']}/{w['nt']}L/{w['h']}H = {w['pt']:,}\n"
        f"  mamba-hybrid:   d{w['d']}/{w['n']}L/e{w['every']} "
        f"({w['b']}m+{w['a']}tf) = {w['pm']:,}\n"
        f"  delta={abs(w['pt'] - w['pm']):,}  rel={w['rel']:.3%}  total ~{w['total'] / 1e6:.1f}M\n"
        f"  layout: {layout_str}  tf at {tf_pos}  ratio mamba:tf = {w['b']}:{w['a']}"
    )

    # Non-vacuous proof on the WINNER's mamba knob dict: +1 layer must push rel > 2%.
    pm_bad = mamba_build(w["d"], w["n"] + 1, w["h"], w["every"])
    rel_bad = abs(w["pt"] - pm_bad) / w["pt"]
    print(
        f"  NON-VACUOUS: mamba {w['n']}L->{w['n'] + 1}L = {pm_bad:,} rel={rel_bad:.3%} -> "
        f"{'REDS >2% (gate FAILS, good)' if rel_bad > TOL else 'STILL GREEN — VACUOUS!'}"
    )
    print("DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
