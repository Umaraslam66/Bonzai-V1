# T9 budget gate DONE → 50M single-scale LOCKED — handoff (2026-06-18)

Supersedes `docs/handoffs/2026-06-17-t9-diagnostic-running.md`. The T9 diagnostic landed,
the budget gate ran, and **Umar locked the scored design.** The ONE remaining open item is
blocked by a CINECA `$WORK` storage outage (below). Branch `phase-2-bakeoff-2backbone`
(unmerged). **Do NOT submit T10 or merge** without Umar's word.

## DECISION (LOCKED 2026-06-18 — supersedes the ladder-sizing open question)
Phase 2 is a **SINGLE fixed-scale bake-off, NOT a scaling curve.** One rung: **N = 50M**,
both backbones (`transformer-ar`, `mamba-hybrid`), **`--no-compile`**. No ladder; the
decision basis is fixed-scale by choice (`FIXED_SCALE_PLUS_S13` family: decide at the single
scale + §13). Recorded in **spec §1A** (`2026-06-17-phase-2-bakeoff-2backbone-delta-design.md`)
and **plan Phase 4 / Task 10**.

**Why 50M:** Chinchilla ≈ 20 tok/param → fully-trained optimum ≈ 30M for our ~624M unique EU
tokens (`TRAIN_TOKENS=623,900,790`); 100M was the measured diagnostic rung; **50M = the chosen
middle** (r≈20 → ~1B tokens → ~1.6× data reuse, inside the safe ≤~4× band → nearly fully-cooked,
more meaningful than 30M).

## T9 DIAGNOSTIC — end-state VERIFIED CLEAN from artifacts (not squeue)
Job `47143523`: `sacct` COMPLETED 0:0, elapsed 3:15:45. Verified against artifacts:
- `.err`: `Trainer.fit stopped: max_steps=110000 reached` → FULL training horizon, not an
  early max-time stop. `fit 8082.8s + eval 3510.3s ≈` elapsed. Eval RAN (the watched failure
  did NOT happen).
- Report `reports/phase-1-training-scaffold/2026-04-15.0-krakow-transformer-ar-89M-seed7-loop-closed.md`:
  `gen_seconds_per_token=0.026779`, decodability 0.983, OGC-valid 0.989, right-angle 0.9985.
  `emergence_verdict: INCOMMENSURATE` is the expected 2048<13312 commensurability gap (F15), not a failure.
- Loss curve `reports/logs/training-scaffold/version_21/metrics.csv`: mean train_loss
  2.60(0-10k)→2.12(80-90k)→2.094(100-110k); last 30k steps ~0.026 nats → **FLATTENED** (r≈40
  via ~5.8× reuse). `val_loss` was not logged; "flattened" = optimization saturated on the
  available data (the held-out eval tests generalization, not this curve).

## UNITS — confirmed 2026-06-18 (saldo -b + scontrol; NOT memory)
Grant AIFAC_P02_548 = **40,000 core-h** (`saldo` "local h"). Booster node = **32 cores + 4×A100**
(`CPUTot=32, Gres=gpu:a100:4`; full node bills `billing=32`). ⇒ **40,000 core-h = 1,250 node-h
= 5,000 GPU-h.** **The "5,000" ceiling is GPU-HOURS** (= the full 548 grant); "5,000 node-h" in
the prior docs was a MISLABEL — corrected here + spec §1A. `gen_seconds_per_token` is a
**single-GPU** rate: post-train eval runs **rank-0 only** (1 GPU works; the node's other 3 GPUs
are allocated-and-billed but idle). (saldo monthConsumed lagged at 27 — grant started 2026-06-17;
unit derivation is from the grant total + fixed node geometry, not the lagging counter.)

## BUDGET GATE — eval dominates; full ladder over budget → single scale
Single-GPU per-token 0.026779 s/tok (@100M) × 13,312 full-cap × **Σ per-city held-out 1,859 cells**
(523/579/156/601 = glasgow/eisenhüttenstadt/munich/krakow) × 2 backbones:
- Full 4-rung {30/100/300M/1B} ladder = **~5,501 node-h EVAL ALONE** (1 seed) = **~22,000 GPU-h
  → 4.4× over the 5,000 GPU-h grant.** Data-feasibility (`r·N ≤ 623.9M·E`) also starved at high r.
  → motivated the single-scale lock.
- **At 50M** (re-scaled, interpolation; per-seed eval ≈336 node-h wall-clock = tf 233 + mamba 103,
  incl. tf full-ctx ×2.7 / mamba ×1.2), vs **1,250 node-h / 5,000 GPU-h**:

  | seeds | AS-IS (rank-0 eval) | 4-GPU eval sharding |
  |---|---|---|
  | 1 | 336 node-h = 1,344 GPU-h (**27%**) | 84 node-h = 336 GPU-h (7%) |
  | 2 | 672 node-h = 2,688 GPU-h (**54%**) | 168 node-h = 672 GPU-h (13%) |
  | 3 | 1,008 node-h = 4,032 GPU-h (**81%**) | 252 node-h = 1,008 GPU-h (20%) |

  Training (r≈20 ≈ 1B tokens) ~<15 node-h total (DDP, no waste). **AS-IS 3 seeds = 81% of the
  whole 3-month grant**; rank-0 eval wastes 3/4 of billed GPU-h, so **4-GPU eval sharding (~4×)
  is the decisive lever**. (Earlier "7/13/20% of 5,000 node-h" was a unit error — node-h vs a
  GPU-h ceiling; AS-IS is 27/54/81%, the 7/13/20% needs sharding.)

## PROVISIONAL 50M PARAM-MATCH (analytic — actual-build verify BLOCKED, see below)
Derived from the EXACT transformer law (`12·d²+13·d`/layer + linear shared scaffold) that
reproduces both known actual rungs (30M, 100M) to the digit:
- **transformer-ar: `d_model=640, n_layers=8, n_heads=10` → 50,219,748** (+0.44%). HIGH confidence.
- **mamba-hybrid: `d_model=640, n_layers≈14, transformer_every=7`** (1 tf + 13 mamba) ≈49,966,948
  (≈0.5% vs tf). **SEED ONLY** — Mamba internal count not exact analytically; actual-build search
  decides n_layers ∈ {13,14,15}.

## ⛔ BLOCKER — CINECA `$WORK` storage outage (2026-06-18)
CINECA reported a `$WORK` filesystem infrastructure issue (slow/unresponsive I/O; some compute
nodes removed; "avoid `$WORK` until further notice"). The login-node param derivation hung at
0% CPU on torch import (paging from `$WORK`). **All Leonardo `$WORK` work is paused.** Diagnostic
artifacts already on disk are fine; do not trust new `$WORK` reads/writes until CINECA clears it.

## NEXT ACTIONS (when `$WORK` recovers) — then STOP for Umar
1. **Param-match 50M on ACTUAL built counts** (full T5 discipline, never eyeball): extend
   `scripts/tune_bakeoff_scales.py` with a 50M seed, run on the unified Leonardo env (construction
   is CPU-safe — login node OK once FS is healthy), confirm transformer-ar vs mamba-hybrid ≤2%,
   **append** the verified pair to `src/cfm/models/bakeoff_scales.py` (append-only; existing rungs
   stay), pass `tests/models/test_bakeoff_param_match.py` at 50M. Do NOT lock the provisional
   analytic numbers. (Temp `scripts/_derive_50m.sh` exists on Leonardo + locally — delete after.)
2. **Re-confirm the budget** with the verified 50M count (replaces the interpolated estimate).
3. **STOP for Umar's word** before T10 (per-run YAMLs, `--no-compile` on `bakeoff_run.sbatch`,
   seed count) and before any merge.

## SEEDS + SHARDING — LOCKED (Umar's word, 2026-06-18)
**3 seeds per backbone + 4-GPU eval-sharding.** Matrix = 2 bb × {50M} × 3 seeds = 6 runs.
Sharding makes 3 seeds ≈ **20%** of the 5,000 GPU-h grant (vs 81% AS-IS). Rationale: seeds
separate skill from luck on the near-tie param-matched models; sharding recovers the 3-idle-GPU
waste.

**SEED→VERDICT RULE (locked, never silent):** per (backbone, city), 3 seeds → **mean KS** =
point estimate, **std/SEM across seeds** = seed-noise. `binding_city_verdict` consumes mean-KS;
the power gate DEMOTES a city unless the winner-vs-runner-up gap exceeds BOTH the feature-
resolution floor (C/√n) AND the seed-noise band → a gap inside seed noise is luck, demoted.
Memorization-first hard-halt still precedes. (Decision-layer extension = a T10 code item; rule locked.)

**EVAL-SHARDING build (Task 11):**
- DONE now (CPU-safe, `$WORK`-independent): `src/cfm/eval/shard.py` (torch-free partition +
  conservation + canonical-order gather) + `tests/eval/test_shard.py` (local, non-vacuous:
  ragged cities 523/579/156/601, conservation, drop/double-count RAISES, shard-order independence).
- DEFERRED to `$WORK` recovery (bundled with the 50M param-match verify): the GPU EQUIVALENCE
  GOLDEN — TWO teeth, non-vacuous: (1) per-cell scores 4-GPU-sharded == rank-0 baseline bit-identical;
  (2) PAIRED structural check on the REAL distributed run incl. a ragged-partition city (523, not ÷4) —
  every cell scored exactly once, no boundary drop/double-count (aggregate equality alone INSUFFICIENT).
  Plus worst-case-city verdict byte-identical across re-runs.

## CARRY-FORWARDS to T10 (unchanged from the prior handoff)
- `--no-compile` for both backbones (T7 verdict); add to `bakeoff_run.sbatch`.
- `bakeoff_run.sbatch` buildability dry-run must inject `'region': '${REGION}'` (item-3 made
  `ScaffoldConfig.region` REQUIRED).
- Per-run YAMLs `configs/experiments/bakeoff-{backbone}-50M.yaml` don't exist — emit at T10, each
  carrying `region` + r-derived `max_steps`.
- Obligations (b)/(c): `model_vs_real_effect` teeth + munich→manchester power-gate reserve +
  EU-train-split resolved-gap recompute — fire at the scored/decision stage.
- Deploy: git bundle to SHARED `/leonardo_work/AIFAC_P02_222/<name>.bundle`. Compute=548;
  repo/data/venv on the 222 tree.
