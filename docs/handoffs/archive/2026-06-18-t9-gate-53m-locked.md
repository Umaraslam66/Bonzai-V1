> ⚠️ SUPERSEDED & CONTAINS A CORRECTED ERROR — historical (flagged 2026-06-23). NOT the live boot doc
> (superseded by `2026-06-19-eval-set-gen-cell-selection-next.md` and by `docs/PROJECT_FOCUS.md`). Verified
> vs `docs/GROUND_TRUTH.md` §3: this doc's §3 BUDGET says "Σ 1,859 held-out **cells**" — that is **1,859
> usable TILES, not cells** (real held-out cells ≈ 94,520); and its "~1,008 GPU-h / ~20%" scored-matrix
> budget is **VOID** (it assumed 1,859 = cells). Do not act on this doc's cell counts or budget. Canon: GROUND_TRUTH §3.

# LIVE BOOT DOC — Phase-2 bake-off: ~53M LOCKED (clean 1:7), param-match DONE; sharding-golden + T10 next (2026-06-18)

Current boot doc for the Phase-2 bake-off. Supersedes the archived handoffs in
`docs/handoffs/archive/`. Branch **`phase-2-bakeoff-2backbone`**, UNMERGED.
**Do NOT merge and do NOT submit scored runs without Umar's word.**

## 0. CONSULT FIRST (in this order)
1. **`docs/GROUND_TRUTH.md`** — canonical facts (units, account, corpus, locked design), each with a
   source pointer. **If anything below disagrees with GROUND_TRUTH, that file wins.** Always state
   the unit (node-h vs GPU-h — differ by 4×).
2. **Your memory** (`MEMORY.md` loaded each session). Recall the `feedback_*` lessons first — esp.
   `feedback_no_marker_without_endstate_verify`, `feedback_tool_output_trustworthiness_layer`,
   `feedback_verify_before_lock_not_after`, `feedback_lock_and_guards_travel_together`. One-line
   state = `project_bakeoff_53m_locked`; Leonardo FS/torch ops = `leonardo_torch_import_and_fs_health`.
3. **Spec** `docs/superpowers/specs/2026-06-17-phase-2-bakeoff-2backbone-delta-design.md` §1A (locked
   decision), §9 (seeds/two-floor). **Plan** `docs/superpowers/plans/2026-06-17-phase-2-bakeoff-2backbone.md`
   Task 9 (DONE), Task 10 (scored matrix, GATED on Umar + the sharding golden), Task 11 (eval-sharding).
4. **`docs/protocols/sub-project-planning-protocol-v3.md`** before any new brainstorm.

## 1. THE LOCKED DECISION (Umar's word, 2026-06-18)
Phase 2 is a **SINGLE fixed-scale bake-off, NOT a scaling curve.** One rung **N ≈ 53M**, two
backbones **`transformer-ar` vs `mamba-hybrid`**, **`--no-compile`** both, **3 seeds** per backbone,
**4-GPU eval-sharding**. Matrix = **2 bb × {53M} × 3 seeds = 6 runs**. Basis = fixed-scale by choice
(`FIXED_SCALE_PLUS_S13`).

**The LOCKED rung** — `src/cfm/models/bakeoff_scales.py`, `"53M"` key (append-only; the
{30/100/300M/1B} ladder rungs untouched). ACTUAL built counts; real `test_bakeoff_param_match.py`
green at 53M and non-vacuous (a >2% perturbation REDS it). Cross-checked `$WORK`↔`$SCRATCH` to the digit:
- transformer-ar: `d512 / 14L / 8H` = **52,798,948**
- mamba-hybrid: `d512 / 24L / transformer_every=7` = **53,733,348** — clean **1:7 Jamba** (21 mamba +
  3 tf, tf at layers 8/16/24). delta = **1.77% ≤ 2%**.

**Why ~53M (not 50M) and clean 1:7:** the original aim was ~50M (Chinchilla ~20 tok/param; 100M was
the diagnostic rung). The param-match gate is ratio-BLIND, so a pure 50M match picked `d640/14L` =
1 tf + 13 mamba (**13:1**, attention-starved, below Jamba's validated 1:7). Clean 1:7 within the 2%
gate is unreachable near 50M (mutually exclusive), so Umar chose ratio fidelity → moved to `d512`/~53M.
At ~53M: r=20 → ~1.06B tokens → ~1.7× reuse (in the safe ≤~4× band). Derived (ratio-constrained) by
`scripts/rederive_53m_ratio.py`.

## 2. T9 DIAGNOSTIC — VERIFIED CLEAN from artifacts (job `47143523`, COMPLETED 0:0, 3:15:45)
- `.err`: `Trainer.fit stopped: max_steps=110000 reached` → full horizon, not an early stop;
  fit 8082.8s + eval 3510.3s ≈ elapsed. **Eval RAN** (the watched failure did not happen).
- Report `reports/phase-1-training-scaffold/2026-04-15.0-krakow-transformer-ar-89M-seed7-loop-closed.md`:
  **`gen_seconds_per_token = 0.026779`** (single-GPU, @100M mixer), decodability 0.983,
  OGC-valid 0.989, right-angle 0.9985. `emergence_verdict: INCOMMENSURATE` = the expected
  2048<13312 gap (F15), not a failure.
- Loss `reports/logs/training-scaffold/version_21/metrics.csv`: mean train_loss 2.60→2.094;
  last 30k steps ~0.026 nats → **FLATTENED** at r≈40 (via ~5.8× reuse). val_loss not logged.

## 3. BUDGET (units canon = GROUND_TRUTH §1; grant = 5,000 GPU-h = 1,250 node-h = 40,000 core-h)
Conservative projection at the 0.026779 s/tok rate (measured @100M; the ~53M eval rate is re-measured
at the scored stage, expect lower) × 13,312 full-cap × Σ 1,859 held-out cells (523/579/156/601 =
glasgow/eisenhüttenstadt/munich/krakow) × 2 backbones:

| seeds | AS-IS rank-0 eval | 4-GPU eval-sharding |
|---|---|---|
| 1 | 336 node-h = 1,344 GPU-h (27%) | 84 node-h = 336 GPU-h (7%) |
| 2 | 672 node-h = 2,688 GPU-h (54%) | 168 node-h = 672 GPU-h (13%) |
| **3 (LOCKED)** | 1,008 node-h = 4,032 GPU-h (**81%**) | 252 node-h = **1,008 GPU-h (~20–21%)** |

The full 4-rung ladder was ~22,000 GPU-h (4.4× over) → drove the single-scale lock. AS-IS 3 seeds =
81% of the grant; **sharding (~4×) makes it ~21%** → that is why sharding is locked, not optional.
Training (r≈20 ≈1.06B tokens) is ~<15 node-h (DDP, no waste).

## 4. DONE THIS SESSION (2026-06-18; UNCOMMITTED working-tree changes — Umar hasn't said commit)
- **`src/cfm/models/bakeoff_scales.py`** — appended the `"53M"` rung (the locked d512 clean-1:7 pair
  above) to `BAKEOFF_SCALES` + both knob dicts + the docstring measured-counts table. Existing rungs
  untouched. Real `test_bakeoff_param_match.py` passes at 53M (genuine pytest, all 5 rungs green);
  non-vacuous proof on the real test (perturb 24→25L → `FAILED 5.0% > 2%`; revert → green).
- **`scripts/rederive_53m_ratio.py`** (NEW, tracked) — the ratio-constrained derivation tool that
  produced the 53M rung. `scripts/tune_bakeoff_scales.py` reverted to its committed state (the generic
  param-match ladder tool; it does NOT produce the ratio-constrained 53M rung). Obsolete `_derive_50m.sh`
  and the other `_`-scratch deleted.
- **Docs honestly relabeled ~53M** (no "50M" label that contradicts the count): GROUND_TRUTH §4/§5,
  spec §1A, plan Task 9/10, this boot doc, memory. Chinchilla reuse ~1.7×, budget ~21%.
- Earlier this session (already committed, base `9a450f4` → `c4131ee`): eval-sharding CORE
  (`src/cfm/eval/shard.py`), NO_DECISIVE_WINNER two-floor verdict (`city_aggregate`), the GPU
  equivalence golden authored (`scripts/eval_sharding_golden.{py,sbatch}`), `bakeoff_run.sbatch`
  `--no-compile` + region-injection carry-forwards.

## 5. NEXT — held for Umar's word (NOT blocked; `$WORK` is healthy)
1. **Eval-sharding GPU equivalence golden**: `sbatch scripts/eval_sharding_golden.sbatch` (torchrun,
   4 ranks). Two teeth: (1) per-cell tokens 4-GPU-sharded == rank-0 baseline bit-identical;
   (2) ragged-city (523) count-conservation. Writes `reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS`
   only if green (marker AFTER on-disk verification, F8). Must PASS before scored runs.
2. **`--test-only` on `bakeoff_run.sbatch`**: `sbatch --test-only scripts/bakeoff_run.sbatch`; also the
   torch-needing `tests/training/test_cli_contract.py` (the ones importing `train_scaffold`).

Then **STOP for Umar** before T10 (emit `configs/experiments/bakeoff-{transformer-ar,mamba-hybrid}-53M.yaml`
from the locked table — each carrying `region` + r-derived `max_steps` + `seeds:[…3…]`) and before any merge.

## 6. `$WORK` / Leonardo FS STATUS — the "outage" was a MISDIAGNOSIS (corrected 2026-06-18)
CINECA reported a `$WORK` incident earlier, but the "heavy `$WORK` I/O still degraded" call was WRONG.
It rested on torch-import *speed* (~63s + a GPU-less interpreter-teardown hang) — which is the NORMAL
Leonardo login-node baseline (confirmed ~81s on a provably-healthy `$SCRATCH`, different Lustre
hardware). **Direct throughput on the 222 tree is healthy: 1.6 GB/s write+fsync, 5.1 GB/s read.**
Judge FS health with `dd if=/dev/zero of=<f> bs=1M count=1024 conv=fsync` (+ readback) vs the
`$SCRATCH` baseline — **NOT** torch-import speed. Details + bypass recipe: memory
`leonardo_torch_import_and_fs_health`. Anything importing `mamba_ssm` needs the gcc-12 libstdc++
`LD_PRELOAD` (see `scripts/rederive_53m_ratio.py` header); capture counts then `os._exit(0)`.

## 7. CAVEATS / RECURRING TRAPS (do not relearn these the hard way)
- **torch-import speed ≠ FS health** — ~60–80s import + GPU-less teardown hang is the NORMAL Leonardo
  baseline (proven on healthy `$SCRATCH`). Use `dd` throughput; see §6.
- **node-h vs GPU-h**: 1 node = 4 GPU = 32 core. The "5,000" ceiling is GPU-h. Always label.
- **Account/storage split**: compute = `AIFAC_P02_548`; repo+data+venv on the **222 tree**
  (`/leonardo_work/AIFAC_P02_222`); 222 compute expired 2026-06-11 (FS durability risk); 548 tree is
  an empty stub. Deploy = git bundle to the shared 222 path (no GitHub creds on login).
- **singapore is HISTORICAL**; bake-off data is `eu-train-union`; `ScaffoldConfig.region` is REQUIRED
  (fail-closed, no default).
- **`bakeoff_scales.py` is the lock surface** — only actual-built-count-verified numbers; the 53M rung
  also carries the clean-1:7 ratio constraint (not just param-match). The gate is ratio-BLIND, so a
  pure param-match silently drifts the Jamba ratio (it picked 13:1 before Umar's ratio call).
- **Markers must follow proven end-state** (F8): `_SHARDING_GOLDEN_PASS` / `JOB_DONE` only after
  on-disk verification.
- **Do NOT solo `model_vs_real_effect` (§7)** — it's a brainstorm with Umar (a conditioning-echoing
  model must FAIL it). Obligations (b)/(c): munich→manchester power-gate reserve + EU-train-split
  resolved-gap recompute fire at the scored/decision stage.
