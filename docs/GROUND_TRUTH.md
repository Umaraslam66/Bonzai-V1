# GROUND_TRUTH — canonical facts (read before quoting any of these)

The facts below keep getting mis-stated across sessions (node-h vs GPU-h, 548 vs 222, singapore
residue). Each is stated **once**, with its **source pointer**. If a summary/memory/handoff
disagrees with this file, **this file wins** — and fix the summary. If a fact here looks wrong,
re-derive it from its source (don't trust this file blindly either); update it here with the new
source. Reports under `reports/` and `logs/` are byte-deterministic primary artifacts and are
never edited to match this file.

Last reconciled: 2026-06-18.

## 1. Compute units — ALWAYS state the unit
- Leonardo Booster node = **32 CPU cores + 4× A100**. Source: `scontrol show node lrdn0375`
  (`CPUTot=32, Gres=gpu:a100:4`); a full-node job bills `billing=32` (`sacct -j 47143523 -o AllocTRES`).
- The grant is in **core-hours** (`saldo -b` column "local h"). `AIFAC_P02_548` total = **40,000 core-h**.
- Therefore: **40,000 core-h = 1,250 node-h = 5,000 GPU-h.** The bake-off "5,000" ceiling is
  **GPU-hours** (= the full 548 grant). Any "5,000 node-h" in older docs is a **MISLABEL**.
- Conversions: `1 node-h = 4 GPU-h = 32 core-h`. A budget % is unit-invariant only if budget and
  ceiling use the SAME unit — convert explicitly.
- `gen_seconds_per_token` measured by the diagnostic is a **single-GPU** rate (post-train eval runs
  on rank 0 only; the node's other 3 GPUs are allocated-and-billed but idle). Source:
  `scripts/train_scaffold.py` (`if not trainer.is_global_zero: return`). Eval-sharding (Task 11)
  recovers that ~4×.

## 2. Storage & account
- **Compute** is charged to **`AIFAC_P02_548`** (`#SBATCH --account=AIFAC_P02_548`; ends 2026-09-17).
- **Repo + EU data + the unified venv** live on the **222 tree**:
  `/leonardo_work/AIFAC_P02_222/Bonzai-OSM` (read-write). Deploy = git bundle to the shared
  `/leonardo_work/AIFAC_P02_222/*.bundle` (no GitHub creds on the login node).
- The **548 tree** (`/leonardo_work/AIFAC_P02_548/Bonzai-OSM`) is an **empty stub**.
- **Durability risk:** the 222 *compute* allocation expired **2026-06-11**; the 222 *filesystem*
  tree still holds repo/data/venv but is at risk post-expiry. `chprj` switches the active project.
  Source: **`saldo -b`** — the `AIFAC_P02_222` row's `start … end` = `20260311 … 20260611`, and the
  `AIFAC_P02_548` row = `20260617 … 20260917` (where the §2 "548 ends 2026-09-17" also comes from).
  (Re-run `saldo -b` to re-confirm; memory `project_allocation` echoes it.)
- Drive Leonardo from the Mac via the SSH ControlMaster socket (`Host leonardo`, user `uaslam00`).
  Source: memory `reference_leonardo_claude_ssh_socket`.

## 3. Corpus & regions
- Bake-off trains on **`eu-train-union`** (EU multiregion), **~624M unique tokens**
  (`TRAIN_TOKENS = 623,900,790`). Source: `src/cfm/eval/ladder.py` (`TRAIN_TOKENS`).
- **Held-out cities** (excluded from training, evaluated against): **glasgow, eisenhüttenstadt,
  munich, krakow** — per-city usable **TILES (NOT cells)**: **glasgow 523 / eisenhüttenstadt 579 /
  munich 156 / krakow 601** (Σ = **1,859 usable tiles**). Source: `reports/2026-06-08-usable-n.yaml`
  (`n_usable_tiles`) + the multiregion holdout manifest.
  **CORRECTION (2026-06-19): 1,859 is the usable-TILE count, NOT a cell count.** Earlier docs that
  said "1,859 held-out cells" conflated tiles with cells. The real held-out **CELL** count is
  **~77,000** (`held_out_tokens 46,130,102 ÷ 596.7` avg cell-body tokens). A tile holds many cells.
- **Held-out eval CELL SELECTION = the multiregion eval-set-gen sub-project — NOT built yet.** The
  tile-level holdout manifest (`data/processed/eval_set/2026-04-15.0/multiregion/`, schema 2.0,
  sha `ae4d5af…`, `_EVAL_SET_LOCKED`; eval-set-gen Phase B) selects which TILES are held out — it does
  NOT select a power-sized CELL set. The realism eval (generate held-out cells → Lane-S → decide) is
  GATED on that selection: power-sized + KS-resolvable per (zoning, road_skeleton, density, coastal)
  4-tuple stratum at ≥min_n=50, frozen + sha-locked like the conditioning floor. The scored-matrix
  budget MUST be re-derived at the true cell scale (the old ~1,008 GPU-h / ~20% assumed 1,859 = cells,
  which is wrong).
- **singapore is HISTORICAL ONLY** (Phase-1 de-risking). `ScaffoldConfig.region` is REQUIRED and
  fail-closed — there is no silent singapore default. Source: commit `dbdf3d5`,
  `src/cfm/training/config.py`.

## 4. Locked Phase-2 bake-off design (spec §1A/§9, plan Task 10/11)
- **Single fixed-scale bake-off, N ≈ 53M** — NOT a scaling curve, NO ladder. Decision basis =
  fixed-scale by choice (`FIXED_SCALE_PLUS_S13` family). Rationale: Chinchilla ~20 tok/param →
  optimum ~30M for 624M tokens; 100M was the measured diagnostic rung; ~50M was the chosen middle.
  **The rung LANDED at ~53M, not 50M** (see next bullet): a clean Jamba 1:7 ratio param-matches
  ≤2% only at ~53M. At N≈53M, r=20 → ~1.06B tokens → **~1.7× reuse** (still in the safe ≤~4× band);
  eval ≈ **~21% of the grant** (3 seeds, 4-GPU sharding). Never label this rung "50M".
- **Two backbones, shared `d_model=512`, param-matched ≤2% on ACTUAL built counts (locked 2026-06-18
  in `src/cfm/models/bakeoff_scales.py` under the `"53M"` key, append-only):**
  - `transformer-ar`: `d512 / 14L / 8H` = **52,798,948**
  - `mamba-hybrid`: `d512 / 24L / transformer_every=7` = **53,733,348** — a **clean 1:7 Jamba**
    interleave (21 mamba + 3 transformer, tf at layers 8/16/24). delta = **1.77% ≤ 2%**.
  - **`--no-compile`** both (T7 verdict). **3 seeds** per backbone. **4-GPU eval-sharding**.
  - WHY clean 1:7 (not the param-match optimum): the pure param-match to 50M picked `d640/14L` =
    **1 tf + 13 mamba (13:1)** — attention-starved, below Jamba's validated 1:7. Clean 1:7 within 2%
    is unreachable near 50M, so the rung moved to `d512`/~53M. Derived (ratio-constrained) by
    `scripts/rederive_53m_ratio.py`; `tests/models/test_bakeoff_param_match.py` passes at 53M and is
    non-vacuous (a >2% perturbation REDS it). Source: spec §1A.
- **Seed→verdict rule:** per (backbone, city) the 3 seeds give mean KS (estimate) + std/SEM
  (seed-noise). A winner is crowned at a city ONLY if the winner-vs-runner-up mean-KS gap clears
  `effective_floor = max(C/√n resolvability, seed-noise reproducibility)`. Clearing one floor but
  not the other (the likely near-tie MIDDLE band) is NOT decisive. If no city is decisive →
  **`NO_DECISIVE_WINNER`** (S13 family), a named verdict. **Memorization-first hard-halt** precedes
  all of this. Source: spec §9, plan Task 10 Step 3.
- **Diagnostic measured facts** (job `47143523`, COMPLETED clean): `gen_seconds_per_token = 0.026779`
  (single-GPU, @100M mixer); loss flattened by r≈40. Source: report
  `reports/phase-1-training-scaffold/2026-04-15.0-krakow-transformer-ar-89M-seed7-loop-closed.md`.

## 5. Open / deferred
**The "`$WORK` outage" was a MISDIAGNOSIS (2026-06-18):** the "heavy `$WORK` I/O degraded" call
rested on torch-import *speed* (~63s + a GPU-less interpreter-teardown hang), which is the NORMAL
Leonardo login-node baseline — confirmed by ~81s on a provably-healthy `$SCRATCH` (different Lustre
hardware). Direct throughput on the 222 tree is healthy: **1.6 GB/s write+fsync, 5.1 GB/s read**.
Judge `$WORK` health with a 1 GB `dd` write+fsync vs the `$SCRATCH` baseline, **NOT** torch-import
speed. (Also: anything importing `mamba_ssm` needs the gcc-12 `libstdc++` `LD_PRELOAD` from
`scripts/rederive_53m_ratio.py`; capture counts then `os._exit(0)` to dodge the teardown hang.)
1. **53M param-match — DONE (2026-06-18):** locked `"53M"` into `bakeoff_scales.py` (ratio-
   constrained, clean 1:7 — see §4); real `test_bakeoff_param_match.py` green + non-vacuous proof.
2. **Eval-sharding GPU equivalence golden** (plan Task 11 Step 4) — DEFERRED (held for Umar's word,
   NOT blocked): per-cell scores 4-GPU-sharded == rank-0 baseline bit-identical + count-conservation
   on a real distributed run incl. a ragged city. CPU-safe core (`src/cfm/eval/shard.py`) + local
   tests DONE; run `sbatch scripts/eval_sharding_golden.sbatch`.

Live boot doc: `docs/handoffs/2026-06-19-eval-set-gen-cell-selection-next.md`
(supersedes `2026-06-18-bakeoff-t10-eval-wiring-open.md`; eval pipeline WIRED + verified — sharding,
4-tuple gen keying, 29MB parquet floor-repro, memorization-halt; the ONE blocker is the held-out
CELL SELECTION = eval-set-gen sub-project, NOT built — see §3).
