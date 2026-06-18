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
  munich, krakow** — per-city usable cells **523 / 579 / 156 / 601** (Σ = 1,859). Source:
  multiregion holdout manifest (release `2026-04-15.0`) + `_union_datamodule` in `train_scaffold.py`.
- **singapore is HISTORICAL ONLY** (Phase-1 de-risking). `ScaffoldConfig.region` is REQUIRED and
  fail-closed — there is no silent singapore default. Source: commit `dbdf3d5`,
  `src/cfm/training/config.py`.

## 4. Locked Phase-2 bake-off design (spec §1A/§9, plan Task 10/11)
- **Single fixed-scale bake-off, N = 50M** — NOT a scaling curve, NO ladder. Decision basis =
  fixed-scale by choice (`FIXED_SCALE_PLUS_S13` family). Rationale: Chinchilla ~20 tok/param →
  optimum ~30M for 624M tokens; 100M was the measured diagnostic rung; 50M = the chosen middle.
- **Two backbones:** `transformer-ar` vs `mamba-hybrid` (7:1 Jamba interleave), param-matched ≤2%.
  **`--no-compile`** for both (T7 verdict). **3 seeds** per backbone. **4-GPU eval-sharding**.
- **Seed→verdict rule:** per (backbone, city) the 3 seeds give mean KS (estimate) + std/SEM
  (seed-noise). A winner is crowned at a city ONLY if the winner-vs-runner-up mean-KS gap clears
  `effective_floor = max(C/√n resolvability, seed-noise reproducibility)`. Clearing one floor but
  not the other (the likely near-tie MIDDLE band) is NOT decisive. If no city is decisive →
  **`NO_DECISIVE_WINNER`** (S13 family), a named verdict. **Memorization-first hard-halt** precedes
  all of this. Source: spec §9, plan Task 10 Step 3.
- **Diagnostic measured facts** (job `47143523`, COMPLETED clean): `gen_seconds_per_token = 0.026779`
  (single-GPU, @100M mixer); loss flattened by r≈40. Source: report
  `reports/phase-1-training-scaffold/2026-04-15.0-krakow-transformer-ar-89M-seed7-loop-closed.md`.

## 5. Open / deferred (both BLOCKED on the CINECA $WORK storage outage, 2026-06-18)
1. **50M param-match on ACTUAL built counts** (≤2%): extend `scripts/tune_bakeoff_scales.py` with a
   50M seed, lock the verified pair into `src/cfm/models/bakeoff_scales.py`, pass
   `tests/models/test_bakeoff_param_match.py` at 50M. Provisional analytic (NOT locked):
   transformer-ar `d640/8L/10H` ≈ 50.2M (high-confidence); mamba-hybrid `d640/~14L/every7` (seed).
2. **Eval-sharding GPU equivalence golden** (plan Task 11 Step 4): per-cell scores 4-GPU-sharded ==
   rank-0 baseline bit-identical, + count-conservation on the real distributed run incl. a ragged
   city. CPU-safe core (`src/cfm/eval/shard.py`) + local tests are DONE.

Live boot doc: `docs/handoffs/2026-06-18-t9-gate-50m-locked.md`.
