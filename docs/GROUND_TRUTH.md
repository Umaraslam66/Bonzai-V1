# GROUND_TRUTH — canonical facts (read before quoting any of these)

The facts below keep getting mis-stated across sessions (node-h vs GPU-h, 548 vs 222, singapore
residue). Each is stated **once**, with its **source pointer**. If a summary/memory/handoff
disagrees with this file, **this file wins** — and fix the summary. **The claude-mem auto-summary
observations injected at session start are NOT a verified source** — they have inverted agent
conclusions and conflated units (e.g. tiles↔cells; obs 8262 "cell-selection BUILT" and 8233 "cells"
for tiles were both FALSE when emitted — the sampler HAS since been built, 2026-06-22, see §3, but
that does not retro-validate the obs; the trust rule stands); treat them as leads and verify here. If a fact here looks wrong,
re-derive it from its source (don't trust this file blindly either); update it here with the new
source. Reports under `reports/` and `logs/` are byte-deterministic primary artifacts and are
never edited to match this file.

Last reconciled: 2026-07-17 (previous: 2026-06-20).

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
  said "1,859 held-out cells" conflated tiles with cells. A tile holds many cells.
  **COUNTED (2026-06-20, job `47473524`, `_heldout_cell_count.py`): 94,520 distinct non-empty
  (conditionable) held-out cells** over the floor's manifest tile set (**1,952** held-out tiles —
  glasgow 549 / eisenhüttenstadt 616 / munich 171 / krakow 616, missing-cells=0; this is the manifest
  `tiles[]` the floor reads, a SUPERSET of the 1,859 "usable" filter). The earlier **~77,000 estimate
  (`46,130,102 ÷ 596.7`) UNDERCOUNTED** — measured avg cell-body ≈ **488 tok**, not 596.7. Distinct
  cells per `(zoning, skeleton, density, coastal)` stratum: **78 global 4-tuples / 185 `(city,stratum)`
  units** (the floor's keying). **Thin-strata is LIVE at cell granularity: 82/185 `(city,stratum)`
  (44%) hold < 50 distinct cells (95 < 75, 103 < 100); 27/78 global < 50.** This is NOT the
  feature-level picture (the floor's `min_n=50` is on FEATURES, where 0/312 held-out strata-metrics fall
  below 50 — a cell emits many features). The {N / thin-strata} call was resolved by the sampler spec
  (feature-count-driven sizing, scarce-metric binding + ceiling-bound — see
  `docs/superpowers/specs/2026-06-21-lane-s-cell-sampler.md` Gates 1–5). Keying code:
  `conditioning_discrimination.py:454-465`.
- **Held-out eval CELL SAMPLER — BUILT + SEALED (2026-06-22; supersedes the 2026-06-20 "NOT built" line).**
  Module `src/cfm/eval/lane_s_sampler.py` + `scripts/build_lane_s_sampler.py` (commits `dfc2b89` spec →
  `c1ce4b8` orchestrator → `bd54ebe` census_sha256 pin → `69bca37` sbatch). Manifest sealed on Leonardo
  (job `47600021`, no-GPU serial, HEAD `69bca37`): **146/146 floored (city,4-tuple) strata built, 5,705
  cells**, `floor_sha256=95abb88…` (==EXPECTED), `census_sha256=236cea99…` recomputes byte-for-byte;
  marker `data/processed/lane_s_sampler/2026-04-15.0/_LANE_S_SAMPLER_LOCKED` (verified present
  2026-07-17). Write-once like the floor. Sources: plan `docs/superpowers/plans/2026-06-21-lane-s-cell-sampler.md`
  (execution log item 1) + Leonardo marker. Generation-cost re-derivation at sealed N: ~154 transformer
  GPU-h ≈ 3.1% of grant (plan log); combined both-backbone matrix ≈ 262 GPU-h ≈ 5.25% with the measured
  mamba rate 0.0652 s/tok, job `47603817` (session record 2026-06-23 — re-derive before any scored run).
  The realism eval (generate held-out cells → `gen_realism` 4-tuple → Lane-S vs the conditioning floor →
  `decide`) is a budget-bounded **stratified DOWN-sampler** over the held-out cells keeping **≥ min_n=50
  generated features per floored 4-tuple stratum** (zoning, road_skeleton, density, coastal). It is a
  DOWN-sampler, **not a selector-from-scratch**: the cells already cover the strata — the **265 floored
  strata are proven feasible** (per-city floor rows: eisenh 56 / glasgow 78 / krakow 70 / munich 61, so
  even the thinnest city clears min_n=50). **This is NOT a continuation of the 2026-06-08 eval-set-gen
  plan** — that built
  the SEPARATE *per-tile coherence* lane (shuffle-gap, per-city); the cell sampler feeds the
  *Lane-S / conditioning-floor* lane (`conditioning_floor.py` + `bakeoff_decision.py` + `gen_realism.py`,
  floor sha `95abb88`). The tile-level holdout manifest (`data/processed/eval_set/2026-04-15.0/multiregion/`,
  schema 2.0, sha `ae4d5af…`, `_EVAL_SET_LOCKED`; eval-set-gen Phase B) selects which TILES are held out —
  it does NOT select cells; it is the sampler's INPUT. Frozen + sha-locked write-once like the floor. The
  scored-matrix budget re-derivation at the chosen N is DONE at seal time (see the sealed-sampler
  bullet above; the old ~1,008 GPU-h / ~20% figure assumed 1,859 = cells and is dead).
- **Eval generation cap LOCKED at `DEFAULT_MAX_CELL_TOKENS = 13,312`** (`datamodule.py:63`) — record-only
  decision **2026-06-20, NOT lowered**. With cell-EOS self-termination (Tooth-1 GPU smoke, job `47470695`:
  **0/64 cells at cap, p99≈601, 100% emit `<cell_end>`=260**) the cap is a rarely-hit SAFETY CEILING, not a
  cost driver — per-cell eval cost scales with the ~600-tok self-terminated length, not the cap (capping
  saves ≈ nothing; per-cell at 600 tok = 0.0045 GPU-h vs at the 13,312 cap = 0.099 GPU-h, but cells stop at
  ~600 so the cap is near-free). Any prior **"cap at ~4,500 to save eval budget" framing is DEAD** (it
  assumed cells run to the cap; they self-terminate — no such framing survives in committed docs). Cap stays
  13,312 to avoid truncating the rare long cell. Source: cell-EOS spec `docs/superpowers/specs/2026-06-20-cell-eos.md`.
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
- **6-run scored TRAINING matrix — DONE (2026-06-23, train-only, `eval_cells 0`,** authorized commit
  `b7ed4de`): all 6 runs (2 backbones × 3 seeds) COMPLETED exit 0, final `last.ckpt` on disk on
  Leonardo, ~112,563 steps each; tf ~6.2 h/run, mamba ~12.3 h/run; **~222 GPU-h ≈ 4.4% of grant**
  (session record 2026-06-23 — checkpoints verifiable on Leonardo; no nan/OOM).
- **Standing methodology-eval matrix — 6/6 COMPLETE (2026-06-25, job `47824952`, code `0d677cc`):**
  architecture **NO_DECISIVE** (macro-only perplexity-gap T +0.0066 vs M +0.0063, Δ inside seed-noise
  0.0012); conditioning IS read (6/6 sign-tests significant); **~99% of conditioning signal =
  char_stats, ~1% = macro buckets**; geometry valid on both backbones. This is METHODOLOGY evidence,
  not a crown. Detail + gates: `docs/PROJECT_FOCUS.md`.
- **Macro-conditioning weakness — DIAGNOSED (2026-06-26→28, jobs `47910130` factorial + `47994849`
  per-field; commits `34e3059`…`69242e2`; JSONs `reports/_diag/`):** cause = **intrinsic data
  weakness** (macro standalone ~0.012–0.021 nats/tok vs char_stats +0.63–0.75; injection + capacity
  ruled out). **road_skeleton is the dominant per-field carrier** on all 3 checkpoints (~25–30% of the
  within-city macro effect); zoning weakest; fields ~additive (81–91%). char_stats dominates because
  it is a leak-grade near-sufficient statistic computed FROM the target cell. Caveat stands: this
  measures NLL, **NOT generation steering/plausibility**. Detail: `docs/PROJECT_FOCUS.md`.

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
2. **Eval-sharding GPU equivalence golden** (plan Task 11 Step 4) — **DONE / PASS** (job `47390793`):
   per-cell scores 4-GPU-sharded == rank-0 baseline bit-identical + count-conservation on a real
   distributed run incl. a ragged city — BOTH backbones tooth1_mismatches=0, 523/523 gathered (holes=0),
   determinism True; `_SHARDING_GOLDEN_PASS` on Leonardo. CPU-safe core (`src/cfm/eval/shard.py`).
3. **Deploy obligation (STILL OPEN as of 2026-07-17):** before any scored or probe run, **re-deploy
   Leonardo to the Mac's committed HEAD** — Leonardo HEAD observed `b7ed4de` (2026-07-17) vs Mac
   `69242e2`; the diagnostic-era commits (`34e3059`…`69242e2`) and standing-eval finalization are not
   in Leonardo's git history (some were synced as working-tree blobs for job `47824952`). A scored run
   needs the full, git-coherent wired path. (The older `d8ea038` vs `46ea757` gap is superseded.)
4. **Next-step candidates (2026-07-17, orchestrator note — NOT a PI decision record; per-act
   authorization lives in the session/commit trail, verify there):** generation-steering probe (does
   macro STEER generation — the NLL diagnostic's open caveat); v2 conditioning candidates from the
   per-field ablation (char_stats dropout at train time; neighbor-derived context stats instead of
   target-derived char_stats; capacity toward structural fields). Standing rules unchanged: merges
   need a green full suite; >10 GPU-h single spends need an explicit budget line; the full realism
   eval (~262 GPU-h est.) and any v2 retrain matrix need a fresh data-in-hand decision.
5. **Grant snapshot (2026-07-17, `saldo -b`):** ~4,093 / 40,000 core-h consumed ≈ **10.2%** (≈ 512
   GPU-h). NOTE: 7 concurrent non-Bonzai Slurm jobs (`serp-*`) observed running under `uaslam00` on
   this grant — Bonzai budget math must not assume sole tenancy.

Live boot doc: `docs/handoffs/2026-06-19-eval-set-gen-cell-selection-next.md`
(supersedes `2026-06-18-bakeoff-t10-eval-wiring-open.md`; eval pipeline WIRED + verified — sharding,
4-tuple gen keying, 29MB parquet floor-repro, memorization-halt). Its "ONE blocker = cell sampler NOT
built" framing is RESOLVED (sampler sealed 2026-06-22 — see §3); current state = §4 matrix trained +
standing eval NO_DECISIVE + macro diagnostics done; next = steering probe → v2 conditioning decision
(see §5 item 4).
