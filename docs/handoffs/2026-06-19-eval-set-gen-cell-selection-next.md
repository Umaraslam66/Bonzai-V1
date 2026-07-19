# LIVE BOOT DOC — Phase-2 bake-off: eval pipeline WIRED + verified; the ONE blocker is the held-out CELL SAMPLER (Lane-S realism eval, next sub-project) (2026-06-19; lane-framing corrected 2026-06-20)

> **Filename note:** the filename still says "eval-set-gen-cell-selection"; the lane was corrected
> 2026-06-20 — the open item is the **Lane-S realism eval's cell SAMPLER**, NOT a resume of the
> 2026-06-08 eval-set-gen (per-tile coherence) plan. Content below is authoritative; GROUND_TRUTH §3 wins.

Supersedes `docs/handoffs/2026-06-18-bakeoff-t10-eval-wiring-open.md`. Branch
**`phase-2-bakeoff-2backbone`**, UNMERGED. **Do NOT launch a scored run or merge without Umar's word.**

## 0. CONSULT FIRST (in order)
1. **`docs/GROUND_TRUTH.md`** — canonical facts. §3 now carries the CORRECTED held-out units (1,859 =
   usable **TILES**, not cells; real cell count ~77k; cell selection not built). **If anything here
   disagrees, GROUND_TRUTH wins.**
2. **Memory** (`MEMORY.md`): `project_bakeoff_prewiring_verified` (the bake-off DONE state),
   `project_eval_set_gen_execution` (eval-set-gen Phase B DONE), `project_multiregion_eval_set_gen_planned`
   (the design/scope), and the `feedback_*` lessons (esp. `feedback_precondition_verify_count_not_estimate`,
   `feedback_verify_count_lineage` — this whole blocker was a tiles↔cells count conflation).
3. **eval-set-gen design/plan:** `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md` +
   `docs/superpowers/plans/2026-06-08-eval-set-gen.md`. **Bake-off** spec §1A / plan Task 10-11.

## 1. THE CORRECTED FACTS (the propagated error — fixed everywhere 2026-06-19)
- **Held-out = 1,859 usable TILES**, NOT cells: **glasgow 523 / eisenhüttenstadt 579 / munich 156 /
  krakow 601** (`reports/2026-06-08-usable-n.yaml` `n_usable_tiles`). The earlier "1,859 cells" in
  GROUND_TRUTH/spec/YAMLs/shard.py/handoffs was a **tiles↔cells conflation** — corrected.
- **Real held-out CELL count ≈ 77,000** (`held_out_tokens 46,130,102 ÷ 596.7` avg cell-body).
- **No power-sized held-out CELL SELECTION exists.** The locked artifact
  (`data/processed/eval_set/2026-04-15.0/multiregion/`, schema 2.0, sha `ae4d5af…`, `_EVAL_SET_LOCKED`,
  eval-set-gen **Phase B**) selects which **TILES** are held out — it does NOT select cells.
- **Budget invalidated:** the old ~1,008 GPU-h / ~20% assumed 1,859 = cells. RE-DERIVE at the true
  cell scale once the selection fixes N.

## 2. DONE + VERIFIED this phase (the eval pipeline is built; the matrix is one blocker away)
On `phase-2-bakeoff-2backbone` (commits up to this doc), all teeth-verified:
- **Both backbones geometry-retired:** transformer-ar right_angle 0.9985 @110k (89M diagnostic);
  mamba-hybrid 1.0 @10k (locked mixer @48M/2048-ctx, not the full 53.7M; job 47275394).
- **53M rung LOCKED + param-match** (52,798,948 / 53,733,348, clean 1:7); both matrix YAMLs load + build
  the locked counts (confirmed on Leonardo).
- **Eval sharding WIRED** into `run_short` (auto under 4-GPU DDP) — the budget lever; **integration
  golden PASS on mamba-hybrid** (wired path via shared `score_cell`: 0 mismatches, 523/523 ragged,
  determinism; job 47390793; `_SHARDING_GOLDEN_PASS`).
- **Gen-side 4-tuple keying** (`cfm.eval.gen_realism`), tripwire-guarded (red-before-green).
- **29 MB parquet reference** (`reports/phase-2-bakeoff/real-features-2026-04-15.0.parquet`) reproduces
  the locked floor EXACT (265/265 vs sha `95abb88`).
- **Memorization-first hard-halt WIRED + PROVEN** (9/9 teeth; refuses a best-excess memorizer by name).

## 3. THE ONE OPEN ITEM — held-out CELL SAMPLER (next sub-project: the Lane-S realism eval's down-sampler)
**LANE CORRECTION (2026-06-20):** this is **NOT** a continuation of the 2026-06-08 eval-set-gen plan.
That plan built the SEPARATE **per-tile coherence** lane (shuffle-gap, per-city). The open item feeds the
**Lane-S / conditioning-floor** lane (`conditioning_floor.py` + `bakeoff_decision.py` + `gen_realism.py`).
The eval-set-gen tile manifest (Phase B) is its INPUT (which tiles are held out), not the thing to resume.

The real problem is a **budget-bounded stratified DOWN-sampler** (NOT a selector-from-scratch): the ~77k
held-out cells ALREADY cover the strata; the job is to pick a bounded subset to generate (the matrix is
6 runs × per-city generation) that keeps **≥ min_n = 50 generated features per floored 4-tuple stratum**.
Design constraints (carry into the sub-project — do NOT scope here):
- **Keep ≥ min_n = 50 gen features** per **(zoning, road_skeleton, cell_density, coastal) 4-tuple
  stratum** (so a `NO_DECISIVE_WINNER` is a true near-tie, not an under-sampling artifact). The **265
  floored strata are proven feasible** — munich (thinnest, 156 tiles) floors 61; the real side already
  clears min_n=50 there, so the sampler need only preserve coverage while bounding generation cost.
- **Frozen + sha-locked**, write-once, like the conditioning floor (`95abb88`) — a `_*_LOCKED` marker +
  a manifest the eval reads; reproducible from a committed regen script.
- **4 held-out cities, corrected counts** (523/579/156/601 usable TILES → a defined CELL subset ⊂ ~77k).
- **Re-derive the scored-matrix budget** at the chosen N (the ~20% figure is void).
- Then the held-out eval wiring is **mechanical** (all built): generate the sampled held-out cells →
  `gen_realism.gen_features_by_city` (4-tuple) → `lane_s_excess` vs the parquet → `decide`
  (memorization-first → power-gated worst-case → winner / `NO_DECISIVE_WINNER`).

## 4. STANDING GATES
- **No scored run, no merge without Umar's word.** The 6 matrix YAMLs are **gated drafts** (untracked,
  corrected comments, `eval_cells:1859` flagged PLACEHOLDER/DO-NOT-RUN) — the matrix is gated on the
  cell sampler above.
- **DEPLOY GATE (2026-06-20):** Leonardo is **7 commits behind** (HEAD `d8ea038` vs Mac `46ea757`); the
  wired path is present there only as LOOSE uncommitted edits and **`gen_realism.py` is not deployed at
  all**. The GPU markers are real, but **re-deploy Leonardo to the Mac's committed HEAD before any scored
  run** (full wired path incl. `gen_realism`, git-coherent).
- **GROUND_TRUTH §3 wins** on any held-out-count question. Verify counts at the precondition with a
  unit-correct method (tiles vs cells) — this blocker was exactly that lesson.
- **Auto-summary memory is NOT a verified source.** claude-mem obs 8262 ("cell-selection BUILT") and 8233
  ("cells" for tiles) are WRONG — haiku auto-summarizer; 8262 inverted its own Explore source. The store is
  read-only in worker runtime so they can't be deleted; verify against source / GROUND_TRUTH §3
  (see memory `feedback_auto_summary_not_verified_source`).
- Leonardo ops unchanged (see `leonardo_torch_import_and_fs_health`).
