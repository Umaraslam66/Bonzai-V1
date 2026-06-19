# LIVE BOOT DOC — Phase-2 bake-off: eval pipeline WIRED + verified; the ONE blocker is the held-out CELL SELECTION (eval-set-gen, next sub-project) (2026-06-19)

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

## 3. THE ONE OPEN ITEM — held-out CELL SELECTION (the next sub-project: eval-set-gen, resume)
eval-set-gen **Phase B (tile manifest) is DONE + locked**; resume it at the **cell selection** for the
realism eval. Design constraints (carry into the sub-project):
- **Power-sized + KS-resolvable** per **(zoning, road_skeleton, cell_density, coastal) 4-tuple stratum**
  at **≥ min_n = 50** generated features/stratum (so a `NO_DECISIVE_WINNER` is a true near-tie, never an
  under-sampling artifact). This is the eval-set-gen's reason to exist.
- **Frozen + sha-locked**, write-once, like the conditioning floor (`95abb88`) — a `_*_LOCKED` marker +
  a manifest the eval reads; reproducible from a committed regen script.
- **4 held-out cities, corrected counts** (523/579/156/601 tiles → a defined CELL set ⊂ ~77k).
- **Re-derive the scored-matrix budget** at the selected N (the ~20% figure is void).
- Then the held-out eval wiring is **mechanical** (all built): generate the selected held-out cells →
  `gen_realism.gen_features_by_city` (4-tuple) → `lane_s_excess` vs the parquet → `decide`
  (memorization-first → power-gated worst-case → winner / `NO_DECISIVE_WINNER`).

## 4. STANDING GATES
- **No scored run, no merge without Umar's word.** The 6 matrix YAMLs are **gated drafts** (untracked,
  corrected comments, `eval_cells:1859` flagged PLACEHOLDER/DO-NOT-RUN) — the matrix is gated on the
  cell selection above.
- **GROUND_TRUTH §3 wins** on any held-out-count question. Verify counts at the precondition with a
  unit-correct method (tiles vs cells) — this blocker was exactly that lesson.
- Leonardo ops unchanged (see `leonardo_torch_import_and_fs_health`).
