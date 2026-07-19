> ⛔ **SUPERSEDED (2026-06-19) by `docs/handoffs/2026-06-19-eval-set-gen-cell-selection-next.md`.**
> [B] (eval-sharding + per-city-KS) is now WIRED + verified. **The "1,859 held-out cells" below is the
> propagated tiles↔cells ERROR** — it is **1,859 usable TILES** (real cells ~77k); the held-out CELL
> SELECTION (eval-set-gen) is the live blocker. Read the new doc + **GROUND_TRUTH §3 (canonical)**, not
> the (historical) body below.

# LIVE BOOT DOC — Phase-2 bake-off: 53M LOCKED + golden fixed (both COMMITTED); T10 YAMLs drafted; [A]/[C]/[D]/[E] resolved; [B] (eval-sharding wiring + per-city-KS emission) OPEN (2026-06-18)

Supersedes `docs/handoffs/2026-06-18-t9-gate-53m-locked.md`. Branch **`phase-2-bakeoff-2backbone`**, UNMERGED.
**Do NOT commit, submit a scored run, or merge without Umar's word.**

## 0. CONSULT FIRST (in order)
1. **`docs/GROUND_TRUTH.md`** — canonical facts (units, account, corpus, locked design). **If anything
   here disagrees, GROUND_TRUTH wins.** Always state the unit (node-h vs GPU-h — they differ 4×).
2. **Memory** (`MEMORY.md` loaded each session): `project_bakeoff_53m_locked` (one-line state),
   `leonardo_torch_import_and_fs_health` (FS/torch ops), and the `feedback_*` lessons — esp.
   `feedback_no_marker_without_endstate_verify`, `feedback_verify_before_lock_not_after`,
   `feedback_gate_must_distinguish_regimes`, `feedback_tool_output_trustworthiness_layer`.
3. **Spec** `docs/superpowers/specs/2026-06-17-phase-2-bakeoff-2backbone-delta-design.md` §1A/§9;
   **Plan** `docs/superpowers/plans/2026-06-17-phase-2-bakeoff-2backbone.md` Task 10/11.
4. **THIS doc** for the live committed/uncommitted state and [B].

## 1. GIT STATE — COMMITTED vs UNCOMMITTED (verified 2026-06-18; a prior note "tip b46a963 / golden uncommitted" was STALE)
**Tip = `d8ea038`** (local Mac == `$WORK`), branch `phase-2-bakeoff-2backbone`. Two scored commits over `c4131ee`:
- **`b46a963`** — the **53M LOCK**: `src/cfm/models/bakeoff_scales.py` "53M" rung + `scripts/rederive_53m_ratio.py`
  + docs honest-relabel (GROUND_TRUTH/spec/plan) + boot-doc rename. COMMITTED, deployed to `$WORK`.
- **`d8ea038`** — the **golden mamba-hybrid fix**: `scripts/eval_sharding_golden.py` exercises mamba-hybrid
  first (+ rank-0 weight broadcast) and `.sbatch` comment. **COMMITTED** (not uncommitted), deployed to `$WORK`.

**UNCOMMITTED — local + `$WORK`, identical content (these are what exists OUTSIDE git):**
- `scripts/train_scaffold.py` (M) — **[A]** `--seed` flag (parser) + `("seed","seed")` in `build_config_from_args`.
- `scripts/bakeoff_run.sbatch` (M) — **[A]** optional `SEED` env → `--seed` pass + the 3-seed matrix-loop doc.
- `configs/experiments/bakeoff-{transformer-ar,mamba-hybrid}-53M.yaml` (??, untracked) — the **T10 DRAFT YAMLs** (§2).
- `docs/GROUND_TRUTH.md` (M) — only the "Live boot doc" pointer, repointed to THIS doc.
- `docs/handoffs/2026-06-18-bakeoff-t10-eval-wiring-open.md` (??, untracked) — THIS boot doc.

**`$WORK`-only scratch (untracked, NOT in local git, deletable):** `scripts/_avg_cell_len.{py,sh}` ([D]),
`scripts/_oom_probe.{py,sbatch}` ([E]), `scripts/_torch_probe.py` + `scripts/_work_recovery_poll.sh` (the
resolved recovery poller), `scripts/_gate_i.sbatch` (pre-existing). On-disk records under
`reports/phase-2-bakeoff/`: `_SHARDING_GOLDEN_PASS`, `_avg_cell_len.out`, `_work_recovery.log`. `$SCRATCH`
holds a fresh torch+mamba insurance venv (`/leonardo_scratch/.../bakeoff_insurance`, harmless leftover).

**The sharding-golden PASS** (`reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS`, job 47258766) rests on
COMMITTED code (`d8ea038`): both backbones 0/523 mismatches, 523/523 ragged-conserved, determinism True.

## 2. [A]/[C]/[D]/[E] — RESOLVED (measured / proven)
- **[A] seed mechanism — DONE + PROVEN (uncommitted code above).** `--seed` flows to `cfg.seed`; proof on a
  fresh process: model-init same-seed **bit-identical** / diff-seed **differ**; train/val tile-split same/diff;
  data sampler order same/diff — i.e. the seed drives ALL run RNG. Matrix launch = loop SEED∈{7,13,23} ×
  2 backbones over the 2 YAMLs (6 scored runs); each seed → distinct checkpoint dir.
- **[C] region / per-city KS — RESOLVED as a DECISION (emission rides [B]).** region = report/floor **TAG only**,
  does NOT filter eval. The eval scores all **1,859** held-out cells, partitions by city, emits **PerCityKS for
  all 4** (glasgow/eisenhüttenstadt/munich/krakow). The per-city-KS LAYER already exists:
  `cfm/eval/multiregion_realism.per_city_ks`/`decision_ks` → `cfm/eval/city_aggregate.PerCityKS` →
  `bakeoff_decision`. Matrix = **6 train runs** (2 bb × 3 seeds), each scoring all 4 cities.
- **[D] max_steps — MEASURED = `112,563`.** avg cell BODY = **596.7 tok** over **941,969** train cells;
  `(20 × 53,733,348) / (16 × 596.7) = 112,563` (r=20 → ~1.07B tokens → **1.72× reuse** of 623.9M). Lands next
  to the diagnostic's 110k. (seq-incl variant 110,708; YAMLs use the body number.)
- **[E] OOM — NO OOM, large headroom.** batch 1 / seq 13322 fwd+bwd peak: **tf 5.6 GB, mamba 6.8 GB** on a
  64 GB A100 → eff-batch-16 (batch 1 × 4 dev × 4 grad_accum) fits with ~57 GB headroom; **no activation
  checkpointing needed**. (Headroom allows a larger batch for speed, but eff-16 is the proven/measured config.)
- **T10 DRAFT YAMLs** (`configs/experiments/bakeoff-*-53M.yaml`) carry: `region: krakow` (tag),
  `train_set: eu-train-union`, `max_len: 13312`, `eval_max_new: 13312`, `eval_cells: 1859`, `batch_size: 1`,
  `grad_accum: 4`, `max_steps: 112563`, `compile: false`, `seed: 7` (matrix via SEED→--seed). Verified loadable
  by the strict `_load_config_yaml` and they build the locked 53M counts (52,798,948 / 53,733,348).

## 3. [B] — THE ONE OPEN ITEM (read-only trace FIRST; do NOT wire/build until Umar's word)
"eval-sharding on" = wire `cfm.eval.shard.sharded_eval` into `run_short`'s post-train eval (rank-0-serial
today: `if not trainer.is_global_zero: return` then a serial loop) so all 4 ranks shard the GENERATION →
gather → rank-0 decode/score; **+** emit per-city KS ([C]); **+** an INTEGRATION golden (the WIRED path's
per-cell tokens == rank-0 baseline on mamba-hybrid — not just the isolated `sharded_eval` primitive already
proven); **+** a GPU re-verify.

**[B]'s FIRST STEP is a READ-ONLY TRACE, not wiring.** Determine what **scalar realism metric per cell** and
what **reference distribution** feed `per_city_ks` today:
- Read `cfm/eval/multiregion_realism.py` (`per_city_ks`/`decision_ks` consume `generated_by_city`,
  `real_by_city`, `metric`), `cfm/eval/realism.py` (the two-sample KS), and trace how `real_by_city` (the
  eval-set reference) is produced/loaded, and what per-cell scalar `metric` is.
- **IF defined + produced by a real harness** → [B]/[C] is **mechanical integration** (point the sharded
  generation at it; wire emission; integration-golden; GPU re-verify).
- **IF the metric/reference is UNDEFINED or AMBIGUOUS** → it **collides with `model_vs_real_effect` (spec §7)**,
  a **brainstorm-with-Umar item** (a conditioning-echoing model must FAIL it), NOT a solo wire.
- **Report WHICH, then STOP. Do not wire or build until Umar says.**

## 4. STANDING GATES (every session)
- **No commit, no scored run, no merge without Umar's word.** The uncommitted [A] code + the YAML drafts
  await his word; the 6-run matrix is gated on [B] + the sharding golden (already PASS).
- **GROUND_TRUTH wins** on any conflict; if a fact here looks wrong, re-derive it from its source.
- **Verify by RECOMPUTATION**, never exit code / a bare marker (F8): a PASS must rest on on-disk recomputed
  evidence (the sharding golden + param-match both do).
- **Budget stakes on [B]:** rank-0-serial eval = **81%** of the 5,000 GPU-h grant for 3 seeds; 4-GPU sharded
  = **~20%**. [B] is what makes the locked budget true — it is not optional polish.
- **Leonardo ops:** torch import ~60–80s + a GPU-less interpreter-teardown hang is the NORMAL baseline (NOT
  `$WORK`-degraded — judge FS health by `dd` throughput); `mamba_ssm` needs the gcc-12 libstdc++ `LD_PRELOAD`;
  capture counts then `os._exit(0)`. Long login-node work goes in tmux (ssh drops mid-run). See memory
  `leonardo_torch_import_and_fs_health`.
