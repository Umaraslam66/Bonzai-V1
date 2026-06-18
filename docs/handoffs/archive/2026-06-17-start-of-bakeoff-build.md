> SUPERSEDED — live boot doc is docs/handoffs/2026-06-18-t9-gate-50m-locked.md (canon: docs/GROUND_TRUTH.md). Archived 2026-06-18.

# START OF 2-BACKBONE BAKE-OFF BUILD — boot here (2026-06-17)

Thin handoff. Pointers, not summaries — read the spec + plan, don't re-derive them.

## STATE
- Branch **`phase-2-bakeoff-2backbone`**. Plan **APPROVED** (`aaf8893`), spec
  (`9775a8d` + `deff870`), off **main `fb3828a`** (suite green 1771/1xfail).
- Allocation **`AIFAC_P02_548`** live (boost_qos_bprod/dbg; lrd_all_serial works).
  Repo on Leonardo: `/leonardo_work/AIFAC_P02_222/Bonzai-OSM` (deploy via git bundle;
  `$WORK=/leonardo_work/AIFAC_P02_548`; artifacts still under the 222 tree, RW).
- **NO MODEL SCORED YET. Nothing in Phase 1–4 executed.** This is the build's first step.

## THE PLAN (authoritative — execute task-by-task, don't re-summarize)
`docs/superpowers/plans/2026-06-17-phase-2-bakeoff-2backbone.md` — 10 tasks / 4 phases.
Spec: `docs/superpowers/specs/2026-06-17-phase-2-bakeoff-2backbone-delta-design.md`.
Protocol: `docs/protocols/sub-project-planning-protocol-v3.md`. Executor:
superpowers:subagent-driven-development (Umar = chat-reviewer, directs execution).

## THE APPROVED GATES (carry verbatim)
- **T5 param-match = a VERIFIED gate** — count the ACTUAL built model's params, assert
  transformer-ar vs mamba-hybrid match **≤2% per scale** {30/100/300M/1B}. Tune the mixer
  to the param target. NEVER eyeball the mapping (equal-depth ≠ equal-capacity).
- **T7 smoke STOPS for Umar's word** — compile verdict: keep compile-ON iff **recompiles
  plateau ≤~10 AND compile overhead <10%** on the REAL cell-length distribution; else
  `--no-compile`, record the finding.
- **T9 diagnostic STOPS for Umar's word** — present the eval-DOMINATED budget (13,312 ≈
  6.5× the AR-gen cost) vs **5000 node-h** BEFORE any scored commit; ladder-collapse →
  `FIXED_SCALE_PLUS_S13` (handled, not a surprise).
- **T10 scored = its own word** — re-planned from T7 + T9 outputs (not pre-pinned).

## NEXT ACTION
Phase 1 / **T1**: gcc/12.2.0 + `CC/CXX/CUDAHOSTCXX` + `LD_PRELOAD` libstdc++ into
`scripts/bakeoff_run.sbatch`; content-contract test; `sbatch --test-only` dry-verify on
548 (NO real submit). Then **T2 golden-freeze FIRST** (capture pre-refactor MicroAR
forward+loss) BEFORE the T3 extract-base refactor. Then **Phase 2 ∥ Phase 3** per the
parallel graph (diagnostic runs on the existing transformer-ar, no mamba dependency).

## DISCIPLINE CARRIED
Behavior-preservation = **bit-identical golden tensor** (capture before touching MicroAR).
Verified-end-state by recomputation. Stop-before-commit on gates/forks. **STOP at the
gated GPU steps (T7/T9/T10) for Umar's word.** No scored hour / merge / push without
Umar's word + `--no-ff`. Main is shared — never force-push. Predecessors discharged:
18.5 resume PASS, (b) mamba-lock GPU verify-before-lock PASS + env_lock extended (merged).
