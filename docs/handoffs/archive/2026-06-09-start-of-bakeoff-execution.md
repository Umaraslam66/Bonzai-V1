> SUPERSEDED — live boot doc is docs/handoffs/2026-06-18-t9-gate-50m-locked.md (canon: docs/GROUND_TRUTH.md). Archived 2026-06-18.

# Handoff — START of bake-off EXECUTION (2026-06-09)

You are a fresh, context-free agent about to **execute** the Phase-2 bake-off delta-reconciliation plan. The brainstorm → spec → plan is **DONE and signed off**; nothing has been executed. This file is forward-looking: what's on disk, how to run it, the gates that can rewrite the work, and what you must NOT relitigate.

---

## ⛳ STATE (verified against git, not memory)

- **main @ `c50c1ff`** (the START-of-bakeoff handoff). eval-set-gen was **merged at `0833ac7`** (one commit back), `--no-ff`, **local, NOT pushed**. (If a summary says "main @ 0833ac7," it's stale — main advanced one commit. Re-read from disk.)
- **`phase-2-bakeoff` @ `605d4b8`**, **+16 / −112 vs main**, merge-base `7910ce1`. **Never rebased onto main.** Holds the 2026-06-02 bake-off spec/plan + ~11 CPU tasks (backbone/micro_ar/diffusion, curve/emergence/feature_resolution/geometry, lit_module/resume).
- **Spec + plan are SIGNED OFF and sit UNCOMMITTED on the working tree** (currently checked out on `main`):
  - Spec: `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md` — content sha `eb03da91b99b8524e989bd7929cf98c3ebfb1257`
  - Plan: `docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md` — content sha `0324840413e2e1a17909cffc9d3086878a87d7e5`
  - Both show as `??` (untracked). **They are not missing** — they land on `phase-2-bakeoff` at **plan Task-1 Step-6**, after the rebase. Do not re-create them; inherit them from the working tree.
- **Nothing executed.** No rebase, no code, no GPU, no branch op.

## 📜 The contract

Execute the **plan EXACTLY**, task by task. The **spec is authoritative**; on any plan-vs-spec conflict, **the spec wins — STOP and ask**, do not improvise. Verify the two content shas above before you start (`git hash-object <path>`); a mismatch means the artifact drifted — STOP.

## ▶️ Execution mode — SUBAGENT-DRIVEN (NOT inline)

Use **superpowers:subagent-driven-development**: a **fresh subagent per task**, **two-stage review between tasks**, **implementer ≠ reviewer**. Umar is the chat reviewer (outer loop). **Subagents are forbidden: new branches, push, PR, merge** (state this in every dispatch — they improvise otherwise).

**Why NOT inline (do not "simplify" to inline):** the plan's entire value is its **halt-gates** (vacuous-green alarm, conditioning-discrimination gate, ∅-ladder, mamba-lock). A gate only means something if the agent that *hits* it is not the agent that *decides to honor it* — inline collapses implementer and reviewer into one agent invested in the green, which is exactly the setup where a vacuous pass gets rationalized. If round-trip friction is the worry, **batch the REVIEW** (group Phase-B's pure-function tasks into fewer review points); never collapse the execution model.

## 🚦 Gating sequence (hold it in order)

1. **Umar's explicit go for Task 1** — it is the **rebase** (the first branch op). **Nothing runs before it**, not even Phase A–C code.
2. **Phases A–C** (rebase + obligation-(a) re-point; the Phase-B rule functions; the T5/T6 net-new builds) — executable **after** that go. All local; no GPU.
3. **Phase D** (Leonardo: shard build, Task-1 diagnostic, scored runs, decision) — **parked on Umar's SEPARATE allocation word.** Do not sequence any GPU work until he gives it.

## 🍴 The fork (why the plan is not a linear list)

The **Task-1 diagnostic gates everything downstream.** The ladder of scales and the decision basis are **OUTPUTS of the Phase-B rule functions** applied to the measured `r` — **never hardcoded.** Plan Task 10 asserts the configured scale set == `feasible_ladder(...).feasible` AND that Task 12's decision path == the persisted `decision_basis(...)`. Two **HALT-gates can rewrite the downstream** — neither is "run and note":
- **Conditioning-discrimination gate (Task 9):** if same-macro-stratum cross-city feature-KS (a **DATA property of REAL held-out tiles — model-independent**, never from pilot generations) exceeds tolerance → **HALT; T5 reopens before any scored run** (the worst-case bar is invalid).
- **∅-ladder (Task 9):** if even 30M can't clear the on-frontier rule → **HALT; escalate to more-data**, no scored runs are feasible.
- **mamba-lock (Task 10.5):** `mamba-ssm` fwd/bwd must run under the exact locked torch stack **before any scored GPU-h**; torch drift → **re-lock-all**.

## 🔒 LOCKED — do NOT relitigate (full list in delta-spec §9)

- **Reading A:** coherence scores the MACRO plan; the bake-off emits sub-F MICRO tokens only → **Phase-2 decides on KS-realism ALONE**; coherence + the §7 power gate are Phase-3.
- **r→ladder (Rule 1) + decision-basis (Rule 2)** as pre-committed functions; conservative boundary rounding; `transformer-AR` §13 tie-break; 1B drops by rule.
- **Branch = REBASE** + the 4-step red-before/green-after opener (GREEN-before-(a) is the vacuous-green **alarm/HALT**).
- **T5 = worst-case generalization bar** (per-city, munich INCLUDED — KS has no null to saturate; #21 binding-city power gate against the city's OWN `C/√n`; pooling rejected) — **gated on the conditioning-discrimination diagnostic**.
- **T6 = build-shape (a)** (per-city manifests + datamodule union; train-city source = G4 roll-up minus held-out-4; region-level whole-city exclusion primary; schema 1.0; CRS-agnostic).
- **Phase-3 parking:** coherence, `assert_coherence_power_sufficient`, obligation (b) `model_vs_real_effect`, obligation (c) munich→manchester swap, macro-planner preview.
- From eval-set-gen (delta-spec §9): 4-city held-out set; all-moderate density cut; density-coherence dropped (perplexity_gap is a correct non-leak); #13+#22 bundled hard gate; munich dense-core #21 stay-and-record.

## 👤 Two Umar-calls riding into Phase D (NOT pre-resolved)

1. **Worst-case vs mean** aggregation (delta-spec §4) — **recommendation = worst-case**; mean is the documented alternative. Changes which architecture can win.
2. **Allocation / timing** — see Infra.

## 🧭 Execution discipline (carried forward)

- **Halt-gates HALT.** On a gate failure or first plan defect, **stop and report to the reviewer** — never improvise a fix or inline it (protocol-v3 Gate 4; `feedback_subagent_branch_pattern`).
- **Verified-end-state, never exit codes.** Re-read the artifact / recompute the sha / count real units. **THIS session caught two stale-summary mismatches** (the start-of-bakeoff handoff's "bake-off = ~4,800 GPU-h" was the *whole envelope* not the slice; "main @ 0833ac7" was actually `c50c1ff`) — so **re-read from disk; do not trust summaries, memory, or handoff prose as ground truth.** (4 prior false-DONEs this project.)
- **ruff check/format unpiped** before commits (don't swallow exit codes); `uv sync --extra dev` before pytest.
- **Gate 2 (pre-dispatch audit):** before any existing-module edit, READ the current module and verify the signatures the plan names still hold — plan snippets may have drifted.
- **Order:** Phase B (local rule functions, TDD) before the Leonardo phase. **No push/merge without Umar's word + `--no-ff`.** Never force-push/rewrite `main`.

## 🏗️ Infra (Leonardo, CINECA, `AIFAC_P02_222`)

- **Bake-off compute = ~1,500 GPU-h slice** (PRD §10) — **NOT 4,800** (that's the whole 1,200-node-h envelope). The shrunk ladder will likely use less.
- **Allocation soft-ends 2026-06-11** but ~**36k / 40k core-h remain** and a **same-size top-up lands ~1–2 days post-expiry** → worst case a short **PAUSE during refill, not lost work**. Align checkpoints to run boundaries.
- **Owe a real `core-h → GPU-h` conversion** from Leonardo boost-node cores-per-A100 (NOT core-h ≈ GPU-h) before any GPU sequencing; order diagnostic/cheapest-first inside the guaranteed pre-11 window.
- **`eval-set-gen-wt` clone still on disk** (`/leonardo_work/AIFAC_P02_222/eval-set-gen-wt`) — cleanup pending Umar's word. Corpus frozen at `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed` (release `2026-04-15.0`, DERIVATION 1.2). **EU corpus is Leonardo-only** (local = Singapore + the frozen holdout manifest) — T5/T6 design+test locally on synthetic fixtures; the real multi-region build/diagnostic run on Leonardo. SSH: user-authed ControlMaster socket `Host leonardo` (re-auth on laptop sleep).

## ▶️ One-liner to start the execution session

> The Phase-2 bake-off delta-spec + plan are SIGNED OFF and **uncommitted on the working tree** (spec `eb03da9…`, plan `0324840…`; they land on `phase-2-bakeoff` at plan Task-1 Step-6). main @ `c50c1ff` (eval-set-gen merged `0833ac7`, not pushed); `phase-2-bakeoff` +16/−112, never rebased. Execute the plan EXACTLY (spec wins, STOP-and-ask) **subagent-driven, NOT inline** (halt-gates need implementer≠reviewer; batch the review, don't collapse execution). Gating: (1) Umar's go for Task 1 (the rebase — first branch op, nothing runs before it); (2) Phases A–C after; (3) Phase D parked on Umar's allocation word. The plan is a **fork on the Task-1 diagnostic** — scales + decision basis are rule-function OUTPUTS, never hardcoded; two HALT-gates rewrite downstream (conditioning-discrimination fail → T5 reopens; ∅-ladder → escalate). Do NOT relitigate delta-spec §9 (Reading A / KS-only Phase-2 axis, the r→ladder rules, rebase+red-before/green-after, worst-case bar [gated on the conditioning diagnostic], build-shape (a), Phase-3 parking). Two Umar-calls into Phase D: worst-case-vs-mean (rec=worst-case), allocation/timing. Verified-end-state never exit codes (this session caught "4,800 GPU-h"=envelope-not-slice and "main@0833ac7"=actually c50c1ff — re-read from disk). Bake-off slice ~1,500 GPU-h; allocation soft-06-11 + top-up. No push/merge without Umar's word + `--no-ff`. Read this handoff + the delta-spec + the plan first.
