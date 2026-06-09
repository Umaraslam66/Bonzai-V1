# Handoff — START of bake-off PHASE D (2026-06-09)

You are a fresh, context-free agent picking up the Phase-2 bake-off at the **Phase C/D boundary**. Phases A–C (all local) are **done, committed, and green**. Phase D (Tasks 8–12) is **all Leonardo** and is **parked on Umar's allocation word**. This is a thin **pointer** doc — the detail lives in the artifacts below; read them, don't reconstruct.

---

## ⛳ STATE (verified against git, not memory)

- **Branch `phase-2-bakeoff` @ `e5cbe9e`** — local-only, **NOT pushed**. `main` **untouched @ `738a8fa`**. Full suite **1317 passed**.
- **Phases A–C COMPLETE.** Read first:
  - **`reports/2026-06-09-phase-2-bakeoff-phases-A-C-local-build.md`** — the interim summary: commits per phase, each gate's proof + reproduced red-on-divergence, the 3 carry-forwards, branch/main state.
  - **Plan** `docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md` (content sha `39d56c7…`) — **Tasks 8–12 are Phase D**; carries the 3 carry-forward notes inline.
  - **Spec (authoritative)** `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md` (content sha `cb4a906…`).
  - The prior **execution handoff** `docs/handoffs/2026-06-09-start-of-bakeoff-execution.md` — the deeper subagent-driven rationale (still applies).
- **Nothing in Phase D has run.** No Leonardo job, no shard build, no GPU, no allocation assumed.

## 🧩 The 3 carry-forwards (honor in Phase D — recorded in the committed plan)

1. **Rec-2 (Task 6 → Task 12):** the dict→list seam. `decision_ks` already localizes `worst_case_city(list(per_city.values())).ks`; Task 12 assembles `{backbone: list(per_city_ks(...).values())}` for `binding_city_verdict`.
2. **munich power-floor (Task 9):** munich (n=156) has the highest #21 floor (~0.1087) → **most likely demoted-for-under-power at first model**; the munich→manchester swap reserve may fire on **POWER** grounds. By design, not a defect — don't be surprised, don't pre-empt.
3. **Task-12 input-completeness precondition:** the held-out exclusion is an **active filter** (the 4 are `validated: true` in the G4 roll-up, removed by `train_cities`, NOT structurally absent) → a held-out city can slip through an upstream gap, so Task 12 must **assert `set(real_by_city.keys()) == {eisenhuttenstadt, glasgow, krakow, munich}` and fail loud.**

## 🚦 Gating sequence (hold it in order)

1. **Umar's allocation word.** Phase D is parked until it. Nothing GPU/shard-build runs before it.
2. **Before any scored GPU-h — bring for review (not after):** a real **core-h → GPU-h conversion** from the Leonardo boost-node spec (NOT core-h ≈ GPU-h), and a **diagnostic/cheapest-first ordering** inside the guaranteed pre-11 window.
3. **Task 8** — build the EU multi-region train shards (Leonardo CPU). **Verified-end-state:** count built city dirs == `len(train_cities)` (≈38 = 42 validated − 4 held-out), sum `n_training_tiles`, assert **no held-out city dir** was built. Never trust the job exit code.
4. **Task 9** — the Task-1 diagnostic (GPU): measures `r` (+CI), emergence floor, per-scale eval-cost, and the **two gate inputs**. **Two HALT-gates:** (a) **conditioning-discrimination** — feed input (i), a **DATA property of REAL held-out tiles (model-INDEPENDENT, never pilot generations)**, to `conditioning_discrimination_gate`; fail → **T5 reopens** before any scored run; (b) **∅-ladder** — `feasible_ladder_conservative(r_ci_high)`; if `escalate_more_data` → **escalate to more-data**, no scored runs feasible. Persist measurements to a `reports/` YAML; re-read before proceeding.
5. **Task 10** — resolve the fork deterministically off the measurement: `ladder = feasible_ladder_conservative(r_ci_high, epoch_factor=E)`; `basis = decision_basis(len(ladder.feasible))`. **Persist BOTH**; assert the configured scale set **== `ladder.feasible`** (anti-hardcoding teeth).
6. **Task 10.5** — `mamba-ssm` verify-before-lock **HALT-gate**, BEFORE any scored GPU-h: import + fwd/bwd under the exact locked torch stack; any torch/CUDA drift → **re-lock-all** (every backbone re-runs under the new lock).
7. **Task 11** — scored runs at the determined scales × {transformer-AR, mamba-hybrid, discrete-diffusion}, identical `E`, across-job `$WORK` resume; verified-end-state per run (checkpoint sha + eval artifact).
8. **Task 12** — decision + report: **assert the path == the persisted `basis`** from Task 10; honor the input-completeness precondition; write the `reports/` summary + PRD update. **Merge to `main` only on Umar's word, `--no-ff`, suite green.**

## 👤 Two Umar-calls riding into Phase D (NOT pre-resolved)

1. **Worst-case vs mean** aggregation (delta-spec §4) — **recommendation = worst-case**; gated on the Task-9 conditioning-discrimination diagnostic (which can reopen T5).
2. **Allocation / timing** — `AIFAC_P02_222` soft-ends 2026-06-11 (~36k/40k core-h remain; same-size top-up ~1–2 d post-expiry → worst case a short PAUSE during refill, not lost work). Align checkpoints to run boundaries.

## 🧭 Execution discipline (carried forward, non-negotiable)

- **Subagent-driven, NOT inline** (fresh subagent per task, implementer ≠ reviewer, Umar = chat reviewer; subagents forbidden new branches/push/PR/merge). The halt-gates only mean something if the agent that *hits* a gate isn't the one that *decides to honor it*. Batch the review only for pure functions; never collapse execution.
- **Stop-before-commit on every HALT-gate and design fork** — bring the teeth (fire-on-bad / pass-on-good + red-on-divergence) to the reviewer before the commit. The whole of Phases A–C was held to this.
- **Verified-end-state, never exit codes** — re-read the artifact / recompute the sha / count real units. (This lineage has caught stale-summary mismatches repeatedly.)
- **Gate 2 (pre-dispatch audit):** READ the current module signatures against SOURCE before any existing-module edit — the EU/SG seam bit Phase A twice.
- `ruff check`/`format` **unpiped** before commits; `uv sync --extra dev --extra training` before pytest.
- **No push/merge without Umar's explicit word + `--no-ff`.** Never force-push / rewrite `main`.

## 🏗️ Infra (Leonardo, CINECA, `AIFAC_P02_222`)

- **Leonardo is SSH-ready** (Umar re-authenticated the user ControlMaster socket; `Host leonardo`, user `uaslam00`). **SSH-ready means the agent CAN execute on Leonardo — it does NOT mean Phase D starts.** Phase D still requires **(a) Umar's allocation word AND (b) the owed core-h→GPU-h conversion reviewed first.** Two separate signals; neither implies the other.
- Repo on Leonardo: `/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, deployed via **git bundle** (no GitHub creds on the login node). Corpus frozen at `data/processed` (release `2026-04-15.0`, DERIVATION 1.2). **The EU corpus is Leonardo-only** (local = Singapore + the frozen holdout manifest). Use `tmux` for login-node monitoring; Slurm batch jobs are disconnect-safe.

## ▶️ One-liner to start the Phase-D session

> Phase-2 bake-off **Phases A–C COMPLETE + committed** (branch `phase-2-bakeoff` @ `e5cbe9e`, local-only; `main` @ `738a8fa` untouched; suite **1317 green**) — read `reports/2026-06-09-phase-2-bakeoff-phases-A-C-local-build.md` + the committed plan (**Tasks 8–12 = Phase D**; 3 carry-forwards recorded). **Phase D is ALL Leonardo, PARKED on Umar's allocation word.** Gating: allocation word → **bring core-h→GPU-h conversion (real boost-node spec, NOT core-h≈GPU-h) + diagnostic/cheapest-first ordering for review** → Task 8 (build EU shards, CPU; verified-end-state: city-dir count == `len(train_cities)`, no held-out dir) → Task 9 (diagnostic: `r`+CI/floor/eval-cost; **TWO HALT-gates** — conditioning-discrimination [input (i) = DATA property of REAL held-out tiles, model-INDEPENDENT] fail→T5 reopens; ∅-ladder→escalate) → Task 10 (resolve fork: `feasible_ladder`→ladder, `decision_basis`→basis, **persist BOTH**, assert scales==`ladder.feasible`) → Task 10.5 (`mamba-ssm` verify-before-lock HALT-gate; torch drift→re-lock-all) → Task 11 (scored runs, identical `E`, `$WORK` resume) → Task 12 (assert path==persisted basis; input-completeness precondition). Two Umar-calls ride in: **worst-case-vs-mean (rec=worst-case)**, allocation/timing. Execution: **subagent-driven NOT inline** (halt-gates need implementer≠reviewer; batch review only for pure fns), **verified-end-state never exit codes**, **stop-before-commit on halt-gates/forks**, ruff unpiped, `uv sync --extra dev --extra training`. **Leonardo SSH-ready** (`Host leonardo`, user `uaslam00`; repo `/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, deploy via git bundle) — **SSH-ready means CAN execute, NOT that Phase D starts** without the allocation word + the owed conversion reviewed. **No push/merge without Umar's word + `--no-ff`; never rewrite `main`.** Read this handoff + the reports A–C summary + the plan (Tasks 8–12) first.
