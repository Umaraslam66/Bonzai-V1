# Handoff — eval-set-generation: design + plan LOCKED, execution NOT started (2026-06-08)

You are a fresh, context-free agent. You inherit ONLY what is written here. Read this whole
file, then the spec and plan it points to, before doing anything. Nothing we locked may be
silently re-decided.

---

## ⛳ STATE

- **`main` @ `8d46217`**, pushed to origin (`github.com/Umaraslam66/Bonzai-V1`), `origin/main ==
  local main` (verified). Design **spec** + implementation **plan** are committed and pushed.
- **Nothing is built.** No eval code, no manifest, no merge. Execution has not started.
- The eval-set-generation sub-project went through a full brainstorm (T1–T7) → spec → plan,
  with **reasons preserved alongside conclusions** (the spec's whole point). Your job is to
  **execute the plan, implementing the spec EXACTLY — nothing added, nothing reinterpreted.**

## 1. Contract pointers (the EXACT documents to implement)

- **Spec:** `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md` (blob `7769b147`).
- **Plan:** `docs/superpowers/plans/2026-06-08-eval-set-gen.md` (blob `5714072b`), 12 tasks
  (T1–T12, incl. T8.5), TDD, gating teeth.
- Both at `main @ 8d46217`. The spec header forbids asserted facts (every claim traces to a
  file:line or a dated measurement); the plan implements the spec section-by-section. If a plan
  step seems to disagree with the spec, the **spec wins** — STOP and ask, do not reinterpret.
- Operating discipline: `docs/protocols/sub-project-planning-protocol-v3.md` (six gates; §9
  construction-identity exclusion; §10.1 write-once sizing; §10.2 relative thresholds).

## 2. LOCKED decisions — do NOT relitigate (each with its one-line reason)

- **Held-out set = `glasgow / eisenhuttenstadt / munich / krakow`** (whole-city). Rule: a
  held-out city's morphology, density, **country**, AND CRS-zone must each remain in train —
  *reason:* a miss must read "didn't generalize," never "never saw this national style" (an
  unrepresented country makes a miss ambiguous and destroys the metric).
- **All-moderate density cut** — the held-out set spans 4 morphologies but only `moderate`
  density. *Reason:* no PRD dense-core-primary target; full per-morphology dense-core coverage
  is infeasible under the 550M train floor (rotterdam, the only admissible modernist dense-core
  city, is 44.6M). It is a **scope cut, revisitable at a phase transition** (project-wide
  principle, PI-ratified — NOT a protocol §10.1 clause), **not a permanent blind spot**.
- **Density-coherence term DROPPED** — *reason:* the conditioning vector hands the model the
  **per-cell `cell_density_bucket`** (the arrangement itself), so a shuffle-gap coherence score
  on it is circular (shuffle cancels the marginal, not a handed-over arrangement). Zoning &
  skeleton stay (tile-MODE conditioned → clean). **PAIRED NON-LEAK — do NOT "fix" it:**
  `perplexity_gap` treats the SAME field oppositely and correctly — it grades FIXED held-out
  tokens under matched-vs-shuffled conditioning (a use-test; conditioning is the independent
  variable, not a leak). Memory `feedback-shuffle-gap-marginal-not-arrangement`.
- **Cross-tile seam coherence DEFERRED, and NEVER an architecture bar** — *reason:* cross-tile
  connectivity is produced by the **rules-based stitcher (PRD §4), identical across every
  architecture**, so it cannot discriminate architectures. It VALIDATES a non-learned component
  (a one-time check when sub-E tile-to-tile semantics land). Do not let any future round
  resurrect it as a per-architecture bar.
- **Within-regime ≠ PRD §9.113** — v1's held-out test (country/zone in train) is
  **necessary-but-not-sufficient** for §9.113's cross-region foundation-model claim. v1 does
  NOT close the headline claim; literal region-holdout (cross-region/cross-national
  extrapolation) is a separate, separately-reported probe.
- **#13 + #22 are ONE bundled HARD GATE** — admin_region (all-None EU) AND
  morphology_class/country/climate_zone (SG constants) must be fixed in the **SAME single corpus
  re-derive, never separately** (the partial-fix trap), before any value-bearing conditioning.
  `docs/known_issues.md` #13/#22.

## 3. Two DORMANT obligations that FIRE at the bake-off (must NOT be orphaned at sub-project close)

These are wired now but activate only when a trained model exists. **Carry them forward into
the bake-off setup explicitly** — they are the part a closing sub-project most easily drops.

- **T8.5 — datamodule re-point:** the EU bake-off datamodule MUST be constructed with
  `multiregion_holdout_manifest_path("2026-04-15.0")`. A **fail-closed schema-2.0 assertion** in
  `holdout_guard.run_holdout_audit` backstops a forgotten re-point (it refuses to audit the EU
  corpus against a non-multi-region manifest → fails loud, never silently leaks).
- **T12 — coherence power gate:** `assert_coherence_power_sufficient` fires at the **first-model
  checkpoint** (it consumes the model-vs-real coherence effect size, which does not exist
  pre-model). If munich's usable-n (**156**) cannot resolve the architecture-distinguishing
  effect, it fires the **munich→manchester swap** — a **deliberate re-lock** of the write-once
  EU eval set (write-once-per-version). The swap is in reserve; the pre-model-detectable
  catastrophe was already ruled out (munich 156/171 usable = 91%, healthy).

## 4. Execution discipline a fresh agent MUST inherit

- **Teeth are gating HALT-steps, not reporting.** T8 (leak-guard 4-case) and T10 (coherence
  3-way) construct the failing case, prove it trips, then prove the clean case passes — and
  **HALT on failure; do not proceed** to the guarded thing. (Case B proves the city-guard is
  non-redundant; disconnected-loops proves the fragmentation term is non-redundant.)
- **Phase B before Phase C.** The manifest + its §2.2 correct-by-construction assertions
  (held-out∩train=∅; tiles **match frozen-corpus tile set** — NOT "fully enumerated", since
  munich is inner-core #21) land and verify FIRST. The leak guard reads the manifest's
  whole-city *declaration*; a faithful guard over a wrong declaration self-passes and still
  leaks. Do not reorder.
- **Verified end-state, never an exit code.** Every unattended step (manifest freeze, usable-n,
  reference, the real-run audit) re-reads the artifact from disk + recomputes the sha / matches
  G4 before being "done." *This phase saw 3 prior false-DONEs from trusting control flow over
  cluster/disk state — a false marker poisons future sessions.*
- **Branch `phase-2-eval-set-gen` off `main`.** Implementation code on the branch; the design
  docs stay on main. **No push / no PR / no merge without Umar's explicit word, and `--no-ff`.**
  Never force-push or rewrite main.
- **Destructive re-derives** (none expected this sub-project) use
  `scripts/multiregion/guarded_rederive.py`, never a hand-rolled loop.

## 5. Open state

- **DONE:** brainstorm (T1–T7) → spec → plan, all locked + pushed (`main @ 8d46217`). Held-out
  set chosen + usable-n measured (read-only): glasgow 523 / eisenhuttenstadt 579 / munich 156 /
  krakow 601 (of 549/616/171/616). Coherence discrimination probe (read-only) confirmed
  real-vs-shuffle (pooled d≈0.8). #21 (munich inner-core) + #22 (SG-constant leak, bundled #13)
  recorded.
- **NOT STARTED:** no eval code, no `macro_graph`/`coherence`/`usable_tiles` modules, no manifest
  built or frozen, no de-Singapore edits, no merge.
- **Execution starts at Task 1, Step 0** — re-verify `paths.py` current state from source first
  (a memory observation claims the per-region CRS label was already added; **source as of this
  handoff says `_EPSG_LABEL="EPSG3414"` is still hardcoded** — trust source, not the memory).

## 6. Execution mode

**Subagent-driven** (`superpowers:subagent-driven-development`): a **fresh subagent per task**,
two-stage review **between** tasks. This preserves the implementer/reviewer **adversarial
separation** — the implementing subagent does NOT self-review as reviewer. **Umar is the chat
reviewer.** Each subagent dispatch must explicitly forbid new branches / push / PR (subagents
improvise otherwise — memory `feedback_subagent_branch_pattern`).

---

## 🖥️ Infra (Leonardo, CINECA, account `AIFAC_P02_222`)

- Corpus (frozen, DERIVATION 1.2, release `2026-04-15.0`):
  `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed`. **Do not reopen it.**
- **Hybrid execution:** code + synthetic-fixture unit tests run local; the manifest build,
  usable-n, real-data teeth (real-vs-permuted; assertion-(b) tile-set match; the T8.5 real-run
  audit) read the corpus and run **on Leonardo** via **`.venv/bin/python`** (NOTE: `uv` is **not**
  on the Leonardo login-node PATH — call the venv python directly).
- **EU eval-set freezes to** `data/processed/eval_set/2026-04-15.0/multiregion/` (the SG set owns
  `eval_set/2026-04-15.0/` write-once; spec §2 path note-back + plan T1/T6).
- **SSH:** user-authed ControlMaster socket `Host leonardo` (user `uaslam00`); dies on laptop
  sleep — re-run `step ssh login '<email>' --provisioner cineca-hpc` then `ssh -fN leonardo`
  (Umar runs these via `!`). Slurm/`setsid` jobs survive socket drops; login-node loops do not.
- Deploy: bring Leonardo to `main @ 8d46217` (git bundle is the proven path; `origin` pull may
  also work now that main is on origin).

## ▶️ One-liner to start the next session

> eval-set-generation is DESIGNED + PLANNED + PUSHED (`main @ 8d46217`): spec
> `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md`, plan
> `docs/superpowers/plans/2026-06-08-eval-set-gen.md`. Execute the 12-task plan **exactly** via
> subagent-driven-development on branch `phase-2-eval-set-gen` (fresh subagent/task, two-stage
> review; Umar is chat reviewer; forbid subagent branch/push/PR). Teeth are HALT-gates; Phase B
> (manifest+§2.2) before Phase C (guard); verified-end-state never exit codes. Do NOT relitigate
> the §2 locked decisions (held-out set, all-moderate cut, density-coherence dropped + the
> perplexity-gap non-leak twin, seam-never-an-arch-bar, within-regime≠§9.113, #13+#22 one
> re-derive). Carry the two dormant bake-off obligations (T8.5 datamodule re-point + schema-2.0
> backstop; T12 first-model power gate + munich→manchester reserve). No merge/push without
> Umar's word + `--no-ff`. Read `docs/handoffs/2026-06-08-end-of-eval-set-gen-planning.md` first.
