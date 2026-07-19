# START OF: Pipeline Readiness Audit + Conditioning Enrichment (2026-06-10)

**This is a READINESS-AUDIT handoff, not an execution handoff.** It boots a FRESH sub-project
in a context-free session. Do not resume the bake-off plan. The first act is an ENUMERATION
brainstorm against the actual codebase — NOT fixes, NOT the conditioning re-derive.

---

## 1. THE PIVOT (the thing no other doc carries yet)

**The Phase-2 cross-city architecture bake-off is HALTED at the T5-reopen. Tasks 10–12 are NOT
to be run.** Gate 2 / Part A (the conditioning-discrimination gate) ran on CPU against the real
held-out tiles (model-independent, zero GPU) and **FAILED on substance**:

- 304 / 321 qualifying cross-city comparisons BH-significant; both metrics; 44 / 45 strata.
- The 4-dim macro plan the model is handed — `(zoning, road_skeleton, cell_density_bucket,
  coastal_inland_river)` — does **not** carry city character. At the *same* stratum
  `(1,2,0,2)`, glasgow buildings median 59 m² vs krakow 25 m² (~2×).
- **Artifact ruled out** (per-city magnitudes all sane same-order; shared CRS-agnostic decoder,
  no per-city branch). It is real morphology.

Consequence: a per-city miss is ambiguous ("wasn't told the city's character" vs "failed to
render"). **The worst-case AND mean per-city bars are invalid as-is.** Umar's call (2026-06-10):
do NOT retreat to a within-city/in-distribution split. Reopen T5 properly via enrichment, and
stop discovering latent failures one-at-a-time at the training threshold.

Full evidence: `reports/2026-06-10-gate-i-conditioning-discrimination-closeout.md` (+ `-design`, `-result.yaml`).

---

## 2. THE NEW MISSION

Two halves, one sub-project:
1. **A full end-to-end pipeline READINESS AUDIT** — enumerate the *classes* of latent failure
   across everything the pipeline does, so we close the class deliberately instead of tripping
   over instances near training.
2. **The CONDITIONING ENRICHMENT** that makes the cross-city comparison valid (so a per-city bar
   can mean "generalization," not "the model wasn't told enough").

---

## 3. THE ORDERING (critical — do not invert)

Fresh sub-project: **brainstorm → enumerate failure classes AGAINST THE ACTUAL CODEBASE → spec →
plan → execute.** The **first act is the ENUMERATION brainstorm**, not fixes. Do **not** dive
into the conditioning / #13 / #22 re-derive before enumerating what else shares that surface.
Enumerate the whole surface first; then fix systematically. Read the sub-project planning
protocol (`docs/protocols/sub-project-planning-protocol-vN.md`) before brainstorming.

**Completeness criterion (the teeth — the audit's red-before/green-after).** An enumeration with
no done-test is just a longer mole-list: the agent lists some classes, feels thorough, declares
done, and a sixth class surfaces at the training threshold anyway because nothing forced the
enumeration to be exhaustive against what actually runs. So:

- **Anchor the enumeration to the ACTUAL END-TO-END EXECUTION PATH the bake-off runs — not a grep.**
  Trace the live path: **EU corpus read → conditioning construction → training-data load → eval
  read → decode → scoring.** Every one of the five bugs lived on the execution path and was
  invisible to the tests — each was found by *running the real thing*, not by inspection. A
  credible audit traces what actually executes (grepping `EPSG3414` would have missed all five).
- **"Done" is NOT "the list feels long."** Done is: for each failure class, on every path the
  bake-off touches, the agent can show EITHER a check that it holds across **all EU zones** OR an
  **explicitly logged gap**. The per-class question the agent must be able to answer **NO** to:
  *"Could a latent failure of this class exist on a path the bake-off executes that this audit did
  not examine?"* If it cannot answer no, enumeration is not done.
- This converts the audit from "list what I can think of" into **"prove coverage of the execution
  surface"** — the only version that closes the class instead of producing a longer mole-list.

---

## 4. THE PATTERN (evidence motivating the audit — a PATTERN, not a task list)

Five latent assumptions hit one-at-a-time at the training threshold across Phase D. They are the
*reason* for the audit; do not treat them as the scope:
1. **Blanket repoint broke Singapore** (Phase A / Task 1) — holdout readers are dual-region;
   resolved region-aware (`holdout_manifest_for_region`).
2. **CRS dir default** (Task 8) — `build_shards_in_memory` defaulted `tile_dirname` to Singapore
   `EPSG3414`; EU FileNotFound. Fixed `ed1138c` (+ I1-safe writer sibling `84edb3b`).
3. **Eval-side CRS twin** (Task 9 step-0) — `geometry.py` / `holdout/pipeline.py` same default;
   silent **vacuous 0.0** on an EU read (not a crash). Fixed `d54424e`.
4. **Building-as-roads decode** (gate-(i) build) — decoder returns building rings as LineString
   by contract; without `promote_building_rings` buildings were miscounted as roads. Fixed `619405a`.
5. **Conditioning-expressivity gap** (gate-(i) verdict) — §1 above. NOT a bug to patch; the
   enrichment mission.

Each was caught by small-before-big / denominator inspection BEFORE poisoning a result. The audit
exists so the *next* class is found by enumeration, not by tripping over it.

---

## 5. SEED CLASSES for the enumeration (starting points from the evidence — expand against code, not exhaustive)

- **Single-region / Singapore-only assumptions** (hardcoded `EPSG3414`, singapore-default region
  args, singapore-only builds in sbatches/runners).
- **CRS handling across ALL zones** (EU spans EPSG:25829–25835; any dir-name / coordinate / area
  path that assumes one zone).
- **Decode / construction-identity correctness** (building-ring promotion; Multi* geometry enum
  gaps — see `[[project_sub_c_multi_geometry_gap]]`; other "consumer must promote/transform" contracts).
- **Conditioning sufficiency / expressivity** (the 4-dim macro plan; `#13` admin_region None for
  EU; `#22` sub_c_morphology_class constant).
- **Eval read paths** (every holdout/geometry read; the gate-(ii) reader is unbuilt; coherence layer parked).
- **Frozen-Singapore-only artifacts** (the SG eval-set/holdout frozen write-once; do validated EU equivalents exist?).

---

## 6. SEED for the conditioning fix

The gate already produced the blueprint: the per-(stratum, city) feature deltas show *which
dimensions distinguish cities at the same macro-stratum* (whatever makes glasgow ≠ krakow). Mine
`reports/2026-06-10-gate-i-conditioning-discrimination-{design,result,closeout}` for it. The
**`#13` admin_region / `#22` morphology bundle** (previously parked to Phase-3) is the likely
enrichment surface — now in scope.

---

## 7. POINTERS (pull specifics from these + the codebase; this handoff does NOT re-summarize them)

- Phase-D + bake-off handoffs: `docs/handoffs/2026-06-09-start-of-bakeoff-phase-d.md`, `…-execution.md`,
  `docs/handoffs/2026-06-02-multiregion-feasibility-audit.md`.
- Delta-spec (locked decisions §9; the gate §4; open decisions §7): `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md`.
- Plan: `docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md`.
- Builds: `reports/2026-06-09-phase-2-bakeoff-phases-A-C-local-build.md`, `reports/2026-06-10-phase-2-bakeoff-task8-leonardo-build.md`.

---

## 8. STATE + discipline

- **CONSOLIDATED TO `main` @ `a4d29ff`** (2026-06-10, `--no-ff` merge of `phase-2-bakeoff` @
  `9693e53`, then pushed). The bake-off machinery + Task-8 multi-region build + the five fixes +
  the gate-(i) machinery now live on `main`. **Start the audit from `main`.** `phase-2-bakeoff` is
  merged (retained, deletable). **No GPU spent, no scored runs.** Full suite **1,343 green** on
  merged main.
- Discipline carries: subagent-driven (implementer ≠ reviewer), verified-end-state (re-read, never
  exit codes), stop-before-commit on gates/forks, ruff unpiped, `uv sync --extra dev --extra training`.
- Leonardo SSH-ready (`leonardo`, `/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, git-bundle deploy) —
  **nothing runs** until the audit reframes the plan and Umar gives a word. No push/merge without
  Umar's word + `--no-ff`.
