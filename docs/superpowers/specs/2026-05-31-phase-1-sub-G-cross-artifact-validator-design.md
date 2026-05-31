# Phase-1 sub-G — cross-artifact consistency validator (PRD stage five) — design

**Status:** DRAFT v2 — brainstorm complete (9 gated decisions); v1 reviewer-approved with four sharpenings (this revision). Pending final read-through before plan-write.
**Date:** 2026-05-31.
**Branch:** `phase-1-sub-G-cross-artifact-validator` (off `main` @ sub-F merge `9336129`).
**Operating discipline:** sub-project-planning-protocol-v1 (six gates + five principles). Sub-G is a candidate v2 bump at close.
**Supersedes naming:** PRD uses no "sub-" labels (Phase/stage only); "sub-G" is the implementation-layer name from the sub-C/D/E specs and handoffs. This document is the design authority for that work.

---

## 1. What sub-G is (one paragraph)

Sub-G is **PRD §5 stage five — consistency validation**: the cross-sub-project
validator that confirms, for a real region, that (1) the macro plan matches the
underlying geometry, (2) every boundary contract corresponds to actual cell
tokens, and (3) token sequences are decodable to valid GeoJSON — quarantining
failures for inspection. To have real artifacts to validate, sub-G also **runs
the pipeline end-to-end** on the de-risk region (Singapore): it chains the
existing `derive_region` orchestrators for the stages whose caches don't yet
exist (sub-E, then sub-F), then runs the validator. A region that passes the
validator (empty quarantine + sanity floor intact) earns a region-level
`_PHASE1_VALIDATED` marker — the validator half of the PRD §11 Phase-1→Phase-2
gate ("tokenize 100 tiles correctly, validator passing on all, round-trip back
to GeoJSON").

### 1.1 What sub-G is NOT (scope boundary)

- **NOT the eval set.** Eval-set generation (PRD §9 held-out real cities) is a
  *separate successor sub-project* (see §2). Sub-G does not select, lock, or
  generate the held-out set.
- **NOT a re-platforming of the pipeline.** Sub-G reuses the four existing
  per-stage `derive_region` orchestrators; it does not build a new unified
  pipeline that supersedes them (Decision 4).
- **NOT a re-validation of within-stage invariants.** Sub-G validates only the
  cross-artifact *seams* that no single stage can check alone; it trusts each
  stage's own `_SUCCESS` for within-stage correctness (Decision 2).
- **NOT a regenerator of sub-C / sub-D.** Those caches already exist
  (`2026-04-15.0`); sub-G fails loud if their `_SUCCESS` is absent rather than
  rebuilding them.

---

## 2. Scope, sequencing, and data state

**Order of remaining Phase-1 + Phase-2 work:** sub-F (done) → **sub-G** (this
spec) → **eval-set-generation** (separate sub-project) → training-scaffold (PRD
Phase 2 bake-off). Naming eval-set-generation explicitly as sub-G's successor is
a hard obligation of the sub-G close-handoff (§8), to prevent the
"training-scaffold next" drift that the sub-F close exhibited.

**Data state at sub-G start (`2026-04-15.0` release):**

| Stage | Real cache exists? | Has `derive_region`? | Validate script? |
|---|---|---|---|
| sub-C | **yes** | `pipeline.py` + `extract_tiles.py` | `validate_extraction.py` |
| sub-D | **yes** | `pipeline.py` + `derive_macro_plan.py` | `validate_macro_plan.py` |
| sub-E | **no** (never run on real) | `pipeline.py::derive_region` (`:109`) + `derive_boundary_contracts.py` | `validate_boundary_contracts.py` |
| sub-F | **no** (never run on real) | `pipeline.py::derive_region` (`:184`) + `scripts/sub_f/derive.py` | `scripts/sub_f/validate.py` |

So the "end-to-end pipeline" half of sub-G reduces to **run sub-E's
`derive_region`, then sub-F's**, using existing code. The defect surface is
concentrated at exactly those two never-executed real paths (sub-E's real
emission; sub-F's first real read of it — the sub-F §8
`test_singapore_encode_layer_real_sub_e` "first read without
`SubEContractViolation`"). **Execution-time precondition:** the sub-C/sub-D
Singapore caches must be present on disk; if sub-C must be regenerated that is a
~8-hour cold Overture fetch (known slow path), a precondition to verify before
any run — not something sub-G builds.

---

## 3. Load-bearing design rules (first-class)

These govern every check in sub-G and are the through-line of the brainstorm.
They are stated **before** the decisions because the decisions (§4) invoke them
by name.

1. **Independence-by-construction.** A cross-artifact check is only meaningful if
   its two sides are measured independently. A validator that passes because both
   sides agree on the same wrong thing is the sub-project-scale analog of sub-F's
   "ambiguous-spec-resolved-silently."
2. **Measure-from-source, not read-and-compare.** Derive the expected value from
   the upstream *source/spec*, not by re-reading the implementation being
   validated (no circular reader reuse; no reading sub-D's verdict).
3. **Provenance-citation.** Every structural invariant cites a spec clause
   *outside the stage it validates*. Circular-by-provenance invariants are
   rejected. The citation doubles as runtime self-documentation in the diagnostic.
4. **Halt-and-revisit, not push-through.** The first real run is a **measurement,
   not a verification**. Sub-F's §8 items will not all pass cleanly; expect
   defects in sub-F's inferred motorway/multi-part handling OR sub-E's real
   emission. Treat the first run as data.
5. **Every measurement carries an action contract.** A reported number with no
   defined response (sanity floor, deferral trigger) is data exhaust
   (long-cell-diagnostic lesson). Reported ≠ ignored.

> Rules 1–3 are strong **planning-protocol-v2** candidates at sub-G close
> (they generalize Gate 6 / threshold-pairing to the cross-artifact-validator
> setting). Per protocol versioning discipline, the bump happens at close.

---

## 4. Architecture (the nine locked decisions)

### Decision 1 — Scope: pipeline-run + validator (eval-set deferred)

Sub-G = (a) the end-to-end pipeline-run that materializes the missing real
sub-E/sub-F caches + (b) the cross-artifact validator. Eval-set generation is a
separate successor sub-project. *Rationale:* the validator is meaningless without
real artifacts to check, and generating them is intrinsic to making it real (and
unblocks sub-F's §8 cache-gated items); eval-set is policy work (which cities,
by what criteria, for which slices) that PRD §9 does not enumerate and that
benefits from sub-G first proving "tokenize correctly" concretely.

### Decision 2 — Trust model: cross-artifact only, with independence-by-provenance

Sub-G validates only the three inter-stage seams (Decisions 3a/3b/3c). It trusts
each upstream stage's `_SUCCESS` for within-stage correctness — it does **not**
re-run within-stage invariants (no full end-to-end re-validation; no "spot
re-check" allow-list). *Rationale:* full re-validation creates two maintenance
points per invariant (lock-and-guards drift at sub-project scale) and gives false
confidence — distrust without independence is theater. The protocol's
"internal-consistency-gates-passed-while-bugs-survived" concern is answered
**not by quantity of checks but by constructing the three seam checks as
genuinely independent measurements** — Rules 1–3 of §3.

### Decision 3a — Seam 1 (macro plan ↔ geometry): structural invariants, citation-mandatory

Do not reuse sub-D's derivation logic and do not re-derive sub-D's full output
(Rule 2). Instead, **recompute the input quantity from sub-C** (footprint area,
land-use mix, road presence/width per cell) and assert sub-D's macro output is
the correct *function* of it (density bin matches the footprint-area range;
dominant-class matches the land-use mix; every road-skeleton cell has ≥1
qualifying sub-C road).

**Independence is in where the truth-statement originates, not in the checking
code (Rule 3).** Every structural invariant carries a **provenance citation** to
a spec clause *outside sub-D* (PRD, or sub-C's extraction spec). An invariant
whose only authority is sub-D itself is circular-by-provenance and is rejected —
find an independent source or drop it. (Worked hazard: if the "qualifying road"
definition comes from sub-D's road-detection logic, the invariant is circular; it
must trace to sub-C's road-extraction contract.)

> **OPEN (→ §9 #1):** the concrete invariant *enumeration* and each invariant's
> `signature_definition` are not yet fixed. This decision names a starter set
> only; the full list + per-invariant provenance citation is a plan-write
> Step-0 deliverable.

### Decision 3b — Seam 2 (boundary contract ↔ cell tokens): transcription-only bijection

Seam 2 asserts sub-F's brefs are a faithful **bijection** of sub-E's
boundary-contract entries — `(active edge, direction, class)` in →
`<bref_DIR_CLASS>` out, **both directions** (catches dropped *and* invented
brefs). It validates that sub-F *transcribed* sub-E correctly; it does **not**
validate that sub-E's classes are semantically right.

- **Independent second parser (Rule 2).** Read sub-E's boundary-contract parquet
  directly; **never** call sub-F's `boundary_contract.py` reader (circular — it
  interprets the same contract being compared).
- **The filtering rule is the real locus of independence.** The bijection is over
  the *active-emission subset* of sub-E rows (per T8.5: `marker != non_active AND
  boundary_class != NONE`), not all 144 edge-rows/tile (naive iteration would
  bury the signal under ~130 false "missing brefs"/tile). The second parser must
  *decide which rows count as expected emission* via the BP7 + T8.5 spec clause
  (Rule 3 provenance) — **not** by calling sub-F's `_classify_feature_for_bref`,
  else it is independent in name only.

> **DEFERRED (→ §8, with trigger):** semantic class-correctness (is sub-E's
> `motorway` really MINOR?) is out of seam-2 scope. It is seam-1-shaped and
> **blocked on a pending human decision** (motorway tiering) with no locked
> `road-class → boundary-class` spec clause. **Trigger:** becomes a seam-1 check
> if the motorway-tiering decision lands and a spec clause locks the mapping.

> **OPEN (→ §9 #4):** exact sub-E contract column names (`marker`,
> `boundary_class`, edge/direction encoding) must be verified against
> `sub_e/io.py` at plan-write (protocol §3 proactive-contract-verification)
> before the second parser is written. This document cites them at the contract
> level only.

### Decision 3c — Seam 3 (decodability): binary gates + reported distributional accuracy

Seam 3 splits into two checks of different shape:

1. **Decodability → valid GeoJSON (hard per-tile gate, quarantinable).** Every
   cell's token sequence decodes (via sub-F's decoder) to a structurally valid
   GeoJSON geometry. *Provenance:* OGC simple-features validity — independent of
   sub-F (Rule 3). **Paired** with a loose per-tile **structural bound** (no
   decoded vertex implausibly far from its cell extent), which catches
   catastrophic decode errors a validity check alone would miss (a
   technically-valid polygon with a vertex 10 km outside the cell). This pairing
   is the protocol's threshold-pairing principle applied.

2. **Round-trip accuracy vs ORIGINAL sub-C geometry (measured + reported,
   region-level — NOT a v1 gate).** Measured end-to-end and **decomposed per
   protocol §5** (canonicalization vs direction-quantization vs encode/decode
   contributions). It is *not* a pass/fail gate in v1 because no independent
   end-to-end threshold exists yet — sub-F's locked thresholds (position
   p99.9 ≤ 4.8m / angle p95 ≤ 4.0°) were measured encoder→decoder against the
   *canonical intermediate* and exclude the canonicalization + quantization loss
   that an original-sub-C baseline adds. Reusing them would repeat the Halt-2
   metric-mismatch (a statistic measured on one regime enforced as a hard bound).

   **Measured-and-reported has a three-part operational definition (Rule 5)** so
   it does not decay into data exhaust:
   - **Artifact:** `_PHASE1_ACCURACY_BASELINE.yaml`, written **every run
     regardless of verdict**; captures position/angle p99.9/p95 + the
     structural-bound histogram; byte-deterministic (§7). Becomes the first
     locked baseline once multi-region stability is shown.
   - **Sanity floor (existence non-negotiable; number reviewer-call):** a level
     below which something is decisively broken without needing a calibrated
     threshold (e.g. position p99.9 > 50m halts as "fundamentally broken
     encode/decode, not a calibration issue"). This makes "reported" actually
     trigger halt-and-revisit.
   - **§8 deferral trigger:** the proper accuracy gate locks when (a) multi-region
     data exists (likely training-scaffold) **and** (b) the baseline distribution
     is shown stable across regions.

> **OPEN (→ §9 #3):** the sanity-floor numbers (position/angle) are reviewer-call.

### Decision 4 — Pipeline-run: thin chain + region gate + resume-from-`_SUCCESS`

Sub-G ships a thin chain runner (not a new per-stage orchestrator) plus the
validator. **Locked contract:**

- **Inputs:** region, release version, `--force` (default false).
- **Precondition:** sub-C + sub-D `_SUCCESS` present, else fail loud (sub-G never
  regenerates C/D).
- **Per stage in `[sub-E, sub-F]`:** `_SUCCESS` present and not `--force` → skip
  with log; else run that stage's `derive_region`.
- **Halt** on any stage *crash*; do not proceed past a failed stage (a crashed
  stage produces no artifacts to validate).
- **Then** run the cross-artifact validator over all four stages' on-disk
  artifacts.
- **On pass** (empty quarantine + sanity floor intact): write `_PHASE1_VALIDATED`
  — the PRD §11 gate. **On fail:** the quarantine workflow (Decision 5) fires.

*Rationale:* the orchestration already exists per stage; resume-from-`_SUCCESS`
makes halt-and-revisit cheap (don't pay full pipeline cost each defect cycle —
sub-F's `derive_region` already gates on `_SUCCESS`; the chain just respects it).
`--force` re-derives from scratch when a re-lock invalidates prior cache.

> **OPEN (→ §9 #5):** *how* the chain runner invokes sub-E/sub-F `derive_region`
> — Python import vs subprocess — is a design decision for plan-write (lean:
> subprocess). See §9 #5.

### Decision 5 — Gate & quarantine semantics: empty-quarantine gate + rich diagnostic

Quarantine is the **workflow** (inspect failures); `_PHASE1_VALIDATED` is the
**verdict** (no failures to inspect). Different levels, not opposing pulls —
reconciling PRD §5 (quarantine for inspection) with PRD §11 (validator passing on
all). Failing tiles are sequestered (by reference — Decision 7) with a structured
diagnostic; the validator exits non-zero and **withholds `_PHASE1_VALIDATED`
while any tile is quarantined**. The PRD §11 "all pass" gate is satisfied only
when the quarantine set is empty.

Accepted known-losses need **no special case** — they are encoded as the check's
*tolerance* (e.g. seam-3's round-trip thresholds; sub-F's 0.22% right-angle
catastrophic loss), so a within-tolerance tile passes and is never quarantined.
No accepted-loss allow-list (registries become dumping grounds).

**Per-tile diagnostic shape** (extends sub-C's `TileValidationError`; cross-artifact
failures need both sides of the seam + the provenance citation, which makes the
runtime diagnostic self-documenting — the same citation that gives the invariant
design-time independence per Rule 3):

```
tile_id
invariant_name
artifact_left  + observed_value     # e.g. "sub_c.footprint_area_m2", 1247.3
artifact_right + observed_value     # e.g. "sub_d.density_bin", "high"
expected_relationship               # e.g. "footprint_area in [a,b) implies density_bin == medium"
spec_clause_citation                # e.g. "PRD §3.2 density binning + sub-C §5.4 footprint extraction"   (illustrative)
# byte-deterministic per sub-C §12.4
```

### Decision 6 — Defect budget: accumulate + signature-grouping

On the first real run (where many defects are expected — Rule 4), the validator
runs ALL checks on ALL tiles, collects every failure, and reports the **complete
defect map in one run** (gate fails if non-empty) — the natural completion of the
empty-quarantine gate and the measurement posture (one run → full map → batch
fixes, not N fix-rerun cycles). Chain-level crash-halt (Decision 4) is unchanged;
accumulation applies only when the validator *can* run but finds defects.

**Grouping by `(invariant_name, signature)` is what separates measurement from
data exhaust.** The **signature is the failure PATTERN, not the values** — e.g.
"density bin one-step-too-high vs footprint range," with specific per-tile values
summarized, not enumerated. Spec obligations:

1. Each invariant defines a `signature_definition` (what makes two failures the
   same pattern — "off-by-one bin direction," not "exact value match").
2. Grouped output sorted by **instance_count desc, then `invariant_name` asc**
   (deterministic tiebreak).
3. Per-group **value-summary** fields per invariant type (min/max/median for
   numeric; enum-distribution for categorical).
4. **Hypothesis field is OPTIONAL** — empty is honest; speculation is not.

Target output (illustrative):

```
invariant: density_bin_matches_footprint
signature: density bin one-step-too-high vs footprint range
instances: 100/100 tiles
value_summary: footprint range [1132, 1389] m², all binned "high" (expected "medium")
spec_clause: PRD §3.2 density binning
hypothesis: bin boundary off-by-one OR footprint extraction systematically inflated   # optional
```

### Decision 7 — Quarantine I/O: reference-only, written every run

Single byte-deterministic `quarantine_report.yaml` at region root, written
**every run regardless of verdict** (symmetry with
`_PHASE1_ACCURACY_BASELINE.yaml`; clean runs are diffable — "did this iteration
shrink the defect set from 47 to 12?" is `git diff quarantine_report.yaml`). Each
group **references tile IDs** — no copying/moving of artifacts (they stay in place
under each stage's region dir; the validator stays **read-only on upstream**,
honoring the no-mutation boundary). Quarantine = "flagged for inspection," not
physically moved; since the gate requires emptiness, a passing region has zero
flagged tiles, so no separate "exclude-from-training" mechanism is needed.

**The empty record has positive meaning** — explicit `groups: []` proves the run
completed validation and found nothing, which is *different* from file-absence
(which could mean validation didn't run / crashed / wasn't invoked). Emit the
empty record explicitly; never skip-on-empty.

**Artifact shape:**
- Location: region root, alongside `_PHASE1_VALIDATED` + `_PHASE1_ACCURACY_BASELINE.yaml`.
- Filename: `quarantine_report.yaml` (singular, region-scoped).
- Written: every run, regardless of verdict.
- Byte-deterministic content (sorted as Decision 6); volatile run-metadata
  segregated per §7 (concrete field list + `EXCLUDED_FROM_SHA` precedent there).
- Each group: `invariant_name`, `signature`, `instance_count`, `tile_ids`
  (sorted), `value_summary`, `spec_clause_citation`, optional `hypothesis`.
- `_PHASE1_VALIDATED` written iff `groups == []` AND sanity floor not violated.

---

## 5. Planning-protocol-v1 gate mapping

| Gate / principle | How it applies to sub-G |
|---|---|
| Gate 1 — Plan review | One-decision-per-message brainstorm done (9 decisions); plan-write must enumerate seam-1 invariants + signatures. |
| Gate 2 — Pre-dispatch audit | **Verify sub-E/sub-F/sub-C exact schemas by grep** before writing the second parser + invariants (the §9 OPEN items). |
| Gate 3 — Implementer test-run | Synthetic-fixture tests for each seam + the real Singapore run (the measurement). |
| Gate 4 — Halt-and-report | First-run defects route to reviewer (halt-and-revisit, Rule 4); no inline push-through fixes. |
| Gate 5 — Pre-code data-flow reasoning | Reason about the bijection cardinality (144 rows vs active subset) before coding seam 2. |
| Gate 6 — External-source-of-truth | This is the *whole point* of Decision 2's independence-by-provenance (Rules 1–3) — every invariant cross-references an upstream spec clause, never the stage under validation. |
| §2 threshold-pairing | Seam-3 decodability gate paired with the structural vertex bound; accuracy threshold deferred (no broken-but-in-range gate shipped). |
| §3 proactive contract verification | §9 OPEN items — read `sub_e/io.py`, `sub_f` schemas, sub-C enums before coding. |
| §5 per-stage prediction decomposition | Seam-3 accuracy decomposed into canonicalization / quantization / encode-decode. |
| §6 rough-numbers heuristic | A first real run that reports "0 defects" or "100% pass with no external cross-reference" is an anti-signal — suspect the validator. |

---

## 6. Proposed module / file layout (plan-write refines)

- `src/cfm/data/sub_g/__init__.py`
- `src/cfm/data/sub_g/pipeline.py` — thin chain runner (Decision 4; invocation
  pattern import-vs-subprocess is §9 OPEN #5).
- `src/cfm/data/sub_g/validator.py` — orchestrates the three seams, accumulates +
  groups (Decision 6), writes the artifacts.
- `src/cfm/data/sub_g/seam_macro_geometry.py` — seam 1 structural invariants.
- `src/cfm/data/sub_g/seam_contract_tokens.py` — seam 2 independent second parser
  + bijection.
- `src/cfm/data/sub_g/seam_decodability.py` — seam 3 decodability gate + accuracy
  measurement/decomposition.
- `src/cfm/data/sub_g/diagnostics.py` — diagnostic record + grouping + report
  writer (Decisions 5–7).
- `src/cfm/data/sub_g/versions.py` — validator version; reuse sub-D's
  `VersionNamespace.VALIDATOR`.
- `scripts/sub_g/derive_phase1_region.py` — CLI entry to the chain runner.
- `scripts/sub_g/validate_phase1_region.py` — CLI entry to the validator alone.
- `tests/data/sub_g/…` — per-seam synthetic fixtures + `test_singapore_integration.py`.

---

## 7. Determinism & provenance of sub-G's own outputs

Byte-determinism follows **sub-C §12.4** (structured-format byte-determinism) and
the **sub-E §9.2 `EXCLUDED_FROM_SHA`** segregation precedent. Every sub-G content
artifact (`quarantine_report.yaml`, `_PHASE1_ACCURACY_BASELINE.yaml`, and the
`_PHASE1_VALIDATED` marker) carries a top-level `run_metadata:` block split into
two parts:

- **Stable identity — IN the digest:** `region`, `release_version`,
  `validator_version` (semantic, via `VersionNamespace.VALIDATOR`; a version
  change invalidates the marker). The digest is computed over this block **plus**
  the validation content (groups / baseline measurements).
- **Volatile / informational — EXCLUDED from the digest:** `run_timestamp`,
  `host`, `run_uuid`, `sub_g_commit_sha`. Recorded for reproducibility (config +
  commit + data) and log-tracing, but never enter the byte-determinism
  comparison — two re-runs of the same `validator_version` over the same
  artifacts produce identical digests despite different timestamps/UUIDs/host.
  (`sub_g_commit_sha` is excluded so that two builds of the same *semantic*
  validator version remain digest-comparable; the *semantic* version is what
  gates marker validity.)

Sub-G does **not** mutate any upstream artifact (read-only-on-upstream boundary).

---

## 8. Deferred items (each with an explicit trigger)

| Deferred | Trigger to un-defer |
|---|---|
| **Seam-2 semantic class-correctness** (sub-E class ↔ sub-C road attributes) | Human motorway-tiering decision lands AND a `road-class → boundary-class` spec clause locks → becomes a seam-1 check. |
| **Seam-3 accuracy gate (proper threshold)** | Multi-region data exists (likely training-scaffold) AND the `_PHASE1_ACCURACY_BASELINE` distribution is shown stable across regions → lock the threshold. |
| **eval-set-generation** | Its own successor sub-project, after sub-G closes (PRD §9). Carried here so it cannot drift into "training-scaffold next." |
| **Cross-tile-boundary bref symmetry** | Inherited from sub-F §8: within-tile only in v1; un-defers if sub-E makes external edges active (e.g. a motorway→MAJOR fix routes an arterial across a tile boundary) → needs sub-D inter-tile neighbour graph. |
| **Sub-F §8 sub-E-cache-gated items** | Unblock *as part of* sub-G's pipeline-run (real sub-E/sub-F caches now exist): real round-trip re-measure, first-real-read, real-region derive, cross-tile BP7 composite on real data, T3c stage-4 empirical re-measure, α-drop on real Singapore. These are the measurement, per Rule 4. |

---

## 9. OPEN questions for reviewer (resolve before / during plan-write)

1. **Seam-1 invariant enumeration** *(plan-write Step-0 deliverable).* Concrete
   list of macro↔geometry invariants, each with its `signature_definition` and an
   out-of-sub-D provenance citation. (Starter set named in Decision 3a; needs
   completion + citation audit.)
2. **The "100 tiles" gate-set — and what the chain runner runs on** *(reviewer-call
   before plan-write; may force re-approval).* The Singapore tile count determines
   both what `_PHASE1_VALIDATED` certifies **and** what the runner executes over.
   Three cases:
   - **>100 tiles:** gate is all-Singapore vs a defined-100-subset (the latter
     borders eval-set scope — do not bleed eval-set policy into sub-G).
   - **=100 tiles:** clean.
   - **<100 tiles:** cannot satisfy "100 tiles" from Singapore alone — either pull
     another region (**sub-G scope expands → surface for re-approval before
     coding**) or interpret PRD §11 "100 tiles" as a *target, not a floor*.
   Plan-write must answer the tile count *early* (a Step-0 read) before the
   gate-set and runner scope lock.
3. **Sanity-floor numbers** (seam-3) *(reviewer-call before plan-write).* Position
   p99.9 and angle p99.9 "broken" thresholds; existence is locked, the numbers
   are reviewer-call.
4. **Exact upstream schemas** *(plan-write Step-0 deliverable, protocol §3).* sub-E
   boundary-contract columns (`sub_e/io.py`), sub-F `cells.parquet` token grammar,
   sub-C feature/enum schemas — read before the second parser + invariants are
   coded.
5. **Chain-runner invocation pattern** *(plan-write design decision; lean:
   subprocess).* Does `sub_g.pipeline` invoke sub-E/sub-F `derive_region` by
   **Python import** (`from cfm.data.sub_e.pipeline import derive_region`) or by
   **subprocess** (shell out to the existing `scripts/.../derive_*.py`)? Import is
   tighter (in-process, typed) but couples sub-G to each stage's module internals;
   subprocess gives a clean process boundary, matches the existing CLI entry
   points, and keeps coupling low (sub-G depends only on the documented script
   contract + `_SUCCESS`/exit code). **Lean: subprocess** — but it's a design
   decision, not an implementation detail.

---

## 10. Decision index (provenance of this design)

| # | Decision | Locked as | Key rationale / deferral |
|---|---|---|---|
| 1 | Scope | pipeline-run + validator; eval-set separate | validator needs real artifacts; eval-set is policy work |
| 2 | Trust model | cross-artifact only, independence-by-provenance | distrust without independence is theater |
| 3a | Seam 1 method | structural invariants, citation-mandatory | independence = provenance of the truth-statement (Rule 3) |
| 3b | Seam 2 method | transcription-only bijection, filtering-rule provenance | semantic correctness deferred (no independent spec clause) |
| 3c | Seam 3 method | binary gates + reported distributional accuracy | sub-F thresholds are canonical-baseline; can't reuse |
| 4 | Pipeline-run | thin chain + region gate + resume-from-`_SUCCESS` | orchestration exists; makes halt-and-revisit cheap |
| 5 | Gate/quarantine | empty-quarantine gate + 3-field-plus-citation diagnostic | reconciles PRD §5 (inspect) + §11 (all-pass) |
| 6 | Defect budget | accumulate + signature-grouping | measurement posture; grouping prevents data exhaust |
| 7 | Quarantine I/O | reference-only, written every run, explicit empty record | diffable iterations; read-only-on-upstream |

---

## Authoritative references

- PRD: §5 (stage five), §9 (eval suite + held-out set), §11 (Phase-1 gate).
- sub-C design §12.3 (the sub-G deferral: the three seam checks), §12.4
  (structured diagnostic format, byte-determinism).
- sub-D design (consumer read-pattern: "sub-G validates sub-D against sub-C
  evidence"), `VersionNamespace.VALIDATOR`.
- sub-E design §9.2 (`EXCLUDED_FROM_SHA`); `sub_e/io.py` schema — read at plan-write.
- sub-F close handoff `docs/handoffs/2026-05-30-sub-F-close-T15-complete.md` §8
  (inherited deferred items), §9 (BP7 ↔ sub-E coupling).
- Planning protocol: `docs/protocols/sub-project-planning-protocol-v1.md`.

*End of sub-G design draft v2 — pending reviewer final read-through before plan-write.*
