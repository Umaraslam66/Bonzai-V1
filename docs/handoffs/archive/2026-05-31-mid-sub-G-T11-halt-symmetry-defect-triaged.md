# Mid sub-G handoff — T11 measurement HALTED on sub-F BP7 symmetry defect, fully triaged — 2026-05-31

> **Entry point for a cold session.** sub-G (cross-artifact consistency validator,
> PRD stage five) is built through T10 (57 tests green); the T11 real-data
> measurement run was fired on Singapore and **halted on a real sub-F defect**
> (the measurement working as designed). The defect is fully triaged into two
> findings (F1 + F2). **We are paused at a DECISION POINT awaiting the reviewer.**
> Read §1 (decision), §2 (state), §6 (triage) first.

---

## 1. THE DECISION POINT (what the next session resumes on)

The T11 measurement surfaced a real defect, triaged into **two findings, both in
merged/sealed sub-projects → human approval required before any code change. I
(the assistant) modified NEITHER sub-F nor sub-C.**

- **F1 — sub-F `_check_symmetry` is over-strict (78.5% of mismatches, ~2021).**
  Roads that *terminate* on an internal cell edge (no sub-C crossing record) emit
  a bref legitimately one-sided; the symmetry check wrongly fails them.
  **Proposed fix:** condition the check on the sub-C crossing record — one-sided
  OK iff no crossing record on the shared edge; still fail when a record exists.
  The §8 "cross-tile-boundary bref symmetry" item un-defers per its trigger.
- **F2 — sub-C asymmetric clipping (20.3% of mismatches, 553 confirmed).** A road
  *crosses* an internal edge (valid crossing record) but its boundary vertex is
  NOT placed symmetrically in both cells' clipped geometry (missing segment, or
  segment clipped short of the boundary). The symmetry check **correctly** catches
  these, so F1's conditioning must KEEP failing on them (not mask them). This is a
  NEW upstream finding for **sub-C**.

**Open F2 sub-question (the assistant offered to drill next, read-only):** is the
asymmetry *sub-C clip-short* (raw geometry genuinely doesn't reach the boundary)
or *canonicalization rounding* (raw reaches it, the 0.5 m round in sub-F moves the
neighbour vertex off)? Inspect **raw** endpoints of a few F2 samples to decide
whether F2's fix lives in sub-C clipping vs sub-F's canon/emission tolerance.

**The reviewer's pending choices:** (i) approve the F1 crossing-record-conditioned
relax; (ii) log F2 as its own §8-equivalent sub-C finding; (iii) drill the F2
clip-vs-canon mechanism now, or defer to a dedicated sub-C session. Both F1 and F2
are sub-F/sub-C revisits needing approval; nothing fires until the reviewer directs.

This is **defect cycle 1**. The other big §8 item (motorway/multi-part inferred
handling) has NOT fired yet — it may surface only after the F1 relax + a re-derive
lets sub-F's cross-tile validator pass and sub-G's own seam checks finally run.

---

## 2. Exact state

- **Branch** `phase-1-sub-G-cross-artifact-validator` (off `main` @ sub-F merge
  `9336129`); working tree CLEAN; **local-only** (unmerged, unpushed).
- **Code: T1–T10 done, 57 tests green, ruff clean** (commit log §3).
- **T11 run state (the halt):**
  - sub-E derive **SUCCEEDED** — full 494 tiles, `_SUCCESS` written at
    `data/processed/sub_e/2026-04-15.0/singapore/`. **The sub-E cache is now
    materialized** (a real artifact gained; unblocks other sub-F §8 cache-gated
    items). sub-F's contract-reader read it with **no `SubEContractViolation`** —
    the §8 "first real read" gate **passed**.
  - sub-F derive **HALTED** in its OWN `validate_cross_tile` → `_check_symmetry`
    at `tile=EPSG3414_i10_j10`. sub-F wrote `cells.parquet` for all **494** tiles
    during the encode loop (they're on disk) but **NO `_SUCCESS`** (validation
    halted before it).
  - **sub-G's own validator NEVER RAN** (it needs sub-F `_SUCCESS` / output). So
    no `quarantine_report.yaml` / `_PHASE1_ACCURACY_BASELINE.yaml` / `_PHASE1_VALIDATED`
    exist yet. sub-G's seam checks are unexercised on real data.

## 3. Commit log (this branch, newest first)

```
d3ca66c T11 Step-0 — gap assertion + densest-N subset selection (read-only)
44493db T10 CLIs (derive_phase1_region + validate_phase1_region)
027d136 T9 subprocess chain runner (resume-from-_SUCCESS + halt-on-crash)
d59cd5e T8 per-region validator (validate_tile + finalize + gate + sanity floor)
267365a T7 validator version + _PHASE1_VALIDATED + accuracy-baseline writers
4099d46 T6 seam 3 decodability gate + position/angle accuracy measurement
d1169f7 T5 seam 2 contract<->tokens transcription bijection
21a698a T4 seam 1 macro<->geometry — SI-1 density + SI-2 road skeleton
3ef991c T3 independent readers (raw sub-E contract, sub-F cells, sub-C features)
e9508ff T2 locked bucket cut-point loaders + bucket_of
6826850/3b5142b/67816a6 T1 Diagnostic foundation (+2 review fixes)
8cca453 plan v2 ; bce1882 plan draft ; 2996b9a/148342a/8db07e5 design v3/v2/v1
```

## 4. What sub-G is + module map (built, T1–T10)

Cross-artifact validator: runs sub-E→sub-F derive on a region (resume-from-
`_SUCCESS`), then validates **three independent seams** per tile, accumulates
defects grouped by signature, writes `quarantine_report.yaml` +
`_PHASE1_ACCURACY_BASELINE.yaml` every run, gates with `_PHASE1_VALIDATED`
(empty quarantine AND sanity floor intact).

- `src/cfm/data/sub_g/diagnostics.py` — `Diagnostic` (9-field), signature grouping, report writer.
- `…/buckets.py` — locked `macro_plan_vocab.yaml locked_buckets.{cell_density,road_skeleton}` loaders + `bucket_of`.
- `…/readers.py` — INDEPENDENT readers (raw sub-E contract via `read_sub_e_contract_rows` — **never** `sub_f.boundary_contract.load_boundary_contract`; sub-F cells; sub-C features-by-cell).
- `…/seam_macro_geometry.py` — seam 1: SI-1 density + SI-2 road-skeleton structural invariants. **SI-3 zoning DEFERRED** (no external clause for the argmax-count→token rule; circular-by-provenance; trigger recorded in module docstring).
- `…/seam_contract_tokens.py` — seam 2: independent bref prediction + bijection.
- `…/seam_decodability.py` — seam 3: decodability gate + position/angle accuracy (per-stage decomposition DEFERRED, diagnostic-only).
- `…/versions.py` — `VALIDATOR_VERSION="1.0.0"`, marker + baseline writers (volatile block excluded from digest).
- `…/validator.py` — `validate_tile` (pure) + `_macro_targets` + `finalize` (gate + sanity floor 50m/20°) + `validate_region` (parquet loop).
- `…/pipeline.py` — `run_chain` (ChainConfig; subprocess sub-E then sub-F; halt-on-crash).
- `…/subset.py` — T11 Step-0: gap assertion + densest-N selection (read-only).
- `…/cli.py` — `derive_main`/`validate_main`, exit codes **0 clean / 1 quarantine / 2 precondition**.
- `scripts/sub_g/{derive_phase1_region,validate_phase1_region,t11_step0}.py` — thin wrappers + Step-0 runner.

## 5. Locked design decisions (9 brainstorm + reviewer-ratified)

scope = pipeline-run + validator (eval-set is a SEPARATE successor sub-project);
trust model = cross-artifact-only with **independence-by-provenance** (every check's
truth-statement traces to a clause OUTSIDE the validated stage — "measure-from-source");
seam 1 = structural invariants w/ provenance-citation; seam 2 = transcription bijection
(filtering-rule provenance); seam 3 = binary gates + reported accuracy + sanity-floor
cliffs **position p99.9 > 50m / angle p95 > 20°** (reviewer-set); pipeline-run = thin
subprocess chain (resume-from-`_SUCCESS`); gate = empty-quarantine + 3-field-plus-citation
diagnostic; defect budget = accumulate + signature-grouping; quarantine I/O = reference-only,
every run. **Sequencing: sub-F → sub-G → eval-set-generation → training-scaffold.**

## 6. T11 triage detail (how F1/F2 were established — read before re-deciding)

- **Tile count corrected:** Singapore = **494 tiles** (sub-C == sub-D; gap = 0), NOT
  the 3457/2963 first reported (that was an `ls dir/tile=*` line-count of dir
  *contents*, ~7/6 files per tile). Step-0a gap-assertion passed vacuously.
- **Step-0b subset (recorded but unused so far):** 369 non-empty tiles; densest-200
  cutoff 993 features, 94% of geometry. List at `reports/sub_g_t11_measurement_subset.txt`.
  **NOTE: subset is now moot for the derive** — at 494 tiles the full derive is
  cheap, so T11 ran the FULL chain (the option-1/fallback decision; the option-2/3
  manifest-synthesis / D-gate analysis was scaffolding for the wrong 3000-tile number).
- **Symmetry-mismatch prevalence:** 306/494 tiles (62%); ~2728 mismatched (cell,dir).
- **Classification (crossing-record discriminator, confirmed against sub-F actual
  emission):** termination **2021 (legit one-sided, F1)** + crossing-asym **553
  (sub-C clip, F2)**; classifier method-noise concentrated in termination (221
  re-derived-only) not crossing-asym (62), so F2≈553 is robust.
- **Root-cause inspection (3 tiles):** all sub-C clip asymmetry — neighbour either
  lacks the crossing road or has it clipped short of the shared edge.
- **Triage scripts are in `/tmp` (NOT committed):** `/tmp/t11_triage.py` (i10_j10
  inspect + prevalence), `/tmp/t11_classify.py` (term-vs-crossing buckets),
  `/tmp/t11_triage2.py` (method-gap + root-cause). Re-create from this handoff if
  needed; they reuse `sub_f.encoder._classify_feature_for_bref`,
  `sub_f.boundary_contract.load_boundary_contract`,
  `sub_f.validator_cross_tile.{_neighbour_cell,_OPPOSITE_DIRECTION}`,
  `sub_g.seam_contract_tokens.parse_actual_brefs_per_cell`.

## 7. How to resume in the new session

1. `git branch --show-current` → expect `phase-1-sub-G-cross-artifact-validator`.
2. Read §1 (decision) + §6 (triage evidence). The reviewer will direct F1/F2 + the
   F2 clip-vs-canon drill.
3. Any sub-F/sub-C change is a **merged-sub-project revisit → human approval first**.
   After F1 relax + re-derive: re-run `scripts/sub_g/derive_phase1_region.py`
   (real region dirs, see §2 paths) → sub-F `_SUCCESS` → sub-G validator finally
   runs → expect the *next* defect cycle (motorway/multi-part).
4. T12 (close handoff + protocol-v2 candidate bump) is still pending after T11.

## 8. Memories saved this session (all in the memory dir, indexed in MEMORY.md)

`project_sub_g_before_training` (sequencing); and feedback: `tool_output_silence_is_not_pass`,
`precondition_verify_count_not_estimate`, `verify_count_lineage`,
`verify_kind_of_yes_not_existence`, `synthetic_fixture_blind_regime_at_validator`,
`validator_check_assumes_dataset_regime`, `aggregate_signal_hides_subsets`.

## Authoritative files

- Spec: `docs/superpowers/specs/2026-05-31-phase-1-sub-G-cross-artifact-validator-design.md` (v3).
- Plan: `docs/superpowers/plans/2026-05-31-phase-1-sub-G-cross-artifact-validator.md` (v2).
- Code: `src/cfm/data/sub_g/*.py`; tests `tests/data/sub_g/*.py`; scripts `scripts/sub_g/*.py`.
- sub-F defect site: `src/cfm/data/sub_f/validator_cross_tile.py::_check_symmetry` (`:211-253`).

*Paused at the F1/F2 decision point. — end of mid-sub-G handoff.*
