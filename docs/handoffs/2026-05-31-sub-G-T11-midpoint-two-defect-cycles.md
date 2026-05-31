# Mid sub-G handoff — T11 real-data measurement, two defect cycles (cycle-1 fixed, cycle-2 scoped) — 2026-05-31

> **Cold-reader resume point.** Assume zero context from the session that wrote this.
> sub-G (cross-artifact consistency validator, PRD stage five) is BUILT (T1–T10) and its
> T11 real-data measurement is IN PROGRESS. The first real run surfaced two upstream
> defects. **Cycle 1 is fixed + committed; cycle 2 is characterized but NOT started.**
> This is NOT the T12 close handoff — sub-G has not written `_PHASE1_VALIDATED` and the
> chain has not completed a clean run. Read §1 (state), §2 (cycles), §3 (next step), §7
> (resume) first.

---

## 1. WHERE SUB-G IS

- **Branch:** `phase-1-sub-G-cross-artifact-validator` (off `main` @ sub-F merge). **Local
  only — NOT pushed** (per-sub-project branch discipline: push at sub-G close, not before).
- **Code: T1–T10 complete** — the validator is fully built. **51 sub-G unit tests pass**
  (`uv run pytest tests/data/sub_g -q` → `51 passed, 9 deselected`). The full sub-F+sub-G
  suite is green (`338 passed` at the cycle-1 commit).
- **T11 (real-data measurement) IN PROGRESS.** First real Singapore run (494 tiles) drove
  the sub-E→sub-F chain and surfaced two defect cycles (§2). sub-G's own validator has
  **never completed a clean run on real data**: no `_PHASE1_VALIDATED`, no
  `quarantine_report.yaml`, no `_PHASE1_ACCURACY_BASELINE.yaml` exist yet. The chain halts
  in sub-F's OWN `validate_cross_tile` before sub-G's seams run.
- **What sub-G is:** runs sub-E→sub-F derive on a region (resume-from-`_SUCCESS`), then
  validates three independent seams per tile, accumulates defects by signature, writes
  `quarantine_report.yaml` + `_PHASE1_ACCURACY_BASELINE.yaml`, gates with
  `_PHASE1_VALIDATED` (empty quarantine AND sanity floor intact). Module map:
  `src/cfm/data/sub_g/{diagnostics,buckets,readers,seam_macro_geometry,seam_contract_tokens,seam_decodability,versions,validator,pipeline,subset,cli}.py`;
  CLIs `scripts/sub_g/{derive_phase1_region,validate_phase1_region}.py`.
- **Sequencing (do not let this drift):** sub-F → **sub-G (here)** → **eval-set-generation**
  (the NAMED successor sub-project) → training-scaffold. Eval-set-generation is separate and
  comes BEFORE training-scaffold. Don't skip it.

## 2. THE TWO DEFECT CYCLES

Both are the same shape (see §4): a sub-F BP7 cross-tile validator **leg** that was CORRECT,
firing on real geometry an upstream stage mis-produced. The validator is not the bug in
either cycle.

### Cycle 1 — N/S direction convention flip — **COMMITTED `98cdeb0`**
- **Symptom:** sub-F `validate_cross_tile._check_symmetry` halted on real Singapore: 2728
  failed (cell,dir) pairs over 306/494 tiles; 100% on the N/S axis, 0 on E/W.
- **Root cause:** the sub-F encoder (`encoder._classify_feature_for_bref._direction_of_endpoint`)
  and, latently, sub-G seam-2 (`seam_contract_tokens._endpoint_edge`) labeled cell faces by
  **geographic compass** (cell-local y=extent → "N"). The BP7 direction **authority** is
  `sub_e.rotation.cell_to_edge_ids` — the locked `configs/sub_f/boundary_reference_vocab.yaml`
  `source_references` defer N/S/E/W to it — and it names a cell's **north edge = the edge
  shared with (i, j-1) = the LOW-y edge** (cell-local y=0). So the encoder looked up the
  boundary contract on the opposite j-edge → dropped/mislabeled every N/S bref.
- **Fix location:** swapped N/S in BOTH functions (y=0→"N", y=extent→"S") to conform to
  `cell_to_edge_ids`. Added external-source-of-truth tests
  (`tests/data/sub_f/test_direction_authority.py`, `tests/data/sub_g/test_seam2_direction_authority.py`)
  that build expected direction from `cell_to_edge_ids` (physical edge_ids from pure lattice
  geometry, never encoder output) — RED before, GREEN after. Corrected two vacuous fixtures
  (`test_seam_contract_tokens.py` N/S asserts; `test_pipeline_writer.py` south-entry road
  **moved** (100,0)→(100,250) because the south edge IS y=250 under the authority — data
  moved to match authority, not expectation relabeled).
- **Status:** COMMITTED `98cdeb0`. Re-derive confirmed the symmetry leg now passes (0
  failures). **v2 debt recorded:** the bref labels are geographically inverted (the
  authority's "north" points geographic south in SVY21); harmless in v1 because bref labels
  are internal stitching tags never surfaced in output (v1 decoder drops bref vertex
  position). **v2 trigger:** if bref direction labels ever surface in output (v2 decoder
  retains bref vertex position) OR an external consumer reads boundary-contract direction
  semantics → fix the inversion at `cell_to_edge_ids`/sub-D, which would invalidate
  v1-trained models (the bref tokens are trained token IDs). Full writeup:
  `reports/2026-05-31-sub-G-T11-symmetry-root-cause.md`.

### Cycle 2 — non-road edges mislabeled MINOR_ROAD — **CHARACTERIZED, NOT STARTED**
- **Symptom:** after cycle-1 fix, the re-derive got PAST `_check_symmetry` and halted on the
  next leg, `_check_coverage` (first at tile i10_j10 cell (7,4) edge S): an edge is active
  (MINOR_ROAD) with a road feature nearby but no `<bref>` emitted.
- **Root cause:** **sub-E violates its own locked spec §5.1.** §5.1
  (`docs/superpowers/specs/2026-05-20-phase-1-sub-E-boundary-contracts-design.md:238-278`)
  is explicit: *"Non-road crossings (water, rail) are ignored in v1. An edge with only water
  or rail crossings becomes NONE."* The implementation
  (`src/cfm/data/sub_e/pipeline.py::_derive_tile_rows`, L308-311) builds `features_by_id`
  from road features only (correct) but then iterates **all** crossings and appends
  `features_by_id.get(c.source_feature_id)` — which is **`None`** for a non-road crossing.
  `derive_boundary_class` (`src/cfm/data/sub_e/derivation.py:84-85`) maps `None` →
  **MINOR_ROAD** (the unknown-*road*-class default). So a water/building polygon whose
  body-chord lies co-linear along an edge yields interval crossing records → threaded in as
  `None` → spurious MINOR_ROAD edge → sub-F correctly emits no bref (non-road) → coverage
  check correctly fires. The `None` is **overloaded** (means both "road, unknown class →
  MINOR" and "non-road → should be excluded").
- **Prevalence:** **2016** coverage-failure edges over **259/494** tiles, **100% non-road**
  confirmed by TWO orthogonal discriminators that agree: by feature_class (building 1478 +
  base 468 + base&building 70 = 2016; **road = 0**) and by event_type (interval-only 2000 +
  mixed 16). The "16 mixed" are non-road features producing both interval+point events, not
  roads.
- **Fix location (sub-E):** exclude non-road crossings from the vote per §5.1 — at
  `pipeline.py:308-311`, build a road-source-id set and append a crossing's class only when
  its source feature is a road; do NOT append `None` for non-road crossings. Also FIX the
  miscited comment at `pipeline.py:319-324` (it cites §5.1 as authority for the OPPOSITE of
  what §5.1 says). Bump `boundary_derivation_version` (on-disk labels change: some
  MINOR_ROAD→NONE; the version axis exists for exactly this). Add the missing lock-and-guards
  test: a non-road-only active edge derives NONE (external-source-of-truth vs §5.1).
- **Status:** CHARACTERIZED only. No code touched. Needs human approval (sealed sub-E). Full
  writeup: `reports/2026-05-31-sub-G-T11-coverage-cycle2-nonroad-edge.md`.

## 3. SUB-E FIX BLAST RADIUS + SEQUENCE (cycle 2)

Larger than cycle 1 (which left sub-E untouched). This changes
`boundary_contract.parquet` content, so:

```
sub-E code fix + boundary_derivation_version bump + lock-and-guards test
  → regen sub-E cache (494 tiles)
  → re-derive sub-F (494 tiles)        [also re-applies the committed cycle-1 N/S fix]
  → re-validate via sub-G              [first real sub-G validator run, if the chain clears]
```
It also touches **sub-G seam-2 semantics** (the contract↔tokens bijection reads the
corrected contract).

**FIRST STEP IN THE NEXT SESSION — the seam-2-fixture check (do this BEFORE writing the fix):**
confirm `tests/data/sub_g/test_seam_contract_tokens.py` fixtures (the `build_cell_contracts`
and `predict_expected_brefs_per_cell` tests) do NOT encode a buggy MINOR_ROAD-on-a-non-road-edge
as expected behavior. If any fixture activates an edge with only non-road crossings and
expects MINOR_ROAD, it is **vacuously passing** the same way the cycle-1 N/S fixtures were —
correct it **against §5.1** (an edge with only non-road crossings is NONE), i.e. move/adjust
the fixture data to match the authority; do NOT relabel the expectation to keep it green.
Same discipline as the cycle-1 `test_pipeline_writer.py` correction (moved the road to the
real south edge rather than relabeling N→S).

## 4. THE META-PATTERN (expect more)

Both cycles are identical in shape: **a BP7 cross-tile validator leg encoded a data-regime
assumption that the synthetic T10 fixtures never exercised.** Real Singapore geometry has
roads that (a) **terminate** on edges, (b) get **clipped** near edges, (c) run **along**
edges (co-linear), and (d) **cross** edges — four regimes; the synthetic fixtures only hit
the clean-crossing one. In BOTH cycles the validator was CORRECT and the UPSTREAM stage had
the bug. **A cycle 3 is plausible** once the sub-E fix lets the chain run further — the
remaining §8 inherited item is **multi-part / motorway handling** (motorway is MINOR-tiered
for v1 as a scoped accept; multi-part LineString emission is code-inferred, not
parquet-observed). **Posture: characterize, don't push through.** The first real run is
measurement / instrument, not pass/fail verification — every halt is the measurement doing
its job, finding a real upstream defect cheaply before training-scaffold.

## 5. TOOLING CAVEAT (keep this posture)

This session had a real tool-output error rate: hallucinated/duplicated stdout from commands
that never ran (`timeout` is absent on macOS → exit 127 with stale-looking output),
output-stream interleaving from background jobs, and one classifier-outage cancellation.
**Three intermediate conclusions were retracted after cleaner reads** (a `_neighbour_cell`
inversion; a "2680/48" axis split; "fix sub-E `cell_to_edge_ids`" / "§5.1 is the defect" —
both from inferring a clause from downstream citations instead of reading it). The discipline
that caught all three BEFORE they became recommendations: **sentinel-terminate each read**
(`echo END_X`), **cross-check load-bearing numbers across ≥2 reads / 2 discriminators**, and
**verify premises against source, not citations** (`feedback_citation_is_not_the_clause`).
`scripts/sub_g/t11_symmetry_diagnosis.py` reproduces cycle-1 in one read-only command if the
next session wants to re-confirm.

## 6. ARTIFACTS + STATE

- **Committed:** `98cdeb0` (cycle-1 N/S fix; 6 files: encoder, seam-2, 2 corrected fixtures,
  2 new authority tests). Branch `phase-1-sub-G-cross-artifact-validator`, **local, NOT
  pushed** (push at sub-G close).
- **Untracked — CARRY to T12 close / measurement artifacts:**
  `reports/2026-05-31-sub-G-T11-symmetry-root-cause.md`,
  `reports/2026-05-31-sub-G-T11-coverage-cycle2-nonroad-edge.md`,
  `scripts/sub_g/t11_symmetry_diagnosis.py`, and THIS handoff.
- **DELETE on next tree touch:** `reports/sub_g_t11_f2_drill.txt`,
  `reports/sub_g_t11_f2_drill_3tiles.txt` (output of a discarded buggy drill — unreliable
  vertex selection; superseded by `t11_symmetry_diagnosis.py`).
- **Re-derived sub-F cache on disk** (`data/processed/sub_f/2026-04-15.0/singapore/`) has the
  cycle-1 N/S fix applied to all 494 tiles but **NO `_SUCCESS`** (halted at the coverage
  leg). Incomplete by design; the sub-E fix requires re-deriving it from scratch anyway.
  sub-C/sub-D/sub-E caches present with `_SUCCESS`.
- **Memories saved this session** (all in the memory dir, indexed in `MEMORY.md`):
  `feedback_citation_is_not_the_clause`, `feedback_loud_false_positive_masks_quiet_defect`,
  `feedback_independence_misses_shared_assumptions`,
  `feedback_tool_output_trustworthiness_layer`, `feedback_root_cause_changes_sibling_fix`,
  `feedback_inspection_script_premise_check`,
  `feedback_synthetic_fixture_blind_regime_at_validator`,
  `feedback_validator_check_assumes_dataset_regime`, `feedback_aggregate_signal_hides_subsets`,
  `feedback_tool_output_silence_is_not_pass`,
  `feedback_precondition_verify_count_not_estimate`, `feedback_verify_count_lineage`,
  `feedback_verify_kind_of_yes_not_existence` (+ the `project_sub_g_before_training`
  sequencing note).

## 7. RESUME INSTRUCTION

Open by scoping the **sub-E §5.1 compliance fix** (cycle 2):
1. **First: the seam-2-fixture check** (§3) — audit `test_seam_contract_tokens.py` for vacuous
   MINOR_ROAD-on-non-road-edge fixtures; correct against §5.1 if found.
2. Write the fix (`pipeline.py:308-311` exclude non-road from the vote; fix the miscited
   `pipeline.py:319-324` comment) + the lock-and-guards test (non-road-only edge → NONE) +
   `boundary_derivation_version` bump, in one commit.
3. **Human approval on the diff** (sealed sub-E) — reviewer-in-the-loop; Umar pastes decision
   points. No sealed-code change before approval.
4. Then the cascade: regen sub-E → re-derive sub-F → re-validate sub-G. Verify each stage
   (sentinel-terminated reads; expect the coverage halt to clear; watch for cycle 3 /
   multi-part-motorway).
5. If sub-G writes `_PHASE1_VALIDATED` on a clean run → that is the real measurement; proceed
   toward T12 close (protocol-v2 candidate bump) and then the eval-set-generation
   sub-project.

The fix itself is clear and small; the discipline lives in (a) correcting fixtures against
the authority (not to-green) and (b) verifying the regen→re-derive→re-validate cascade rather
than assuming it.

*Paused after committing cycle-1 (`98cdeb0`); cycle-2 sub-E fix scoped and awaiting the next session. — end of mid-sub-G handoff.*
