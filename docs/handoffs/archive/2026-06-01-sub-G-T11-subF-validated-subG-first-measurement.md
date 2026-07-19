# Mid sub-G handoff — sub-F VALIDATED on 494 tiles; sub-G's own validator ran end-to-end, first real measurement (5 finding groups) — 2026-06-01

> **Cold-reader resume point.** Assume zero context. sub-G (cross-artifact
> consistency validator, PRD stage five) is BUILT (T1–T10). Its T11 real-data
> measurement on Singapore (494 tiles) is the live work. This session drove the
> sub-E→sub-F→sub-G chain from a halt-on-symmetry state all the way THROUGH a
> **clean, non-vacuous sub-F cross-tile validation** and into **sub-G's own
> validator running end-to-end** for the first time. sub-G `passed=False` with
> **5 quarantine groups + a sanity-floor violation** — the validator WORKING and
> producing its first measurement, not crashing. Read §1 (state), §2 (what
> shipped), §3 (the milestone), §4 (the 5 findings + hypotheses), §5 (resume).

---

## 1. WHERE THINGS ARE

- **Branch:** `phase-1-sub-G-cross-artifact-validator` (off `main` @ sub-F merge).
  **Local only — NOT pushed** (push at sub-G close).
- **sub-F is VALIDATED on real Singapore (494 tiles).** sub-F
  `data/processed/sub_f/2026-04-15.0/singapore/` has `_SUCCESS` + `manifest.yaml`
  (`sub_f_derivation_version: 1.1`, `sub_f_validator_version: 1.1`). The
  validate-then-touch discipline means **`validate_cross_tile` passed on all 494
  tiles** (cross-reference, symmetry, coverage, non-road). Verified
  NON-VACUOUSLY (§3). sub-E cache is regenerated to `boundary_derivation_version
  1.1` (also `_SUCCESS`).
- **sub-G's validator ran end-to-end** on the 494-tile sub-F output and wrote
  `data/processed/sub_g/2026-04-15.0/singapore/{quarantine_report.yaml,
  _PHASE1_ACCURACY_BASELINE.yaml}`. **NO `_PHASE1_VALIDATED`** (passed=False).
  This is the chain runner's `EXIT_QUARANTINE` (1) — validator ran, found defects.
- **Tests green:** `uv run pytest tests/data/sub_e tests/data/sub_f
  tests/data/sub_g -m "not slow" -q` (sub-G now **62** with the cycle-5 tests).
- **Full suite green at each commit** this session (sub-F 281, sub-G 62).

## 2. WHAT SHIPPED THIS SESSION (all committed, local)

| Commit | What | Sealed? |
|---|---|---|
| `98cdeb0` | (prior session) cycle-1 N/S direction convention fix (encoder + seam-2) | sealed sub-F |
| `99f9e43` | **cycle-2**: sub-E excludes non-road crossings from boundary-class vote (§5.1); `boundary_derivation_version` 1.0→1.1 + 6 guards + lock-and-guards tests | sealed sub-E |
| `d157e47` | sub-F `SUB_F_DERIVATION_VERSION` 1.0→1.1 (closes cycle-1's reproducibility gap — cycle-1 changed encoder output without a version bump) | sealed sub-F |
| `c9f623c` | **cycle-3**: sub-F validator resolves BP4 `<unknown_highway>`→`highway` (false-positive non-road halt); `SUB_F_VALIDATOR_VERSION` 1.0→1.1 (verdict-only) | sealed sub-F |
| `5b44cfd` | **cycle-4**: encoder gates bref emission on road key; **promotes `vocab.semantic_tag_to_l1_key` to the SINGLE road-key authority** used by BOTH encoder + validator (kills the re-determine-road-ness-locally bug class). No version move (folds into pre-ship 1.1, verified) | sealed sub-F |
| `56d422b` | tooling: alpha-drop advisory report made non-fatal (import moved inside try) + repo root on sys.path so `scripts` namespace pkg imports; + coverage non-vacuity drill | sub-F + scripts |
| `6f2473a` | **cycle-5**: sub-G decodability seam `_original_coords` recurses Multi* first-part through per-type dispatch (MultiPolygon original crashed shapely `Polygon.coords`); + POI alpha-drop open item | **sub-G (active, not sealed)** |
| (`e669a50`) | carry: pre-flight absent-feature drill + §3.7 subway erratum | docs/scripts |

**Cycle shape (all five):** a real-data run halted; the halt was a real upstream
defect (cycles 1–4) or sub-G's own seam bug (cycle 5), NOT the validator
over-firing. Each fix: characterize against SOURCE (read the clause, not the
citation), RED-before-GREEN lock-and-guards test, surface sealed-code diffs for
approval before applying.

## 3. THE MILESTONE — sub-F validation passed NON-VACUOUSLY (locked)

"Didn't raise" ≠ "ran non-vacuously." Confirmed both, on real data
(`scripts/sub_g/t11_coverage_nonvacuity_drill.py`, reusing the validator's own
helpers + mirroring `_check_coverage` exactly):
- **Coverage leg evaluated 41,252 active road edges** across **330/494 tiles**,
  **all covered, 0 uncovered**. Not "found zero edges to check."
- **Independent cross-check:** that 41,252 equals the cycle-4 drill's count of
  `(cell,dir)` bref groups — two drills, same number, agree.
- **Motorway (4,929 features) and multi-part (5,605 MultiLineString roads)
  regimes ACTUALLY APPEARED** in the 494 tiles and passed — not absent. Road geom
  mix: 296,666 LineString + 5,605 MultiLineString.

So cycles 1–4 cleared every sub-F cross-tile data leg, with the motorway and
multi-part regimes exercised. **sub-F data-cycle chain is genuinely done.**

## 4. SUB-G'S FIRST MEASUREMENT — 5 finding groups (the live work)

`sub-G derive: passed=False groups=5 sanity_floor_violated=True`.
From `data/processed/sub_g/2026-04-15.0/singapore/quarantine_report.yaml`:

| Seam | Signature | Count |
|---|---|---|
| **decodability** | decoded geometry not OGC-valid | **27,958** |
| **contract↔tokens** (bref bijection) | bref missing (sub-F dropped) | 1,325 |
| contract↔tokens | bref multiset mismatch (missing+extra) | 282 |
| contract↔tokens | bref extra (sub-F invented) | 42 |
| **accuracy sanity floor** | position p99.9 318.7m > 50m; angle p95 179.9° > 20° | 1 |

`_PHASE1_ACCURACY_BASELINE.yaml`: position **p95 227.4m / p99.9 318.7m**, angle
**p95 179.9° / p99.9 180.0°**, **n=862,436** features, **structural_bound_breaches 0**,
validator_version 1.0.0.

### THREE HYPOTHESES — carry as hypotheses-to-VERIFY, not conclusions

**H1 — Accuracy 318m/180° is most likely a sub-G COMPARISON artifact (not a real
decode defect).** The encoder quantizes to 0.5m, so 318m CANNOT come from
round-trip quantization. 318m < cell diagonal (354m) and 180° = reversed
traversal — the signature of **comparing mis-ordered vertices**.
`_accuracy_record` (`seam_decodability.py:94-110`) matches `decoded[i]` vs
`original[i]` POSITIONALLY (line 100-102), but the encoder **canonicalizes**
(lex-min start, CCW for polygons; `canonicalize_geometry`) BEFORE encoding, so
decoded vertex order legitimately differs from source order.
- **Drill:** one feature — compare canonical-to-canonical (canonicalize the
  original before `_original_coords`) or match set-wise. If confirmed, the fix is
  in **sub-G's comparison**, NOT the decoder.

**H2 — bref bijection (1,649) is the REAL tension; both branches live.** sub-F's
OWN cross-reference leg PASSED (every emitted bref agrees with sub-E's contract),
yet sub-G's bijection reports 1,325 missing + 42 extra + 282 mixed. They are
DIFFERENT comparisons: sub-G **predicts** brefs from geometry
(`seam_contract_tokens.py::predict_expected_brefs_per_cell`) while sub-F checks
emitted-vs-contract-CLASS (`validator_cross_tile.py::_check_cross_reference`).
- Branch A: sub-G's prediction **reimplements emission and drifted** from the
  encoder's actual logic — this is the **cycles-3/4 reimplementation bug class, a
  THIRD instance** if so (both prior were "re-determine road-ness locally"). Note
  cycle-4 changed the ENCODER's emission gate; did sub-G's prediction get the
  same `semantic_tag_to_l1_key` + endpoint logic? Likely NOT fully.
- Branch B: real drops/inventions sub-F's class-only check structurally cannot
  see (it checks class agreement, not presence-from-geometry).
- **Drill decides which. Do NOT assume the benign one.**

**H3 — OGC-validity 27,958 (3.2% of 862,436) is independent.** Real
self-intersecting decoded polygons vs a decoder bug. Check at
`seam_decodability.py:145` (`shape(geom).is_valid`). Separate drill.

## 5. RESUME INSTRUCTION (next session)

**Sequence (cheapest-scariest-number first, then highest-information):**
1. **Drill H1 (accuracy)** first — most likely an artifact; clears the 318m/180°
   scare cheaply. One feature: decode vs original, compare canonical-to-canonical.
   If artifact → fix `_accuracy_record`/`_original_coords` comparison + re-run.
2. **Drill H2 (bijection)** — highest information: tells us if reimplementation
   drift struck a third time. Pick a "bref missing" instance, hand-trace
   `predict_expected_brefs_per_cell` vs the encoder's actual emission for that
   cell×edge. Verify against the encoder, not sub-G's prediction.
3. **Drill H3 (OGC-validity)** — pick an invalid decoded geometry; is it a real
   self-intersection or a decoder bug?
4. **POI alpha-drop open item** (`docs/known_issues.md`): POI drop 10.7% vs <1.5%
   elsewhere; one-line hypothesis = density-correlation (POIs cluster in dense
   high-budget warning-band cells). Confirm by cross-tab vs cell density at close.

**Posture (unchanged):** characterize, don't push through; verify against SOURCE
not citations; RED-before-GREEN; sub-G code is ACTIVE (fixable with TDD, no
seal-approval), sub-E/sub-F are SEALED (surface diffs for approval). Each fix may
surface the next finding — first-real-data runs cluster findings.

**Re-run commands:**
- Full chain (regen everything): `uv run python scripts/sub_g/derive_phase1_region.py
  --region singapore --release 2026-04-15.0 --sub-c-region-dir <C> --sub-d-region-dir
  <D> --sub-e-region-dir <E> --sub-f-region-dir <F> --output-dir <G> --force`
  (paths under `data/processed/<stage>/2026-04-15.0/singapore`).
- **sub-G-only re-run** (when only sub-G code changed): drop `--force` → sub-E/sub-F
  skip on `_SUCCESS`, only `validate_region` re-runs (~6 min). This is the fast loop
  for H1–H3 fixes.
- Long runs: launch in background, read `/tmp/*.log` + the on-disk `_SUCCESS`/
  `quarantine_report.yaml` to confirm (don't trust the traceback alone — verify the
  artifacts; `feedback_tool_output_trustworthiness_layer`).

**Reproducible drills committed (`scripts/sub_g/`):**
`t11_preflight_absent_feature_crossings.py` (cycle-2 gate, 0 absent),
`t11_cycle4_coverage_safety_drill.py` (cycle-4 coverage-safety + non-road
footprint), `t11_coverage_nonvacuity_drill.py` (milestone confirmation + regime
presence).

## 6. META-PATTERN

A first-real-data run on a validator surfaces CLUSTERED findings (sub-F cycles
1–4; now sub-G's 5 groups). **sub-G hardening is its own phase**, analogous to the
sub-F cycle chain — its three seams (macro-geometry, contract↔tokens,
decodability) had never run on real data; two of the three plus the accuracy
floor fired on the first run. Expect H1–H3 fixes to surface further findings
(e.g., the macro-geometry seam's behavior is still largely unobserved). The
measurement is the instrument; every halt/finding is it doing its job cheaply
before training-scaffold.

*Paused at a clean boundary after cycle-5 (`6f2473a`): sub-F validated
non-vacuously, sub-G's validator ran end-to-end, first measurement banked. — end
of handoff.*
