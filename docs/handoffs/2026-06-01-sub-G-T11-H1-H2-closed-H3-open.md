# Mid sub-G handoff — H1 (accuracy) + H2 (bref-bijection) CLOSED; H3 (OGC-validity 27,958) is the SOLE remaining group — 2026-06-01

> **Cold-reader resume point.** Assume zero context. sub-G (cross-artifact
> consistency validator, PRD stage five) is BUILT. Its T11 real-data measurement
> on Singapore (494 tiles) is the live work. The prior handoff
> (`2026-06-01-sub-G-T11-subF-validated-subG-first-measurement.md`) left sub-G's
> first run with **5 quarantine groups + a sanity-floor violation** and three
> hypotheses (H1 accuracy, H2 bref-bijection, H3 OGC-validity). This session
> **closed H1 and H2** — both were the validator firing on its OWN wrong premise,
> not upstream defects. The validator now reports **1 group, sanity floor clean**.
> H3 is the only group left. Read §1 (state), §2 (what shipped), §3 (H1+H2), §4
> (H3 resume — drill cold), §5 (re-run), §6 (posture).

---

## 1. WHERE THINGS ARE

- **Branch:** `phase-1-sub-G-cross-artifact-validator` (off `main`). **Local only —
  NOT pushed** (push at sub-G close).
- **Validator state (494 tiles, validator_version 1.1.0):**
  `sub-G validate: passed=False groups=1 sanity_floor_violated=False`.
  Was 5 groups + floor violation. **Only the OGC-validity group remains** (H3).
- **Accuracy baseline** (`data/processed/sub_g/2026-04-15.0/singapore/_PHASE1_ACCURACY_BASELINE.yaml`,
  gitignored): position_core p99.9 **3.61m** (≪50m), angle_core p95 **0.99°**
  (≪20°), n=862,436; position_full p99.9 229m (reported bref residual). Monotonic.
- **Tests green:** sub-G + sub-E + sub-F = **437 passed** (`uv run pytest
  tests/data/sub_e tests/data/sub_f tests/data/sub_g -m "not slow" -q`). ruff clean.
- **No `_PHASE1_VALIDATED`** yet (H3 group still open).

## 2. WHAT SHIPPED THIS SESSION (committed, local)

| Commit | What |
|---|---|
| `07e37fe` | **H1 fix** — seam-3 accuracy: geometry-aware symmetric Hausdorff vs the CANONICAL original + multi-part pairing; core(gated)/full(reported) split; floor gates on CORE; `_percentile` made NaN-safe; VALIDATOR_VERSION 1.0.0→1.1.0. |
| `443cec7` | **H2 fix** — seam-2 on-edge tolerance: extract encoder `ON_EDGE_EPS_M=1e-6` (behavior-identical), sub-G `_EDGE_TOL_M` IMPORTS it (one source); two-directional guard + single-source lock. Folds into 1.1.0. **Touches sealed sub-F encoder.py** (constant extraction only — surfaced + approved). |

Reports: `reports/2026-06-01-sub-G-T11-H1-accuracy-metric-root-cause.md`,
`reports/2026-06-01-sub-G-T11-H2-bref-bijection-tolerance.md`. All investigation
drills were deleted (scaffolding); the corrected behavior lives in the tests.

## 3. H1 + H2 — both were the validator firing on its OWN wrong premise

**H1 (accuracy 318m/179.9°):** NOT the canonicalization artifact the prior handoff
guessed, NOT a broken decoder. Three causes: (1) Multi\* feature **mispairing**
(`encode_cell` splits a Multi\* sub-C row into one block PER PART; sub-G paired
`decoded[k]↔sub_c[k]` by index), (2) index-positional vertex metric (broken by
canonicalize-reorder + chunking), (3) the v1-by-design **unencoded outbound bref
vertex** (known, not a bug; decoder.py:13-22). Fix measures geometry-aware
Hausdorff vs canonical original, reports core+full, gates floor on CORE, excluding
the bref vertex by **construction identity** (`_has_outbound_bref`, never
magnitude) — guard test fires core on a non-bref displacement. **Loud-masks-quiet:
the mispairing artifact (all types ~300m) hid a genuine — but designed — crossing-
road residual (full p99.9 229m).**

**H2 (bref-bijection 1,649):** a sub-G **false positive** — sub-G `_endpoint_edge`
used 0.5m vs the encoder's 1e-6m on the SAME raw coords; a near-corner endpoint
(e.g. (0.071,0.0): exactly on N(y=0), 0.071 off W) was misattributed to W. Ruled
out by verification (not assumption): road-gate (encoder `is_road` ≡
`feature_class==0` via `_semantic_tag_from_row`), contract (0/15,360
disagreements), real drop (encoder emission ≡ tokens). Fix: cell edge is a
STRUCTURAL boundary → structural epsilon, shared via one constant.

**Pattern:** both fixes corrected the validator's premise; neither weakened a
gate (both carry guards proving the gate still fires on a real defect). Each was
verified against SOURCE, RED-before-GREEN, confirmed on all 494 tiles.

## 4. H3 RESUME — OGC-validity 27,958 (drill COLD; it's the subtlest)

`quarantine_report.yaml` group: `invariant_name: decodable_to_valid_geojson`,
`signature: decoded geometry not OGC-valid`, instance_count **27,958** (3.2% of
862,436). Gate: `seam_decodability.py` ~line 148, `shape(geom).is_valid` for
decoded type in (Polygon, LineString).

**Carry these four, all live:**

1. **TWO branches, both live — do NOT pre-frame as a third false-positive just
   because H1/H2 were.** Different seam (decoded-geometry validity, not endpoint
   attribution). (a) **Real decode degeneracy** vs (b) the **gate mis-judging the
   decoder's convention**: `decode_feature` NEVER returns `Polygon` — a closed
   building ring comes back as a `LineString` "by default" (decoder.py:144-155),
   and Case B/D crossing roads append a placeholder duplicate vertex
   (decoder.py:128-134). So every geometry the gate sees is a LineString. Note
   shapely `LineString.is_valid` is False essentially ONLY for degeneracy
   (<2 distinct points / zero-length / NaN) — a self-intersecting LineString is
   VALID. So 27,958 invalid ⇒ 27,958 degenerate decoded LineStrings.

2. **Loud-masks-quiet, now twice-seen (H1, H2).** The headline 27,958 is a
   symptom; **characterize by ACTUAL geometry shape before concluding** —
   zero-length `[(x,y),(x,y)]` / ≤1-distinct-vertex / closed-ring-flagged. The
   prime suspect is the **2-vertex crossing-road bref placeholder** (anchor +
   duplicate → zero-length); the H1 full-NaN geoms (~6/126k in a sample) are a
   subset, but 3.2% is much larger — decompose it, don't assume one mode.

3. **If H3 splits into "gate bug" + "real degeneracy" (like H1 split into
   measurement-bug + designed-residual), the real-degeneracy remainder may itself
   need a gate-vs-known-limitation DECISION** — analogous to the bref-vertex call.
   If the degeneracy is *only* because of the v2-scoped unencoded bref vertex (a
   designed v1 limitation), quarantining it as "not decodable to valid GeoJSON"
   may be the gate over-reaching the same way the sanity floor did. Flag it as a
   possible reviewer decision; do NOT assume it's purely a gate fix.

4. **Then POI alpha-drop** 10.7% (`docs/known_issues.md`; one-line hypothesis =
   density-correlation), **then sub-G close → `_PHASE1_VALIDATED` → merge gate**
   (PRD §11; reviewer approval before merge to main).

**First H3 step:** a read-only drill — for the 27,958, decode each, bucket by
(n_distinct_vertices, is_closed, has_outbound_bref, feature_class). That table
decides branch (a) vs (b) and whether a known-limitation decision is needed.

## 5. RE-RUN COMMANDS

- **sub-G-only (fast loop, ~8–11 min)** — when only sub-G code changed:
  `uv run python scripts/sub_g/validate_phase1_region.py --region singapore
  --release 2026-04-15.0 --sub-c-region-dir <C> --sub-d-region-dir <D>
  --sub-e-region-dir <E> --sub-f-region-dir <F> --output-dir <G>` (paths under
  `data/processed/<stage>/2026-04-15.0/singapore`). Re-runs `validate_region`
  only; does NOT regenerate sub-E/sub-F.
- **Full chain (regen everything):** `scripts/sub_g/derive_phase1_region.py … --force`.
- Launch long runs in the background; verify the on-disk `quarantine_report.yaml`
  + `_PHASE1_ACCURACY_BASELINE.yaml` + exit code, NOT the traceback alone
  (`feedback_tool_output_trustworthiness_layer`).

## 6. POSTURE (unchanged)

Characterize, don't push through; verify against SOURCE not citations; RED-before-
GREEN; every gate fix carries a guard proving it still fires on a real defect
(`feedback_structural_exclusion_not_magnitude`, `feedback_gate_must_distinguish_regimes`).
sub-G code is ACTIVE (TDD, no seal-approval); sub-E/sub-F are SEALED (surface
diffs for approval — H2's encoder.py constant extraction was behavior-identical +
surfaced). Each fix may surface the next finding — first-real-data runs cluster.

*Paused at a clean boundary after H2 (`443cec7`): 5 quarantine groups + floor →
1 group, floor clean; H1+H2 closed and verified on 494 tiles; H3 is the sole
remaining group, to be drilled cold. — end of handoff.*
