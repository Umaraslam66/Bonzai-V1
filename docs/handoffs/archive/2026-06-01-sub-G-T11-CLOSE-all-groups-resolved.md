# sub-G CLOSE — all 5 quarantine groups resolved, validator clean on 494 tiles, `_PHASE1_VALIDATED` written; MERGE-TO-MAIN PENDING reviewer go — 2026-06-01

> **Cold-reader resume point.** sub-G (cross-artifact consistency validator, PRD
> stage five) is BUILT and now VALIDATED on Singapore (494 tiles). Its T11
> real-data measurement found 5 quarantine groups + a sanity-floor violation across
> three hypotheses (H1 accuracy, H2 bref-bijection, H3 OGC-validity) plus one POI
> alpha-drop advisory. All are now resolved. The validator reports
> `passed=True groups=0 sanity_floor_violated=False`; `_PHASE1_VALIDATED` is written
> to the canonical sub_g dir. **The only thing left is the merge of branch
> `phase-1-sub-G-cross-artifact-validator` → `main`, held for explicit reviewer go
> (the PRD §11 merge gate).** Read §1 (state), §2 (the 5 groups + dispositions), §3
> (the two known-limitation exclusions), §4 (POI), §5 (what's pending).

## 1. STATE

- **Branch:** `phase-1-sub-G-cross-artifact-validator` (off `main`). **Local only —
  NOT pushed.** Merge to main is the pending step.
- **Validator (494 tiles, validator_version 1.1.0):**
  `passed=True groups=0 sanity_floor_violated=False`. `_PHASE1_VALIDATED` written to
  `data/processed/sub_g/2026-04-15.0/singapore/` (gitignored).
- **Accuracy baseline:** position_core p99.9 **3.61m** (≪50m), angle_core p95
  **0.99°** (≪20°), position_full p99.9 **229m** (reported bref residual),
  `ogc_bref_collapse_excluded_from_gate: 27958`.
- **Tests:** sub-E/F/G **441 passed** (`uv run pytest tests/data/sub_e tests/data/sub_f
  tests/data/sub_g -m "not slow" -q`), ruff clean.
- **This-session commits (local):** `0e8a359` (H3 fix), `da8aea5` (POI advisory
  resolution). Prior: `07e37fe` (H1), `443cec7` (H2), plus the handoff/cycle commits.
- **Protocol bumped:** `docs/protocols/sub-project-planning-protocol-v2.md` adds §9
  (construction-identity exclusion with regime-distinguishing guard).

## 2. THE 5 QUARANTINE GROUPS → 0

First measurement: `passed=False groups=5 sanity_floor_violated=True`.

| # | Seam · signature | Instances | H | Disposition |
|---|---|---|---|---|
| 1 | accuracy sanity floor (pos p99.9 318.7m; angle p95 179.9°) | 1 | H1 | **FIXED** — validator measurement bug (Multi* mispairing + index-positional vertex metric). Geometry-aware symmetric Hausdorff vs the canonical original; core/full split; gate floor on core. `07e37fe` |
| 2 | bref missing (sub-F dropped) | 1,325 | H2 | **FIXED** — sub-G false positive: `_endpoint_edge` used 0.5m vs encoder's 1e-6m on-edge tol. Shared `ON_EDGE_EPS_M`. `443cec7` |
| 3 | bref multiset mismatch | 282 | H2 | FIXED (same root cause) |
| 4 | bref extra (sub-F invented) | 42 | H2 | FIXED (same root cause) |
| 5 | decoded geometry not OGC-valid | 27,958 | H3 | **KNOWN-LIMITATION EXCLUSION** — v1 outbound-bref placeholder collapse; report-not-gate by construction identity. `0e8a359` |

H1/H2 were the validator firing on its OWN wrong premise (measurement bug, tolerance
mismatch) — fixed, both carry a guard proving the gate still fires on a real defect.
H3 is a genuine v1 limitation, excluded (not fixed) — see §3.

## 3. THE TWO KNOWN-LIMITATION EXCLUSIONS — ONE v1 LIMITATION, TWO SEAMS

The v1 micro-tokenizer drops the position of a road's edge-crossing vertex by design
(`<bref>` = direction + class, not position; v2-scoped per spec §1.4). This single
limitation surfaced in two seams; both excluded by CONSTRUCTION IDENTITY (never
magnitude), each with a regime-distinguishing guard:

- **H1 (accuracy):** the bref VERTEX is excluded from the gated CORE metric
  (`_has_outbound_bref`), reported in FULL (p99.9 229m). Guard
  `test_feature_accuracy_core_FIRES_on_displaced_non_bref_vertex`.
- **H3 (decodability):** the SAME vertex collapses a V=2 crossing road to a
  zero-length `[anchor,anchor]` LineString; excluded from the OGC gate
  (`_is_bref_placeholder_collapse` = `_has_outbound_bref` AND `<2 distinct vertices`),
  count reported (`ogc_bref_collapse_excluded_from_gate`). Guard
  `test_check_decodability_GATE_FIRES_on_degenerate_without_outbound_bref` (a
  synthetic no-bref zero-length twin still quarantines — two blocks, one geometry).

The 27,958 ⊆ the 229m full-residual family — same crossing roads. A read-only drill
reproduced the gate's invalid set bit-identically (27,958 == 27,958) and found 100%
construction-identity, 0 genuine-degeneracy remainder. Reports:
`reports/2026-06-01-sub-G-T11-{H1,H2,H3}-*.md`. This discipline is now protocol §9.

## 4. POI ALPHA-DROP — confirmed density artifact, ACCEPTED

`docs/known_issues.md`: the POI 10.69% over-drop is a density-correlation artifact,
not a POI-specific over-drop or budget bug. A dropped-cell × POI-density cross-tab
(reproduced the official 36-cell / 15,991-POI aggregate exactly) shows POIs cost the
7-token floor; the drop rule is type-agnostic (`total cell tokens > 5760`); 9/36
dropped cells are building-dense with ~2–4 POIs; retained cells exist with up to 593
POIs. POIs over-represent only because they cluster in the densest cells. Accepted;
no code change. `da8aea5`.

## 5. WHAT'S PENDING — MERGE TO MAIN (reviewer go)

Reviewer chose "write marker, HOLD merge" (2026-06-01). Pre-conditions cleared before
the marker was written: (1) both exclusion guards green by name; (2) `groups: []`
certified on the canonical 494-tile output at the committed code; (3) byte-determinism
spot-check (two independent runs → identical quarantine_report.yaml + baseline +
identical marker content_digest).

**Next action:** on reviewer go, merge `phase-1-sub-G-cross-artifact-validator` → `main`
(local merge, per branch pattern — local-first, no PR/push). After merge, the next
sub-project is **eval-set generation** (separate sub-project; precedes training
scaffold — see `project_sub_g_before_training`). Read protocol v2 §9 before its
brainstorm.

*sub-G is the validator half of the PRD §11 gate. With `_PHASE1_VALIDATED` written and
the merge pending, Phase 1 stage five is functionally complete.*
