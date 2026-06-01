# sub-G T11 H1 — the 318m/180° accuracy sanity-floor violation is a sub-G measurement bug, not a broken decoder

**Date:** 2026-06-01 · **Branch:** `phase-1-sub-G-cross-artifact-validator` · **Region:** singapore (494 tiles, release 2026-04-15.0)
**Validator:** `VALIDATOR_VERSION` 1.0.0 → **1.1.0** (semantic change to seam-3 accuracy).

## Finding (overturns the handoff's H1)

The handoff's first measurement reported `accuracy_sanity_floor` violated: position p99.9 **318.7m** (> 50m) and angle p95 **179.9°** (> 20°), on n=862,436 features. The handoff hypothesised a **canonicalization** comparison artifact. Drilling on real Singapore data **refuted** that and found the real cause has three independent parts — **none a broken decoder**:

1. **Feature mispairing (sub-G bug).** `encode_cell` (encoder.py:578-601) emits **one `<feature>` block per part** of a Multi\* geometry, but sub-C stores a Multi\* as **one row**. sub-G's `check_decodability` paired `decoded[k] ↔ sub_c_features[k]` by index, so every feature after the first Multi\* in a cell was compared against the **wrong** original (29% of cells affected; +1,113 decoded blocks over 76,094 sub-C features in a 30-tile sample, from 971 Multi\* features).
2. **Index-positional vertex metric (sub-G flaw).** Even correctly paired, polygon canonicalization reorders ring vertices (lex-min rotation + CCW winding → 180° on reversed rings) and long-segment chunking adds collinear vertices, so a `decoded[i]`-vs-`original[i]` match compares non-corresponding vertices.
3. **bref-vertex placeholder (known v1 limitation, NOT a bug).** Crossing roads (Case B/D) don't encode the exit edge-crossing vertex's *position* — only its edge + class (decoder.py:13-22, v2-scoped per spec §1.4). The decoder emits a placeholder. Error up to ~cell-diagonal. This is documented and explicitly **not** L_inf-gated in v1.

Evidence (30-tile sample, reproduces the 494-tile baseline under the old metric: ALL pos p99.9 = 320m, angle p95 = 179.8°):
- Refutation: **roads** (canonicalize is a no-op for LineStrings) and **POIs** (single point, no order) both show ~300m under the old metric; order-invariant vertex Hausdorff and canonical-aligned matching do **not** collapse it → not an ordering artifact.
- Residual isolation: with correct multi-part pairing + geometry-aware distance, the residual lives **entirely** in crossing roads — Case A (`nobref`) roads round-trip to p99.9 **3.27m** / max 4.48m (0% > 5m); Case B/D (`bref`) roads p99.9 **271m** (51% > 5m). The worst cases are 2-vertex roads whose only error is the unencoded exit vertex.

## Fix (sub-G code, TDD; sub-E/sub-F untouched)

Seam-3 accuracy is now **geometry-aware** and reports **two** baselines:
- **Pairing:** walk sub-C features, advance the decoded pointer by `#parts of canonicalize(orig)` (not `decoded[k]↔sub_c[k]`).
- **position_core_m** (gated): symmetric vertex Hausdorff between decoded and the **canonical** original, **excluding** the v1-unencoded outbound bref vertex.
- **position_full_m** (reported, not gated): includes it, so the bref residual stays visible.
- **angle_core_deg** (gated): max segment-bearing diff vs the canonical original, defined only where vertex counts match (no chunking) → no nearest-segment noise.
- **Sanity floor gates on CORE** (reviewer decision 2026-06-01): the floor means "broken encode/decode," and the bref vertex carries no encoded position, so gating on it would fire on every real region forever and block the PRD §11 gate.

### Reviewer guard — exclusion is by CONSTRUCTION IDENTITY, not magnitude
`_has_outbound_bref(block)` = "the token body ends in a bref token" (Case B/D outbound; encoder.py:438-456 + decoder.py:104-134). Inbound brefs (Case C/D) carry position via the anchor and are **not** excluded. This is a token-structure fact, never an error-size cut. A lock-and-guards test (`test_feature_accuracy_core_FIRES_on_displaced_non_bref_vertex`) displaces an **encoded** interior vertex on a crossing road and asserts the **core** metric fires — proving the exclusion did not blind the floor. See [[feedback_structural_exclusion_not_magnitude]].

## Corrected metric — measured on the 30-tile representative sample

| class | CORE p95 | CORE p99.9 | CORE max | FULL p99.9 | angle CORE p95 |
|---|---|---|---|---|---|
| road | 1.84m | 3.64m | 5.47m | 253m | 0.99° |
| building | 0.99m | 2.66m | 4.95m | 2.66m | 1.00° |
| poi | 0.30m | 0.35m | 0.35m | 0.35m | 0.00° |
| base | 2.12m | 4.22m | 4.74m | 4.22m | 1.00° |
| **ALL** | 1.37m | **3.42m** | 5.47m | **217m** | **0.99°** |

Core position p99.9 **3.42m ≪ 50m**, angle p95 **0.99° ≪ 20°** — inside the spec's own "end-to-end expected 5–15m / 3–8°" estimate (Decision 3c). FULL p99.9 217m = the reported crossing-road bref residual.

### Confirmed on the full 494 tiles (`_PHASE1_ACCURACY_BASELINE.yaml`, validator 1.1.0)
`sub-G validate: passed=False groups=4 sanity_floor_violated=False`, n_features **862,436** (non-vacuous), n_angle_features 527,589:
- **position_core_p99_9 = 3.61m** (≪ 50m), position_core_p95 = 1.35m
- **angle_core_p95 = 0.99°** (≪ 20°), angle_core_p99_9 = 1.00°
- position_full_p99_9 = **229.1m**, position_full_p95 = 18.6m (monotonic; the reported bref residual)
- structural_bound_breaches = 0

The accuracy `accuracy_sanity_floor` group is gone (5→4 quarantine groups). A NaN-poisoned non-monotonic full percentile (123.6m p95 > 8.2m p99.9, from 6/126k degenerate zero-length bref-placeholder geometries) was fixed by making `_percentile` non-finite-safe; **nan_core was 0** (gated distribution clean), and the degenerate geometries are the expected v1 2-vertex-crossing-road placeholder (encoded anchor faithful: core 0.12m), not a new defect.

## Tests / verification
`tests/data/sub_g/` 62 → 70 (new: pairing, subdivision-faithfulness, core/full split, the floor-still-fires guard, angle-vs-canonicalization, `_has_outbound_bref` identity). Full E/F/G suite 433 passed. `ruff check` clean.

## Still open (separate drills)
The accuracy floor was the loud finding masking others. Remaining sub-G quarantine groups to drill next: **H2** bref-bijection (1,649; `seam_contract_tokens` 0.5m vs encoder 1e-6m tolerance — verify against encoder), **H3** OGC-validity (27,958 — decode emits LineString for all, so these are LineStrings; check for degenerate/zero-length), and the **POI alpha-drop** 10.7% open item.
