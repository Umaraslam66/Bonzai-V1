# sub-G T11 H2 — the 1,649 bref-bijection mismatch is a sub-G false positive (on-edge tolerance)

**Date:** 2026-06-01 · **Branch:** `phase-1-sub-G-cross-artifact-validator` · **Region:** singapore (494 tiles)

## Finding

The seam-2 bref-bijection group (1,325 missing + 282 mixed + 42 extra = 1,649) is a **sub-G false positive**, the same class as H1 (the validator firing on its own wrong premise) — **not** a sub-F drop and **not** the cycles-3/4 reimplementation-drift third instance.

**Root cause:** sub-G's `_endpoint_edge` used a **0.5m** tolerance; the encoder's `_direction_of_endpoint` uses **1e-6m** on the *same raw coords*. Both check edges in the same order (W, E, N, S). A **near-corner** endpoint is misattributed by the loose band. Traced case (`i10_j10` cell (5,0)): a road endpoint at `(0.071, 0.0)` is *exactly* on the **N** edge (y=0) but 0.071m off **W**:
- sub-G (0.5m): `|0.071| ≤ 0.5` → **W** → W contract MINOR → predicts `W_MINOR`.
- encoder (1e-6m): `|0.071| > 1e-6` (not W) → `|0.0| ≤ 1e-6` → **N** → N contract NONE → emits nothing.

→ sub-G predicts a bref the encoder never emits ("missing"; wrong-edge variants → "mixed"/"extra").

## Branches ruled out (verified against the encoder, not assumed)
- **Road-gate drift (B):** the encoder's `is_road = semantic_tag_to_l1_key(tag)=="highway"` reduces to `feature_class==0` — its `semantic_tag` key is *derived* from `feature_class` (`_semantic_tag_from_row`: 0→"highway"). Identical to sub-G's gate. Refuted analytically.
- **Contract drift (B′):** `build_cell_contracts` vs `load_boundary_contract` — **0 disagreements / 15,360 (cell,dir) checks**. Refuted.
- **Real sub-F drop:** the encoder's emission reproduces the tokens exactly. Refuted.
- (A first decomposition drill reported a misleading "exact-missing=493"; its discriminator was too loose — `feedback_inspection_script_premise_check`. Tracing the concrete case corrected it.)

## Fix (sub-G seam-2 + a behavior-identical sub-F constant extraction)
A cell edge is a **structural boundary** (sub-C's clip snaps crossing endpoints float-exact onto it), so it gets a structural epsilon, not a 0.5m metric band (`feedback_epsilon_structural_vs_user_threshold`). The 0.5m "absorb canonicalization" rationale was a **false premise**: canonicalize is a LineString no-op and sub-G reads raw coords.

- **Shared authority (one source, both import):** extracted the encoder's `1e-6` to a named constant `encoder.ON_EDGE_EPS_M` (used as the `_classify_feature_for_bref` default — behavior-identical); `seam_contract_tokens._EDGE_TOL_M = ON_EDGE_EPS_M` (imported, not re-hardcoded). A test asserts `_EDGE_TOL_M is ON_EDGE_EPS_M` so a future encoder-epsilon change can't silently re-open this (`feedback_independence_misses_shared_assumptions` applied to the epsilon itself). Importing a *constant* (not the classification logic) preserves seam-2 independence (Decision 3b).

### Two-directional guard (reviewer 2026-06-01) — test BOTH band edges
- (a) `test_bijection_FIRES_on_genuine_dropped_bref`: a road endpoint *exactly* on an active edge with its token **absent** → still fires "missing". Tightening did not blind the seam to a real drop.
- (b) `test_bijection_SILENT_on_near_corner_offedge`: the `(0.071, 0.0)` trace case, token correctly absent → must **not** fire. Regime-discrimination confirmed: `_endpoint_edge(0.071, 0.0, tol=0.5)` → "W" (the false positive), `tol=1e-6` → "N" (correct).

## Verification
494-tile bijection cross-check: missing/extra **0/0** region-wide with the shipped prediction. Validator re-run: bref-bijection group gone (quarantine groups 4 → 1; only the OGC-validity group remains, = H3). sub-G tests +3 (shared-authority lock + two-directional guard); E/F/G suite green; ruff clean. Folds into the unblessed validator 1.1.0 (no `_PHASE1_VALIDATED` ever written).

## Still open
**H3** OGC-validity 27,958 (the only remaining group — decode emits LineString for all, so degenerate/zero-length LineStrings; some overlap the H1 bref-placeholder zero-length geoms) + the **POI alpha-drop** 10.7% open item.
