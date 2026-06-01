# sub-G T11 H3 — the 27,958 OGC-invalid geometries are the v1 outbound-bref placeholder collapse (report-not-gate)

**Date:** 2026-06-01 · **Branch:** `phase-1-sub-G-cross-artifact-validator` · **Region:** singapore (494 tiles)

## Finding

The seam-3 OGC-validity group (`decodable_to_valid_geojson` / `decoded geometry not
OGC-valid`, **27,958** instances = 3.2% of 862,436) is **100% the v1-by-design
outbound-bref placeholder collapse** — the most degenerate form of the SAME
v1-unencoded outbound bref vertex H1 reports-not-gates. It is **not** a decode bug,
and there is **no genuine-degeneracy remainder** hiding underneath.

Unlike H1/H2 (the validator firing on its own *wrong premise* — a measurement bug, a
tolerance mismatch), here the decoded geometry **really is** OGC-invalid. The call is
gate-vs-known-limitation, and it is resolved **consistently with H1**: exclude the
designed bref info-loss from the blocking gate by construction identity, keep the
count reported (reviewer-approved 2026-06-01).

## Characterization (read-only drill, reproduced bit-identical to the gate)

A drill decoded every feature block on all 494 tiles, reproduced the gate's
`shape(geom).is_valid` check, and bucketed every invalid instance. The reproduced
count was **27,958 — exactly the gate's count** (so the characterization is the gate,
not a proxy):

| Dimension | Result over all 27,958 |
|---|---|
| `n_distinct_vertices` | **1** (100%) — every one is `[anchor, anchor]`, `n_coords==2` |
| `has_outbound_bref` | **True** (100%) |
| `feature_class` | **0 (road)** (100%) |
| Quiet subset: `n_distinct==1` **without** outbound bref | **0** |
| NaN / non-degenerate-invalid (`n_distinct≥2`) | **0** |
| Case split | 22,045 Case B (outbound only) · 5,913 Case D (inbound+outbound) |

The loud-masks-quiet check (twice-seen in H1/H2) came up **clean**: a single mode,
all of it the same v1 limitation. No real degeneracy under the headline symptom.

## Why these geometries are degenerate (construction identity)

A `<bref>` token carries direction + class but **not position** — v1 drops the
crossing vertex (decoder.py:13-22, §1.4 scope lock #1, v2-scoped). For a road with
interior bends this is a bounded position residual (the H1 `position_full` story).
But these 27,958 are roads that cross a 250m cell with **no interior vertex** (V=2,
entry + exit only):

- Encoder (`encode_feature`, Case B/D): emits (V−2)=0 inner pairs → tokens are
  `<feature> tag (inbound?) anchor×4 <outbound_bref> <feature_end>`. Only the anchor
  (entry vertex) carries a position.
- Decoder: reconstructs `[anchor]`, then appends the outbound-bref placeholder
  `coords.append(coords[-1])` → `[anchor, anchor]`, a **zero-length LineString** →
  OGC-invalid.

`decode_feature` never returns Polygon (closed rings come back as LineString,
decoder.py:144-155), and a LineString is OGC-invalid essentially *only* under
degeneracy (a self-intersecting LineString is valid). Interior pairs always move
(`magnitude_q ≥ 1`), so the **only** way to reach a single-distinct-vertex LineString
is this outbound-bref collapse. Hence the construction-identity predicate is
total-coverage of the invalid set, not a magnitude heuristic.

## Fix (sub-G seam-3 only; reviewer Option 1)

Exclude the bref-placeholder collapse from the blocking OGC-validity gate **by
construction identity, never by magnitude**, and report the excluded count in the
accuracy baseline next to the `position_full` residual (same roads, one limitation
in two seams). This mirrors H1's core/full mechanism (the excluded thing is counted +
reported, not gated), not a separate advisory section.

- `_is_bref_placeholder_collapse(block, geom)` = `_has_outbound_bref(block)` **AND**
  `<2 distinct decoded vertices`. The gate skips (and counts) only when both hold.
- The count threads `check_decodability → validate_tile → validate_region →
  finalize → render_accuracy_baseline` as `ogc_bref_collapse_excluded_from_gate`,
  with a note cross-referencing the `position_full` residual.

## Regime-distinguishing guard (reviewer-mandated) — two blocks, one geometry

Two token blocks decode to the **identical** zero-length `[(0,195.5),(0,195.5)]`
LineString; the gate must diverge **only** on construction identity:

- (a) `test_check_decodability_excludes_bref_placeholder_collapse_from_gate`: the real
  Singapore block `[509,41,300,323,363,369,1506,510]` (anchor + outbound bref) →
  **no** OGC diagnostic, count == 1.
- (b) `test_check_decodability_GATE_FIRES_on_degenerate_without_outbound_bref`: the
  same anchor + a synthetic magnitude-0 inner pair (`m=443=_MAGNITUDE_BASE-1`) and
  **no** outbound bref → **still quarantines** (1 OGC diagnostic), count == 0.

(b) is the proof the exclusion is by construction identity, not a bare zero-length /
magnitude test: a naive "skip if zero-length" would wrongly pass (b)
(`feedback_gate_must_distinguish_regimes`, `feedback_structural_exclusion_not_magnitude`).
A degenerate-without-bref geometry is unreachable from a real encoder (interior pairs
always move), so the guard is necessarily synthetic — and that unreachability is
exactly why the exclusion is safe.

## Verification

- Read-only drill: reproduced **27,958** invalid, 100% bref-placeholder, 0 remainder.
- sub-E/F/G suite: **441 passed** (437 + 4 new), ruff clean.
- 494-tile validator re-run (post-fix, temp output dir): **`groups: []`** (was 1),
  `sanity_floor_violated=False`; baseline reports
  `ogc_bref_collapse_excluded_from_gate: 27958` (== the prior OGC group count == the
  drill count, three-way agreement), `position_full_p99_9: 229.09` (same roads),
  `position_core_p99_9: 3.61`, `angle_core_p95: 0.99`. Marker written to the temp dir
  only — the canonical `_PHASE1_VALIDATED` awaits the sub-G close-out merge-gate review.
- Folds into the unblessed validator 1.1.0 (no `_PHASE1_VALIDATED` ever written;
  artifact-changing fix before any 1.1.0 artifact is blessed).

## Disposition

Report-not-gate, consistent with H1. The 27,958 are the same crossing roads as the
`position_full` p99.9 229m residual — one v1 limitation (the unencoded outbound bref
vertex, v2-scoped per §1.4) surfacing in two seams (accuracy + decodability). This is
a **known-limitation exclusion** (alongside H1's core/full bref-vertex exclusion), to
be listed in the sub-G close-out for merge-gate review.
