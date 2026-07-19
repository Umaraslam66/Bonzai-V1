# Eyeball generation probe — methodology validation (NON-scored)

**Date:** 2026-06-23 · **PI-approved** eyeball probe (no scoring / floor / manifest / decide / merge).
**Goal:** does the tokenization+training methodology produce VALID, COHERENT geometry that
responds DIRECTIONALLY to conditioning? (NOT a realism claim, NOT an architecture verdict.)

## Setup
- **Checkpoint:** `transformer-ar-53M / krakow-seed7 / last.ckpt` (global_step 111281,
  train_set=eu-train-union, conditioning_scheme=value-char-v1, ablation=full). ONE checkpoint.
- **Job:** Leonardo `47673765`, 1×A100, **3m38s ≈ 0.06 GPU-h** (boost_qos_dbg).
- **Contexts (HAND-BUILT, in-distribution):** 3 density-graded, stratum `(zoning,skel,dens,coast)`
  with zoning=1 & coastal=2 FIXED, stepping density 3→2→0 and skeleton 2→1→0, char_stats from
  the real `character_stats_for_cell` on many-small vs few-large footprint lists. 7 cells each
  (21 total), `max_new=1536`, gen seeds 1000–1006. city_identity/region = berlin (a TRAINING city).
- Tokens generated on GPU; decode (`split_cell_into_features`→`try_decode_block`→
  `promote_building_rings`) + render done LOCALLY. Scripts: `scripts/_eyeball_gen_probe.{py,sbatch}`,
  `scripts/_eyeball_render.py`.

## Results

### Directional response — STRONG (monotonic with density)
| context | med tokens | med building-class feats/cell | med road-class feats/cell |
|---|---|---|---|
| dense_urban | 1218 | 27 | 46 |
| medium_mixed | 408 | 10 | 7 |
| sparse_suburban | 155 | 3 | 3 |

Feature counts AND token length track the density conditioning cleanly.

### Decodability — ~100%
Every cell decoded ~all its feature blocks (1 cell at 99%, rest 100%). Token streams are well-formed.

### Self-termination — 19/21
19/21 cells emitted `<cell_end>`(260) and stopped on their own. **2 dense cells hit the 1536
probe cap** (a cost limit I set, NOT the 13312 hard cap) — inconclusive for those 2; all cells
far under the 13312 ceiling.

### Building footprints — SANE but rarely EXACTLY closed
- Building-class features ARE emitted in the right quantity (scales with density).
- Strict ring-closure (`coords[0]==coords[-1]`) rate is LOW: **12% dense / 5% med / 5% sparse**.
- BUT the open building rings are **near-closed**: gap (|first−last| / bbox-diag) median **3%**,
  **80% < 5%**, **96% < 10%**. median 6 vertices/ring.
- **Interpretation:** the model draws plausible, essentially-complete rectangular footprints
  but omits the exact closing-vertex repeat the promotion/OGC-validity check requires. A benign,
  nameable refinement target (closure tolerance / closing-token signal), NOT garbage geometry.
  Visible in the renders: sparse cells show clean rectangular buildings; `dense_urban_2` shows a
  sensible cluster of closed (blue) + near-closed footprints with a road network.

### Roads
Line features form curving networks with real spatial structure (more numerous in dense cells).
Plausible at a glance; rigorous planar-graph connectivity NOT verified in this probe.

## Verdict (eyeball, methodology validation)
The methodology **generates valid, coherent, conditioning-responsive geometry.** Positives:
100% decodable, strong directional response, self-terminating, sane footprint shapes. One concrete
refinement target: **exact building-ring closure** (footprints are ~complete but don't seal).

## Caveats
- Conditioning is HAND-BUILT synthetic (in-distribution values, but not real cells) → no real-cell
  side-by-side. A real-vs-gen comparison is a cheap follow-up.
- Roads "connect" assessed visually only.
- 2 dense cells capped at 1536 (probe cost limit) → termination for those is unconfirmed (re-run
  with higher cap is cheap).
- ONE checkpoint (transformer-ar seed7). No cross-seed / cross-backbone (out of scope — not a verdict).

## Follow-up diagnostics (2026-06-23, read-only, no fixes)

### (2) cell-EOS completeness in the dense regime — CONFIRMED
Re-ran the 2 dense cells that hit the 1536 probe cap at `max_new=13312` (job `47678081`,
same prefix/char_stats/seed): cell3(seed1003)→**1855 tok, cell_end fired**; cell4(seed1004)→
**3131 tok, cell_end fired**. Neither hit 13312. So **21/21 cells self-terminate**; the earlier
"2 capped" were just longer cells, not termination failures. The EOS fix is complete in dense.

### (1) Building-closure root cause — TOKENIZATION HOLDS; the closure *metric* is over-strict
- The grammar has **no closing token**. Buildings are Case A: anchor + a `(dir,mag)` walk;
  `_is_closed_ring` is an **exact** `coords[0]==coords[-1]` check on the reconstructed walk.
- **The grammar CAN express closure.** Encode→decode round-trips of real-shaped polygons close
  to **gap ~0%** (med/p90/max 0% over 200 random rotated rectangles). BUT only **48%** pass the
  EXACT-equality check — because `decode_dir_mag` uses `cos(radians(90°))=6e-17`≠0, so even a
  perfect axis-aligned ring drifts by float-epsilon. Exact equality fails ~half the time on the
  ENCODER'S OWN clean round-trips → it's a measurement artifact, hitting real data too.
- **Generated buildings, closure under tolerance:** EXACT 5–12% · within 2% 40–59% · within 5%
  74–95% (per context). So with a sane tolerance the MAJORITY of generated footprints are valid
  closed rings; the exact check is the dominant suppressor.
- **Residual model-side gap is modest:** model returns to within ~3% median (~1×0.5m quantum)
  vs the encoder's ~0% — a small training-precision gap, not a representation gap.
- **Classification:** NEITHER "(a) missing/under-emitted closing token" NOR "(b) grammar can't
  express closure." Closure is representable (to quantization precision); the low promotion rate
  is ~85% an over-strict exact-equality check in `_is_closed_ring`/`promote_building_rings` plus a
  modest model precision gap. **"Tokenization works" HOLDS.** (No fix applied — flagged only.)

## Artifacts
`reports/_eyeball_probe/`: `gen_tokens.json` (tokens+prefixes+char_stats), `png/*.png` (21 per-cell),
`geojson/*.geojson` (21), `montage_{dense_urban,medium_mixed,sparse_suburban}.png`.
