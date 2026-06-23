# PROJECT FOCUS — read this before touching the eval / bake-off

Last set: **2026-06-23** by Umar (PI ruling). Canon for state = `docs/GROUND_TRUTH.md`.

## Current focus: VALIDATE the methodology, not crown a winner

The active goal is to confirm the **tokenization + training pipeline works end to end**:

- **Does it train?** — yes; the 6-run scored matrix trained clean and self-terminating.
- **Does it generate VALID, COHERENT geometry?** — the open question (eyeball-probe stage).

The bar right now is **plausibility + geometric validity** (sane building footprints, roads
that connect, output that responds to conditioning). It is **NOT** distributional realism,
**NOT** a sales/marketing claim, and **NOT** an architecture verdict.

## Explicitly OUT of scope right now — do NOT do these unprompted

- **Do NOT crown transformer-vs-mamba.** No `decide()`, no Lane-S excess, no Lane-M verdict,
  no `floor_all` scoring, no NO_DECISIVE_WINNER analysis. The crown is Umar-gated.
- **Do NOT "fix" the char_stats↔KS echo.** It is a **KNOWN, DEFERRED** issue: the per-cell
  conditioning hands the model the median/IQR/p90-ratio of building area + median road length,
  which Lane-S then KS-scores → pass-by-echo (the same §3.4 circularity class density-coherence
  was dropped for). It is **parked for a future *sellable* realism version.** Do not re-litigate,
  re-derive, estimate, or patch it unless Umar explicitly asks.
- **Do NOT merge** (`phase-2-bakeoff-*`→main, `cell-eos`→main) — Umar-gated.
- **Do NOT re-pin the floor** or rebuild the sealed sampler manifest for the current focus.

## Why the echo is parked (so it isn't "rediscovered" as new each session)

The realism eval (Lane-S / Lane-M) is contaminated by the per-cell `char_stats` conditioning.
An honest de-echo is a real metric change that requires a floor re-pin AND a Leonardo
cell-grouped extraction *even to evaluate whether a crownable signal survives* (UNKNOWN locally;
do **not** estimate it with proxies — singapore is a banned proxy). None of that is needed to
answer "does the methodology produce coherent geometry," which is the current question. The full
leak analysis lives in this session's findings; the short version is the bullet above.

## Deferred defects — RECORDED so they aren't rediscovered (do NOT fix unprompted)

Both surfaced by the 2026-06-23 eyeball probe; both DEFERRED (no fix taken). Evidence:
`reports/_eyeball_probe/SUMMARY.md` (+ this session's encode→decode round-trip experiment).
They matter **only when the realism eval revives** — irrelevant to the current coherence focus.

### (a) EVAL-SIDE BUG — over-strict ring-closure check (suppresses ~half of building promotion)
- **Where:** `src/cfm/eval/geometry.py` — `_is_closed_ring` / `promote_building_rings` use an
  **exact** `coords[0] == coords[-1]` test.
- **Root cause:** the decoder's `decode_dir_mag` (`src/cfm/data/sub_f/decoder.py`) reconstructs
  vertices with `cos(radians(90°)) = 6.1e-17` (≠ 0) + sub-quantum accumulation, so even a
  perfect ring drifts by float-epsilon. **Exact equality fails on ~52% of the ENCODER'S OWN clean
  round-trips** (only 48% pass, measured over 200 random rotated rectangles; gap itself ~0%).
- **Impact:** same code path runs on real data → when the eval revives it will silently drop
  ~half of REAL building polygons (depressing `n_polygons`, building-area KS, the emergence floor).
  It is NOT a tokenization or model failure — the grammar expresses closure to quantization precision.
- **Fix direction (deferred):** replace exact equality with a closure TOLERANCE (e.g., gap < a few %
  of the ring bbox-diagonal) or snap the closing vertex to the anchor. Do NOT apply unprompted.

### (b) MODEL GAP — modest building closing-precision (training, not representation)
- **What:** generated building footprints return to within **~3% of the anchor (median; ≈ one 0.5 m
  magnitude quantum)**; **74–95% close within a 5% tolerance**, 40–59% within 2% — vs the encoder's
  ~0% (≈100% within a tiny tolerance). So the model's closing edge is ~1 quantum looser than the data.
- **Classification:** a small **training-precision** gap, NOT a representation gap (closure IS
  expressible; see defect (a)). Measured on `transformer-ar-53M/krakow-seed7`.
- **Fix direction (deferred):** out of scope now; revisit only if/when footprint exactness becomes a
  product bar. Do NOT chase unprompted.

## What IS in scope

- Generation probes / eyeball renders of generated geometry (GeoJSON / PNG).
- Confirming self-termination (cell-EOS), token-length sanity, decode round-trips.
- Anything that validates the **pipeline**, not the **realism verdict**.

All verdict / merge / scoring gates remain **Umar-gated**.
