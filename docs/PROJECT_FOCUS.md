# PROJECT FOCUS — read this before touching the eval / bake-off

Last set: **2026-06-23** by Umar (PI ruling). Canon for state = `docs/GROUND_TRUTH.md`.

## Current focus: VALIDATE the methodology, not crown a winner

The active goal is to confirm the **tokenization + training pipeline works end to end**:

- **Does it train?** — yes; the 6-run scored matrix trained clean and self-terminating.
- **Does it generate VALID, COHERENT geometry?** — the open question (eyeball-probe stage).

The bar right now is **plausibility + geometric validity** (sane building footprints, roads
that connect, output that responds to conditioning). It is **NOT** distributional realism,
**NOT** a sales/marketing claim, and **NOT** an architecture verdict.

## Standing-eval RESULT (2026-06-25) — a METHODOLOGY result, not a crown

Standing-eval matrix COMPLETE (6/6, job 47824952; code commit `0d677cc`). **Architecture is
`NO_DECISIVE`** — macro-only perplexity-gap **T +0.0066 vs M +0.0063, Δ0.0003 inside seed-noise
0.0012**. Conditioning **is** read (all 6 sign-tests significant at 100% effective-shuffle), but
**macro-only ~0.006 vs full ~0.45 → ~99% of the conditioning signal is char_stats; the macro
buckets contribute ~1%**. Saturation is **seed-dependent** (seed7 DESCENDING, seed13 PLATEAUED on
**both** backbones; mamba-seed23 saturation UNAVAILABLE — no training log on disk, D4). Geometry
**valid on both** (100% decodable, self-terminating, closure tight, roads fragmented = defect (c)).
This is descriptive methodology evidence, **not** a `decide()` verdict — the crown stays Umar-gated.

## Why macro conditioning is only ~1% (2026-06-26 diagnostic — read-only)

Direct factorial on the checkpoints (job 47910130 + local; `scripts/_diag_*.py`,
`reports/_diag/*.json`): the macro channel is **intrinsically weak for per-token NLL**, NOT
ignored. Macro's **standalone** value (char_stats also ablated) is only **~0.012–0.021 nats/tok**
vs char_stats's **+0.63–0.75** — and ablating char_stats does **not** resurrect macro, so char
isn't *masking* it. **Injection** and **capacity/training** are ruled out (macro responds
monotonically to id changes; flat across seeds/saturation/backbone). The random within-city donor
**understated** macro ~2.5× → true within-city effect ≈ **+0.018**. char_stats dominates because it
is a **leak-grade near-sufficient statistic computed from the target cell** (this quantifies the
parked char_stats↔KS echo). **Caveat: NLL ≠ generation plausibility** — macro still shapes coarse
structure (denser context → more buildings) in ways per-token NLL barely rewards; do NOT read this
as "drop macro." A v2 question (which macro fields to keep/strengthen), not a current-focus fix.

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

All three surfaced by the 2026-06-23 eyeball probe; all DEFERRED (no fix taken). Evidence:
`reports/_eyeball_probe/SUMMARY.md` (+ this session's encode→decode round-trip experiment and
`scripts/_road_connectivity_diag.py`). (a) and (b) matter **only when the realism eval revives**;
(c) is a **v2 grammar** question (relevant to v2 scoping, not the current coherence focus). None
changes the validated result: the methodology produces coherent geometry.

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

### (c) v2 REPRESENTATION GAP — road topology is not expressible in sub-F-v1 (NOT a model/training bug)
- **What:** on the 21 probe cells, after reclassifying by construction identity (so the **234 / 673
  (35%)** raw "road" LineStrings that were actually unsealed buildings are NOT counted as roads), the
  **401 true road segments are genuinely fragmented as drawn** — components÷segments ≈ **0.87 dense /
  1.00 medium / 1.00 sparse**, largest connected component **7–33%** of segments, **85–100% of road
  endpoints dangling**. Robust to an **8× endpoint tolerance** sweep (0.5→4 m), so it is NOT a
  float-drift / near-miss artifact. So the model emits road **geometry, not road topology.**
- **Root cause (why it's representation, not training):** `src/cfm/data/sub_f/decoder.py` decodes
  **every feature independently** (one `decode_feature` per feature); there is **no junction /
  shared-vertex primitive**. The only connectivity mechanism is `<bref>`, which references the **cell
  boundary only** (not other features) and **drops the crossing position** (decoder error
  UNBOUNDED-BY-TEST, bounded above by `cell_extent/2 = 125 m`). Connected topology **cannot be
  expressed in v1, so it cannot be trained in** — fragmentation is a property of the grammar, not a
  learning failure. (Same "representation vs model" distinction as defect (a)/(b).)
- **Classification:** DEFERRED to a **v2 grammar** (a junction-node / shared-vertex primitive). Out of
  scope for the current methodology-validation focus and for the realism eval.
- **Open question for v2 scoping (UNKNOWN locally):** do the **real training roads carry noded
  junctions** (coincident junction vertices that survive encoding)? If yes, v2 noding is well-motivated;
  if not, fragmented output is faithful to source. Not verifiable on the Mac — needs a Leonardo check;
  **singapore is a banned proxy** (do NOT estimate it with singapore). Measured on
  `transformer-ar-53M/krakow-seed7`; one checkpoint, synthetic hand-built conditioning.

## What IS in scope

- Generation probes / eyeball renders of generated geometry (GeoJSON / PNG).
- Confirming self-termination (cell-EOS), token-length sanity, decode round-trips.
- Anything that validates the **pipeline**, not the **realism verdict**.

All verdict / merge / scoring gates remain **Umar-gated**.
