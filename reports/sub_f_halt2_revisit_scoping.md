# Halt 2 round-trip-threshold revisit — scoping report (NO lock changes)

**Date:** 2026-05-29 · **Branch:** `phase-1-sub-F-micro-tokenizer` · **Status:** scoping only; proposal pending reviewer decision. No locks changed.

## What triggered this

T8.7 (decoder round-trip gate) surfaced that the BP2 Halt 2 lock — round-trip
`L_∞ ≤ 4.8m` — does not hold on real Singapore data. Root cause is a
`feedback_sample_regime_blind_locks` instance, now fully characterised:

- The encoder describes each segment as *"head in compass-direction D for distance M."* With 48 directions the bins are 7.5° wide, so any heading is rounded to within ±3.75°. The decoder walks these from a single anchor, so position error **accumulates**: per segment ≈ `L·sin(3.75°)`, and over a multi-vertex feature the drift compounds.
- **4.8m was the p95 of a 1,000-polyline sample, not a max.** The Halt 2 sweep at N=48 recorded p95 = 4.790393, p99 = 8.34, **sample max = 12.88m**. The T8.7 gate asserts `max ≤ 4.8m`, which was never true even on the Halt 2 sample.
- The Halt 2 measurement method (`analyze_geometry_primitives.py:_encode_decode`) already chunked and used the exact cumulative per-vertex metric used here — so the methods are comparable. The gap is **sample representativeness**: the 1,000-polyline sample materially under-represented the real distribution.

## Deliverable 5 (done first): honest re-measure on real data

The plan's T8.7 test metric was buggy (1:1 vertex zip; crashes once chunking
inserts collinear vertices). Fixed metric = each **source** vertex mapped to its
decoded counterpart via cumulative `chunked_segment_pairs` (collinear chunk
vertices admitted per spec §3.8). Honest round-trip `L_∞` at the **current 48
directions** over all 862,436 real features
(`reports/sub_f_halt2_roundtrip_scoping.yaml`):

| type | p50 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| roads | 2.05 | 13.87 | 21.72 | 29.90 | 33.00 |
| buildings | 0.89 | 4.15 | 8.20 | 15.47 | 32.18 |
| base/landuse | 2.65 | 18.01 | 26.89 | 32.71 | 33.46 |
| pois | 0 | 0 | 0 | 0 | 0 |
| **ALL** | 0.87 | **8.85** | **17.41** | 27.87 | 33.46 |

Synthetic aggregate (plan's seed 20260529) under the honest metric = **25.81m**
(the subagent's nearest-vertex metric under-reported it at 21.57m). Real P99.9
**segment** length = 205.8m; max segment = 344.1m.

## Deliverable 1: direction-count curve (single-segment anchor)

`L·sin(180°/N)` at the real P99.9 segment length (205.8m),
`reports/sub_f_halt2_roundtrip_scoping.yaml::direction_count_curve`:

| N | bin° | max seg within 4.8m | err @ P99.9 seg | err @ max seg |
|---|---|---|---|---|
| 48 | 7.5 | 73m | 13.46m | 22.50m |
| 96 | 3.75 | 147m | 6.73m | 11.26m |
| 128 | 2.81 | 196m | 5.05m | 8.45m |
| **144** | 2.50 | 220m | **4.49m** | 7.51m |
| 192 | 1.88 | 294m | 3.37m | 5.63m |
| 256 | 1.41 | 392m | 2.53m | 4.22m |

Min N holding 4.8m **at the P99.9 single segment** = **144**. BUT this is the
single-segment bound; **end-to-end accumulation makes open-loop 144 insufficient**
(measured below: openloop_144 ALL p99.9 = 8.74m). So "more directions" alone needs
N ≫ 144 to hold end-to-end.

## Deliverables 2 + 3: densification is a dead end; the working mechanisms

**Naive densification does not reduce drift.** Inserting collinear vertices on a
straight segment leaves every sub-segment at the same true heading → same
quantized bin → the decoded path is the same drifting ray. Proven analytically
and consistent with the mechanism. Densification only helps if it carries **fresh
absolute anchors** (re-anchoring) — which is what actually bounds the error.

Two mechanisms bound accumulation:
- **Re-anchor periodically** (emit a fresh absolute anchor every ≤73m of path). Bounds error to `73m·sin(3.75°) ≤ 4.8m`. Cost: +4 tokens per re-anchor → **BP3 budget impact** (a third budget pass). Grammar change (multi-anchor feature).
- **Error-feedback direction selection** (re-aim each chunk from the running *decoded* position toward the true target — Bresenham/error-diffusion). Bounds drift to ~one chunk regardless of length **at no token cost**.

## Deliverable 4 + the decisive experiment: fix comparison on real data

`reports/sub_f_halt2_fix_comparison.yaml`, all 862,436 features:

| variant | ALL p95 | p99 | p99.9 | max | roads p95 | roads max | lock cost |
|---|---|---|---|---|---|---|---|
| baseline_48 (current) | 8.85 | 17.41 | 27.87 | 33.46 | 13.87 | 33.00 | — |
| **feedback_48** | **2.86** | **3.57** | **4.13** | 9.66 | 2.96 | 9.66 | **NONE** |
| openloop_144 | 3.04 | 5.79 | 8.74 | 14.01 | 4.68 | 14.01 | vocab/sentinel/BP2 re-lock |

**Error-feedback (feedback_48) dominates** — better than tripling the direction
count, at zero lock cost. Token-count parity vs baseline is **IDENTICAL on all
862k features** (verified), so the BP3 budget is untouched.

### Cascade-scope inventory

| Option | encoder.py | decoder.py | token_cost / BP3 budget | encoding_primitives (BP2 lock) | sentinel_inventory IDs | grammar |
|---|---|---|---|---|---|---|
| **Error-feedback** | chunk-direction selection (tracks running decoded pos) | **unchanged** | **unchanged** | direction/mag/anchor/chunk **unchanged**; only `round_trip_l_inf_threshold` re-affirmed | **unchanged** | unchanged |
| More directions (N) | direction binning + sub-block base | direction reconstruction | unchanged (tokens/pair same) | `direction_count` re-lock | direction block 396..443 expands → cascade | unchanged |
| Re-anchor | emit periodic anchors | multi-anchor parse | **budget re-lock (3rd pass)** | possibly anchor semantics | possibly | multi-anchor feature |
| Accept as known-loss | unchanged | unchanged | unchanged | `round_trip_l_inf_threshold` re-defined | unchanged | unchanged |

## Recommendation

**Adopt error-feedback encoding (feedback_48).** It is the cheapest *and* most
accurate option: holds the 4.8m bar through p99.9 (4.13m) on real data, leaves
the BP3 budget, the 48-direction vocab, the sentinel inventory, the grammar, and
the decoder all untouched. Only `encoder.py`'s chunk-direction logic changes — the
same surface as the chunking fix (86f0c99). It dominates the more-directions and
re-anchor options on every axis.

Two reviewer sub-decisions remain even with feedback adopted:

1. **Gate posture.** Feedback holds p99.9 = 4.13m but max = 9.66m (≈1 feature).
   Options: (a) gate at **p99.9 ≤ 4.8m** (matches how 4.8m was originally derived —
   a percentile, not a max) and document the extreme-tail max as v1 known-loss;
   (b) feedback **+** a modest direction bump to push max < 4.8m (likely overkill);
   (c) re-derive the threshold from the feedback distribution.
2. **The 4.8m number itself.** It was the p95 of an unrepresentative sample. Under
   feedback it becomes a comfortable p99.9 bound, so keeping it is defensible — but
   the reviewer should consciously re-affirm it rather than inherit it silently.

## Residual checks before locking the fix (not yet done)

- Confirm error-feedback preserves the **angle / right-angle round-trip** (the other Halt 2 threshold, 7.5°) — feedback keeps vertices near true, expected same-or-better, but measure.
- Confirm existing T8.4 §3.5 chunking tests still pass (the axis-aligned 50m example is unchanged under feedback; off-axis ones change direction values — update expectations).
- Characterise the feedback max=9.66m tail (degenerate geometry? near-180° turns?).
- Re-affirm determinism (feedback is a deterministic function of canonical coords — BP5 preserved).

## ADDENDUM (2026-05-29, post-reviewer pre-lock checks) — error-feedback DISQUALIFIED; recommendation revised

The reviewer approved error-feedback conditionally and required three pre-lock
checks. They reopened the comparison:

**Check 1 — feedback tail (`reports/sub_f_halt2_residual_checks.yaml`).** The 9.66m
feedback max is a *wiggly road* (MultiLineString, 80 vertices, max segment 1.2m,
total 66m, `is_simple=True`) — accumulated drift over many tiny segments, not
degenerate geometry. Benign in shape.

**Check 2 — angle round-trip (the OTHER Halt 2 threshold). FAILS for feedback.**
Right-angle building-corner post-deviation, 2.0M corners:

| variant | non-catastrophic p95 | p99 | catastrophic (>45°) |
|---|---|---|---|
| baseline 48 | 7.5° (= locked) | 11.0° | 3,495 |
| **feedback 48** | **22.5°** | 37.5° | **33,735 (10×)** |

Error-feedback's per-chunk re-aiming dithers edge directions; on short building
edges (drift comparable to edge length) this swings reconstructed corners wildly.
Feedback fixes position but **regresses the angle threshold 3×** — a real lock
cost the position-only comparison hid. **Feedback is disqualified.**

**Check 3 — determinism (`...residual_checks.yaml`).** Within-process: DETERMINISTIC
(0 token-stream mismatches). Cross-runtime tie exposure (chunk directions within
1e-7° of a bin boundary, where a cross-libm atan2 1-ULP diff could flip the bin):
feedback 7,672 vs baseline 26,507 of 847k chunks — feedback is *not worse*. Not a
blocker (and moot now that feedback is out). Cross-env remains the existing
end-of-Phase-1 deferral (§1.4 #4).

**Reopened comparison — open-loop more-directions (`...direction_sweep.yaml`).** No
dithering, so angle is preserved/improved at every N:

| N | bin | pos p95 | pos p99.9 | pos max | angle p95 | angle cat |
|---|---|---|---|---|---|---|
| 48 | 7.5° | 8.85 | 28.3 | 33.5 | 7.5° | 3,495 |
| 144 | 2.5° | 3.04 | 8.84 | 14.0 | 2.5° | 1,771 |
| 256 | 1.41° | — | 5.00 | 14.2 | 2.8° | 1,553 |
| 360 | 1.0° | — | 3.68 | 14.1 | 3.0° | 1,558 |

Two facts: (a) **angle holds p95 ≤ 7.5° at every N** (open-loop never regresses it;
more N halves catastrophic). (b) **position max plateaus at ~14m regardless of N** —
an accumulation floor (wiggly features) that directions cannot remove; only
re-anchoring or feedback bounds it.

**The metric-mismatch fix changes the answer.** 4.8m was *derived* as the sample
p95 (§ root cause above). Enforcing it at the same statistic (p95 ≤ 4.8m), the
position holds at **N ≈ 96–144** (measured: 144 → p95 = 3.04m). Only a stricter
p99.9 gate needs N = 360.

### Revised recommendation

**Open-loop more directions**, gate at **p95 ≤ 4.8m** (honouring the lock's
derivation statistic — the metric-mismatch lesson), at the minimal N that holds
it with margin (measured safe at 144; ~96–128 likely sufficient — pin precisely
during implementation). This:
- fixes the real gap (position p95 8.85 → ~3m) **without** the angle regression
  feedback caused;
- leaves the BP3 budget untouched (tokens-per-pair unchanged);
- preserves the angle threshold;
- documents the irreducible position max (~14m, wiggly features) and the residual
  catastrophic corners (~1,771 at N=144) as v1 known-loss — consistent with the
  already-accepted right-angle catastrophic known-loss.

Cost: a direction-sub-block vocab expansion (48 → N) → BP2 / sentinel-inventory
ID-layout re-lock (mechanical but non-trivial; the dir block 396..443 grows and
cascades the magnitude / structural / BP7 / reserved bases).

**Open question for the reviewer (gate statistic):** p95 (per derivation; N≈96–144;
~14m max tail documented as known-loss) **vs** a stricter p99.9 gate (N=360) **vs**
also bounding the max (requires re-anchor — +tokens, a BP3 budget pass). The
position-max tail cannot be removed by directions alone, so a max-based gate is
not achievable on the directions axis.

## Artifacts

- `scripts/sub_f/scope_halt2_roundtrip.py` → `reports/sub_f_halt2_roundtrip_scoping.yaml`
- `scripts/sub_f/scope_halt2_fixes.py` → `reports/sub_f_halt2_fix_comparison.yaml`
- Uncommitted (subagent T8.7, failing as designed): `src/cfm/data/sub_f/decoder.py`, `tests/data/sub_f/test_decoder.py`. The decoder is reusable as-is for feedback (decoder is unchanged); the test metric needs the vertex-count-aware fix + the chosen gate posture.
