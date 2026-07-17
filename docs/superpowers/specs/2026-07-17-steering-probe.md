# Generation-steering probe — does macro conditioning STEER generated geometry?

Status: SPEC (2026-07-17). NON-scored methodology experiment (eyeball-probe class, statistical).
Answers the open caveat of the 2026-06-26 macro diagnostic: NLL sensitivity ≠ generation
steering. Product relevance: at generation time no target-derived char_stats exists — the macro
buckets are the only knobs a user can turn. If they do not steer, v2 conditioning surgery
(char_stats dropout / neighbor-derived stats) is REQUIRED before any realism-eval spend.

## Hypothesis (written in advance)

The eyeball probe (2026-06-23) showed qualitative directional response when density+skeleton+char
were stepped TOGETHER (confounded). We expect the char control (C4) and joint-macro (C3) contrasts
to steer clearly; single-field road_skeleton (C1) and the char-ablated regime (C5) are genuinely
uncertain — C5 is the product-relevant unknown this probe exists to measure.

## Design

Checkpoints (same 3 as the diagnostics): transformer-ar-seed7, transformer-ar-seed13,
mamba-hybrid-seed7 (final `last.ckpt` of the 6-run matrix, on Leonardo).

Base context (from the eyeball probe's REAL in-distribution strata): city/region=berlin (a
TRAINING city; held-out cities deliberately avoided), zoning=1, coastal=2, pop_density=2,
cell_density=2, road_skeleton=1, seed-field=7. char_fixed = `character_stats_for_cell` of the
eyeball `medium_mixed` areas/lengths. char_mean = the factorial diagnostic's dataset-mean char
convention (`_diag_conditioning_factorial._mean_char` over the held-out cache).

Contrasts — each is TWO arms differing in EXACTLY the named field(s); everything else identical,
including paired generation seeds (gen_seed 2000..2039 shared across arms of a contrast):

| id | swapped field(s) | arm A → arm B | char regime |
|----|------------------|----------------|-------------|
| C1 | road_skeleton | 0 → 2 | char_fixed |
| C2 | cell_density | 0 → 3 | char_fixed |
| C3 | joint (pop,skel,dens) | (1,0,0) → (3,2,3) | char_fixed |
| C4 | char_stats ONLY | sparse_suburban chars → dense_urban chars | macro fixed at base |
| C5 | road_skeleton | 0 → 2 | char_mean (ablated) |

C4 is the POSITIVE CONTROL / sensitivity gate: char is known-strong (NLL +0.63–0.75). A probe in
which C4 shows no effect cannot detect steering → verdict UNRELIABLE, no macro conclusion drawn
(a gate must be able to fail: feedback_gate_must_distinguish_regimes).

N = 40 paired generations per arm; max_new = 1536 (eyeball-probe cap; cells self-terminate ~600).
Yield: 10 arms × 40 = 400 cells/ckpt, 1,200 total.

Off-manifold annotation: for each arm, record whether its (zoning, skeleton, density, coastal)
4-tuple exists among the locked floor's 78 global tuples. Annotate, do not block — counterfactual
off-manifold response is itself informative.

## Outcome metrics (decode locally off dumped tokens; GPU job emits tokens only)

Per cell, after decode + the ONE feature-classification rule
(`conditioning_discrimination._tile_features`, same as gen_realism): n_features,
n_road_segments, total_road_length, n_buildings, total_building_area, median_building_area,
n_tokens. Registered direction predictions:

- C1/C5: skeleton 2 > 0 ⇒ road segments and total road length HIGHER in arm B.
- C2: density 3 > 0 ⇒ building count HIGHER, median building area LOWER in arm B.
- C3: dense joint ⇒ counts/lengths HIGHER, median area LOWER in arm B.
- C4 (control): dense char ⇒ building count HIGHER, median area LOWER in arm B (expected LARGE).

## Verdict rule (pre-registered)

Per (contrast, primary metric): paired per-seed deltas (arm B − arm A); report median delta,
two-sided exact binomial sign-test p, rank-biserial effect size. A contrast STEERS iff sign-test
p < 0.01 AND direction matches the registered prediction, on ≥ 2 of 3 checkpoints. Primary
metrics: C1/C5 → total_road_length; C2/C4 → n_buildings; C3 → n_features. All other metrics
reported as secondary (no verdict weight).

Probe VALID iff C4 STEERS. Then:
- C5 STEERS (± C1) → macro usable at product time; v2 = amplify (train-time char dropout), realism
  eval worth running on v1.
- Only C4 steers (C1/C2/C3/C5 flat) → model is char-driven; v2 retrain (char_stats dropout +
  neighbor-derived stats) REQUIRED before the ~262 GPU-h realism eval.
- Mixed → PI decision with data + budget line.

## Budget

Rates: tf 0.0268 s/tok (job 47143523), mamba 0.0652 s/tok (job 47603817), ~600 tok/cell
self-terminated. 800 tf-cells ≈ 3.6 h + 400 mamba-cells ≈ 4.3 h single-GPU ≈ 7.9 GPU-h compute;
one full node, 4 concurrent single-GPU workers (full-node discipline) ≈ 2 h wall = 2 node-h =
**~8 GPU-h billed** (< 10 GPU-h line; PI-blessed small-spend class).

## Files

- `scripts/steering_probe_gen.py` — arm construction (pure, unit-tested: arms differ ONLY at the
  swapped position(s); paired seeds identical) + generation loop modeled on `_eyeball_gen_probe.py`.
- `scripts/steering_probe.sbatch` — 1 node / 4 workers / account AIFAC_P02_548.
- `src/cfm/eval/steering_stats.py` (+ tests) — SCORING CORE (paired deltas, exact sign test,
  rank-biserial, verdict rule). Orchestrator-authored, not subagent-authored.
- `scripts/steering_probe_analyze.py` — thin CLI: decode dumps → `_tile_features` → steering_stats.

## Assumptions (flagged per CLAUDE.md)

1. berlin as the fixed city generalizes (same choice as the eyeball probe); steering measured
   within ONE training-city context is decision-grade for the v2 go/no-go. Trigger to revisit:
   verdicts differ across checkpoints in a city-flavored way.
2. Decoded-feature statistics via `_tile_features` are the right steering observables (they are
   the floor/eval grammar — consistency with the scored lane).
3. The 6-run matrix `last.ckpt`s on Leonardo are the correct artifacts (verified COMPLETED
   2026-06-23, reports steps=112563).
