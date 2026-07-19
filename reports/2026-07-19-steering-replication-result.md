# Steering-probe REPLICATION result (2026-07-19) — product-time steering CONFIRMED

Pre-registration: addendum in `reports/2026-07-19-steering-probe-result.md` (decided before
any new data; disjoint seeds 3000–3159, n=160 pairs, contrasts C1/C4/C5, judged on this
sample ALONE, FIXED-closure classifier b30d604). Job: Leonardo `49835918` (COMPLETED 0:0,
57m46s ≈ 3.9 GPU-h). 2,880 cells, 0 dropped pairs. Artifacts: `reports/_steering_repl/`.

## Verdict (rule: sign-test p<0.01 + registered direction on ≥2/3 checkpoints)

| contrast | result | per-checkpoint p (T7 / T13 / M7) | median Δ |
|---|---|---|---|
| C4 control (char) | **3/3 VALID** | 9e-40 / 2e-37 / 4e-39 (r_rb 0.99–1.00) | +6 to +11.5 buildings |
| C1 road_skeleton, char fixed | **3/3 STEERS** | 2e-10 / 8e-08 / 1e-06 (r_rb 0.51–0.65) | +310 to +504 m road |
| C5 road_skeleton, char ABLATED | **3/3 STEERS** | 7e-06 / 1e-05 / 3e-11 (r_rb 0.45–0.61) | +212 to +410 m road |

`probe_valid=True`, `macro_steers_replicated=True` (C1 3/3),
**`product_steers_replicated=True` (C5 3/3)** — the n=40 main run was simply underpowered,
exactly as the addendum hypothesized.

## What this establishes

1. **The macro conditioning channel steers generation, including in the product regime**
   (no target-derived char_stats): stepping road_skeleton 0→2 adds ~200–400 m median road
   length per cell with char ablated to the dataset mean, on every checkpoint, both
   backbones. The v1 model supports a controllable-generation claim for structural
   conditioning.
2. Reconciles the NLL diagnostic: macro is worth ~0.018 nats/tok for token prediction
   (tiny) yet visibly shapes what is drawn — per-token likelihood sensitivity is not a
   proxy for generation control. This is itself a publishable methodological point.
3. Standing v2 design findings (unchanged): the cell_density knob is generation-dead in
   the presence of char_stats (main run C2, direction ~negative); road topology remains
   geometry-not-graph (defect (c), v2 grammar); char_stats dominance (C4) is the
   quantified leak.

## Consequences

- The realism eval (~262 GPU-h est., prerequisite defect-(a) fix landed in `b30d604`) can
  proceed on v1 with the controllability story intact — PI budget decision.
- The v2 char-dropout retrain is now OPTIONAL strengthening (amplify a confirmed-present
  channel), not a rescue — defer until after the realism eval unless it fails.
