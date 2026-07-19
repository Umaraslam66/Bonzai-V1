# Steering-probe result (2026-07-19)

Spec: `docs/superpowers/specs/2026-07-17-steering-probe.md` (pre-registered contrasts + verdict
rule). Code: commit `51aaea9`. GPU job: Leonardo `49831125` (COMPLETED 0:0, 24m37s wall on one
full node ≈ **1.6 GPU-h billed** — the ~8 GPU-h estimate used the 100M-mixer gen rate; the 53M
models generate ~3× faster). Inputs: 3 checkpoints (transformer-ar seed7/seed13, mamba-hybrid
seed7, the 6-run-matrix `last.ckpt`s), 5 contrasts × 2 arms × 40 paired seeds = 1,200 cells;
0 dropped pairs; 13/1,200 hit the 1536 cap (98.9% self-terminated). Artifacts:
`reports/_steering/steering-*.json` (raw tokens), `steering_per_cell_metrics.json`,
`steering_per_cell_metrics_verdicts.json` (verdict layer: `cfm.eval.steering_stats.judge`).

## Verdict (pre-registered rule: sign-test p<0.01 + registered direction on ≥2/3 checkpoints)

| contrast | what varies | primary metric | result |
|---|---|---|---|
| C4 control | char_stats only | n_buildings | **STEERS 3/3** (r_rb 0.98–1.00, p ≤ 1e-8) — probe VALID |
| C1 | road_skeleton 0→2, char fixed | total_road_length | **STEERS 2/3** (T7 p=.0022 Δmed +522 m; M7 p=.0064 Δmed +488 m; T13 p=.15 Δmed +235 m, right direction) |
| C2 | cell_density 0→3, char fixed | n_buildings | **NO STEER 0/3** — direction slightly NEGATIVE (r_rb −0.14..−0.30) |
| C3 | joint (pop,skel,dens), char fixed | n_features | 1/3 (all directions positive; T7 p=.024, T13 p=.099 — trend, under the bar) |
| C5 | road_skeleton 0→2, char ABLATED (mean-char) | total_road_length | **0/3 at p<.01, but all 3 in the registered direction** (p=.038/.053/.15; Δmed +331/+144/+186 m; r_rb 0.36–0.57) |

Headline: `probe_valid=True`, `macro_steers=True` (via C1), `product_steers=False` (C5 under
the strict bar).

## Interpretation

1. **The NLL "~1% macro" number understated generation-level control.** road_skeleton — the
   field the per-field NLL ablation ranked dominant — visibly steers generated road geometry
   (+~500 m median total road length for a 2-step knob change) when char_stats is present.
   NLL-weak is not steer-dead; the probe existed to test exactly this and it paid off.
2. **The product-time regime (no target-derived char_stats) shows a consistent, sub-threshold
   trend, not absence**: every checkpoint moves the registered direction with meaningful
   medians, at p≈0.04–0.15 with n=40. This is an UNDERPOWERED positive, not a null. It must
   not be reported as "steers" (rule is the rule) nor as "does not steer" (directionally wrong
   reading). Resolution: replication at 4× n (addendum below).
3. **The cell_density knob is generation-dead when char_stats is present** (C2, direction even
   mildly negative): char_stats directly encodes building count/size statistics, so the density
   bucket has nothing left to control. Direct v2 design input: density conditioning via a
   redundant coarse bucket is wasted capacity in the presence of a char channel.
4. **C4 quantifies char dominance at the generation level** (near-perfect separation) —
   consistent with the NLL factorial's +0.63–0.75 nats/tok and the leak diagnosis.

## Pre-registered replication addendum (decided before any new data)

To resolve C5 (the paper's "controllable without target stats" claim): independent replication,
NEW disjoint seeds 3000–3159 (n=160 pairs), contrasts C4 (validity re-check), C1 (confirmation),
C5 (the claim). Same rule, applied to the replication sample ALONE (no pooling with this run —
pooling after peeking is sequential-testing contamination). Power at the observed pooled C5
positive-rate (~0.65): ≈0.88 per checkpoint at p<0.01. Estimated cost ≈ 4 GPU-h (small-spend
class). Verdict names: `C5_replication` decides `product_steers_replicated`.

## Decision consequences (for the PI checkpoint)

- The **realism eval (~262 GPU-h)** conditions generation on matched REAL cells (char_stats
  included, per the locked Lane-S design), so its validity is independent of C5 — it measures
  held-out distributional realism under full conditioning. Prerequisite before any scored run:
  fix eval defect (a) (exact-equality ring closure, `_is_closed_ring`) — it silently drops ~52%
  of clean building polygons.
- The **v2 char-dropout retrain** decision is gated on the C5 replication: if replication passes,
  v1 already supports a (modest) controllable-generation claim and v2 is optional strengthening;
  if it fails, v2 (train-time char_stats dropout + neighbor-derived stats, capacity toward
  structural fields) is the fix that makes the product story true.
