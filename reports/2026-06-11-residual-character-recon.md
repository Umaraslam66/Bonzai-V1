# Residual city-character reconnaissance (PI-ordered, 2026-06-11)

**Status: investigation only.** No wiring, no Task 24, no recommendation-as-commitment.
**Run:** Leonardo login node CPU, 00:33–00:51 CEST, code @ `d3f27b9`, RC=0.
**Pipeline trust:** the script hard-aborts unless it reproduces verified run-3
(V0 321/141, V4 1056/392) through the same harness, AND unless its own parallel
building-area walk restacks V4 to the identical 1056/392. Both anchors held.
**Artifact:** `reports/2026-06-11-residual-character-recon.yaml`.

## 1. The residual is REAL — every artifact probe came back clean

- **Zero-length roads: 0.0% in all four cities** — Task-22's bref exclusion fully
  decontaminated the road metric; nothing degenerate is left in the pool.
- **No power artifact:** munich (the 171-tile floor) holds 49.1% of all pairs and
  49.5% of significant ones — no over/under-representation. Significant pairs are
  *smaller*-n than non-significant ones (median min-n 164 vs 388): this is not
  "huge n makes dust significant" — the δ floor already killed that regime.
- **Not the δ threshold:** the sweep decays smoothly (V4 rate 37%→22%→12%→6% at
  δ=0.15/0.20/0.25/0.30) and only dies near δ≈0.40–0.50 — far beyond the locked
  "differences a consumer would notice" framing. At a generous δ=0.25, 127 pairs
  still stand.
- **Not one city:** all six city pairs sit in a tight 0.27–0.43 significance-rate
  band (krakow–munich highest 0.43, eisenhüttenstadt–krakow lowest 0.27). The
  residual is symmetric, not glasgow-vs-everyone.
- **Broad, not concentrated:** among V4-significant pairs, median KS ≈ 0.21–0.23,
  ~120 pairs per metric at KS≥0.20, only 4–7 at KS≥0.40. This is two hundred
  moderate differences, not a handful of outliers to excuse.

## 2. WHAT survives: thread 1 — buildings are SHAPE, beyond any summary

Median-normalizing each significant pair's samples (divide by own median →
location removed; surviving KS = distribution shape):

- **building_area_m2: 70% of significant pairs survive normalization**
  (124/176 stay ≥0.15; normalized median KS 0.187). Within strata identical down
  to median building size, cities differ in the *spread and tails* of their
  building-size distributions.
- The **richer-dims upper bound** confirms no summary-bucket captures this:
  stacking IQR (rate 0.327), p90/p50 (0.349), count (0.324), or a
  median+IQR+ratio kitchen sink (**0.297, 781 significant pairs**) on top of V4
  barely dents the rate from 0.371. A three-summary description of the
  within-cell distribution still leaves ~30% of comparisons significant.

## 3. WHAT survives: thread 2 — roads are HALF fine-location, half shape

- Pooled road-length medians are nearly identical across cities (37–41 m) — no
  gross shift; the residual lives within strata.
- **54% of significant road pairs die under median normalization** (117/216):
  within identical strata, cities' roads are systematically slightly longer or
  shorter — a fine-grained location effect no tested stratum carries. The other
  46% is shape. Significant road pairs spread across density buckets 0–2 and
  concentrate in skeleton class 1 — broad again, not localized.

## 4. The decisive implication (quantified, not inferred)

The gate's PASS bar is **zero** BH+δ-significant pairs. The recon bounds every
bucketed-conditioning enrichment at a ~30% significance-rate floor with
*hundreds* of surviving pairs. **PASS is unreachable by conditioning-stratum
enrichment of any tested or extrapolated kind.** This is no longer a
feature-selection problem.

Honest classification against the PI's (a)/(b)/(c):

- **(a) richer single-dim conditioning carrier: ruled out as a gate-closer** —
  directly bounded (kitchen sink: 781 significant pairs).
- **(b) partially solvable for MODEL QUALITY:** the building residual is
  predominantly transmissible *shape* information — a continuous/distributional
  carrier (e.g. real per-cell stats via the empty `macro_tokens` channel) could
  give the model character signal a bucket can't. But it cannot flip the gate:
  the gate measures via strata, and the strata test is bounded above.
- **(c) genuinely city-idiosyncratic at this granularity — for the GATE'S
  question, yes.** Same-conditioning ⇒ same-distribution is empirically false
  for these cities at δ=0.15 under any practical conditioning vocabulary. The
  T5 worst-case cross-city bar rests on that premise.

## 5. Option space this evidence frames (decisions are Umar's, none taken)

1. **Re-scope the bar, not the conditioning** (spec/PRD-level): with a
   city-identity floor wired, a *per-seen-city* bar is valid by construction
   ("a miss can't mean 'wasn't told the city'"); cross-city generalization
   becomes a separate claim measured differently (e.g. held-out-city quality at
   matched conditioning, judged against this recon's measured KS floor rather
   than zero-difference).
2. **Identity floor + distributional carrier** (b-shaped): wire `city_identity`
   now; carry shape via `macro_tokens` for model quality; accept the
   recalibrated gate as a MEASUREMENT (report KS profile) rather than a
   PASS/FAIL halt at Task 25.
3. **Raise δ to where the residual dies** (~0.40): mechanically closes the gate
   but at a floor the spec's own consumer framing calls absurd — listed for
   completeness, the sweep argues against it.

## Verified-end-state

In-run hard anchors (harness + restacked walk) both reproduced; YAML re-read
locally; internal consistency checked (392 = 176+216; sweep@0.15 ≡ run-3 row;
six pair-cells sum to per-metric totals). Coverage 1,952/1,952.
