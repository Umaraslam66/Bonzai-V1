# Localization diagnostic — 4 held-out cities (Task 23 step 5)

**Run:** 2026-06-10 22:23–22:28 CEST, Leonardo login node (CPU, zero GPU), tmux, RC=0.
**Command:** `.venv/bin/python scripts/run_localization_diagnostic.py --release 2026-04-15.0`
(defaults: cities = eisenhuttenstadt/glasgow/krakow/munich, min_n=50, alpha=0.05,
effect_size_floor=0.15). **Code:** branch `phase-2-readiness-closure` @ `c32d41a`
(deployed by git bundle). **Artifact:** `reports/2026-06-10-localization-diagnostic.yaml`
(re-read from disk; verified identical to the printed summary on every variant).

## Variant table (recalibrated verdict, δ=0.15)

| variant | metric | n_pairs | n_sig_effect | median KS | | total raw_bh | total effect | effect rate |
|---|---|---|---|---|---|---|---|---|
| V0 (gate-(i) baseline) | building_area_m2 | 141 | 82 | 0.1854 | | 303 | **141** | 43.9% |
| | road_length_m | 180 | 59 | 0.1186 | | | | |
| V1 (per-cell un-collapse) | building_area_m2 | 49 | 23 | 0.1445 | | 112 | **42** | 34.4% |
| | road_length_m | 73 | 19 | 0.1055 | | | | |
| V2_8 (8-bucket density) | building_area_m2 | 123 | 71 | 0.1747 | | 258 | **117** | 42.2% |
| | road_length_m | 154 | 46 | 0.1222 | | | | |
| V2_16 (16-bucket density) | building_area_m2 | 198 | 122 | 0.1963 | | 423 | **210** | 45.4% |
| | road_length_m | 265 | 88 | 0.1215 | | | | |
| V3 (+ per-cell sea bucket) | building_area_m2 | 141 | 82 | 0.1854 | | 303 | **141** | 43.9% |
| | road_length_m | 180 | 59 | 0.1186 | | | | |

Tile coverage (F3): eisenhuttenstadt 616/616, glasgow 549/549, krakow 616/616,
munich 171/171 — zero skipped. Bref-excluded (identity-keyed, counted): 43,326 /
77,735 / 102,104 / 60,965. Feature pool identical across variants per city
(202,202 / 737,295 / 827,805 / 494,909) — the ±20%-of-V0 denominator check passes
as exact equality (uniform-pool construction).

## Verified-end-state checks performed

1. YAML re-read from Leonardo disk; every per-metric count/median and per-variant
   verdict matches the printed table (full-precision comparison).
2. Per-variant per-city n within ±20% of V0: exact equality (by construction).
3. Rough numbers: V0 still FAILs at δ=0.15 (recalibration alone does not close T5 —
   expected; this is why enrichment exists). Bref counters noisy-nonzero per city.
4. **V3 ≡ V0 bit-identical was drilled, not accepted at face value.** Glasgow has
   187 sea-positive cells carrying 2,067 pooled features, so a no-op was NOT
   trivially expected. Root cause is structural: `sub_c/conditioning.py` labels a
   tile coastal iff ANY cell has sea_water_fraction > 0, and glasgow is the only
   held-out city with coastal tiles (14; the other three are inland/river-only).
   Every sea-positive cell therefore lives in a (zoning, skeleton, density,
   coastal=3) stratum that exists in ONE city only — never paired, in V0 and V3
   alike. **V3 was structurally incapable of moving the verdict on this held-out
   set** (the existing coastal dim screens the candidate sea dim whenever only one
   city has sea tiles). "Sea adds nothing" must NOT be read as evidence about the
   sea feature in general — it is a single-sea-city artifact of this holdout.

## Reading (for the step-6 PI gate; decision is Umar's)

> **SUPERSEDED IN PART by the V1b/V1d attribution run below (same evening):** the
> "V1 kills the most signal → collapse layer" bullet survives only in count terms
> and is shown by the 2×2 to be mostly a denominator artifact of dropping dims.
> Read §"Attribution run" before acting on anything in this section.

- **V1 kills the most signal** (141 → 42 effect-significant; rate 43.9% → 34.4%):
  a large share of apparent cross-city character is tile-collapse mis-assignment
  (features inheriting tile-dominant zoning/skeleton rather than their own cell's).
  By the locked selection criterion (largest drop in n_significant_effect at
  δ=0.15) the collapse layer is where most recoverable character lives.
- **V2 does not kill signal** (rate ≈ flat at 8 buckets, slightly UP at 16):
  density quantization is not where character hides; finer buckets just expose
  more qualifying comparisons at the same discrimination rate.
- **V3 is structurally untestable here** (above) — its candidate dim needs a
  ≥2-sea-city holdout to be measurable at all.
- **Residual:** even at per-cell granularity 42 pairs (34.4%) remain effect-
  significant — genuine city character beyond per-cell zoning+density that the
  conditioning vector does not carry today. This residual is what Task 24's
  character feature must target.

Methodology knobs serialized in the YAML (V1 stratum-REPLACEMENT reading; V2
equal-width [0,1] top-inclusive buckets; V3 sea buckets {≤1e-9, (·,0.5], >0.5})
— judge against the data at the gate, not on faith.

---

# Attribution run — V1b/V1d 2×2 (PI-ordered at the step-6 gate, same evening)

**Run 2:** 2026-06-10 23:17–23:24 CEST, Leonardo login node (CPU), code @ `e08177c`
(V1b/V1d variants added at `f69cbc7`; methodology-label harmonization at `e08177c`),
RC=0. The PI-requested "V1c = per-cell-density-only" is identical to V0 by
construction (V0's density slot is already per-cell) — recorded loudly as
`methodology.v1c_note` in the YAML; the correct decomposition is the 2×2 below.

**Verified-end-state:** YAML re-read and matched to the printed table at full
precision on all 7 variants; per-city pools exactly equal across variants;
**run-1's five variant blocks reproduced bit-identically in run 2** (whole-pipeline
determinism check, free); coverage unchanged (1952/1952, zero skipped).

## The 2×2 (n_significant_effect / n_pairs / rate)

| | dims KEPT (skeleton+coastal) | dims DROPPED |
|---|---|---|
| **tile zoning** | V0: 141 / 321 / **43.9%** | V1d: 36 / 97 / **37.1%** |
| **per-cell zoning** | V1b: 160 / 391 / **40.9%** | V1: 42 / 122 / **34.4%** |

## What the isolation shows

1. **Run-1's headline inverted.** The 141→42 count drop attributed to "per-cell
   un-collapse" is carried almost entirely by the DIM DROP (V1d alone: 36), and
   that drop is largely mechanical: removing skeleton+coastal collapses 321
   comparisons into 97 — fewer tests can only produce fewer significant counts.
   The locked count-based criterion ("largest drop in n_significant_effect") is
   confounded by the changing pair denominator across variants.
2. **Per-cell zoning is NOT the character feature.** V1b (the clean swap, dims
   held) RAISES the count (141→160; per-cell zoning creates more qualifying
   strata, 391 pairs) and trims the rate only 43.9%→40.9%. If cross-city
   character were mis-assigned tile-dominant zoning, V1b would have absorbed it.
   It did not.
3. **Denominator-honest view (rates):** zoning swap −3.0pp, dim drop −6.8pp,
   both −9.5pp (≈ additive). No tested layer kills the discrimination signal;
   it persists at 34–45% under every stratum refinement (V2 flat, V3
   structurally untestable here).
4. **Net data statement for the gate:** the cross-city character is NOT
   localized in any tested stratum layer (collapse, quantization, sea dim).
   The residual discrimination is city-shaped — consistent with the spec's
   city-identity floor (#13) carrying the load, with the "character" axis still
   unidentified by this diagnostic. What to wire as Task 24's character field
   is therefore a genuinely open PI decision; the diagnostic's contribution is
   to RULE OUT per-cell zoning and density quantization as candidates.
