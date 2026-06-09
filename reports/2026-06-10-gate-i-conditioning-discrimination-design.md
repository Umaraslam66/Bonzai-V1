# Gate input (i) — conditioning-discrimination, locked operationalization (2026-06-10)

The Task-9 conditioning-discrimination HALT-gate, computed on CPU from REAL held-out
tiles **before any GPU pilot** (model-independent). PASS ⇒ the worst-case bar is valid;
FAIL ⇒ T5 reopens; UNSUPPORTED ⇒ the held-out set can't support the test at full
granularity (report, do NOT coarsen to force a verdict). Design locked by PI 2026-06-10.

## Meta-principle (PI)
Test EXACTLY what the model is conditioned on, with thresholds paired to actual
per-comparison n, guarded against multiple-comparison false-FAILs, every n reported.

## Stratum = the FULL conditioning tuple (do NOT coarsen)
`(zoning_class, road_skeleton_class, cell_density_bucket, coastal_inland_river)` — the
exact macro plan the model is handed (T5). Per-feature assembly:
- `zoning_class`, `road_skeleton_class`, `coastal_inland_river` — TILE-level (`read_tile_labels` → `MorphologyStratum` + `coastal_inland_river`).
- `cell_density_bucket` — PER-CELL (`decode_region_blocks(tokens, cell_density_by_cell)` tags each feature with its cell's density).
Leaving any dimension out lets real same-conditioning variation masquerade as cross-city
difference (false FAIL) or hides it (false PASS). Thin-n is handled by EXCLUDING-AND-
REPORTING thin cells (below), NEVER by dropping a stratum dimension.

## Features (per metric, separately)
`building_area_m2` (from (Multi)Polygon geoms) and `road_length_m` (from (Multi)LineString),
via `realism.feature_samples`. Computed and reported SEPARATELY; the verdict fires if
EITHER metric shows real cross-city difference (worst-across-both).

## Extraction (Leonardo CPU; reuses sealed pieces)
Per held-out city ∈ {eisenhuttenstadt, glasgow, krakow, munich}, per held-out tile:
`read_tile_labels` (tile dims) + `read_sub_f_cells` + `_cell_density_by_cell` →
`decode_region_blocks` → per-feature (geom, density). Tag each feature with the full
stratum + metric. Accumulate `features[(city, stratum, metric)] -> list[float]`.
(Tile dir reads go through the **step-0-fixed** eval path — region-CRS labels.)

## Verdict (PURE, unit-tested locally)
`conditioning_discrimination_verdict(features_by_city_stratum_metric, *, min_n, alpha=0.05)`:
- For each metric, for each stratum present in ≥2 cities each with n ≥ `min_n`:
  per unordered city-pair → `D = ks_distance`, **per-comparison noise floor**
  `floor = 1.36·√((n₁+n₂)/(n₁n₂))` (α=0.05 two-sample KS critical value — the threshold
  PAIRED to the n that made it), and the asymptotic two-sample KS p-value
  `p = Q_KS(D·√(n₁n₂/(n₁+n₂)))`, `Q_KS(λ)=2Σ(-1)^{k-1}e^{-2k²λ²}` (clamped [0,1]).
- **Multiple-comparisons guard (REQUIRED):** Benjamini-Hochberg correct the p-values
  across ALL per-pair tests (the "pair count"). A pair "really differs" iff its BH-adjusted
  p < α. The raw worst KS must NOT fire the HALT alone — 40 same-distribution pairs expect
  ~2 raw exceedances by chance.
- **Per-metric verdict:** FAIL if ≥1 pair in that metric is BH-significant. **Overall:**
  FAIL if either metric FAILs; else PASS.
- **UNSUPPORTED:** if thin-n exclusion leaves too few comparable strata (0 qualifying
  comparisons, or below a reported floor), verdict = UNSUPPORTED — report, escalate, do
  NOT coarsen.

## Reported fields (every number with its denominator)
per-(stratum, city, metric) n; per-pair (D, floor, raw p, BH-adjusted p); the distribution
of (D − floor) across pairs; per-metric verdict; overall verdict; min_n; count of
(stratum,city,metric) cells excluded thin-n; count of strata with <2 comparable cities;
number of qualifying comparisons (the "is the test even supported" signal).

## TDD teeth (red-on-divergence)
1. PASS: same-distribution same-stratum across cities → all D ≤ floor → PASS.
2. FAIL: a stratum where cities genuinely differ beyond floor, surviving BH → FAIL.
3. **MC-guard (own teeth):** many same-distribution pairs (~40) → ~2 raw KS>floor by
   chance → BH correction → NOT FAIL (a single noise-tail outlier does NOT reopen T5).
4. thin-n: (stratum,city,metric) below min_n excluded + counted.
5. UNSUPPORTED: all strata thin → verdict UNSUPPORTED (not a silent PASS).
6. per-metric: real difference in road-length but not building-area → still FAIL.
