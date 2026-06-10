# Gate input (i) — conditioning-discrimination: VERDICT = FAIL → T5 reopens (2026-06-10)

**Run:** Leonardo job `45452784` (`lrd_all_serial`, budget-free CPU), build `619405a`.
Result YAML: `reports/2026-06-10-gate-i-conditioning-discrimination-result.yaml`.
Design: `reports/2026-06-10-gate-i-conditioning-discrimination-design.md` (PI-locked).
**No GPU was touched.** The gate ran model-independently on real held-out tiles, before any pilot.

## Verdict: FAIL (robust, well-supported, artifact ruled out)
`verdict: FAIL`; both metrics FAIL. The macro-plan conditioning the model is handed
`(zoning, road_skeleton, cell_density_bucket, coastal_inland_river)` does **not** explain
the cross-city variation in real geometry. Per the locked design, **T5 reopens** — the
per-city worst-case bar is invalid, because a per-city miss is ambiguous ("wasn't told
the city's character" vs "failed to render the handed structure").

## Evidence (every number with its denominator)
- **304 of 321** qualifying comparisons BH-significant (94.7%); building_area 136/141, road_length 168/180.
- **44 of 45** compared strata have ≥1 failing pair. Only 17/321 within the noise floor.
- Test well-supported (NOT UNSUPPORTED): 321 comparisons / 45 strata at min_n=50; 54 (city,stratum,metric) cells excluded thin-n; 48 strata had <2 comparable cities.
- (ks − floor) across pairs: min −0.054, p25 0.037, median 0.078, p75 0.122, **max 0.554**.
- Effect range: many pairs are *modest-but-certain* (ks≈0.08 made significant by huge per-stratum n, tens of thousands of features → tiny floor); SOME are *large*. Example (worst building_area stratum `(1,2,0,2)`): **glasgow median 59.3 m² (p75 112) vs krakow 25.0 m² (p75 36)** — ~2× at identical conditioning. The large practical differences mean the FAIL is substance, not merely a huge-n significance artifact.

## Artifact ruled out
Per-city overall magnitudes are all physically sane and same-order: building-area medians
40.5 / 55.2 / 45.5 / 60.0 m² (eisenhuttenstadt/glasgow/krakow/munich); road-length medians
all ~38–40 m. No order-of-magnitude offset. The decoder is shared across cities and the
token layer is CRS-agnostic (CRS baked out at encode) — there is no per-city decode branch
to introduce a bias. The differences are real morphology.

## Found-and-fixed during the build (caught before any verdict, by small-before-big)
- **`bb26cf2`** verdict module (BH multiple-comparison guard, per-pair noise floors, per-metric, UNSUPPORTED). MC-guard non-vacuous (a raw exceedance existed; BH suppressed it).
- **`619405a`** extraction promotion fix: the sanity-extract showed munich `building_area=0` — the construction-identity trap (decoder returns building closed-rings as LineString; without `promote_building_rings` every building was miscounted as a road). Fixed; 87,517 building features reappeared (468,357 + 87,517 = 555,874 exactly).

## Implication / open decision (PI)
T5 (multi-region eval semantics) reopens. This is the foreseen "conditioning-expressivity
gap" the delta-design flagged as a real possibility. The per-city worst-case (or mean) bar
cannot serve as the Phase-2 generalization decision axis at this conditioning. Options to be
decided with the PI — NOT pre-resolved here. **HALT: no scored runs, no GPU, no Task 10.**

> Erratum 2026-06-10 (post-audit): "the macro-plan conditioning the model is handed" describes the DATA plan (derived + stored), not the model INPUT — the running model received a constant value-agnostic slot prefix (readiness enumeration F6). The gate's verdict is model-independent and unaffected.
