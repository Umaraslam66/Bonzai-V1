# Known issues

A short, in-tree list of accepted-but-not-yet-fixed issues. Each entry says: where the issue is, why we accepted it, what blocks fixing, and when we have to fix it.

Add new entries on top. Remove entries when they're fixed.

---

## #22 — SG-constant conditioning leak (morphology_class / country / climate_zone hardcoded across the frozen EU corpus) — BUNDLED WITH #13

- **Filed:** 2026-06-08 (eval-set-gen T6 de-Singapore source check)
- **Severity:** medium (latent; same class as #13 — inert today, contaminates value-bearing conditioning)
- **Status:** DEFERRED — **⛔ HARD GATE, BUNDLED WITH #13.** Both are SG value-bearing leaks; both block the same value-bearing conditioning; **both MUST be fixed in the SAME single corpus re-derive — never separately.** Re-deriving for admin_region alone and leaving morphology_class/country/climate SG is the partial-fix trap this bundling prevents.
- **Affects:** `src/cfm/data/sub_c/conditioning.py:29` (`morphology_class="Asian-megacity"`, `era_class="contemporary"` defaults never overridden for EU); the frozen `effective_conditioning.yaml` across all 42 EU cities.

### Context

`compute_conditioning_per_tile` defaults `morphology_class="Asian-megacity"`; the multi-region pipeline never passed a per-city value, so every EU tile's frozen conditioning carries `morphology_class=Asian-megacity`, `country=SG`, `climate_zone=tropical_rainforest` (verified 2026-06-08 across manchester/munich/krakow/malmo). `coastal_inland_river` is the exception — genuinely derived per-tile (`conditioning.py:56-72`), real and varying.

### Inert under value-agnostic v1 (why not a v1 blocker)

v1 conditioning is value-AGNOSTIC (8 field-SLOT tokens, no value channel — see #13): the model never sees these values, so the SG constants don't reach training or the v1 eval bars (loss + coherence + macro→micro perplexity-gap, which score the macro plan / micro tokens, NOT conditioning values). The eval `labels.py` routes morphology via the REAL `morphology_stratum` (sub-D), keeps `sub_c_morphology_class` UNSCORED, and never extracts country/climate/era into the scored surface → **no SG constant is scored anywhere.**

### HARD GATE (bundled with #13)

BEFORE value-bearing conditioning / full §9.115 conditioning-compliance: re-derive admin_region (#13/#14) **AND** morphology_class/country/climate_zone (#22) in **ONE** corpus re-derive. Fixing one without the other ships a half-de-Singapored conditioning vector.

### Tracking

- Source: `sub_c/conditioning.py:29`. Couples to **#13/#14** (same re-derive, bundled). Surfaced: eval-set-gen T6 de-Singapore source check.

---

## #21 — munich's fallback_bbox is inner-core only (~3.7× smaller than peer cities)

- **Filed:** 2026-06-08 (eval-set-gen, held-out-set fork — munich tile-count trace)
- **Severity:** low (frozen-corpus scoping fact; NOT a derivation/clip defect)
- **Status:** ACCEPTED, WON'T-FIX for v1 (corpus is frozen). Revisit only at a corpus regen if munich full-extent is ever needed.
- **Affects:** `configs/data/regions/munich.yaml` (`fallback_bbox: [11.36, 48.06, 11.72, 48.25]`)

### Context

munich's `fallback_bbox` is `0.360 × 0.190°` (≈561.7 km²) vs the peer-standard `0.600 × 0.480°` (≈2000+ km²) used by manchester/malmo/szczecin/lisbon — a **~3.7× smaller geographic extent** (and NOT zone-forced: a full box stays inside UTM zone 32). So munich = **inner-core only** → 171 tiles (vs mixed/moderate peer-median 588). Traced to source 2026-06-08: on-disk `sub_f/munich` = **171 tile dirs** (matches the G4 yaml); tile *density* is **30.45 tiles/100km², slightly ABOVE peers (~27–28)** — the derivation is consistent and nothing was clipped/dropped. **The low count is the small bbox, NOT under-derivation.**

### Consequence

**munich-derived statistics are core-scoped.** Any per-city aggregate (including the eval-set held-out coherence metric, where munich is the n=171 per-stratum power floor) reflects munich's inner core, not its metro extent. The held-out-set design accepts munich as a valid (real-count) city with this caveat recorded.

### Tracking

- Source: `configs/data/regions/munich.yaml`. Couples to **#15** (fallback bbox is the tiled extent). Surfaced: eval-set-gen held-out fork, munich-171 trace.

---

## #20 — rotterdam + warsaw are DEGRADED SOURCE DATA (~12–13× quantum-inflation), not a pipeline bug

- **Filed:** 2026-06-06 (Phase-2 corpus completion, the 6-city inflation gate)
- **Severity:** medium (these two cities carry ~12–13× the corpus-normal building-inflation rate; an OVERTURE source-data quality problem)
- **Status:** **OVERTURNED + RE-ADMITTED (2026-06-07).** The "degraded source, do-not-re-run" verdict was correct *given the encoder of the time* (a plain re-run reproduced the inflation), but the degradation was inflation **severity**, not a different corruption class — and the #19 de-densify fix removes it **at root**. After the de-densify re-derive, rotterdam + warsaw pass sub_g clean AND the path-length spot-check (`spotcheck_pathlength.py`) shows them undistorted vs the accepted eindhoven baseline (mean ≈1.0×, **zero** buildings >1.5×; worst real building ~1.3×, the same codec-quantization ceiling accepted cities show — see #19). PI re-admitted them as full corpus members (+77.1M). amsterdam (~3× borderline) likewise re-derived + validated and counts.
- **Affects:** (historical) the shipped-corpus selection; Overture buildings in rotterdam/warsaw boxes — now re-admitted via the #19 fix.

### Context

The decoded-vertex inflation defect (#19) is a corpus-wide ~0.05% token-share tail. The 6-city prevalence gate (`measure_inflation_prevalence.py`, 2026-06-06) measured each sub-G-failed city's own building-inflation distribution against the 29-validated reference (≥2× building token-share = 0.05%):

- **lodz 0.03%, a_coruna 0.04%, almere 0.04%** — indistinguishable from the corpus tail (corpus-normal, edge-tripped only).
- **amsterdam 0.14%** (~3×) — elevated, borderline.
- **rotterdam 0.64% (~13×), warsaw 0.62% (~12×)** — genuinely degraded: their Overture buildings are far more over-densified (sub-quantum vertices) than normal. rotterdam ≥4× token-share = 0.50% (50× the norm).

These two are not the pipeline mis-handling normal data; they are abnormally-densified SOURCE polygons. A blanket "construction-identity exclusion" of the decoded-vertex check (recover-3 / B2) would have MASKED this degraded subset — caught by the per-city gate (the "aggregate hides subsets" / touch-as-cross discipline).

### Fix / handling

Initially excluded (B1-simple, 2026-06-06). The prescribed "de-densify the source first, don't just re-run" path was then executed: the #19 `dedensify_coords` fix + guarded sub_f re-derive (2026-06-07) — exactly the regen-era fix this entry pointed to. Re-admitted after the spot-check confirmed undistortion vs the accepted baseline (Status above). The 6-city prevalence numbers stand as the *historical* characterization of the pre-fix inflation severity; they are no longer an exclusion rationale.

### Tracking

- Source: Overture source densification, removed at encode by the #19 de-densify fix. Evidence: `reports/2026-06-06-inflation-prevalence-the6.txt` (pre-fix severity) + `logs/rederive/spotcheck_3city.txt` (post-fix undistortion). Couples to **#19**.

---

## #19 — 0.5 m magnitude-quantum inflates over-densified source polygons (decoded-vertex bloat)

- **Filed:** 2026-06-06 (Phase-2 corpus completion)
- **Severity:** low corpus-wide (0.05% building token-share at ≥2× inflation; 0.01% at ≥4×) — a documented v1 limitation alongside alpha-drop (0.114%) and touch-as-cross (#17, 0.0064%)
- **Status:** FIX IN PROGRESS (2026-06-07, branch `phase-2-corpus-completion`). PI elevated Path A (de-densify + re-derive) to the critical path after add-cities failed to deliver — G4 failed all 3 gates (431M < 550M, eindhoven/tilburg/szczecin not validated, NL uncovered) and #19 recurred in the new NL cities, blocking NL coverage + ~170M corpus-normal tokens. Fix implemented + teeth-proven (see "Fix landed" below); corpus re-derive + G4 re-validation pending. Was ACCEPTED for v1 when this was a 0.05% tail; the cost-benefit inverted once it became the dominant failure in cities the corpus must include. The `decoded_vertex_within_cell_bound` sub-G check trips on the edge-adjacent worst cases.
- **Affects:** `src/cfm/data/sub_f/encoder.py` magnitude quantization; over-densified Overture polygons (buildings) with sub-quantum segments.

### Context

The sub-F encoder quantizes each (dir, mag) segment to a **0.5 m magnitude floor** (`DEFAULT_MAGNITUDE_QUANTUM_M`). Overture buildings that are over-densified (segments as short as ~0.04 m — a curved-facade digitization artifact) have each sub-quantum segment **rounded UP to 0.5 m**, inflating decoded perimeter **1.5–6.8×** vs the sub_c source (proven 2026-06-06 by authoritative-pairing path-length trace + segment-level evidence: `SRC 0.04 m → DEC 0.5 m`). The decoded vertex lands past the 300 m structural bound because it was *inflated* there, not because the feature is there — so relaxing the bound would bless 2–6× bloated buildings ("coherent ≠ correct"). Prevalence measured corpus-wide (`measure_inflation_prevalence.py`): **≥2× = 0.05% / ≥4× = 0.01%** of building token-share, **uniform across morphologies** (medieval-cores hypothesis falsified).

### Fix (regen era)

De-densify / simplify source polygons to the ~0.5 m quantum BEFORE encoding (drop or merge sub-quantum vertices — no information loss at the representational floor), so the encoder stops rounding sub-quantum segments up. Then re-derive. Verify whether the spec intended a pre-simplification step (missing-step) vs a design gap. The recover-3 option (per-city-prevalence-gated structural exclusion of the decoded-vertex check, with a teeth-proof) is banked for this regen window — it was deliberately NOT shipped under deadline as new validator surface.

### Fix landed — `dedensify_coords` (2026-06-07, pending corpus re-derive validation)

`encoder.dedensify_coords` (called at the top of `encode_feature`, mirrored in the
`token_cost.py` budget twin) does **radial-distance simplification at tolerance =
magnitude quantum**: keep `coords[0]`; keep each later vertex only when its straight-line
distance from the last KEPT vertex is `>= 0.5 m`; always keep the final vertex (preserves
anchor, ring closure, road bref endpoints exactly). This removes only sub-quantum
micro-vertices the 0.5 m quantum cannot represent anyway, so it CANNOT destroy
representable geometry — a ring already spaced `>= quantum` is returned vertex-for-vertex.

- **Teeth-proof:** `tests/data/sub_f/test_dedensify.py` — (a) an over-densified building
  (0.0735 m segmentation) decoded **6.80× / max-coord 372 m before → ~1.0× / within the
  300 m bound after**; (b) over-simplify GUARD — a legit-detailed building is untouched,
  with a discrimination teeth test proving a large tolerance DOES drop vertices (the
  preservation assertion is not vacuous).
- **DERIVATION version bumped 1.1 → 1.2** (`sub_f/versions.py`): the fix changes
  `cells.parquet` token bytes for the same input on over-densified features, so the axis
  bump forces the whole-corpus re-derive and blocks a pre/post-fix version skew (same
  defect class as the cycle-1 N/S bump).
- **Degenerate-extent addendum (v1 edge):** a feature whose WHOLE extent is `< 0.5 m`
  de-densifies below a valid vertex count; `dedensify_coords` falls back to the original
  coords (no collapse to an invalid ring). Such sub-resolution features can still inflate
  but stay far below the 300 m bound (bounded, and effectively non-existent in Overture —
  min building footprints are several m²). Not gated; documented.
- **Real-data confirmation (DONE 2026-06-07):** eindhoven gate — #19 cleared
  (decoded_vertex 1→0), sub_c/d/e byte-untouched (sub_f-only), token yield 26.9M→26.0M
  (−3.32%, correct shrinkage of sub-resolution noise), geometry valid. Corpus-wide:
  tiles_changed ≈ 603/611 → over-densification is **pervasive**, not a rare tail, but the
  shrink is small (NL ~3%, DE/FR <1.5%). rotterdam/warsaw recovered + re-admitted (#20).
- **CODEC CEILING — distinct from #19, do NOT mistake for unfixed inflation:** after the
  fix, the per-building decoded/source path-length ratio is centered at ~1.0× (mean 0.997–
  0.999×, p99 ~1.05×) but has a thin upper tail — the **worst single building is ~1.3×
  (max ~1.6×)** in EVERY city, INCLUDING accepted ones (eindhoven worst-large 1.314×). This
  is the inherent encode/decode quantization ceiling (0.5 m magnitude + 1° direction bins
  on certain building shapes), governed by the BP2 round-trip L_inf lock, NOT residual #19
  inflation (#19 was pervasive and would lift the *mean/p99*, not a lone tail). The
  spot-check therefore judges the BULK (mean/p99) against the accepted baseline, treating
  the ~1.3× tail as info. Sub-quantum-extent clip slivers (source <2 m) decode to a bounded
  ≤4 m and are the degenerate edge above, not distortion.

### Tracking

- Source: `encoder.py` magnitude quantization. Evidence: `reports/2026-06-06-inflation-prevalence.txt` (29-ref) + close-out. Couples to **#20**.

---

## #18 — destructive in-place re-derives must use the guarded tool (not hand-rolled shell)

- **Filed:** 2026-06-05 (sub-F v1.2 corpus re-derive; near-miss)
- **Severity:** medium (process/safety — a corruption hazard against the single-copy corpus)
- **Status:** MITIGATED on BOTH layers — `scripts/multiregion/guarded_rederive.py` (sub-F destructive re-derive) **and** a baked-in guard on the sub-C run path (2026-06-05, branch `phase-2-corpus-completion`): `driver.run_city` holds a per-city `extract_lock` flock for the whole stage chain (a second concurrent extract refuses) and `io.write_parquet` is now atomic (temp + `os.replace`, the #18 truncation root cause). This entry is the standing mandate.
- **Affects:** any operation that re-derives a region in place (sub-F re-derive; the upcoming sub_c re-runs for the 13 timeouts / almere).

### Context

`pq.write_table` (`src/cfm/data/io.py:30`) writes `cells.parquet` **in place, not atomically**. An in-place re-derive therefore (a) truncates the prior-good tile during its write window (a kill there leaves an unreadable parquet — detectable but destroyed), and (b) for an already-`_SUCCESS` city, leaves the **stale `_SUCCESS`** in place during the rewrite, so a mid-rewrite kill yields a city that *looks* blessed but fails leg-5 version-consistency. On 2026-06-05 a hand-rolled `nohup` loop was also accidentally launched **twice**; two concurrent in-place re-derives of the same dirs would have corrupted the corpus — prevented only by wait-loop timing, NOT a safeguard. Zero corruption occurred, but the exposure was real.

### Mandate / fix

Use `python -m scripts.multiregion.guarded_rederive --city <c> ...` for ALL destructive re-derives. It enforces: **lockfile** (`fcntl.flock` — a second invocation refuses), **atomic temp-swap** (derive into a temp dir; replace live only on full success; live untouched during the kill-prone derive), and **halt-on-non-identical** (compare temp vs live before swap; HALT with live untouched unless `--allow-content-change`). 10 tests in `tests/data/multiregion/test_guarded_rederive.py`. This matters most for the **sub_c re-runs** (the EXPENSIVE extraction layer), where in-place corruption would destroy real compute, not cheaply-regenerable sub-F.

### sub-C guard (added 2026-06-05, branch `phase-2-corpus-completion`)

The sub-C re-runs go through the standard driver (`extract_region_batch.py → driver.run_batch → run_city`), not `guarded_rederive.py` (which is sub-F-only). So the guard is baked into that path instead of being an opt-in tool that can be forgotten:

- **Per-city lock** (`src/cfm/data/multiregion/extract_lock.py`): `run_city` acquires an exclusive non-blocking `fcntl.flock` (`data/processed/multiregion/.locks/<city>.extract.lock`) before the stage chain; a second concurrent extract of the same city returns `failed` immediately (continue-but-loud). Different cities still extract concurrently. The flock auto-releases on process death, so a watchdog kill leaves NO stale lock.
- **Crash-safe write** (`src/cfm/data/io.py::write_parquet`): now writes to a per-pid temp in the same dir and `os.replace`s into place — atomic on a POSIX same-fs rename, so a kill/write-failure mid-derive leaves the destination untouched (a prior-good tile is never truncated). Bytes are unchanged → byte-identity guarantees preserved. This is the direct fix for the `pq.write_table`-is-not-atomic root cause described above, and it covers ALL stages, not just sub-C.

Tests: `tests/data/test_io.py` (crash-safety), `tests/data/multiregion/test_extract_lock.py` (lock), `tests/data/multiregion/test_driver.py` (refuse-when-held + release-on-success/failure).

### Tracking

- Source: `guarded_rederive.py` (commit `edcb1b8`). Evidence/analysis: `reports/2026-06-05-subf-v1.2-revalidation-closeout.md` §8.

---

## #17 — sub_c records a §8.3 touch-at-boundary as a crossing (touch-as-cross root)

- **Filed:** 2026-06-05 (Phase-2 multiregion, sub-F validator v1.2)
- **Severity:** low (0.0064% of crossings; census anomaly=0 — no fragment is dropped) but **spec-violating**
- **Status:** DEFERRED to next regen — spec-violating per §8.3, **TOLERATED under validator v1.2**, MUST fix at the next sub-C regeneration. **Not "benign" — tolerated.**
- **Affects:** `src/cfm/data/sub_c/geom.py` — crossing records are derived from `per_cell_pieces` (`:141`, consumed at `:156–160`) which is computed BEFORE `apply_sliver_drop` (`:182`, the <0.01 m sliver / 0-d-collapse discard). (Handoff's `geom.py:557` cite was stale; anchor on the `per_cell_pieces`→crossing path vs `apply_sliver_drop` ordering.)

### Context

A road terminating exactly on an internal cell boundary (§8.3 touch-not-cross, which the spec says must produce **0** crossing records) is recorded AS a crossing: the crossing-record derivation keys off `per_cell_pieces` before the sliver/0-d discard removes the boundary-touch fragment, so the touch looks like "present in 2 cells" → a crossing row → sub-E marks the shared edge `MINOR_ROAD` → sub-F emits a `<bref>` on the one side that has the endpoint. The **corpus-wide census** (`symmetry_probe.py --touch-census`, 2026-06-05) measured this directly: **187 / 2,927,731 crossings = 0.0064%, anomaly = 0** (the teeth-proven `anomaly` column confirms NO fragment was actually dropped — these are terminations, not clip-drops). Under v1.2 the road-presence-conditioned symmetry/coverage legs tolerate the one-sided emission.

### Fix (regen era)

Require **positive-length** presence in `_partition_geometry_into_cells` / the crossing derivation (apply the sliver/0-d discard BEFORE deciding "both cells present"). **When fixed, the corpus SHIFTS at the ~54–65 symmetric touch-as-cross edges** — they lose their spurious `<bref>`. So this fix is NOT verdict-only; it changes token bytes and must re-derive + re-bless the affected tiles.

### Tracking

- Source: `geom.py` `per_cell_pieces`/`apply_sliver_drop` ordering. Surfaced: batch-2 sub-F BP7 symmetry-FP investigation (`reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md`). Couples to **#16** (both are regen-era sub-C clip-path prerequisites).

---

## #16 — the v1.2 relax's drop-guard `assert_lossless_clip` is TESTED-BUT-UNWIRED (no production caller)

- **Filed:** 2026-06-05 (Phase-2 multiregion, sub-F validator v1.2)
- **Severity:** medium (the intended **recurring** drop-guard does not run; until wired, source-trace + census-anomaly are the ONLY drop checks)
- **Status:** DEFERRED to next regen — wiring requires sub-C to persist `source_clipped_length_m` (a clip-path change + re-run). **Regen-era prerequisite alongside #17.**
- **Affects:** `src/cfm/data/multiregion/lossless_clip.py::assert_lossless_clip` (no production caller); the sub-C clip path (where it must be wired).

### Context

The v1.2 symmetry + coverage relax **deliberately blinds** those legs to a sub-C clip-DROP (a true crossing whose neighbour fragment was dropped — it looks identical to a §8.3 termination). `assert_lossless_clip` is the intended in-corpus **twin** that catches a drop by length conservation (`Σ per-cell fragment lengths == len(source ∩ bbox)`, tol 0.1 m). But it is **tested-but-unwired**: a repo-wide grep finds its only callers are its own 4 tests — it is NOT called by `pipeline.py` (sub-F derive), NOT by `validate_cross_tile` (docstring mention only, `validator_cross_tile.py:249`), NOT by any sub-C clip code. It needs `len(source ∩ bbox)` per feature computed at sub-C clip time, which sub-C does not currently persist or pass through.

### Consequence (un-missable)

- **No executing in-corpus drop-guard exists today.** The relax shipped on two widened legs with the twin wired to nothing.
- The ONLY running drop checks are: `symmetry_probe.py --source-trace` (traces symmetry-DISAGREEMENT edges only — blind on cities with 0 disagreements) **+** the touch-census `anomaly` column (corpus-wide; teeth-proven non-vacuous by `tests/data/multiregion/test_touch_census_teeth.py`). Both passed clean on 2026-06-05 (anomaly=0 corpus-wide; `len_in_MISS=0` on all SYM8 traced edges).
- **Future re-derives (the 13 sub-C timeouts, any regen) do NOT get the guard** unless the sub-C `source_clipped_length_m` wiring lands first. Wiring `assert_lossless_clip` into the sub-C clip path is a **regen-era prerequisite**.

### Tracking

- Source: `lossless_clip.py`. Evidence: `reports/2026-06-05-subf-v1.2-revalidation-closeout.md` §3. Couples to **#17**.

---

## #15 — sub_c tiles the fallback bbox, not the real Overture admin polygon

- **Filed:** 2026-06-04 (Phase-2 multiregion G3 / batch-2 scoping)
- **Severity:** low (diversity corpus tolerates over-inclusion; tile/token counts are NOT the sizing basis)
- **Status:** DEFERRED — acceptable for a diversity corpus; revisit only if true-extent counts ever become load-bearing
- **Affects:** `src/cfm/data/overture/loader.py::_build_region_geometry` (Phase-1 placeholder), `src/cfm/data/sub_c/pipeline.py:348`

### Context

`_build_region_geometry` is a Phase-1 placeholder: `admin_polygon = box(fallback_bbox)`. sub_c partitions tiles over (and clips features to) that box (`pipeline.py:348`), NOT the real divisions polygon — the divisions theme *is* fetched but feeds only the (separately-broken, see #13) admin_region lookup. The manifests' `admin_polygon_source: overture://divisions:...` is a cosmetic label, not evidence of a real polygon. **Confirmed in code 2026-06-04.** The boxes over-include (Prague's box ≈937 km² > its municipality ≈496 km²) — benign for a diversity corpus (extra fringe, not a clip). Consequence: per-city **tile/token counts are not trustworthy as true-extent measures**; per-tile tok/tile *direction* is robust. Sizing is by **diversity, not counts** (see handoff), so this does not block batch-2.

### For batch-2

Draw fallback bboxes **generously** (over-include rather than risk clipping a dense core). NOT bundled with #13/#14 (one reopen, one change, clean attribution).

### Tracking

- Source: `loader.py:193`, `pipeline.py:348`. Surfaced: G3 batch-2 scoping 2026-06-04.

---

## #14 — admin_region granularity is not comparable across countries (subtype='region')

- **Filed:** 2026-06-04 (Phase-2 multiregion batch-2 scoping)
- **Severity:** medium (semantics; couples to #13's hard gate)
- **Status:** DEFERRED — the correct cross-country granularity is a **value-bearing-conditioning (Task 7) design decision**; spec TBD. Do NOT guess it in an operational reopen.
- **Affects:** `src/cfm/data/sub_c/pipeline.py::_derive_region_lookup_svy21` (the `subtype='region'` filter)

### Context

`subtype='region'` means different administrative levels per country (inspected on the real cached divisions themes 2026-06-04): Singapore = **sub-city district** (5 regions, varies tile-to-tile); Spain = **Catalunya** (whole autonomous community — constant for every Barcelona tile); Germany = **Bayern** (whole Bundesland — constant for Munich); Czechia = **Praha / kraj** (Prague ≈ the city). So naively de-hardcoding `country_code` while keeping `subtype='region'` would NOT fix #13 — it would replace "null = European" with "near-unique province ID per EU city vs sub-city district for SG", a subtler asymmetry. The spec itself files the right subtype as TBD ("equivalent second-level subtype for other regions"). The correct choice depends on what value-bearing conditioning is *for*, which is a Task-7 design decision.

### Tracking

- Source: `pipeline.py::_derive_region_lookup_svy21`. Couples to #13. Surfaced: batch-2 scoping 2026-06-04.

---

## #13 — sub_c admin_region lookup hardcodes country_code='SG' → None for all non-SG tiles

- **Filed:** 2026-06-04 (Phase-2 multiregion G2/G3)
- **Severity:** medium (latent train-contamination: INERT today, becomes a systematic SG-vs-EU confound the moment value-bearing conditioning is enabled)
- **Status:** DEFERRED — **⛔ HARD GATE (see below): must be fixed before ANY value-bearing conditioning (Task 7 / bake-off candidate).**
- **Affects:** `src/cfm/data/sub_c/pipeline.py:337` (`country_code="SG"` hardcoded)

### Context

The admin_region lookup filters the divisions theme by `country='SG'`, so every non-Singapore tile gets `admin_region=None`. **INERT under slice-v1 value-agnostic conditioning** — the model consumes 8 field-SLOT tokens with no value channel (`micro_ar.py:15-21`), `datamodule.build_conditioning_prefix` (`datamodule.py:53-60`) returns the field-slot id-block, and `n_cond=8` means there is *no embedding id for any region value* — the model physically cannot see `None` vs a region string. The VALUE is recorded in the shard (`build_shards.py:88`) and `conditioning_prefix_ids` (the schema artifact) but is not trained on. Re-derivable from the **already-cached divisions theme** (no re-fetch needed).

### Why deferred (not fixed now)

Defer = ONE reopen at Task 7 (when the #14 granularity is decided regardless). Fixing now = that same Task-7 reopen PLUS a needless 5-city canary reopen — strictly worse, and it would block batch-2 for no durable gain. **`None` is the safer placeholder:** all-EU-null is glaringly unfinished (Task 7 MUST notice and re-derive); a plausible-wrong "Catalunya"-on-every-tile invites silent training on a mismatched signal.

### ⛔ HARD GATE

**BEFORE enabling any value-bearing conditioning (Task 7 / a bake-off candidate): admin_region MUST be re-derived with a deliberate cross-country granularity choice (#14) and the corpus reopened. Do NOT train value-bearing conditioning on the existing admin_region values — EU is all-`None` and SG is hardcoded.** Also recorded in spec §7 and the Phase-2 handoff. **BUNDLED WITH #22:** the same re-derive MUST also fix the morphology_class/country/climate_zone SG-constant leak (#22) — one re-derive, all SG value-bearing leaks, never separately (the partial-fix trap).

### Tracking

- Source: `pipeline.py:337`; conditioning path `conditioning.py`, `datamodule.py`, `micro_ar.py`, `build_shards.py`. Couples to #14. Surfaced: G2/G3 2026-06-04.

---

## #12 — Eval-set is FROZEN; three load-bearing carry-forward triggers for the training-scaffold / eval-harness phase

- **Filed:** 2026-06-01 (eval-set-generation close)
- **Severity:** medium (not a defect — these are trigger conditions the successor MUST honor, or eval numbers become invalid or silently under-powered)
- **Status:** open obligations on the successor sub-project (training scaffold + eval harness)
- **Affects:** `src/cfm/eval/holdout/*`, the frozen `data/processed/eval_set/2026-04-15.0/holdout_manifest.yaml` + `_EVAL_SET_LOCKED`

### Context

The Singapore held-out set is locked write-once: 132 tiles, KS target gap 0.08 (resolves 0.076), ρ=0.5, δ_floor=0.005, ceiling 0.968. Full record: `reports/phase-1-eval-set/2026-06-01-singapore-eval-set-FROZEN.md`.

### Triggers the successor MUST honor

1. **Holdout exclusion (one source):** the training data loader MUST call `cfm.eval.holdout.lineage_audit.audit_no_holdout_leak(manifest, training_reachable)` against the frozen manifest, fail-closed on absent lineage (G-F4). A contaminated holdout invalidates every eval number undetectably.
2. **Eval-harness fail-loud on resolution:** when models exist, assert the bake-off's needed architecture-distinguishing gap is **≥ 0.076** (the frozen set's resolved gap). Finer → the **second-region extraction trigger** (the deferred B-decision), NEVER silent under-power. The single-region hard floor is **0.049** — finer is categorically a second-region need, not an N-tuning knob.
3. **Conditioning vector (one source):** the model's conditioning MUST consume the same sub-C/sub-D quantities `cfm.eval.holdout.labels` reads (`population_density_bucket`, `cell_density_bucket`, the `morphology_stratum` = sub-D `road_skeleton_class`+`zoning_class` — NOT sub-C's constant `morphology_class`), or conditioning-compliance scoring is apples-to-oranges.

### Owed deferred items (spec §7)

The tokenizer-on-**model** side of R2, the Wasserstein/KS **distance** computation against model output, simulation-viability execution, and model-scoring orchestration are all deferred to the eval-harness. `ρ` is tunable down to ~0.25 once the model's natural over-emission variation is observed (does not move N).

### Tracking

- Source: `reports/phase-1-eval-set/2026-06-01-singapore-eval-set-FROZEN.md`; protocol §10 (`docs/protocols/sub-project-planning-protocol-v3.md`) records the freeze-gate principles this close earned.

---

## #11 — Layer-3 subset selector skips sparse-side dimensions (negated-positive-score interaction with eligibility guard)

- **Filed:** 2026-05-19 (Phase 1 sub-D Gate 2B review)
- **Severity:** low (Layer-3 subset is still diverse on the positive-side dimensions; sparse-side coverage missing but not load-bearing)
- **Status:** deferred — fix before any region enrollment whose Layer-3 subset must exercise sparse / low-density tiles
- **Affects:** `src/cfm/data/sub_d/frequency_analysis.py::_SUBSET_DIMENSIONS` (sparse-side entries with negated scores)

### Context

`select_layer3_subset` ranks tiles by a fixed dimension list. Sparse-side dimensions like `density_low`, `road_skeleton_sparse`, `scope_sparse_tile` use a key function that negates the underlying positive quantity (`lambda e: -e["density_signal"]["max"]`, etc.). The "top" candidate then has the least-negative score, which is still `<= 0`. The selector's downstream guard `if key_fn(top) <= 0 and top_key not in selected_keys: continue` skips these dimensions entirely.

Empirical evidence from Gate 2B on Singapore: 9 tiles selected, 3 sparse-side dimensions never picked a new tile.

### Fix

Replace the negation with a positive-magnitude reciprocal so sparse-side dimensions produce a positive score whose maximum corresponds to the sparsest tile:

```python
("density_low", lambda e: 1.0 / (e["density_signal"]["max"] + 1e-9)),
```

Same pattern for `road_skeleton_sparse`, `scope_sparse_tile`. The eligibility predicate (`active_cell_count > 0`) still filters empty tiles out before ranking; this fix just lets the survivors be ranked correctly.

### Tracking

- Source: `src/cfm/data/sub_d/frequency_analysis.py::_SUBSET_DIMENSIONS`
- Surfaced by: Gate 2B review of real Singapore proposal, section G #3

---

## #10 — Bucket-merge marginal-cost-of-cut metric is degenerate

- **Filed:** 2026-05-19 (Phase 1 sub-D Gate 2B review)
- **Severity:** medium (reviewer-facing: hides the cut-point elbow for bucket-based vocabs; not a correctness issue but degrades Gate 2 review quality)
- **Status:** deferred — replace before the next region enrollment that re-derives bucket cuts
- **Affects:** `src/cfm/data/sub_d/frequency_analysis.py::_fill_marginal_cost` applied to `_density_proposal_section`, `_road_proposal_section`, `_tile_population_density_proposal_section`

### Context

The marginal-cost-of-cut formula `(Δcoverage) / (Δcategories)` is well-defined for token-dropping (zoning's case, where merging a class into "other" reduces coverage of the surviving tokens). For bucket-merging strategies (density, road skeleton, tile population density), every value still falls into some bucket regardless of strategy, so coverage stays at 1.0 across all cut strategies — marginal cost is 0.0 for every entry, and the elbow the reviewer wants to see cannot be derived from the metric.

Empirical evidence from Gate 2B on Singapore: cell_density, road_skeleton, and all four tile_population_density proxies returned coverage=1.0, marginal_cost=0.0 for every strategy in `_DENSITY_CANDIDATE_BUCKETS` / `_ROAD_CANDIDATE_BUCKETS`. The Gate 2 reviewer made cut decisions from the section-C distribution summary instead.

### Fix

Replace coverage with a quantity that varies meaningfully under bucket-merging. Candidates:

1. **Entropy loss**: information lost when merging buckets. Sensitive to which buckets merge and how mass is distributed.
2. **Largest-bucket mass**: fraction of values in the biggest bucket. A bucketing that puts >50% in one bucket is degenerate; this metric catches that.
3. **Quantile-fit goodness**: KL-divergence between the bucket distribution and an idealized equal-quantile bucketing of the same N.

Recommend (1) or (3) for richer signal. Keep the `marginal_cost` field name on the candidate_strategies entries so the reviewer-facing table layout doesn't change.

### Tracking

- Source: `src/cfm/data/sub_d/frequency_analysis.py::_fill_marginal_cost`
- Surfaced by: Gate 2B review of real Singapore proposal, section G #2

---

## #9 — Cell density ratio exceeds 1.0 in real Singapore data

- **Filed:** 2026-05-19 (Phase 1 sub-D Gate 2B review)
- **Severity:** medium (mathematical invariant violation; affects 0.03% of cells but the upper bound is no longer guaranteed by construction)
- **Status:** deferred — investigate root cause before any region whose density bucketing depends on a strict `[0, 1]` bound
- **Affects:** Sub-C extraction output. Sub-D `derive_density_evidence` consumes the per-cell ratio without checking it, and the Gate 2B-locked `cell_density.locked_buckets` top bucket `[0.35, inf)` absorbs the anomaly gracefully — sub-D is not currently mis-derived.

### Context

The `building_footprint_ratio` metric in `derive_density_evidence` is `sum(building polygon area within cell) / cell_area_admin_clipped_m2`. Mathematically this must be `≤ 1.0` (a cell cannot be more than 100% covered by buildings). Real Singapore sub-C data violates this: across 17,049 active cells, `max = 1.4096` and ~0.03% of values are above 1.0.

Three plausible root causes:

1. **Overlapping Overture building polygons.** A single physical building represented by two overlapping polygons in `buildings.parquet` would double-count area in the sum.
2. **Multi-polygon cell-clipping edge cases.** Sub-C's int8 `GEOMETRY_TYPE` enum extension (known issue, project-memory `project_sub_c_multi_geometry_gap.md`) suggests Multi\* geometries are present; the cell-clipping logic might leave portions extending beyond cell bounds in some edge cases.
3. **Sliver-drop rule not pruning microscopic polygons that round up.** `sliver_drop_rule: drop iff geometry has area < 0.01 m² OR length < 0.01 m` — if a building polygon barely passes the threshold but extends slightly beyond a cell boundary, area accounting could over-sum.

### Investigation plan

Pick 3-5 cells with `building_footprint_ratio > 1.0` from Singapore output. For each, list the `source_feature_id`s of buildings clipped into that cell. Inspect whether any pair has overlapping geometry, Multi\* parts, or sub-cell-edge slivers. Update sub-C clipping or de-duplication accordingly.

### Why not weaken sub-D's invariant

Sub-D's `derive_tile_population_density_evidence` F2 test asserts `0.0 ≤ value ≤ 1.0`. Tile aggregates of >1.0 per-cell ratios happen to stay under 1.0 in current data (max p75 = 0.50, max area_weighted = 0.36), but the bound is no longer guaranteed by construction. Per `feedback_test_weakening_to_pass`: when data violates an invariant, the assumption failed; fix the upstream, do not weaken the test. The F2 [0, 1] assertion stays strict.

### Tracking

- Source: `src/cfm/data/sub_c/pipeline.py` (sub-C extraction)
- Spec: §11.3 (cells.parquet schema), §9.2 (sea-mask + sliver-drop order)
- Surfaced by: Gate 2B review of real Singapore proposal, section G #1
- Related: `project_sub_c_multi_geometry_gap.md` (memory) — Multi\* geometry handling
- Mitigation in place: sub-D `cell_density.locked_buckets` top bucket open-ended `[0.35, inf)` absorbs anomalous values without leaking into intermediate buckets.

---

## #8 — Cross-tile validator invariant #1 over-couples YAML-format version to data-shape version

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** low (works today; brittle on next schema change)
- **Status:** deferred — fix at the next schema version bump in either dimension
- **Affects:** `src/cfm/data/sub_c/validator_cross_tile.py::_check_schema_version_consistency`

### Context

`validate_extraction_cross_tile` invariant #1 (`sub_c_schema_version_consistency`) compares `manifest.sub_c_schema_version` (data-shape version) against `meta.yaml.schema_version` (YAML-format version). These are conceptually independent version series per spec §14.9. The current implementation forces `_SCHEMA_VERSION == _SUB_C_SCHEMA_VERSION` to satisfy the validator, which means any future YAML-format change forces a spurious data-shape version bump (or vice versa).

### Fix

Invariant #1 should compare like-for-like via a dedicated `features_parquet_schema_version` (or analogous data-shape version) on meta.yaml, keeping YAML-format version and data-shape version as separate fields.

### Tracking

- Source: `src/cfm/data/sub_c/validator_cross_tile.py::_check_schema_version_consistency`
- Spec: §14.9
- Surfaced by: Task 17 code review

---

## #7 — `--rerun` CLI path is a stub (`NotImplementedError`)

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** low (Singapore initial extraction works without re-run; deferred until needed)
- **Status:** deferred — implement when the first per-tile re-extraction need arises
- **Affects:** `scripts/extract_tiles.py`

### Context

`scripts/extract_tiles.py --rerun <i,j>` raises `NotImplementedError`. Per spec §11.8, per-tile re-extraction protocol is documented: read the existing manifest, re-extract just the named tile, update `manifest.tiles[<this_tile>].provenance_sha256`, and re-run the cross-tile validator. Not on the Singapore Phase 1 critical path.

### Effort estimate

Half a day of work. The extraction plumbing and manifest update logic are already in place; it is purely wiring and the per-tile re-run of the cross-tile validator.

### Tracking

- Source: `scripts/extract_tiles.py` (see `# DECISION:` near `--rerun` handler)
- Spec: §11.8

---


## #4 — RESOLVED (training-scaffold, 2026-06-01): `emit_unknown_token` fall-through handled on the live sub-F path (superseded by cascade #7)

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** high (training-critical path) — now RESOLVED
- **Status:** **RESOLVED 2026-06-01 (verify-before-lock).** The obligation is met on the **live Phase-1 path**; #4 was filed against the **non-training-reachable Phase-0** tokenizer. See Resolution below.
- **Affects (filed):** `src/cfm/tokenizer/encode.py:59-60` — the **Phase-0 proof-of-concept** tokenizer, imported **only** by `scripts/smoke.py`. Training never reaches it. It remains **knowingly unfixed** (it still hard-raises `UnsupportedFeatureClass`); that is acceptable because it is not on any training/production path.

### Resolution (2026-06-01, training-scaffold Task 1)

The Phase-1 micro-tokenizer is `src/cfm/data/sub_f/encoder.py`, built *after* #4 was filed. Its `_resolve_semantic_tag_to_token_id` (encoder.py:253-283, "cascade #7") already implements the fall-through #4 asked for: a non-sentinel not-in-vocab value buckets to the `<unknown_KEY>` BP4 family; it raises `KeyError` only when a key has **no** unknown-family slot (correct fail-loud on a true gap). Sub-C stores raw not-in-vocab building/POI values (`sub_c/policy.py:205`); they are turned into `key=value` semantic_tags by `_semantic_tag_from_row` (pipeline_writer.py:42, key ∈ {highway, building, amenity, natural}) and resolved via cascade #7.

Verified on the real frozen Singapore sub-C output by `tests/eval/test_unknown_class_fallthrough.py` (3 assertions): (1) **88 distinct real semantic_tags, 10 route to `<unknown_KEY>`** — non-vacuous, the unknown-class regime is genuinely exercised, and none raise; (2) a key with no `<unknown_KEY>` slot still raises `KeyError` (twin: a key with a slot resolves); (3) the round-trip is a **known lossy collapse** (`building=<unknown> → <unknown_building> → generic building`, identity lost by design — a v1 limitation, reported-not-gated per protocol v3 §9).

Cross-reference: spec `docs/superpowers/specs/2026-06-01-phase-1-training-scaffold-design.md` §2; memory `feedback_filed_issue_location_can_be_stale`.

### Context

Sub-C output is unusable end-to-end for training without an enhancement at `src/cfm/tokenizer/encode.py:59-60`. Currently `_encode_feature` hard-raises `UnsupportedFeatureClass` on any class value not in the vocab YAML. Under Topic 3b Option A, sub-C stores raw class values; downstream tokenization at training time must fall through to `<prefix>__UNK__` when the field's `missing_value_policy` is `emit_unknown_token`. The change is estimated at ~10 lines but warrants its own brainstorm-spec-test cycle and a separate sub-project.

Sub-D through sub-G consume per-cell rows directly, not tokens, so this is not a blocker for those sub-projects.

### Fix

In `_encode_feature`, after failing to look up the class in the vocab, check the field's policy. If `emit_unknown_token`, emit the appropriate `__UNK__` token. If `raise_error` (current behaviour), keep the raise.

### Tracking

- Source: `src/cfm/tokenizer/encode.py:59-60`
- Spec: §3 + §20

---

## #3 — Sweden densification revisit required before first Sweden extraction (spec §7.4)

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** medium (correctness on coastline-heavy regions)
- **Status:** deferred — measure and tune before `extract_tiles.py --region sweden` is first run
- **Affects:** `src/cfm/data/sub_c/coords.py::densify_polygon`

### Context

Sub-C's `densify_polygon` is called with `max_edge_length_m=None` for Singapore. This is empirically a no-op: the Singapore polygon's maximum edge is 775 m and 99 % of edges are under 500 m, so inserting intermediate vertices would change nothing. Sweden is a different story: higher latitudes introduce a cos(lat) projection-compression effect, and archipelago coastlines can have far longer edges. Skipping densification on a 20 km coastline edge would place a sea/land boundary vertex far from the true geodesic path, corrupting cell-coverage fractions.

### Fix

Before the first Sweden extraction run:
1. Measure the edge-length distribution of Sweden's administrative boundary polygon (same method as Singapore was measured).
2. If any edge exceeds ~5 km, pass an appropriate `max_edge_length_m` value (e.g. 1000 m) to `densify_polygon`.
3. Document the chosen value in `configs/regions/sweden.yaml`.

The function signature is already in place; only the argument value needs tuning.

### Tracking

- Source: `src/cfm/data/sub_c/coords.py::densify_polygon`
- Spec: §7.4

---

## #2 — Subtype / subclass fields analyzed in B1 but not tokenized in Phase 1

- **Filed:** 2026-05-16 (Phase 1 sub-B2 spec)
- **Severity:** low (scope decision, not a bug)
- **Status:** deferred — picked up by a future sub-project after the encoder design extends to multi-token-per-feature
- **Affects:** `buildings.subtype`, `transportation.subclass`, `base.subtype` fields from the B1 report

### Context

The B1 frequency analysis covered nine fields including the three subtype/subclass fields above. B2's vocab YAML only tokenizes the four locked feature_class sections (road, building, poi, base) plus the alternate categories folded into the POI section via union. The three subtype/subclass fields are deferred.

### Why

The current tokenizer encoder is one-token-per-feature: `cfm.tokenizer.encode._encode_feature` reads `feature["properties"]["class"]` and emits exactly one feature_class token. Integrating subtype as a *second* token per feature (option B from the B2 brainstorm) or a *crossed* class×subtype token (option C) is a tokenizer architectural decision that warrants its own sub-project with its own brainstorm. B2 deliberately keeps subtype out of scope to avoid quietly expanding the encoder contract via a vocab YAML.

### Future

When subtype integration is on the table, a future sub-project picks between the options. Either way:

- The B1 numbers for `buildings.subtype` (Moderate keeps 11 cats), `transportation.subclass` (all 7 retained at every floor), and `base.subtype` (11 → 7 at Moderate) are already analyzed and ready to use.
- The Sweden re-run (B1') re-runs both class and subtype frequencies in parallel; subtype data will continue to land in the B1' report.

### Tracking

- B2 spec §2 (out-of-scope deferrals): `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md`
- B1 report §3.2, §3.4, §3.5: `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md`

---

## #1 — Cold-fetch of a fresh region takes ~8 hours

- **Filed:** 2026-05-16 (Phase 1 sub-A shipping checklist)
- **Severity:** medium (perf, not correctness)
- **Status:** deferred — **fix before adding Sweden as a region**
- **Affects:** `cfm.data.overture.load_region` cold path. Cache-hit path is unaffected (~1 s).

### Symptom

Calling `load_region("singapore")` against an empty cache against the pinned Overture release `2026-04-15.0` took **29,479.8 s (≈ 8.2 hours)** end to end on a normal home connection (2026-05-16 run). All five themes downloaded correctly; the manifest is valid; subsequent calls hit cache in ≈ 1 s.

### Root cause

`cfm.data.overture.loader._check_total_size` runs a `COUNT(*)` query against every theme via `S3DuckDBBackend.build_count_query` **before any read_theme call**. Each `COUNT(*)` scans the metadata of every parquet in the theme's S3 prefix (`s3://overturemaps-us-west-2/release/<release>/theme=<theme>/type=<type>/*`). For buildings/places/transportation that is hundreds of partitioned parquet files distributed globally. DuckDB has to open each one to read its row-group bbox stats before it can prove the file is outside Singapore. With httpfs latency this is the slow path.

The actual data reads (the `read_theme` calls after the COUNT phase) are the smaller portion of total time.

### Planned fix

Push the Singapore bbox into Overture's partition selection so that DuckDB only opens parquets that geographically cover Singapore, not the whole world. Overture's theme directories use coarse spatial partitioning (Hilbert-style); the right glob or a manual partition prune should reduce the metadata-scan workload by 1–2 orders of magnitude.

Concretely, three candidates worth trying in order:

1. **Skip or stub the COUNT pre-estimate.** Use a static heuristic per theme + bbox area for the `OversizedFetch` guard. Cheapest change; loses the precise size print but keeps the safety threshold.
2. **Glob the partition layer directly.** Replace `theme=<theme>/type=<type>/*` with a path that limits to relevant geographic partitions. Requires inspecting Overture's actual partition layout for the pinned release.
3. **Stream-and-write batches.** Skip materialising a `pyarrow.Table` per theme; stream `pq.write_table` from the DuckDB record-batch reader so we never hold a full theme in RAM. Orthogonal to the COUNT issue but worth doing while we're in there.

### Effort estimate

Half a day of work + verification (re-run a real cold fetch against Singapore and confirm wall-clock drops below an hour). Not a multi-day fix.

### Why we're not fixing it now

Phase 1 sub-A's contract is verified end-to-end. Phase 1 sub-projects B1–G read from the cache, never the cold path. The next time the cold path matters is when we add Sweden as a second region — at that point fixing this is a hard prerequisite, not optional.

### Tracking

- Source: `src/cfm/data/overture/loader.py::_check_total_size` and `src/cfm/data/overture/backend.py::S3DuckDBBackend.build_count_query`.
- Project memory: `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/project_overture_cold_fetch_slow.md`.
- Pinning policy reminder (`docs/data/overture_pinning_policy.md`) says re-pinning invalidates caches — re-pinning Singapore today would re-incur this 8-hour cost.

---

## sub-F spec §3.7 `subway`/`path`/`track` "emit-as-MINOR" examples are stale after the cycle-2 sub-E non-road-exclusion fix (doc-vs-behavior erratum)

**Status:** v2 erratum — documentation-vs-behavior drift, NOT a code bug. Logged 2026-06-01 during sub-G T11 cycle-3. Do **not** edit the spec mid-cascade.

**What:** Sub-F design spec `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md` §3.7 L292 (Class assignment semantics, Halt 7) names `subway`, `path`, `pedestrian`, `track` as examples of values that "also emit as MINOR" via sub-E's default-bucket fallthrough. That text predates the sub-G T11 **cycle-2** sub-E fix (commit `99f9e43`), which brought `_derive_tile_rows` into compliance with sub-E spec §5.1: **non-road crossings are excluded from the boundary-class vote**.

**The drift:** §3.7's "emits as MINOR" examples are only accurate for values whose sub-C `feature_class == road`. After cycle-2, any of these whose sub-C `feature_class != road` (e.g. a `subway` carried as rail/transit rather than highway) now correctly derives **NONE** (non-emitting) at the edge, contradicting the §3.7 example. The per-value classification has NOT been verified here — the drift is *conditional* on each value's sub-C feature_class.

**Why it is not a bug:** cycle-2 is correct (§5.1 mandates non-road exclusion); the §3.7 example list is documentation that was not updated when cycle-2 landed. The encoder still faithfully passes through whatever class sub-E derives (the actual §3.7 correctness criterion — "faithful passthrough of sub-E's class per edge" — is unchanged).

**Resolve at:** sub-E-v2 / sub-F-v2. Verify the sub-C `feature_class` of each of {`subway`, `path`, `pedestrian`, `track`}; update §3.7 L292's example list to name only values that remain road-classified (hence still emit MINOR).

**Surfaced by:** sub-G T11 cycle-3 (the `<unknown_highway>` validator fix, commit `c9f623c`); recorded as a non-blocking note so it does not resurface later as a phantom cycle.

---

## sub-F spec §1.4 / §8 non-emission rule enumerates "buildings/POIs" but should name non-road LineStrings too (doc-completeness erratum)

**Status:** v2 erratum — spec wording is incomplete, not wrong. Logged 2026-06-01 during sub-G T11 cycle-4. Do **not** edit the spec mid-cascade; the encoder gate already implements the correct principle.

**What:** §1.4 L59 states the operative principle — *"Token layer represents roads only for cross-cell references"* — but its examples and the §8 L803 non-road-non-emission check are framed around **buildings/POIs** (polygons/points clipped at the geometry layer). They do **not** explicitly name **non-road LineStrings** (e.g. `natural=coastline`, waterways), which are LineStrings (so not clipped like polygons) yet non-road (so must not emit brefs).

**Why it surfaced:** sub-G T11 cycle-4 found the sub-F encoder emitting `<bref>` for `natural` LineStrings clipped to active road edges (4,862 emissions / 224 tiles, 100% `natural`). The encoder's emission gate had no road-key check — it emitted for any LineString. The fix (commit below) gates emission on the feature's L1 key == `highway` via the shared `vocab.semantic_tag_to_l1_key` authority, which **implements §1.4's roads-only principle** for the LineString case the enumeration omitted.

**Resolve at:** sub-F-v2. Reword §1.4 / §8 so the non-emission rule reads "only highway-keyed features emit `<bref>`; all non-road features (buildings, POIs, **and non-road LineStrings**) emit zero `<bref>`" — matching the encoder gate and the validator's highway-only `_check_non_road_non_emission`.

**Surfaced by:** sub-G T11 cycle-4 (the encoder road-key emission gate).

---

## RESOLVED (sub-G close, 2026-06-01): POI alpha-drop rate is ~10× the other feature types — confirmed density-correlation artifact, ACCEPTED

**Status:** confirmed + accepted at sub-G close — advisory, never a blocker. Logged 2026-06-01 from the first real-data run of the sub-F alpha-drop warning-band diagnostic; confirmed the same day by a dropped-cell × POI-density cross-tab (sub-G T11 H3 follow-up).

**What:** The alpha-drop warning-band report (`reports/sub_f_task_3c_warning_band_singapore.yaml`, budget_raw 5760) on Singapore dropped 36/31,616 cells (0.114%; drop rule = cell token length > 5760). Per feature-type fraction dropped: road (fc=0) **0.76%**, building (fc=1) **1.51%**, base (fc=3) **0.13%** — but **POI (fc=2) 10.69%** (15,991 / 149,655).

**Verdict: density-correlation artifact, NOT a POI-specific over-drop or budget-accounting bug. Accepted.** A read-only cross-tab (reproduced the official 36-cell / 15,991-POI aggregate exactly = count lineage) shows:

- **POIs cost the 7-token floor** (a Point hits `token_cost.chunked_per_feature_tokens` `n<2` branch = `_STRUCTURAL_TOKENS(3) + n_anchor(4)`). No per-POI cost inflation; the drop rule is purely `total cell tokens > 5760`, type-agnostic.
- **Dropped cells are the densest cells, full stop:** POIs/cell median **430** (mean 444, max 975) vs retained median **0** (mean 4.2); total features/cell median **671** vs retained **0**. ~100× POI density, ~25× total-feature density.
- **The drop is not POI-targeted:** 9 of 36 dropped cells have POIs < 50% of features; the `(tile 24,8,*)` cluster is dropped on **building** density (~2–4 POIs, 350–393 buildings each). Building-dense cells with near-zero POIs are dropped by the same rule.
- **POI count alone does not drive drops:** retained cells exist with up to **593 POIs** (under budget); of the top-40 cells by POI count, only 20 are dropped — total density decides, not POI count.

So POIs are over-represented in the drop set only because POIs concentrate in the densest city-centre cells, which are exactly the cells that exceed the per-cell token budget. In the most POI-saturated dropped cells, POIs are 75–81% of the cell's tokens — hundreds of correctly-costed (7-token) POIs, a real density fact about those cells.

**Accepted because:** the per-cell budget (P99.9, Halt-4 lock 2026-05-29) deliberately rejects the extreme density tail (0.114% of cells); the loss is the densest urban cores, acceptable for v1 (the budget tradeoff was locked with this understood). No code change. If a future region needs the densest cores retained, raise the per-cell budget or split dense cells — not a POI-specific fix.

**Surfaced by:** sub-G T11 cascade-4 (first run of the alpha-drop diagnostic on real data). **Confirmed by:** sub-G T11 H3 follow-up cross-tab (2026-06-01).
