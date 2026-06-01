# Known issues

A short, in-tree list of accepted-but-not-yet-fixed issues. Each entry says: where the issue is, why we accepted it, what blocks fixing, and when we have to fix it.

Add new entries on top. Remove entries when they're fixed.

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


## #4 — Tokenizer `emit_unknown_token` fall-through not yet implemented (training-critical)

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** high (training-critical path)
- **Status:** deferred — fix before Phase 2 training run; NOT a sub-D/E/F/G blocker
- **Affects:** `src/cfm/tokenizer/encode.py:59-60`

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
