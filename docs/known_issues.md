# Known issues

A short, in-tree list of accepted-but-not-yet-fixed issues. Each entry says: where the issue is, why we accepted it, what blocks fixing, and when we have to fix it.

Add new entries on top. Remove entries when they're fixed.

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

## #6 — `admin_region = "Central Region"` placeholder

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** low (conditioning vector; sub-D uses it as a feature)
- **Status:** deferred — fix before sub-D needs disambiguated per-tile region conditioning
- **Affects:** `src/cfm/data/sub_c/pipeline.py` (conditioning computation)

### Context

Sub-C currently writes a fixed string `"Central Region"` for `tile.meta.conditioning_per_tile.admin_region`. Per spec §11.9 the correct value is the second-level division (subtype=region for Singapore) from the divisions theme covering the tile centroid. The value is informational-only at sub-C scope; every Singapore tile happens to fall inside Central Region for the initial extraction, so the placeholder is not wrong, just imprecise.

### Fix

Look up the divisions theme (already cached by sub-A) for the tile centroid point, filter to `subtype = "region"`, and return the matching `name` field.

### Tracking

- Source: `src/cfm/data/sub_c/pipeline.py` (see `# DECISION:` near conditioning computation)
- Spec: §11.9

---

## #5 — `water_fraction` placeholder (inland-water computation deferred)

- **Filed:** 2026-05-18 (Phase 1 sub-C closeout)
- **Severity:** low-medium (correctness on inland-river cells; doesn't break any invariant)
- **Status:** deferred — fix before sub-D macro-plan derivation needs accurate `water_fraction` distinction
- **Affects:** `src/cfm/data/sub_c/pipeline.py` (see `# DECISION:` in `_extract_tile`)

### Context

Sub-C sets `water_fraction = sea_water_fraction` for every cell. Spec §11.3 defines `water_fraction` as "all-water (sea + inland)" coverage. Computing inland-water contribution requires intersecting inland-water base features (river, stream, reservoir, etc.) with each cell box. Today's value is a safe under-estimate: it satisfies inline-validator invariant #5 (`sea ≤ wf ≤ 1`). Affected cells are coastal inland-river tiles such as Singapore River → Marina Bay.

### Fix

Load the base theme, filter to water subtypes (river, stream, reservoir, …), intersect each filtered geometry with the cell box, compute area coverage, and add to `sea_water_fraction`.

### Tracking

- Source: `src/cfm/data/sub_c/pipeline.py` (see `# DECISION:` in `_extract_tile`)
- Spec: §11.3

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
