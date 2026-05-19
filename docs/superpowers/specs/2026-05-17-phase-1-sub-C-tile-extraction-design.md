# Phase 1 sub-project C — Multi-cell tile extraction design

- **Date:** 2026-05-17
- **Phase:** 1, sub-project C (multi-cell tile extraction)
- **Status:** Draft, pending user review
- **Owner:** umar

## 1. Goal

Turn the cached Overture themes that sub-A produces (per region, bbox-filtered) into a per-tile structured dataset that the downstream Phase 1 sub-projects (sub-D macro plan, sub-E boundary contracts, sub-F stitcher, sub-G validator) can consume. Sub-C is the geometric extraction stage: it reprojects to a local metric frame, clips to the admin polygon, partitions into 2 km × 2 km tiles with 8 × 8 cell grids, splits multi-cell features at cell boundaries with crossing records, applies sea-masking and missing-value policies, and writes per-tile parquet + YAML artifacts under an integrity-checked region manifest.

The per-tile output IS sub-C's contract with sub-D/E/F/G. Getting it wrong cascades through every downstream sub-project, so the storage layout, schema, and determinism guarantees are all locked here and cannot drift without a `sub_c_schema_version` bump + re-extraction.

## 2. Scope (in / out)

**In scope for sub-C (Singapore only):**

- Library `cfm.data.sub_c` with the extraction pipeline and pure-function helpers for each stage.
- CLI script `scripts/extract_tiles.py` that drives extraction for a region from cached parquets.
- Cross-tile validator script `scripts/validate_extraction.py` that checks the digest-chain integrity and serves as the `_SUCCESS` gate.
- Per-tile output: `cells.parquet`, `features.parquet`, `crossings.parquet`, `meta.yaml`, `provenance.yaml`.
- Region-level output: `manifest.yaml`, `_SUCCESS`.
- Pre-commit lint rule forbidding `import pandas` in sub-C write-path modules.
- Test suite per Topic 6: ~60 named tests across three layers (unit, pipeline-stage on torture-test fixture + cross-tile-validator failure-mode fixture, cached-Singapore integration).
- A `docs/known_issues.md` entry recording the Sweden-densification revisit requirement and the tokenizer-enhancement training-critical-path dependency.

**Explicitly out of sub-C scope (cross-referenced; required by sub-C output to be useful, but not sub-C's deliverables):**

- **Tokenizer enhancement at `src/cfm/tokenizer/encode.py:59-60`** for `emit_unknown_token` fall-through. Today's encoder hard-raises `UnsupportedFeatureClass` on not-in-vocab classes; under Topic 3b Option A sub-C stores raw class values and downstream tokenization at training time must fall through to `<prefix>__UNK__` when the field's policy is `emit_unknown_token`. The change is small (~10 lines) but is its own brainstorm-spec-test cycle and a separate sub-project. **It is on the training critical path, NOT on the sub-D/E/F/G critical path** — sub-D consumes per-cell rows, not tokens; same for sub-E/F/G.
- **B2 follow-up: `_LOCKED_MISSING_POLICIES` extension** to the four-case `{missing_value, not_in_vocab}` schema. See §3.
- **Conditioning vocabulary sub-project** (analogue of B2's vocab work). Sub-C stores raw conditioning-vector strings (`"Asian-megacity"`, `"contemporary"`, etc.); the conditioning-vocab YAML + str→int8 mapping is a future sub-project per Topic 4e D2-A. The future sub-project inherits Topic 7c's α/β EPSILON principle and the append-only-within-phase discipline.
- **Sweden enrollment sub-project.** Sub-A's cold-fetch fix (`known_issues.md` #1) is the hard prerequisite. Sweden enrollment is where the CRS choice for Sweden is made (1a's polymorphic `tile.crs` is the structural enabler), where polygon densification is re-measured against Sweden's longer-edge coastlines (§7.4), and where extraction parallelism (`--pool-size N`) gets tuned for Sweden's tile count.
- **Sub-D macro plan derivation, sub-E boundary contracts, sub-F stitcher, sub-G end-to-end validator.** Consumers of sub-C's output.

## 3. PREREQUISITE: B2 follow-up

Sub-C reads `configs/data/missing_value_policy.yaml` and applies a four-case rule per field (Topic 3b). The four-case rule requires the policy YAML to carry both a `missing_value` axis (NULL handling, already in B2) and a `not_in_vocab` axis (present-but-not-in-Phase-1-vocab handling, NEW).

**Concrete prerequisite artifact (NOT sub-C scope; should have its own spec/plan):**

| Artifact | Change | Owner |
|---|---|---|
| `src/cfm/data/vocab_derivation.py::_LOCKED_MISSING_POLICIES` (line 326) | Extend dict shape from `{field: (type, rationale, is_provisional)}` to `{field: {missing_value: (type, rationale, is_provisional), not_in_vocab: (type, rationale, is_provisional)}}`. Populate `not_in_vocab` values per the Topic 3b four-case lock (see §10.2). | B2 follow-up |
| `configs/data/missing_value_policy.yaml` | Regenerate via `scripts/derive_phase1_vocab.py` so the YAML carries both axes. | B2 follow-up (script auto-generates) |
| `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md` §8, §10 | Update Missing-value policy and Policy YAML shape sections to reflect the four-case schema. | B2 follow-up |
| `tests/data/test_derive_phase1_vocab.py` | Update the four-case integration test for the new schema. | B2 follow-up |

Estimated half-day of work, B2-scoped. Sub-C implementation cannot start until this lands.

## 4. Design principles

Five principles drive sub-C's decisions. Two are inherited from B2 and now generalized; three are new in sub-C.

### 4.1 Cost-asymmetry — cheap to keep, impossible to recover (inherited from B2 §3.1, generalized)

Choices that move data in only one direction (irreversible) deserve extra weight on the keep-it-now side. The asymmetry generalized beyond vocab to every sub-C layer:

- **Raw class storage** (3b Option A): sub-C stores raw Overture class values; downstream tokenizer maps not-in-vocab to `__UNK__`. Phase 1.1 Sweden vocab expansion correctly tokenizes Singapore's originally-rare classes (e.g., "barn") WITHOUT re-extracting any tile.
- **Raw `edge_position_m`** (2b): un-quantized float; sub-E quantizes at its own granularity. Quantizing here would force re-extraction if sub-E ever wants finer resolution.
- **`subtype_raw` storage** (4b): stored as nullable string even though Phase 1 vocab doesn't tokenize subtype; future subtype-integration sub-project gets the data for free.
- **`sea_overlap_fraction` raw float** (2.5b): downstream consumers pick their own threshold.
- **`water_fraction` per cell** (2.5a): cheap to compute now, expensive to re-derive.

### 4.2 Schema-polymorphism is cheap; re-deriving stored data is not

When evaluating "we can change this later" arguments, separate schema-layer cost (cheap) from data-layer cost (expensive). The polymorphic `tile.crs` field per region is the canonical example: Singapore tiles carry `EPSG:3414`, Sweden tiles can carry any other EPSG, and the schema accommodates both without modification — but the **coordinate values stored under that field** are committed to the chosen CRS and cannot be reinterpreted under another CRS without re-extraction. See auto-memory `feedback_schema_vs_data_cost_asymmetry.md`.

### 4.3 Apply EPSILON at structural boundaries, not user thresholds (new in 7c)

Two categories of float comparison:

- **(α) Structural-boundary comparisons.** Boundary value carries semantic weight by definition (0, 1, equality between computed values). FP noise on the boundary side is what EPSILON exists to absorb. Apply per-quantity-type EPSILON from the central table (§14.4).
- **(β) User-threshold comparisons.** Threshold is a designer-chosen tuning knob (500 m river-length cutoff, 0.01 m² sliver area). Shifting by EPSILON just shifts the threshold; the threshold itself IS the policy. Use strict `<`, `>`, `>=`, `<=`.

See auto-memory `feedback_epsilon_structural_vs_user_threshold.md`.

### 4.4 Denormalize iff (a) every consumer benefits AND (b) the access pattern is established

The decision rule for whether a derivable field is stored as primary or derived-at-read-time. Both conditions must hold. Across Topics 2–4, ten denormalization candidates were considered; eight were rejected (`was_cut`, `feature_class on crossing record`, `over_sea`, `class_canonical`, `corner_pair_id`, per-cell water columns repeated on feature rows, region-uniform config blocks repeated per tile, conditioning_vector region-constant defaults repeated per tile). Two were accepted (`geometry_type` and `bbox_*` on features.parquet) — both with consistency tests in Topic 5's inline validator. Speculative-for-one-consumer denormalization is dead weight; predicate-pushdown wins that benefit every consumer with established access patterns are justified additions.

### 4.5 Per-tile granularity is the unit of determinism

Per Topic 4a + Topic 7 Category J: a single tile's directory `tile=EPSG3414_i<i>_j<j>/` is the atomic unit. Per-tile re-extraction does NOT perturb the bytes of any other tile. Topic 6's Layer 2 `test_torture_tile_reextract_byte_identical_modulo_excluded_fields` certifies this at fixture scale; Topic 7d's `test_extraction_pool_size_independence` certifies the parallelism-safety extension of the same property.

## 5. Cross-decision dependencies

Several pairs of decisions are co-dependent; the artifact must enforce the joint constraints.

- **(1a polymorphic `tile.crs`) ↔ (4a tile-dir naming).** `tile=EPSG<code>_i<i>_j<j>` includes the CRS in the path. Future regions with different CRSs land in unambiguously-named directories under the same region root.
- **(1b half-open grid) ↔ (2b co-linear feature attachment).** Boundary points `x = i*2000` belong to tile/cell `i`. A feature co-linear with that edge attaches entirely to cell `i` and generates no crossing record. Half-open is locked in 1b; co-linear-rule in 2b derives from it.
- **(2a split-at-boundaries) ↔ (2b crossing-record schema).** Sub-E's per-edge contract derives from per-crossing records that 2b stores. `source_feature_id` is primary on both sides of every cut; `was_cut` is derived (≥2 cell-rows with same `source_feature_id`).
- **(2.5a cell-mask rule) ↔ (2.5b feature-overlap policy).** Cells that survive 2.5a contain ≥1 non-sea feature; those features are tagged with `sea_overlap_fraction`. Option B (keep + tag) for 2.5b was justified by 2.5a's cell-mask discrimination already happening.
- **(2.5a pipeline order) ↔ (sliver-drop placement).** Sliver-drop runs before sea-mask: clip → reproject → partition → sliver-drop → sea-mask. Reverse order would let a 1 cm road sliver in a sea cell survive sea-mask, then orphan after sliver-drop.
- **(9.1 sea-polygon derivation) ↔ (10.2 base.class not-in-vocab drop_row).** Sea polygons (`base.class IN {ocean, strait, bay} OR subtype=ocean`) are below the Strict-300 floor and would be dropped by the policy step's not-in-vocab handler. `derive_sea_polygons` must run on RAW themes BEFORE `apply_missing_value_policy` so the derived view retains sea polygons while the policied themes correctly drop them from feature emission (sea polygons are masks, not features). Verified by inline test `test_derive_sea_polygons_runs_against_raw_base_not_policied_themes`.
- **(3b raw class storage) ↔ (tokenizer enhancement).** Sub-C produces output that today's tokenizer cannot encode end-to-end without the `emit_unknown_token` fall-through enhancement. **Not a sub-D/E/F/G blocker** (they consume per-cell rows, not tokens); a training-critical-path dependency only.
- **(4b geometry_type denormalization) ↔ (Topic 5 inline validator).** Denormalization-with-consistency-test is allowed under §4.4; inline validator asserts `decode_wkb_header(geometry).type_name == geometry_type` for every row.
- **(4b bbox denormalization) ↔ (Topic 5 inline validator).** Same pattern: bbox stored for predicate-pushdown, validator asserts bbox matches WKB.
- **(4d digest chain) ↔ (Topic 7e EXCLUDED_FROM_SHA).** Timestamps + free-form rerun fields excluded from sha computation so `provenance_sha256` is byte-deterministic across runs. Without exclusion, the digest chain drifts every run.
- **(4d per-tile re-extraction protocol) ↔ (manifest.tiles[] update).** Single-tile re-extraction MUST update `manifest.tiles[<this_tile>].provenance_sha256`. `initial_extraction` block stays frozen; `tiles[]` tracks current state.
- **(4e conditioning_vector storage split) ↔ (4d region-uniform constants in manifest).** Region-constant conditioning fields (`country`, `climate_zone`) live in `manifest.conditioning_defaults`; per-tile fields in `tile.meta.conditioning_per_tile`. Same pattern as 4d (A4) config split.
- **(Topic 7d densification once-per-region) ↔ (worker isolation).** Densified polygon computed in main process before workers start; serialized as WKB to workers; workers MUST NOT re-densify. (For Singapore the densification is a no-op per §7.4, but the function-signature placement is locked so Sweden doesn't re-open sub-C code.)

## 6. Pipeline overview

```
load_region("singapore")                                 # sub-A; cache-hit ~1 s
   ↓ Region(themes, fetch_bbox, geometry, ...)
sea_polygons_raw = derive_sea_polygons(themes["base"])   # §9.1; derived BEFORE policy
   ↓ sea_polygons_raw: MultiPolygon (4326); ~35 polygons for SG
                       (Sea polygons are masks, not features. Derived from raw
                       base before apply_missing_value_policy so the base.class
                       not-in-vocab drop doesn't remove ocean/strait/bay rows
                       before sea-mask can use them.)
apply_missing_value_policy(themes, policy_yaml)          # §10.1; raw-row level
   ↓ policied_themes
       (transportation NULL rows dropped;
        buildings NULL → B__UNK__; POI primary NULL → POI__UNK__;
        base not-in-vocab rows incl. ocean/strait/bay DROPPED from feature emission;
        sea_polygons_raw is unaffected — it's a separate derived view.)
densify_admin_polygon(geometry.admin_polygon, max_edge_length_m=None)  # §7.4; no-op for SG
   ↓ densified_polygon (== geometry.admin_polygon for SG)
reproject_to_local_metric(                               # §7.1
    policied_themes, sea_polygons_raw, densified_polygon,
    target_crs="EPSG:3414")
   ↓ policied_themes (SVY21), sea_polygons (SVY21), densified_polygon (SVY21)
clip_to_admin_polygon(policied_themes_svy21, densified_polygon_svy21)  # §7.3
   ↓ clipped_themes
partition_into_tiles(clipped_themes, densified_polygon_svy21, tile_size_m=2000)  # §7.2
   ↓ tile_inventory: dict[(tile_i, tile_j), TileInputs]
   FOR EACH tile, in dynamic-queue worker (§14.5):
      partition_into_cells(tile_inputs, cell_size_m=250)        # §8
         ↓ per-cell sub-features + crossing events
      apply_sliver_drop(per_cell_subfeatures, rule)             # §11.5
         ↓ per-cell sub-features (slivers removed)
      apply_sea_mask(per_cell_subfeatures, sea_polygons_svy21)  # §9; uses pre-derived sea
         ↓ kept cells + sea_water_fraction (per cell) + sea_overlap_fraction (per feature)
      compute_conditioning_per_tile(...)                         # §11.9
      write parquets + meta.yaml                                 # §11; canonicalize encoding (§14.3)
      run inline validator (§12)                                 # blocks on failure
      write provenance.yaml                                      # commit marker
write manifest.yaml                                              # §11; aggregate tiles[]; sea_polygons_sha256 recorded
run cross-tile validator (§12.2)                                 # _SUCCESS gate
write _SUCCESS                                                   # iff cross-tile validator passes
```

## 7. Coordinate frame, tile grid, clipping (Topic 1)

### 7.1 Coordinate frame (1a)

**Sub-C uses EPSG:3414 (SVY21) for Singapore.** Per-tile `tile.crs` is polymorphic by design — future regions choose their own CRS (Sweden picks in its enrollment sub-project; the tile-origin pattern in 7.2 is locked here as CRS-origin-anchored across all regions).

Rationale recap: SVY21 is Singapore Land Authority's national grid; integer EPSG code = byte-deterministic CRS identifier (no FP origin to drift); no UTM zone-seam logic; consumer GIS tools handle EPSG codes uniformly. Per `feedback_dont_optimize_multiregion_under_singleregion_scope.md`: don't pick UTM (multi-region-friendly but zone-seam-prone on Stockholm-on-33N/34N) for a Singapore-only scope.

### 7.2 Tile origin / grid alignment (1b)

**Tiles are aligned to the CRS origin on a global 2 km grid.** Tile `(i, j)` covers easting `[i*2000, (i+1)*2000)` and northing `[j*2000, (j+1)*2000)` in SVY21. Tile ID is `(crs, i, j)` with `i, j` integer.

- **Half-open interval explicit.** Feature at exactly `x = 2000.0` lands in tile `i = 1`, not `i = 0`.
- **Co-linear-feature tie-break.** A feature lying exactly on edge `x = i*2000` belongs entirely to cell on the higher-i side (cell `i`) by the half-open convention. Generates no crossing record (touch-but-not-cross; see §8.3).
- **Tile-ID stability under admin-polygon refinement.** Refining the admin polygon (Overture release update, more precise source) changes the SET of tiles that intersect Singapore but does NOT change tile IDs. Tile (12, 17) is always the same 2 km × 2 km patch.

For Singapore in SVY21: easting ~6 km–50 km, northing ~18 km–50 km → tile indices roughly `i ∈ [3, 25], j ∈ [9, 25]`, ~150–250 tiles after admin clipping + sea masking.

**Determinism property (precise framing for Topic 7's audit):** reprojection of fixed `(lon, lat)` → fixed `(x, y)` is deterministic (given the locked pyproj version); `floor(float / 2000)` is deterministic. Tile IDs are integer-valued and reproducible across runs.

### 7.3 Clipping pipeline ordering (1c)

**Reproject everything to SVY21 first, then clip in SVY21.** Both themes and admin polygon are reprojected before any spatial operation; the clip is then a metric-correct intersection in SVY21.

Rationale (refined from 1c precision items):
- **Single canonical metric pipeline.** Cell partition, boundary-crossing detection, sea-masking, distance checks all operate in SVY21. One CRS for all spatial ops; no coordinate-system gymnastics.
- **Clip-cut points are exactly metric-correct in the frame we train on.** A road clipped at the coastline is cut at the actual metric intersection in SVY21, not at a 4326-clipped point then reprojected (sub-meter difference but free with this ordering, vs. extra determinism complexity in the alternative).
- **The 4326-Cartesian distortion argument generalizes by latitude.** Singapore at 1.3°N: cos(lat) ≈ 0.9997, 4326-Cartesian distortion ~0.03%. Stockholm at 59°N: cos(lat) ≈ 0.515, ~50% stretch. "Mildly incorrect for Singapore" becomes "significantly incorrect for Sweden" — same precedent-locking as 7.2's tile-origin pattern.
- **Disambiguating Option B's two distinct concerns** (recorded here so spec readers don't blur them): (a) `intersects/contains` predicate accuracy — negligible for Singapore at 0.03%; (b) cut-geometry location — `GEOS-in-4326` cut points reprojected to SVY21 ≠ `GEOS-in-SVY21` cut points directly, regardless of latitude.

Sub-A's contract (handoffs.md §A→C) is honored: admin polygon clipping happens before tile partitioning.

### 7.4 Polygon densification (1c implementation note; F1 lock)

**No densification for Singapore.** Empirical measurement (cached 2026-04-15.0 Singapore divisions parquet, country=SG): max edge 775 m, 99% of edges < 500 m, median 57 m. At Singapore's latitude (1.3°N), the 4326-Cartesian distortion on the longest edge is ~23 cm — well below the 250 m cell quantization scale and below sub-C's coordinate EPSILON.

**Function signature locked even though it's a no-op for Singapore** (F1):

```python
def densify_polygon(
    polygon: BaseGeometry,
    max_edge_length_m: float | None,
) -> BaseGeometry:
    """If max_edge_length_m is None, return polygon unchanged.
    Otherwise insert vertices on every edge longer than max_edge_length_m
    so the densified polygon has no edge exceeding the threshold.
    """
```

Sub-C invokes with `max_edge_length_m=None` for Singapore. Sweden enrollment passes a real value (e.g., `100.0` based on Sweden's measured edge distribution) without re-opening sub-C code.

**`known_issues.md` entry recorded at sub-C ship time:** "Sweden's coastline edge-length distribution MUST be measured before reaching the same no-densification conclusion. At higher latitudes (cos(lat) effect) and with archipelago coastlines (longer edges), densification may become essential. Function signature in place; only the `max_edge_length_m` argument needs tuning."

## 8. Cell extraction (Topic 2)

### 8.1 Split-at-boundaries with crossing records (2a)

For each tile, partition clipped themes into the 8 × 8 cell grid (each cell 250 m × 250 m). Multi-cell features are **sliced at every cell boundary they cross.** Each cell receives the sub-geometry that lies within it. Each cut produces a crossing record (§8.2 schema) linking the two sides via `source_feature_id`.

**Hard constraint that drives this choice:** `src/cfm/tokenizer/encode.py::_require_in_bounds` (Phase 0) rejects any vertex outside `[0, cell_size_m]` within a cell. The tokenizer's "cell-local coordinates" contract is non-negotiable in sub-C. Primary-cell assignment (no splitting) would break it for every multi-cell road and polygon. Split-at-boundaries also matches PRD §5 verbatim ("clips features at cell boundaries, recording crossings").

**Per-cell-local-sub-feature row representation:**
- Each cut piece carries the source feature's `id` as `source_feature_id`.
- `was_cut` is **DERIVED**, not stored: a feature was cut iff its `source_feature_id` appears in ≥ 2 distinct cell rows. Consumers compute this via a join on `(source_feature_id)` at read time. Single source of truth (see §4.4).
- Polygon pieces (a building straddling a cell boundary becomes two polygon pieces) preserve `source_feature_id` on both pieces; the macro plan (sub-D) operates on cell-level zoning, not on individual building polygons.

### 8.2 Crossing-record schema (2b)

`crossings.parquet` is per-tile, edge-keyed. Sub-E reads it to derive PRD-§5 boundary contracts.

**8-column flat schema** (NO `corner_pair_id`; rejected per §4.4 — speculative for one consumer's unknown access pattern):

| column | type | meaning |
|---|---|---|
| `source_feature_id` | string | Overture feature ID; primary |
| `lower_cell_i` | int8 | i-coord of the cell on the lower-i/lower-j side of the edge |
| `lower_cell_j` | int8 | j-coord of same |
| `axis` | int8 (enum: 0=x, 1=y) | edge orientation; 0 = x-axis edge (between cells (i, j) and (i+1, j)); 1 = y-axis edge |
| `ring_index` | int16 | 0 for polygon exterior shell; ≥ 1 for interior rings |
| `event_type` | int8 (enum: 0=enter, 1=exit, 2=interval) | road point-crossing = enter or exit; polygon edge-interval = interval |
| `edge_position_m` | float64 | raw SVY21 meter along the edge; un-quantized (sub-E quantizes at its granularity) |
| `edge_extent_length_m` | float64 | 0 for point crossings (roads); > 0 for polygon edge-intervals |

**Canonical `edge_id` = `(lower_cell_i, lower_cell_j, axis)`.** Encoded as three flat columns (not a struct) for per-column predicate-pushdown. The upper cell on the other side is derived: `axis=0` → upper cell is `(lower_cell_i + 1, lower_cell_j)`; `axis=1` → `(lower_cell_i, lower_cell_j + 1)`. Not stored.

**`feature_class` is NOT on the crossing record.** Derive via `source_feature_id` lookup in `features.parquet`. Same precedent as `was_cut`.

**In-file sort key (locked):** `(lower_cell_i, lower_cell_j, axis, source_feature_id, ring_index, edge_position_m, event_type)`. `source_feature_id` is the byte-deterministic tie-break for the rare case of two source features crossing the same edge at the same `edge_position_m` and `event_type`.

### 8.3 Edge cases (2b + precision items)

Explicit handling so first readers don't have to derive these:

| edge case | rule | rows produced |
|---|---|---|
| Multi-cell road (linear, spans N cells) | Split at each cell boundary; one record per crossing. | N - 1 crossing records, all sharing `source_feature_id` |
| Polygon with interior ring crossing an edge | Each ring intersection generates events; `ring_index` distinguishes shell vs each hole. | Multiple records per `(source_feature_id, edge_id)`; different `ring_index` values |
| Corner-crossing (feature crosses exactly at a cell corner) | Two simultaneous edge records, one per axis (the x-edge and the y-edge that meet at the corner). | 2 records, one with `axis=0`, one with `axis=1`, same `source_feature_id` and (likely) same `edge_position_m` |
| Touch-but-not-cross (road ends exactly at boundary) | Crossing requires non-zero presence in BOTH adjacent cells. Touching the shared edge alone is not enough. | 0 crossing records; feature wholly in one cell |
| Co-linear-entirety (feature lies entirely on a cell boundary) | Per 7.2 half-open: feature attaches to the higher-i / higher-j cell. | 0 crossing records; feature wholly in that cell |
| Partial co-linearity (polygon shell along an edge while body spans both cells) | Distinct from co-linear-entirety. Record as `event_type=interval` with `edge_extent_length_m = co-linear length`. | 1 record per co-linear-shell segment |
| Multi-crossing same edge (zigzag road) | Multiple alternating `enter`/`exit` records on same `edge_id`, ordered by `edge_position_m`. | N records per edge; `event_type` alternates |

## 9. Sea masking (Topic 2.5)

### 9.1 Sea definition (α subtype-based; β documented fallback)

**Sea polygon set (α, locked for Singapore):**

```
sea_polygon ≡ base row where (class IN {ocean, strait, bay}) OR (subtype = ocean)
```

Empirically (cached 2026-04-15.0 Singapore base parquet): 35 polygons match (5 class=ocean + 22 class=strait + 8 class=bay). Subtype coverage is 100% on Singapore.

**Sea polygons are masks, NOT features.** They are derived from the **RAW base theme BEFORE `apply_missing_value_policy`** as a separate view (`derive_sea_polygons(themes["base"])` in §6's pipeline). Reasoning: Phase 1 vocab does not include ocean/strait/bay in BASE_*; the four-case rule for base (§10.2) has `not_in_vocab.type = drop_row`, which would drop sea-defining rows from the policied themes. If sea-masking sourced its polygons from the policied themes, there'd be no sea polygons left to mask with. The pre-policy derived view sidesteps this.

`sea_polygons` is a shared input to all per-tile workers (computed once in main process per §14.5; serialized as MultiPolygon union for cheaper per-cell intersection; sha256 recorded in `manifest.sea_polygons_sha256`). The policy step continues to drop sea-defining base rows from feature emission — correct, because sea polygons aren't features; they're masks.

**Inland water is NOT sea — it is data.** Reservoirs (MacRitchie, Bedok, Pierce — `class IN {reservoir, lake}`), rivers (`class IN {river}`), streams (`stream`), canals (`canal`), drains (`drain`), ponds (`pond`), swimming pools (`swimming_pool`), and the rest of base's water-typed classes survive the policy step (they're in vocab) and feed `features.parquet` for downstream model use.

Sub-C uses raw `base.class` and `base.subtype` for the sea-polygon derivation directly from the cached parquet. The Phase 1 vocab deferral of subtype tokenization (`known_issues.md` #2) is about token-emission, not about geometric operations.

**β (boundary-touching fallback) documented for future regions only:** if a future region (e.g., Sweden's archipelago) shows poor subtype/class coverage on sea polygons, fall back to computing boundary-touching-water area (sea polygons touch the admin polygon boundary; inland polygons don't). Not implemented in sub-C; documented as escape hatch.

### 9.2 Cell-level sea-masking rule (2.5a)

**Locked rule:**

```
sea_water_fraction(cell) = area(cell ∩ admin ∩ sea_polygons) / area(cell ∩ admin)

drop cell iff (sea_water_fraction >= 1.0 - EPS_RATIO)
        AND  (cell has zero non-sea features after sliver-drop)
```

Where:
- `EPS_RATIO = 1e-9` (Topic 7c α — see §14.4).
- Non-sea features ≡ any feature whose source row is NOT a sea polygon. Roads, buildings, POIs, AND inland-water base features (rivers, streams, reservoirs, canals, drains, ponds) all count as kept-worthy presence. Inland-water feature in a coastal-mouth cell (e.g., Singapore River draining into Marina Bay) keeps the cell.
- The denominator uses the admin-clipped cell area, not the raw cell area. Coastal cells where admin trims off the maritime portion get smaller denominators; a fully-sea admin-clipped fragment still qualifies as drop-candidate. (First readers shouldn't have to derive why thin coastal admin slivers behave correctly.)

**Pipeline order (locked under Topic 4):**
```
clip (admin polygon, in SVY21)
  → reproject themes to SVY21
  → partition into 2 km tiles, then 8 × 8 cells per tile
  → sliver-drop (§11.5)
  → sea-mask (this rule)
  → write
```

The order matters: sliver-drop runs first so a 1 cm road sliver crossing into a sea cell is removed before the sea-mask runs, leaving the cell genuinely feature-empty and droppable. Reverse order would let the sliver survive sea-mask (one feature present), then orphan after sliver-drop (kept sea cell with no features).

### 9.3 Feature-level sea-overlap (2.5b)

**Lock: keep all sea-intersecting features in kept cells; store `sea_overlap_fraction` (float, primary) on every cell-local sub-feature row.**

Rationale (recap): cells that survived 9.2's mask contain ≥ 1 non-sea feature. The features in those cells are the reason the cells were kept (bridges, piers, offshore POIs, port infrastructure are Singapore's distinctive spatial signature). Dropping them contradicts the cell-mask logic. Tagging adds optionality without losing data; raw float defers thresholding to downstream consumers (cost-asymmetry).

**Storage:**
- `sea_overlap_fraction` is per cell-local sub-feature row in `features.parquet`.
- For point features (POIs): values in `{0.0, 1.0}` using GEOS **`intersects`** semantics (NOT `contains`) — coastline POIs that sit on the sea-polygon boundary count as sea-adjacent.
- For linestring features: `length(feature ∩ sea_geometry_in_cell) / length(feature_in_cell)`.
- For polygon features: `area(feature ∩ sea_geometry_in_cell) / area(feature_in_cell)`.
- `over_sea` is DERIVED at read-time as `sea_overlap_fraction > EPS_RATIO` (same EPSILON as 9.2). NOT stored.

**Optimization (explicit, not implicit):**
```python
if cell.sea_water_fraction == 0.0:
    # no per-feature compute; default fast-path
    all features in cell get sea_overlap_fraction = 0.0
else:
    cell_local_sea = intersection(cell, sea_polygons)   # compute ONCE per cell, cached
    for feature in cell.features:
        sea_overlap_fraction = ratio(feature ∩ cell_local_sea, feature)
```

Per-feature reprojection or per-feature polygon-intersection-from-scratch would be wasteful. The cell-local sea geometry must be cache-deterministic (Topic 7 audit).

## 10. Missing-value + not-in-vocab policy (Topic 3)

### 10.1 Pipeline stage (3a)

**Apply policies at raw-row level, immediately after `load_region` returns** (and after `derive_sea_polygons` extracts the sea-mask view per §9.1), BEFORE reproject / clip / partition.

```python
def apply_missing_value_policy(
    themes: dict[str, pa.Table],
    policy_yaml_path: Path,
) -> dict[str, pa.Table]:
    """Returns a NEW themes dict; signature enforces non-mutation.
    sub-A's Region object is untouched.

    The base.class not-in-vocab drop step removes ocean/strait/bay
    rows from the returned policied_themes — correct, because sea polygons
    are masks (not features). Sea-masking sources its sea-polygon set from
    the pre-policy derive_sea_polygons view (see §6 pipeline + §9.1)."""
```

The functional signature IS the enforcement — non-mutation is not "by convention" (which is a bug waiting for a tired implementer, per the 3a precision lock).

**Closed-set handler-map policy:** `{policy_type → handler_callable}` registered in code. Unknown policy types raise `PolicyError`. Adding a new policy type (e.g., hypothetical `bucket_low_count`) requires both a new entry in `_LOCKED_MISSING_POLICIES` AND a handler registration. Closed is the right choice for Phase 1; open is over-engineered without a second policy type.

**Policy YAML provenance:** `configs/data/missing_value_policy.yaml` is AUTO-GENERATED by B2's `scripts/derive_phase1_vocab.py` from `_LOCKED_MISSING_POLICIES` at `src/cfm/data/vocab_derivation.py:326`. To change a policy, edit the dict and regenerate; do NOT hand-edit the YAML — B2's next regeneration clobbers manual edits. Same convention as the auto-generated reports.

Sub-C reads the YAML at runtime as the source of truth (so YAML provenance + commit-sha-pinning continue to flow into sub-C's per-tile manifests).

### 10.2 Four-case schema (3b — depends on B2 follow-up per §3)

For each of the 5 fields in `missing_value_policy.yaml`, both `missing_value` (NULL handling) and `not_in_vocab` (present-but-not-in-Phase-1-vocab handling) axes are defined:

| Field | `missing_value.type` | `not_in_vocab.type` | Sub-C behavior |
|---|---|---|---|
| `buildings.class` | `emit_unknown_token` | `emit_unknown_token` | NULL → set class to `B__UNK__`. Not-in-vocab → store raw value as `class_raw`; tokenizer enhancement (out-of-scope; see §3) maps to `B__UNK__` at encode time. |
| `transportation.class` | `drop_row` | `drop_row` | NULL OR not-in-vocab → drop row from themes BEFORE partition. Symmetric extension of NULL policy to not-in-vocab. |
| `base.class` | `n_a` (100% non-null on SG) | `drop_row` | NULL doesn't occur. Not-in-vocab → drop row (Strict-300 floor's explicit drop decision applies; ~4.69% of base rows below Strict get dropped). **Sea-defining rows (`class IN {ocean, strait, bay}`, ~35 SG rows below floor) are correctly dropped here** — they're not features. Sea-mask sources its polygons from the pre-policy `derive_sea_polygons` view per §6 + §9.1; the policy drop does not break sea-masking. |
| `places.categories.primary` | `emit_unknown_token` | `emit_unknown_token` | NULL → set primary to `POI__UNK__`. Not-in-vocab → store raw value; tokenizer maps at encode. |
| `places.categories.alternate` | `n_a` (list field; empty list = "no secondary categories") | `drop_element` | Sub-C stores full alternate list raw (`storage_policy=preserve_all` from B2). Tokenizer at encode time filters not-in-vocab elements per-element. Per-row drop never applies. |

**Phase 1.1 Sweden expansion benefit (lead rationale for 3b Option A):** Singapore's "barn" rows (below Moderate-100 floor today) correctly tokenize as `B_barn` after Phase 1.1 vocab promotion WITHOUT re-extracting any tile. Cost-asymmetry working in our favour on a near-term event.

### 10.3 Reserved-symbol audit result

The `__UNK__` double-underscore marker convention disambiguates Phase-1 placeholders from data-derived "unknown" categories (e.g., Overture's `transportation.class == "unknown"` with 6,066 rows in Singapore — `R_unknown` is a real data token, NOT a placeholder).

**Extended audit recorded in spec** (date 2026-05-17, command `uv run python -c "..."` against cached 2026-04-15.0 Singapore parquets):

| Field | Unique values audited | `__UNK__` substring found? |
|---|---:|---:|
| `buildings.class` | 62 | none |
| `transportation.class` | 21 | none |
| `base.class` | 23 | none |
| `base.subtype` | 11 | none |
| `places.categories.primary` | 1,235 | none |
| `places.categories.alternate` | 1,281 | none |

Full Overture class/category space (not just in-vocab subset B2 audited) is collision-free with `__UNK__`. Sub-C's Option A "ship raw" plan is safe.

## 11. Storage layout (Topic 4)

### 11.1 Directory structure (4a)

```
data/processed/sub_c/<release>/<region>/
├── _SUCCESS                                  # zero-byte sentinel; presence = full extraction complete
├── manifest.yaml                             # region-level: tile inventory, sub_c_schema_version, integrity root
├── tile=EPSG3414_i12_j17/                    # i/j named in path; signed coords parse cleanly
│   ├── cells.parquet                         # per-cell rows (≤64 per tile)
│   ├── features.parquet                      # per cell-local sub-feature rows
│   ├── crossings.parquet                     # per crossing event
│   ├── meta.yaml                             # tile-level aggregates + applied config
│   └── provenance.yaml                       # most-recent extraction record for THIS tile
├── tile=EPSG3414_i12_j18/
│   └── ...
└── ...
```

Tile-directory naming uses `tile=EPSG<code>_i<i>_j<j>` for self-documentation; named coordinates handle negative values cleanly (none for Singapore; relevant for future regions).

### 11.2 `features.parquet` schema (4b — per cell-local sub-feature row)

```
cell_i                    int8
cell_j                    int8
feature_class             int8     # enum: 0=road, 1=building, 2=poi, 3=base
source_feature_id         string
geometry                  binary   # WKB, little-endian (NDR); cell-local SVY21 coords
geometry_type             int8     # enum: 0=Point, 1=LineString, 2=Polygon (denormalized; consistency-tested per §12)
bbox_min_x                float64  # precomputed from WKB (denormalized; consistency-tested per §12)
bbox_min_y                float64
bbox_max_x                float64
bbox_max_y                float64
class_raw                 string?  # raw Overture class for road/building/base; null for poi
subtype_raw               string?  # raw Overture subtype for building/base (stored even though Phase 1 vocab doesn't tokenize; cost-asymmetry forward storage); null for road/poi
categories_primary        string?  # raw Overture primary for poi; null for others
categories_alternate      list<string>?  # full alternate list for poi (storage_policy=preserve_all); null for others
sea_overlap_fraction      float64
```

**Sort key:** `(cell_i, cell_j, feature_class, source_feature_id)`. `source_feature_id` is the ultimate tie-break for byte-determinism.

**Denormalization justifications (per §4.4 every-consumer-benefits AND established-access-pattern):**
- `geometry_type` (int8): predicate-pushdown for sub-D/G filters by geometry kind. Inline validator asserts `decode_wkb_header(geometry).type_name == geometry_type` per row.
- `bbox_*` (float64): spatial-predicate pushdown without WKB parse. Inline validator asserts bbox matches WKB-derived bbox per row.

**Subtype storage rationale (cost-asymmetry forward):** `subtype_raw` is stored as nullable string. Phase 1 vocab doesn't tokenize subtype (`known_issues.md` #2), but the future subtype-integration sub-project (whenever it ships) gets the data for free without re-extraction.

**WKB byte order: explicit little-endian (NDR).** Default has historically drifted across shapely versions; explicit pin kills the drift. Encoded via `shapely.wkb.dumps(geom, hex=False, byteorder=1)`.

### 11.3 `cells.parquet` schema (4b — per-cell row, ≤ 64 per tile)

```
cell_i                          int8
cell_j                          int8
water_fraction                  float64   # all-water (sea + inland) coverage of cell, in [0, 1]
sea_water_fraction              float64   # sea-only coverage (per 9.1 sea definition), in [0, 1]
cell_area_admin_clipped_m2      float64   # area of cell after admin polygon clip, in m²; > 0 for kept cells
kept_features_count             int32     # defensive count; inline validator asserts matches features.parquet row count for this cell
```

**Sort key:** `(cell_i, cell_j)`. Naturally unique.

Cells dropped by §9.2 do NOT appear in this file. Inline validator (§12.1) asserts no kept-cell row violates `(sea_water_fraction >= 1.0 - EPS_RATIO) AND (kept_features_count == 0)` — catches "drop rule wasn't applied" bugs.

### 11.4 `crossings.parquet` schema

Locked in §8.2. 8 columns. In-file sort key locked in §8.2.

### 11.5 `meta.yaml` schema (per-tile)

```yaml
schema_version: 1.0
tile_i: 12
tile_j: 17

aggregates:
  kept_cell_count: 47                    # of ≤64 possible cells
  sea_mask_drop_count: 17                # cells dropped by §9.2 in this tile
  mean_water_fraction: 0.23              # AREA-WEIGHTED across kept cells (per 4d A2):
                                         # Σ(water_fraction × cell_area_admin_clipped_m2) / Σ(cell_area_admin_clipped_m2)
  mean_sea_water_fraction: 0.04          # same area-weighting
  feature_count_by_class:
    road: 1834
    building: 6712
    poi: 412
    base: 89
  crossing_count: 2107

config:                                  # tile-specific applied config only (region-uniform constants live in manifest.config)
  sliver_drop_rule: "drop iff geometry has area < 0.01 m² OR length < 0.01 m"

conditioning_per_tile:                   # see §11.9
  admin_region: "Central Region"
  morphology_class: "Asian-megacity"     # per-tile from day one; same value for every SG tile initially
  era_class: "contemporary"
  coastal_inland_river: 1                # int8 enum: 0=inland, 1=coastal, 2=riverside, 3=coastal_riverside
  population_density_bucket: null
  population_density_bucket_owner: sub-D
```

**Sliver-drop rule (4d A3):** ship default with the rule as a stringified, per-tile-tunable field. Default scope is area/length thresholds only (per the 4d A3 brainstorm lock). Future tuning recorded in this field per tile. `known_issues.md` entry: "Sliver-dropping default chosen empirically; may be tunable per region or per Overture release."

### 11.6 `provenance.yaml` schema (per-tile)

```yaml
schema_version: 1.0
tile_i: 12                               # redundant with path; defensive
tile_j: 17
crs: "EPSG:3414"                         # redundant with manifest.region_crs; defensive

extraction:
  commit_sha: <40-char sha>              # commit at MOST-RECENT extraction of THIS tile
  extracted_utc: 2026-05-17T08:12:14Z    # EXCLUDED from sha computation per §14.6
  rerun_count: 0                         # 0 for initial; ≥1 for re-extractions; INCLUDED in sha (deterministic)
  rerun_reason: initial                  # free-form; INCLUDED in sha per F2 fix (audit-trail purpose: different reasons → different shas)

inputs:                                  # digests of artifacts read at extraction
  release: 2026-04-15.0
  admin_polygon_sha256: <sha>
  policy_yaml_sha256: <sha>
  vocab_yaml_sha256: <sha>

outputs:                                 # digests of artifacts written; per-tile reproducibility check
  cells_parquet_sha256: <sha>
  features_parquet_sha256: <sha>
  crossings_parquet_sha256: <sha>
  meta_yaml_sha256: <sha>
```

Re-running the same tile with the same inputs (digests match) on the same code commit MUST produce matching `outputs.*` digests — Topic 7's per-tile determinism contract recorded in-band.

### 11.7 `manifest.yaml` schema (region-level)

```yaml
schema_version: 1.0                      # version of manifest.yaml format itself
sub_c_schema_version: 1.0                # AUTHORITATIVE version covering ALL sub-C output shapes
release: 2026-04-15.0
region: singapore
region_crs: "EPSG:3414"
admin_polygon_source: "overture://divisions:country:SG"
admin_polygon_sha256: <sha>              # the polygon used for clipping
densified_admin_polygon_sha256: <sha>    # for SG, == admin_polygon_sha256 (no-op densification)
sea_polygons_sha256: <sha>               # pre-policy derived sea-mask view (§9.1); audit-pinned
policy_yaml_sha256: <sha>
vocab_yaml_sha256: <sha>

config:                                  # region-uniform constants (4d A4: do not repeat per-tile)
  tile_size_m: 2000
  cell_size_m: 250
  cell_grid: [8, 8]
  epsilon_ratio: 1.0e-9
  epsilon_coord_m: 1.0e-6
  epsilon_area_m2: 1.0e-6
  epsilon_length_m: 1.0e-6
  sea_definition: "base.class IN {ocean, strait, bay} OR base.subtype = ocean"
  sea_water_fraction_threshold: 1.0      # T_sea in §9.2; minus EPS_RATIO in compare
  coastal_inland_river_min_river_length_m: 500.0
  pipeline_order: [clip, reproject, partition, sliver_drop, sea_mask]

conditioning_defaults:                   # region-constant subset of conditioning vector (§11.9)
  country: SG
  climate_zone: tropical_rainforest

initial_extraction:
  commit_sha: <40-char sha>              # commit at first full extraction (FROZEN after first run)
  started_utc: 2026-05-17T08:00:00Z      # EXCLUDED from sha
  completed_utc: 2026-05-17T08:42:31Z    # EXCLUDED from sha
  tile_count: 187

tiles:                                   # canonical inventory; consumers iterate this (NOT filesystem glob)
                                         # SORTED by (tile_i, tile_j) for byte-determinism
  - {tile_i: 12, tile_j: 17, provenance_sha256: <sha>}
  - {tile_i: 12, tile_j: 18, provenance_sha256: <sha>}
  - ...

# Asymmetry note (state in spec text below the YAML):
# - initial_extraction.commit_sha is FROZEN after first full extraction.
# - tiles[*] tracks CURRENT state: single-tile re-extraction updates that tile's provenance_sha256 here.
# - Therefore tiles[<re-extracted-tile>].provenance_sha256 may correspond to a different commit_sha
#   than initial_extraction.commit_sha. This is by design — manifest does NOT pretend siblings share
#   the most-recent commit of an independently re-extracted tile.
```

**Integrity chain (4d A1):** `_SUCCESS → manifest.tiles[*].provenance_sha256 → provenance.outputs.*_sha256 → file bytes`. Single-tile re-extraction MUST update `manifest.tiles[<this_tile>].provenance_sha256` (the only-consistent-split asymmetry).

### 11.8 `_SUCCESS` semantics + write order

**Write order, locked (matters for partial-extraction recovery):**

```
1. Compute densified_admin_polygon once (main process; for SG this is a no-op pass-through).
2. For each tile to extract (workers; see §14.5):
     write tile_dir/cells.parquet
     write tile_dir/features.parquet
     write tile_dir/crossings.parquet
     write tile_dir/meta.yaml
     run inline validator (§12.1) — blocks on failure; failure leaves provenance.yaml absent
     write tile_dir/provenance.yaml          # written LAST in tile; presence = tile complete
3. Write/update manifest.yaml (main process; aggregate tiles[] sorted by (tile_i, tile_j)).
4. Run cross-tile validator (§12.2).
5. Iff cross-tile validator passes: write _SUCCESS.
```

**`_SUCCESS` semantics (4d):**
- Presence indicates: all tiles in `manifest.yaml` are present with valid `provenance.yaml`, AND cross-tile validator has passed.
- Single-tile re-extraction: updates that tile's `provenance.yaml` + recomputes its `outputs.*_sha256` + updates `manifest.tiles[<this_tile>].provenance_sha256` + reruns cross-tile validator + leaves `_SUCCESS` in place iff validator still passes.
- Full re-extraction: deletes `_SUCCESS` first, writes everything, writes `_SUCCESS` last.
- Consumer protocol: a region directory without `_SUCCESS` is in-flight or post-corruption; consumers should refuse to read.

**Partial-extraction recovery protocol:** list tile_dirs on disk; for each, check `provenance.yaml` is present and `outputs.*_sha256` match the actual file digests; tiles failing either check are re-extracted.

### 11.9 Conditioning vector (4e)

Seven fields (down from PRD §8's eight; `deterministic_seed` removed per 4e E1 — it's reproducibility metadata, not generation conditioning, and lives in training-time run config).

| Field | Type | Granularity | Owner | Manifest or tile |
|---|---|---|---|---|
| `country` | string (ISO α-2) | region-constant | sub-C | manifest.conditioning_defaults |
| `climate_zone` | string | region-constant | sub-C | manifest.conditioning_defaults |
| `morphology_class` | string | per-tile from day one (per 4e E2; cost-asymmetry forward) | sub-C | tile.meta.conditioning_per_tile |
| `era_class` | string | per-tile from day one | sub-C | tile.meta.conditioning_per_tile |
| `admin_region` | string | per-tile | sub-C | tile.meta.conditioning_per_tile |
| `coastal_inland_river` | int8 enum | per-tile | sub-C | tile.meta.conditioning_per_tile |
| `population_density_bucket` | int8 enum | per-tile | **sub-D** | tile.meta.conditioning_per_tile (null at sub-C with `_owner: sub-D`) |

**`coastal_inland_river` enum:** `0=inland, 1=coastal, 2=riverside, 3=coastal_riverside`.

**Derivation rule (sub-C-applied):**
- `coastal` iff `sea_water_fraction > 0` for any cell in tile.
- `riverside` iff `Σ length(features WHERE class IN {river, stream}) in tile >= 500 m` (β user-threshold; strict `>=`, no EPSILON per §4.3).
- `coastal_riverside` iff both.
- `inland` otherwise.

The 500 m threshold is pinned in `manifest.config.coastal_inland_river_min_river_length_m` for tunability without code change. Reasoning: a 2 km × 2 km tile, 500 m ≈ one cell-side of river presence — substantial. 250 m would be too noisy.

**`admin_region` granularity:** second-level admin division per Overture divisions theme (`subtype = region` for Singapore; equivalent second-level subtype for other regions; Sweden TBD by Sweden enrollment sub-project).

**Expanded theme dependency:** sub-C reads the divisions theme for `admin_region` population, beyond the admin_polygon clipping that sub-A's contract already requires. Divisions queries happen at per-tile level (find which `subtype=region` polygon contains the tile centroid).

**Enum-tightening deferral (D2-A):** sub-C stores raw strings for `morphology_class`, `era_class`, `climate_zone`. A future "conditioning vocabulary" sub-project locks the enum sets, generates `configs/conditioning/conditioning_vocab.yaml`, and provides the str→int8 mapping for training. **That future YAML inherits the same append-only-within-phase discipline as `vocab_phase1.yaml`** — enum reordering or deletion requires a phase transition, not a minor version bump.

## 12. Validator placement (Topic 5)

Three-way scope split: inline per-tile (during extraction) / cross-tile separate script (`_SUCCESS` gate) / sub-G (cross-sub-project, requires sub-D/E artifacts).

### 12.1 Inline per-tile validator (during extraction)

Runs after parquets + meta.yaml are written, BEFORE provenance.yaml is written. **Failure halts; treated as bug-in-sub-C, not data-issue** per `feedback_test_weakening_to_pass.md` — if a test fails because real data violates an assumed invariant, the assumption is what failed; STOP and escalate, don't adapt the test.

| # | Invariant | Topic of origin |
|---|---|---|
| 1 | Schema correctness: every expected column present with right type; sort key canonical for all parquets. | §8.2, §11.2, §11.3, §11.4 |
| 2 | `bbox_*` ↔ WKB consistency: `bbox_xxx` columns match WKB-derived bbox for every feature row, within `EPS_COORD_M`. | §11.2 (4b denormalization-with-test) |
| 3 | `geometry_type` ↔ WKB header consistency: int8 enum matches WKB header type for every row. | §11.2 (4b denormalization-with-test) |
| 4 | crossings ↔ features `source_feature_id` consistency: every `source_feature_id` in `crossings.parquet` appears on ≥ 2 distinct cells in `features.parquet`. | §8.2 (was_cut→had-crossing implicit) |
| 5 | Water fraction bounds (combined-pass with NaN check per 7b precision item): `0 - EPS_RATIO ≤ sea_water_fraction ≤ water_fraction ≤ 1 + EPS_RATIO` AND `not isnan(water_fraction)` AND `not isnan(sea_water_fraction)`. SINGLE COLUMN TRAVERSAL. | §9.2 + §14.3 NaN policy |
| 6 | Kept-cell rule consistency: for every cell in `cells.parquet`, `NOT (sea_water_fraction ≥ 1.0 - EPS_RATIO AND kept_features_count == 0)`. Catches "drop rule wasn't applied" bugs. | §9.2 |
| 7 | `kept_features_count` matches `features.parquet` row count per cell. | §11.3 |
| 8 | `meta.yaml.mean_water_fraction` matches area-weighted formula from `cells.parquet`: \|stored - computed\| < EPS_RATIO. | §11.5 (4d A2 area-weighting) |
| 9 | `cell_area_admin_clipped_m2 > EPS_AREA_M2` for every kept cell (α structural-boundary comparison per §4.3). | §11.3 + §14.4 |
| 10 | NaN-free on every numeric column in cells/features/crossings.parquet (folded into #5 for water fractions; standalone for `edge_position_m`, `edge_extent_length_m`, `bbox_*`, `cell_area_admin_clipped_m2`, `sea_overlap_fraction`). | §14.3 |

### 12.2 Cross-tile validator (`_SUCCESS` gate; separate script)

`scripts/validate_extraction.py --region <region>`. **Mandatory `_SUCCESS` gate, not optional.** Runs after manifest written; `_SUCCESS` is only written iff cross-tile validator passes.

| # | Invariant |
|---|---|
| 1 | `sub_c_schema_version` consistency across manifest + every tile's meta.yaml + every tile's provenance.yaml. |
| 2 | `manifest.tiles[]` inventory matches filesystem tree (no orphan tile dirs; no missing tile dirs). |
| 3 | `manifest.tiles[<i,j>].provenance_sha256` matches actual `provenance.yaml` digest on disk. |
| 4 | Per-tile `outputs.*_sha256` match actual file digests (full integrity-chain verification). |

**Per-tile re-extraction protocol:** MUST invoke cross-tile validator on completion, not just inline. Inline checks tile self-consistency; cross-tile checks manifest-update reflects the new digest (catches orphaned-manifest bugs).

**Runtime budget:** cross-tile validator MUST complete in < 60 seconds for region size up to 10K tiles. At Sweden-scale, parallelize per-tile digest checks via process pool (default `--pool-size 1`; pool size affects wall-clock, not pass/fail).

### 12.3 Sub-G deferral (cross-sub-project per PRD §5)

Sub-G validator handles invariants that require sub-D/E artifacts:
- macro plan ↔ underlying geometry consistency (needs sub-D)
- boundary contract ↔ cell tokens correspondence (needs sub-E + tokenizer)
- token sequences decodable to valid GeoJSON (needs tokenizer round-trip)
- per-tile failures quarantined for inspection (sub-G's workflow)

### 12.4 Structured diagnostic format

```python
class TileValidationError(Exception):
    tile: str               # e.g. "tile=EPSG3414_i12_j17"
    invariant: str          # named id, e.g. "bbox_matches_wkb"
    failed_row: dict        # identifying fields, e.g. {"source_feature_id": "0123abc...", "row_index": 341}
    detail: dict            # expected vs actual, e.g. {"stored_bbox": (...), "wkb_bbox": (...)}
```

Each named invariant has a structured failure payload. Topic 7e reuses this format for determinism-test failure reporting. The structured format itself MUST be byte-deterministic across runs (named test in §13).

## 13. Test strategy (Topic 6)

Three-layer pattern matching B2 (`docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md` §13), scaled to sub-C's larger surface. ~60 named tests total; fast suite under ~15 seconds.

### 13.1 Layer 1 — Unit tests (~40 tests)

Per pure function across all topics. Examples (not exhaustive):
- `test_reprojection_lonlat_to_svy21_byte_deterministic` [§7.1]
- `test_tile_id_derivation_half_open_boundary_at_exact_x_equals_2000` [§7.2]
- `test_co_linear_feature_attaches_to_higher_ij_cell` [§7.2 + §8.3]
- `test_split_at_boundaries_cut_points_byte_stable_across_runs` [§8.1 + §14]
- `test_corner_crossing_produces_two_records_one_per_axis` [§8.3]
- `test_polygon_interior_ring_crossing_produces_multiple_records_per_source_feature_id` [§8.3]
- `test_co_linear_entirety_produces_zero_records_feature_attaches_to_higher_ij` [§8.3]
- `test_touch_but_not_cross_produces_zero_records` [§8.3]
- `test_partial_co_linearity_emits_interval_event_with_extent` [§8.3]
- `test_multi_crossing_same_edge_alternating_enter_exit_sorted_by_position` [§8.3]
- `test_sea_definition_subtype_based_matches_singapore_35_polygons` [§9.1]
- `test_derive_sea_polygons_runs_against_raw_base_not_policied_themes` [§9.1; verifies pre-policy derivation order; if the order is reversed, sea polygons would be empty after the base.class not-in-vocab drop]
- `test_apply_missing_value_policy_drops_sea_defining_base_rows_from_features` [§9.1 + §10.2; sea polygons removed from feature emission while sea_polygons view retains them]
- `test_inland_water_cell_not_dropped_macritchie_like` [§9.2]
- `test_pure_sea_cell_with_zero_features_dropped` [§9.2]
- `test_coastal_cell_with_bridge_not_dropped` [§9.2]
- `test_sea_water_fraction_epsilon_boundary_at_exactly_1_minus_eps` [§9.2 + §14.4]
- `test_sea_overlap_fraction_intersects_predicate_includes_coastline_pois` [§9.3 + §14.4]
- `test_apply_missing_value_policy_returns_new_themes_dict_no_mutation` [§10.1]
- `test_apply_missing_value_policy_drops_transportation_null_class_rows` [§10.2]
- `test_apply_missing_value_policy_assigns_b_unk_to_null_buildings_class` [§10.2]
- `test_not_in_vocab_buildings_class_stored_as_class_raw` [§10.2]
- `test_not_in_vocab_transportation_class_dropped` [§10.2]
- `test_not_in_vocab_base_class_dropped_per_strict_decision` [§10.2]
- `test_not_in_vocab_alternate_categories_filtered_at_tokenize_time_not_sub_c` [§10.2]
- `test_river_length_threshold_strict_no_epsilon_at_499_99` [§4.3 β + §11.9]
- `test_sliver_drop_threshold_strict_no_epsilon` [§4.3 β + §11.5]
- `test_canonicalize_yaml_sorted_keys_byte_deterministic` [§14.3]
- `test_int8_enum_mapping_stable` [§14.3]
- `test_wkb_byte_order_explicit_little_endian` [§14.3]
- `test_sha_excludes_timestamps_and_sha_field_but_includes_rerun_reason` [§14.6]
- `test_densify_polygon_none_returns_unchanged` [§7.4]
- `test_densify_polygon_with_real_threshold_inserts_vertices` [§7.4 — exercises Sweden-ready signature]

### 13.2 Layer 2 — Pipeline-stage tests on hybrid fixtures (~16 tests)

**Torture-test tile fixture** (one comprehensive 4×4-cell synthetic tile generated by `tests/fixtures/sub_c/build_torture_tile.py` — declarative list of feature definitions, each tagged with the topic decision it exercises). Generated parquets are byte-deterministic.

Session-scoped pytest fixture: torture-test tile extracted ONCE per pytest session; tests assert on shared output. Corruption tests copy baseline to temp dir and corrupt there.

- `test_torture_tile_extraction_succeeds` (baseline)
- `test_torture_tile_inline_validator_passes_on_clean_output`
- `test_torture_tile_reextract_byte_identical_modulo_excluded_fields` [primary per-tile determinism; §14]
- Per-invariant diagnostic payload tests (per Topic 6 P2; 8 named tests for 12.1's invariants 2,3,6,5,4,7,8,1):
  - `test_bbox_matches_wkb_diagnostic_includes_row_and_both_bboxes`
  - `test_geometry_type_matches_wkb_diagnostic_includes_row_and_both_types`
  - `test_kept_cell_rule_diagnostic_includes_cell_and_water_fractions`
  - `test_water_fraction_bounds_diagnostic_includes_cell_and_offending_value`
  - `test_crossings_features_consistency_diagnostic_includes_source_feature_id`
  - `test_kept_features_count_diagnostic_includes_cell_counts`
  - `test_mean_water_fraction_diagnostic_includes_expected_vs_actual_formula`
  - `test_schema_correctness_diagnostic_includes_missing_column_name`
- `test_validator_diagnostic_payloads_byte_deterministic` [Topic 7 L3 — corrupt same row same way across two pytest sessions; assert TileValidationError payload bytes identical]
- `test_provenance_sha256_byte_deterministic_across_runs` [§14.6 E1]
- `test_extraction_pool_size_independence` [§14.5 D5; pool_size=1 vs pool_size=4]
- `test_extraction_pool_size_independence_more_workers_than_tiles` [§14.5 D5 P1; pool_size=1 vs pool_size=N>tile_count; catches empty-queue worker shutdown bugs]
- `test_pyarrow_version_2_6_parquet_format_correct` [§14.3 verify-at-impl]
- `test_pyproj_uses_formula_path_for_svy21` [§14.3 verify-at-impl]

**Cross-tile-validator failure-mode micro-fixture** (2 tiles, each 1 cell + 1 feature; for negative tests of cross-tile validator only):
- `test_cross_tile_validator_detects_orphan_tile_dir`
- `test_cross_tile_validator_detects_missing_tile_dir`
- `test_cross_tile_validator_detects_provenance_sha256_mismatch`
- `test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun`

### 13.3 Layer 3 — Cached-Singapore integration tests (~4 tests; cache-hit ~1 s)

Shape-only assertions (real Overture data; not pinned to specific category counts because Overture changes between releases — inherited wart from sub-B1/B2).

- `test_singapore_two_tile_extraction_shape` [one coastal tile near Marina Bay; one inland tile near central reservoir; pick specific (tile_i, tile_j) at implementation]
- `test_singapore_tile_reextract_byte_identical_modulo_excluded_fields` [real-data per-tile determinism]
- `test_singapore_two_tile_cross_tile_validator_pass` [end-to-end integrity chain]
- `test_singapore_manifest_sub_c_schema_version_consistency` [chain across manifest + tile yamls]

### 13.4 Test budget summary

| Layer | Tests | Wall-clock |
|---|---:|---:|
| Layer 1 (unit) | ~40 | ~1–2 s |
| Layer 2 (torture-tile + cross-tile-fixture) | ~16 | ~8 s (session-scoped extraction) |
| Layer 3 (cached Singapore) | ~4 | ~5 s |
| **Total** | **~60** | **~15 s** |

Slow-marked: none. The only network-touching path would be sub-A's cold fetch, already covered by sub-A's slow-marked tests.

## 14. Determinism contract (Topic 7)

**Goal:** sub-C output is byte-identical across repeated extractions on a given commit + lock-file + CI-canonical platform, modulo the documented excluded-fields set.

Categories below consolidate all locks from Topics 1–6 plus 7's new additions. Each rule cites its topic-of-origin so future contributors can trace.

### 14.1 Coordinate-pipeline determinism (Category A)

- **EPSG:3414 integer code** → no FP CRS origin drift. [§7.1; 1a]
- **GEOS-in-SVY21 cut-point determinism** → all geometric ops happen in SVY21. [§7.3; 1c]
- **Polygon densification before reprojection** → no-op for Singapore; locked function signature for Sweden. [§7.4; 1c + F1]
- **Shapely intersection of feature × cell-box → byte-stable cut points** → output vertex coords reproduce given fixed input. [§8.1; 2a]
- **Admin-clipped denominator computation** → `area(cell ∩ admin)` reproduces; depends on the (no-op) densified polygon. [§9.2; 2.5a]
- **Cell-local sea geometry cached once per cell** → cache build is deterministic across runs and across the per-feature iteration order. [§9.3; 2.5b]

### 14.2 Integer-arithmetic determinism (Category B)

- **`floor(easting / 2000)` for tile-ID** is deterministic given deterministic reprojection. [§7.2; 1b]
- **Half-open interval `[i·2000, (i+1)·2000)`** → boundary point at exactly `x = i·2000` lands in tile `i`; co-linear features attach to higher-ij side. [§7.2 + §8.3; 1b]

### 14.3 Encoding determinism (Category E, 7b)

**Parquet writer config** (pinned in `cfm.data.sub_c.io._PARQUET_WRITE_KWARGS`):

| Parameter | Value | Reasoning |
|---|---|---|
| `compression` | `"snappy"` | Deterministic, no level parameter. [§14.3; 7b] |
| `row_group_size` | `50_000` | Bounded; per-tile feature counts typically fit in one row-group. |
| `data_page_size` | `1_048_576` (1 MiB) | Pinned for explicitness, not known drift. |
| `write_batch_size` | `10_000` | Pinned for explicitness, not known drift. |
| `use_dictionary` | `True` (with explicit per-column override list) | Dictionary encoding policy can shift across pyarrow versions; explicit per-column pin is the deterministic default. |
| `write_statistics` | `True` | Enables row-group predicate-pushdown for sub-D/E (load-bearing). **Drift risk** per 7a; tagged `revisit_when: [sub-D, sub-E, sub-F, sub-G]` if drift occurs. |
| `use_compliant_nested_type` | `True` | Required for `categories_alternate` (list<string>) cross-tool compatibility. |
| `version` | `"2.6"` | Parquet format version 2.6 chosen for column statistics features sub-D will use for predicate-pushdown — not just "default has shifted." Verify-at-impl test in Layer 1. |

**Dictionary determinism rides on sort determinism.** pyarrow's dictionary contents depend on row order; sort key (per-column-table) determines that order; therefore dictionary-encoding bytes are deterministic iff sort keys are deterministic. The sort keys in §11.2/§11.3/§11.4/§8.2 close the loop.

**WKB byte order:** explicit little-endian (NDR) via `shapely.wkb.dumps(geom, hex=False, byteorder=1)`. [§11.2; 7b]

**YAML canonicalization:** reuse B1/B2's `cfm.data.vocab_derivation.canonicalize_yaml` (or refactor to shared `cfm.data.io.canonicalize_yaml`). Settings: `sort_keys=True`, `default_flow_style=False`, `allow_unicode=True`, `Dumper=yaml.SafeDumper`, no machine-generated comments. [§11.5/§11.6/§11.7; 4d + 7b]

**int8 enum mappings** centralized in `cfm.data.sub_c.enums`:
```python
GEOMETRY_TYPE = {0: "Point", 1: "LineString", 2: "Polygon"}
FEATURE_CLASS = {0: "road", 1: "building", 2: "poi", 3: "base"}
AXIS          = {0: "x", 1: "y"}
EVENT_TYPE    = {0: "enter", 1: "exit", 2: "interval"}
COASTAL_RIVER = {0: "inland", 1: "coastal", 2: "riverside", 3: "coastal_riverside"}
```
Adding values = append-only-within-phase; triggers `sub_c_schema_version` bump.

**NaN policy** (NEW in 7b): sub-C output contains no NaN values in any numeric column. Inline validator (§12.1 #5, #10) asserts NaN-free per column, folded into bounds-check pass for water-fractions (single column traversal: `0 ≤ x ≤ 1 AND not isnan(x)` checked together — critical at Sweden-scale 10K tiles for avoiding double-scan).

**`open string columns`** (class_raw, subtype_raw, categories_primary, categories_alternate, admin_region, morphology_class, era_class): stay as strings, not enums — Overture-driven unbounded domain.

### 14.4 Float-comparison policy + EPSILON (Category C, 7c)

**Principle (locked 7c):** Apply EPSILON when comparing against structural boundaries (0, 1, computed-value equality). Do NOT apply EPSILON when comparing against user-chosen thresholds; the threshold itself is the policy. Hedging ("bool-shaped definitional check") is a symptom of unclear classification — pick α or β.

**Per-quantity-type EPSILON table** (centralized in `cfm.data.sub_c.epsilon`):

| Constant | Value | Quantity | Topic of origin |
|---|---|---|---|
| `EPS_RATIO` | `1e-9` | [0, 1] ratios — `sea_water_fraction`, `water_fraction`, `sea_overlap_fraction` | §9.2 (2.5a), §9.3 (2.5b), §12.1 (Topic 5) |
| `EPS_COORD_M` | `1e-6` | coordinate equality in SVY21 m — `bbox_match` validator, cross-run coord comparisons | §12.1 #2 (Topic 5 P4), §14 |
| `EPS_AREA_M2` | `1e-6` | area equality in m² — area-weighted-mean validator, `cell_area_admin_clipped_m2 > EPS_AREA_M2` | §12.1 #8, #9 |
| `EPS_LENGTH_M` | `1e-6` | length equality in m (NOT for the 500 m river threshold) | §14 |

**α examples (apply EPSILON):**
- `sea_water_fraction >= 1.0 - EPS_RATIO` (cell drop rule) [§9.2]
- `sea_overlap_fraction > EPS_RATIO` (over_sea derivation at consumer-read-time) [§9.3]
- `cell_area_admin_clipped_m2 > EPS_AREA_M2` (0 is structural boundary: "no admin coverage at all"; was misclassified during brainstorm and fixed) [§11.3, §12.1 #9]
- Validator `0 - EPS_RATIO ≤ sea_water_fraction ≤ water_fraction ≤ 1 + EPS_RATIO` [§12.1 #5]
- Validator `\|bbox_min_x - wkb.min_x\| < EPS_COORD_M` [§12.1 #2]

**β examples (strict comparison; no EPSILON):**
- `river_stream_length_in_tile >= 500.0` (coastal_inland_river) [§11.9; 4e]
- `feature_area_m2 < 0.01` (sliver-drop) [§11.5; 4d A3]
- `feature_length_m < 0.01` (sliver-drop) [§11.5; 4d A3]

**Future inheritance:** the future conditioning-vocab sub-project's derivation logic inherits α/β principle + per-quantity-type EPSILON table. New thresholds added there classify α/β explicitly; new quantity-types extend the central table, not bypass.

### 14.5 Parallelization safety (Category J, 7d)

**Invariant naming:** *"Pool size affects wall-clock; byte output is invariant under `pool_size ∈ [1, N]` for any N."*

**Parallelism model:** process pool (`multiprocessing.Pool`) with dynamic tile-queue. CLI flag `--pool-size N`; default `1` (sequential). Sweden tunes without re-opening sub-C.

**Shared inputs computed once in main process:**
- **Densified admin polygon** (per Category J L2): computed once at extraction start; for Singapore this is the no-op `densify_polygon(polygon, None)` returning polygon unchanged. Sha256 recorded in `manifest.densified_admin_polygon_sha256`. Serialized as WKB bytes to workers; workers MUST NOT re-densify (parallel workers could otherwise produce different densified polygons under floating-point vertex-insertion-order artifacts).
- **Sea polygons (derived view)** (per §9.1): `sea_polygons_raw = derive_sea_polygons(raw_themes["base"])`, reprojected to SVY21, computed once in main process. Singapore: ~35 polygons unioned to a single MultiPolygon for cheaper per-cell intersection. Serialized as WKB to workers. Sha256 recorded in `manifest.sea_polygons_sha256`. Workers MUST NOT re-derive (the raw base theme may have been mutated by lazy-loading order otherwise).
- **Cached Overture parquets**: workers open lazily per-tile via pyarrow filter pushdown; read-only access, no cross-worker conflict.

**Sequential main-process operations** (NOT parallelized):
- Densification (once per region, before workers start).
- `manifest.yaml` assembly (after all workers complete; reads each tile's provenance.yaml, sorts `tiles[]` by `(tile_i, tile_j)`, writes manifest).
- Cross-tile validator (after manifest; runs all digest checks; can internally parallelize its checks via same pool model — pool size affects wall-clock, not pass/fail).
- `_SUCCESS` write (last, after cross-tile validator passes).

### 14.6 EXCLUDED_FROM_SHA + determinism test surface (7e)

**Extension of B2's sha-exclusion pattern.** sha256 is computed over canonicalized YAML content with the sha field itself excluded (inherited from B2) PLUS timestamp fields excluded (NEW in sub-C). `rerun_reason` is NOT excluded (per F2 fix — different rerun_reasons should produce different shas for audit-trail purposes; tests pass a canonical fixed value).

```python
# cfm.data.sub_c.determinism
#
# Wildcard semantics (pinned, NOT glob):
# - file key "*" matches any YAML file (applies to all files).
# - field-path "*_sha256" matches any field whose final dotted-path
#   segment ends with the suffix "_sha256" (string endswith match,
#   applied to the LAST segment of the dotted path). Examples:
#     "vocab_sha256"                  → matches
#     "tiles[3].provenance_sha256"    → matches (final segment "provenance_sha256")
#     "outputs.cells_parquet_sha256"  → matches
#     "sha256_input"                  → does NOT match (suffix only)
# - All other entries are exact dotted-path matches (no wildcards).
EXCLUDED_FROM_SHA = {
    "*": ["*_sha256"],                                   # inherited from B2; suffix-match on final segment
    "manifest.yaml": [
        "initial_extraction.started_utc",                # exact dotted-path match
        "initial_extraction.completed_utc",
    ],
    "provenance.yaml": [
        "extraction.extracted_utc",                      # exact dotted-path match
    ],
    # rerun_reason: NOT excluded (per F2: audit-trail purpose; tests use canonical fixed value)
}
EXCLUDED_FROM_TEST_COMPARE = EXCLUDED_FROM_SHA  # one source of truth
```

**Result:** `provenance_sha256`, `manifest.tiles[*].provenance_sha256`, and any computed `manifest_sha256` are byte-deterministic across runs. The digest chain `_SUCCESS → manifest.tiles[*].provenance_sha256 → provenance.outputs.*_sha256 → file bytes` works without parallel sha calculations.

**Update protocol:** future contributor adding a new timestamp field MUST update `EXCLUDED_FROM_SHA` in the same PR. CI assertion parses YAML files for fields ending in `_utc`, `_at`, `_timestamp`; fails if any aren't in the exclusion set.

**Determinism test surface** (subset of §13; specifically determinism tests):
- Layer 1: ~13 named tests on per-function determinism (listed in §13.1).
- Layer 2: 6 named determinism tests (`test_torture_tile_reextract_byte_identical_modulo_excluded_fields`, `test_extraction_pool_size_independence`, `test_extraction_pool_size_independence_more_workers_than_tiles`, `test_provenance_sha256_byte_deterministic_across_runs`, `test_manifest_sha256_byte_deterministic_across_runs`, `test_validator_diagnostic_payloads_byte_deterministic`).
- Layer 3: 2 named determinism tests (cached-Singapore tile reextract; two-tile cross-tile validator).

### 14.7 Library pinning (Category I, 7a)

**Strategy:** range pins in `pyproject.toml` + `uv.lock` as canonical (matches Phase 0 / sub-A / B1 / B2 project pattern).

**CI authority:**
- CI MUST invoke `uv sync --frozen`. Local development MAY use `uv sync`. Determinism tests assert against frozen lock-resolved versions.
- Lock-file updates: explicit commits with their own PR; not piggy-backed on feature commits.
- **CI-canonical platform:** Linux x86_64, CPython 3.11. Different shapely wheels (Linux/macOS/Windows × CPython versions) may vendor different GEOS builds. Local development on macOS dev workstations is best-effort; CI is authoritative.

**Pinned libraries (concrete via uv.lock at sub-C ship):**
- `shapely` — GEOS-vendored; CI-canonical wheel.
- `pyproj` — PROJ-vendored; **formula-based for SVY21** (verify-at-impl in Layer 1; EPSG:3414 is a Transverse Mercator projection with parameters in the EPSG database, no datum-grid file dependency for sub-C's lon/lat → SVY21 transformation).
- `pyarrow` — parquet writer behavior per §14.3.
- `PyYAML` — canonicalize_yaml settings per §14.3.
- `pandas` — **FORBIDDEN in write path (lint-enforced).** Pre-commit lint rule: grep-fail on `import pandas` (or `from pandas`) in sub-C write-path modules. Test fixtures, analysis scripts, exploratory notebooks retain pandas access. Spec text without lint = convention; lint rule = enforcement (matches 3a's don't-mutate-by-signature-not-by-comment precedent).

### 14.8 Drift exception protocol (7a)

When a `uv lock` update causes a determinism test failure:

1. **Investigate:** which library / version bump; which artifact's bytes differ; which bytes specifically (parquet metadata? geometry coords? YAML field order?).
2. **Classify:**
   - **Load-bearing:** any downstream consumer's logic (sub-D/E/F/G/training) would read different values, OR the diff changes a sha256 in the digest chain.
   - **Non-load-bearing:** bytes differ but every downstream consumer reads the same logical content (e.g., parquet page-statistics fields that are summary cache, not data).
3. **Decide:**
   - Load-bearing → fix: (a) tighten the pin to exclude the drifted version, OR (b) modify sub-C code to not depend on the drifted behavior. Update spec + add regression test.
   - Non-load-bearing → exempt:
     (a) Document in spec §Determinism exceptions with: library, version range, affected artifact, bytes that differ, reasoning that diff is non-load-bearing.
     (b) Update the affected determinism test to xfail with a comment pointing to the documented exception.
     (c) Update the test-strip protocol if the exempted field needs stripping in compare logic.
     (d) **Record `revisit_when: [sub-D, sub-E, sub-F, sub-G, training]`** — list of downstream sub-projects whose specs must re-audit this exception when written.

**Cross-cutting project rule (new):** every future sub-project spec MUST include a "§Determinism-exception re-audit" section that enumerates entries tagged `revisit_when: <this sub-project>` and re-classifies each against current consumer behavior. Makes the temporal-prediction problem tractable instead of silent.

### 14.9 Schema versioning (Category G)

- `sub_c_schema_version` (region-wide; covers all parquet + yaml shapes uniformly): in `manifest.yaml` at region root.
- Per-file `schema_version` (narrow scope; versions the YAML format itself, not the data shape): in each `manifest.yaml`, `provenance.yaml`, `meta.yaml`.
- **`generated_at_commit` lags HEAD by one commit** — inherited B1/B2 caveat; not a bug. The script captures `git rev-parse HEAD` at run time; committing the artifact makes a new commit. Documented behaviour, not fixed.

## 15. Public API

### 15.1 Library (`cfm.data.sub_c`)

```python
from cfm.data.sub_c import (
    # Pipeline stages (pure where possible)
    derive_sea_polygons,                       # §9.1; pre-policy derived view
    apply_missing_value_policy,
    densify_polygon,
    reproject_to_local_metric,
    clip_to_admin_polygon,
    partition_into_tiles,
    partition_into_cells,
    apply_sliver_drop,
    apply_sea_mask,
    compute_sea_overlap_fraction,
    compute_conditioning_per_tile,
    # I/O
    write_tile_artifacts,
    write_manifest,
    write_success_marker,
    # Validators
    validate_tile_inline,
    validate_extraction_cross_tile,
    # Dataclasses
    Tile, Cell, FeatureRow, CrossingRecord, TileMeta, TileProvenance, RegionManifest,
    # Errors
    PolicyError, TileValidationError,
    # Constants
    EPS_RATIO, EPS_COORD_M, EPS_AREA_M2, EPS_LENGTH_M,
    EXCLUDED_FROM_SHA,
)
```

### 15.2 CLI scripts

```
scripts/extract_tiles.py
  --region <region>           e.g. singapore
  --release <release>         e.g. 2026-04-15.0 (defaults to sub-A pinned release)
  --output-dir <path>         defaults to data/processed/sub_c/<release>/<region>/
  --pool-size N               default 1; sequential
  --rerun <tile_i,tile_j>     re-extract a single tile; updates manifest in place
  --rerun-reason <str>        free-form audit string; included in sha
```

```
scripts/validate_extraction.py
  --region <region>           e.g. singapore
  --release <release>         e.g. 2026-04-15.0
  --pool-size N               default 1; digest-check parallelism
  # Exits non-zero on validation failure; structured TileValidationError JSON on stderr.
```

## 16. Module and file layout

```
src/cfm/data/sub_c/
├── __init__.py                              # public exports per §15.1
├── pipeline.py                              # orchestrator (apply_missing_value_policy → densify → reproject → ...)
├── coords.py                                # reprojection, tile/cell partitioning (§7)
├── geom.py                                  # split-at-boundaries, crossing-record derivation (§8)
├── sea_mask.py                              # sea definition, cell-mask rule, feature-overlap (§9)
├── policy.py                                # apply_missing_value_policy, handler-map (§10)
├── conditioning.py                          # compute_conditioning_per_tile (§11.9)
├── io.py                                    # parquet/yaml write helpers; _PARQUET_WRITE_KWARGS; canonicalize_yaml (re-exported)
├── enums.py                                 # int8 enum mappings (§14.3)
├── epsilon.py                               # EPS_* constants (§14.4)
├── determinism.py                           # EXCLUDED_FROM_SHA, sha helpers (§14.6)
├── manifest.py                              # RegionManifest dataclass, write_manifest, _SUCCESS protocol
├── validator_inline.py                      # per-tile inline validator (§12.1)
├── validator_cross_tile.py                  # cross-tile validator script logic (§12.2)
└── errors.py                                # PolicyError, TileValidationError

scripts/
├── extract_tiles.py                         # CLI per §15.2
└── validate_extraction.py                   # CLI per §15.2

tests/data/sub_c/
├── test_coords.py                           # Layer 1 — §7 unit tests
├── test_geom.py                             # Layer 1 — §8 unit tests
├── test_sea_mask.py                         # Layer 1 — §9 unit tests
├── test_policy.py                           # Layer 1 — §10 unit tests
├── test_conditioning.py                     # Layer 1 — §11.9 unit tests
├── test_io.py                               # Layer 1 — encoding determinism (§14.3)
├── test_torture_tile.py                     # Layer 2 — pipeline-stage on torture-test fixture
├── test_cross_tile_validator.py             # Layer 2 — failure-mode micro-fixture (§13.2)
└── test_singapore_integration.py            # Layer 3 — cached Singapore (§13.3)

tests/fixtures/sub_c/
├── build_torture_tile.py                    # declarative fixture builder (§13.2 P4)
└── build_cross_tile_fixture.py              # 2-tile micro-fixture for cross-tile validator failure tests

configs/data/
└── missing_value_policy.yaml                # AUTO-GENERATED via B2 follow-up (§3); sub-C reads at runtime

.pre-commit-hooks/
└── no-pandas-in-write-path.sh               # lint rule per §14.7

docs/
└── known_issues.md                          # +2 entries (Sweden densification revisit; tokenizer enhancement training-path dependency)
```

`src/cfm/data/__init__.py` adds the new `sub_c` exports per §15.1.

## 17. Errors

- **`PolicyError`** (in `cfm.data.sub_c.errors`): raised by `apply_missing_value_policy` for unknown policy types in the closed-set handler map.
- **`TileValidationError`** (in `cfm.data.sub_c.errors`): raised by inline + cross-tile validators with structured payload per §12.4.
- **`ValueError`** from pure-function library calls when input dataclass invariants are violated (e.g., negative tile coords, geometry with NaN coordinates pre-extraction).
- **`RuntimeError`** from CLI script if `git rev-parse HEAD` fails — abort rather than write a manifest with a bogus commit sha.

## 18. Done criteria

Sub-C is done when:

- B2 follow-up per §3 has landed (PREREQUISITE).
- `uv run pytest -q` passes: ~60 new tests + previous fast-suite (187 from B2 merge), wall-clock ~30 s total fast suite.
- `uv run python scripts/extract_tiles.py --region singapore` produces `data/processed/sub_c/2026-04-15.0/singapore/` with manifest + per-tile dirs + `_SUCCESS` (verified by `test_singapore_two_tile_extraction_shape`).
- `uv run python scripts/validate_extraction.py --region singapore` exits 0 (verified by `test_singapore_two_tile_cross_tile_validator_pass`).
- Byte-identical re-extraction across runs verified by `test_singapore_tile_reextract_byte_identical_modulo_excluded_fields` (real-data Layer 3) and `test_torture_tile_reextract_byte_identical_modulo_excluded_fields` (fixture Layer 2).
- Pool-size independence verified by `test_extraction_pool_size_independence` AND `test_extraction_pool_size_independence_more_workers_than_tiles`.
- Per-tile re-extraction protocol verified by `test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun` (Layer 2 negative test ensures the protocol is enforced).
- Digest-chain integrity verified by `test_provenance_sha256_byte_deterministic_across_runs`, `test_manifest_sha256_byte_deterministic_across_runs`, and the cross-tile-validator invariants 1-4 (§12.2).
- Pre-commit hook blocks `import pandas` additions to write-path modules (verified by the pre-commit lint rule itself + a test that runs the hook against a synthetic offending diff).
- `docs/known_issues.md` has the 2 new entries (Sweden densification revisit; tokenizer enhancement training-path dependency).
- User has reviewed the artifacts + spec + approved.

## 19. Risks specific to sub-C

- **B2 follow-up sequencing.** Sub-C cannot start until B2's `_LOCKED_MISSING_POLICIES` extension lands. If B2 follow-up slips, sub-C blocks. Mitigation: it's half-day of work; do it first.
- **Inline validator runtime overhead.** Not benchmarked. If > ~10% of extraction wall-clock at Sweden-scale, may need a `--skip-inline-validator` flag (dangerous; should be CI-gated). Mitigation: measure at implementation; per-row checks on ≤ 10K rows per tile should be sub-second.
- **pyarrow `version="2.6"` write correctness.** Verify-at-impl in Layer 1; failure means pinning a different parquet format version and re-recording.
- **pyproj formula-vs-grid path for SVY21.** Verify-at-impl in Layer 1; if pyproj uses a grid file, pin the grid via PROJ_DATA env var or document the exception per 14.8.
- **Singapore admin polygon edge distribution (§7.4 measurement) is from one Overture release.** Future Overture release may change polygon densification needs. Mitigation: the densification function signature is locked; only the `max_edge_length_m` argument changes.
- **Cross-tile validator runtime at Sweden-scale.** Spec budget 60s at 10K tiles; if exceeded, pool-parallelization tuning needed. Mitigation: parallelism via the same process-pool pattern as extraction.
- **Determinism drift across uv.lock updates.** Per the drift exception protocol; expected to occur occasionally; the protocol is the mitigation.
- **Tokenizer enhancement dependency.** Sub-C output is unusable for training without the enhancement. Mitigation: clear documentation in `known_issues.md`; framing in §3 makes this visible.

## 20. Out of scope — deferrals + cross-references

| Item | Reason | Picked up by |
|---|---|---|
| Tokenizer `emit_unknown_token` fall-through enhancement | Training-critical-path dependency; not sub-C's scope | Separate sub-project (brainstorm-spec-test cycle); `known_issues.md` entry at sub-C ship |
| B2 `_LOCKED_MISSING_POLICIES` four-case extension | Configuration prerequisite per §3 | B2 follow-up (half-day; should get its own spec/plan) |
| Conditioning vocabulary YAML + enum tightening | Stores raw strings now; future "conditioning vocab" sub-project locks enums | Future sub-project (D2-A deferral); inherits §4.3 α/β EPSILON and append-only-within-phase disciplines |
| Sub-D macro plan derivation | Consumes sub-C per-cell rows | Sub-D enrollment |
| Sub-E boundary contracts derivation | Consumes sub-C crossing records | Sub-E enrollment |
| Sub-F stitcher | Consumes sub-C per-tile data | Sub-F enrollment |
| Sub-G end-to-end validator (PRD §5 stage five) | Cross-sub-project invariants needing macro plan + tokens | Sub-G enrollment |
| Sweden enrollment | Sub-A cold-fetch dependency; polygon densification re-measurement; parallelism tuning | Future sub-project after sub-A fix |
| β boundary-touching sea-detection fallback | Documented escape hatch; not implemented | Future region's enrollment iff α subtype-based proves insufficient |
| Subtype / subclass tokenization | Requires encoder design change (one-token-per-feature contract) | Future sub-project per `known_issues.md` #2; sub-C stores `subtype_raw` so future work doesn't re-extract |
| Population density bucket (per-tile conditioning field) | Needs building footprint analysis | Sub-D; sub-C writes null with `_owner: sub-D` |
| Deterministic seed (per-tile conditioning field) | Reproducibility metadata, not generation conditioning | Training-time run config |

## 21. Implementation order (advisory)

1. **B2 follow-up** (PREREQUISITE per §3) — half-day; should be its own spec/plan.
2. Library skeleton: `cfm.data.sub_c` package with dataclasses + pure functions + enums + epsilon. No I/O yet.
3. Pure-function unit tests (Layer 1), TDD per function. Coordinate handling first (§7), then cell extraction (§8), then sea masking (§9), then policy (§10), then conditioning (§11.9).
4. I/O layer (`io.py`, `enums.py`, `manifest.py`) + encoding determinism tests.
5. Inline validator (`validator_inline.py`) + per-invariant unit tests.
6. Pipeline orchestrator (`pipeline.py`) — composes pure functions; first end-to-end run against a 1-tile fixture.
7. Torture-test fixture builder (`tests/fixtures/sub_c/build_torture_tile.py`).
8. Layer 2 pipeline-stage tests against torture-test fixture.
9. Cross-tile validator (`validator_cross_tile.py`) + 2-tile micro-fixture failure-mode tests.
10. CLI script `extract_tiles.py` + manifest assembly + `_SUCCESS` protocol.
11. CLI script `validate_extraction.py`.
12. Pre-commit lint rule for `import pandas`.
13. Layer 3 cached-Singapore integration tests.
14. Generate Singapore artifacts once locally; eyeball; commit if shape looks right.
15. Add `docs/known_issues.md` entries (2 new).
16. Final full-suite test run.

## 22. References

- Sub-A spec: `docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md`
- Sub-A handoff contract: `docs/data/handoffs.md`
- Sub-B1 spec: `docs/superpowers/specs/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis-design.md`
- Sub-B1 report: `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md`
- Sub-B2 spec: `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md`
- End-of-sub-B2 handoff (sub-C's directive): `docs/handoffs/2026-05-16-end-of-sub-B2.md`
- Phase 1 vocab: `configs/tokenizer/vocab_phase1.yaml`
- Phase 1 missing-value policy: `configs/data/missing_value_policy.yaml` (post-B2-follow-up four-case schema)
- Phase 0 tokenizer: `src/cfm/tokenizer/` (with `encode.py:59-60` `UnsupportedFeatureClass` hard-raise → driver of Topic 3b Option A; and `_require_in_bounds` → driver of Topic 2a split-at-boundaries)
- Sub-A loader: `src/cfm/data/overture/loader.py`, `src/cfm/data/overture/region.py` (Region / RegionGeometry / BboxScope contracts)
- B2 library (`_LOCKED_MISSING_POLICIES`, `canonicalize_yaml`, sha-exclusion pattern): `src/cfm/data/vocab_derivation.py`
- Known issues: `docs/known_issues.md` (sub-A cold fetch #1; subtype deferral #2; +2 new from sub-C)
- PRD: `PRD.md` (§5 data pipeline, §8 conditioning vocabulary, §5 stage five validator)
- Auto-memory (cross-sub-project principles applied; referenced by name without absolute paths):
  - `feedback_schema_vs_data_cost_asymmetry.md`
  - `feedback_dont_optimize_multiregion_under_singleregion_scope.md`
  - `feedback_epsilon_structural_vs_user_threshold.md`
  - `feedback_test_weakening_to_pass.md`
  - `feedback_append_only_vocab_safety.md`
  - `feedback_subagent_branch_pattern.md`
  - `feedback_handoff_agenda_is_floor.md`
  - `feedback_brainstorm_gate_discipline.md`
