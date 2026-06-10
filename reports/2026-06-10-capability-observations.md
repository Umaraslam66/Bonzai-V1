# Capability & use-case observations (side-note, 2026-06-10)

**Status: observations only.** Harvested during the readiness-audit surface trace per standing
instruction. These feed a separate strategy discussion; they are NOT audit findings, NOT scope,
and nothing here implies action. Grouped by layer.

## What the conditioning layer discards (enrichment-adjacent raw material)

- **Raw continuous signals are quantized to 4 buckets very early.** `building_footprint_ratio`
  (per cell AND tile-p75), `road_crossing_count` (per edge), `sea_water_fraction` (per cell),
  river/stream lengths all exist upstream (sub-C cells/meta; sub-D `derivation_evidence.parquet`
  retains raw metric values) but only 4-way buckets reach conditioning. Gate-(i)'s FAIL says
  exactly this stratum is too coarse to carry city character.
- **Grid structure collapses to tile scalars.** `macro_core.parquet` carries up to 36 per-cell
  zoning + 36 per-cell density + 112 per-edge skeleton values per tile; `TileLabels` reduces
  zoning to one dominant and skeleton to one modal class. The `TrainingShard.macro_tokens` field
  is the schema-anticipated carrier for the full grid — provisioned, always empty.
- **Per-cell water signal exists at the same granularity as cell_density** (`sea_water_fraction`,
  per-cell river lengths) while `coastal_inland_river` is tile-level.
- **admin_region for EU is recoverable, not missing-by-nature** — full lookup machinery exists;
  only the hardcoded `country_code="SG"` blanks it (sub-C-regen cost, data-layer).
- **G4 roll-up per-city metadata is discarded at the corpus-read layer**: morphology, density,
  geography, crs, tokens, tok_per_tile are available per train city; only name+validated are read.
  Potential training stratification / curriculum / per-city weighting signal.
- The value-bearing prefix machinery (ids, embedding rows, builder) is fully provisioned — the
  enrichment's mechanical cost on the model side is small; the binding constraint is upstream
  (what values exist and their EU correctness).

## What the token representation encodes that nothing exposes

- **bref tokens encode crossing direction × road class** (8 ids: 4 edges × MAJOR/MINOR) — decode
  discards both; the cross-cell stitching signal is unexposed to every consumer. A future
  tile-coherence or connectivity eval could read it for free.
- **The semantic tag (`body[0]`) carries feature class**; `decode_feature` discards it. Only the
  building class is reconstructed (via token-id scan); road sub-class, POI category, and
  `<unknown_*>` distinctions are decoded away.
- **`decode_region_blocks` holds `(cell_i, cell_j)` and discards it** — no consumer can place
  decoded features in the tile frame or score cell-to-cell adjacency from decode output. Tile-frame
  placement + cell adjacency are one plumbing change away.
- No georeferencing path exists (cell-local meters, no cell-origin offset, no CRS re-application) —
  the v1-persona "GeoJSON in a real place" step is greenfield.

## What evals could measure with data already in hand

- **Gate-(i)'s report already persists the full per-(city, stratum, metric) n-map** — per-city
  emergence floors and per-stratum feature-resolution gaps are computable with zero extra IO.
- **Density-stratified KS (per-stratum realism) needs only a groupby** on the real side —
  `decode_region_blocks` already tags features with `cell_density_bucket`. (Generated side
  currently hardcodes stratum 0 — the blocker is conditioning, not plumbing.)
- **`_right_angle_stats` computes all corner angles** and keeps only the 90°±10° rate; the full
  angle histogram is free and is a plausible city-character fingerprint (Glasgow grid vs Kraków
  old town) — i.e. an eval-side discriminator for exactly the morphology the conditioning misses.
- **Per-feature token-length vs decode-success correlation** (truncation-cause attribution) is
  computable from data `slice_eval` already touches.
- `curve.ScalingFit` retains all bootstrap parameters — per-backbone slope-CI (is b distinguishable
  from 0) is one percentile call away, mechanizing the structural check.
- Holdout manifests carry `provenance_sha256` + `macro_vocab_sha256` per tile and per-region
  `n_usable_tiles`/`crs` — none read by code; a cheap read-time integrity + power-bookkeeping layer
  is available without new artifacts.

## Harvested during Step 2 (failure-class enumeration)

- **Gate-(i)'s FAIL cannot localize which coarseness layer binds** (stratum dims vs grid→scalar
  collapse vs 4-bucket quantization — it conditions on already-collapsed labels). A cheap
  pre-enrichment diagnostic: re-run discrimination varying ONE layer at a time (e.g. per-cell
  zoning instead of tile-dominant) to find where the city character actually lives before choosing
  what to enrich.
- The full per-pair (D, floor, n) table gate-(i) already persists supports a practical-effect-size
  re-analysis (how many pairs survive a KS≥some-δ floor) with zero new compute — useful both for
  gate recalibration and as a "how much character is missing" magnitude estimate for enrichment
  sizing.
- The eval layer's strongest guards cluster where a failure already happened once (tile_dirname,
  manifest routing) — coverage tracks incident history, not design. A capability angle: the same
  incident-driven counters (n_excluded_thin, bref_collapse, dropped{empty,too_long}) form a
  ready-made template for denominator-integrity counters elsewhere.
- A "wiring check" (who calls this in production?) would have caught all four tested-but-unwired
  conditioning machineries mechanically — candidate for the protocol as a cheap standing gate.

## Misc

- The eval-set generator's per-stratum machinery (selector, sizing, floors) is general — pointing
  it at a second region is mostly data cost, not code cost (consistent with the second-region
  escalation trigger already recorded in the SG marker).
- `shuffles.py` WITHIN_BUCKET keys on country/climate/morphology/era — all currently region-constant
  (and partly wrong) — so WITHIN_BUCKET == CROSS_TILE in practice; if conditioning enrichment lands,
  this dormant eval becomes meaningful for free.

## Segment-2 execution observations (readiness-closure Phases 3–5, harvested 2026-06-10)

- **A plan test can be mathematically unsatisfiable and still read plausible.** Task 15's
  "two manifests each individually under threshold, together over" is impossible — a union
  rate is a convex combination of per-manifest rates, bounded by their max. The implementer
  caught it pre-RED and substituted a direction-discriminating pair (cross-manifest
  accumulation proven by message counts; union-level-not-per-manifest proven by dilution).
  Pattern to screen future plan tests for: aggregate-threshold tests whose fixtures cannot
  exist.
- **A reviewer's empirical reproduction can use the wrong construction path.** The Task-14
  quality reviewer "reproduced" a generation crash by building `MicroAR` directly with
  `max_len=cfg.max_len`, bypassing `build_backbone`'s `+CONDITIONING_PREFIX_LEN` position
  sizing — the production path fits exactly. Orchestrator adjudicated by reading the real
  build path before dispatching a fix; the disputed fact is now pinned as a test
  (`test_generation_at_exact_positional_capacity_through_production_build`). Companion fact
  discovered during the pin: the true crash boundary is `max_new >= max_len + 2` (the last
  sampled token is appended but never fed back).
- **Long-running subagents can stall on a buffered slow-suite run** (Task 20 implementer
  ended its turn waiting on output). The work was complete and correct in the tree;
  orchestrator ran the full verification (suite, slow e2e, ruff) and committed with
  provenance noted in the commit body. Verified-end-state discipline covered the gap.
- **n_cond axis conflation is a recurring trap**: `n_cond=conditioning_id_span()=512` is
  embedding-table ROWS; sequence positions come from `CONDITIONING_PREFIX_LEN=8`. The
  conflation propagated from an orchestrator dispatch text into an sbatch comment before
  review caught it. Both axes now documented at config.py max_len and the diagnostic sbatch.
