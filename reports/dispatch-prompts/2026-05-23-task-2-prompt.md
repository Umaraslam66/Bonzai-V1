# Task 2 implementer dispatch prompt

**Status:** Revised v2; pending reviewer read-through / approval before dispatch.
**Target:** General-purpose subagent / Codex agent.
**Suggested model:** Sonnet-class.
**Branch:** `phase-1-sub-F-micro-tokenizer` (base includes Task 4 close commit `8721b2b`).

> The prompt below is the verbatim text to give the implementer agent. Everything between the `===` markers is the agent's prompt body.

===

Task: Sub-F Task 2 - BP2 encoder primitives + round-trip thresholds, Halt 2 surface only.

You are working in `/Users/umaraslam/Projects/Bonzai-OSM` on branch `phase-1-sub-F-micro-tokenizer`. You are not alone in the codebase: do not revert edits made by others; inspect current state and work with it. Do not push. Do not create a PR. Do not proceed past Halt 2 approval. Do not write a locked `encoding_primitives.yaml`.

## Preconditions

- Branch: `phase-1-sub-F-micro-tokenizer`.
- Task 4 is closed at `8721b2b`.
- `configs/sub_f/semantic_vocab.yaml`, `configs/sub_f/unknown_family.yaml`, and `configs/sub_f/sentinel_inventory.yaml` are `LOCKED`.
- `configs/sub_f/sentinel_inventory.yaml` locks BP1/BP4/dataloader IDs and leaves BP2 block `300..1499` as PLACEHOLDER; Task 2 surfaces whether BP2 fits before reviewer lock.
- Do not push. Do not PR. Do not proceed past Halt 2 approval.

## Pre-dispatch audits

### Audit step 1: confirm sub-C WKB writer/source symbols still anchor geometry contract

Run:

```bash
grep -n "byte_order" src/cfm/data/sub_c/io.py
grep -n "shapely\|dump_wkb" src/cfm/data/sub_c/io.py
```

Expected:
- A `byte_order=1` WKB writer path exists in `src/cfm/data/sub_c/io.py`.
- `dump_wkb` and shapely/WKB-related symbols are present.

If the WKB contract moved or no longer forces little-endian bytes: STOP, report BLOCKED. That is a verify-before-lock mismatch requiring reviewer classification.

### Audit step 2: confirm cached Singapore feature geometry decodes across all EPSG3414 tiles

Run:

```bash
uv run python -c "import pyarrow.parquet as pq; from pathlib import Path; from shapely.wkb import loads; root=Path('data/processed/sub_c/2026-04-15.0/singapore'); paths=sorted(root.glob('tile=EPSG3414_*/features.parquet')); print('tile_count', len(paths)); assert paths; rows=0; first=None; classes={}; [classes.__setitem__(int(r['geometry_type']), classes.get(int(r['geometry_type']), 0)+1) or None for p in paths for r in pq.ParquetFile(p).read(columns=['geometry_type']).to_pylist()]; p=paths[0]; raw=pq.ParquetFile(p).read(columns=['geometry']).column('geometry')[0].as_py(); print(p.parent.name, raw[0], loads(raw).geom_type); print('geometry_type_counts', classes)"
```

Expected:
- `tile_count` is greater than `1`.
- Tile names match `tile=EPSG3414_*`.
- First geometry WKB byte is `1`.
- `shapely.wkb.loads` returns a geometry type.

If this fails: STOP, report BLOCKED. Do not analyze a single tile or non-projected source.

### Audit step 3: confirm BP2 placeholder block remains available

Run:

```bash
uv run python -c "import yaml; d=yaml.safe_load(open('configs/sub_f/sentinel_inventory.yaml')); b=d['bp2_encoding_primitives_placeholder']; print(d['_status'], b['start_id'], b['end_id'], b['placeholder'])"
```

Expected: `LOCKED 300 1499 True`.

If status/range/placeholder drifted: STOP, report BLOCKED.

## Implementation

Create:
- `src/cfm/data/sub_f/__init__.py`
- `src/cfm/data/sub_f/enums.py`
- `scripts/sub_f/analyze_geometry_primitives.py`
- `configs/sub_f/encoding_primitives.yaml` with `_status: PROPOSED`
- `tests/data/sub_f/test_encoder.py`
- `reports/2026-05-23-phase-1-sub-F-task-2-halt.md`

## Analysis input scope

- Iterate every `data/processed/sub_c/2026-04-15.0/singapore/tile=EPSG3414_*/features.parquet`.
- Use `pq.ParquetFile(path).read()`, not parent-directory `pq.read_table()`.
- Decode `geometry` with `shapely.wkb.loads`.
- Aggregate globally across all Singapore tiles, not first tile.
- Halt report must enumerate:
  - tile count,
  - total feature count,
  - total feature count per geometry class / `geometry_type`,
  - sample count per measured geometry class.

## Candidate surface

Evaluate every combination:
- `direction_count`: `8`, `16`, `24`
- `magnitude_quantum_m`: `0.25`, `0.5`, `1.0`

For all 9 rows, report both analytical and measured behavior. Analytical-only threshold locks are not sufficient.

## Geometry primitive distributions

Aggregate across all tiles:
- turn-angle distribution for polylines and polygon exterior rings,
- vertex-spacing distribution for polylines and polygon exterior rings,
- building corner-angle distribution.

For building corner-angle characterization, report:
- total building polygon corner count,
- fraction within `+/-5 deg` of `90 deg`,
- p50/p95/p99 absolute deviation from `90 deg`.

This is input characterization. Do not reuse the POC city's 95% claim as Singapore evidence.

## Measured round-trip L_inf per candidate

For each of the 9 candidate rows:
- Deterministically sample up to `1000` random Singapore features per measured geometry class:
  - polylines,
  - polygon exterior rings.
- Use `random.Random(20260523)` or equivalent deterministic seed.
- Sampling procedure:
  - collect all candidate features for the geometry class,
  - sort by stable key `(tile_id, source_feature_id)` ascending,
  - run `random.sample(sorted_features, min(1000, N))` using the seeded generator.
- Do not rely on parquet read iteration order for sample determinism.
- If a class has fewer than `1000` features, use all available features and report the actual count.
- Implement analysis-local encode/decode helpers for each candidate:
  - anchor is vertex 1 per spec section 3.2,
  - direction bins use the candidate `direction_count`,
  - magnitude values use candidate `magnitude_quantum_m`,
  - segments longer than `32m` split into same-direction chunks,
  - decoded geometry may include synthetic collinear vertices from chunking.
- Compute actual per-feature L_inf vertex error against original vertices. Synthetic collinear vertices are admitted; original vertices must be recovered within tolerance.
- L_inf measurement convention:
  - Polylines are open: compute across all `V` original vertices.
  - Polygon rings are closed: compute across `V` original vertices excluding the implicit closure vertex where last coordinate equals first coordinate; decoder reconstructs closure.
- Report per candidate:
  - `roundtrip_l_inf_mean_m`,
  - `roundtrip_l_inf_p50_m`,
  - `roundtrip_l_inf_p95_m`,
  - `roundtrip_l_inf_p99_m`,
  - `roundtrip_l_inf_max_m`,
  - sample counts and skip counts.

The proposed L_inf threshold must anchor on measured p95/p99 behavior, not only the analytical half-bin/perpendicular/quantization bound.

## Right-angle measured behavior per candidate

First characterize input right-angle corners from Singapore building polygons:
- Define input-right-angled corners as corners with absolute deviation from `90 deg` <= `5 deg`.
- Surface total count and fraction of building corners meeting that definition.

For each of the 9 candidate rows:
- On input-right-angled corners only, encode/decode the parent polygon ring.
- Measure post-round-trip angle behavior:
  - absolute decoded deviation from `90 deg`,
  - absolute change from input corner angle.
- Report distributions:
  - mean,
  - p50,
  - p95,
  - p99,
  - max,
  - measured corner count,
  - skip count.

The proposed 95th-percentile angle threshold must anchor on measured Singapore post-round-trip distribution.

## Collinearity admission threshold derivation

Derive a proposed collinearity admission threshold explicitly. Do not only include a field name.

Method:
- Identify Singapore polyline collinear-candidate triples: consecutive segments whose turn angle is within `+/-5 deg` of straight continuation.
- For each triple, compute the middle vertex's perpendicular deviation from the straight line through its neighboring vertices.
- Halt report must enumerate `X` collinear-candidate triples out of `Y` total polyline interior triples, where `Y = sum(V - 2)` across sampled/eligible polylines.
- If `X < 500`, flag the empirical p95 basis as statistically weak for reviewer lock.
- Report distribution:
  - count,
  - mean,
  - p50,
  - p95,
  - p99,
  - max.
- Also report fixed multiples of the candidate magnitude quantum, such as `1x` and `2x`.
- Proposed threshold may be either empirical p95 or a fixed multiple of magnitude quantum, but the halt report must state which and why.

Spec framing: collinearity admission threshold means max perpendicular deviation from the straight line through neighboring decoded vertices.

## Anchor scheme comparison

Compare both anchor schemes:
- `flat`: 2 tokens per anchor; coordinate vocab size derived from `[0,250]m` grid and chosen quantum.
- `hierarchical`: 4 tokens per anchor; reduced coordinate vocab per spec section 3.6.

Anchor vocab size derivation is locked to spec section 2.2 / 3.6:
- Flat anchor vocab = `2 * ceil(250 / magnitude_quantum_m)`, with separate x and y coordinate vocabularies.
  - At `0.5m` quantum: `2 * 500 = 1000` slots.
- Hierarchical anchor vocab = `(16 coarse + 32 fine) * 2 axes = 96` slots.
  - The `16 coarse + 32 fine` split is locked by spec section 3.6; do not invent a different split.

For each anchor scheme, report:
- anchor vocab size,
- tokens per anchor,
- mean sequence length per cell,
- p95 sequence length per cell.

Use measured all-tile Singapore feature/cell distribution. Boundary-reference overhead is out of scope for Task 2; note that Tasks 3/7 cover cross-cell overhead later.

Anchor scheme has no default. Lock must be based on measured vocab-size vs mean/p95 sequence-length-per-cell tradeoff.

## Cheap-to-keep rationale to include in Halt 2 report

- `direction_count`: default-toward-16. Use 24 if measured curved-road precision materially improves; use 8 only if measurement shows angular noise dominates and 16/24 buy no structural value.
- `magnitude_quantum_m`: default-toward-0.5m. 0.25m increases magnitude vocab and may improve measured precision; 1.0m loses sub-meter precision that is expensive to recover.
- `anchor_scheme`: no default. Flat costs larger vocab but fewer tokens per anchor; hierarchical costs smaller vocab but more tokens per anchor. Lock by measured mean/p95 sequence-length-per-cell tradeoff.

## YAML output

Write `configs/sub_f/encoding_primitives.yaml` with:
- `_status: PROPOSED`
- `release: 2026-04-15.0`
- `source_scope` including all-tile Singapore path and tile/feature counts
- `joint_surface` with all 9 rows
- `input_geometry_characterization`
- `right_angle_input_characterization`
- `collinearity_candidate_distribution`
- `anchor_scheme_comparison`
- `bp2_placeholder_fit` for block `300..1499`
- `proposed_lock` containing:
  - `direction_count`
  - `magnitude_quantum_m`
  - `anchor_scheme`
  - `round_trip_l_inf_threshold_m`
  - `round_trip_angle_threshold_deg`
  - `collinearity_admission_perpendicular_m`
  - rationale tied to measured data

Do not set `_status: LOCKED`.

## Tests

Add tests in `tests/data/sub_f/test_encoder.py` asserting:
- `encoding_primitives.yaml` is `_status: PROPOSED`.
- `joint_surface` contains exactly the 9 candidate pairs.
- Every joint-surface row includes analytical fields and measured L_inf fields.
- Every joint-surface row includes measured right-angle post-round-trip fields.
- Input characterization reports more than one tile and per-geometry-class feature counts.
- Right-angle input characterization reports fraction within `+/-5 deg` of `90 deg`.
- Collinearity derivation reports empirical distribution and proposed threshold method.
- Collinearity derivation reports candidate count `X`, total interior triples `Y`, and whether `X < 500`.
- Anchor comparison includes both `flat` and `hierarchical`, each with vocab size, tokens per anchor, mean sequence length per cell, and p95 sequence length per cell.
- Anchor comparison uses the explicit spec-locked vocab derivation: flat `2 * ceil(250 / magnitude_quantum_m)`, hierarchical `96`.
- BP2 placeholder fit references `300..1499`.

## Verification

Run expected red/green:

```bash
uv run pytest tests/data/sub_f/test_encoder.py -v
```

Expected before YAML exists: FAIL with missing config or missing fields.

Then run:

```bash
uv run python scripts/sub_f/analyze_geometry_primitives.py \
  --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
uv run pytest tests/data/sub_f/test_encoder.py -v
git diff --check
```

Expected after analysis: tests PASS and diff check clean.

## Halt report

Create `reports/2026-05-23-phase-1-sub-F-task-2-halt.md` containing:
- Audit outcomes.
- All-tile input inventory: tile count, feature counts, geometry-class counts, sample counts.
- Joint 9-row surface with analytical and measured L_inf fields.
- Singapore building right-angle input characterization.
- Per-candidate post-round-trip right-angle deviation distribution.
- Polygon-ring L_inf convention applied: exclude implicit closure vertex from original-vertex error measurement.
- Collinearity admission threshold derivation, including candidate count `X`, total interior triples `Y`, and weak-basis flag if `X < 500`.
- Anchor flat vs hierarchical comparison with vocab size and mean/p95 sequence length.
- Proposed direction count, magnitude quantum, and anchor scheme.
- Proposed L_inf, 95th-pct angle, and collinearity thresholds with measured-data rationale.
- BP2 placeholder fit check for `300..1499`.
- Section 10.5 telemetry.
- Status: `DONE_WITH_CONCERNS` unless an audit mismatch requires `BLOCKED`.

## Commit

Commit halt-pending work as:

```text
wip(sub_f): T2 pre-halt - BP2 encoder primitives + round-trip surface
```

Report final status as `DONE_WITH_CONCERNS` when Halt 2 surface is committed, or `BLOCKED` with the report content if an audit/plan mismatch surfaces.

Do not lock `encoding_primitives.yaml`. Do not update `sentinel_inventory.yaml` from BP2 PLACEHOLDER to final BP2 lock. Halt 2 reviewer approval is required first.

===
