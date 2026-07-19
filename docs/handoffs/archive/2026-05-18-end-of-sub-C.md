# Session handoff — end of Phase 1 sub-C / start of sub-D (2026-05-18)

> **For the new session:** read this doc, then propose a sub-D brainstorm agenda. Do not re-read sub-C's brainstorm history; the decisions are committed in the spec.

## 1. Sub-C closeout state

Phase 1 sub-C (multi-cell tile extraction) merged to main on 2026-05-18 via `git merge --no-ff phase-1-sub-C-tile-extraction` (merge commit `d41c0af`, 43 commits on the feature branch). 372 fast tests passing + 4 Layer 3 cached-Singapore slow tests + the 12-minute byte-identical re-extraction determinism check. Branch retained on disk (not deleted, not pushed to remote — same pattern as sub-A/B1/B2). Shipped: pure-pyarrow + shapely + pyproj pipeline producing per-tile `cells.parquet` + `features.parquet` + `crossings.parquet` + `meta.yaml` + `provenance.yaml` under a region-level `manifest.yaml` + `_SUCCESS` integrity chain; 10 inline invariants + 4 cross-tile invariants; pre-commit lint forbidding pandas in write-path; `sub_c_schema_version = 1.1` (Multi\* enum extension landed mid-implementation).

## 2. Sub-C's output contract (what sub-D consumes)

Schema definitions are authoritative in the sub-C spec — do NOT duplicate, just cite:

- `cells.parquet` schema (per-cell rows, ≤64 per tile): **spec §11.3**
  - Likely sub-D primary input: `cell_i, cell_j, water_fraction, sea_water_fraction, cell_area_admin_clipped_m2, kept_features_count`.
  - `water_fraction` is all-water (sea + inland) per spec §11.3 — landed real in pre-merge fix `db92caf`; sub-D may rely on it for wet-cell discrimination.
- `features.parquet` schema (per cell-local sub-feature row): **spec §11.2**
  - 15 columns including denormalized `bbox_*` (predicate-pushdown ready per §4.4) and `geometry_type` int8 enum (codes 0–5 covering Point/LineString/Polygon + Multi\* variants).
  - Sub-D joins to cells via `(cell_i, cell_j)`.
- `crossings.parquet` schema (per crossing event): **spec §11.4** → **spec §8.2** (8-column flat schema; canonical sort key locked).
  - Sub-D probably ignores crossings (that's sub-E territory) but `source_feature_id` is the join key if needed.
- `meta.yaml` per-tile: **spec §11.5**
  - `aggregates`: `kept_cell_count, sea_mask_drop_count, mean_water_fraction (area-weighted), mean_sea_water_fraction, feature_count_by_class, crossing_count`.
  - `conditioning_per_tile`: **spec §11.9** (7 fields — see below for which are sub-D's responsibility).
- `provenance.yaml` per-tile: **spec §11.6** (extraction record + input/output digests).
- `manifest.yaml` region-level: **spec §11.7** (tile inventory sorted by `(tile_i, tile_j)`; `config` block with epsilon constants + pipeline_order; `conditioning_defaults` for region-constant fields).
- `_SUCCESS` semantics + write order: **spec §11.8**.

**Conditioning vector ownership (§11.9):** sub-C populates `country, climate_zone (manifest); morphology_class, era_class, admin_region, coastal_inland_river (per-tile)`. **sub-D owns `population_density_bucket`** — currently `null` with `_owner: sub-D` in every tile's `meta.yaml.conditioning_per_tile`. Filling it is on the sub-D scope ledger.

**Consumer API:** `cfm.data.sub_c.{io,manifest}` re-exports `read_manifest`, the dataclasses (`CellAggregate`, `TileMeta`, `TileProvenance`, `RegionManifest`), and the schema constants `_FEATURES_SCHEMA`, `_CELLS_SCHEMA`, `_CROSSINGS_SCHEMA`. Use `pyarrow.parquet.ParquetFile(path).read()` for per-tile reads (NOT `pq.read_table` — Hive partition trap; see memory `feedback_pyarrow_hive_partition_inference.md`).

## 3. Open cross-sub-project decisions sub-D inherits

**Tokenizer `emit_unknown_token` fall-through (known_issues #4) — sequencing question.** Sub-C stores raw class values; the encoder at `src/cfm/tokenizer/encode.py:59-60` hard-raises on not-in-vocab. The fix is ~10 lines. The question is WHEN:

- **α — land between sub-D and sub-E.** Cleaner because sub-G's end-to-end validator needs token-round-trip; landing the enhancement earlier means sub-G doesn't gate on it.
- **β — land after sub-G.** Defers the brainstorm-spec-test cycle; sub-G can be designed to validate token-decodability without the enhancement existing yet (sub-G itself doesn't run the encoder on extracted output; it consumes downstream).

**Recommendation: β.** Reasoning: (a) sub-D/E/F all consume per-cell rows, not tokens — the enhancement is on the *training* critical path, not the data-pipeline critical path; (b) deferring keeps the sub-project boundaries cleaner (one tokenizer change = one tokenizer sub-project); (c) the enhancement is small enough that landing it after sub-G doesn't gate anything time-sensitive.

Note for sub-D: this decision doesn't affect sub-D scope directly. Sub-D consumes per-cell rows; the tokenizer choice is downstream.

**Schema versioning coupling (known_issues #8).** Sub-C's cross-tile validator invariant #1 compares `manifest.sub_c_schema_version` against `meta.yaml.schema_version` — over-coupling YAML-format version to data-shape version. Sub-D output (whatever shape it takes) should adopt the like-for-like comparison pattern, NOT propagate the coupling. Specifically: if sub-D adds new per-tile YAML artifacts or new manifest fields, version YAML-format and data-shape separately.

**Coastal-inland-river enum, morphology_class, era_class, climate_zone** are currently raw strings at sub-C scope (per spec §11.9 D2-A deferral). A future "conditioning vocabulary" sub-project locks the enum sets. Sub-D consumes the raw strings; do not introduce a parallel mapping in sub-D.

## 4. Feedback-memory list relevant to sub-D

Read these from `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/`; auto-loaded via `MEMORY.md` at session start. Cited, not duplicated:

- **Cost-asymmetry** — `feedback_schema_vs_data_cost_asymmetry.md`. Sub-C applied this 10+ times; sub-D should default to storing raw / cheap-to-keep / impossible-to-recover values.
- **Denormalization criterion** — spec §4.4 (the rule itself; not in memory). "Iff every consumer benefits AND access pattern is established." Sub-C accepted 2 of 10 candidates under this rule.
- **α/β EPSILON discipline** — `feedback_epsilon_structural_vs_user_threshold.md`. Apply EPSILON at structural boundaries (0, 1, equality); strict comparison at user thresholds.
- **Branch discipline (verbatim in implementer dispatches)** — `feedback_subagent_branch_pattern.md` + `project_branch_pattern.md`. Every implementer dispatch must forbid new branches / push / PR.
- **Halt on validator failure** — `feedback_test_weakening_to_pass.md`. If real data violates an invariant, the assumption failed; do not weaken the test.
- **Handoff agenda is a floor, not a ceiling** — `feedback_handoff_agenda_is_floor.md`. Audit this doc for missing topics before invoking brainstorming.
- **Topic-by-topic gate discipline** — `feedback_brainstorm_gate_discipline.md`. One topic per assistant message during brainstorms.
- **Follow the data, not the PRD** — `feedback_follow_data_over_prd.md`. If sub-D's empirical findings disagree with the PRD, update the PRD.
- **Don't optimize multi-region under single-region scope** — `feedback_dont_optimize_multiregion_under_singleregion_scope.md`. Sub-D is Singapore-only; defer Sweden-specific design choices to Sweden enrollment.
- **Pyarrow Hive partition inference** — `feedback_pyarrow_hive_partition_inference.md`. If sub-D reads `tile=*` parquets, use `pq.ParquetFile(path).read()`.
- **Marginal cost-of-cut for floor selection** — `feedback_marginal_cost_of_cut.md`. Relevant if sub-D introduces any vocabulary/quantization decisions.
- **Append-only-within-phase discipline** — `feedback_append_only_vocab_safety.md`. Sub-C extended GEOMETRY_TYPE this way; sub-D inherits the pattern for any enum it introduces.
- **uv sync after pyproject changes** — `feedback_uv_sync_dev_extras.md`. Run `uv sync --extra dev` after any pyproject edit.

**Read the codebase before designing.** Not a memory entry per se — but sub-D should inspect sub-C's actual on-disk output before locking the consumer contract. A 2-tile cached-Singapore extraction is available via `scripts/extract_tiles.py --region singapore --output-dir data/processed/sub_c/2026-04-15.0/singapore/` (or run the Layer 3 slow test which produces a 12-tile subset at `tmp_path` for inspection). Empirical inspection beats reading the spec in isolation — sub-C surfaced 3 architectural surprises (Multi\* enum, Hive partition, divisions theme name path) that the spec didn't anticipate.

## 5. Cross-references to known_issues.md

Existing entries that sub-D should be aware of (verbatim text in `docs/known_issues.md`):

- **#1** Sub-A cold-fetch (~8 hours). Not a sub-D blocker (sub-D reads sub-C output, which is cache-served). Becomes relevant only when adding a second region.
- **#3** Sweden densification revisit. Not a sub-D blocker; Sweden enrollment dependency.
- **#4** Tokenizer `emit_unknown_token` fall-through. **Not a sub-D blocker** (per §3 above — sub-D consumes per-cell rows, not tokens). Sequencing question is sub-D-adjacent.
- **#7** `--rerun` CLI stub. Loud `NotImplementedError`; not silent corruption. Not a sub-D blocker.
- **#8** Cross-tile validator invariant #1 version coupling. **Sub-D should not propagate this pattern** — see §3 above.

#2 (subtype/subclass tokenization deferral) and #5/#6 (resolved pre-merge) are not sub-D-relevant.

## 6. First-message instructions for the new session

The new session should, in order:

1. **Confirm it has read this handoff + the §2 cited spec sections** (sub-C spec §11.2, §11.3, §11.4, §11.5, §11.6, §11.7, §11.8, §11.9). Reply explicitly: "Read end-of-sub-C handoff, sub-C spec §11.\*."
2. **Sanity-check workspace:**
   ```bash
   git rev-parse --abbrev-ref HEAD     # expect: main
   git log --oneline -1                # expect: d41c0af Merge sub-C: ...
   uv run pytest -q                     # expect: 372 passed, 10 deselected, 1 xfailed
   ```
3. **Propose a sub-D brainstorm agenda.** Per memory `feedback_handoff_agenda_is_floor.md`: audit this handoff for missing topics before locking the agenda. Specifically, consider:
   - What does "macro plan" mean concretely? PRD §5 references it; lock the operational definition.
   - What does sub-D output to sub-E/F/G? (The output contract is the most load-bearing decision.)
   - Per-cell vs per-tile vs per-region operations.
   - Does sub-D introduce its own conditioning fields beyond `population_density_bucket`?
   - Determinism contract scope (file format pinning; sort keys; sha exclusions).
   - Test strategy (Layer 1/2/3 same pattern as sub-C?).
4. **Do not enter implementation mode** until brainstorm → spec → plan cycle completes per the project pattern.

---

End of handoff. Next session: §6, in order.
