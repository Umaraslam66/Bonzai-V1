# Readiness audit — execution-surface map (Step 1 artifact, 2026-06-10)

**Purpose.** The DENOMINATOR for the failure-class enumeration (handoff §3): the set of code,
artifacts, and dispatch points the Phase-2 bake-off executes end-to-end — including the unrun
Tasks 9–12 — proven complete by reconciling a static per-stage trace against an independent
runtime trace, with every locally-untraceable branch logged as an explicit gap. Classes are
enumerated against THIS map; a path absent from this map is a map defect, not an audit pass.

**Base:** main @ `b583fdc`, tree clean. Method: 6 per-stage static-trace subagents +
1 runtime-trace subagent (import-closure audit hooks on 5 entry points; live CPU smoke of
`train_scaffold.py`; real SG read→decode→promote chain over manifest tiles; field-level
reconciliation of the two already-ran Leonardo artifacts). Orchestrator spot-verified the
load-bearing claims independently (implementer ≠ reviewer).

---

## 1. Entry-point inventory (the surface's roots)

**Already ran (runtime evidence exists):**
| Entry point | What ran | Runtime evidence |
|---|---|---|
| `scripts/build_multiregion_train_shards.py` (+`.sbatch`, lrd_all_serial) | Task-8 EU train-shard build, 38 cities / 22,019 tiles | `reports/2026-06-10-task8-multiregion-train-shards-build.yaml` — field set matches writer line-for-line; `train_cities` in artifact == locally recomputed 38 names; totals cross-add |
| `scripts/run_gate_i_conditioning_discrimination.py` | gate-(i) extraction + BH verdict, job 45452784, build `619405a` | `reports/2026-06-10-gate-i-conditioning-discrimination-result.yaml` — field set matches dump dict; len(pairs)=321=n_qualifying; verdict FAIL coherent |
| `scripts/scaffold_ddp_validate.sbatch` → `ddp_audit_halt_check.py` + `ddp_resume_check.py` | DDP validation (2026-06-02, 4×A100) | historical reports; NOT re-traced (logged gap §6) |

**Unrun (Tasks 9–12 — audited as written, because the audit's scope is readiness):**
| Entry point | Plan task | State on disk |
|---|---|---|
| `scripts/bakeoff_diagnostic.sbatch` → `train_scaffold.py` | Task 9 | pre-builds `build_training_shards('2026-04-15.0','singapore')` (hardcoded); header says "on real Singapore"; `--emergence-floor 1.96` hand-typed SG figure |
| `scripts/bakeoff_run.sbatch` → `train_scaffold.py --config …` | Task 11 | same singapore pre-build; passes `--config` which **argparse does not accept** (known xfail `tests/training/test_cli_contract.py`); `configs/experiments/` does not exist |
| Task 10 / Task 12 runners (config generation, decision orchestrator) | Tasks 10, 12 | **do not exist** — planned-but-absent surface (§5) |

**No CLI path to EU training exists today:** `train_scaffold.py` exposes no `--region`/`--release`/
`--seed`/`--lr`; `ScaffoldConfig` defaults `region="singapore"`; the multi-region union mode
`CellDataModule(training_manifests=[…])` is implemented + tested but has **zero production callers**.
Task-8's 38 EU manifests are currently consumed by nothing.

---

## 2. Stage-by-stage live surface (consolidated; cites are content-anchored)

### 2a. EU corpus read
- `build_multiregion_train_shards.py::build_all_train_cities` → `build_shards.py::train_cities`
  (G4 roll-up ∩ validated − held-out; reads ONLY `name`+`validated`) → `build_train_city_manifest`/
  `build_train_city_shards` (ALL validated tiles; deliberately bypasses `_holdout_ids` — I1 handled
  at the loop) → `build_shards_in_memory` (per-tile: `read_tile_labels` + `_cell_density_by_cell` +
  `read_sub_f_cells`; `epsg_label = epsg_label_for_region(region)`).
- Readers: `sub_d/io.py::read_macro_core_parquet` (pinned 11-col schema), `sub_g/readers.py::read_sub_f_cells`
  (`pq.ParquetFile(path).read()` form), `holdout/labels.py::read_tile_labels` → `_derive_tile_conditioning`.
- Artifacts read: G4 roll-up (`reports/2026-06-05-phase-2-g4-corpus-dod.yaml`), sub-D `manifest.yaml`
  `tiles[]`, per-tile `macro_core.parquet` (**read twice per tile**: labels then density, two module-local
  `_cell_density_by_cell` copies — build_shards vs holdout.pipeline), `effective_conditioning.yaml`,
  sub-F `cells.parquet`, `configs/data/regions/<region>.yaml` (**only `projected_crs` read**),
  holdout manifests (routed; §2d).
- CRS authority is dual-source: region-config `projected_crs` is THE live source; sub-D manifest's
  `region_crs` and the multiregion manifest's per-region `crs` are present but **read by no consumer**.
  The plan's three-way CRS consistency check (Task-9 step-0) has no script.

### 2b. Conditioning construction (the enrichment target)
- **Headline structural fact (orchestrator-verified):** at training AND generation time the model
  receives ZERO conditioning values. The prefix is the value-agnostic 8 field-SLOT id constant
  (`datamodule.py::build_conditioning_prefix`, used at `flatten_shards_to_cells` and
  `train_scaffold._generate_and_score`). The 4-dim macro plan is derived, stored on shards
  (`TrainingShard.tile_conditioning`), and read at gate time — it never reaches the model input.
  Explicit slice-v1 DECISION (`micro_ar.py` docstring). `build_value_bearing_prefix` +
  `conditioning_id_span()`(=512 embedding rows) exist, sized, **unwired** on the data path.
  The delta-spec's prior "conditioned on city D means handed D's macro plans" is true of shard
  CONTENT, not of model INPUT under current wiring.
- Conditioning schema is **8 fields** (`_CONDITIONING_FIELDS`), not 4: population_density_bucket,
  zoning_class (tile dominant), road_skeleton_class (tile modal), cell_density_bucket (PER-CELL),
  region (=admin_region), coastal_inland_river, sub_c_morphology_class, seed.
- Per-dim provenance: zoning/skeleton/cell_density derive in sub-D (`pipeline.py::_derive_tile_targets`
  + `evidence.py`, buckets from locked `configs/macro_plan/v1/macro_plan_vocab.yaml`) → `macro_core.parquet`;
  coastal_inland_river + admin_region + morphology derive in sub-C → `meta.yaml` → copied into sub-D's
  `effective_conditioning.yaml` (schema-driven copy: a new sub-C field flows through with zero sub-D change).
- **#13 CONFIRMED:** `sub_c/pipeline.py` hardcodes `country_code="SG"` in the divisions lookup →
  `admin_region=None` for every EU tile (machinery exists; only the country filter blanks it; restoring
  it is a sub-C-regen-cost fix, data-layer not schema-layer).
- **#22 CONFIRMED:** `morphology_class="Asian-megacity"` literal at the only production call site;
  also `era_class="contemporary"`, `country="SG"`, `climate_zone="tropical_rainforest"` — three
  Singapore constants baked factually WRONG into every frozen EU `effective_conditioning.yaml`
  (currently unread by the model path; inherited by any enrichment that reads the dict wholesale).
- Tile labels collapse per-cell/per-edge grid structure to tile scalars: up to 36 per-cell zoning +
  112 per-edge skeleton values reduce to ONE dominant + ONE modal; only cell_density survives at
  cell granularity (and only into reporting strata / gate strata, not the model).
- Id-space hazard: `CONDITIONING_VALUE_BASE == CONDITIONING_ID_BASE` (=1508, runtime-confirmed) —
  slot ids collide with field-0 value-buckets 1..7; value-bearing must REPLACE the slot prefix, not augment.
- Import-time side effect (runtime-confirmed): importing `cfm.data.training.conditioning` reads the
  4 `configs/sub_f/*.yaml` and computes `CONDITIONING_ID_BASE` from deployed YAML content.

### 2c. Training-data load + training execution
- `train_scaffold.py::main` → `run_short`/`run_smoke` → `_datamodule` (region-conditional manifest +
  `expected_holdout_schema_for_region(cfg.region)` travel together) → `CellDataModule.setup` on ALL
  ranks: `load_training_manifest` → `run_holdout_audit` (fail-closed schema backstop + `lineage_audit
  .audit_no_holdout_leak`) → `build_shards_in_memory` (tokens re-read from parquet **in memory on every
  rank at every setup** — manifests carry lineage only, shards are not files) → `flatten_shards_to_cells`
  (drops empty + >5760-token cells, logged) → seeded tile-level split → pad-collate.
- Model: `backbone.py::build_backbone` string dispatch — `transformer-ar`→`MicroAR` (runtime: 7,119,844
  params at defaults); `mamba-hybrid`/`discrete-diffusion` **raise `BackboneNotYetBuilt`**; no `mamba_ssm`
  import exists anywhere in src/scripts (runtime-confirmed zero mamba modules). The generation hook
  `_generate_and_score` is typed against the AR interface only — a built diffusion model would have no eval path.
- Trainer: ddp iff devices>1; bf16 on gpu else 32-true; `ModelCheckpoint(train_time_interval=30min)`;
  `WorldSizeGuard` iff devices>1 (runtime-confirmed attach/absence); logger = TB if installed else CSV
  (runtime: CSV fallback taken locally); `maybe_compile` swallows compile exceptions (warn-and-proceed).
- **$WORK resume is split-brain:** `training/resume.py::work_checkpoint_dir`/`resume_ckpt_path` are
  built + tested + **never called**; `trainer.fit` gets no `ckpt_path`; checkpoints land in cwd
  `lightning_logs/`, not `$WORK`. `bakeoff_run.sbatch`'s USR1 across-job-resume header claim is
  currently false — a relaunch restarts from step 0.
- `env_lock.assert_training_env_locked` (torch 2.5.1+cu121 / lightning 2.6.5 / pydantic 2.13.4) — GPU
  entry points only. `deviation_log.py` built, unconsumed.
- CosineAnnealingLR `T_max = max(1, cfg.max_steps)` — a `--max-time`-bounded run without `--max-steps`
  anneals against the 2000-step default.

### 2d. Eval read
- Router (obligation-a): `holdout/paths.py::holdout_manifest_for_region` + `expected_holdout_schema_for_region`
  — singapore→SG manifest/1.0; {eisenhuttenstadt, glasgow, krakow, munich} (hardcoded `_EU_HELD_OUT_CITIES`
  frozenset, drift-guard test cross-refs the manifest)→multiregion/2.0; EU train city→ValueError (I1);
  unknown→ValueError. All four branches runtime-executed.
- Consumers: `geometry.py::holdout_polygons_per_active_cell` (emergence-density source — **test-only**;
  production floor arrives as the hand-typed `--emergence-floor 1.96` CLI float, default None ⇒ verdict
  silently skipped), `conditioning_discrimination.py::extract_features_by_city_stratum_metric` (gate-(i)),
  `build_shards.py::_holdout_ids`/`compute_training_tile_ids` (dual-region, SG=local fixture),
  `holdout/pipeline.py::generate_eval_set` (eval-set-gen; SG-fixed freeze path by design),
  `resolution.py::assert_resolution_sufficient` (**always the SG marker** — obligation (c) open; dormant, no caller).
- Frozen artifacts: SG manifest (1.0, 132 tiles, sha self-excluding) + SG `_EVAL_SET_LOCKED` (carries the
  KS numbers) + multiregion manifest (2.0, 4 cities, 46.1M held-out / 623.9M train tokens) + multiregion
  `_EVAL_SET_LOCKED` (**no KS fields, zero readers**). `manifest_sha256` is computed at freeze and
  **verified by no reader**. No reader checks `_EVAL_SET_LOCKED` presence before consuming a manifest.
  `partition_path` fields + `holdout_partition_dir` have zero consumers.
- `tile_dirname(…, epsg_label="EPSG3414")` default SURVIVES in the signature; all 6 live call sites pass
  explicit labels (d54424e + ed1138c on main, regression-locked), but any future caller inherits Singapore
  silently. The vacuous-0.0 MECHANISM survives in `holdout_polygons_per_active_cell` (`if not cells_path.exists():
  continue` + 0.0 on zero active cells, no log); the EU trigger was fixed, not the mechanism.
- EU region CRS labels (runtime-confirmed from configs): glasgow EPSG25830, munich EPSG25832,
  krakow EPSG25834 — ETRS89/UTM zones, not national grids.
- `train_scaffold._write_report` stamps the SG `eval_set_locked_marker(cfg.release)` path as provenance
  regardless of region (string only, never parsed).

### 2e. Decode
- One sealed chain: `sub_g/seam_decodability.py::split_cell_into_features` (trailing incomplete block
  dropped) → `sub_f/decoder.py::decode_feature` — **never returns Polygon** (Point or LineString only;
  the function's own docstring still claims Polygon — stale doc contradicting the code, a
  citation-is-not-the-clause hazard) → consumers must transform.
- **Consumer-must-transform contract compliance** (the load-bearing table):
  | Contract | gate-(i) `_tile_features` | scaffold `slice_eval` | `holdout_polygons_per_active_cell` | eval-set gen | sub-G gate |
  |---|---|---|---|---|---|
  | C1 promote building rings (`promote_building_rings`, 77-id construction identity, runtime-confirmed) | ✅ (619405a) | ✅ | ✅ | n/a (type-blind bref-rate) | n/a |
  | C2 bref-collapse exclusion | **❌ — V=2 collapses enter road_length as 0.0-length samples; bref roads carry ≤125 m last-vertex placeholder error into the KS distributions** | ✅ (excluded from OGC denominator, reported) | ✅ by construction | ✅ (is the instrument) | ✅ (guarded) |
  | C4 Points (POIs) | silently dropped, **no counter** | counted in decodability AND OGC denominator (Point always valid → inflates) | ignored | counted in denominator | skip validity, keep vertex bound |
  | C5 Multi* split per part | per-part samples (accepted) | per-part | per-part | per-part | ✅ explicit (H1 fix) |
  | C7 no-density cells skipped by `decode_region_blocks` | inherits (features invisible to gate) | n/a (strata hardcoded 0) | **does NOT density-filter** (counts all non-empty cells) — denominator differs from consumers that do | inherits | n/a |
- Plausibility: the 300 m `_VERTEX_BOUND_M` check exists ONLY in sub-G's `check_decodability` — generated
  output in `slice_eval` gets OGC-validity only; implausible coords pass. `try_decode_block` is bare
  `except Exception → None` by design (decodability as a rate).
- `realism.feature_samples` does NOT promote internally and has zero live callers — the future Task-12
  caller must promote first; **nothing structurally enforces it** (third consumer in a row relying on convention).
- GEOMETRY_TYPE enum gap (memory `project_sub_c_multi_geometry_gap`) is **CLOSED** on main
  (`sub_c/enums.py` has MultiPoint/MultiLineString/MultiPolygon, wired in pipeline; stale comments
  remain at `sub_c/io.py` + `sub_c/geom.py`). Memory updated.
- No georeferencing exists on the decode path: decoded coords are cell-local meters; no cell-origin
  offset, no CRS re-application anywhere (v1-persona GeoJSON georeferencing step absent).
- Decoder constants `_BLOCK=23`, `_DIRECTION_BASE=511`, `_MAGNITUDE_BASE=444`, bref ids 1500..1507
  hardcoded (twin in `validator_cross_tile`); must stay synced with encoder.

### 2f. Scoring + decision
- **WIRED today:** gate-(i) (`conditioning_discrimination_verdict` — BH guard, per-pair floors paired to n,
  UNSUPPORTED branch; ran, FAIL on disk) and the scaffold in-loop eval (`slice_metrics.slice_eval` —
  decodability/OGC/right-angle/bref-collapse/emergence verdict; reported-not-gated).
- **LIBRARY-ONLY (no caller; Tasks 10–12 would wire them; runtime-confirmed unloaded):**
  `multiregion_realism.{per_city_ks, decision_ks}` (its own docstring records the gap),
  `city_aggregate.{worst_case_city, binding_city_verdict}` (#21 power demotion),
  `ladder.{feasible_ladder, feasible_ladder_conservative, decision_basis}`,
  `curve.{fit_scaling_curve, structural_check_ok, pick_winner}`, `feature_resolution.*`,
  `resolution.assert_resolution_sufficient`, `perplexity_gap`, `conditioning_gate.py`
  (the older tolerance-shaped twin of the same §4 question — dead; the BH module superseded it).
- Threshold/teeth facts: `TRAIN_TOKENS = 623_900_790` hardcoded in ladder.py, claimed from
  `_EVAL_SET_LOCKED`, cross-referenced by **no guard test** (lock-without-guard). Two near-identical
  KS constants: `1.36` (conditioning_discrimination.noise_floor) vs `1.358` (feature_resolution), no
  cross-guard. `pick_winner` does NOT call `structural_check_ok` (pairing is the absent runner's
  obligation); `pick_winner({})` raises bare StopIteration; single-entry dict auto-wins.
  `binding_city_verdict` with one backbone → raw IndexError. `ks_distance` empty-side → 1.0 (fail-closed,
  but indistinguishable from a genuinely bad backbone without n). The Task-12 **4-city completeness
  assertion exists nowhere in code** (prose-only in the Phase-D handoff).
- Fail-open default recorded: `--emergence-floor` None ⇒ emergence verdict silently omitted.

---

## 3. Runtime-trace reconciliation (the map's own done-test, Sharpen 1)

Three independent legs: (A) import-closure audit (5 entry points, `sys.addaudithook` + sys.modules
delta); (B) live smoke `train_scaffold.py --smoke --devices 1` (rc=0; 60 cfm modules loaded ==
import closure exactly; 374 repo file opens incl. 362 per-tile reads = 494−132 holdout exclusion
visible in the open pattern); (C) real SG read→decode→promote chain over 9 manifest tiles (12,084
blocks, 3,346 promoted polygons; noisy non-round counts; no-density-cell skip observed live);
(D) field-level reconciliation of both already-ran Leonardo artifacts against the writers.

**Result:** all 25 statically-mapped modules runtime-confirmed live; all 14 claimed-dead modules
confirmed unloaded everywhere (incl. the smoke); all 9 flagged dynamic-dispatch points confirmed
(values: `CONDITIONING_ID_BASE=1508`, span 512; building ids = 77; routing branches all executed;
WorldSizeGuard attach behavior; CSV-logger fallback observed). One expectation corrected: EU region
configs are UTM 258xx.

**Static map was INCOMPLETE — 30 runtime-loaded modules folded in**, of which 4 are directly
load-bearing and are now first-class map entries:
- `cfm.data.training.shard_schema` (build_shards/datamodule dependency)
- `cfm.training.config` (`ScaffoldConfig` — carries the singapore/seed/lr defaults)
- `cfm.data.sub_f.vocab` (`vocab_tag_to_id` — sealed-vocab reader feeding conditioning ids + backbone head size)
- `cfm.eval.holdout.bref_rate` (via slice_metrics)
plus one missed import-time side-effect class: **`cfm/data/__init__.py` eagerly imports sub_c +
vocab_derivation** — the entire extraction layer loads into every process including training entry
points. Remaining 25 are transitive closure (sub_e/sub_f internals, sub_d enums/errors, holdout
manifest/selector/sizing, sub_c modules via the eager import).

---

## 4. Explicitly logged residual gaps (not locally traceable; the audit must carry these)

| Gap | Compensating evidence | Status |
|---|---|---|
| EU corpus parquet-read branches (38-city build; gate-(i) EU extraction) | Trace-D artifact reconciliation (field-level + cross-computed counts) — strong but indirect | LOGGED |
| 4-GPU DDP branches (WorldSizeGuard fire, ddp strategy, bf16, env-lock GPU path) | devices=4 construction shows guard attached; 2026-06-02 DDP reports are historical, not re-traced | LOGGED |
| TensorBoardLogger branch (tensorboard absent locally; CSV fallback is the proven path) | none | LOGGED |
| Leonardo $WORK pathing + sbatch shell layer (module load, venv, traps, env inheritance through `sbatch $0`) | none (shell layer never traced) | LOGGED |
| pyarrow C++ file opens invisible to audit hook | reads evidenced by returned data | instrumentation note |
| mamba / diffusion construction | confirmed-ABSENT surface (raises by design), not a gap | n/a |

## 5. Planned-but-absent surface (the bake-off plan executes these; they do not exist)

1. Task-9 EU repoint of the diagnostic (sbatch is Singapore-hardcoded; no `--region` plumbing).
2. Task-9 step-0 three-way CRS consistency check script (config CRS == sub-D `region_crs` == dir label).
3. Gate-input-(ii) reader (pilot-generation per-city extraction feeding `binding_city_verdict`).
4. Obligation (c): EU resolved-gap recompute; EU `_EVAL_SET_LOCKED` has no KS fields and no reader.
5. Task-10: `configs/experiments/` generation from `ladder.feasible`; ladder/basis persistence + re-read.
6. Task-11: `--config` CLI loader; $WORK resume wiring (`resume.py` → `trainer.fit(ckpt_path=…)`).
7. Task-12: decision orchestrator — path==persisted-basis assertion, 4-city completeness guard,
   promote-before-`feature_samples`, `pick_winner`⟷`structural_check_ok` pairing, curve y-value repoint to `decision_ks`.

## 6. Map completeness statement

The denominator = (6-stage static inventory) ∪ (30 runtime-discovered modules) with: zero
phantom static entries, zero runtime-loaded-but-unmapped modules remaining, the dead-list
runtime-confirmed dead, and the untraceable surface enumerated in §4 as named gaps rather than
silent absences. The per-class enumeration (Step 2) checks classes against §2's stages, §4's
gaps, and §5's absent-but-planned surface. The map's known limits: shell/sbatch layer and
EU-data/GPU branches rest on artifact-level evidence, not local re-execution.
