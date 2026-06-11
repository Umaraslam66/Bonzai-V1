# Readiness-Closure + Conditioning-Enrichment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 17-class audit's gap register (Mission A) and make the cross-city eval valid
via the two-axis conditioning fix (Mission B), so the bake-off can resume on a surface with no
unexamined latent-failure class and a ~~reopened T5 gate that can actually render PASS~~
**[REVISED 2026-06-11, spec §8] re-scoped T5 bar that is valid by construction: floor-judged
generalization (excess over the measured same-conditioning cross-city floor, worst-case across
held-out cities) + the Lane-M memorization discriminator.** Phases 7–9 below carry dated
revisions; the spec's §8 is authoritative where older task text conflicts.

**Architecture:** The **F16 sequencing spine is hard ordering**: Phase 1 cheap debt → Phase 2
F6 delivery (before ANY GPU — zero scored checkpoints exist; the prefix-scheme change is free
now) → Phase 3 EU-training path → Phase 4 F8 resume + F17 isolation (before any post-renewal
job) → Phase 5 reader-integrity → Phase 6 recalibrated gate + localization diagnostic →
Phase 7 expressivity wire-in (feature chosen BY the diagnostic; character-anchored per PI lock,
city-name floor kept separable) → Phase 8 re-derive (HALT-gate: PASS at δ=0.15 closes T5) →
Phase 9 decision layer. **No version-skewed partial fixes** — an artifact-changing fix applies
uniformly or defers whole.

**Tech Stack:** Python 3.11+, pytest, ruff (unpiped), pydantic, PyTorch Lightning (existing).
No new deps. Spec: `docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md`
(LOCKED, PI-resolved §5: δ=0.15; both-asymmetric character-anchored; #13 city-name shortcut;
T0 through the renewal gap).

---

## Cross-cutting discipline (EVERY task)

- **TDD:** failing test → run-FAIL → minimal impl → run-PASS → commit. `from __future__ import
  annotations`; type hints on public functions; `ruff format` + `ruff check` UNPIPED before
  every commit.
- **Gate 2 (pre-dispatch audit):** every task that edits an existing module READS the current
  module first and verifies the signatures this plan names still hold; corrections handed
  forward. Plan snippets drift (`feedback_reverify_plan_snippets_at_dispatch`).
- **Halt-gates:** a step marked **HALT-GATE** stops the plan on failure and routes to the
  reviewer. The implementer never improvises a fix.
- **Branch:** all work on `phase-2-readiness-closure` (created Task 0). Subagents: **no new
  branches, no push, no PR, no merge, no Leonardo.**
- **Leonardo-gated steps** are marked **[LEONARDO — GATED]**: build + test locally; the run
  itself waits for Umar's explicit word per step. Verified-end-state on every remote artifact
  (re-read, recompute, count real units).
- Reverse-locks: where a task changes behavior an existing test PINS (named per task), the test
  edit is part of the task's diff and called out in the commit message — deliberate flip, never
  silent weakening.

## File structure (created / modified)

**New:** `scripts/check_crs_consistency.py`, `scripts/measure_emergence_floor.py`,
`scripts/measure_cell_token_lengths.py`, `scripts/run_localization_diagnostic.py`,
`scripts/run_gate_i_rederive.py` (thin variant of the existing runner),
`configs/eval/emergence_floors.yaml`, `src/cfm/data/training/atomic_io.py`,
`tests/…` mirrors per task.
**Modified (Gate-2 read first):** `src/cfm/data/training/{conditioning,datamodule,build_shards}.py`,
`src/cfm/training/{config,resume}.py`, `scripts/train_scaffold.py`,
`src/cfm/eval/{conditioning_discrimination,slice_metrics,resolution}.py`,
`src/cfm/eval/holdout/labels.py`, `scripts/bakeoff_{diagnostic,run}.sbatch`,
`src/cfm/eval/conditioning_gate.py` (tombstone), `src/cfm/data/sub_f/decoder.py` (docstring),
two spec/plan docs (erratum edits).

---

## PHASE 0 — Branch + lock bookkeeping

### Task 0: Branch, commit spec+plan (orchestrator-only — subagents never branch)
- [ ] `git checkout -b phase-2-readiness-closure` from `main@77fdb6d`.
- [ ] `git add docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md docs/superpowers/plans/2026-06-10-readiness-closure-and-conditioning-enrichment.md && git commit -m "docs(readiness): locked spec + implementation plan"`
- [ ] Verify: `git log --oneline -1` shows the commit; `git status --porcelain` clean for these files.

---

## PHASE 1 — Cheap test debt + tombstones (T0; spec A-4, §2)

### Task 1: Tombstone the dead `conditioning_gate` twin + correct the two stale plan snippets
**Files:** Modify `src/cfm/eval/conditioning_gate.py`, `tests/eval/test_conditioning_gate.py`,
`docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md`,
`docs/superpowers/plans/2026-06-02-phase-2-bakeoff.md`.
- [ ] **Step 1 (failing test):** in `tests/eval/test_conditioning_gate.py`, REPLACE the existing
  behavior tests with one tombstone test:
```python
import pytest

def test_conditioning_gate_module_is_tombstoned():
    with pytest.raises(ImportError, match="superseded by cfm.eval.conditioning_discrimination"):
        import cfm.eval.conditioning_gate  # noqa: F401
```
- [ ] **Step 2:** run `uv run pytest tests/eval/test_conditioning_gate.py -v` → FAIL (module imports fine).
- [ ] **Step 3:** replace `conditioning_gate.py`'s body with a module-docstring + immediate
  `raise ImportError("cfm.eval.conditioning_gate is superseded by cfm.eval.conditioning_discrimination "
  "(the BH-corrected gate that actually ran 2026-06-10). Wiring this module is the F10 dead-twin hazard.")`.
- [ ] **Step 4:** run test → PASS; run `uv run pytest tests/eval -q` → no regressions.
- [ ] **Step 5:** edit BOTH plan docs: at each `conditioning_gate` import/reference, insert an
  erratum line: `> **ERRATUM 2026-06-10 (F10):** conditioning_gate is tombstoned; the live gate
  is cfm.eval.conditioning_discrimination. See readiness spec §A-4.` In the 2026-06-02 plan's
  Task-2 snippet that calls `feature_samples` on raw geoms, add: `> **ERRATUM 2026-06-10 (F4-C1d):
  feature_samples consumers MUST apply promote_building_rings first (or use the promoting
  wrapper from readiness Task 24).**`
- [ ] **Step 6:** ruff format+check (unpiped); commit `fix(audit): tombstone conditioning_gate dead twin + plan errata (F10, F4-C1d)`.

### Task 2: slice_eval promotion live-call test (closes F4-C1b)
**Files:** Test `tests/eval/test_slice_metrics.py` (append). **Gate-2:** read
`src/cfm/eval/slice_metrics.py::slice_eval` and `tests/eval/test_geometry_promote.py` fixtures
for a valid building-ring token-block + closed-ring geom fixture shape; reuse them.
- [ ] **Step 1 (failing-by-deletion test — non-vacuous by construction):**
```python
def test_slice_eval_promotes_building_rings_on_its_live_path():
    """Deleting the promote_building_rings call inside slice_eval must turn this RED.
    Feed a building-class closed-ring LineString (decoder contract: never Polygon);
    assert it is counted as a polygon by the building metrics."""
    blocks, geoms = _building_ring_fixture()  # reuse test_geometry_promote's fixture shape
    out = slice_eval(blocks=blocks, geoms=geoms, strata=[0] * len(blocks))
    assert out["n_polygons"] >= 1  # un-promoted, the ring stays LineString -> 0
```
- [ ] **Step 2:** run → PASS expected (the call exists). **Prove non-vacuity once:** locally
  comment out the `promote_building_rings` call, run → must FAIL, restore. Record in the
  implementer report ("red-on-divergence demonstrated by reversion").
- [ ] **Step 3:** ruff; commit `test(eval): slice_eval promotion live-call guard (F4-C1b)`.

### Task 3: Delta-spec §4 prior correction + closeout erratum + stale docstrings
**Files:** Modify `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md` (§4 PRIOR
paragraph), `reports/2026-06-10-gate-i-conditioning-discrimination-closeout.md` (one-line
erratum), `src/cfm/data/sub_f/decoder.py` (docstring), `src/cfm/data/sub_c/io.py` +
`src/cfm/data/sub_c/geom.py` (stale 3-type comments), `src/cfm/models/micro_ar.py` (n_cond doc).
- [ ] **Step 1:** delta-spec §4: replace the [PRIOR] paragraph with: `**[PRIOR — CORRECTED
  2026-06-10]** The original prior ("conditioned on city D means handed D's macro plans") was
  true only of the unread TrainingShard.tile_conditioning field; the running model received a
  constant value-agnostic slot prefix (readiness enumeration F6). Conditioning has TWO axes:
  DELIVERY (wired by readiness Phase 2) and EXPRESSIVITY (gate-(i) FAIL; enrichment per
  readiness Phase 7).` Closeout gets: `> Erratum 2026-06-10: "conditioning the model is handed"
  describes the data plan, not the model input — see readiness enumeration F6.`
- [ ] **Step 2:** `decode_feature` docstring: state Point-or-LineString-only + consumer-must-
  promote, citing `geometry.promote_building_rings`. micro_ar doc: `n_cond=conditioning_id_span()`
  (=512), not 8. sub_c comments: append Multi* codes 3–5.
- [ ] **Step 3:** `uv run pytest tests/data/sub_f tests/models -q` → no regressions (docstrings
  only). Commit `docs(audit): correct delta-spec §4 prior + stale contract docstrings (F6, F10)`.

### Task 4: Delete the dead `UNSCORED_V1_DIMENSIONS` frozenset (YAGNI) **(Gate-2 first)**
**Files:** Modify `src/cfm/eval/holdout/labels.py`, `tests/eval/holdout/test_labels_no_sg_constant.py`.
- [ ] **Step 1 (Gate-2):** confirm zero src consumers (`grep -rn UNSCORED_V1_DIMENSIONS src/`).
- [ ] **Step 2:** delete the frozenset; fold its content into the module docstring ("unscored in
  v1: region/admin_region, sub_c_morphology_class — see readiness spec §4.4"). Keep the
  STRUCTURAL half of the test (country/climate/era not TileLabels fields) — that one is the
  non-vacuous guard; delete only the membership assertions.
- [ ] **Step 3:** run the test file → PASS; full `tests/eval/holdout` → no regressions; commit
  `refactor(eval): drop dead UNSCORED_V1_DIMENSIONS; keep the structural SG-constant guard (F9)`.

---

## PHASE 2 — F6 delivery (T0; spec §3; BEFORE any GPU)

### Task 5: Disambiguate the `"region"` key collision (pre-wiring; F6 trap §3.2)
**Files:** Modify `src/cfm/data/training/build_shards.py::_tile_conditioning_dict`;
Test `tests/data/training/test_build_shards.py` (append). **Gate-2:** confirm
`tile_conditioning` still has zero readers (`grep -rn 'tile_conditioning\[' src/ scripts/`).
- [ ] **Step 1 (failing test):**
```python
def test_tile_conditioning_distinguishes_admin_region_from_city():
    d = _tile_conditioning_dict(_labels_fixture(admin_region=None))
    assert "admin_region" in d and "region" not in d  # the city lives on TrainingShard.region
    assert d["admin_region"] is None
```
- [ ] **Step 2:** run → FAIL (`region` key present). **Step 3:** rename the key to
  `admin_region` in `_tile_conditioning_dict` (writer-only change; zero readers verified).
- [ ] **Step 4:** run + full `tests/data/training -q` → PASS. **Step 5:** commit
  `fix(training): tile_conditioning key admin_region, never shadow the city name (F6 trap)`.

### Task 6: `conditioning_scheme` into config + checkpoint hparams + report (F16 killer)
**Files:** Modify `src/cfm/training/config.py`, `scripts/train_scaffold.py::_write_report`;
Test `tests/training/test_config.py`, `tests/training/test_lit_module.py` (append).
**Gate-2:** read `ScaffoldConfig` (verified 2026-06-10: release/region/seed/backbone/… fields)
and `ScaffoldLit.__init__`'s `save_hyperparameters(cfg.model_dump())`.
- [ ] **Step 1 (failing tests):**
```python
def test_config_carries_conditioning_scheme_default_value():
    assert ScaffoldConfig().conditioning_scheme == "value"  # flipped by this phase

def test_checkpoint_hparams_record_the_scheme(tmp_path):
    lit = ScaffoldLit(ScaffoldConfig(d_model=32, n_layers=1, n_heads=2))
    assert lit.hparams["conditioning_scheme"] == "value"
```
- [ ] **Step 2:** run → FAIL. **Step 3:** add `conditioning_scheme: str = "value"` (allowed:
  `{"slot","value"}` via pydantic validator) to `ScaffoldConfig`; `_write_report` prints it
  (it already dumps `cfg.model_dump()` — verify the line; if the report only dumps selected
  fields, add it explicitly). **Step 4:** PASS + suite. **Step 5:** commit
  `feat(training): conditioning_scheme tagged into config/checkpoint/report (F16)`.

### Task 7: Wire value-bearing prefixes — REPLACE, never augment (the core F6 fix)
**Files:** Modify `src/cfm/data/training/datamodule.py::flatten_shards_to_cells`;
Test `tests/training/test_datamodule.py`. **Gate-2:** read current `flatten_shards_to_cells`
(prefix at the `prefix = tuple(build_conditioning_prefix())` line) + `TrainingShard` fields +
`build_value_bearing_prefix` signature (verified 2026-06-10: kwargs population_density_bucket,
zoning_class, road_skeleton_class, cell_density_bucket, region, coastal_inland_river,
sub_c_morphology_class, seed).
- [ ] **Step 1 (failing test — the §3.3 red-before; FAILS on today's constant prefix):**
```python
def test_model_input_prefix_differs_across_differing_tile_conditioning():
    """THE F6 tooth: two shards with different tile_conditioning must produce different
    example prefixes. RED today (constant slot prefix), GREEN after the wire-in."""
    a = _shard_fixture(tile_conditioning={"population_density_bucket": 0, "zoning_class": 0,
        "road_skeleton_class": 1, "admin_region": None, "coastal_inland_river": 0,
        "sub_c_morphology_class": "Asian-megacity"})
    b = _shard_fixture(tile_conditioning={**a.tile_conditioning, "zoning_class": 1})
    ca = flatten_shards_to_cells([a], seed=7)[0]
    cb = flatten_shards_to_cells([b], seed=7)[0]
    assert ca.prefix_ids != cb.prefix_ids

def test_prefix_scheme_mutual_exclusivity():
    """No example may mix slot ids and value ids (id-space collision: slot block ==
    field-0 value block). All prefix ids must lie in the VALUE layout."""
    ex = flatten_shards_to_cells([_shard_fixture()], seed=7)[0]
    slot_block = set(range(CONDITIONING_ID_BASE, CONDITIONING_ID_BASE + 8))
    # Under the value scheme, field i's id lives in block i; only field 0's block
    # overlaps the slot block numerically — assert ids match build_value_bearing_prefix exactly.
    assert list(ex.prefix_ids) == build_value_bearing_prefix(**_expected_kwargs_for(_shard_fixture()))
```
- [ ] **Step 2:** run → FAIL (constant prefix). **Step 3 (the wire-in):** in
  `flatten_shards_to_cells`, per shard+cell build the prefix via `build_value_bearing_prefix(
  population_density_bucket=tc["population_density_bucket"], zoning_class=tc["zoning_class"],
  road_skeleton_class=tc["road_skeleton_class"], cell_density_bucket=cell.cell_density_bucket,
  region=tc["admin_region"], coastal_inland_river=tc["coastal_inland_river"],
  sub_c_morphology_class=tc["sub_c_morphology_class"], seed=seed)` — **explicit field list,
  never `**tile_conditioning`** (spec §4.4: the SG-wrong constants must not ride in by dict-splat;
  morphology passes through today as a constant and is replaced at Task 24).
- [ ] **Step 4 (the deliberate reverse-lock flips — named test edits, same commit):**
  `test_conditioning_prefix_is_the_field_slot_id_block` and the flatten-prefix slot assertion
  flip to assert the VALUE layout (keep a renamed slot-builder unit test for the legacy builder).
- [ ] **Step 5:** run task tests + `uv run pytest tests/training tests/data/training -q` → PASS.
- [ ] **Step 6:** commit `feat(training): value-bearing conditioning prefix replaces slot prefix
  (F6 delivery; deliberate flip of 2 locking tests)`.

### Task 8: Generation-side conditioning + real strata (kills `strata.append(0)`)
**Files:** Modify `scripts/train_scaffold.py::_generate_and_score`; Test
`tests/training/test_generate_and_score_conditioning.py` (new). **Gate-2:** read
`_generate_and_score` (prefix + strata lines; eval-cell loop) and `generate_cell_tokens` signature.
- [ ] **Step 1 (failing test):** with a stub model, assert (a) each generated cell's prefix ==
  `build_value_bearing_prefix(...)` of the conditioning handed for that cell (matched-conditioning
  generation: conditioning sampled from the region's real tiles via `read_tile_labels` +
  `_cell_density_by_cell` — the eval generates UNDER real conditioning, scored per stratum);
  (b) `strata` == the handed cell-density buckets, not `[0,...]`.
- [ ] **Step 2:** FAIL. **Step 3:** implement: `_generate_and_score(cfg, model, dm)` draws
  per-cell conditioning from the datamodule's loaded shards (`dm` already holds them — no new
  IO), builds the value prefix per cell, passes the real `cell_density_bucket` to `strata`.
- [ ] **Step 4:** PASS + `tests/slow/test_e2e_scaffold.py` run EXPLICITLY (it is deselected by
  default — run it by path; loop-closure must survive). **Step 5:** commit
  `feat(eval): matched-conditioning generation + real strata in _generate_and_score (F6)`.

### Task 9: Delivery hygiene — injectivity, live-shape tests, exclusivity guard
**Files:** Test `tests/data/training/test_conditioning_value_bearing.py`,
`tests/models/test_micro_ar.py` (append). **Gate-2:** read `_value_bucket` (SHA-256 % 63).
- [ ] **Step 1 (failing/new tests):** (a) `_value_bucket` injectivity over the LIVE value sets:
  all 4-bucket int fields trivially injective; for strings assert the CURRENT live set
  ({None, "Asian-megacity"}) maps collision-free AND add the documented expectation test:
  38 train-city names → record collision count, assert == the precomputed value (pin it; if a
  future change makes city names value-bearing through `_value_bucket`, the pinned number forces
  the implementer to confront aliasing — content-anchored, not silent); (b) micro_ar prefix-mask
  invariants re-run at `n_cond=conditioning_id_span()` (512), the live shape; (c) a
  collate-layer guard: every batch's prefix ids ∈ value layout (no slot id 1508..1515 pattern
  as a CONSTANT row — assert at least one prefix differs across a 2-tile batch fixture).
- [ ] **Step 2-4:** FAIL→impl(test-only task: fixtures)→PASS. **Step 5:** commit
  `test(training): value-prefix injectivity + live-shape n_cond=512 + exclusivity guards (F6/F16)`.

---

## PHASE 3 — EU-training path (T0; spec A-1)

### Task 10: `--region`/`--release` CLI + sbatch de-Singaporization + sbatch content test
**Files:** Modify `scripts/train_scaffold.py` (argparse + overrides), `scripts/bakeoff_diagnostic.sbatch`,
`scripts/bakeoff_run.sbatch`; Test `tests/training/test_cli_contract.py` (append).
**Gate-2:** read `main()`'s overrides dict; the sbatch preamble lines.
- [ ] **Step 1 (failing tests):** (a) CLI: `--region krakow --release 2026-04-15.0` reaches
  `ScaffoldConfig` (parse-only test via a `build_config_from_args` refactor — extract pure
  function from `main`); (b) **sbatch content test**: read both sbatch files as text; assert
  NO `'singapore'` literal; assert the pre-build line uses `"$REGION"`/`${TRAIN_SET}`; assert
  `: "${REGION:?set REGION}"` guards exist.
- [ ] **Step 2:** FAIL. **Step 3:** add flags + env-driven sbatches (`REGION` env; preamble
  `python -c "...build_training_shards('${RELEASE}','${REGION}')"` → replaced in Task 11 for the
  union). **Step 4:** PASS + suite. **Step 5:** commit
  `feat(training): region/release CLI + env-driven sbatches + content guard (F1/F7)`.

### Task 11: Production caller for the multi-region union (the 38 manifests get a consumer)
**Files:** Modify `scripts/train_scaffold.py::_datamodule`, `src/cfm/training/config.py`
(`train_set: str = "single"` | `"eu-train-union"`); Test `tests/training/test_datamodule_union_path.py` (new).
**Gate-2:** read `CellDataModule(training_manifests=[…])` union mode + `train_cities` signature
(`build_shards.py:90`) + the Task-8 report's city list.
- [ ] **Step 1 (failing test):** `_datamodule(cfg)` with `train_set="eu-train-union"` constructs
  `CellDataModule` with `training_manifests` = the per-city manifest paths for
  `train_cities(release, g4_rollup=…, holdout_manifest=…)` (synthetic 2-city fixture), with
  `expected_holdout_schema="2.0"`; held-out cities absent.
- [ ] **Step 2-4:** FAIL→impl→PASS (+ assert the strict `holdout["held_out_cities"]` read — the
  fail-closed caller pattern, never `.get`). **Step 5:** commit
  `feat(training): eu-train-union datamodule path — Task-8 manifests get their consumer (F7)`.

### Task 12: Three-way CRS consistency check + meters assertion
**Files:** Create `scripts/check_crs_consistency.py`; Test `tests/scripts/test_check_crs_consistency.py`.
**Gate-2:** read `epsg_label_for_region`, sub-D `manifest.yaml` keys (`region_crs`, `tiles[]`),
multiregion holdout manifest per-region `crs`.
- [ ] **Step 1 (failing tests, synthetic tmp_path tree):** the checker (a) for each region:
  config `projected_crs` == sub-D `region_crs` == on-disk `tile=EPSG…` dir labels == holdout
  manifest per-region `crs` where present; (b) asserts the CRS is projected/meters (pin: label
  matches `^EPSG\d+$` AND `projected_crs` in an allowlist derived from the 42 configs — plus a
  pyproj-free unit heuristic: reject 4326-class codes); (c) exit nonzero + named per-city diff
  on mismatch; (d) `--report` writes YAML with per-city verdicts.
- [ ] **Step 2-4:** FAIL→impl→PASS. Run LOCALLY for singapore (real data) → PASS expected.
- [ ] **Step 5 [LEONARDO — GATED]:** run over all 42 cities on Leonardo CPU; verified-end-state:
  re-read the report YAML, count per-city verdicts == 42. **HALT-GATE** on any mismatch.
- [ ] **Step 6:** commit `feat(eval): three-way CRS consistency checker + meters guard (F2)`.

### Task 13: Emergence floors as provenance-bearing artifact (kills the 1.96 literal; F13/F15)
**Files:** Create `configs/eval/emergence_floors.yaml`, `scripts/measure_emergence_floor.py`;
Modify `scripts/train_scaffold.py` (floor resolution + loud-if-absent), `src/cfm/eval/slice_metrics.py`
(record floor provenance + denominator convention in the metrics dict); Tests
`tests/training/test_emergence_floor_resolution.py`, `tests/scripts/test_measure_emergence_floor.py`.
**Gate-2:** read `slice_eval` emergence params + `holdout_polygons_per_active_cell`.
- [ ] **Step 1 (failing tests):** (a) floor resolution: a cell-generating run (`eval_cells>0`)
  with NO floor entry for `cfg.region` **raises** (fail-open closed; the old `--emergence-floor`
  flag is removed — reverse-lock: `test_slice_eval_verdict_absent_when_no_emergence_inputs`
  flips to assert the RAISE at the scaffold layer while `slice_eval` keeps the None-tolerant
  library behavior for non-generating callers); (b) the YAML schema requires per-region:
  `floor`, `holdout_density`, `frac` (0.25), `derived_at` (sha), `derivation_regime:
  {cell_length: full, denominator: all_nonempty_cells}`; (c) the measure script computes density
  via `holdout_polygons_per_active_cell(release, region)` and writes the entry (synthetic test).
- [ ] **Step 2-4:** FAIL→impl→PASS. Seed `singapore: floor 1.96, holdout_density 7.85` with
  provenance `hand-carried from 2026-06 scaffold; re-derive on next SG run`.
- [ ] **Step 5 [LEONARDO — GATED]:** run the measure script for the 4 EU held-out cities (CPU);
  verified-end-state: re-read YAML, floors present+noisy (rough-numbers check), commit artifact.
- [ ] **Step 6:** commit `feat(eval): per-region emergence floors with provenance; CLI literal dies (F13/F15)`.

### Task 14: F15 commensurability — lengths into config/report + refuse-verdict rule
**Files:** Modify `src/cfm/training/config.py` (`eval_cells`, `eval_max_new` become config
fields), `scripts/train_scaffold.py` (plumb from config; CLI flags map to config),
`src/cfm/eval/slice_metrics.py`; Test `tests/eval/test_commensurability.py` (new).
- [ ] **Step 1 (failing tests):** (a) `ScaffoldConfig().eval_max_new` exists; report/hparams
  carry it (F9 reproducibility); (b) **the refuse rule:** `slice_eval(..., emergence_floor_per_cell=f,
  generated_length_cap=L, floor_regime_cell_length=R)` REFUSES the §2 verdict (returns
  `emergence_verdict="INCOMMENSURATE"` — loud, distinct from None) when `L < R` (a 512-cap
  generation cannot be scored against a 5760-regime floor); (c) metrics dict records both
  lengths + the floor's denominator convention.
- [ ] **Step 2-4:** FAIL→impl→PASS (diagnostic sbatch sets eval_max_new ≥ the floor regime or
  the verdict is INCOMMENSURATE by construction). **Step 5:** commit
  `feat(eval): length-commensurability guard on the emergence verdict (F15)`.

### Task 15: EU cell-token-length stats + drop-rate action contract
**Files:** Create `scripts/measure_cell_token_lengths.py`; Modify
`src/cfm/data/training/datamodule.py::flatten_shards_to_cells` + `CellDataModule.setup`;
Tests `tests/scripts/test_measure_cell_token_lengths.py`, `tests/training/test_datamodule.py` (append).
- [ ] **Step 1 (failing tests):** (a) the script scans sub-F `cells.parquet` token lengths per
  city → YAML (p50/p99/p99.9/max, frac>5760) (synthetic fixture); (b) **action contract:**
  `setup` raises `DropRateExceeded` when `too_long/total > 0.005` (0.5% — 5× the SG P99.9
  design point) with the named escalation in the message ("raise DEFAULT_MAX_CELL_TOKENS via a
  recorded decision or re-chunk; see readiness F13"); the `dropped` dict is no longer discarded.
- [ ] **Step 2-4:** FAIL→impl→PASS. **Step 5 [LEONARDO — GATED]:** run the script over the 38
  train cities; verified-end-state re-read; **HALT-GATE** if any city's frac>5760 exceeds 0.5%.
- [ ] **Step 6:** commit `feat(data): EU token-length measurement + drop-rate action contract (F13)`.

---

## PHASE 4 — F8 resume + F17 isolation (T0; spec A-2; before any post-renewal job)

### Task 16: Atomic training-manifest writes
**Files:** Create `src/cfm/data/training/atomic_io.py`; Modify `build_shards.py:284` region;
Test `tests/data/training/test_atomic_io.py`. **Gate-2:** read the sub-C crash-safe writer
(`src/cfm/data/sub_c/` — locate `write`-guard used by guarded_rederive) and mirror its shape.
- [ ] **Step 1 (failing test):** `atomic_write_text(path, text)` writes `path.with_suffix(".tmp")`
  then `os.replace`; test asserts (a) content lands; (b) on injected failure between write and
  replace, the original file is untouched (no torn state); (c) `_write_training_manifest` uses it
  (monkeypatch `os.replace` to observe).
- [ ] **Step 2-4:** FAIL→impl→PASS. **Step 5:** commit `fix(data): atomic manifest writes (F17)`.

### Task 17: Run-keyed reports + per-run dirs
**Files:** Modify `scripts/train_scaffold.py::_write_report` (+ checkpoint/log dirpath wiring in
`run_short`/`build_trainer` call); Test `tests/training/test_report_naming.py` (new).
**Gate-2:** read `_write_report:174-181` + `build_trainer` signature.
- [ ] **Step 1 (failing test):** report path contains `{backbone}-{params}M-{seed}`; two configs
  differing only in backbone produce different paths; checkpoint dirpath ==
  `work_checkpoint_dir(backbone, scale_label)` (from `cfm.training.resume`).
- [ ] **Step 2-4:** FAIL→impl→PASS. **Step 5:** commit
  `fix(training): run-keyed report + checkpoint dirs — no cross-run overwrite (F17)`.

### Task 18: Wire across-job resume (the F8 core)
**Files:** Modify `scripts/train_scaffold.py::run_short`, `src/cfm/training/train.py::build_trainer`
(checkpoint `dirpath`); Test `tests/training/test_resume_wiring.py` (new).
**Gate-2:** read `resume.py::{work_checkpoint_dir, resume_ckpt_path}` and `run_short`.
- [ ] **Step 1 (failing test):** with `WORK=tmp_path` env: (a) `run_short` passes
  `ckpt_path=resume_ckpt_path(backbone, scale)` to `trainer.fit` when a `last.ckpt` exists in
  `work_checkpoint_dir(...)` and `None` otherwise (stub trainer records the kwarg); (b) the
  ModelCheckpoint `dirpath` IS the `$WORK` dir.
- [ ] **Step 2-4:** FAIL→impl→PASS; run `uv run pytest tests/training -q` full.
- [ ] **Step 5 [LEONARDO — GATED, the plan's own HARD ORDERING]:** short-job proof:
  start → kill → resubmit → **continues from last.ckpt, not step 0** (compare `trained_steps` +
  global_step in the resumed checkpoint; verified-end-state re-read of both checkpoints' steps).
  **HALT-GATE** on restart-from-0. Runs only on Umar's word, post-renewal.
- [ ] **Step 6:** commit `feat(training): across-job $WORK resume wired into run_short (F8)`.

### Task 19: sbatch trap + end-state markers
**Files:** Modify `scripts/bakeoff_run.sbatch`, `scripts/bakeoff_diagnostic.sbatch`; Test
extend `tests/training/test_cli_contract.py` sbatch-content assertions.
- [ ] **Step 1 (failing content test):** assert the run sbatch (a) traps USR1 → `kill -USR1 $SRUN_PID`
  (signal forwarded to ranks) then `JID=$(sbatch --export=ALL "$0") && [[ -n "$JID" ]] || exit 1`
  (resubmit VERIFIED — no exit-0-on-failure); (b) `JOB_DONE` is printed only after an end-state
  block: `test -f` the expected checkpoint AND report paths (region/backbone/seed-keyed from
  Task 17), else `JOB_FAILED_ENDSTATE` + exit 1; diagnostic sbatch loses the `|| echo "(report
  not found)"` masking.
- [ ] **Step 2-4:** FAIL→edit sbatches→PASS. **Step 5:** commit
  `fix(orchestration): USR1 forward + verified resubmit + end-state markers (F8)`.

---

## PHASE 5 — Reader-side integrity (T1; spec A-2/F9 + gate-(i) counters)

### Task 20: sha + lock-marker verification at read; TRAIN_TOKENS guard
**Files:** Modify `src/cfm/data/training/holdout_guard.py::run_holdout_audit` (verify
`manifest_sha256` of the loaded holdout manifest; require `_EVAL_SET_LOCKED` present beside it),
`src/cfm/eval/resolution.py` (region-aware `marker_path` selection helper); Tests
`tests/training/test_holdout_guard.py` (append), `tests/eval/test_resolution_seam.py` (append),
`tests/eval/test_ladder.py` (append). **Gate-2:** read `manifest.py::manifest_sha256` grammar.
- [ ] **Step 1 (failing tests):** (a) a holdout manifest whose recomputed sha ≠ its
  `manifest_sha256` field → `HoldoutLeakError` at `run_holdout_audit`; (b) missing
  `_EVAL_SET_LOCKED` beside the manifest → raise; (c) **TRAIN_TOKENS guard:**
  `ladder.TRAIN_TOKENS == yaml(multiregion _EVAL_SET_LOCKED)["train_tokens"]` (real artifact,
  default suite — the idiom `test_reads_the_real_frozen_marker_fields` already uses);
  (d) `assert_resolution_sufficient(..., region=...)` selects SG vs EU marker; EU marker
  missing KS fields → loud KeyError (existing behavior, now reachable + tested).
- [ ] **Step 2-4:** FAIL→impl→PASS (+ full suite — the planted-leak realrun tests must stay green).
- [ ] **Step 5:** commit `feat(eval): reader-side sha/lock verification + TRAIN_TOKENS guard (F9)`.

### Task 21: Gate-(i) extraction counters + CRS regression test
**Files:** Modify `src/cfm/eval/conditioning_discrimination.py` (per-city
`n_tiles_expected/read/skipped` into the result dataclass + report YAML); Test
`tests/eval/test_conditioning_discrimination.py` (append).
- [ ] **Step 1 (failing tests):** (a) synthetic 3-tile city with 1 missing `cells.parquet` →
  result carries `n_tiles_expected=3, n_tiles_read=2, n_tiles_skipped=1`; **HALT-threshold:**
  `extract…` raises if any city's skipped/expected > 0.1 (the silent-shrinkage ceiling — a
  partial city can no longer quietly thin); (b) the extraction-site CRS regression test (the
  4th `tile_dirname` call site): monkeypatch capture, assert region label used, RED if the
  signature default rides (same shape as `test_build_shards_in_memory_uses_region_crs_label`).
- [ ] **Step 2-4:** FAIL→impl→PASS. **Step 5:** commit
  `feat(eval): gate-(i) tile-coverage counters + extraction CRS guard (F3/F1)`.

---

## PHASE 6 — Recalibrated gate + localization diagnostic (spec §4.2/§4.3)

### Task 22: Recalibrated verdict — δ=0.15 effect floor + bref exclusion (construction identity)
**Files:** Modify `src/cfm/eval/conditioning_discrimination.py` (verdict: `effect_size_floor`
param; extraction: exclude outbound-bref features from `road_length_m` BY `_has_outbound_bref`
+ count them); Test `tests/eval/test_conditioning_discrimination.py` (append).
**Gate-2:** read `_tile_features` + `sub_g.seam_decodability._has_outbound_bref` import path.
- [ ] **Step 1 (failing tests — the gate must distinguish regimes):**
```python
def test_recalibrated_gate_CAN_pass_at_real_n():
    """Huge-n tiny-shift (KS≈0.03 << δ=0.15) is BH-significant but NOT a FAIL — the δ floor
    makes PASS reachable at real sample sizes (the structural-incapacity fix)."""
def test_recalibrated_gate_still_FAILS_on_large_effects():
    """KS≈0.4 at modest n → FAIL survives the recalibration (glasgow-vs-krakow regime)."""
def test_bref_features_excluded_from_road_length_by_construction_identity_and_counted():
    """Outbound-bref road excluded + counted (n_bref_excluded per city); a zero-length
    geometry WITHOUT the bref identity is NOT excluded (regime-distinguishing twin —
    symptom-keyed exclusion would pass this; identity-keyed must)."""
```
- [ ] **Step 2:** FAIL ×3. **Step 3:** implement (`significant = (adj < alpha) and (ks >= effect_size_floor)`;
  default `effect_size_floor=0.15` at the runner layer, explicit param in the pure verdict;
  report both `n_significant_raw_bh` and `n_significant_effect` — the δ's effect visible).
- [ ] **Step 4:** PASS + full eval suite. **Step 5:** commit
  `feat(eval): gate-(i) recalibration — δ=0.15 effect floor + bref exclusion-by-identity (PI-call #1, F4-C2)`.

### Task 23: Localization diagnostic (CPU; characterize-before-recommend)
**Files:** Create `scripts/run_localization_diagnostic.py`; Test
`tests/scripts/test_localization_diagnostic.py` (synthetic). **Gate-2:** read
`extract_features_by_city_stratum_metric` + `read_macro_core_parquet` (per-cell rows) +
`derivation_evidence.parquet` schema in sub-D.
- [ ] **Step 1 (failing tests, synthetic 2-city fixture):** the script computes the recalibrated
  verdict under stratum VARIANTS, one layer at a time: V0 baseline (tile-dominant/modal, 4-bucket)
  — must reproduce gate-(i) shape; V1 un-collapse (per-cell zoning + per-cell density as the
  stratum, features assigned to their cell's stratum); V2 un-quantize (8/16-bucket density from
  raw `building_footprint_ratio` in `derivation_evidence.parquet`); V3 candidate dims (per-cell
  `sea_water_fraction` bucket). Output YAML: per variant × metric → n_pairs,
  n_significant_effect(δ=0.15), median KS — **the variant that kills the most discrimination
  signal localizes where city character lives.**
- [ ] **Step 2-4:** FAIL→impl→PASS locally on synthetic.
- [ ] **Step 5 [LEONARDO — GATED]:** run on the 4 held-out cities (CPU, zero GPU). Verified-end-
  state: re-read YAML; rough-numbers check; per-variant n within ±20% of V0's (denominator sanity).
- [ ] **Step 6 — HALT-GATE (PI review):** bring the variant table to Umar; **the character
  feature for Task 24 is chosen HERE, by the data, with Umar's word** (δ↔character coupling:
  selection criterion = largest drop in n_significant_effect at δ=0.15).
- [ ] **Step 7:** commit `feat(eval): coarseness-localization diagnostic (F5; spec §4.2)`.

---

## PHASE 7 — Expressivity wire-in (after Task 23's PI gate)

### Task 24: City-name floor + character feature into the conditioning vector — SEPARABLE
> **[REVISED 2026-06-11, spec §8 — Task 24 SPLITS:]**
> **24a (identity floor + separability)** survives ~verbatim from the steps below: `city_identity`
> value-bearing from `TrainingShard.region`, the `conditioning_ablation` switch
> (`"full"/"no_city"/"no_character"`), constant-column + all-None guards, vocab append-only
> checks. Lane S is computed under `"no_city"` — the switch is the scored lane's instrument.
> **24b (character)** is RE-PARAMETERIZED: not a bucketed `character_<chosen>` field (every
> candidate eliminated by the diagnostic — V1b/V2/V3/V4) but the **continuous distributional
> carrier via `macro_tokens`** (per-cell building log-median / log-IQR / p90-p50 / count +
> road median length; spec §8). Bigger than the original task: gets its OWN mini-spec at
> dispatch (Gate-2 read of the empty carrier; quantization/schema; wholesale shard rebuild
> under the uniform-defect rule; NO sub-C regen). The bucketed-character steps below apply to
> 24a's city field only.
**Files:** Modify `src/cfm/data/training/conditioning.py` (append-only: 2 new `_CONDITIONING_FIELDS`
entries — `city_identity`, `character_<chosen>` — extending `conditioning_id_span`),
`build_shards.py::_tile_conditioning_dict` (+ the per-cell carrier if the diagnostic picks a
per-cell feature: `CellPayload` gains the field, mirrored from `_cell_density_by_cell`'s shape),
`datamodule.py::flatten_shards_to_cells` (explicit-field wiring extended); Tests across the three.
> **Intentionally parameterized:** the CHARACTER field's derivation is the Task-23 output —
> its steps are finalized at dispatch with Umar's chosen feature (flagged per plan-discipline;
> not a silent TODO). The CITY field is fully specified now.
- [ ] **Step 1 (failing tests):** (a) `city_identity` value-bearing from `TrainingShard.region`
  (the city name — zero regen; #13 shortcut per PI-call #3); (b) **separability tooth:** the two
  new fields occupy DISTINCT id blocks; a config switch
  (`conditioning_ablation: {"full","no_city","no_character"}` on `ScaffoldConfig`) zeroes one
  block (bucket-0) without touching the other — the bake-off can show character does the work
  (PI-call #2's locked rationale); (c) **constant-column guard:** building shards across ≥2
  cities where `character_*` is constant → loud error (kills the #22 class structurally);
  (d) **all-None guard:** a region whose `city_identity`/character column is all-None → loud.
- [ ] **Step 2-4:** FAIL→impl→PASS (+ vocab append-only check: `CONDITIONING_ID_BASE` unmoved;
  span grows by 2×64; MASK_ID test from Phase-2 must stay green — it computes from the live span).
- [ ] **Step 5:** commit `feat(conditioning): city-identity floor + diagnostic-chosen character
  field, separable by construction (F5; PI-calls #2/#3)`.

---

## PHASE 8 — Floor-artifact production (T5 closes as re-scoped; revised 2026-06-11)

### Task 25 [LEONARDO — GATED]: conditioning-floor artifact production (re-scoped 2026-06-11)
> **[REVISED 2026-06-11, spec §8 — HALT semantics replaced.]** No PASS/FAIL on cities-differ.
**Files:** Create the floor-artifact runner (the Task-22 measurement machinery + Task-21
counters + bref exclusion); artifact + report `reports/2026-06-XX-conditioning-floor-*.yaml|.md`.
- [ ] **Step 1:** local synthetic test of the runner wiring — floor table computed; integrity
  halts reachable BOTH directions (floor-collapse < 0.049 fixture; floor-explosion > 0.5
  fixture; UNSUPPORTED); sha-stamp + write-once grammar (mirrors Task-20 / `_EVAL_SET_LOCKED`).
- [ ] **Step 2 [GATED]:** run on Leonardo CPU. (a) 4 held-out cities → per-(city-pair, metric,
  stratum) real-real KS table + `floor_D = min_T KS(real_D, real_T)` (median-over-T as
  context) + δ ladder. (b) the 38 training cities' per-stratum real distributions (Lane-M
  inputs; ALL 38, PI knob 2). Verified-end-state: re-read YAML; every city
  `n_tiles_skipped/expected ≤ 0.1`; `n_bref_excluded` noisy-nonzero; per-pair table internally
  consistent; sanity halts clean.
- [ ] **Step 3:** **floor artifact written + sha-stamped + sanity clean ⇒ T5 CLOSES as
  re-scoped** (the bar is valid by construction). Report committed; PRD §9 already revised
  with this re-scope (PRD §6/§10 re-checked 2026-06-11 — no edit needed). Any anomaly ⇒ halt
  to Umar, never improvised.

---

## PHASE 9 — Decision layer (T1; spec A-3; before any scored run)

### Task 26: Task-12 runner obligations, tests-first (paired teeth)
**Files:** Create `src/cfm/eval/bakeoff_decision.py` + `scripts/run_bakeoff_decision.py`;
Modify `src/cfm/eval/realism.py::feature_samples` (promotes internally — resolves the F4-C1d
fork structurally: the helper becomes safe for every future caller), `src/cfm/eval/feature_resolution.py`
(constant unified to 1.358 == conditioning_discrimination's? **Gate-2 decides**: read both
derivations; pick the exact two-sample KS α=0.05 coefficient, change the OTHER, cross-guard
test pins them equal), `scripts/train_scaffold.py` (--config loader → resolves the xfail),
`src/cfm/eval/slice_metrics.py` (Point semantics pinned: Points excluded from the OGC
denominator + counted as `n_points`), compile-OUTCOME into the report; sbatch preamble
buildability dry-run (`python -c "from cfm.models.backbone import build_backbone; ..."`).
> **[REVISED 2026-06-11, spec §8]** The decision quantity is **per-city excess-over-floor**
> (Lane S; floor artifact is a sha-verified input — the runner REFUSES an absent/mismatched
> sha/lock, mirroring Task-20), aggregated worst-case with the binding-city power gate.
> `pick_winner` requires `memorization_check_ok` (Lane M, all 38 training cities) PAIRED with
> `structural_check_ok`. Lane-M must-fire pair is part of Step 1: a synthetic regurgitator
> fixture (generated := training city T's real samples) MUST FAIL Lane M; an oracle fixture
> (generated := D's own held-out samples) MUST PASS both lanes. No-leakage pin: the
> discriminating-strata selection provably reads only real data.
- [ ] **Step 1 (failing tests):** (a) `decide(...)` asserts path == persisted basis (YAML in,
  enum compared); (b) 4-city completeness: `set(real_by_city) == frozenset(held_out_cities from
  the manifest)` else loud; (c) `pick_winner` requires ≥2 backbones AND `structural_check_ok`
  per fit (the pairing test: garbage non-monotone fit → no crowning); (d) `feature_samples`
  promotion: building-ring fixture → polygon sample (red on today's non-promoting helper);
  (e) `--config` round-trip (xfail flips to pass — named reverse-lock); (f) Point-semantics
  pin; (g) constants cross-guard; **(h) [2026-06-11] the Lane-M must-fire pair + floor-sha
  refusal + no-leakage pin + excess-over-floor quantity + `memorization_check_ok` pairing
  (per the revision note above).**
- [ ] **Step 2-4:** FAIL→impl→PASS + full suite. **Step 5:** commit
  `feat(eval): bake-off decision runner with paired obligations (F11/F4/F12; Task-12 surface)`.

---

## Execution + merge discipline

Task order is the spine; within a phase tasks are sequential. After every phase: full suite
(`uv run pytest -q`, unpiped) + a one-paragraph phase report to Umar. **Merge to `main` only at
sub-project end, on Umar's word, `--no-ff`, suite green, `reports/` summary written.** All
[LEONARDO — GATED] steps wait for Umar's explicit per-step word; nothing on GPU anywhere in this
plan (Task 25/12/13/15/23 Leonardo steps are CPU; Task 18's proof is the first post-renewal GPU
job and is additionally gated on T0 closure per PI-call #4).

## Self-review

**Spec coverage:** A-1→Tasks 10–15; A-2→16–20; A-3→26; A-4→1–4; A-5 deferred-with-triggers
(unchanged, carried in spec §1); §2 delta-spec correction→Task 3; §3 delivery→Tasks 5–9 (3.1
mechanism→7, scheme tag→6, traps 3.2→5/7/9, teeth 3.3→7, §3.4 honored by Task 24's data);
§4.2 diagnostic→23; §4.3 recalibration→22 (δ=0.15 locked); §4.4 both-asymmetric+separable→24;
§4.5 teeth→22/24/25; §7 protocol patterns→named per task (regime-distinguishing guards in
22/24; construction-identity exclusion in 22; threshold pairing in 22's raw-vs-effect counts +
13's provenance; correct-unit in 23's per-variant n checks).
**Placeholder scan:** Task 24's character-field steps are intentionally parameterized on Task
23's PI-gated output — flagged, not silent; all other steps carry code/assertions or exact
content directives with Gate-2 reads.
**Type consistency:** `build_value_bearing_prefix` kwargs match the verified live signature
(incl. `region=` taking the admin value — Task 5's rename feeds `tc["admin_region"]`);
`conditioning_scheme`/`train_set`/`conditioning_ablation` are ScaffoldConfig fields introduced
before use (Tasks 6/11/24); `work_checkpoint_dir/resume_ckpt_path` per verified `resume.py`.
**Known tension recorded:** Task 24 appends 2 conditioning fields ⇒ `conditioning_id_span`
grows ⇒ embedding rows grow ⇒ any Phase-2-trained checkpoint is shape-incompatible — acceptable
because NO checkpoint that matters exists before Phase 8 (F16 sequencing holds it: Tasks 5–9 and
24 all land before the first real training job); the scheme/shape tag (Task 6) makes any
violation loud.
