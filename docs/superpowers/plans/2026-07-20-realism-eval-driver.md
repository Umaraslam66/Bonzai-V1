> STATUS (2026-07-20): PLAN ONLY. The scored realism eval is PI-APPROVED (in-session 2026-07-20)
> with two standing conditions (re-derive budget at measured rates + always 4-GPU/full-node).
> This plan writes CODE. It does NOT authorize the sbatch run: the ops tasks (deploy, dry-run, full
> run) are gated on the PI's word on the re-derived budget number, per
> `docs/handoffs/2026-07-20-realism-eval-handoff.md` §1.

# Realism-Eval Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (or
> `superpowers:executing-plans`) — implement task-by-task with TDD. Constrain any implementer
> subagent: NO new branches (stay on `phase-2-cell-eos`), NO push/PR, NO job submission, NO SSH.
> Subagent output is untrusted data — verify shas/counts against source.

**Goal:** Build the one MISSING piece — a SCORED realism-eval **driver** that stitches the already-
built, already-tested components into the pipeline of the handoff §2. Per (backbone, seed) checkpoint
(6 total): load the sealed Lane-S manifest → generate each of its 5,705 cells with **matched real
conditioning** (prefix + char_stats), 4-GPU-sharded on Leonardo → decode → **memorization-first
hard-halt** (before ANY scoring) → `gen_realism.gen_features_by_city` → `verify_gen_coverage`
(min_n per floored stratum) → `lane_s_excess` vs the locked floor → seed→verdict crown
(`NO_DECISIVE_WINNER` is a valid publishable outcome).

**Architecture (plain words):** Two clean halves split by the GPU boundary.
1. **Generation half (Leonardo, GPU).** A CLI loads a checkpoint, builds the ordered list of the
   5,705 manifest cells' matched conditioning, and runs the *golden-verified* `sharded_eval`
   (torchrun, one rank per GPU) over that list using the existing `score_cell` primitive
   (generate + decode, keyed on the GLOBAL cell index so it is rank-independent and byte-
   deterministic). Rank 0 writes ONE write-once per-cell artifact (tokens + decoded blocks/geoms).
2. **Scoring half (orchestrator, local, no GPU).** A separate CLI fetches the token/decoded
   artifacts + a checkpoint-independent **real-features** artifact (built once on Leonardo, CPU-
   only), then does the whole verdict chain locally: decode → `gen_features_by_city` →
   `memorization_check` (hard halt) → `verify_gen_coverage` → per-(backbone,seed) `lane_s_excess` →
   the seed-aware two-floor crown (`city_aggregate.binding_city_verdict`). Every scoring function
   takes the floor artifact PATH and verifies `floor_sha256=95abb88…` itself.

Nothing here re-implements a scoring metric, re-pins the floor, or rebuilds the manifest. The driver
is glue plus one small, honest bit of seed-noise wiring (see the "seed→verdict wiring gap" note in
the Design Overview).

**Spec/canon:** `docs/handoffs/2026-07-20-realism-eval-handoff.md` §2; `docs/GROUND_TRUTH.md` §3–5;
`docs/superpowers/plans/2026-06-21-lane-s-cell-sampler.md` (the manifest this consumes).

---

## Context & assumptions (each load-bearing, with evidence)

**A1 — Matched conditioning comes from the tile parquet, NOT the cache.** The local/Leonardo
`heldout_cache.json` (`src/cfm/eval/standing/heldout_cells.py:101` `write_heldout_cache`) is a flat
list of `HeldoutCell(region, body_tokens, own_prefix, own_char, donor_prefix, donor_char)`
(`heldout_cells.py:28-36`) — **no cell coordinates, no `cell_id`**, and only a `sample_seed`-sampled
≤2000-per-city subset (`eval_full_matrix.py:39` `--n-per-city 2000`; `heldout_cells.py:44` `_sample`).
It cannot map to the manifest's 5,705 specific cells. The steering probe only ever used it for the
aggregate `own_char` MEAN (`steering_probe_gen.py:222` `_mean_char_from_cache`), never a per-cell
lookup — consistent with A1. **The only viable path** is to derive conditioning from the raw tile
parquet exactly as training does: `build_shards_in_memory(release, region, tile_ids)` →
`flatten_shards_to_cells(shards, seed, ablation)` (`heldout_cells.py:73-75`), producing `CellExample`
objects that carry `prefix_ids` (10 ids) + `character_stats` (7 floats) (`datamodule.py:226-233`).

**A2 — Manifest cell identity == `CellExample.key`, a total join.** Manifest/census cells are keyed
`(city, tile_i, tile_j, cell_i, cell_j)` (`lane_s_sampler.py:82-91` `SampledCell`; `:144`
`_cell_sort_key`; census cols `:252-262`; manifest `cells[]` at `:407-417`). `CellExample.key` =
`(region, tile_i, tile_j, cell_i, cell_j)` (`datamodule.py:166`), with `city == region`. `density_bucket`
is NOT part of identity — it is encoded inside `prefix_ids` via `cell_density_bucket`. The census
records only non-empty (conditionable) cells (`scripts/_heldout_cell_count.py:90-103`), so every one
of the 5,705 manifest cells is guaranteed a matching non-empty `CellExample` — the join is total, no
missing keys expected. **The driver joins on this 5-tuple.**

**A3 — Ablation must match how the checkpoint was trained (`"full"`).** `flatten_shards_to_cells`
takes an `ablation` arg that changes the value-bearing prefix (`datamodule.py:230` →
`conditioning.py:269` `build_value_bearing_prefix`). The bake-off trained under the full value-char-v1
scheme; `load_heldout_cells` and the steering probe both use `ablation="full"` (`heldout_cells.py:75`).
The driver MUST pass the ablation the checkpoint was trained under. **Load-bearing:** read the ablation
from the checkpoint's `hyper_parameters`/`ScaffoldConfig` and assert it, rather than hardcoding — a
mismatch silently changes the conditioning and voids "matched real". Default assumption: `"full"`.

**A4 — Checkpoint loading is raw `torch.load`, backbone from config not path.** `load_model(ckpt_path,
device)` (`src/cfm/eval/standing/harness.py:54`): `torch.load(..., weights_only=False)` →
`ScaffoldConfig(**ck["hyper_parameters"])` → `build_backbone(cfg.backbone, cfg)`
(`src/cfm/models/backbone.py:59`) → strip `model.` prefix → `load_state_dict(strict=False)` with a
loud `SystemExit` on any mismatch. `cfg.backbone` (`"transformer-ar"`/`"mamba-hybrid"`) drives the
dispatch — the driver never infers backbone from the path. Returns `(model, meta)` with
`meta["backbone"]`, `meta["seed"]`.

**A5 — The generation primitive is `score_cell`, keyed on the global index; the sharding primitive is
the golden-verified `sharded_eval`.** `score_cell(model, *, prefix_ids, char_stats, max_new, seed) ->
{gen_seconds, n_tokens, has_building, n_attempted, blocks, geoms}` (`scripts/train_scaffold.py:184`)
does generate (`generate_cell_tokens`, `cfm.inference.generate:40`) + decode
(`split_cell_into_features` from `cfm.data.sub_g.seam_decodability`; `try_decode_block` from
`cfm.inference.generate`) (`train_scaffold.py:200-205`). `sharded_eval(n_items, score_one, *, rank,
world_size) -> list[T]` (`src/cfm/eval/shard.py:116`) round-robins by rank
(`range(rank, n_items, world_size)`), then `all_gather_object` + `gather_in_order` merges into ONE
count-conserving, byte-deterministic list on every rank. The golden run
(`scripts/eval_sharding_golden.py`, job `47390793`, `_SHARDING_GOLDEN_PASS`) drives exactly
`sharded_eval(N, score_one)` where `score_one(i)` wraps `score_cell` with `seed = GEN_SEED_BASE + i`
(rank-independent). **The driver reuses this pattern verbatim, substituting the manifest cells for
`dm.val_cells`** (the only thing `_generate_and_score` binds to `val_cells` is the context list;
`score_cell` itself is already parameterized — `train_scaffold.py:251,279-280`).

**A6 — `generate_cell_tokens` is deterministic given the seed.** `generate_cell_tokens(model, *,
prefix, max_new, seed, char_stats=None) -> list[int]` (`cfm/inference/generate.py:40`) seeds a
dedicated `torch.Generator(device).manual_seed(seed)` (`:75`), multinomial-samples each step threading
`char_stats` into every forward (`:83`), stops at `CELL_END_TOKEN_ID=260` (`:91`, kept in the tail),
and returns only the generated tail. Same seed → identical tokens. Layout guard: if `char_stats` is
given, `len(prefix)` must be `CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS` (=10) else
`ValueError` (`:61-70`).

**A7 — The scoring chain: exact interfaces (all take the floor PATH, verify sha internally).**
- `gen_features_by_city(cells: list[DecodedCell], *, release: str) -> dict[str, GenFeatures]`
  (`gen_realism.py:56`). `DecodedCell(city, tile_i, tile_j, cell_density_bucket, blocks, geoms)`
  (`:40`). Output keyed `city -> {(metric, (zoning, road_skeleton, density, coastal)): [floats]}`
  where zoning/skeleton/coastal come from `read_tile_labels` (per-tile cache) and features from the
  shared `_tile_features` (`:66-85`). `GenFeatures = dict[tuple[str, tuple], list[float]]`.
- `verify_gen_coverage(gen_by_city, manifest, *, min_n=None) -> CoverageReport`
  (`lane_s_sampler.py:449`). Defaults `min_n = manifest["methodology"]["target_features"]` (=50).
  Per floored `(city, metric, stratum)` × `owed_metrics`: `ok` if achieved ≥ min_n; short AND
  binding-metric AND ceiling_bound → `ceiling_bound_excluded`; any other short → **raises
  `SamplerCoverageError`** (`:506`).
- `lane_s_excess(gen_features, real_features, artifact, *, city, min_n=None) -> LaneSResult`
  (`conditioning_floor.py:875`). For ONE city. `artifact` = path (→ `load_verified_floor`, verifies
  `floor_sha256==floor_artifact_sha256(data)`, `:776`) OR a `VerifiedFloorArtifact`. Per qualifying
  stratum: `excess = max(0, ks_distance(gen, real) - floor_all[key])`. Returns `LaneSResult(city,
  per_stratum_excess, median_excess, p90_excess, n_qualifying, n_skipped_thin)`.
- `memorization_check(gen_by_city, real_by_city, real_train_by_city, artifact, *, min_n=None) ->
  MemorizationCheck` (`bakeoff_decision.py:172`). Needs THREE by-city feature dicts + the verified
  floor. `real_train_by_city` must cover **EXACTLY** `artifact.payload["train_cities"]` (ValueError
  otherwise, `:214`). Discriminating strata come only from the artifact. Returns
  `MemorizationCheck(ok, verdicts, failing_pairs, n_pairs_no_strata)`; `ok=False` → the caller MUST
  halt (mirror `pick_winner`'s `MemorizationRefusal`, `:319`).
- `binding_city_verdict(per_backbone_per_city: dict[str, list[PerCityKS]]) -> BindingVerdict |
  NoDecisiveWinner` (`city_aggregate.py:74`). `PerCityKS(city, ks, seed_sem=0.0, n_features)`
  (`:18`). Per worst-first city: `gap = ks(runner_up) - ks(winner)`; decisive only if
  `gap > effective_floor = max(single_region_floor_gap(n), max(sem_winner, sem_runnerup))`
  (`:119-127`). No decisive city → `NoDecisiveWinner` (a value, not a raise).

**A8 — `train_cities` is the ~29-city set the floor artifact freezes** (`conditioning-floor.yaml`
`train_cities:` block: a_coruna, almere, amsterdam, barcelona, bologna, bruges, budapest, cergy,
copenhagen, debrecen, edinburgh, eindhoven, espoo, hamburg, helsinki, karlsruhe, linz, lisbon,
ljubljana, lodz, malmo, manchester, mannheim, milton_keynes, prague, rotterdam, szczecin, tallinn,
telford, …). `held_out_cities` = eisenhuttenstadt, glasgow, krakow, munich. **Load-bearing cost
implication:** the memorization check forces real-feature extraction over ALL train_cities (CPU-only,
checkpoint-independent) — an ops precondition to size before the run (Task 5 / ops T8).

**A9 — Neither the manifest nor the floor exists in the local checkout.** `data/processed/lane_s_
sampler/2026-04-15.0/` is Leonardo-only; the local floor YAML is
`reports/conditioning_floor/2026-04-15.0/conditioning-floor.yaml` (present, but the SEALED artifact
with its lock marker + sha lives on Leonardo). Therefore **all unit tests use tiny synthetic manifest
+ floor fixtures** (the `lane_s_sampler`/`bakeoff_decision` test suites already build these in
`tmp_path`), and every GPU/Leonardo touch is an ops task, never a unit test.

**A10 — Budget (re-derive-before-sbatch, PI condition 1).** From the probe jobs (`49831125`:
1,200 cells/24m37s/1 node; `49835918`: 2,880 cells/57m46s/1 node): tf ≈ 900–930 cells/GPU-h, mamba
≈ 530–560 cells/GPU-h. Full manifest per checkpoint: tf 5,705/≈915 ≈ **6.2 GPU-h**; mamba 5,705/≈545
≈ **10.5 GPU-h**. Matrix (3 tf + 3 mamba) ≈ **≈50 GPU-h ≈ 1% of the 5,000-GPU-h grant** (memory obs
13341 records ~46 GPU-h; consistent). 4-GPU-sharded wall ≈ 1.6 h (tf) / 2.9 h (mamba) per checkpoint.
Proposed ceiling to PI: 100 GPU-h (headroom for re-runs/real-feature extraction). **This number is
re-derived and shown to the PI before ANY `sbatch` — an ops gate, not a code task.**

---

## Global constraints (do not re-litigate)

- **Memorization halt fires → STOP and report. No scoring past it.** The decide/aggregate CLI runs
  `memorization_check` FIRST and raises before touching coverage/excess/verdict.
- **Do NOT:** fix the char_stats↔KS echo; re-pin the floor; rebuild the sealed manifest; weaken any
  assertion to pass; crown outside the locked two-floor rule (`NO_DECISIVE_WINNER` is a valid,
  publishable verdict — do not soften it); touch `_CLOSURE_EPS_M` (float-drift-only); absorb deferred
  defect (b) into eval tolerance.
- **Requires Leonardo repo ≥ `b30d604`** (defect-(a) closure fix — the classifier the scored lane runs
  through). Deploy is an ops precondition BEFORE the dry-run.
- **Verify before generating:** `_LANE_S_SAMPLER_LOCKED` present; `floor_sha256 == 95abb88…`;
  `census_sha256 == 236cea99…`; 5,705 cells / 146 strata. Fail loud on any mismatch.
- **Full-node always** (billing is per node): `--gres=gpu:4`, 32 cpus, `boost_usr_prod`, account
  `AIFAC_P02_548`. Never a rank-0-only or 1-GPU job. gcc-12 `LD_PRELOAD` for `mamba_ssm` import.
- **Write-once outputs** under `reports/_realism_eval/…`; every result lands in `reports/` with config
  + commit hash + job id + prose summary. Verify end-state before any DONE marker/sentinel.
- Python 3.11+, `from __future__ import annotations`, type hints on public fns, `ruff format`/`check`,
  `pytest`. TDD per task. Fast tests default; `@pytest.mark.slow` for anything heavy. The driver core
  must be **unit-testable without GPU/Leonardo** — inject the generation fn, use synthetic fixtures.
- Conventional commits, task-by-task, on `phase-2-cell-eos`. No push/merge without the PI's word.

---

## Design overview (~1 page)

**The pipeline, end to end.** For each of the 6 checkpoints:

1. **Conditioning (Task 1).** Load + verify the sealed manifest (shas, lock, counts). For each
   held-out city, `build_shards_in_memory` + `flatten_shards_to_cells(ablation)` → index
   `CellExample`s by `.key`. Left-join the manifest's ordered `cells[]` onto that index → an ordered
   `list[ConditionedCell]` carrying `(cell_key, density_bucket, prefix_ids, char_stats, real_body_
   tokens)`. Fail loud on any unmatched manifest cell (A2 says there should be none).

2. **Sharded generation (Tasks 2–4).** `sharded_eval(n=5705, score_one)` under torchrun, one rank per
   GPU. `score_one(i)` = `score_cell(model, prefix_ids=cells[i].prefix_ids, char_stats=cells[i].char_
   stats, max_new=DEFAULT_MAX_CELL_TOKENS, seed=BASE_SEED + i)` — **global-index-keyed, rank-
   independent** (A5). Rank 0 writes ONE write-once JSON: per cell `{cell_key, density_bucket, tokens,
   blocks, geoms, self_terminated}`. Emitting BOTH tokens and decoded blocks lets the local dry-run
   re-exercise decode for a determinism check.

3. **Real features, once (Task 5).** A checkpoint-independent Leonardo CPU step decodes the REAL body
   tokens for (a) the held-out manifest cells and (b) every `train_cities` city (A8), and runs them
   through the SAME `gen_features_by_city` classifier → `real-features.yaml`
   (`real_by_city` + `real_train_by_city`) in the `run_bakeoff_decision.py` record schema.

4. **Scoring + verdict, local (Task 6).** Fetch artifacts. Decode (or reuse decoded) → `Decoded
   Cell`s → `gen_features_by_city` per (backbone, seed). Then: **`memorization_check` FIRST → halt on
   `ok=False`.** Then `verify_gen_coverage`. Then per (backbone, seed, city) `lane_s_excess` →
   `median_excess`. Aggregate the 3 seeds → `PerCityKS(ks=mean, seed_sem=SEM)` → `binding_city_
   verdict` → `BindingVerdict` or `NoDecisiveWinner`. Write `decision.yaml` + coverage + memorization
   reports + the prose summary.

**The seed→verdict wiring gap (a load-bearing design decision, NOT a rule change).** The locked crown
is the two-floor rule in `city_aggregate.binding_city_verdict`, which already consumes a per-city
**`seed_sem`** (`city_aggregate.py:18-22`). BUT the wired `decide()`/`pick_winner` path builds
`PerCityKS` with the default `seed_sem=0.0` (`bakeoff_decision.py:334-340`) and its `--eval-results`
YAML schema (`run_bakeoff_decision.py:69-83`) carries a `scales[]` (ladder) dimension, **no seed
dimension**. The realism eval is fixed-scale + 3-seed, not a ladder — so `decide()` cannot be called
verbatim to honor the seed-noise floor. **Decision:** the Task-6 aggregation CLI orchestrates the
locked primitives directly — replicating `decide()`'s teeth as explicit guards (floor sha-verify;
STRICT held-out/train city set-equality of manifest vs artifact vs features; memorization-first) —
then computes the real per-seed SEM and calls `binding_city_verdict` with a populated `seed_sem`.
This uses the EXISTING seed→verdict rule with its intended input; it does not weaken or re-implement
the rule. `decide()`/`run_bakeoff_decision.py` stay untouched. (Rationale: "default to simplicity" +
"crown only via the locked rule" — the honest path is to feed the rule its seed-noise input, not to
shoehorn seeds into a ladder slot or to run with `seed_sem=0` and lose the reproducibility floor.)
This is flagged as **OPEN-FOR-PI-CONFIRMATION** in the summary.

**Where scoring runs (Q3 resolved).** Generation on Leonardo GPUs (Tasks 2–4). Decode is pure
deterministic geometry (no GPU) — it can run either on Leonardo inside `score_cell` (which the golden
path already does) OR locally; the driver emits tokens so the **feature-scoring + verdict always runs
locally with the orchestrator** (handoff §5), and the local dry-run re-decodes to prove determinism.
Real-feature extraction is Leonardo-CPU (needs the tile parquet) but checkpoint-independent and done
once.

**Job layout (Q4 resolved).** ONE torchrun sbatch **per checkpoint** (6 jobs), `sharded_eval`-driven
(the golden-verified 4-GPU path the handoff §1 condition 2 explicitly says to use), each ≈1.6–2.9 h
wall — comfortably under a 4 h `--time`, with model-load overhead amortized once per job. Chosen over
(a) the bash 4-worker `CUDA_VISIBLE_DEVICES` probe pattern (no cross-GPU count-conservation guarantee;
the golden proves `sharded_eval` gathers all N with holes=0) and (b) one mega-job (a single failure
loses all 6; longer than needed). Per-checkpoint jobs also let mamba (slower) and tf run independently.

---

## File structure

| File | Responsibility |
|---|---|
| `src/cfm/eval/realism/__init__.py` (create) | Package marker for the driver's pure-logic core. |
| `src/cfm/eval/realism/conditioning.py` (create) | Load+verify manifest; join manifest cells → matched `(prefix_ids, char_stats, real_body_tokens)` via the training datamodule. Injectable shard-builder. |
| `src/cfm/eval/realism/driver.py` (create) | Ordered cell list + `run_generation(cells, gen_fn, rank, world_size)` over `sharded_eval`; per-cell artifact schema (`GenCellRecord`) serialize/read. Injectable `gen_fn` (no torch in the core). |
| `src/cfm/eval/realism/scoring.py` (create) | Decode artifact → `DecodedCell`; per-(backbone,seed) `gen_features_by_city`; the seed-aware aggregation (`aggregate_seed_verdict`) wrapping `binding_city_verdict`; explicit set-equality + memorization-first guards. |
| `scripts/realism_eval_gen.py` (create) | Generation CLI (Leonardo): `load_model` → build cells → `sharded_eval` → write artifact + sentinel + end-state verify. `--dry-run`/`--limit-cells`/`--stratum`. |
| `scripts/realism_eval_real_features.py` (create) | Leonardo no-GPU CLI: extract real held-out + `train_cities` features → `real-features.yaml`. |
| `scripts/realism_eval_decide.py` (create) | Local scoring/aggregation CLI: fetch → decode → memorization-first halt → coverage → excess → seed verdict → `decision.yaml` + reports. |
| `scripts/realism_eval.sbatch` (create) | Leonardo torchrun 4-GPU per-checkpoint job (copy `steering_probe.sbatch` verification discipline; `eval_sharding_golden.sbatch` torchrun launch). |
| `docs/handoffs/2026-07-20-realism-eval-result-TEMPLATE.md` (create) | Prose `reports/` summary template (filled at run time as `reports/2026-07-XX-realism-eval-result.md`). |
| `tests/eval/realism/test_conditioning.py`, `test_driver.py`, `test_scoring.py` (create) | Fast unit tests with synthetic manifest/floor/CellExample fixtures. |

Artifacts (NOT committed — data): `reports/_realism_eval/2026-04-15.0/{gen-<bb>-seed<k>.json,
gen-features-<bb>-seed<k>.yaml, real-features.yaml, coverage-<bb>-seed<k>.yaml, memorization.yaml,
decision.yaml}` + sentinels.

---

## Task 1 — Conditioning source: manifest → matched conditioning join  *(local, TDD)*

**Files:** create `src/cfm/eval/realism/__init__.py`, `src/cfm/eval/realism/conditioning.py`; test
`tests/eval/realism/test_conditioning.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class ConditionedCell:
    cell_key: tuple[str, int, int, int, int]   # (city, tile_i, tile_j, cell_i, cell_j) — A2
    density_bucket: int
    prefix_ids: tuple[int, ...]                 # 10 ids incl. char placeholder (A1/A6)
    char_stats: tuple[float, ...]               # 7 floats (A1)
    real_body_tokens: tuple[int, ...]           # CellExample.tokens (real cell body, for real-features)

def load_verified_manifest_or_raise(path: Path) -> dict:
    """lane_s_sampler.load_verified_manifest + assert shas/counts:
    floor_sha256==95abb88…, census_sha256==236cea99…, len(cells)==5705, len(strata)==146."""

def build_conditioned_cells(
    manifest: dict,
    *,
    release: str,
    ablation: str,                              # A3 — MUST match the checkpoint's training scheme
    shard_builder: Callable[[str, str, list[tuple[int,int]]], list] = build_shards_in_memory,
    flattener: Callable[..., tuple[list, object]] = flatten_shards_to_cells,
    conditioning_seed: int = 0,
) -> list[ConditionedCell]:
    """Per held-out city: shard_builder(release, city, tile_ids) -> flattener(..., ablation) ->
    index CellExample by .key -> left-join manifest cells[] in manifest ORDER. Raise
    ConditioningJoinError listing any manifest cell_key absent from the index (A2: expect none)."""
```
- Inject `shard_builder`/`flattener` so tests never touch Leonardo parquet. The CLI (Task 3) passes
  the real `cfm.data.training.build_shards.build_shards_in_memory` /
  `cfm.data.training.datamodule.flatten_shards_to_cells`.
- The ablation is passed through, NOT hardcoded; Task 3 reads it from the checkpoint config (A3).

**Test plan** (`test_conditioning.py`, fast):
- `test_join_maps_every_manifest_cell` — synthetic 4-cell manifest + fake flattener returning matching
  `CellExample`-like objects (namedtuple with `.key/.prefix_ids/.character_stats/.tokens`); assert
  each `ConditionedCell` carries the right prefix/char and manifest ORDER is preserved.
- `test_unmatched_manifest_cell_raises` — flattener omits one manifest key → `ConditioningJoinError`
  naming the missing 5-tuple.
- `test_manifest_verification_rejects_bad_sha` — tampered synthetic manifest → the lane_s verify path
  raises; and a good one with wrong `census_sha256` constant → assertion error.
- `test_prefix_len_is_10_when_char_present` — guards the A6 layout invariant on the fixtures.
- `test_ablation_is_threaded_not_hardcoded` — spy flattener records the `ablation` kwarg it received.

**DONE:** all cases pass; `ruff` clean; no torch/Leonardo import in the core module (imports of
`build_shards_in_memory`/`flatten_shards_to_cells` are the injected defaults, tolerated but not
exercised in unit tests).

---

## Task 2 — Driver core: ordered generation over `sharded_eval` (injectable gen fn)  *(local, TDD)*

**Files:** create `src/cfm/eval/realism/driver.py`; test `tests/eval/realism/test_driver.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class GenCellRecord:
    cell_key: tuple[str, int, int, int, int]
    density_bucket: int
    tokens: list[int]
    blocks: list[list[int]]
    geoms: list[dict]
    self_terminated: bool

def run_generation(
    cells: Sequence[ConditionedCell],
    gen_fn: Callable[[ConditionedCell, int], dict],   # (cell, seed) -> score_cell-shaped dict
    *,
    base_seed: int,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[GenCellRecord]:
    """score_one(i) = gen_fn(cells[i], base_seed + i); dispatched through
    cfm.eval.shard.sharded_eval(len(cells), score_one, rank=rank, world_size=world_size).
    Global-index seed keying == rank independence (A5). Returns the count-conserved, ordered list."""

def write_gen_artifact(records, path, *, meta: dict) -> None:   # write-once; refuse existing
def read_gen_artifact(path) -> tuple[dict, list[GenCellRecord]]  # (meta, records)
```
- `gen_fn` is injected. Task 3 supplies a closure over `score_cell(model, prefix_ids=cell.prefix_ids,
  char_stats=list(cell.char_stats), max_new=DEFAULT_MAX_CELL_TOKENS, seed=seed)`. Tests supply a pure
  deterministic fake (`seed -> tokens`), so the core is GPU-free.
- `world_size=1` path (default when not distributed) must produce the full ordered list — matches
  `sharded_eval` single-process behavior for the local dry-run.

**Test plan** (`test_driver.py`, fast — patch `sharded_eval` to run in-process single-rank):
- `test_seed_is_global_index_keyed` — fake `gen_fn` records `(i, seed)`; assert `seed == base_seed+i`
  for all i (rank-independence contract).
- `test_records_in_manifest_order_and_count_conserved` — N=7 (ragged vs a would-be 4-shard),
  assert `len(out)==7` and order.
- `test_self_terminated_flag` — token list ending in 260 (not at cap) → `self_terminated True`; a
  cap-length list → `False`.
- `test_write_artifact_is_write_once` — second `write_gen_artifact` to the same path raises
  `FileExistsError`.
- `test_roundtrip_read_write` — `read_gen_artifact(write_gen_artifact(...))` reproduces records + meta.

**DONE:** all pass; no torch import in `driver.py`; `sharded_eval` is the only sharding mechanism used.

---

## Task 3 — Generation CLI  *(local code; run is ops)*

**Files:** create `scripts/realism_eval_gen.py`; extend `tests/eval/realism/test_driver.py` with a CLI
arg-parse test (no torch).

**CLI (argparse), mirroring `steering_probe_gen.py`:**
```
--ckpt PATH (required)         --manifest PATH (required)     --release STR (default 2026-04-15.0)
--out PATH (required)          --base-seed INT (default e.g. 20260720)
--max-new INT (default 4096)   # DECISION (orchestrator review 2026-07-20): 4096, not the 13312
                               # context cap. Measured probe median 394-400 tok/cell, mean ~486,
                               # 1.6% hit the probe's 1536 cap. 4096 ≈ 8x median bounds a
                               # pathological non-terminating tail to ~+12% budget worst-case,
                               # while a 13312 cap could add ~40%+. It is a CLI flag — the PI can
                               # override at submit without a code change; flagged in the budget
                               # checkpoint. Revisit if the dry-run shows >2% of cells at cap.
--ablation STR (default read-from-ckpt; assert match)   --limit-cells INT (dry-run)   --stratum STR (dry-run)
```
- **torch-touching imports are lazy** (`load_model` from `cfm.eval.standing.harness`; `score_cell`
  from `scripts.train_scaffold`; `torch.distributed`) so `build_conditioned_cells` and arg-parsing
  stay GPU-free and unit-testable (A4/A5 pattern; steering probe does this at `:261-265`).
- Flow: `dist.init_process_group("nccl")`; `torch.cuda.set_device(local_rank)`; refuse `WORLD_SIZE<2`
  unless `--dry-run` (mirror `eval_sharding_golden.py:39-45`); `model, meta = load_model(ckpt,
  device)`; read `ablation` from `meta`/config and assert vs `--ablation` (A3); `cells =
  build_conditioned_cells(manifest, release=…, ablation=…)`; optional `--limit-cells`/`--stratum`
  filter (dry-run only); `records = run_generation(cells, gen_fn, base_seed=…, rank=rank,
  world_size=world)`; **rank 0** `write_gen_artifact(records, out, meta=meta)`; **end-state verify**
  (re-read, assert `len(records)==len(cells)` [or the filtered count], holes=0); print sentinel
  `REALISM_EVAL_GEN_DONE` (rank 0).
- `gen_fn(cell, seed)` wraps `score_cell(model, prefix_ids=list(cell.prefix_ids),
  char_stats=list(cell.char_stats), max_new=args.max_new, seed=seed)`.

**Test plan:** `test_cli_parses_and_filters` (no torch): build args → assert the `--limit-cells`/
`--stratum` filter selects the right synthetic cells; `test_ablation_mismatch_raises` (config ablation
≠ `--ablation` → SystemExit). Real GPU behavior is validated by the ops dry-run (T8), not a unit test.

**DONE:** CLI imports cleanly without a GPU; arg tests pass; `ruff` clean. (No `sbatch` yet.)

---

## Task 4 — sbatch: torchrun 4-GPU per-checkpoint  *(local code; submit is ops)*

**Files:** create `scripts/realism_eval.sbatch`.

- **Headers:** `--partition=boost_usr_prod`, `--account=AIFAC_P02_548`, `--nodes=1`,
  `--ntasks-per-node=1`, `--cpus-per-task=32`, `--mem=240G`, `--gres=gpu:4`, `--time=04:00:00`,
  `--output/--error=logs/%x-%j.{out,err}`. (Match `steering_probe.sbatch:22-32`; add
  `--qos=boost_qos_lprod` per the golden sbatch if the ≥30 min runtime needs it — decide at deploy.)
- **Preamble:** `set -euo pipefail`; `cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM`; `module load
  python/3.11.7 cuda/12.2 gcc/12.2.0`; gcc-12 `LD_PRELOAD` block (`steering_probe.sbatch:35-39` — the
  `mamba_ssm` import needs it even for tf); `source .venv/bin/activate`.
- **Launch:** `torchrun --standalone --nproc_per_node=4 scripts/realism_eval_gen.py --ckpt "$CKPT"
  --manifest "$MANIFEST" --out "$OUT" --base-seed "$BASE_SEED"` (one checkpoint per job; pass via
  `--export`). This is the golden torchrun launch (`eval_sharding_golden.sbatch:35`), NOT the bash
  4-worker fan-out.
- **Verification discipline (copy the probe's "no marker without end-state" pattern,
  `steering_probe.sbatch:58-95`):** after torchrun exits, check its status; require the rank-0
  artifact non-empty (`[ -s "$OUT" ]`); `grep -q REALISM_EVAL_GEN_DONE logs/…`; only then echo
  `REALISM_EVAL_CKPT_DONE`. A bare success must never lie.

**DONE:** file lints as shell (shellcheck-clean where practical); headers/paths reviewed against
GROUND_TRUTH §1–2. NOT submitted — submission is ops T8 after the PI budget word.

---

## Task 5 — Real-feature extraction CLI (checkpoint-independent)  *(local code; run is ops, no GPU)*

**Files:** create `scripts/realism_eval_real_features.py`; reuse `scoring.py` decode helper (Task 6).

- **Purpose:** produce `real-features.yaml` with `real_by_city` (the 4 held-out cities, from the
  manifest cells' REAL body tokens carried by `ConditionedCell.real_body_tokens`) and
  `real_train_by_city` (EXACTLY the `train_cities` from the floor artifact, A8) — the memorization
  check's required input (`memorization_check` refuses a train-city-set mismatch, A7).
- **Held-out real:** decode `real_body_tokens` → `DecodedCell` → `gen_features_by_city`. **Training
  real:** `build_shards_in_memory`+`flatten_shards_to_cells` over each train city's tiles → decode
  `CellExample.tokens` → `gen_features_by_city`. CPU-only; no checkpoint loaded.
- Serialize to the `run_bakeoff_decision.py` record schema: `{city: [{metric, stratum, samples}, …]}`
  for both `real_by_city` and `real_train_by_city` (`run_bakeoff_decision.py:56-62,86-91`).
- **Ops precondition surfaced here (A8):** the train_cities extraction is the heaviest CPU step; size
  it (tile counts × cities) before running and note it in the budget line to the PI.

**Test plan** (`test_scoring.py`): `test_real_features_schema_roundtrips` — synthetic decoded cells →
YAML records → reload via the same `_features_from_records` shape used by `run_bakeoff_decision.py`;
`test_train_city_set_is_exactly_artifact` — the emitted `real_train_by_city` keys equal a synthetic
floor artifact's `train_cities` (else the downstream memorization check would refuse).

**DONE:** CLI runs on synthetic fixtures locally; emitted YAML loads with the target schema; `ruff`
clean. Real run is ops (T8), no GPU.

---

## Task 6 — Scoring / aggregation CLI: memorization-first → seed verdict  *(local, TDD)*

**Files:** create `src/cfm/eval/realism/scoring.py`, `scripts/realism_eval_decide.py`; test
`tests/eval/realism/test_scoring.py`.

**Interfaces (`scoring.py`):**
```python
def decoded_cells_from_artifact(meta, records, *, release: str) -> list[DecodedCell]:
    """GenCellRecord -> gen_realism.DecodedCell (reuse blocks/geoms; optionally re-decode tokens
    for a determinism assert in dry-run)."""

def assert_city_sets(manifest, artifact, real_by_city, gen_by_city_per_ckpt) -> None:
    """STRICT held-out set-equality across manifest, floor artifact, real features, and every
    checkpoint's gen — replicates decide()'s Tooth-2 as an explicit guard (bakeoff_decision.py)."""

def aggregate_seed_verdict(
    lane_s_by_ckpt: dict[tuple[str, int], dict[str, LaneSResult]],   # (backbone, seed) -> {city: LaneSResult}
    *,
    n_reference_by_city: dict[str, int],
) -> BindingVerdict | NoDecisiveWinner:
    """Per (backbone, city): mean median_excess over its 3 seeds = ks; std-error over seeds = seed_sem.
    Build PerCityKS(city, ks, seed_sem, n_features) per backbone; call
    city_aggregate.binding_city_verdict(per_backbone). This is the ONLY crown path — the locked
    two-floor rule with its seed-noise input populated (see 'seed→verdict wiring gap')."""
```

**CLI (`scripts/realism_eval_decide.py`) flow (all local, no GPU):**
1. Load the 6 gen artifacts + `real-features.yaml` + the floor artifact PATH + the manifest.
2. `assert_city_sets(...)` (fail loud on any mismatch).
3. Per (backbone, seed): `decoded_cells_from_artifact` → `gen_features_by_city` →
   `gen-features-<bb>-seed<k>.yaml` (target schema).
4. **`memorization_check(gen_by_city, real_by_city, real_train_by_city, floor_path)` FIRST**, per
   (backbone, seed). If ANY `ok=False` → write `memorization.yaml`, print the halt, `raise
   MemorizationHalt` (mirror `MemorizationRefusal`, `bakeoff_decision.py:319`). **No scoring past it.**
5. `verify_gen_coverage(gen_by_city, manifest)` per (backbone, seed); collect `CoverageReport`
   (ceiling-bound exclusions recorded; any non-ceiling short → `SamplerCoverageError` propagates).
6. Per (backbone, seed, city): `lane_s_excess(gen_features[city], real_features[city], floor_path,
   city=city)` → `median_excess`.
7. `aggregate_seed_verdict(...)` → `BindingVerdict` or `NoDecisiveWinner`. Write `decision.yaml` and
   the prose summary. `NO_DECISIVE_WINNER` is written as the named verdict — never softened.

**Test plan** (`test_scoring.py`, fast, synthetic floor+manifest in `tmp_path`, reuse the
`bakeoff_decision`/`gen_realism` fixture builders):
- `test_memorization_halt_blocks_scoring` — a `_GEN_MEMORIZER`-style feature set → `MemorizationHalt`
  raised BEFORE any `lane_s_excess` call (spy asserts excess never invoked).
- `test_city_set_mismatch_raises` — drop a held-out city from one gen dict → `assert_city_sets` raises.
- `test_seed_sem_drives_no_decisive` — construct 3-seed per-backbone `median_excess` with a small mean
  gap but large seed spread → `NoDecisiveWinner` (seed-noise floor binds); shrink the spread → a
  `BindingVerdict`. Proves the seed-noise floor is actually wired (the whole point of the gap fix).
- `test_decisive_winner_when_gap_clears_both_floors` — clean separation → `BindingVerdict` naming the
  winner + binding city.
- `test_coverage_ceiling_bound_excluded_not_raised` — a ceiling-bound short stratum lands in
  `ceiling_bound_excluded`, not a raise; a non-ceiling short raises `SamplerCoverageError`.
- `test_no_decisive_is_reported_verbatim` — the YAML written for a `NoDecisiveWinner` contains the
  named verdict + per-city `(gap, resolution_floor, seed_noise_floor)`.

**DONE:** all cases pass; the crown is reached ONLY via `binding_city_verdict`; memorization is
provably first; `ruff` clean.

---

## Task 7 — Dry-run modes + reports summary template  *(local code; the Leonardo dry-run is ops)*

**Files:** extend the CLIs with `--dry-run`; create
`docs/handoffs/2026-07-20-realism-eval-result-TEMPLATE.md`.

- **7a — Local scoring dry-run (no GPU, no Leonardo).** Using the LOCAL `data/_diag/heldout_cache.json`
  `body_tokens` (real cells) as a stand-in gen artifact, run `decoded_cells_from_artifact` →
  `gen_features_by_city` → `verify_gen_coverage`(on a synthetic 1-stratum manifest) → `lane_s_excess`
  against the LOCAL floor YAML fixture. PASS = the decode+feature+excess wiring runs end-to-end and
  emits the expected 4-tuple stratum keys with no exception. This is the "exercise decode+scoring
  locally" requirement (handoff §5) and needs NO checkpoint.
- **7b — Leonardo generation dry-run (tiny, minutes — ops).** `realism_eval_gen.py --dry-run
  --stratum <one> --limit-cells <~10> --ckpt <one>` under a short torchrun job. DECISION
  (orchestrator review 2026-07-20): the Leonardo dry-run ALSO runs torchrun 4-rank — it bills the
  full node either way (per-node billing) and exercises the exact sharded path the full run uses.
  Single-process (`world_size=1`) is permitted ONLY for the local no-GPU scoring dry-run (7a). Fetch the tiny artifact; run 7a's
  scoring locally on it. PASS = artifact has the expected N cells, `self_terminated` mostly True,
  re-decoding the tokens locally reproduces the on-Leonardo `blocks/geoms` bit-identically
  (determinism), and the local scoring chain runs clean.
- **Reports template:** sections — config (manifest shas, floor sha, ablation, base_seed, commit hash,
  job ids), budget-as-run (GPU-h per checkpoint vs the PI-confirmed ceiling), memorization result
  (PASS/HALT), coverage summary (ok / ceiling-bound-excluded counts), per-(backbone,seed,city)
  `median_excess` table, the seed→verdict (`BindingVerdict` or `NO_DECISIVE_WINNER` with both floors
  per city), and a plain-language interpretation (incl. the deferred-defect (b)/(c) caveats: interpret
  road-metric KS as geometry-not-topology, do NOT patch).

**Test plan:** `test_local_scoring_dryrun_smoke` (`@pytest.mark.slow` if the cache read is heavy, else
fast on a truncated fixture) — runs 7a on a 5-record synthetic cache and asserts a non-empty
`LaneSResult`/coverage report.

**DONE:** 7a runs green locally; template committed; 7b is documented as the ops gate before the full
run.

---

## Task 8 — Ops: deploy, verify, budget-to-PI, dry-run, full run, fetch, report  *(ops — orchestrator-run)*

**NOT subagent-implementable. Sequenced; each gated on the prior.** Uses the Leonardo SSH ControlMaster
(`Host leonardo`, `uaslam00`); auto-mode may deny remote writes → hand the PI a `!`-prefixed one-liner.

1. **Deploy** Leonardo `/leonardo_work/AIFAC_P02_222/Bonzai-OSM` to the Mac's committed HEAD (must be
   **≥ `b30d604`**) via git-bundle + `leo_deploy2.sh` collision-safe apply (GROUND_TRUTH §5 item 3).
2. **Verify preconditions** on Leonardo: `_LANE_S_SAMPLER_LOCKED` present; recompute
   `census_sha256==236cea99…`; manifest records `floor_sha256==95abb88…`; 5,705 cells / 146 strata;
   the 6 `last.ckpt`s exist under `/leonardo_work/AIFAC_P02_548/…/checkpoints/bakeoff/{transformer-ar-
   53M,mamba-hybrid-54M}/krakow-seed{7,13,23}/` (**explicitly check seed23 — its training log is
   UNAVAILABLE but the ckpt may exist; if a ckpt is missing, report and stop, don't assume**).
3. **Re-derive the budget** at measured rates (A10) from the probe-job timings; present the GPU-h
   number + the 100 GPU-h ceiling to the PI; **get the word before ANY `sbatch`** (PI condition 1).
4. **Real-feature extraction** (Task 5 CLI, Leonardo no-GPU) → `real-features.yaml`; fetch.
5. **Dry-run** (Task 7b): tiny Leonardo gen job + local scoring; confirm PASS.
6. **Full run:** submit `realism_eval.sbatch` per checkpoint (6 jobs); monitor via
   `logs/%x-%j.out` + the `REALISM_EVAL_CKPT_DONE` sentinel + non-empty artifacts; do NOT trust exit
   codes alone.
7. **Fetch** the 6 gen artifacts; run `realism_eval_decide.py` locally (memorization-first).
8. **Report:** fill the Task-7 template → `reports/2026-07-XX-realism-eval-result.md` with config +
   commit + job ids + verdict + prose. **PI checkpoint before ANY merge or crown language.** If
   memorization halts or the verdict is `NO_DECISIVE_WINNER`, report it verbatim — it is a valid,
   publishable outcome.

**DONE:** result report committed on `phase-2-cell-eos`; no merge/push without the PI's word.

---

## Open questions resolved (map to the 7)

1. **Conditioning source** → derive from tile parquet (`build_shards_in_memory` +
   `flatten_shards_to_cells`, join on `(city,tile_i,tile_j,cell_i,cell_j)`); the cache is a dead end
   (A1/A2). 2. **Memorization inputs** → three by-city feature dicts incl. `real_train_by_city`
   covering EXACTLY the floor's ~29 `train_cities` (A7/A8); halt is FIRST in Task 6. 3. **Where scoring
   runs** → generation on Leonardo GPU; feature-scoring + verdict local with the orchestrator; real
   features Leonardo-CPU once (Design Overview). 4. **Job layout** → one torchrun `sharded_eval` job
   per checkpoint (golden-verified path). 5. **Determinism** → `seed = base_seed + global_index` via
   `score_cell`/`sharded_eval` (A5). 6. **Dry-run** → local scoring dry-run (no GPU, from the cache) +
   tiny Leonardo gen dry-run (Task 7). 7. **Output** → write-once `reports/_realism_eval/…`, per-cell
   JSON + per-(backbone,seed) feature YAML in the `run_bakeoff_decision.py` schema, sentinels, prose
   summary (Task 7 / File structure).

**Seed→verdict wiring — REVIEWED AND APPROVED (orchestrator, 2026-07-20), flagged to PI.**
Verified against canon: `docs/GROUND_TRUTH.md:119-121` states the locked rule verbatim — "the 3
seeds give mean KS (estimate) + std/SEM (seed-noise). A winner is crowned at a city ONLY if the
winner-vs-runner-up mean-KS gap clears `effective_floor = max(C/√n resolvability, seed-noise
reproducibility)`." Calling `decide()` verbatim would run with `seed_sem=0` and silently DROP the
seed-noise floor — that would violate the locked rule, not honor it. Task 6's approach (orchestrate
the locked primitives, replicate `decide()`'s guards explicitly, feed `binding_city_verdict` the
real per-seed SEM) is the canon-compliant path. This decision is recorded here and will be restated
at the PI budget checkpoint before any sbatch; the PI can veto before any verdict is computed.
