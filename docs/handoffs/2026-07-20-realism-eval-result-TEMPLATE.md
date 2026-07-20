# Scored held-out realism eval — RESULT (TEMPLATE)

> **This is a fill-in template, not a result.** Copy it to
> `reports/YYYY-MM-DD-realism-eval-result.md`, replace every `<…>` placeholder with the
> value read from `decision.yaml` (and its sibling `memorization.yaml` on a halt), and delete
> this blockquote. `decision.yaml` is the sole authority — this prose never overrides it. Every
> number below must come from an artifact, not from memory. Reproducibility is mandatory:
> config + commit + job ids + data snapshot, or the result does not count.

## 0. One-line verdict

`<DECISIVE — winner <backbone>>` **or** `<NO_DECISIVE_WINNER>` (a named, publishable verdict,
NOT a failure — never softened, never retried).

## 1. Config (provenance — what was run)

| Field | Value | Source |
| --- | --- | --- |
| Commit (repo) | `<git sha>` | `decision.yaml: config.commit` |
| Overture release | `<2026-04-15.0>` | `config.release` |
| Floor artifact path | `<…/conditioning-floor.yaml>` | `config.floor_artifact_path` |
| **floor_sha256** | `<sha>` | `config.floor_sha256` (self-verified at load) |
| Manifest path | `<…/lane_s_manifest.yaml>` | `config.manifest_path` |
| manifest sampler_sha256 | `<sha>` | `config.manifest_sampler_sha256` |
| manifest floor_sha256 | `<sha>` | `config.manifest_floor_sha256` (must equal the floor's) |
| manifest census_sha256 | `<sha>` | `config.manifest_census_sha256` |
| Real-features path | `<…/real-features.yaml>` | `config.real_features_path` |
| Checkpoints scored | `<tf-seed7, tf-seed13, tf-seed23, mamba-seed7, mamba-seed13, mamba-seed23>` | `config.checkpoints` (LOCKED shape: 2 backbones × 3 seeds) |
| min_n | `<None → floor's frozen methodology.min_n=50>` | `config.min_n` |
| **Slurm job ids** | gen: `<…>`; decision: `<…>`; deploy/bundle: `<…>` | ops log (record every job) |
| Ablation | `<matched conditioning / char-ablated / …>` | run config |
| base_seed | `<…>` | gen run config |

Shas MUST match across manifest / floor / real-features / every checkpoint's gen — the runner
fails loud (`assert_city_sets`, `load_verified_floor`) on any lineage skew. If it ran, they matched.

## 2. Budget as-run vs PI ceiling

State the unit explicitly (GPU-h; 1 node = 4 GPU = 4 GPU-h/h; billing is per NODE).

| | GPU-h | Notes |
| --- | --- | --- |
| PI-confirmed ceiling (pre-`sbatch`) | `<the number the PI approved>` | re-derived at MEASURED 53M gen rates from the probe jobs, NOT the stale 262 estimate |
| Actual gen (6 checkpoints, 4-GPU) | `<…>` | sum of the 6 sharded gen jobs' node-h × 4 |
| Actual decision (CPU/local) | `<~0>` | GPU-free |
| **Total as-run** | `<…>` | must be ≤ ceiling; if it drifted, say why |

Expectation from the boot doc: ~90–130 GPU-h (the 53M models generate ~3× faster than the
100M-scale rate the 262 GPU-h estimate assumed). If the actual is far outside this, investigate
before trusting the run.

## 3. Memorization (RUN FIRST — a hard gate before any fidelity scoring)

`<PASS — all 6 checkpoints cleared the Lane-M discriminator>`
**or**
`<HALT — memorization.yaml written; NO coverage/excess/verdict ran>` (a regurgitator passes
realism by construction, so scoring is meaningless past a halt).

| backbone | seed | ok | failing_pairs | n_pairs_no_strata |
| --- | --- | --- | --- | --- |
| `<tf>` | `<7>` | `<True>` | `<[]>` | `<…>` |
| … | … | … | … | … |

If HALT: name the failing (backbone, seed) + pairs verbatim, then STOP — this is a PI decision,
not something to route around. Source: `memorization.yaml` (on halt) / `decision.yaml:
memorization` (on pass).

## 4. Coverage summary

Per (backbone, seed): floored (city, metric, stratum) slots that reached min_n on the GENERATED
side. A ceiling-bound short is EXCLUDED-and-reported (data limit, demoted downstream); ANY other
short RAISES `SamplerCoverageError` (sampler under-sized — the run would not have completed).

| backbone | seed | n_ok | ceiling_bound_excluded |
| --- | --- | --- | --- |
| `<tf>` | `<7>` | `<…>` | `<[(city, metric, stratum), …]>` |
| … | … | … | … |

Source: `decision.yaml: coverage`.

## 5. Per-(backbone, seed, city) median_excess (the raw scored quantity)

`excess = max(0, KS(gen, real_D) − floor_all_D)` per qualifying (metric, stratum); the
per-city value is the median over strata.

| backbone | seed | city | median_excess | p90_excess | n_qualifying | n_skipped_thin |
| --- | --- | --- | --- | --- | --- | --- |
| `<tf>` | `<7>` | `<glasgow>` | `<…>` | `<…>` | `<…>` | `<…>` |
| … | … | … | … | … | … | … |

Source: `decision.yaml: lane_s_median_excess`. Lower excess = closer to real.

### 5b. Seed-aggregated inputs to the verdict (what fed the two-floor rule)

Per (backbone, city): `ks` = mean of the 3 seeds' median_excess; `seed_sem` = std-error over
seeds (the seed-noise-floor input). `n_features` = floored-strata real reference count (the
resolution-floor input — floored strata ONLY, never all-strata).

| backbone | city | ks (mean excess) | seed_sem | n_features |
| --- | --- | --- | --- | --- |
| `<tf>` | `<glasgow>` | `<…>` | `<…>` | `<…>` |
| … | … | … | … | … |

Source: `decision.yaml: per_city_aggregation`.

## 6. Verdict (both floors, per city)

The ONLY crown path: a city is decisive only if the winner-vs-runner-up gap clears
**max(resolution floor `C/√n`, seed-noise floor)**. `decide()` was deliberately NOT used (it
sets `seed_sem=0` and would silently drop the seed-noise floor). Fill exactly ONE block.

**If DECISIVE:**

- winner: `<backbone>`  ·  runner-up: `<backbone>`
- binding held-out city: `<city>`
- gap: `<…>`  cleared  city_floor `<…>` (= max(resolution `<…>`, seed-noise `<…>`))
- demoted (under-powered) cities: `<[…]>`

**If NO_DECISIVE_WINNER** (written verbatim — route to the spec §13 simplest-backbone
tie-break, never improvise a winner):

| city | gap | resolution_floor | seed_noise_floor | bound by |
| --- | --- | --- | --- | --- |
| `<glasgow>` | `<…>` | `<…>` | `<…>` | `<resolution / seed-noise>` |
| … | … | … | … | … |

Source: `decision.yaml: verdict`.

## 7. Plain-language interpretation

Write 3–6 sentences a non-specialist can follow (Sketcher/Inker analogies welcome). Cover:

- **What the verdict means.** DECISIVE = one backbone drew the held-out cities' geometry
  measurably closer to real than the other, beyond both the resolution limit (how finely our
  reference sample can even tell distributions apart) and seed noise (how much the answer
  wobbles just from the random seed). NO_DECISIVE_WINNER = neither backbone separated beyond
  that noise — a real, publishable result, not a failed run. Do not soften it.
- **What "close to real" is and is NOT.** This scores per-(metric, stratum) distribution
  distance (KS) of feature quantities against the closest real city — plausibility and
  geometric validity, not photo-realism (the v1 AV/robotics-sim bar).

### Deferred-defect caveats — INTERPRET, do NOT patch

- **(c) Roads are geometry, not topology (v1 representation property).** The model emits
  plausible road *segments* but does not stitch them into a connected *graph* — they are
  fragmented. This WILL show up in the road-metric KS (e.g. `road_length_m`): read a road-metric
  gap as a **geometry-vs-topology** artifact of the v1 grammar, NOT as a fidelity failure to fix
  in this eval. It is a v2-grammar item. Do not add a topology repair or a tolerance to hide it.
- **(b) ~1-quantum building-closing gap (REAL but small).** Building rings can miss closing by
  about one coordinate quantum. It is genuine and must NOT be absorbed by widening eval
  tolerance; the closure epsilon `_CLOSURE_EPS_M=1e-6` is float-drift-only — leave it. If a
  building-area KS looks slightly inflated, name (b) as a candidate cause; do not patch the eval.

## 8. Files & lineage

- `decision.yaml` — the authority (config, per-ckpt excess table, aggregation, verdict, coverage,
  memorization). Write-once.
- `memorization.yaml` — present ONLY on a halt.
- `summary.md` — the runner's short prose verdict (never the sole authority).
- `gen-features-<backbone>-seed<seed>.yaml` — per-checkpoint gen-feature audit dumps.
- Gen artifacts (6): `<paths>` — write-once, one per (backbone, seed).

## Appendix — dry-run provenance (pre-flight, NOT a result)

Before the scored run, the wiring was exercised GPU-free with
`scripts/realism_eval_decide.py --dry-run` (decode → gen features → coverage → Lane-S excess on
ONE artifact; NO verdict, writes nothing; sentinel `REALISM_EVAL_DRY_RUN_OK`). A dry run is
structurally incapable of a crown and never emits a `decision.yaml`. `--verify-tokens` (default
ON in `--dry-run`) re-decodes each cell's tokens and asserts they reproduce the stored
`(blocks, geoms)` bit-identically — the on-Leonardo → local decode-determinism check the 7b PASS
criteria require. Record the dry-run job id / local invocation here for lineage:
`<local: heldout-cache slice, DRY_RUN_OK>` · `<Leonardo 7b: job …, N cells, self_terminated …/N>`.
