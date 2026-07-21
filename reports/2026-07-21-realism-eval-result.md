# Scored held-out realism eval — RESULT: MEMORIZATION HALT (all 6 checkpoints)

## 0. One-line verdict

**MEMORIZATION HALT — all 6 checkpoints failed the Lane-M discriminator; NO fidelity scoring
ran (no coverage, no excess, no crown), by design.** A regurgitator passes realism by
construction, so scoring past this gate would be meaningless. This is a named, publishable
outcome and a PI decision point — it was not softened, retried, or routed around.

## 1. Config (provenance — what was run)

| Field | Value | Source |
| --- | --- | --- |
| Commit (repo, at generation time) | `acd077f` (Leonardo HEAD; gen path identical from `be04a39`) | deploy log |
| Commit (repo, at decision time) | `acd077f` (local == Leonardo) | `git log` |
| Overture release | `2026-04-15.0` | run config |
| Floor artifact path | `reports/conditioning_floor/2026-04-15.0/conditioning-floor.yaml` | decide invocation |
| **floor_sha256** | `95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6` | self-verified at load |
| Manifest path | `data/processed/lane_s_sampler/2026-04-15.0/sampler-manifest.yaml` (sealed; `_LANE_S_SAMPLER_LOCKED`) | decide invocation |
| manifest floor_sha256 | `95abb88…` (== floor's) | pinned loader (`ManifestLineageError` otherwise) |
| manifest census_sha256 | `236cea99dc370021113352c9c737da2404791ad200ca6d8d7e908e81ca6cb373` (recomputed locally: match) | pinned loader + `shasum` |
| manifest sampler_sha256 | `391979f31a0f4ef251e2e66f5b8c097ee4f5fbe93a0d87ea994e0f4625bcda3a` | manifest |
| Cells / strata | 5,705 / 146 (verified at load) | pinned loader |
| Real-features | `real-features.yaml` (canonical: 4 held-out + 38 train cities, `REALISM_EVAL_REAL_FEATURES_DONE`) | job 49909104 |
| Checkpoints scored | tf-53M × seeds{7,13,23}, mamba-54M × seeds{7,13,23} (locked 2×3 shape enforced) | `assert_locked_run_shape` |
| min_n | None → floor's frozen `methodology.min_n=50` | decide default |
| **Slurm job ids** | gen: `49904826/27/28` (tf s7/s13/s23), `49904829/30/32` (mamba s7/s13/s23); real-features: `49909104` (first attempt `49904814` FAILED — train-city tile-enumeration defect, fixed in `acd077f`); dry-run: `49896367` | ops log |
| Ablation | `full` (matched real conditioning; asserted vs each checkpoint) | gen CLI |
| base_seed / max_new | `20260720` / **8192** (PI-approved revision from 4096; measured real max 7,367 tok) | gen CLI + PI word |

## 2. Budget as-run vs PI ceiling (unit: GPU-h; 1 node = 4 GPU; billed per node)

| | GPU-h | Notes |
| --- | --- | --- |
| PI-confirmed ceiling | **100** ("go at 100/8192", 2026-07-20) | re-derived at measured 53M rates |
| Actual gen (6 ckpts, torchrun 4-GPU) | **64.8** | 16.21 node-h × 4 (2:54:36 + 3:15:27 + 2:25:58 + 2:28:26 + 2:28:43 + 2:39:22) |
| Leonardo gen dry-run (job 49896367) | 0.44 | 6m36s × 1 node |
| Real-features (CPU, `lrd_all_serial`) | ~0 GPU-h | 1h42m × 4 cores ≈ 6.8 core-h |
| Decision (local Mac) | 0 | GPU-free |
| **Total as-run** | **≈ 65.3** | well under the 100 ceiling; also under the 90–130 boot-doc expectation (the manifest is 86% density-bucket-0 cells, mean gen length ~340 tok vs the conservative 600 planning figure) |

Generation health: **100.0% self-termination** (34,229/34,230 cells; exactly 1 cell hit the
8192 cap). The 8192 cap revision was correct: per-artifact max lengths ran 6,027–8,192 — a
4096 cap would have truncated real-plausible dense cells.

## 3. Memorization (RUN FIRST — hard gate): **HALT**

`memorization.yaml` written; **no coverage/excess/verdict ran.** Per (D=held-out, T=train)
pair with frozen discriminating strata (derived exclusively from the sha-verified floor
artifact; 4 D × 38 T sweep, no top-k): PASS iff `median KS(gen, real_D) < median KS(gen,
real_T)`. Failing pairs verbatim from `memorization.yaml`:

| backbone | seed | ok | failing_pairs | n_pairs_no_strata |
| --- | --- | --- | --- | --- |
| mamba-hybrid | 7 | False | (munich,hamburg) (munich,vienna) (munich,warsaw) | 0 |
| mamba-hybrid | 13 | False | (munich,espoo) (munich,hamburg) (munich,lodz) (munich,warsaw) | 0 |
| mamba-hybrid | 23 | False | (munich,tallinn) | 0 |
| transformer-ar | 7 | False | (eisenhuttenstadt,tallinn) (glasgow,eindhoven) (glasgow,hamburg) (glasgow,malmo) (glasgow,prague) (glasgow,tallinn) (glasgow,warsaw) (munich,hamburg) (munich,lodz) (munich,tychy) (munich,vienna) (munich,warsaw) | 0 |
| transformer-ar | 13 | False | (eisenhuttenstadt,prague) (eisenhuttenstadt,szczecin) (glasgow,edinburgh) (glasgow,hamburg) (glasgow,prague) (krakow,prague) (krakow,warsaw) (munich,barcelona) (munich,copenhagen) (munich,lodz) (munich,prague) (munich,tallinn) (munich,tychy) (munich,warsaw) | 0 |
| transformer-ar | 23 | False | (eisenhuttenstadt,tychy) (eisenhuttenstadt,valencia) (glasgow,hamburg) (munich,hamburg) (munich,lodz) (munich,prague) (munich,tallinn) (munich,tychy) (munich,valencia) (munich,vienna) (munich,warsaw) | 0 |

Failing-pair counts (of 152 pairs swept per checkpoint): mamba 3 / 4 / 1; transformer
12 / 14 / 11.

**Margin characterization (orchestrator recomputation — indicative, NOT the check's exact
rule; strata filtered at ≥1 sample rather than the frozen min_n, which explains a few
small-negative entries).** Margins `medKS(own) − medKS(train)` run **+0.005 to +0.101**;
the largest are all munich (munich-vs-hamburg up to +0.101, munich-vs-warsaw +0.074,
munich-vs-tychy +0.064). Full table in the session ledger; recompute script inputs are the
committed `gen-features-*.yaml` + `real-features.yaml` + floor artifact.

**Pattern (facts, not verdict):**
- **munich fails in all 6 checkpoints** and carries the largest margins. glasgow fails in all
  3 transformer seeds; krakow only in tf-seed13; eisenhüttenstadt in 4 of 6.
- **transformer fails ~3–14× more pairs than mamba** (12/14/11 vs 3/4/1). No fidelity
  scoring ran, so this is NOT a crown or a fidelity comparison — it is a memorization-gate
  observation only.
- Margins are modest and diffuse (closer to several big training cities at once), which reads
  as *distributional* gravitation toward training-city statistics rather than verbatim
  tile copying — consistent with the standing finding that ~99% of conditioning signal is
  char_stats and macro conditioning is intrinsically weak (the model has little city-identity
  signal to adapt held-out generations with).

## 4. Coverage — NOT RUN (halted upstream)
## 5. Per-city excess / 5b. Seed aggregation — NOT RUN (halted upstream)
## 6. Verdict — NOT RUN; **no winner, no NO_DECISIVE_WINNER — the run halted at the
memorization gate.** `decision.yaml` deliberately does not exist.

## 7. Plain-language interpretation

The eval asks: "draw me munich's held-out squares, given munich-shaped hints — does the
drawing look like munich, or does it look like the cities you practiced on?" For every model
we tested, at least one held-out city's drawings sat *at least as close* to some practice
city's real statistics as to the target city's own — worst for munich, whose generations look
more like hamburg/vienna/warsaw than like munich. The gate that stops us from scoring
"realism" on such drawings fired exactly as designed: a model that replays its practice
material would otherwise score deceptively well.

Two readings, in decreasing likelihood given prior evidence: (1) **weak city-identity
conditioning** — we already measured that ~99% of the conditioning signal is character
statistics and only ~1% is macro/city context, so the model *cannot* strongly adapt to a
held-out city and defaults to generic big-city fabric; the discriminator correctly catches
that as training-gravitating output. (2) True tile-level regurgitation — less consistent with
the diffuse, small margins across many training cities. Distinguishing these is a v2
question (char-dropout retrain and neighbor-derived context stats were already queued as v2
conditioning candidates and are now directly motivated).

What this does NOT change: geometry remains 100% valid and self-terminating, and product-time
steering remains confirmed (3/3 replication). The v1 bar (plausibility + geometric validity)
and the memorization bar (city-faithfulness under held-out conditioning) are different bars;
this run shows the second is where the current models fall short.

### Deferred-defect caveats — INTERPRET, do NOT patch
- **(c) roads are geometry-not-topology**: not implicated — no road-metric fidelity scoring ran.
- **(b) ~1-quantum building-closing gap**: not implicated — halt fired upstream of any
  building-area KS; `_CLOSURE_EPS_M=1e-6` untouched.

## 8. Files & lineage

- `reports/_realism_eval/2026-04-15.0/memorization.yaml` — the halt record (committed).
- `reports/_realism_eval/2026-04-15.0/gen-features-<bb>-seed<k>.yaml` × 6 — audit dumps
  (committed; ~1.7 MB each).
- `decision.yaml` — **does not exist** (halt upstream; write-once discipline intact).
- Gen artifacts ×6 (~130 MB each) + `real-features.yaml` (629 MB): Leonardo
  `reports/_realism_eval/2026-04-15.0/` (write-once; NOT committed — data).
- Sentinels verified per job: `REALISM_EVAL_GEN_DONE` + `REALISM_EVAL_CKPT_DONE` ×6;
  `REALISM_EVAL_REAL_FEATURES_DONE`; artifact counts re-read (5,705 ×6).

## Appendix — dry-run provenance (pre-flight, NOT a result)

- Leonardo 7b: job `49896367`, 12 cells (mamba-seed23, one density stratum), 6m36s,
  10/12 self-terminated (under the old 4096 cap), `REALISM_DRYRUN_OK`.
- Local 7a: fetched artifact scored end-to-end (`REALISM_EVAL_DRY_RUN_OK`, Lane-S
  median_excess 0.0656→0.0355 across harness iterations, nothing written, crown unreachable).
- Cross-platform decode determinism: blocks bit-identical 12/12; geoms max float drift
  **5.7e-14** (x86→ARM) → `verify_tokens` tolerance set to blocks-exact + geoms atol 1e-9
  (`ad09b49`), three orders below `_CLOSURE_EPS_M`.
- First real-features attempt (job `49904814`) FAILED: train-city extraction routed through
  the held-out-only holdout-manifest path; fixed by reusing `build_train_city_shards`
  (`acd077f`), regime test proven RED pre-fix.
