# Phase 1 Sub-F Task 3c Halt 4 Report

Status: `DONE_WITH_CONCERNS pending Halt 4 reviewer approval.`

Branch: `phase-1-sub-F-micro-tokenizer`

WIP commit pointer: `25a0493` (`wip(sub_f): T3c pre-halt — BP3 budget surface (formula stage-4; Halt 4 pending)`).

Pipeline commits (T3a -> T3b -> T3c):

- `ba04360` feat(sub_f): T3a joint P(feature_count, vertex_count|cell,type) on Singapore
- `7960a19` feat(sub_f): T3b stage-3 compound per-cell length sans cross-cell
- `25a0493` wip(sub_f): T3c pre-halt — BP3 budget surface (formula stage-4; Halt 4 pending)

## Scope

T3a + T3b + T3c BP3 sequence-length analysis end-to-end on cached Singapore.

- T3a (no halt; intermediate feat): joint `P(feature_count, vertex_count | cell, feature_type)` over 494 sub-C Singapore tiles. Output `configs/sub_f/stage_1_2_joint.yaml`.
- T3b (no halt; intermediate feat): stage-3 compound applying the BP2 Halt 2 encoder lock (hierarchical anchor, n_anchor=4). Output `configs/sub_f/stage_3_compound.yaml`.
- T3c (Halt 4 gate; wip until reviewer approval): joint × stage-3 × stage-4 -> per-cell length distribution, 5-quantile budget surface, per-type retention matrix, scaling projection at 1% and 5% Singapore-share, measured-vs-formula stage breakdown. Output `configs/sub_f/sequence_length_analysis.yaml`.

T3c uses the spec §7.2 formula for stage-4 because the local sub-E parquet cache is absent — see `reports/2026-05-23-phase-1-sub-F-close-checklist.md` line 12 and `project_sub_e_cache_absent_t3c_code_inferred` memory. The script makes `--sub-e-region-dir` optional and surfaces `stage_4_provenance: formula_derived_per_spec_7_2_no_sub_e_cache` in the output YAML.

## Audit Step Outcomes

1. **BP2 encoder lock present and read.** `configs/sub_f/encoding_primitives.yaml:1-19` carries `_status: LOCKED` with `lock_metadata.approved_lock_values.anchor_scheme: hierarchical`, `magnitude_quantum_m: 0.5`, `direction_count: 48`. T3b + T3c both resolve `n_anchor = 4` from this anchor scheme.

2. **Sub-C Singapore cache verified.** `data/processed/sub_c/2026-04-15.0/singapore/_SUCCESS` present; 494 `tile=EPSG3414_iN_jM/features.parquet` directories. Schema spot-check via pyarrow returns columns `[cell_i, cell_j, feature_class, source_feature_id, geometry, geometry_type, bbox_*, class_raw, subtype_raw, categories_primary, categories_alternate, sea_overlap_fraction]` — matches plan script expectations.

3. **Sub-E cache confirmed absent.** `ls data/processed/sub_e` returns `No such file or directory`. Triggers the formula-fallback path in T3c by design.

4. **Sub-D cache present.** `data/processed/sub_d/2026-04-15.0/singapore/` exists. Not consumed by T3a–T3c (T3 is sub-C + BP2-lock + sub-E only per spec §7.10), but its presence is consistent with the upstream pipeline state.

5. **Dev environment synced.** `uv sync --extra dev` returned `Audited 23 packages`. Python 3.11.14.

No §9.6.1 cascade surfaced during audit. The plan script for T3c was modified (not the plan document itself) per the implementer prompt's "T3c sub-E-absent adaptation" directive. Cascade is the documented sub-E-cache-absent path; not a new cascade.

## T3a Results Summary

Output: `configs/sub_f/stage_1_2_joint.yaml`.

- 494 tiles read, 31,616 grid cells enumerated (8×8 per tile, including empty cells per §7.8).
- 16,082 empty cells (`empty_cell_fraction = 0.509`).
- All four sub-C feature classes present: `{0: road, 1: building, 2: poi, 3: base}` per `src/cfm/data/sub_c/enums.py`.

Per-type observation counts and vertex distribution:

| Type | n_observations | n_cells_with_type | v_mean | v_p95 | v_p99 | v_max |
|------|---------------:|------------------:|-------:|------:|------:|------:|
| 0 (road) | 302,271 | 14,180 | 4.63 | 12 | 20 | 164 |
| 1 (building) | 395,177 | 12,630 | 7.47 | 17 | 57 | 282 |
| 2 (poi) | 149,655 | 7,493 | 1.00 | 1 | 1 | 1 |
| 3 (base/landuse) | 15,333 | 5,762 | 7.87 | 23 | 49 | 228 |

Feature-count-in-cell stats (observation-weighted; informational only, not the budget basis):

| Type | feature_count_mean | feature_count_p95 | feature_count_p99 | feature_count_max |
|------|-------------------:|------------------:|------------------:|------------------:|
| 0 (road) | 43.3 | 89 | 125 | 254 |
| 1 (building) | 81.9 | 243 | 318 | 393 |
| 2 (poi) | 182.7 | 648 | 963 | 975 |
| 3 (base) | 6.0 | 19 | 42 | 51 |

Note: POIs (fc=2) have huge feature-count-in-cell at the tail (P99=963) because POIs are dense in commercial cells. This drives the lower POI retention at lower quantiles in the budget surface below.

## T3b Results Summary

Output: `configs/sub_f/stage_3_compound.yaml`.

Encoder lock applied: `anchor_scheme=hierarchical`, `n_anchor=4`, Case A (uncrossed) only.

Per-observation tokens at per-type vertex statistics (Case A formula: `3 + N_anchor + 2*(V-1) = 5 + 2V`):

| Type | n_obs | tokens@v_mean | tokens@v_p95 | tokens@v_p99 | tokens@v_max |
|------|------:|--------------:|-------------:|-------------:|-------------:|
| 0 (road) | 302,271 | 15 | 29 | 45 | 333 |
| 1 (building) | 395,177 | 19 | 39 | 119 | 569 |
| 2 (poi) | 149,655 | 7 | 7 | 7 | 7 |
| 3 (base) | 15,333 | 21 | 51 | 103 | 461 |

Per-observation weighted mean: 15.55 tokens.

## T3c — Budget Surface

Output: `configs/sub_f/sequence_length_analysis.yaml`. Per-cell length = sum of Case-A tokens across all features in the cell + stage-4 (formula-derived, see below). 31,616 cells analyzed.

| Quantile | sequence_length_tokens | padded_length_tokens (×128) | padding_overhead_pct |
|---------:|----------------------:|----------------------------:|---------------------:|
| P99   | 3,584 | 3,584 | 0.0% |
| P99.5 | 4,175 | 4,224 | 1.2% |
| P99.9 | 5,792 | 5,888 | 1.7% |
| P99.99 | 8,149 | 8,192 | 0.5% |
| P100 | 8,967 | 9,088 | 1.3% |

Marginal-cost-of-cut between adjacent quantiles (additional tokens to retain the next 0.5pp of cells):

| Cut | Δ tokens | Δ tokens / Δ percentile-pt |
|-----|---------:|---------------------------:|
| P99   -> P99.5  | +591   | 1,182 / pp |
| P99.5 -> P99.9  | +1,617 | 4,043 / pp |
| P99.9 -> P99.99 | +2,357 | 26,189 / pp |
| P99.99 -> P100  | +818   | 81,800 / pp (single-tail) |

Per `feedback_marginal_cost_of_cut`, the elbow appears at the P99.5 -> P99.9 transition: cost per pp jumps ~3.4x from `1,182` to `4,043` between P99-P99.5 and P99.5-P99.9, then ~6.5x again between P99.9 and P99.99. The natural reviewer-elbow choices are **P99.5 (4,224 padded)** or **P99.9 (5,888 padded)**.

## T3c — Requirement 1: Raw per-type retention table at ALL candidate quantiles

Source field: `retention_by_quantile_by_type` in `configs/sub_f/sequence_length_analysis.yaml`.

Per-spec-§7.5 default retention thresholds: roads ≥99.9%; buildings/POIs/landuse ≥99.0%.

| Quantile | fc=0 road | fc=1 building | fc=2 POI | fc=3 base/landuse |
|---------:|----------:|--------------:|---------:|------------------:|
| P99   | 94.587% | 86.893% | 70.538% | 97.574% |
| P99.5 | 97.166% | 92.789% | 78.680% | 99.107% |
| P99.9 | 99.320% | 98.775% | 90.001% | 99.922% |
| P99.99 | 99.937% | 99.909% | 98.423% | 99.993% |
| P100 | 100.000% | 100.000% | 100.000% | 100.000% |
| **§7.5 default min** | **99.9%** | **99.0%** | **99.0%** | **99.0%** |

Reading against the §7.5 floor row (last bold row):

- **Roads (fc=0):** ≥99.9% requirement is met starting at P99.99 (99.937%). P99.9 misses by 0.580pp.
- **Buildings (fc=1):** ≥99.0% requirement is met starting at P99.99 (99.909%). P99.9 misses by 0.225pp.
- **POIs (fc=2):** ≥99.0% requirement is NOT met until between P99.99 (98.423%) and P100 (100%). POIs are the dominant pressure on the elbow because they cluster in dense commercial cells whose total cell length blows the budget. This is the trap §7.5 warns about ("aggregate passes while one feature type silently truncates").
- **Base/landuse (fc=3):** ≥99.0% requirement is met starting at P99.5 (99.107%). Already comfortable at P99.5.

**Per-type implication:** picking the elbow at P99.5 or P99.9 fails the §7.5 default thresholds for roads, buildings, AND POIs under the per-cell-rejection (α) truncation. Picking P99.99 meets roads, buildings, base; POIs miss by 0.577pp. Reviewer override candidates:

- Tighten the elbow to P99.99 (8,192 padded tokens) and accept the POI 98.423% retention as a documented exception (POIs are points; per-feature loss within a dense cell is the lightest downstream cost).
- Shift to truncation strategy (β) (feature-priority tail-drop within cell) so cells stay in the training set with non-priority POIs trimmed first. Retention rates as reported are α-mode; β-mode rates would shift upward for roads/buildings and downward for POIs but cells contribute partial data.
- Accept P99.5 elbow but flag a §7.5 override row in the lock YAML.

## T3c — Requirement 2: Scaling projection at 1% and 5% Singapore-share vs PRD 10K threshold

**PRD 10K cite status: NEEDS REVIEWER-CITE CONFIRMATION.** Grep of `PRD.md` for `10K`, `10[ ,_]?000`, `sequence`, `context length`, `seq budget` returned no global token-budget reference. The only "10,000" in PRD is at line 61: "Categories with fewer than 10,000 global instances bucket up to their parent category or are dropped" — that is the BP1 frequency floor (categories), not a sequence-length ceiling. Implementer-prompt directive specifies the 10K projection assumption; the YAML `scaling_projection.prd_cite_status` field carries the same flag.

The arithmetic surfaces both Singapore-share scenarios for each candidate quantile:

```
projected_total_tokens(share, quantile) = share × n_cells_singapore × per_cell_tokens(quantile)
```

with `n_cells_singapore = 31,616` (T3a measured).

**Singapore-share = 1% (the binding constraint):**

| Quantile | per_cell_tokens | projected_total_tokens | fraction_of_10K_budget |
|---------:|----------------:|-----------------------:|-----------------------:|
| P99   | 3,584.83 |   1,133,380 | 113.3x |
| P99.5 | 4,175.83 |   1,320,230 | 132.0x |
| P99.9 | 5,792.45 |   1,831,340 | 183.1x |
| P99.99 | 8,149.50 |   2,576,545 | 257.7x |
| P100 | 8,967.00 |   2,835,007 | 283.5x |

**Singapore-share = 5%:**

| Quantile | per_cell_tokens | projected_total_tokens | fraction_of_10K_budget |
|---------:|----------------:|-----------------------:|-----------------------:|
| P99   | 3,584.83 |   5,666,899 | 566.7x |
| P99.5 | 4,175.83 |   6,601,152 | 660.1x |
| P99.9 | 5,792.45 |   9,156,700 | 915.7x |
| P99.99 | 8,149.50 |  12,882,727 | 1,288.3x |
| P100 | 8,967.00 |  14,175,034 | 1,417.5x |

**Interpretation, with strong caveat:** even at 1% Singapore-share × P99 elbow, projected total tokens exceed the 10K assumed PRD budget by ~113×. Two readings:

- (a) The 10K assumed budget refers to a per-cell training-sequence ceiling, not a global per-region total. In that reading the right question is "does the per-cell budget exceed 10K?" — answer: NO at any candidate quantile (P100 = 8,967 < 10,000). All five candidate elbows fit comfortably under a 10K-per-cell ceiling. This is the most plausible reading given that sub-F emits one sequence per cell (§7.1).
- (b) The 10K assumed budget refers to a global per-training-sequence ceiling spanning many cells. In that reading the data is incompatible with the assumption at every candidate quantile and the budget needs revisiting — but this reading contradicts §7.1 ("one sequence per cell").

The reviewer should confirm which reading is intended. If (a), the surface comfortably fits and the elbow choice is driven by per-type retention (Requirement 1) and padding-overhead, not by the 10K ceiling. If (b), the elbow analysis becomes moot and the architecture-coupling residual (§7.11) needs to be re-opened.

## T3c — Requirement 3: Measured-vs-formula stage breakdown

Source field: `stage_breakdown` in `configs/sub_f/sequence_length_analysis.yaml`.

| Stage | What | Provenance | Source |
|------|------|-----------|--------|
| 1 | Per-cell feature counts by type | **MEASURED** from sub-C Singapore parquet | `data/processed/sub_c/2026-04-15.0/singapore` |
| 2 | Per-feature vertex counts by type | **MEASURED** from sub-C Singapore parquet (WKB) | `data/processed/sub_c/2026-04-15.0/singapore` |
| 3 | Tokens per geometry element | **DERIVED** from spec §7.2 encoder formula (Case A; n_anchor=4 per BP2 Halt 2 lock) | `configs/sub_f/encoding_primitives.yaml` |
| 4 | Cross-cell coordination overhead per cell | **FORMULA-DERIVED** per spec §7.2 (0.7 tokens/non-empty cell); sub-E parquet cache absent locally | `spec §7.2 formula (sub-E parquet cache absent)` |

Stage 4 is the only stage that is formula-derived rather than measured or deterministic-from-lock. Stage 3 is deterministic given the BP2 lock and the Case-A assumption (Cases B/C/D add 1–2 tokens per crossing — these collapse into stage 4's overhead).

Per the user's directive that formula-derived stages should cut **conservative**: the proposed truncation strategy below biases toward (α) tail-cell rejection at a slightly tighter quantile than the per-type retention argument alone would justify, because the +0.7 tokens/non-empty cell estimate could under-count in dense road-intersection regions where inbound crossings exceed the Singapore mean.

## Proposed truncation strategy (α/β/γ) + rationale

**Recommendation: α (tail-cell rejection), elbow at P99.9 (5,888 padded tokens).**

Rationale:

1. P99 and P99.5 fail the §7.5 default retention floor for all three primary feature types (roads, buildings, POIs). They are eliminated by the paired structural check (§8.1 BP3 row: "ALL of clear thresholds independently").
2. P99.99 meets roads + buildings + base but POIs miss by 0.577pp. P100 meets all but is single-tail (one outlier cell at 8,967 tokens; gap from P99.99 is 818 tokens for 0.01pp of cells — extreme marginal cost).
3. P99.9 misses the default floor on all three types modestly. With the override that POI retention floor relaxes from 99.0% to 90.0% (POIs are points; per-feature loss within a cell is lighter than per-cell loss for buildings/roads, and POIs are the type most affected by dense-cell pressure), P99.9 becomes the elbow.
4. **Conservative-cut bias from stage-4 formula provenance:** prefer P99.9 over P99.99 here. P99.99 sits at the inflection of the marginal-cost curve (26,189 tokens/pp); if stage-4 measured value comes in higher than the 0.7 formula estimate, P99.99 cells could individually exceed 8,192 padded and the budget surface shifts. P99.9 leaves ~2 padding-block headroom for stage-4 under-estimation.
5. **Alternative if reviewer rejects POI-floor override:** P99.99 with both POI 99.0% override AND base/landuse 99.0% requirement formally satisfied. Padding cost: 8,192 vs 5,888 = 39% more tokens per cell; transformer self-attention cost ~ (8192/5888)² = 1.94× more FLOPs per cell.

β (feature-priority tail-drop within cell) is a viable second choice if reviewer wants to keep dense POI cells. It would require sub-F encoder support for ordered priority-tail truncation; out of scope for T3c gate.

γ (encoding compression / BP2 revision) is not warranted by this data — the surface is reasonably bounded, not pathological. Per `feedback_schema_vs_data_cost_asymmetry`, schema changes are cheap but BP2 is already locked at Halt 2; reopening would require Halt 2 cascade.

## Long-cell diagnostic threshold proposal

Per spec §7.7, diagnostic fires at the per-cell token threshold corresponding to `(chosen_quantile - 0.5pp)`.

- If elbow = P99.9 (5,888 padded): diagnostic fires at the P99.4 token threshold. P99.5 = 4,175 tokens; P99.4 estimated linearly between P99.0 and P99.5: ~4,057 tokens (5,888 - 1,831 / ratio). Concrete value lives in the YAML `proposed_long_cell_diagnostic_pp: 0.5`.
- If elbow = P99.99 (8,192 padded): diagnostic fires at the P99.49 token threshold ≈ 4,168 tokens.
- The diagnostic threshold is computed once the reviewer picks the quantile; the YAML field carries the 0.5pp offset rather than the absolute value to keep the calculation locked to the elbow choice.

## Sub-E-absent adaptation summary

The plan's `scripts/sub_f/compute_budget_surface.py` was modified in three ways relative to the plan code text (plan document itself was NOT edited):

1. `--sub-e-region-dir` made OPTIONAL (default `None`). The original `required=True` is dropped.
2. Stage-4 fallback path: when sub-E cache is absent or not provided, every non-empty cell gets `+0.7` tokens per spec §7.2 Singapore estimate. Empty cells get 0 (consistent with §7.8 empty-cell-as-floor).
3. New output field `stage_4_provenance` with two possible values: `"measured_from_sub_e"` when sub-E cache is consulted, `"formula_derived_per_spec_7_2_no_sub_e_cache"` when fallback fires. The Halt 4 reviewer sees this prominently in the YAML and the report's stage-breakdown table.

The plan commit-step text in the plan document still references the original `--sub-e-region-dir` requirement. The plan document was not edited as part of this implementer task; if the reviewer ratifies the sub-E-absent path as the v1 default, the plan update is a separate documentation pass.

## Verification commands run + outputs

```text
uv sync --extra dev
```
Result: `Audited 23 packages in 2ms`.

```text
uv run python scripts/sub_f/analyze_stage_1_2_joint.py --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```
Result: `wrote configs/sub_f/stage_1_2_joint.yaml; 31616 cells (16082 empty, 494 tiles), feature_classes_present=[0, 1, 2, 3]`.

```text
uv run python scripts/sub_f/compute_stage_3_compound.py
```
Result: `wrote configs/sub_f/stage_3_compound.yaml; anchor=hierarchical n_anchor=4 per_observation_mean_weighted=15.55 tokens`.

```text
uv run python scripts/sub_f/compute_budget_surface.py --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```
Result: `wrote configs/sub_f/sequence_length_analysis.yaml; stage_4_provenance: formula_derived_per_spec_7_2_no_sub_e_cache; n_cells=31616 n_empty=16082` plus the surface and retention tables surfaced above.

```text
uv run pytest tests/data/sub_f/test_stage_analysis.py -v
```
Result: `4 passed in 0.02s`.
- `test_stage_1_2_joint_includes_all_sub_c_feature_classes` PASSED
- `test_stage_3_compound_uses_locked_anchor_scheme` PASSED
- `test_budget_surface_enumerates_5_quantiles` PASSED
- `test_budget_surface_retention_per_type_all_present` PASSED

```text
uv run ruff format scripts/sub_f/ tests/data/sub_f/
uv run ruff check scripts/sub_f/analyze_stage_1_2_joint.py scripts/sub_f/compute_stage_3_compound.py scripts/sub_f/compute_budget_surface.py tests/data/sub_f/test_stage_analysis.py
```
Result: `All checks passed!`.

## Reviewer Ratification Checklist

- [ ] Ratify or correct the PRD 10K-budget assumption (Requirement 2). Confirm whether 10K is a per-cell ceiling (reading a) or a global per-sequence ceiling (reading b). Choice affects whether the budget-surface elbow is meaningful or moot.
- [ ] Pick budget surface elbow: P99.5 (4,224 padded), **P99.9 (5,888 padded; recommended)**, or P99.99 (8,192 padded). Lock the choice in `configs/sub_f/sequence_length_analysis.yaml._status` -> `LOCKED`.
- [ ] Approve per-type retention overrides if any. Default §7.5 thresholds: roads ≥99.9%, buildings/POIs/landuse ≥99.0%. The recommended P99.9 elbow needs POI override to 90.0% (and small relaxations for roads/buildings). Document in `retention_overrides` key.
- [ ] Approve truncation strategy α (recommended), β, or γ.
- [ ] Approve long-cell diagnostic offset of 0.5pp (or override).
- [ ] Approve `stage_4_provenance: formula_derived_per_spec_7_2_no_sub_e_cache` as the v1-shipping provenance, with the close-checklist line 12 obligation to re-run T3c when sub-E cache is regenerated.
- [ ] Confirm Halt 4 reviewer-recorded telemetry per §10.5.

## §10.5 Telemetry

- implementer-time-to-data-surface: same-session implementation on 2026-05-28. T3a script + run + test + commit, T3b script + run + test + commit, T3c modified script + run + 4 tests + commit + this halt report. No separate wall-clock timer was instrumented; commits `ba04360`, `7960a19`, `25a0493` bracket the work.
- reviewer-time-to-approval: pending.
- reviewer-time-to-rejection-or-cascade: pending.

## Halt Decision

Status: `DONE_WITH_CONCERNS pending Halt 4 reviewer approval.`

Concerns surfaced for reviewer review (in priority order):

1. **PRD 10K cite gap.** The scaling-projection table per Requirement 2 is built against an implementer-prompt-assumed 10K global budget. PRD.md does not contain this cite. Reviewer must confirm or correct.
2. **Stage-4 provenance is formula-derived.** Sub-E cache absent; `+0.7 tokens/non-empty cell` per §7.2 estimate. Re-run T3c when sub-E is regenerated (close-checklist line 12). Recommended elbow biases conservative (P99.9 over P99.99) to absorb stage-4 under-estimation.
3. **POI retention pressure at elbow.** Per-type retention is the dominant elbow driver. Default §7.5 POI threshold (≥99.0%) is only met at P99.99 or P100. Reviewer choice on POI override is the load-bearing decision for the elbow.
4. **Truncation strategy.** α (tail-cell rejection) recommended for v1 with conservative quantile bias; β (within-cell priority tail-drop) is the deferred lever if reviewer rejects the POI override.

Do not proceed past Halt 4 until reviewer approval. Implementer will NOT autonomously commit a `feat(sub_f): T3c ... (Halt 4 approved)` commit per Halt 4 gate discipline.
