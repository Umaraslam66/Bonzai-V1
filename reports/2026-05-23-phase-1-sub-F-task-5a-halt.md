# Phase 1 Sub-F Task 5a Halt 5 Report

Status: DONE_WITH_CONCERNS pending Halt 5 reviewer approval.

Branch: `phase-1-sub-F-micro-tokenizer`

WIP commit: pending at report creation time (to be set by the `wip(sub_f): T5a pre-halt - BP5 verifications (Halt 5 pending)` commit immediately after this report lands).

## Scope

Implemented the Halt 5 surface only:

- Added `scripts/sub_f/verify_vertex_order_chain.py` (plan §Task 5a, Step 1, verbatim).
- Added `scripts/sub_f/verify_sub_d_rounding.py` (plan §Task 5a, Step 2, verbatim).
- Ran both scripts against cached sub-C Singapore.
- Produced `reports/sub_f_task_5a_vertex_order.yaml` and `reports/sub_f_task_5a_rounding.yaml` with `_status: PROPOSED`.

No encoder, decoder, per-axis test suite, writer, or `src/cfm/data/sub_f/versions.py` doc update was made. Task 5b (per-axis test implementation) and Task 8 (writer) remain blocked on this halt's outcome.

## Audit Step Outcomes

Pre-implementation audits per spec §10.3 + step-3 of the dispatch:

1. **Sub-C feature sort key (spec §5.2 row "OSM feature iteration order"; cited in plan as `src/cfm/data/sub_c/io.py:218-220`):** present and current. The lambda spans lines 219-221; the exact tuple-construction line is:
   - `src/cfm/data/sub_c/io.py:221`: `key=lambda r: (r.cell_i, r.cell_j, r.feature_class, r.source_feature_id),`
   - The plan's cited range (218-220) is one line off the lambda's actual final line; the spec range is still close enough that the §5.2 commitment is satisfied. Surfaced here for the reviewer to decide whether to amend the cite to `:219-221` in §5.2 + Test 4 of `tests/data/sub_f/test_per_axis_determinism.py` (`expected_key = "(r.cell_i, r.cell_j, r.feature_class, r.source_feature_id)"` — string-match is line-agnostic, so the test will pass; only the comment range needs touching).
2. **`src/cfm/data/sub_d/io.py` exists** (`grep` ran against it cleanly).
3. **Real features.parquet shape matches script assumptions:** sampled `data/processed/sub_c/2026-04-15.0/singapore/tile=EPSG3414_i9_j18/features.parquet`. Columns include `source_feature_id` (str) and `geometry` (WKB bytes, NDR / `\x01` byte-order prefix per sub-C `dump_wkb`). The script's `wkb_loads(r["geometry"])` and `r["source_feature_id"]` keying both apply.

No defect surfaced; scripts were implemented verbatim from the plan and run.

## Vertex-Order Verification

### What the script actually tests

Per the script docstring's own honesty: it samples the SAME sub-C parquet twice via cold `pq.ParquetFile(...).read()` calls and compares geometry vertex lists. **It does NOT cross the Overture → sub-A → sub-C chain.** Cross-chain (Overture → sub-A) requires re-fetching from Overture; the plan explicitly chose "out of scope for cheap halt input."

This is meaningful evidence about ONE link of the chain (sub-C parquet round-trip stability), not about the full chain the §5.6 verification commitment names.

### Results

`reports/sub_f_task_5a_vertex_order.yaml`:

```yaml
sample_size: 20
exact_match_count: 20
outcome_branch: a
recommendation: INHERIT (no canonicalization)
_status: "PROPOSED - pending Halt 5 reviewer approval per spec §10.3."
```

20/20 exact match across two cold reads of sub-C parquet vertices.

### Three-outcome enumeration (per `feedback_ambiguous_third_branch_in_verification`)

Spec §5.6:

- **(a) Chain guarantees stable vertex order** (docs + source say so for ALL three hops: Overture → sub-A → sub-C). Inheritance is valid; no canonicalization needed.
- **(b) Chain documents absence of guarantee** (any hop documents "order may vary"). Sub-F canonicalizes via lex-min polygon-ring rotation.
- **(c) Ambiguous** (docs don't guarantee for at least one hop, empirical sample shows stability). Canonicalize anyway. Cheap insurance. Per the memory: "Empirical stability under sampling is NOT a guarantee — same logic as sub-E's `feature_class` defect."

### Recommendation (per `feedback_verify_before_lock_not_after` + `feedback_ambiguous_third_branch_in_verification`)

The script's reported `outcome_branch: a` is misleading as written, because:

- The script only verified the sub-C ↔ parquet round-trip link (one of three hops). 20/20 match is conclusive evidence ONLY for "pyarrow reads of the same file are bit-stable" — i.e., the trivially-true link.
- The Overture → sub-A hop was not exercised. Overture parquet ordering across DuckDB httpfs scans is not documented in `docs/data/overture_pinning_policy.md` as vertex-order-preserving. Sub-A's reader code was not consulted by the script.
- Per `feedback_ambiguous_third_branch_in_verification`'s explicit default ("ambiguous defaults to defend, not to trust"), and per spec §5.6 ("Default for (c) is (b)'s treatment, NOT (a)'s. Decision lands at Halt 5"), the correct surface for the reviewer is:

  > **Effective outcome = (c) ambiguous-but-locally-stable.** Recommend CANONICALIZE via lex-min polygon-ring rotation in the sub-F encoder. The 20/20 match documents one hop's empirical stability, not a chain guarantee.

This contradicts the YAML's `recommendation: INHERIT`. The YAML reflects what the script's narrow comparison literally proved; this report reflects what spec §5.6's three-outcome framing requires under partial evidence. **Reviewer should approve the canonicalize default unless they ratify the script's narrow scope as "good enough for v1" and accept the documented risk as a §12 deferral candidate.**

If the reviewer chooses INHERIT anyway, the §13 revision ledger should record the deviation from `feedback_ambiguous_third_branch_in_verification`'s default with explicit rationale.

## Sub-D Rounding Verification

### Python `round()` behavior on constructed boundary inputs

`reports/sub_f_task_5a_rounding.yaml`:

```yaml
python_round_is_banker: true
sub_d_io_uses_round: false
sub_d_io_uses_int_cast: true
test_cases:
- {input: 0.5,  round_output: 0,  expected_banker: 0}
- {input: 1.5,  round_output: 2,  expected_banker: 2}
- {input: 2.5,  round_output: 2,  expected_banker: 2}
- {input: 3.5,  round_output: 4,  expected_banker: 4}
- {input: -0.5, round_output: 0,  expected_banker: 0}
- {input: -1.5, round_output: -2, expected_banker: -2}
recommendation: LOCK Python round() round-half-to-even (PEP 3141 default) for sub-F
_status: "PROPOSED - pending Halt 5 reviewer approval per spec §10.3."
```

All 6 constructed boundary cases match PEP 3141 banker's rounding. No environment-specific deviation.

### Sub-D source grep findings (expanded beyond plan)

The plan's script only greps `src/cfm/data/sub_d/io.py` for `"round("`. Per the dispatch instructions, I also greped all of `src/cfm/data/sub_d/` for `round(`, `int(round(`, `math.floor(`, `math.ceil(`, and related quantization-relevant patterns. Findings:

- **`src/cfm/data/sub_d/io.py`: NO `round(` calls.** Only `int(...)` type-casts on already-integer-typed values (slot_kind, slot_index, metric_namespace, scope, value). These do not introduce FP→int rounding semantics; they are noop or trivial truncation of already-integral values. The script's `sub_d_io_uses_round: false` is correct and consistent with this.
- **`src/cfm/data/sub_d/frequency_analysis.py:721`:** `idx = int(round(q * (len(ordered) - 1)))` — percentile-index quantization via `int(round(...))`. Uses Python `round()` (banker's).
- **`src/cfm/data/sub_d/evidence.py:359`:** `idx = int(round(q * (len(ordered) - 1)))` — identical percentile-index pattern via `int(round(...))`. Uses Python `round()` (banker's).
- **`src/cfm/data/sub_d/lattice.py`: no `round`, `floor`, `ceil`, or `quantum` calls** (lattice arithmetic is integer-only; cell sizes locked at 250m, no FP rounding step).
- **No `math.floor`, `math.ceil`, `math.trunc`, or `numpy` rounding calls anywhere in `src/cfm/data/sub_d/`.**

Adjacent finding (sub-C, not sub-D, but in the upstream chain feeding sub-F's encoder):

- **`src/cfm/data/sub_c/coords.py`:** uses `math.floor(...)` for cell-i/j assignment (lines 56, 57, 66, 67, 117, 118, 122, 123). This is the tile-assignment / cell-assignment quantization; it is floor, not round.
- **`src/cfm/data/sub_c/geom.py`:** mixes `math.floor` (lines 229, 230, 236, 237, 543-ish, 553, 615) and `round(...)` (lines 543, 552, 591, 595). The `round(...)` calls are used for cell-boundary detection on already-projected float coordinates — relevant for `cell_assignment` correctness, not for vertex-coordinate output. Vertex geometry itself is preserved through `dump_wkb` without rounding.

### Recommendation

- Sub-F encoder's coordinate quantization step should use **`int(round(coord_m / quantum))`** — Python `round()` (banker's). This matches sub-D's existing `int(round(...))` usage in `frequency_analysis.py:721` and `evidence.py:359` and is consistent with PEP 3141.
- **The assumed default ("Python `round()` round-half-to-even") is confirmed to be sub-D's actual mechanism** for the quantization-relevant call sites that exist. Sub-D `io.py` does not perform FP quantization (no `round(` calls); the percentile-quantization sites use `int(round(...))`. No §9.6.1 cascade is triggered.
- **No §9.6.1 cascade needed** for rounding axis. §5.2 row 2 ("Round tie-breaking") can lock as `int(round(coord_m / quantum))` with Python `round()` banker's semantics at Halt 5 approval.

## Sub-C Sort-Key Re-Verification

Spec §5.2 row "OSM feature iteration order" pins the sort key as `(cell_i, cell_j, feature_class, source_feature_id)` to be verified against `src/cfm/data/sub_c/io.py` at Task 5a.

File:line evidence:

- `src/cfm/data/sub_c/io.py:212-222`: `write_features_parquet` function.
- `src/cfm/data/sub_c/io.py:219-222`: `sorted_rows = sorted(features, key=lambda r: (r.cell_i, r.cell_j, r.feature_class, r.source_feature_id),)`.
- Exact sort-key tuple-construction line: **`src/cfm/data/sub_c/io.py:221`**: `key=lambda r: (r.cell_i, r.cell_j, r.feature_class, r.source_feature_id),`.

The spec's cited range `:218-220` is shifted by ~1 line from the actual lambda lines `:219-221`. The string match used by `tests/data/sub_f/test_per_axis_determinism.py::test_sub_c_feature_sort_key_is_4_tuple` (per plan Task 5b, Step 1, Test 4) is line-agnostic and will pass. Recommend the reviewer at Halt 5 approval also touch §5.2 + Test 4's comment to cite `:219-221` precisely, OR leave the existing `:218-220` as approximate and accept the off-by-one.

## Verification Commands Run + Outputs

```
uv sync --extra dev
```

```
Resolved 24 packages in 26ms
Audited 23 packages in 7ms
```

```
uv run python scripts/sub_f/verify_vertex_order_chain.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```

```
[vertex order] wrote /Users/umaraslam/Projects/Bonzai-OSM/reports/sub_f_task_5a_vertex_order.yaml; outcome=a
```

```
uv run python scripts/sub_f/verify_sub_d_rounding.py
```

```
[rounding] wrote /Users/umaraslam/Projects/Bonzai-OSM/reports/sub_f_task_5a_rounding.yaml; banker=True
```

Pre-implementation audit greps (already cited in §"Sub-D Rounding Verification → Sub-D source grep findings" above):

```
grep -rn -E "round\(|int\(|math\.(floor|ceil|trunc)" src/cfm/data/sub_d/ --include="*.py"
```

```
grep -rn -E "round\(|math\.(floor|ceil|trunc)|np\.(round|floor|ceil)" src/cfm/data/sub_d/ --include="*.py"
```

```
grep -n -E "round|floor|ceil|quantum|quantize" src/cfm/data/sub_d/lattice.py
```

(lattice.py grep returned no matches.)

Sample of features.parquet to confirm columns:

```
uv run python -c "
import pyarrow.parquet as pq
from pathlib import Path
p = Path('data/processed/sub_c/2026-04-15.0/singapore/').glob('tile=*/features.parquet')
tp = next(p)
t = pq.ParquetFile(tp).read()
print('columns:', t.column_names)
print('n rows:', t.num_rows)
"
```

```
columns: ['cell_i', 'cell_j', 'feature_class', 'source_feature_id', 'geometry', 'geometry_type', 'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y', 'class_raw', 'subtype_raw', 'categories_primary', 'categories_alternate', 'sea_overlap_fraction']
n rows: 6614
```

## Reviewer Ratification Checklist

To unblock Task 8 (writer) + Task 5b (per-axis test suite), the reviewer needs to approve:

1. **Vertex-order chain outcome.** Choose between:
   - **(a) INHERIT** — accept the script's narrow scope as sufficient evidence for v1; document the un-verified Overture → sub-A hop as a §12 residual + §13 ledger entry noting deviation from `feedback_ambiguous_third_branch_in_verification`'s default.
   - **(b/c default) CANONICALIZE** — adopt lex-min polygon-ring rotation in sub-F encoder; spec §5.2 row "Vertex iteration within feature" locks to "canonicalize"; Task 5b test 5 (currently `@pytest.mark.skip`) gains the canonicalization assertion shape; Task 8 writer implements the canonicalizer.
2. **Rounding mechanism lock.** Approve `int(round(coord_m / quantum))` with Python `round()` banker's semantics for sub-F coordinate quantization. No §9.6.1 cascade.
3. **Sub-C sort-key cite range.** Either:
   - Amend spec §5.2 + plan Task 5b Test 4 comment to cite `:219-221` (precise), OR
   - Accept current `:218-220` as approximate (string-match test is line-agnostic).
4. **Sub-D `io.py` does not perform FP quantization.** Accept that the rounding lock is justified by sub-D's `frequency_analysis.py:721` + `evidence.py:359` precedent rather than by `io.py`. The plan's script grep on `io.py` alone was insufficient; the expanded grep above is the actual evidence.

## Section 10.5 Telemetry

- **Implementer-time-to-data-surface:** same-session implementation + verification on 2026-05-28; wall-clock from script-write to halt-report-commit ~25 minutes (estimated; no separate timer instrumented). The two verification scripts completed in <1 second of wall-clock combined.
- **Reviewer-time-to-decision:** pending.
- **Total-halt-duration:** pending.

## Halt Decision

Status: DONE_WITH_CONCERNS.

Concerns are limited to expected Halt 5 reviewer ratification items above, plus one expectation-mismatch surfaced for reviewer awareness:

- The script literally proved sub-C parquet round-trip stability (one hop), not Overture → sub-A → sub-C chain stability (three hops). Per `feedback_ambiguous_third_branch_in_verification`'s explicit default, the correct treatment is CANONICALIZE for v1, with a documented residual for cross-chain verification deferred to a later sub-project. Reviewer may override and accept INHERIT with explicit rationale.

Do not proceed past Halt 5 until reviewer approval.
