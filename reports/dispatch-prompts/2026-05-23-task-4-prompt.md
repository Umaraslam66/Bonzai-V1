# Task 4 implementer dispatch prompt

**Status:** Revised v2; pending reviewer read-through / approval before dispatch.
**Target:** General-purpose subagent / Codex agent.
**Suggested model:** Sonnet-class.
**Branch:** `phase-1-sub-F-micro-tokenizer` (base includes Task 1 lock commit `afc9aa5`).

> The prompt below is the verbatim text to give the implementer agent. Everything between the `===` markers is the agent's prompt body.

===

Task: Sub-F Task 4 — BP4 unknown family + sentinel inventory, Halt 3 surface only.

You are working in `/Users/umaraslam/Projects/Bonzai-OSM` on branch `phase-1-sub-F-micro-tokenizer`. You are not alone in the codebase: do not revert edits made by others; inspect current state and work with it. Do not push. Do not create a PR. Do not proceed past Halt 3 approval. Do not proceed to any later Task 4 post-approval lock step.

## Preconditions

- Branch: `phase-1-sub-F-micro-tokenizer`.
- Task 1 is closed at `afc9aa5`.
- `configs/sub_f/semantic_vocab.yaml` is `LOCKED` with 127 semantic slots.
- Do not push. Do not PR. Do not proceed past Halt 3 approval.

## Pre-dispatch audits

### Audit step 1: confirm sub-C sentinel anchor data contract holds in cached Singapore

Run:

```bash
python -c "import pyarrow.parquet as pq; from pathlib import Path; rows = []; [rows.extend(pq.ParquetFile(p).read().to_pylist()) for p in Path('data/processed/sub_c/2026-04-15.0/singapore/').rglob('features.parquet')]; from collections import Counter; c = Counter((r['feature_class'], r.get('class_raw')) for r in rows); print('B__UNK__:', c.get((1, 'B__UNK__'), 0)); print('unknown:', c.get((0, 'unknown'), 0))"
```

Expected:
- `B__UNK__` count > `200000`
- `unknown` count > `5000`

Cascade #7 anchor values were `301418` and `9748`; thresholds allow roughly 30% drift for upstream growth/refresh.

If either count is below threshold: STOP, report BLOCKED. Sub-C may have stopped emitting sentinels; this is a real cascade #8 candidate requiring reviewer classification.

### Audit step 2: confirm semantic_vocab.yaml still has expected locked structure

Run:

```bash
python -c "import yaml; d=yaml.safe_load(open('configs/sub_f/semantic_vocab.yaml')); print(d['_status'], len(d['slots']), d['slots'][0]['tag'])"
```

Expected: `LOCKED 127 <first L1 tag>`.

If status/count drifted: STOP, report BLOCKED.

### Audit step 3: check spec §2.4 reserved-block status

Run:

```bash
grep -A 5 "per-family reserved blocks" docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md | head -20
```

Expected: §2.4 still defers reserve sizes; Task 1/2 did not lock all blocks.

Halt 3 should propose the ID-namespace anchor below. If already locked, consume existing lock instead.

## Implementation

- Create `scripts/sub_f/derive_unknown_family.py`.
- Create `configs/sub_f/unknown_family.yaml` with `_status: PROPOSED` pending Halt 3.
- Create `configs/sub_f/sentinel_inventory.yaml` with `_status: PROPOSED` pending Halt 3 + training-scaffold consumption.
- Append focused tests to `tests/data/sub_f/test_vocab.py`.
- Create `reports/2026-05-23-phase-1-sub-F-task-4-halt.md`.

## Unknown family derivation

- Derive one `<unknown_{key}>` slot per BP1 Gate 6 L1 must-appear key from locked `semantic_vocab.yaml`.
- Expected count: `28`.
- Preserve `semantic_vocab.yaml` L1 order, not sorted order.
- Do NOT derive slot list from empirical Singapore data.

## Singapore occurrence counts

- Compose key/value using sub-C `feature_class` mapping where valid:
  - `feature_class=0` -> `highway`
  - `feature_class=1` -> `building`
- Compose semantic tag as `f"{key}={class_raw}"`.
- Compare against the full locked semantic_vocab tag set: all `127` slots, including the `43` Singapore-empirical X exceptions.
- Count real OSM unknown only when composed tag is NOT in `semantic_vocab` tags and `class_raw` is NOT a sub-C sentinel.
- Count sub-C sentinel unknown when `class_raw` matches `unknown` / `__UNK__` / `B_*`.
- POI/base remain deferred; do not invent category mapping.

For each unknown slot, report:
- `singapore_count_real_osm_below_F`
- `singapore_count_subc_sentinels`
- `singapore_count_total = real + sentinel`

## Proposed thresholds

- `over_firing_flag`: true when unknown total >= 10% of that key's locked semantic-pair Singapore coverage.
- `zero_firing_flag`: true when unknown total == 0.
- Include numerator, denominator, ratio, and threshold rationale per slot.
- Status: `PROPOSED` pending Halt 3 reviewer lock.

## Proposed ID namespace anchor at Halt 3

- BP1 semantic family: IDs `0..199`
  - `127` used, `72` reserved for v2 semantic growth.
- BP4 unknown family: IDs `200..255`
  - `28` used at `200..227`, `28` reserved at `228..255` for v2/per-pair unknown growth.
- Dataloader-side sentinels: IDs `256..299`
  - `<pad>=256`
  - `<eos>=257`
  - `<bos>=258`
  - `<cell_start>=259`
  - `<cell_end>=260`
  - These are NOT sub-F on-disk vocab tokens.
- BP2 encoding primitives: placeholder block `300..1499`, values lock at Task 2 halt.
- BP7 boundary-ref: placeholder block `1500..1599`, values lock at Task 7 halt; `8` expected used, `92` reserved.
- Surface this in `sentinel_inventory.yaml` and Halt 3 report for reviewer lock.

## Additional Halt 3 report caveat required by reviewer

- Explicitly state that BP2 (`300..1499`) and BP7 (`1500..1599`) are PLACEHOLDER blocks.
- Their final sizes are empirically locked at Tasks 2 and 7 halts respectively.
- If Task 2 encoding-primitive count lands above `1200` or Task 7 boundary-ref count above `100`, BP2/BP7 blocks slide.
- Only BP1 + BP4 + dataloader sentinel IDs are proposed for lock at Halt 3.

## Tests

- Unknown family count = `28`.
- Unknown slot order follows semantic_vocab L1 order.
- Real-OSM unknown counts exclude all `127` semantic_vocab tags.
- Sub-C sentinel counts are reported separately.
- Unknown family used IDs are `200..227` and family block is `200..255`.
- `sentinel_inventory.yaml` reserves dataloader sentinels at `256..260` and marks them not on-disk.
- BP2/BP7 placeholder blocks are present.
- `unknown_family.yaml` and `sentinel_inventory.yaml` are `PROPOSED` at Halt 3.

## Verification

- `uv run python scripts/sub_f/derive_unknown_family.py`
- `uv run pytest tests/data/sub_f/test_vocab.py -v`
- `git diff --check`

## Halt report

Create `reports/2026-05-23-phase-1-sub-F-task-4-halt.md` containing:
- Enumerated `<unknown_*>` slot list.
- Singapore occurrence table with real-OSM below-F, sub-C sentinel, and total counts.
- Over-firing / zero-firing proposal table.
- Proposed Halt 3 ID namespace anchor.
- Proposed `sentinel_inventory.yaml` post-N/dataloader reservation.
- Explicit placeholder-block caveat for BP2/BP7 as described above.
- §10.5 telemetry.
- Status: `DONE_WITH_CONCERNS` unless BLOCKED audit/plan mismatch surfaces.

## Commit

Commit message:

```text
wip(sub_f): T4 pre-halt — BP4 unknown family + Singapore occurrence counts
```

Report final status as `DONE_WITH_CONCERNS` when Halt 3 surface is committed, or `BLOCKED` with the report content if an audit/plan mismatch surfaces.

===
