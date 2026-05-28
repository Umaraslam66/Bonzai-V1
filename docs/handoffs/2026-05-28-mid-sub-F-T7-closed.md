# Session handoff - mid-sub-F, Task 7 closed (2026-05-28)

> For the incoming agent: this is a resumable handoff. Assume no prior
> conversation memory. This document plus the referenced repo files are the
> entry point for completing Phase 1 sub-F.

## Branch State

- Branch: `phase-1-sub-F-micro-tokenizer`
- Latest implementation commit before this handoff: `c86af13`
  (`feat(sub_f): T7 BP7 boundary-ref vocab LOCKED - 8 tokens, sub-E authority (Halt 7 approved)`)
- Divergence from `main` before this handoff commit: 29 commits.
- Worktree state before this handoff file: clean.
- Do not push, create PRs, or merge. The human reviewer owns merge at sub-F close.

Chronological sub-F commit chain from `main`:

```text
38ff72b docs(sub_f): phase 1 sub-F micro-tokenizer design spec
07bd4bd docs(sub_f): phase 1 sub-F micro-tokenizer implementation plan
b52df6e docs(sub_f): spec sync - four-axis manifest -> six-axis (audit-after-fixup)
cd5d332 docs(sub_f): spec sync - BP1 scope amendments from cascades #4 + #5
69b835c docs(sub_f): plan revision - Task 1 cascades #4 + #5 + 5 code-bug fixes
c5362ae docs(sub_f): commit Task 1 dispatch prompt + mid-sub-F handoff
2fa9cd7 fix(sub_f): T1 prompt - audit step 2 API shape (action=query, not deprecated action=raw)
9d59e13 fix(sub_f): T1 plan + spec - cascade #6 outcome (rp=999 + paginate building only)
0332026 fix(sub_f): T1 prompt v2 - cascade #6 audit + 8-test halt
cfab21e wip(sub_f): T1 pre-halt - snapshots + L1+L2 curve + Singapore X (Halt 1 pending)
42efe4c fix(sub_f): T1 cascade #7 filter sub-C UNK from X threshold
afc9aa5 feat(sub_f): T1 BP1 vocab floor LOCKED - semantic_vocab.yaml, F + X (Halt 1 approved)
93016dc fix(sub_f): T4 prompt v2 - data-diagnostic sentinel audit
4355b92 wip(sub_f): T4 pre-halt - BP4 unknown family + Singapore occurrence counts
8721b2b feat(sub_f): T4 BP4 unknown family + sentinel inventory LOCKED (Halt 3 approved)
62d69db docs(sub_f): T2 prompt v3 - measured round-trip + anchor vocab conventions
081cea4 wip(sub_f): T2 pre-halt - BP2 encoder primitives + round-trip surface
5428704 feat(sub_f): T2 BP2 encoder primitives LOCKED - 48 dirs / 0.5m / hierarchical (Halt 2 approved)
d98f0d8 docs(sub_f): T6 prompt v1 - six-axis manifest halt surface
1ce1434 docs(sub_f): T6 prompt v2 - provisional manifest + SOURCE question
cf32d81 feat(sub_f): T6 BP6 6-axis version manifest + SOURCE composite (Halt 6 approved)
5504d38 docs(sub_f): T7 prompt v1 - BP7 boundary-ref halt surface
66ab540 docs(sub_f): T7 prompt v2 - add BP7 class coverage audit
01c4c92 docs(sub_f): T7 prompt v3 - add BP7 coverage evidence
ff40d3a wip(sub_f): T7 pre-halt - BP7 boundary-ref vocab + sub-C feature-splitting (Halt 7 pending)
6347d52 fix(sub_f): T7 plan + spec - cascade #9 BP7 drivable override
5fb7fdf docs(sub_f): T7 BP7 purpose reclassification surface
91d07b9 fix(sub_f): T7 discard local BP7 override; surface sub-E authority
c86af13 feat(sub_f): T7 BP7 boundary-ref vocab LOCKED - 8 tokens, sub-E authority (Halt 7 approved)
```

## Where Sub-F Stands

Closed and approved:

- Task 1 / Halt 1: BP1 semantic vocab floor locked.
- Task 2 / Halt 2: BP2 encoder primitives locked.
- Task 4 / Halt 3: BP4 unknown family + sentinel inventory locked.
- Task 6 / Halt 6: BP6 six-axis versioning and provisional manifest code locked.
- Task 7 / Halt 7: BP7 boundary-reference vocab locked.

Still open:

- Task 3a, 3b, 3c: BP3 sequence-length / budget surface. Halt 4 occurs at T3c.
- Task 5a: BP5 verification halt. Halt 5 blocks Task 8.
- Task 5b: BP5 per-axis test suite after Halt 5.
- Tasks 8-15: writer, validators, pipeline, CLI, integration, empirical gate, final handoff.

Recommended next work:

1. Dispatch or implement Task 5a first if prioritizing writer unblock. It is the remaining direct blocker for Task 8.
2. Run Task 3a -> Task 3b -> Task 3c to close Halt 4 and lock BP3 budget/retention/truncation.
3. After T5a and the already-closed T1/T2/T4/T6/T7, proceed through T8-15.

## Authoritative Files

- Spec: `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md`
- Plan: `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md`
- Protocol: `docs/protocols/sub-project-planning-protocol-v1.md`
- Running sub-F-close checklist: `reports/2026-05-23-phase-1-sub-F-close-checklist.md`
- Latest halt reports:
  - `reports/2026-05-23-phase-1-sub-F-task-1-halt.md`
  - `reports/2026-05-23-phase-1-sub-F-task-2-halt.md`
  - `reports/2026-05-23-phase-1-sub-F-task-4-halt.md`
  - `reports/2026-05-23-phase-1-sub-F-task-6-halt.md`
  - `reports/2026-05-23-phase-1-sub-F-task-7-halt.md`

## Locked Artifacts

### BP1 semantic vocab

- Path: `configs/sub_f/semantic_vocab.yaml`
- Status: `LOCKED`
- Slots: 127 first-class semantic slots.
- Lock values:
  - `F = 9.95794913319044e-08`
  - `X = 2.5887822885870944e-06`
  - granularity: `L1+L2-mixed`
  - F exception list: empty
- Shape: 28 L1 keys + 56 L2 highway/building pairs + 43 Singapore-empirical X-pass-list pairs.

### BP4 unknown family

- Path: `configs/sub_f/unknown_family.yaml`
- Status: `LOCKED`
- Slots: 28 unknown slots in semantic-vocab L1 order.
- IDs: `200..227`
- Reserve: `228..255`
- Notable decision: 26 zero-firing Singapore slots retained for multi-region expansion.

### BP2 encoding primitives

- Path: `configs/sub_f/encoding_primitives.yaml`
- Status: `LOCKED`
- Lock values:
  - `direction_count: 48`
  - `magnitude_quantum_m: 0.5`
  - `anchor_scheme: hierarchical`
  - `chunk_threshold_m: 32`
  - `round_trip_l_inf_threshold_m: 4.8`
  - `round_trip_angle_threshold_deg: 7.5`
  - `collinearity_admission_perpendicular_m: 0.928048`
- BP2 block in sentinel inventory: `300..1499`
- Used count: 209 slots (anchor 96 + direction 48 + magnitude 65)
- Reserved count: 991.

### ID namespace / dataloader sentinels

- Path: `configs/sub_f/sentinel_inventory.yaml`
- Status: `LOCKED`
- Blocks:
  - BP1 semantic: `0..199`, 127 used.
  - BP4 unknown family: `200..255`, 28 used at `200..227`.
  - Dataloader sentinels: `256..299`, on-disk false.
  - BP2 encoding primitives: `300..1499`, 209 used.
  - BP7 boundary-ref: `1500..1599`, 8 used at `1500..1507`.
- Named dataloader sentinel IDs:
  - `<pad>=256`
  - `<eos>=257`
  - `<bos>=258`
  - `<cell_start>=259`
  - `<cell_end>=260`

### BP7 boundary-reference vocab

- Path: `configs/sub_f/boundary_reference_vocab.yaml`
- Status: `LOCKED`
- Slots: 8.
- IDs:
  - `1500 <bref_N_MAJOR>`
  - `1501 <bref_E_MAJOR>`
  - `1502 <bref_S_MAJOR>`
  - `1503 <bref_W_MAJOR>`
  - `1504 <bref_N_MINOR>`
  - `1505 <bref_E_MINOR>`
  - `1506 <bref_S_MINOR>`
  - `1507 <bref_W_MINOR>`
- Semantics: sub-F tokenizes sub-E `boundary_contract.parquet` `BoundaryClass` values verbatim. No local `highway=*` -> class override.

### BP6 version / manifest code

- Code paths:
  - `src/cfm/data/sub_f/versions.py`
  - `src/cfm/data/sub_f/manifest.py`
  - `src/cfm/data/sub_f/provenance.py`
- Status: code locked at Task 6; no status-gated YAML artifact.
- Version axes: sub-D's 5 namespaces plus SOURCE = 6 total.
- SOURCE value is scalar because sub-D `VersionRef` is scalar-only, encoded in canonical format:
  `overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>`
- Structured component fields are also available in `manifest["sub_f_source_version"]`.

## Provisional Manifest Obligation

Task 6 deliberately left region manifest `vocab_sources` partial because BP7 had not locked yet.

Current obligation, tracked in `reports/2026-05-23-phase-1-sub-F-close-checklist.md`:

- Add `configs/sub_f/boundary_reference_vocab.yaml` to region manifest `vocab_sources`.
- Recompute final complete `manifest_sha256`.
- Clear or update `vocab_sources_status` away from `partial_pending_bp7`.
- Do this at sub-F close / Task 15 after the writer/orchestrator produces the region manifest.

Do not forget this. It is the main known provenance gap left by the T6 -> T7 ordering.

## Verification Debt

Sub-E output is absent in this workspace:

- `data/processed/sub_e` is absent.
- No `boundary_contract.parquet` files were found under the repo at T7 close.

Consequences:

- Task 7's statement that a motorway-only edge emits MINOR under sub-E is code-inferred from `src/cfm/data/sub_e/derivation.py`, `pipeline.py`, and `writer.py`, not observed in parquet.
- Task 7's statement that same-edge MultiLineString parts collapse to one per-edge class is also code-inferred, not parquet-observed.
- This does not block BP7 lock because sub-F correctness is faithful passthrough of sub-E's authoritative contract.
- When sub-E output is regenerated or restored, spot-check actual `boundary_contract.parquet` emission for motorway-only and same-edge MultiLineString edges. If actual output diverges, update BP7 limitation docs. The BP7 lock remains faithful-passthrough either way.

## Cascades / Cascade-Equivalent Defects

Carry these forward. Do not re-litigate unless new source evidence contradicts them.

1. **Cascade #1: compare_version mechanism.** sub-D uses enum-add `VersionNamespace`, not kwarg-add. sub-F extends by adding SOURCE.
2. **Cascade #2: six-axis manifest.** sub-D has 5 namespaces, not 3; sub-F adopts all 5 plus SOURCE.
3. **Cascade #3: sub-C sort key concrete.** sub-C feature sort key is `(cell_i, cell_j, feature_class, source_feature_id)`.
4. **Cascade #4: Singapore X scope narrowed.** sub-C POI/base classes do not give direct OSM key/value signal; X-threshold v1 scope is highway + building.
5. **Cascade #5: L1 must-appears corrected.** Reviewer-supplied 15-key list was wrong; Map_features primary set is 28 keys.
6. **Cascade #6: taginfo `rp=1000` rejected.** Use `rp=999`; paginate `building` only; cap non-scope value tails at first 999 where applicable.
7. **Cascade #7: sub-C unknown sentinels filtered.** `building=B__UNK__`, `highway=unknown`, and related sentinel patterns are not BP1 semantic slots; they map through BP4 unknown family.
8. **Cascade #8: BP2 right-angle catastrophic known-loss.** 0.22% catastrophic right-angle cases from anchor/direction-bin alignment accepted as v1 known-loss; sub-F-v2 candidate, no v1 block.
9. **Cascade #9: sub-E boundary class authority.** sub-F consumes sub-E `boundary_contract.parquet` verbatim. sub-E MINOR-default can under-tier `motorway` and over-emit non-vehicular/ambiguous ways as MINOR. Accepted for sub-F-v1 as inherited sub-E limitation; sub-E-v2 candidate.

Related non-cascade defects worth remembering:

- Task 1 prompt audit used deprecated wiki `action=raw` while implementation used correct `action=query`. Fixed as prompt-derivation defect.
- Task 4 prompt audit initially grepped wrong source path for sub-C sentinels. Replaced with data-diagnostic audit.
- Task 7 invalid sub-F-local BP7 override commit `6347d52` is superseded by `91d07b9` and `c86af13`; do not reintroduce a local override.

## Protocol-v2 Candidates

The count to carry to sub-F close is 10 candidates:

1. **Direct upstream source read.** Gate 6 trigger phrases like "extending X" require reading X's defining source file, not downstream docs summarizing it.
2. **Hand-enumeration with complete-count assertions.** Every hand-enumerated upstream list needs an independent count assertion.
3. **Reviewer-supplied lists are untrusted input.** Reviewer-authored lists still require canonical-source verification.
4. **Dispatch audit must reuse implementation call/code path.** Audit steps must not invent shorthand API calls that differ from implementation.
5. **Exact-parameter upstream diagnostics.** Audit endpoint parameters, not just URL paths; `rp=1000` passed path review but failed API validation.
6. **Reviewer-supplied parameter values and premises are untrusted.** Includes `rp=999 is safe`, BP7 drivable-only purpose, and BP7 missing-means-NONE premise.
7. **Singapore-frequency pass-lists must filter normalization sentinels.** Upstream sentinel artifacts cannot become semantic vocab slots.
8. **Audit anchor paths must verify actual code/data structure.** Prefer data-contract diagnostics when source path is incidental.
9. **Hypothesis falsification must be explicit.** If a diagnostic falsifies the hypothesis, say so; do not reframe as confirmation.
10. **Late-stage integration / architecture-ownership audit.** For any cascade proposing sub-F local behavior against an upstream, first ask whether sub-F owns that derivation or consumes upstream output as authoritative. Architecture ownership precedes resolution design.

## Cross-Sub-Project Couplings

- **sub-C SOURCE coupling:** `src/cfm/data/sub_f/versions.py` depends on sub-C manifest field names `sub_c_schema_version` and `initial_extraction.commit_sha`. A future sub-C schema rename must update sub-F SOURCE loading.
- **sub-E BP7 coupling:** BP7 classes are 100% sub-E-derived under architecture (b). Any sub-E boundary-contract schema or semantic change propagates directly into sub-F BP7 output.
- **sub-E-v2 candidates inherited by sub-F:** refine highway tiering (`motorway` -> MAJOR; decide non-vehicular way handling) and evaluate richer multi-crossing representation for same-edge MultiLineString roads.
- **sub-D versioning coupling:** sub-F SOURCE is encoded as scalar string only because sub-D `VersionRef` is scalar-only. If sub-D supports structured VersionRefs later, sub-F can preserve both structured and canonical scalar forms.
- **sub-D lattice / round coupling:** T5a still must verify sub-D rounding mechanism and vertex-order chain. Do not assume it from memory.
- **training-scaffold coupling:** dataloader sentinel IDs `256..260` are `on_disk: false`. Training-scaffold must consume them as training-side IDs, not expect them in sub-F parquet vocab.

## Remaining Task Shape

### Task 3a -> 3b -> 3c (Halt 4)

- T3a: Stage-1 + Stage-2 joint distribution by feature type.
- T3b: Stage-3 compound per-cell length sans cross-cell.
- T3c: Stage-4 compound + budget surface, Halt 4.
- Halt 4 surfaces budget quantile elbow, per-type retention table, truncation strategy (alpha / beta / gamma), and long-cell diagnostic threshold.
- Output target: `configs/sub_f/sequence_length_analysis.yaml`.

### Task 5a / 5b (Halt 5 + tests)

- T5a: BP5 verifications. Halt 5.
  - Verify vertex-order chain Overture -> sub-A -> sub-C.
  - Verify sub-D `round()` mechanism.
  - Reviewer locks outcome before implementation bakes assumptions into encoder/decoder.
- T5b: per-axis test suite implementation after Halt 5 outcome.
- T5a blocks T8.

### Tasks 8-15

- T8: Writer: encoder/decoder, `cells.parquet`, per-tile provenance, region manifest with `vocab_sources`.
- T9: Inline validator.
- T10: Cross-tile validator with BP7 four-test composite and version consistency.
- T11: Pipeline orchestrator.
- T12: CLI scripts.
- T13: Singapore integration tests.
- T14: Empirical gate + round-trip correctness against real cached Singapore.
- T15: final handoff document and close checklist.

## Verification Status At T7 Close

Fresh verification at `c86af13`:

```text
./.venv/bin/python -m pytest tests/data/sub_f/test_rotation.py -q
16 passed

./.venv/bin/python -m pytest tests/data/sub_f/test_manifest.py tests/data/sub_f/test_provenance.py -q
14 passed

./.venv/bin/python -m pytest tests/data/sub_f/test_vocab.py tests/data/sub_f/test_encoder.py -q
63 passed

./.venv/bin/ruff check src/cfm/data/sub_f/rotation.py scripts/sub_f/verify_sub_c_feature_splitting.py tests/data/sub_f/test_rotation.py tests/data/sub_f/test_manifest.py tests/data/sub_f/test_vocab.py tests/data/sub_f/test_encoder.py
All checks passed

git diff --check
passed with no output
```

Do not claim broader suite status without rerunning it.

## Non-Negotiable Discipline For The Next Session

- Stay on `phase-1-sub-F-micro-tokenizer`.
- Do not push, open PRs, or merge.
- For halt tasks, stop at the halt and surface the report. The reviewer approves continuation.
- Use `wip:` commits for halt-pending work and `feat:` commits only after reviewer approval.
- Verify before lock. If a plan assumption conflicts with current source or data, stop and surface the cascade.
- Do not reintroduce sub-F-local BP7 boundary-class derivation. BP7 class authority is sub-E.

End of handoff. Next session should start by reading this file, the spec, the plan, and `reports/2026-05-23-phase-1-sub-F-close-checklist.md`, then proceed with T5a and/or T3a depending on scheduling.
