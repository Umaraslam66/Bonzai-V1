# Sub-F micro-tokenizer — CLOSE handoff (T15): Phase-1 sub-F COMPLETE — 2026-05-30

> **Sub-F is DONE.** T1–T14 + T5a + T5b landed; T15 (this document) seals it.
> This is the entry point for anyone picking up sub-F **without this session's
> context**: what sub-F is, what it locks, what's deferred, every interpretive
> decision, the lessons, and what to do next. Read top-to-bottom. The two
> highest-value sections for a cold reader are **§6 Interpretive Decisions
> Index** (every resolved ambiguity, findable in 30 seconds) and **§7 Lessons**
> (the recurring pattern that dominated this sub-project).

---

## 1. What sub-F is (one paragraph)

Sub-F is the per-cell **micro-tokenizer** (PRD stage four). For each 250m × 250m
lattice cell it emits a flat integer token sequence over a **686-slot vocab**
that encodes the cell's features (roads / buildings / POIs / base) as a four-case
grammar (A/B/C/D) built from a hierarchical coordinate **anchor** + a chunked
**(direction, magnitude)** vertex stream + boundary-reference **`<bref>`** tokens
where a road meets an active cell edge. On-disk output per tile: `cells.parquet`
(exactly 64 rows, one per cell; empty cells carry `token_sequence=[]`) +
`provenance.yaml`; per region: `manifest.yaml`; all behind a `_SUCCESS` marker.
The pipeline composes encode → inline-validate → cross-tile-validate →
provenance → manifest → `_SUCCESS`, halting on any validator failure.

## 2. Branch state + this-session commit log

- Branch `phase-1-sub-F-micro-tokenizer`. **Do NOT merge to `main` here** — the
  human reviewer owns the merge at sub-F review. This handoff + its push close
  the implementation.
- Suite at close: **827 passed, 30 deselected, 1 xfailed**. The 30 deselected are
  `@pytest.mark.slow` (real-cache integration + the OS-pipe round-trip); the
  xfail is a pre-existing unrelated Phase-0 marker. `ruff` clean.
- This session (T12 → T15), most recent first (the T15 doc commit follows this file):

  ```text
  7d7c16f fix(sub_f): T15 discharge close-checklist line 8 — add BP7 vocab source, clear partial_pending_bp7
  77ed2a1 test(sub_f): T14 per-type retention empirical gate — T13's complement, floors read from operative lock
  d9dd46f test(sub_f): T13 Singapore integration — consolidate real-cache legs into one fail-loud suite
  ac009dd feat(sub_f): T12 CLI scripts (derive/validate/encode/decode) — per-feature inspection contract + Step-0 stale-snippet fixes
  ```
  (Prior session HEAD `7b21f2d`; the two Halt re-locks landed earlier at `c1eb2a1` (Halt 4) and `9a7d347` (Halt 2). Spine tasks T8.1–T11 at `b913611`…`1adf360`.)

## 3. Task status

**DONE: T1–T14 + T5a + T5b.** T15 (this handoff) closes sub-F.

| Task | Delivered |
|---|---|
| T1 | BP1 semantic vocab floor (`semantic_vocab.yaml`) |
| T2 | BP2 encoding primitives (`encoding_primitives.yaml`) |
| T3a/b/c | Stage 1+2+3 + BP3 budget (Halt 4) |
| T5a/T5b | BP5 vertex-order verification + per-axis determinism suite (`test_per_axis_determinism.py`) |
| T6 | BP6 six-axis version manifest |
| T7 | BP7 boundary-reference vocab (`boundary_reference_vocab.yaml`) |
| T8.1–T8.8 | Writer: `vocab.py`, `io.py` (`CELLS_SCHEMA`), `encoder.py` (canonicalize + 4-case grammar + §3.5 chunking), `boundary_contract.py` (source-derived sub-E reader), `decoder.py`, `pipeline_writer.py` (`encode_tile`) |
| T9 | Inline validator (`validator_inline.py`) — schema/rows/empty-cell/derivation/vocab-membership/sha |
| T10 (+backfill) | Cross-tile validator (`validator_cross_tile.py`) — BP7 four-test composite + version + sha-uniqueness + all-64-cells |
| T11 | Orchestrator (`pipeline.py::derive_region`) — halt-on-fail + restartable + α-drop wiring |
| T12 | CLIs `scripts/sub_f/{derive,validate,encode,decode}.py` |
| T13 | `test_singapore_integration.py` — consolidated real-cache integration (fail-loud) |
| T14 | `run_empirical_gate.py` + `test_empirical_gate.py` — per-type retention gate |

## 4. Locked values — read the OPERATIVE block, never the historical shadow

| Locked value | OPERATIVE source | NOT (historical shadow) |
|---|---|---|
| BP3 budget **5,899 raw / 6,016 padded** | `sequence_length_analysis.yaml lock.elbow_budget_{raw,padded}_tokens` | top-level `budget_surface` 5792/5888 (pre-chunking audit; see its `_prechunking_analysis_note`) |
| Per-type retention floors **roads 0.9936 / buildings 0.9889 / pois 0.9027 / base 0.9992** | `lock.retention_floors_per_type[*].floor` | `retention_defaults_per_spec_7_5` 0.999/0.99 (§7.5 audit trail) |
| Long-cell diagnostic **5,760** (= padded − 256) | `lock.long_cell_diagnostic.threshold_tokens` | 5,632 (pre-Halt-4) |
| BP2 **direction count 360** (1° bins), IDs **511–870** | `encoding_primitives.yaml proposed_lock.revisit_2026_05_29` + `sentinel_inventory.yaml` | `approved_lock` direction 48 (SUPERSEDED); IDs **396–443 RETIRED** (`direction_v1_deprecated`, never reused) |
| Round-trip thresholds **position p99.9 ≤ 4.8m / angle p95 ≤ 4.0°** | `proposed_lock.revisit_2026_05_29` (re-derived FRESH at N=360; measured 3.7m / 3.0°) | `approved_lock` 4.8m-as-p95 / 7.5° (1k-sample, superseded) |
| On-disk vocab **686 slots** | `load_sub_f_vocab()` / `sentinel_inventory.yaml` | 374 (pre-Halt-2) |
| Structural sentinels `<feature>`=**509**, `<feature_end>`=**510**; BP7 brefs **1500–1507** | `encoder.py` / `boundary_reference_vocab.yaml` | — |
| SOURCE `VersionRef.value` = `overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>` | `versions.py::_encode_source_version` | — |

## 5. The two Halt revisits absorbed this session

- **Halt 4 (`c1eb2a1`)** — BP3 budget re-lock for the §3.5 encoder-chunking discovered missing at T8.7 (fix `86f0c99`). 5,792/5,888 → **5,899 / 6,016**. Per-type floors recomputed from the chunked padded cut (all drifted up/flat). Long-cell re-anchored 5,632 → **5,760**. Per-feature cost pinned in `token_cost.py` + `test_token_cost.py`.
- **Halt 2 (`9a7d347`)** — BP2 `direction_count` 48 → **360** (1° bins; the only mechanism improving BOTH position and angle without regressing). Direction IDs relocated append-safely **396–443 → 511–870** (old range retired). Round-trip thresholds re-derived FRESH (position p99.9 ≤ 4.8m, angle p95 ≤ 4.0°). On-disk vocab 374 → **686**. BP3 budget UNAFFECTED (tokens-per-pair unchanged).

## 6. INTERPRETIVE DECISIONS INDEX

Every resolved ambiguity where the spec/plan underspecified a load-bearing point.
Format: **decision** — why — what forces a v2 revisit — §13.1 cite.

1. **Per-cell `provenance_sha256` = content-equivalence anchor** (hash of emitted tokens + cell coords), NOT a cross-layer provenance chain. *Why:* byte-determinism (T5b) gives the v1 integrity property; a chain to sub-C/sub-E lineage is v2-shaped. Per-cell shas legitimately collide across tiles for byte-identical cells (T9 checks format only; T10 sha-uniqueness is tile-level). *v2 trigger:* cross-layer lineage is needed. §13.1 "T11 dispatch → §4.2 per-cell provenance_sha256" row; commit `e556c31`.
2. **`encode.py` / `decode.py` = per-feature DEVELOPER-INSPECTION tools**, NOT pipeline consumers (the pipeline persona uses `derive.py` + `validate.py`). Bref flag grammar = `DIR:CLASS` via `resolve_bref_tag`; decode emits and encode consumes a **bare geometry** (no Feature wrapper) so `decode | encode` round-trips. `--cell-origin` is a CLI-side pre-translation (`encode_feature` has no such param). *v2 trigger:* a real consumer needs per-cell or batch encode/decode. §13.1 "T12 dispatch → CLI user-contract" row; commit `ac009dd`.
3. **Bref vertex position = v2-deferred** (§1.4 scope lock #1). v1 ships class-only `<bref_DIR_CLASS>`; the crossing-position vertex is dropped. Honest imprecision bound = `cell_extent/2 = 125m`; **NOT round-trip-gated** — a NEGATIVE test in `test_decoder.py` forbids silently re-adding the gate. *v2 trigger:* sub-E-v2 ships exact crossing positions (joint sub-F-v2 + sub-E-v2). §13.1 "T8.7 plan-write → BP7 bref vertex" row.
4. **Two accepted v1 known-losses.** (a) Right-angle **catastrophic** corner loss 0.22% accepted (BP2; categorical, post-roundtrip deviation > 45°). (b) ~**14m position MAX** for long multi-vertex features accepted (Halt-2; a *continuous* accumulation floor inherent to direction-quantize + single-anchor — NOT the categorical right-angle loss; do not conflate). *v2 lever:* re-anchor (scoped and REJECTED at Halt 2 — it regresses the angle axis). §13.1 "Halt 2 revisit → BP2 direction" row + `encoding_primitives.yaml`.
5. **Open-polyline direction PRESERVED** (this DOF was dropped from canonicalization at Halt 5). *Why:* a BP1 grep found NO direction-encoding tokens (no `oneway`, `waterway`-flow, `cycleway:left/right`); canonicalizing traversal direction would silently destroy recoverable OSM semantics. Closed LineStrings (roundabouts) also preserved (conservative). *v2 trigger:* BP1 adds direction-encoding tokens. §13.1 "Halt 5 follow-up → BP5 open-polyline" row.
6. **α-truncation (tail-cell rejection), not β (within-cell tail-drop)** for v1. *Why:* β would tension the just-locked §5.2 feature-iteration order + BP5 canonical form (a Halt-5 cascade). The α-drop report makes the β decision data-driven for the training-scaffold sub-project. §13.1 "Halt 4 ratification → BP3" row.
7. **vocab_sources_status = `complete`** (BP1/BP2/BP4/BP7) at sub-F close — BP7's source (`boundary_reference_vocab.yaml`) was added once T7 locked it (close-checklist line 8, discharged at T15 `7d7c16f`).

## 7. LESSONS — the recurring pattern (worked examples)

**Name it: _ambiguous-spec-resolved-silently_.** Once a spec is mature and
heavily-locked, the load-bearing risk shifts from *"is the design right"* to
*"how is an underspecified point INTERPRETED between locks."* At this lock
density, **Step-0 reverify is mandatory, not optional** — every dispatch surfaces
at least one interpretive ambiguity or a stale-since-relock value. *"It's just
CLI wrappers / just integration tests / just the empirical gate"* is **exactly**
when the guard must not drop. This session caught **nine** instances:

1. **Unread-source inference** (T8.5) — inferred the sub-E schema instead of reading source-of-truth.
2. **Spec clause absent from plan; happy-path tests miss it** (T8.7) — encoder dropped §3.5 chunking; synthetic tests never hit a >32m segment.
3. **Lock validated on a non-representative sample + metric mismatch** (Halt 2) — 4.8m was a 1k-sample p95 (no >32m segments) enforced as a max.
4. **Plan code-snippets stale after later re-locks** (T8.8) — vocab 374 vs 686; `bytes()` vs `struct.pack`; 3-col vs 7-col sub-E fixture.
5. **A gate that can't fail the regime it guards** (Halt 2 angle) — passed identically broken (N=48) and fixed (N=360).
6. **Dispatch framing silently narrows scope** (T10) — BP7 emphasis let sha-uniqueness + all-64-cells slip; caught at T11 Step-0, backfilled.
7. **The dispatcher's own contract is unimplementable-as-written** (T12) — `--inbound-bref CLASS` can't work (`encode_feature` does `tag_to_id[...]`; a bref needs dir+class); `resolve_bref_tag` returns `None` (doesn't raise) on a bad class.
8. **The REUSE target is stale post-relock** (T13) — `scope_halt2_*.py` are hardcoded 48-bin; reusing them would re-measure at the rejected quantization. Plus four scattered EMPTY skip-stubs the plan would have duplicated → consolidated.
9. **A later task's plan re-implements what an earlier task just consolidated + hardcodes superseded thresholds** (T14) — plan re-did round-trip/BP7 that T13 consolidated, and hardcoded the pre-Halt-4 §7.5 floors (pois 0.99 vs operative 0.9027).

*(Plus the T15 Step-0 catch: a **mandatory** close-checklist item (line 8) silently undischarged across T7→T14 — the dishonest-sign-off-by-default, caught before sealing.)*

**Operating rule:** at every dispatch boundary, hunt the ambiguity in Step-0;
read CURRENT source for the interfaces the task composes (and for any helper you
plan to *reuse* — "proven once" ≠ "current"); resolve load-bearing
interpretations OUT LOUD (ask, or record a §13.1 entry), never by silent default.
Read locked values from their **operative** source — never hardcode a lock in a
test (a guard that reads the lock can't drift from it). Durable memory:
`feedback_ambiguous_spec_resolved_silently` (umbrella, 9 instances) +
`feedback_reverify_plan_snippets_at_dispatch`, `feedback_lock_and_guards_travel_together`,
`feedback_gate_must_distinguish_regimes`, `feedback_test_spec_not_just_plan`,
`feedback_sample_regime_blind_locks`.

## 8. Deferred items — every one carries an explicit condition

**sub-E-cache-gated (real-data; all fail-loud `@slow`, run when the sub-E cache regenerates):**
- Round-trip re-measure (position p99.9 / angle p95) — `test_singapore_integration.py::test_singapore_roundtrip_position_and_angle`. **Do NOT** reuse `scope_halt2_*.py` (48-bin-stale).
- First-real-read without `SubEContractViolation` + encode-layer well-formedness — `::test_singapore_encode_layer_real_sub_e`.
- T11 real-region derive — `::test_singapore_end_to_end_derive`. Determinism (same/fresh) — `::test_singapore_determinism_*`. Cross-tile BP7 composite — `::test_singapore_cross_tile_composite`.
- **T3c stage-4 empirical re-measure** (replace the §7.2 `0.7 tok/cell` formula) → if it diverges >2 tok/cell, Halt-4-revisit candidate. The **retention gate** (`test_empirical_gate.py`) must then re-run with empirical stage-4 + the empirical-stage-4 golden committed.
- α-drop warning-band trigger evaluation on real Singapore (count >14 OR >0.1% OR stage-4 >2 tok/cell OR 3 monotonic increases).

**Human-owned (Phase-1 level):**
- **Motorway tiering** — sub-E tiers `motorway` MINOR (scoped accept, NOT data-validated; the green T10/T14 tests pass on *agreement*, they do not validate MINOR is right). Decide whether the sub-E `motorway`→MAJOR fix lands before the first Leonardo run.
- **Cross-tile-boundary bref symmetry** — within-tile only in v1 (external edges are sub-E non-active→NONE→no bref). *Trigger:* if sub-E makes external edges active (e.g. the motorway→MAJOR fix routes an arterial across a tile boundary), T10 needs cross-tile-boundary symmetry (needs the sub-D inter-tile neighbour graph).
- **β-upgrade re-eval** (training-scaffold) when sub-E + multi-region data exist; the Singapore α-drop set is dominated by cells 15–52% over budget, so β recovers little.

**Code-debt:**
- Fixture dedup: `_make_full_tile_contract_rows` is triplicated (`test_pipeline_writer.py` / `test_per_axis_determinism.py` / `test_boundary_contract.py`'s `_make_full_tile_rows`) → extract to `tests/data/sub_f/conftest.py` (lock-and-guards: a sub-E schema bump must not leave one copy stale).
- §9.1–§9.5 line-number cite sweep (prefer content-anchored cites).

## 9. Document-in-handoff obligations (close-checklist lines 9–13 — discharged here)

- **SOURCE hard-dependency:** sub-F SOURCE derivation hard-depends on sub-C manifest field names `sub_c_schema_version` and `initial_extraction.commit_sha` (`versions.py::load_sub_f_source_version`). A future sub-C schema bump or field rename **must** update `src/cfm/data/sub_f/versions.py`.
- **SOURCE `VersionRef.value` canonical format:** `overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>` (scalar-encoded because sub-D `VersionRef.value` is scalar-only).
- **sub-E-v2 candidate:** refine highway tiering in the sub-E grouping map (`motorway`→MAJOR; decide non-vehicular way handling). Sub-F BP7 inherits this through architecture (b) — faithful passthrough.
- **sub-E-v2 candidate:** evaluate whether same-edge MultiLineString / multi-part road crossings need richer than one-class-per-edge representation.
- **Cross-sub-project coupling:** sub-F BP7 classes are **100% sub-E-derived**; any sub-E `boundary_contract` schema or semantic change propagates directly into sub-F BP7 output, and the training-scaffold sub-project must know this.

## 10. Dual-lock read hazard (forward-warning — structural, not task-specific)

Both companion lock YAMLs carry a **historical block shadowing the operative one**.
A naive top-level / `approved_lock` read returns the SUPERSEDED value:
- `encoding_primitives.yaml`: read `proposed_lock.revisit_2026_05_29` (direction 360 / angle 4.0), **not** `approved_lock` (48 / 7.5).
- `sequence_length_analysis.yaml`: read `lock.*` (5,899/6,016 + floors), **not** the top-level `budget_surface` / `retention_by_quantile` (pre-chunking 5,792/5,888).

§13.1 "T14 dispatch → … DUAL-LOCK read forward-warning" row.

## 11. Ledger-hygiene observations (non-gating)

- §13.1 has ~26 table rows; the closing "Fifteen cross-bite-point revisions"
  prose counts the cross-bite-point-*proper* subset (the remainder are §13.5
  protocol-bump candidates, cite drifts, and honesty notes). Pre-existing
  narrative-vs-rowcount looseness; intentionally not rewritten at T15.
- `test_encoder.py::test_direction_bin_lower_at_boundary_48_directions` exercises
  the RETIRED 48-count as a function-generality test (annotated at T15 to say so);
  the LOCKED count (360) is covered by `test_per_axis_determinism::test_direction_bin_locked_360_*`.

## 12. Memories saved/refined this session

- `feedback_ambiguous_spec_resolved_silently` — umbrella, now **9 instances** (the worked examples in §7).
- `feedback_reverify_plan_snippets_at_dispatch` — extended with the **reuse-target staleness** dimension (T13).
- `feedback_lock_and_guards_travel_together` — extended with **read-from-source, don't-hardcode** (T14).

## 13. SIGN-OFF

- **All MANDATORY close-checklist items DISCHARGED:** Halt-2 + Halt-4 revisits; T5b per-axis suite; T11 α-drop wiring; T14 retention gate; **line 8 (BP7 vocab source) at `7d7c16f`**; the five document-in-handoff obligations (§9).
- **All DEFERRED items carry explicit conditions** (§8) — sub-E-cache regeneration, named human decisions, or code-debt tickets.
- **All interpretive decisions indexed** (§6) with rationale + v2-revisit triggers + §13.1 cites.
- Suite green (827 passed); `ruff` clean; lock-and-guards verified.

**Sub-F is sealed.** Next action: push `T12 → T13 → T14 → line-8-discharge → T15`
to `origin/phase-1-sub-F-micro-tokenizer` as one push event (the session's
no-push-until-close discipline; T15 is the trigger). After push: T15 is the
last sub-F task — the next sub-project (training-scaffold) inherits the deferred
items in §8 and the cross-sub-project coupling in §9.

## Authoritative files

- Spec: `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md` (§13.1 ledger)
- Master plan: `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md` (snippets STALE post-re-locks — see §7)
- Locks: `configs/sub_f/{encoding_primitives,sequence_length_analysis,sentinel_inventory,boundary_reference_vocab,semantic_vocab,unknown_family}.yaml`
- Close-checklist: `reports/2026-05-23-phase-1-sub-F-close-checklist.md`
- Spine: `src/cfm/data/sub_f/{vocab,io,encoder,decoder,boundary_contract,pipeline_writer,validator_inline,validator_cross_tile,pipeline,provenance,manifest,versions,token_cost,rotation}.py`
- CLIs: `scripts/sub_f/{derive,validate,encode,decode,run_empirical_gate,compute_alpha_drop_report}.py`
- Integration + gate: `tests/data/sub_f/{test_singapore_integration,test_empirical_gate}.py`

End of sub-F. — T15
