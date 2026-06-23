# Session handoff — mid sub-F, writer→validator→orchestrator SPINE COMPLETE (T8.8→T9→T10→T10-backfill→T11) — 2026-05-30

> Resumable handoff. Assume no prior conversation memory. This document + the
> referenced repo files are the entry point for completing Phase 1 sub-F from
> this state. Clean break point: the writer→validator→orchestrator spine has
> landed, both lock revisits are absorbed, and the §13.1 per-cell-sha decision
> is recorded. Architectural risk is behind us; T12–T15 are a different phase
> (CLI surface, region integration, empirical gate, close).

## 1. Branch state

- Branch: `phase-1-sub-F-micro-tokenizer`
- HEAD: `1adf360` `feat(sub_f): T11 pipeline orchestrator (derive_region) — halt-on-validator-fail + restartable + alpha-drop wiring`
- Worktree: clean. Full repo suite: **797 passed, 4 skipped, 21 deselected, 1 xfailed** (the xfail is a pre-existing unrelated Phase-0 marker). sub-F suite: 245 passed, 4 skipped. ruff clean on all changed files.
- Do NOT push, PR, or merge. Human reviewer owns merge at sub-F close. No new branches.

**Sub-F task status (master plan task index, T1–T15):**
- DONE: **T1–T11** + T5a + T5b. (Writer T8.1–T8.8, per-axis determinism T5b, inline validator T9, cross-tile validator T10 + backfill, pipeline orchestrator T11.)
- REMAINING: **T12** (CLI: derive/validate/encode/decode), **T13** (Singapore integration tests), **T14** (empirical gate, terminal), **T15** (close handoff). Plus the cache-gated close-checklist obligations (§7 below).

**This-session commit chain (most recent first):**
```text
1adf360 feat(sub_f): T11 pipeline orchestrator (derive_region) — halt-on-validator-fail + restartable + alpha-drop wiring
e556c31 docs(sub_f): §13.1 — resolve §4.2 per-cell sha to content-equivalence anchor (v1); defer cross-layer lineage to v2
91ac47a feat(sub_f): T10 backfill — sha-uniqueness + all-64-cells-present cross-tile legs (completes spec §4.7)
dba5965 test(sub_f): T10 rule-isolate cross-reference negative + capture cross-tile-boundary symmetry trigger
2707cbf feat(sub_f): T10 cross-tile validator — BP7 four-test composite (per-leg) + version consistency + class-map gate
3ad706b test(sub_f): T5b review follow-ups — non-vacuous 5a reverse test + drop dead subprocess script + fix tie-break docstring
0154dc4 feat(sub_f): T5b BP5 per-axis determinism suite — real-fn axis tests + cross-process determinism + §5.6 canonical DOFs
1ce2e17 refactor(sub_f): T8.8 code-review follow-ups — feature_class guard + test hardening
1926651 test(sub_f): T8.8 add Case D bref structural test + fix stale bref id comment
6fbb082 feat(sub_f): T8.8 per-tile encode_tile orchestrator + close-checklist BP7 integration obligation
```
(Prior session HEAD was `408f95e`; the Halt-2 + Halt-4 re-locks landed before that at `9a7d347` / `c1eb2a1`.)

## 2. Architectural summary — the spine

The four layers now compose end to end, each enforcing the layer below:

- **T8 family (writer)** — `src/cfm/data/sub_f/`: `vocab.py` (686-slot loader), `io.py` (pinned `CELLS_SCHEMA` + `write_cells_parquet`, 64-row invariant), `encoder.py` (`canonicalize_geometry` 3-DOF canonical form + 4-case grammar A/B/C/D + §3.5 magnitude chunking + `encode_cell`), `boundary_contract.py` (T8.5 source-derived sub-E reader, `load_boundary_contract`, `resolve_bref_tag`), `decoder.py` (`decode_feature` + `serialize_geojson`, Case-A round-trip gate), `pipeline_writer.py` (T8.8 `encode_tile` — per-tile sub-C+sub-E → cells.parquet, incl. the per-cell content sha).
- **T9 inline validator** — `validator_inline.py` `validate_inline(cells_path)`: per-tile contract (schema conformance, 64 rows, empty-cell biconditional, `cell_slot_index == cell_i*8+cell_j` derivation, **token-vocab MEMBERSHIP** not just 0..1599 bound, sha format). The membership check is what makes the Halt-2 direction relocation ENFORCED: a retired-range token (e.g. 420 ∈ 396–443) is rejected because 396–443 is absent from the 686-slot vocab. 14 tests, 12 adversarial.
- **T10 cross-tile validator** — `validator_cross_tile.py` `validate_cross_tile(sub_f_region_dir, sub_e_region_dir)`: region-level. SEVEN legs now (after backfill): BP7 four-test composite [cross-reference, within-tile **symmetry**, non-road non-emission, coverage] + version-manifest consistency + **sha-uniqueness across tiles** + **all-64-distinct-cells-present**, PLUS a standalone Gate-6 BP1→sub-E class-map test. Every leg is a separate `_check_*` function so each adversarial negative is rule-isolated (proven by leg-neutering).
- **T11 orchestrator** — `pipeline.py` `derive_region(cfg)`: require sub-D `_SUCCESS` → require sub-E `_SUCCESS` → per tile {`encode_tile` → `validate_inline` → write `provenance.yaml`} → `validate_cross_tile` → `build_region_manifest` → write `manifest.yaml` → **`_SUCCESS.touch()` LAST**. Halt-on-validator-fail (any validator exception propagates → no manifest, no `_SUCCESS`, no partial output). **Restartability test binds**: clean run byte-snapshot → poisoned run failing after tile 1 is on disk → clean re-run into the same partial dir is byte-identical (no resume-from-partial). Also wires the α-drop diagnostic report.

Every threshold-bearing leg has ≥1 adversarial negative; the review standard throughout was **leg-neutering** (no-op the target check → confirm the negative no longer raises → proves rule isolation, not just "something fires"). This caught a real double-firing bug in T10's cross-reference negative (dba5965).

## 3. Lock revisits absorbed (both landed before this session's HEAD; live now)

- **Halt 4 re-lock (commit `c1eb2a1`)** — BP3 budget re-locked for the §3.5 encoder-chunking discovery: **5,899 raw / 6,016 padded** (was 5,792 / 5,888). Per-type floors recomputed (all drifted up/flat). Long-cell diagnostic re-anchored to **5,760** (2 padding blocks below 6,016). Per-feature cost pinned in `cfm.data.sub_f.token_cost` + `test_token_cost.py`. Source: `configs/sub_f/sequence_length_analysis.yaml lock.revisit_2026_05_29`.
- **Halt 2 re-lock (commit `9a7d347`)** — BP2 `direction_count` 48→**360** (1° bins, the only mechanism improving BOTH axes). Direction ID block relocated **396–443 → 511–870** append-safely (old range RETIRED to `direction_v1_deprecated`, NOT reused). Round-trip thresholds re-derived FRESH: position **p99.9 ≤ 4.8m** (measured 3.7m), angle **p95 ≤ 4.0°** (measured 3.0°). On-disk vocab 374→**686**. BP3 budget UNAFFECTED (tokens-per-pair unchanged). Source: `encoding_primitives.yaml proposed_lock.revisit_2026_05_29` (the OPERATIVE lock — the top `approved_lock` 48 is HISTORICAL/superseded) + `sentinel_inventory.yaml direction_relocation_2026_05_29`.

## 4. §13.1 interpretive decisions recorded (read these before touching provenance / bref)

- **Per-cell `provenance_sha256` = content-equivalence anchor (v1)** — commit `e556c31`. Spec §4.2 "sha chain anchor" was ambiguous between content-equivalence (what was emitted) and provenance-chain (upstream lineage). Resolved to content-equivalence: the value `encode_tile` writes — `sha256(struct.pack(">{n}H", *tokens) + bytes([cell_i,cell_j]))` — IS the v1 anchor; T11 KEEPS it (does NOT recompute). The "stub for Task 11 to overwrite" framing is DROPPED. T11 owns only the TILE-level provenance.yaml sha + manifest. **Per-cell shas legitimately collide across tiles for byte-identical cells** (content-equivalence, NOT a unique id; T9 checks format only, T10 sha-uniqueness is tile-level). Cross-layer lineage chaining deferred to v2. Step-0 reverified byte-determinism (endianness pinned `>H`; integer coords) before locking the interpretation.
- **Bref vertex position v2-scoped (Halt 5 / T8.7 era)** — Cases B/C/D bref-vertex position is NOT round-trip-asserted in v1 (§1.4 scope lock #1 + §13.1). The honest v1 imprecision bound is `cell_extent/2 = 125m`; the ~14m position **max** for long multi-vertex features at N=360 is documented v1 known-loss (continuous accumulation floor, NOT the categorical right-angle known-loss). A NEGATIVE test in `test_decoder.py` asserts bref vertices are not round-trip-checked so a future contributor can't silently re-add the gate.

## 5. The recurring pattern — now at SIX instances; umbrella memory created

This session's dominant theme: with the spec mature and heavily-locked, the risk has shifted from "is the design right" to "how is an underspecified point INTERPRETED between locks." Six worked instances:
1. T8.5 — inferred sub-E schema vs reading source of truth.
2. T8.7 — §3.5 chunking absent from plan; synthetic tests never exercised >32m segments.
3. Halt 2 — sample-regime-blind (4.8m was a 1k-sample p95, no >32m segments) + metric mismatch (p95 derived, enforced as max).
4. T8.8 — plan drift (vocab 374 vs 686; `bytes()` vs `struct.pack`; 3-col vs 7-col sub-E fixture). T5b — `direction_count=16` stale placeholder.
5. Halt 2 angle test — a gate that passed identically broken (N=48) and fixed (N=360).
6. T10 — dispatch framing silently narrowed scope (BP7 emphasis → sha-uniqueness/all-64-cells slipped; caught at T11 Step-0, backfilled).

New umbrella memory **`feedback_ambiguous_spec_resolved_silently`** names the meta-pattern and references the five existing worked-example memories. The operating rule: at every dispatch, hunt for the ambiguity in Step-0; resolve load-bearing interpretations OUT LOUD (ask the human or record a §13.1 entry), never by silent default. **Expect each of T12–T15 to surface at least one.**

## 6. T12 dispatch posture (NEXT — fresh context recommended)

- Files: `src/cfm/data/sub_f/cli/{derive,validate,encode,decode}.py` (or per the master-plan Task 12 file map; verify) + tests.
- **"Glue over verified pieces" is the failure-mode TRIGGER, not reassurance.** Same shape as T8.6 looking like smaller-scope semantic work right before the empty-cell question surfaced. Hold the SAME elevated halt posture as T8.6/T8.7 — do not let "it's just CLI wrappers" lower the guard.
- The CLI surface is where **v1 user contracts get encoded** (arg names, exit codes, output format, stdout/stderr shape). Each is a small spec decision. **Halt on anything underspecified rather than picking a default** — these are exactly the `feedback_ambiguous_spec_resolved_silently` decisions.
- **Step-0 reverify is mandatory and the snippets are GUARANTEED stale:** every function the CLIs wrap changed shape this session — `encode_tile(sub_c_features_parquet, sub_e_boundary_contract_parquet, out_cells_parquet)`, `derive_region(PipelineConfig)` (config now includes `sub_e_region_dir` + `extracted_utc`), `validate_inline(cells_path)`, `validate_cross_tile(sub_f_region_dir, sub_e_region_dir)` (TWO args), `decode_feature(tokens)` / `serialize_geojson(geom)`. Read current signatures; do not trust the plan's Task 12 snippet.
- Check for a sub-E CLI precedent (`src/cfm/data/sub_e/` or `scripts/`) for the arg-shape / entrypoint convention before inventing one.

## 7. Close-checklist carry-forwards (`reports/2026-05-23-phase-1-sub-F-close-checklist.md`)

Cache-gated (mostly un-skip-when-sub-E-regenerates / first-Leonardo-run):
- Real-data round-trip re-measure: position honest p99.9 ≤ 4.8m AND right-angle-corner post-deviation p95 ≤ 4.0° (the angle gate has NO synthetic unit test — binds only via YAML + real-data measurement).
- Un-skip real-region/real-sub-E integration stubs: T8.7 real-data round-trip, T8.8 `test_encode_tile_against_real_sub_e_singapore`, T11 `test_derive_region_against_real_singapore`, T5b §5.5 real-Singapore determinism leg. Each is a `@pytest.mark.skip("awaiting … cache regeneration")` stub today.
- T8.5 sub-E first-real-data read without `SubEContractViolation`; T3c stage-4 empirical ratio; **BP7 multi-part / Multi\* code-inferred → verify against real sub-E output**; motorway/MultiLineString spot-checks.
- α-drop wiring DISCHARGED (T11) but the trigger-evaluation-on-real-data piece stays open: on the first real region run, emit + evaluate the warning-band cell count against the locked action contract (>14 cells OR >0.1% OR sub-E stage-4 >2 tok/cell OR 3 consecutive monotonic increases).
- **Cross-tile-boundary bref symmetry** (T10 deferral): within-tile only in v1 (external edges are sub-E scope=non-active→NONE→no bref). TRIGGER: if sub-E makes external edges active (e.g. motorway→MAJOR across a tile boundary), T10 needs cross-tile-boundary symmetry (needs the sub-D inter-tile neighbour graph). Tied to the motorway-tiering decision.
- **Motorway tiering** (Phase-1-level, human-owned): sub-E tiers `motorway` as MINOR for v1 (scoped accept, NOT data-validated). The T10 cross-reference test passes whenever encoder + contract AGREE on MINOR — it does NOT validate MINOR is semantically right. Decide before first Leonardo run whether the sub-E `motorway`→MAJOR fix lands first.
- Test-fixture dedup: the valid 144-row sub-E contract builder is now triplicated across `test_boundary_contract.py` / `test_pipeline_writer.py` / `test_validator_cross_tile.py` (and used by `test_pipeline.py`). Extract to a shared `tests/data/sub_f/conftest.py` fixture.
- §9.1–§9.5 line-number cite sweep (backlogged; prefer content-anchored cites per `feedback_content_anchored_cites`).
- Remaining work: T13 (Singapore integration), T14 (empirical gate, terminal), T15 (close handoff).

## 8. Memories saved/refined this session

- `feedback_decompose_verification_debt_before_inferring` (prior; worked example #1)
- `feedback_test_spec_not_just_plan` (prior; worked example #2)
- `feedback_sample_regime_blind_locks` (prior; worked example #3)
- `feedback_reverify_plan_snippets_at_dispatch` (created last session, **refined this session** with own-session-generated staleness dimension; worked example #4)
- `feedback_gate_must_distinguish_regimes` (prior; worked example #5; the leg-neutering review standard operationalizes it)
- `feedback_ambiguous_spec_resolved_silently` (**NEW this session** — the umbrella over the five above; the meta-pattern at six instances)

## Authoritative files

- Spec: `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md` (§13.1 ledger now has 13 cross-bite-point revisions incl. the per-cell-sha resolution)
- Master plan: `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md` (Task 12 §, Task 13 §, etc. — snippets STALE post-re-locks)
- T8 writer plan: `docs/superpowers/plans/2026-05-28-phase-1-sub-F-task-8-writer.md`
- Locks: `configs/sub_f/{encoding_primitives,sequence_length_analysis,sentinel_inventory,boundary_reference_vocab,semantic_vocab,unknown_family}.yaml`
- Close-checklist: `reports/2026-05-23-phase-1-sub-F-close-checklist.md`
- Spine code: `src/cfm/data/sub_f/{vocab,io,encoder,decoder,boundary_contract,pipeline_writer,validator_inline,validator_cross_tile,pipeline,provenance,manifest,versions,token_cost,rotation}.py`

End of handoff. Next session: read this + the spec/master-plan + the locked YAMLs, then dispatch T12 against fresh context with the elevated halt posture (glue-over-verified-pieces is the trigger, not reassurance) and a mandatory Step-0 signature re-verify.
