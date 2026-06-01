# Phase-1 Training Scaffold — design

**Status:** design locked via brainstorm 2026-06-01; awaiting spec review → implementation plan.
**Branch:** `phase-1-training-scaffold` (local-first; merge to `main` + push at sub-project end once suite-green + `reports/` summary written; PR optional per CLAUDE.md).
**Protocol:** `docs/protocols/sub-project-planning-protocol-v3.md` (six gates + six principles + §9 construction-identity exclusion + §10 freeze-gate principles).
**Predecessor:** eval-set-generation (CLOSED 2026-06-01, `7db977c`). This sub-project is the named successor that owns the three carry-forward triggers (`known_issues #12`).

## 1. Goal & scope boundary

Make **one tiny micro-generator model train end-to-end on Singapore cell tokens and produce one real eval number on the frozen holdout, reproducibly**, with the three carry-forward entry conditions wired in. This closes — for the first time on this project — the loop `tokens → train → decode → eval`. Everything upstream (sub-C/D/E/F/G, the frozen eval set) is built and validated; everything downstream of the first run is volume, not novelty.

**The scaffold proves the LOOP, not the architecture, not the model quality.** It is the smallest vertical slice through data → model → train → eval with minimum new machinery (brainstorm Topic 2, Option 1: thin vertical slice).

### Explicitly OUT of scope — named follow-ons (not deferred-vaguely, named)

- The Phase-2 **bake-off** (4 architectures × 3 scales, ~1,500 GPU-h, PRD §11).
- The **deferred eval-harness depth** (`known_issues #12`, spec §7): KS/Wasserstein distance vs model output, tokenizer-on-**model** side of R2, simulation-viability execution, conditioning-compliance **scoring**, model-scoring orchestration.
- The **macro planner** + **boundary-contract conditioning** + **cell-stitching** (PRD §4 — the architecture's core innovation).
- **Generalization** metric (needs a held-out region D; single-region Singapore cannot test it).
- **Second-region extraction** (the trigger-3 escalation; the deferred B-decision).

### UNSCORED-in-slice — named, never implied by a passing number

A green per-cell eval means "the micro generator emits decodable, locally-valid cells." It does **NOT** mean "the tile generator works." The following capabilities are unproven by the slice and stated as such (the unscored-not-passing discipline at slice scope): **tile-level cell-to-cell coherence, boundary-contract stitching, macro-planner conditioning.**

## 2. Topic 1 — known_issues #4 resolution (verify inside scaffold; no predecessor sub-project)

**Finding (verified by source-read 2026-06-01):** there are two tokenizers. `cfm/tokenizer/encode.py` (the Phase-0 proof-of-concept; axis-aligned rectangles, dyadic moves) is the file `known_issues #4` points at — but it is imported **only** by `scripts/smoke.py` and is **not** on the training path. The Phase-1 micro-tokenizer is `cfm/data/sub_f/encoder.py`. Sub-C deliberately stores raw not-in-vocab building/POI class values on disk (`sub_c/policy.py:205`, "tokenizer maps to `__UNK__` at encode time"). The sub-F encoder, built *after* #4 was filed, already handles this in `_resolve_semantic_tag_to_token_id` (`encoder.py:279-281`): a non-sentinel value missing from BP1 is bucketed to the `<unknown_KEY>` BP4 family ("cascade #7"). It raises `KeyError` (`encoder.py:283`) only when a *key* has no unknown-family slot at all. So #4 is very likely already solved on the path training uses, and filed against a file training does not reach.

**Resolution (verify-before-lock, NOT close-on-read):** the scaffold's first task is a Gate-6 verification test. Three assertions:

1. **Non-vacuous coverage, counted.** Enumerate every *distinct* building/POI class value present in the frozen Singapore sub-C output; assert the test exercised a non-zero count of *unknown* (not-in-vocab) ones; **report the number**. Proves the test saw the unknown-class regime, not merely that nothing crashed (the synthetic-fixture-blind / didn't-raise≠ran-non-vacuously trap).
2. **Regime-distinguishing negative + positive twin.** A key genuinely absent from the BP4 `<unknown_KEY>` family must still raise `KeyError` (proves the encoder fails loud on true gaps rather than the catch-all silently swallowing everything — which would be *worse* than #4). Its minimally-different twin: a key *with* a BP4 slot resolves cleanly (proves the negative isn't always-raising).
3. **Round-trip asserts the known loss.** `pavilion → <unknown_building> → generic building` is the documented v1 lossy collapse. The test asserts that expected loss **explicitly** (§9 construction-identity: reported-not-gated), so it neither accidentally passes nor fails on it.

**Two branches (a stated precondition, not implicit task numbering):**

- **Branch A — clean (likely):** all three assertions green → **annotate** (do not delete) `known_issues #4`: Phase-0 `encode.py` stays knowingly-unfixed-but-non-training-reachable; the live sub-F path is verified to handle the obligation. #4 closed as superseded by cascade #7.
- **Branch B — gap found:** a real key with no unknown-family slot appears in real Singapore data → inline ~10-line fix in the sub-F encoder (per #4's own estimate) → re-verify all three assertions → then close.

**Why this gates the tier-1 locks (adjustment 1):** Branch B touches the unknown-family vocab, which is **upstream** of the conditioning id-block and the shard format that Tasks 2–3 lock as tier-1 write-once. Locking tier-1 artifacts against a vocab that Task 1 just changed is the deferred-param-binds-a-lock shape (§10.1) one level over. **Therefore: Tasks 2–3 (tier-1 locks) do NOT freeze until Task 1 is SETTLED — clean-closed (Branch A) OR fixed-and-revalidated (Branch B) — exactly as the trigger-2 schema did not freeze until the sequence unit resolved.** This is a stated precondition in the task graph (§12).

## 3. Architecture & data flow

```
frozen holdout_manifest (132 tiles)  ──┐
sub-G validated Singapore (494 tiles) ─┼─► build_training_shards  [byte-deterministic, built TWICE & diffed]
                                       │     • train set = validated − holdout (≈362), BY TILE ID from the frozen manifest (single source)
                                       │     • per-tile shard = FULL tile structure:
                                       │       { tile conditioning, macro tokens,
                                       │         per-cell { tokens, cell_density, boundary contracts },
                                       │         lineage STAMPED from provenance_sha256 }
                                       └─►  training_manifest
                                                │
        ┌────────────────────────────────────────┘
        ▼
  CellDataModule.setup()
        • audit_no_holdout_leak(frozen_manifest, training_reachable)  — fail-closed, ALL ranks,
          HALTS before batch 0 (a failed audit raises → zero training steps execute)
        • internal train/val split carved from the 362, DISJOINT from the 132 by construction
        ▼
  per-cell example = [ conditioning prefix tokens | cell tokens ]   (seeded DistributedSampler, 4×A100 DDP)
        • conditioning VALUES via the SHARED derivation factored out of eval/holdout/labels.read_tile_labels
        ▼
  toy decoder-only transformer (~10–30M)
        • vocab = sealed sub-F vocab + conditioning id-block (separate offset, append-only)
        • next-token CE; loss MASKED on the conditioning prefix; bf16; torch.compile (default-on)
        • checkpoint { weights, optimizer, RNG, step, data-position }  — 30-min, bit-identical resume
        ▼
  [FINAL EVAL, run ONCE at end] generate cells on holdout-tile conditioning ─► sub-F decoder ─► GeoJSON
        ▼
  decodability rate + OGC-validity (reuse sub_g.seam_decodability) + 90°-corner %   (PER-CELL scope)
        • bref-placeholder collapse excluded via the SHARED D3 instrument (reported-not-gated)
        ▼
  assert_resolution_sufficient(needed_gap)  — built + tested SEAM, marker-sourced, fully activates at bake-off
```

## 4. New code surface

- `src/cfm/data/training/` — `build_training_shards` (materialization + lineage stamping), the locked shard schema, `CellDataModule`, and the **shared conditioning derivation factored out of** `eval/holdout/labels.py`.
- `src/cfm/models/micro_ar.py` — the toy decoder-only micro-generator (candidate-2's micro half).
- `src/cfm/training/` — Lightning module + training loop, checkpointing. (Per PRD §12; a `src/cfm/training/CLAUDE.md` may be added for loop-specific rules.)
- `src/cfm/inference/` — generation + decode-to-GeoJSON.
- `src/cfm/eval/slice_metrics.py` — decodability/validity/corner metrics; `assert_resolution_sufficient` (trigger-3 seam).
- `configs/experiments/training-scaffold-v1.yaml` — pydantic-validated. Config + code commit + data snapshot fully determine the experiment (CLAUDE.md reproducibility mandate).

## 5. §10.1 two-tier lock ledger

Every artifact is classified. Tier-1 = comparability-critical, write-once-ish, shared identically across all 12 bake-off runs — locked NOW to a today-justifiable bar. Tier-2 = slice-local, expected to iterate — built, not locked. The §10.1 trap is **both** directions: locking a slice-local choice as if comparability-critical, AND letting a comparability-critical schema stay provisional because "it's just the toy run."

| Artifact | Tier | Locked to / rationale |
|---|---|---|
| Conditioning-vector **schema** | **1 — lock now** | The existing sub-C/sub-D contract via `read_tile_labels`. Fields: `{population_density_bucket, morphology_stratum = dominant_zoning_class + modal_road_skeleton_class, cell_density_bucket (PER-CELL SCALAR)}` scored + `{region, sub_c morphology_class, coastal_inland_river}` unscored-recorded + deterministic seed. (Externally constrained AND tier-1 → least free of all.) |
| Token **shard format** | **1 — lock now** | FULL tile structure `{tile conditioning, macro tokens, per-cell {tokens, cell_density, boundary contracts}, lineage}` — provisioned to serve all 4 bake-off candidates, NOT just the slice's micro subset. Under-provisioning to the slice's needs is the fatal write-once direction. |
| Conditioning **id-block** (offset + field-value→id mapping) | **1 — lock now** | Separate offset ABOVE the sealed sub-F vocab (sub-F vocab untouched, not reindexed); append-only (a future conditioning dimension appends, never reindexes); the mapping lives in ONE place that both the shard-build and the model read (one-source). |
| **Holdout-source-identity invariant** | **1 — lock now** | The holdout set IS the frozen manifest's 132 tile IDs, addressed BY ID, single source. Both layer-1 exclusion (`494 − 132`) and layer-2 audit key on it. **No "recompute-which-tiles-are-holdout" path exists.** Both defense layers must defend the SAME definition of holdout via independent mechanisms — else they defend different definitions. (Adjustment 2: this is a correctness contract every bake-off run inherits, as load-bearing as the shard format.) |
| **Eval-protocol identity** | **1 — lock now** | The metric set (decodability + OGC-validity + 90°-corner), the per-cell scope, the holdout-touched-once discipline, the bref-collapse-via-D3 exclusion. Must not drift run-1→run-12 or scaling curves aren't comparable. |
| Toy model architecture / scale | **2 — build, iterate** | Slice-local; the slice proves the loop, not the model. |
| Conditioning **entry transform** (prepend + loss-mask) | **2 — build, iterate** | Model-side, OUTSIDE the trigger-2 compared surface. Line: *what* gets prepended (values, tier-1, from `read_tile_labels`) vs *how* it's prepended (tier-2). |
| Metric subset / thresholds | **2 — build, iterate** | The pass bar is a slice sanity floor, iterable. |

## 6. The three entry-condition triggers (`known_issues #12`)

### Trigger 1 — holdout exclusion (live, fail-closed, two defense layers)

Two layers defending **different** failures, both keying on the same frozen manifest (§5 invariant):
- **Layer 1 (expected leak):** training tile set = `validated − holdout`, by ID from the frozen manifest (≈362; the exact count emerges from the set-difference-by-ID at build — see §11 assumption). Exclusion by construction.
- **Layer 2 (unexpected leak):** the tier-1 shard format carries lineage **stamped at build** from `provenance_sha256`; the DataModule **reads** that stamped lineage (never synthesizes it from the path — synthesis makes G-F4 vacuous, the gate-must-distinguish-regimes trap) and calls `audit_no_holdout_leak` at `setup()`, fail-closed, on all ranks, **halting before batch 0**.

Four proven regimes, each with its must-pass twin (Topic-7 pin):
- **F1/F2** — inject a holdout TileRef into a shard's recorded lineage → raises; clean sibling → passes.
- **F4 (critical)** — drop the lineage field → raises on the *absence*; **assert the loader does not backfill/synthesize** (reaches the audit as `None`); present-lineage sibling → passes. This is the test that proves the synthesis trap was *avoided*, not merely *intended*.
- **Clean** — passes with a reported **non-zero** shard count (audited real shards, not zero).
- **Stamped-lineage integrity** — a shard whose lineage points at a real *training* tile passes and is counted (proves "passes" = "audited and cleared," not "audited nothing"; closes the degenerate-stamp loophole).
- **Halt** — a planted holdout ref → `setup()` raises → **zero training steps execute** (a leak trained-through-with-a-log-line is the same unrecoverable contamination).

### Trigger 2 — conditioning vector (one source, proven structurally)

Single source via a function **factored out of** `read_tile_labels` (not called-through-and-hope). The shared derivation extends through to the quantities both consumers (model conditioning, eval conditioning-compliance) compare on; any model-side transform lives inside the shared function or is provably outside the compared surface (the spec states where the shared derivation ENDS). Gate-6 test asserts **same-source** (an identity assertion — builder and `read_tile_labels` resolve to the same derivation call, like sub-G's `is`-lock on the bref predicate — fails the moment someone forks the derivation), beyond the hand-enumerated value cross-reference (which proves agreement *today*).

### Trigger 3 — eval-harness fail-loud resolution seam (built + tested, bake-off-activated)

`assert_resolution_sufficient(needed_gap)` — a real, pure, tested function (the honest form: it *can't* be live with no architecture pair, so it awaits real input, not a stub). Two pins:
- **Marker-sourced, fail-closed on absence** (same shape as G-F4): reads `resolved_gap` (0.076) and `single_region_floor` (0.049) from the frozen eval-set marker; if the marker is missing/unreadable/lacks the fields → **raises**, never defaults permissive or no-ops.
- **Two failure KINDS, distinct messages + escalations:** `needed_gap` in `[0.049, 0.076)` → "this frozen set can't resolve it; a larger/second-region set in principle could" → second-region extraction. `needed_gap < 0.049` → "no single-region set can *ever* resolve this; single-region is fundamentally insufficient" → categorically different escalation. Tests assert the two messages differ and name the right escalation; `0.10` passes; marker-absent raises.

## 7. Sequence unit, model, conditioning entry

- **Sequence unit = CELL** (the micro generator on individual cells, ≤5,760 tokens/cell per the sub-F P99.9 lock). Reuses the sub-F decoder unit, the eval roundtrip unit, and `roundtrip.py`'s per-cell `cell_density` pairing. The whole-tile pure-AR alternative was rejected: 64 concatenated cells = tens of thousands of tokens (worst ~368k), the most memory-bound option, possibly infeasible at toy scale.
- **Model** = decoder-only autoregressive transformer (~10–30M, the simplest building block shared by bake-off candidates 1–3), next-token CE, AdamW + cosine, bf16, `torch.compile` default-on (disable if it breaks, CLAUDE.md), gradient checkpointing if memory-bound, 30-min checkpointing, loss curves to tensorboard, 4×A100 DDP (`feedback_leonardo_full_node`; never 1-GPU jobs).
- **Conditioning entry** = prepended conditioning tokens from the separate append-only id-block; loss masked on the conditioning prefix (given, not predicted); the decoder predicts only over the sub-F vocab range.

## 8. Eval scope & holdout discipline

- Slice eval (on the frozen holdout, run ONCE at the end): decodability rate + OGC-validity (reuse `sub_g.seam_decodability`'s predicate, don't reimplement) + 90°-corner % (the PoC 95% bar). Plus train/val CE loss on the internal split.
- **Selection-loop leak (a SECOND protection the trigger-1 audit cannot reach):** the lineage audit catches holdout in *training data*; it does NOT catch holdout in the *selection loop* (early-stop, best-checkpoint). The internal val split is carved from the 362, disjoint from the 132 by construction; the holdout eval runs once at the end, never as a monitored Lightning val-loop metric. Tested both ways: no holdout id in the val-split manifest, AND the holdout eval is not registered as a val-loop metric (can't drive checkpoint selection).
- **Bref-collapse via the shared D3 instrument:** the bref-placeholder collapse (zero-length crossing roads) is a known v1 limitation that decodes OGC-invalid by design. The slice applies the SAME construction-identity exclusion as sub-G / the eval set (`_is_bref_placeholder_collapse` / `bref_placeholder_rate`), per-instance-exclude + rate-report (D3 — on ungrounded model output the bref is ungrounded). Reported-not-gated; NOT a slice-local validity-minus-bref reimplementation (which would repeat the C2/D2 mistake of penalizing the model for faithfully reproducing the v1 ceiling).

## 9. Test strategy & fixture discipline

- **Synthetic fixtures** build the controlled regime-distinguishing NEGATIVES real data can't reliably produce (planted leak, absent lineage, no-BP4-slot key, conditioning fork, marker-absent). **Real frozen-Singapore slices** exercise COVERAGE/SHAPE (#4 unknown-class enumeration — mandated real by the Topic-1 lock — byte-determinism, Gate-6 cross-references).
- **Every "must-raise" synthetic negative has a minimally-different "must-pass" twin from the same fixture family** (the actual sub-E fix: the fixture proves the code *distinguishes* regimes, not that it merely reacts to malformed input that could raise for the wrong reason).
- **Byte-determinism tests build twice and diff** — never assert-once (a single hash is stable within a run; determinism is across runs).
- **Fast vs slow:** unit tests fast; the end-to-end run is `@pytest.mark.slow`. Validation sequence = `fast_dev_run` smoke first (`num_workers=0`, ≥world_size val shards per `feedback_leonardo_ddp_smoke_pattern`; proves loop + checkpoint + bit-identical resume + decode) → short real run for the eval number.

## 10. Task decomposition (gate→test matrix — the spec backbone)

Each task names its protocol gate and its discrimination test(s). **Dependency precondition (adjustment 1):** Tasks 2–3 (tier-1 locks) are BLOCKED on Task 1 being SETTLED (Branch A clean-closed OR Branch B fixed-and-revalidated), because Branch B mutates the unknown-family vocab upstream of those locks.

| # | Task | Gate | Discrimination test(s) | Depends on |
|---|---|---|---|---|
| 1 | #4 verify → Branch A annotate/close OR Branch B fix-and-revalidate | 6 + §9 | counted non-vacuous unknown enum (real); no-BP4 key raises ↔ has-slot twin resolves; round-trip asserts known lossy collapse | — |
| 2 | Tier-1 lock definitions: conditioning schema, shard format, id-block, holdout-source-identity invariant, eval-protocol | 1 + §10.1 | format provisions FULL tile structure (not slice subset); id-block mapping read from ONE source by both build & model (identity assertion) + append-only (adding a dim appends, never reindexes); holdout-source-identity: both layers reference the SAME manifest object, no recompute path | **1 settled** |
| 3 | Shared conditioning derivation (factor-out of `read_tile_labels`) | 6 + §3 | identity assertion (fork-detection) + hand-enum value cross-ref (no builder in expected) | **1 settled**, 2 |
| 4 | `build_training_shards` (materialize + stamp lineage) | §10.1 + determinism | build-twice-and-diff byte-identical; 362-set BY ID from frozen manifest (no recompute path) | 2, 3 |
| 5 | Holdout audit wiring (`DataModule.setup`) | 2 + 6 | F1/F2 inject↔clean twin; F4 no-synthesis↔present twin; clean non-zero count; stamped-integrity; **halt → zero steps** (all ranks) | 4 |
| 6 | Cell DataModule + DDP + determinism | determinism | val-split disjoint from 132; bit-identical 4→4 resume; sampler seed from recorded config | 4, 5 |
| 7 | Toy micro-generator model | 1 (tier-2) | conditioning prefix masked from loss; predicts only sub-F range | 2 |
| 8 | Lightning training loop + checkpointing | small-before-big | 30-min checkpoint + bit-identical resume | 6, 7 |
| 9 | Inference / decode path | §3 | decode reuses sub-F decoder (one source) | 7 |
| 10 | Slice eval metrics | 2 + §9 | holdout-not-a-monitored-metric; bref-collapse via shared D3 (reported-not-gated); per-cell scope stated | 9 |
| 11 | Resolution seam (trigger 3) | 2 + §10.1/§10.3 | 0.10 pass / 0.06 & 0.03 distinct-message fails; marker-absent raises | — |
| 12 | E2E `fast_dev_run` smoke → short run → `reports/` summary | small-before-big + reproducibility | smoke proves loop+ckpt+resume+decode before the real run; config+commit+snapshot recorded | 8, 10, 11 |

## 11. Risks

- **Torch / `torch.compile` env friction on Leonardo.** The slice uses a plain transformer (Mamba deferred to the bake-off), so `mamba-ssm` is not a slice dependency; `torch.compile` is disable-if-it-breaks (CLAUDE.md).
- **Tiny training set** (~362 tiles → a few thousand cells) overfits fast. Acceptable for a loop-proof; the eval number is a sanity floor, not a quality claim (and tile-coherence is unscored regardless).
- **iCloud editable-install gotcha** (`project_repo_location_icloud`): the env reports the working dir as `~/Projects/Bonzai-OSM`; confirm the venv / `pythonpath = ["src"]` setup before the first run.

**Assumption (verify at Task 4):** the training count ≈362 assumes the 132 holdout tiles ⊆ the 494 sub-G-validated tiles. The build computes the training set as `validated − holdout` BY TILE ID, so correctness does not depend on the assumption — but if any holdout tile is absent from the validated set, the count differs and that itself signals a holdout/validated lineage mismatch worth surfacing (not silently absorbing).

## 12. Decisions log (load-bearing, for future readers)

- #4 is resolved INSIDE the scaffold (verify-before-lock), not via a predecessor sub-project; the live path (sub-F cascade #7) is the verified one; Phase-0 `encode.py` is knowingly-unfixed-non-training-reachable.
- Scaffold = thin vertical slice; bake-off / full eval-harness / macro-planner / stitching / generalization / second-region are named follow-ons.
- Sequence unit = cell (decisive: the 5,760-token/cell budget makes whole-tile infeasible at toy scale; cell reuses decoder/eval/cell_density).
- Tier-1 locks (conditioning schema, shard format, id-block, holdout-source-identity, eval-protocol) freeze only after #4 settles.
- Three triggers wired: stamped-lineage fail-closed audit (live), factored-out one-source conditioning (proven structurally), marker-sourced resolution seam (built + tested, bake-off-activated).
