# Readiness-closure + conditioning-enrichment — design spec (2026-06-10)

**Status:** LOCKED — PI-approved 2026-06-10 (reviewed against the file). All four §5 PI-CALLs
resolved by Umar 2026-06-10:
1. **δ = 0.15** effect-size floor for the recalibrated gate (accepted as recommended).
2. **Both, asymmetric — CHARACTER-anchored** (locked rationale: identity-only conditioning would
   make the bake-off measure memorization of four labels — the Singapore-overfit failure the
   project exists to kill; character conditioning forces genuine generalization, morphology →
   geometry, transferable to unseen cities. Name is a free floor; CHARACTER is the real learning
   signal. Keep the two separable so the bake-off can show character is doing the work, not the name).
3. **#13:** city-name shortcut now; admin_region proper at the already-deferred regen bundle.
4. **Allocation:** T0 is CPU-local and proceeds through the renewal gap; first post-renewal GPU
   job waits for T0 closure including the F8 resume fix.
**δ↔character coupling (PI note, binding on the plan):** δ=0.15 is only meaningful relative to
CHARACTER conditioning — under identity it would measure label-memorization. The recalibrated
gate measures character capture at δ=0.15, and the localization diagnostic picks the character
feature against that same δ.
**Inputs:** the gate-passed audit artifacts (`reports/2026-06-10-readiness-audit-execution-surface-map.md`,
`…-failure-class-enumeration.md` — esp. the §5 gap register, clusters G-A…G-E), the PI's four
carry-forwards (F5/F6 spine + delta-spec correction; gate-(i) calibration as first-class;
F8 resume on the allocation critical path; F15 length-confound + bref quantification folded in),
and a fresh local mining of the gate-(i) result YAML (effect-size sweep — §4.3 table below).
**Baseline:** main @ `77fdb6d` (audit artifacts committed+pushed). Suite 1,343 green at merge;
no GPU spent; bake-off HALTED at T5-reopen.

---

## 0. The shape: two missions, one sequencing spine

**Mission A — readiness closure:** fix the audit's gaps so the *next* latent failure class
cannot exist unexamined on the execution surface (the audit's purpose: close the class, not the
instances).
**Mission B — conditioning:** make the cross-city eval valid, on the PI-mandated two-axis split:
**B-delivery (F6)** — values must reach the model at all; **B-expressivity (F5)** — the values
must carry city character. Separate obligations, separate fixes, separate teeth.

**The sequencing spine is F16 (fix-sequencing is itself a failure class):**
1. **F6 delivery lands BEFORE any GPU training.** Today zero scored checkpoints exist — the
   prefix-scheme change is free now and poisonous later (a slot-trained checkpoint loads
   silently-wrong under value-bearing code; §3.1 scheme-tagging makes that loud forever after).
2. **F8 resume + F17 run-isolation land BEFORE any multi-job or post-renewal training** — the
   allocation soft-ends 2026-06-11; "pause and resume after renewal" is currently a false
   assumption (relaunch restarts at step 0; reports overwrite).
3. **F5 expressivity decides AFTER the localization diagnostic** (§4.2) — characterize before
   recommend; the gate cannot currently say *which* coarseness layer binds.
4. **Gate-(i) recalibration lands WITH the re-derive**, never after (verify-before-lock).
5. **No partial/version-skewed fixes:** any artifact-changing fix applies uniformly or is
   deferred whole (`feedback_uniform_defect_level_no_version_skew`).

---

## 1. Mission A — readiness closure (by gap cluster; every fix ships with teeth)

Tiering: **T0** = before ANY training job (incl. the diagnostic); **T1** = before scored runs /
Task-12; **T2** = deferred, logged with trigger. Every fix carries the discipline pair:
fire-on-bad/pass-on-good + red-on-divergence demonstrated at review.

### A-1. EU-training path (cluster G-B; mostly T0)
- `--region`/`--release` (or config) plumbing into `train_scaffold.py`; both bake-off sbatches
  lose their `b('…','singapore')` literals; a **content test on the sbatches** (region literals,
  required flags+values) so the next stale driver is caught by the suite, not on-node.
- Wire the multi-region union: a production caller for `CellDataModule(training_manifests=[…])`
  over `train_cities(...)` (Task-8's 38 manifests currently feed nothing).
- **Three-way CRS consistency check** (config `projected_crs` == sub-D `region_crs` == on-disk
  dir label) as a script + test over all 42 cities; plus a one-line **meters-unit assertion** on
  `projected_crs`.
- **EU emergence floor**: compute per-region via the existing `holdout_polygons_per_active_cell`
  (region-parameterized, currently zero callers); the hand-typed SG 1.96 dies. Floor arrives via
  config/artifact with provenance, not a CLI literal; `--emergence-floor` absent on a
  cell-generating run becomes LOUD (fail-open closed).
- **EU token-length stats vs `DEFAULT_MAX_CELL_TOKENS=5760`** measured from the Task-8 manifests
  (CPU); drop-rate gets a threshold + action contract (`feedback_diagnostic_threshold_design`).
- Gate-(i) extraction site: CRS regression test + `n_tiles_read/n_tiles_skipped` per city in the
  result YAML (closes the silent-shrinkage regime; T1 — needed at re-derive anyway).

### A-2. Run integrity (clusters G-C + F8/F15/F17; T0 for resume+isolation, T1 rest)
- **F8:** wire `resume.py` → `trainer.fit(ckpt_path=resume_ckpt_path(...))`; checkpoints to
  `$WORK` layout; USR1 trap forwards the signal to ranks (checkpoint-on-signal) and **verifies
  the resubmit succeeded** (no exit-0-on-failed-sbatch — the false-completion family); JOB_DONE
  markers verify end-state (checkpoint exists + report exists), not rc. Proven by the plan's own
  HARD ORDERING: kill→resubmit→continues on a SHORT job before any long run.
- **F17:** training-manifest writes become atomic (tmp+rename — the sub-C crash-safe writer
  pattern exists, reuse it); report filenames keyed `{backbone}-{scale}-{seed}`; per-run
  checkpoint/log dirs.
- **F15:** `--eval-max-new`, `--eval-cells`, emergence-floor provenance, and `--max-len` all
  enter `ScaffoldConfig` (hence the report and checkpoint hparams); a **commensurability
  assertion** relates the three lengths (eval_max_new vs floor-derivation length regime vs train
  max_len) — at minimum: refuse a §2 emergence verdict when generated-length cap < the floor's
  derivation regime, and record the floor's denominator convention (all-non-empty-cells) beside
  the gate's (density-bucketed cells).
- **F9 reader-side integrity (T1):** manifest `manifest_sha256` verified at load (datamodule +
  eval readers); `_EVAL_SET_LOCKED`-presence check before consuming holdout manifests;
  `TRAIN_TOKENS` guard test against the EU marker; `assert_resolution_sufficient` grows a
  region-aware marker argument (obligation (c) lands with the EU recompute at re-derive time).

### A-3. Decision layer (cluster G-D; T1 — none of it runs before Task-12 exists)
The Task-12 runner is built WITH its five paired obligations as tests-first: path==persisted-basis
assertion; 4-city completeness guard; promote-before-`feature_samples` (or `feature_samples`
learns to promote internally — pick one, structurally); `pick_winner`⟷`structural_check_ok`
pairing + minimum-backbones assertion (kills the live single-entry auto-win); n-floors before KS
is decision-bearing. Pre-submit buildability gate (CPU `build_backbone` dry-run in sbatch
preamble). Compile OUTCOME recorded in the report, not intent. Point/bref semantics pinned by
test in `slice_eval` (C4). 1.36/1.358 constants unified or cross-guarded.

### A-4. Cheap test debt (cluster G-E; T0, one sitting)
slice_eval promotion live-call test; `conditioning_gate.py` tombstoned (module-level raise or
deletion — the live plan's import snippet corrected at the same time); `decode_feature` docstring
fixed; `tokenizer.decode` tombstone note; stale sub-C comments; `UNSCORED_V1_DIMENSIONS` either
consumed or dropped; the two reverse-locks flagged for deliberate flipping (EPSG3414-default pin;
emergence fail-open pin).

### A-5. Explicitly deferred (T2, logged with triggers)
Diffusion/mamba eval-hook generalization (trigger: Task-5 builds a second backbone); WorldSizeGuard
fire-regime observation (trigger: first Leonardo session — 2-min single-task srun); EU-scale
per-rank setup memory/time measurement (trigger: first EU training job, measured before the long
run); DistributedSampler padding semantics (trigger: same); `deviation_log` wiring (trigger:
first recipe deviation).

---

## 2. Delta-spec §4 correction (PI carry-forward 1)

The 2026-06-09 delta-spec §4 [PRIOR] — "conditioned on city D means handed D's macro plans" —
is **false of the running system** (true only of the unread `TrainingShard.tile_conditioning`
field). The spec amendment: replace the PRIOR paragraph with the two-axis statement (delivery
absent by slice-v1 decision; expressivity FAILed by gate-(i)), cross-referencing the enumeration
artifact. The gate-(i) closeout gets a one-line erratum pointer (its "conditioning the model is
handed" phrasing), not a rewrite — reports are point-in-time records.

---

## 3. Mission B-delivery (F6) — wire the values (T0)

**3.1 Mechanism.** `flatten_shards_to_cells` and `_generate_and_score` switch from
`build_conditioning_prefix()` (constant slot block) to `build_value_bearing_prefix(...)` fed from
`shard.tile_conditioning` + per-cell `cell_density_bucket` + seed — **REPLACE, never augment**
(slot ids 1508..1515 == field-0 value buckets; the schemes are mutually exclusive by id-space).
A **`conditioning_scheme` field enters `ScaffoldConfig`** → checkpoint hparams + report — every
checkpoint is forever tagged with the prefix scheme it was trained under (the F16 silent-flip
killer). A mutual-exclusivity guard test asserts no example can contain both schemes.
**3.2 Known traps, handled by name:** the `"region"` key collision (`tile_conditioning["region"]`
= admin_region (None for EU) vs `TrainingShard.region` = city name) — disambiguate keys BEFORE
wiring; the two suite tests that LOCK the constant prefix flip red→green as part of the change
(planned test edits, not weakening); generated-side `strata.append(0)` becomes the real per-cell
bucket; micro_ar prefix-mask tests re-run at the live n_cond=512 shape; `_value_bucket` SHA-256
mod 63 hash aliasing gets an **injectivity test over the live value sets** (38 train cities ⇒
~11 expected collisions if region-as-string is ever value-bearing — see §4.4 fork).
**3.3 Teeth.** Red-before: a test asserting model inputs differ across two tiles with different
`tile_conditioning` (fails on the constant prefix today); green-after with the wiring. Plus an
end-to-end: train the smoke on two synthetic tiles with opposite zoning, assert the prefix ids
differ at the collate layer.
**3.4 What delivery does NOT claim:** with today's data, value-bearing delivery hands the model
≤4 informative dims (#13 region None, #22 morphology/era/country/climate constants, density
all-moderate at tile level). Delivery is necessary, not sufficient — that is exactly the F5/F6
independence the PI mandated.

---

## 4. Mission B-expressivity (F5) — enrich what the cards say

**4.1 Evidence anchor (mined 2026-06-10 from the gate-(i) result YAML, local CPU).**
- `building_area_m2` carries the dominant missing character: 136/141 pairs significant, **60
  pairs at KS≥0.20 across 29 strata, 22 pairs ≥0.30** — large, broad, all four cities involved.
  Survives any candidate effect floor.
- `road_length_m` effects are much smaller (168/180 significant but only **17 pairs ≥0.20**;
  39 ≥0.15) — AND this is the bref-contaminated metric (zero-length V=2 collapses + ≤125 m
  placeholder error ride the distributions). **Owed at re-derive:** per-city bref/collapse-rate
  quantification + exclusion-by-construction-identity, then re-verdict road_length. It is
  plausible road_length's failure shrinks substantially once decontaminated; building_area's
  cannot (building blocks carry no brefs — pinned by test).
- Inland strata differentiate more than riverside (meanKS 0.236 vs 0.183 building-area).

**4.2 Localization diagnostic FIRST (CPU, existing artifacts, before choosing the enrichment).**
Gate-(i) conditions on already-collapsed, already-quantized labels, so its FAIL aggregates three
candidate coarseness layers. Re-run discrimination varying ONE layer at a time:
(a) per-cell zoning/density instead of tile-dominant/modal (un-collapse), (b) finer buckets from
the retained raw evidence (`derivation_evidence.parquet` keeps raw `building_footprint_ratio`
etc.) (un-quantize), (c) + candidate new dims (§4.4). The layer whose variation kills the most
discrimination signal is where city character lives — that, not intuition, picks the enrichment.
Zero GPU; held-out tiles only; same BH machinery (with the §4.3 floor applied).

**4.3 Gate-(i) recalibration (PI carry-forward 2 — first-class).**
As built the gate is structurally incapable of PASSing at real n (significance = `p_bh < alpha`
alone; min significant KS in the artifact = 0.0285 at n up to 169k). The reopened gate adds a
**practical-effect-size floor δ**: a pair "really differs" iff BH-significant **AND** KS ≥ δ
(threshold-pairing: the statistical tooth and the practical tooth are separate and both required).
Sweep from the real artifact, to frame the PI decision:

| δ (KS floor) | significant pairs surviving (of 304) | building_area | road_length |
|---|---|---|---|
| 0.05 | 290 | 135 | 155 |
| 0.10 | 215 | 117 | 98 |
| 0.15 | 121 | 82 | 39 |
| 0.20 | 77 | 60 | 17 |

**[PI-CALL #1]** δ is partly a product decision — "what level of city-distinguishability counts
as conditioning-sufficient" for the v1 persona (plausibility for AV/robotics sim, not realism).
Framing: δ=0.10 ≈ "modest-but-visible distribution shift"; δ=0.15–0.20 ≈ "differences a consumer
of generated tiles would notice." Recommendation to react to: **δ=0.15**, justified as ~2× the
SG single-region KS floor (0.049) and the threshold where road_length's (contaminated) tail
separates from building_area's (real) signal — but the number carries Umar's name, not mine.
The recalibrated gate must satisfy `feedback_gate_must_distinguish_regimes`: demonstrate it CAN
pass (synthetic same-character cities at real n) and still fails on the real artifact.

**4.4 Enrichment candidates (decided AFTER §4.2; the fork framed now).**
**[PI-CALL #2] Identity-conditioning vs character-conditioning** — the central fork:
- *(i) Identity:* a per-city/region label the model can condition on (#13 admin_region restored,
  or simply the city name the pipeline already carries as `TrainingShard.region`). Cheap (city
  name needs NO sub-C regen — it is already on every shard); directly closes the "wasn't told
  the city" ambiguity for SEEN cities; does NOT generalize to unseen conditioning (a new city's
  label is untrained).
- *(ii) Character:* continuous/structural per-tile or per-cell features derived from data the
  pipeline already retains — building-footprint statistics beyond the 4-bucket density (the
  mined evidence says building geometry IS the missing character), finer quantization, per-cell
  grid delivery via the provisioned-but-empty `macro_tokens` carrier. Generalizes; costs more
  (derivation + possibly schema); the §4.2 diagnostic tells us which feature actually carries
  the signal.
- Recommendation to react to: **both, asymmetrically** — city-identity as the v1 floor (restores
  a valid per-city bar immediately: a miss can no longer mean "wasn't told which city"), one
  character feature chosen by the §4.2 diagnostic as the v1 ceiling (keeps the bar meaningful
  as generalization, which is what the bake-off claims to measure). Pure-identity risks the
  bake-off measuring memorization-of-four-labels.
**[PI-CALL #3] #13 admin_region scope:** restoring it properly = sub-C regen (data-layer,
weeks-scale per the multiregion feasibility audit) vs the city-name shortcut above (zero regen).
If identity-conditioning is wanted, recommend the shortcut for v1 and defer #13 to the already-
deferred regen sub-project (#16/#17 bundle) — uniform-defect-level rule forbids a partial regen.
**#22 morphology + the three SG-wrong constants:** excluded from any wholesale dict-read by the
delivery wiring (explicit field list, never `dict(**tile_conditioning)`); their data-layer fix
rides the same deferred regen.

**4.5 Expressivity teeth.** The recalibrated gate-(i) re-run (with bref exclusion + coverage
counters) on the enriched conditioning is the acceptance check: **PASS at the PI-chosen δ = the
enrichment worked; FAIL = T5 stays open.** Plus constant-column guards (a conditioning field that
is constant across all 38 train cities fails loud — kills the #22 class structurally) and an
all-None guard per region (kills the #13 silent regime).

---

## 5. PI-CALLs (decisions carrying Umar's name; nothing pre-resolved)

1. **δ effect-size floor** for the recalibrated gate (§4.3; sweep table provided).
2. **Identity vs character vs both** enrichment shape (§4.4; recommendation: both, asymmetric).
3. **#13 scope:** city-name shortcut now + admin_region at the deferred regen, vs regen now.
4. **Allocation/timing:** the soft-end is 2026-06-11; T0 work is CPU-local and can proceed
   through the renewal gap; first post-renewal GPU job waits for T0 closure (incl. F8 resume) —
   confirm this ordering against the renewal timeline.

## 6. Out of scope (logged, not silently dropped)

Coherence layer (Phase-3, per delta-spec ledger); munich→manchester swap (Phase-3 reserve);
mamba/diffusion construction (Task-5 gate unchanged); the corpus regen bundle (#16/#17 + #13
proper + #22 + touch-as-cross — dedicated sub-project, unchanged); "all Europe" expansion
(feasibility verdict stands).

## 7. Verification discipline (protocol v3, applied)

Every threshold gets its paired structural check (§2); every new abstraction over an existing
module gets a Gate-6 external cross-reference; every exclusion is construction-identity with a
regime-distinguishing guard (§9 — the bref exclusion in §4.1 explicitly); write-once artifacts
are not sized by provisional parameters (§10.1 — the EU emergence floor records its derivation
regime); per-stratum thresholds are relative-to-base-rate where base rates differ (§10.2);
detection power verified in the correct unit (§10.3 — pairs vs strata vs features in the
recalibrated gate). Subagent-driven, implementer ≠ reviewer, stop-before-commit at every gate.
