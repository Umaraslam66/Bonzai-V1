# Readiness audit — failure-class enumeration + coverage matrix (Step 2 artifact, 2026-06-10)

**Denominator:** the accepted execution-surface map
(`reports/2026-06-10-readiness-audit-execution-surface-map.md`, gate-passed by PI 2026-06-10).
**Completeness criterion (handoff §3):** for every class, on every path the bake-off executes
(incl. unrun Tasks 9–12), show EITHER a non-vacuous check OR an explicitly logged gap, such that
the per-class question *"could a latent failure of this class exist on an executed path this audit
did not examine?"* answers **NO**.
**Non-vacuity bar (PI Sharpen 2):** a check counts ONLY if it would actually FAIL with the failure
present. Tiers: T1 = red-on-divergence demonstrated; T2 = non-vacuous by construction on the LIVE
path. A test that exists but provably never exercises the failing regime (SG-only fixtures,
slow-deselected, helper-not-live-path) is a GAP with a note owed. Default suite runs
`-m 'not slow'` — slow/deselected tests guard nothing day-to-day; selection status is recorded.

**Method:** 4 coverage-verification subagents (classes grouped; targeted local test runs: 26+30+43
default-suite tests re-run green; edge behaviors reproduced empirically — `pick_winner({})`
StopIteration, single-backbone IndexError, `ks_distance` empty→1.0) + 1 adversarial completeness
critic. Orchestrator independently spot-verified the four claims that change the taxonomy
(no-stop-condition generation, report-filename collision, bare `write_text` manifest, BH-only
significance). Implementer ≠ reviewer throughout.

Paths: P1 corpus read · P2 conditioning construction · P3 training load+exec · P4 eval read ·
P5 decode · P6 scoring/decision · P7 orchestration/shell+frozen-artifacts.

---

## 1. The taxonomy — 17 classes

The 14 structural classes survived the adversarial critique intact at the SURFACE level (every
mapped fact has a class home). The critique added **three cross-cutting MODE classes (F15–F17)**
the structural axes missed — each anchored in verified code, each a sixth-class-style surprise
candidate had it stayed unnamed.

| # | Class | One-line definition |
|---|---|---|
| F1 | Single-region residue | SG literals/defaults riding paths the bake-off executes |
| F2 | CRS/zone correctness | dir-labels, multi-authority CRS, projected-meters assumptions |
| F3 | Silent-vacuous reads & fail-open defaults | missing-data paths that return clean-looking nothing |
| F4 | Construction-identity contracts | decoder-emits-X / consumer-must-transform obligations |
| F5 | Conditioning EXPRESSIVITY (axis a) | content too coarse to carry city character (gate-(i) FAIL) |
| F6 | Conditioning DELIVERY (axis b) | values never reach the model (constant slot prefix) |
| F7 | Planned-but-absent surface | Tasks 9–12 machinery that does not exist; loud vs SILENT absence |
| F8 | Checkpoint/resume & orchestration integrity | resume split-brain, traps, markers, shell layer |
| F9 | Lock/artifact integrity | unguarded locks, freeze-only shas, unread markers |
| F10 | Dead-twin & stale-doc hazards | superseded modules/false docstrings that wiring can follow |
| F11 | Verdict calibration & aggregation (renamed from "teeth gaps") | fail-open edges, missing guards, AND over-firing/calibration |
| F12 | Backbone/scale readiness | 2-of-3 backbones absent, AR-only eval hook, compile-outcome opacity |
| F13 | Sample-regime transfer | SG-derived constants applied to EU/scale regimes |
| F14 | Distributed/scale execution regime | EU-scale per-rank rebuild, DDP collectives, sampler fallbacks |
| **F15** | **Measurement commensurability** (NEW) | compared quantities measured under different budgets/lengths/protocols |
| **F16** | **Generation coherence / version-skew / fix-sequencing** (NEW) | nothing pins artifact↔code↔checkpoint compatibility across changes |
| **F17** | **Multi-run interference / run-isolation** (NEW) | 12 bake-off runs sharing cwd/manifests/report names |

**F5 vs F6 (PI-mandated split, held):** the full 8-field × 5-lifecycle-stage table (derived /
stored / supposed-read / actually-read / reaches-model-input) shows machinery EXECUTES through
"actually-read" for all 8 fields and reaches the model for **zero** of them — the only model input
is the constant `[1508..1515]` slot block on both training and generation. Fixing delivery without
expressivity hands the model ≤4 informative dims (region None #13, morphology/era/country/climate
constants #22, density all-moderate); fixing expressivity without delivery enriches a field the
model never sees. The classes compose and do not overlap.

---

## 2. Summary coverage matrix (class × path; cell = worst instance status; details §3)

✅ = all named instances covered non-vacuously · ⚠ = mixed (some T1/T2, some gaps) · ✗ = gaps
dominate (no non-vacuous check for the class's core instances on that path) · — = N/A.

| Class | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|
| F1 | ⚠ | ⚠ | ✗ | ⚠ | — | ✗ | ✗ |
| F2 | ⚠ | — | ⚠ | ⚠ | — | ✗ | ✗ |
| F3 | ⚠ | ✗ | ⚠ | ✗ | ⚠ | ✗ | ✗ |
| F4 | — | — | — | ⚠ | ⚠ | ✗ | — |
| F5 | ✗ | ✗ | — | ✅* | — | ✅* | — |
| F6 | — | ✗ | ✗ | — | ✗ | ✗ | ⚠ |
| F7 | — | — | ✗ | — | — | ✗ | ✗ |
| F8 | — | — | ✗ | — | — | — | ✗ |
| F9 | ✗ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| F10 | ⚠ | — | — | — | ⚠ | ✗ | — |
| F11 | — | — | — | — | — | ⚠ | — |
| F12 | — | — | ⚠ | — | ⚠ | ✗ | ✗ |
| F13 | ✗ | — | ✗ | ✗ | — | ✗ | ✗ |
| F14 | — | — | ⚠ | — | — | — | ✗ |
| F15 | — | — | ✗ | ✗ | — | ✗ | ✗ |
| F16 | ✗ | ✗ | ✗ | — | — | — | ✗ |
| F17 | — | — | ✗ | — | — | ✗ | ✗ |

\* F5's ✅ is the gate-(i) FAIL — the check exists, is non-vacuous, and FIRED. Coverage ≠ health.

**No cell is blank.** Every ✗/⚠ decomposes below into named instances each carrying exactly one of
COVERED-NONVACUOUS (cited) / GAP-VACUITY-UNPROVEN (note owed) / GAP-NO-CHECK / N/A-reason.

---

## 3. Per-class detail (instances, evidence, §3 answer)

### F1 — Single-region residue. §3: **NO — gaps logged.**
COVERED-NONVACUOUS: `tile_dirname` regression locks at 3 sites (T1, `tests/eval/test_tile_dirname_crs_eval.py`
geometry+pipeline; `tests/data/training/test_build_shards_multiregion.py::test_build_shards_in_memory_uses_region_crs_label`;
all default-suite); manifest routing two-sided test (T1, EU-red-was-`KeyError: 'munich'` →
green / SG-stays-green, `tests/data/training/test_holdout_repoint_multiregion.py`);
`_EU_HELD_OUT_CITIES` drift guard (T2 external-source, `tests/eval/holdout/test_paths.py`);
SG-constants-cannot-reach-model (T2 via the slot-prefix lock — note this same lock is F6's defect).
GAPS: **no test asserts sbatch content** (the singapore pre-build literals in both bake-off
sbatches are unguarded — confirmed no region-literal assertion anywhere); `ScaffoldConfig.region`
default unguardable; `_write_report` SG-marker stamp on any region; `--emergence-floor 1.96`
SG figure (also F13/F15); **NEW: the 4th `tile_dirname` call site —
`conditioning_discrimination.extract_features_by_city_stratum_metric` — has NO regression test
and a silent warn-and-skip failure mode, on the one EU path that already executed**; NEW:
`test_tile_dirname_matches_sub_d_convention` actively PINS the EPSG3414 default (reverse-lock —
removing the default requires editing this test).

### F2 — CRS/zone correctness. §3: **NO — gaps logged.**
COVERED: dir-label derivation per region config (T2, `test_epsg_label_for_region_reads_region_config`).
GAPS: the **three-way CRS consistency check does not exist** (config `projected_crs` is the sole
live authority; sub-D `region_crs` and manifest `crs` have zero consumers); config↔data agreement
for all 42 cities rests on two one-time, non-repeatable run artifacts (Task-8 build = fail-loud →
genuine evidence at that commit/filesystem; gate-(i) = warn-and-skip → proves only "some tiles
resolved"); **no check anywhere asserts the CRS unit is meters** — a degree-based `projected_crs`
would flow through silently into area/length (holds today by enumeration of the 42 configs only).

### F3 — Silent-vacuous reads & fail-open defaults. §3: **NO — gaps logged.**
COVERED: `train_cities` held-out leak (T1 structural guard `test_structural_guard_fires_if_heldout_leaks`
+ T1 lineage-audit teeth + planted-leak datamodule halt, all default-suite — but the library
function itself is fail-open `.get("held_out_cities", [])`; protection is call-site convention);
decodability-as-rate denominator (T1 `test_decodability_uses_attempted_block_count`); gate-(i)
thin-n/UNSUPPORTED (T1 teeth 4–5).
GAPS: vacuous-0.0 MECHANISM in `holdout_polygons_per_active_cell` (only the slow/deselected
real-data test guards it); **gate-(i) missing-tile skips have NO surfaced count in the verdict
report** (a city can silently shrink while surviving strata stay ≥ min_n — partial shrinkage is
indistinguishable from thin data; the executed Leonardo run inherited this); `bakeoff_run.sbatch`
passes NO emergence floor → every Task-11 run silently omits the §2 verdict (fail-open is
test-PINNED as intended); all-None labels from a missing `conditioning` key flow to gate strata
with no check; drop-rate never asserted (`dropped` dict discarded at `setup`; zero-example
training would proceed silently — `test_e2e_scaffold` asserts `decoded_cells >= 0`, an explicitly
vacuous bar, and is slow-deselected); no decode-yield floor on any wired path.

### F4 — Construction-identity contracts. §3: **NO — gaps logged.**
COVERED: gate-(i) ring promotion (T1, `test_tile_features_promotes_building_rings_to_area`,
red-before demonstrated on Leonardo munich building_area=0); Multi* per-part pairing (T1, sub-G
H1 guard tests, default-suite); decoder↔encoder constants (T2 transitive via round-trip L∞≤4.8m
gate + vocab YAML pins; residual: no direct decoder-constants-vs-YAML test).
GAPS: **slice_eval's promotion call is vacuity-unproven** — every fixture passes pre-made
Polygons; deleting the live `promote_building_rings` call at `slice_metrics.py:133` passes the
entire default suite (the `is`-identity test proves shared definition, not invocation — ~10-line
test owed); `holdout_polygons_per_active_cell` promotion guard is slow-deselected only;
**Task-12 promote-before-`feature_samples` obligation is recorded nowhere executable AND the
2026-06-02 plan's own snippet demonstrates `feature_samples` on unpromoted geoms** (stale-snippet
hazard); **C2 bref-collapse: gate-(i) road_length includes V=2 zero-length collapses and ≤125 m
placeholder-vertex error — unquantified.** Bound established from existing artifacts: building
blocks carry no bref tokens (pinned by test) → **building_area's FAIL is bref-independent; the
overall gate-(i) FAIL stands; road_length_m's 168/180 significant pairs are the suspect subset**
(SG reference: ~3.2% collapse rate; no EU per-city bref-rate artifact exists — quantification
owed at the re-derive); C4 Point semantics divergent and unpinned (dropped-no-counter in gate-(i)
vs always-valid-inflating-OGC in slice_eval); C7 density-skip denominator divergence (gate/eval-set
skip no-density cells; the emergence-floor source counts ALL non-empty cells — incommensurate
denominators, undocumented — also feeds F15); 300 m plausibility bound exists ONLY in sub-G —
implausible-but-OGC-valid generated geometry passes scoring unchecked.

### F5 — Conditioning expressivity. §3: **NO — one covered cell (which FIRED); rest gaps.**
COVERED: gate-(i) (T1 locally — fail-on-genuine-difference + MC-guard-non-vacuous demonstrated;
the one non-vacuous expressivity check on the entire surface; FAIL 304/321). Precisely scoped:
4-dim stratum × {building_area, road_length} × 4 held-out cities × real round-tripped geometry.
It does NOT cover: train-city expressivity, density dim (all-moderate cut), non-geometry character
(Points dropped), per-instance arrangement (KS is marginal — `feedback_shuffle_gap` applies), and
**it cannot LOCALIZE which coarseness layer binds** (stratum dims vs grid→scalar collapse vs
4-bucket quantization — it conditions on already-collapsed, already-quantized labels).
GAPS (all GAP-NO-CHECK): #13 all-None admin_region has no firing check (documentation substitutes:
known_issues #13 + orchestrator HARD GATE); #22 morphology constant — the existing test asserts
the value EQUALS "Asian-megacity" (goes red only if someone FIXES it); grid→scalar collapse
information-preservation unchecked (the trigger-2 identity lock guarantees eval and model share
the SAME collapse — consistency, not adequacy); EU bucket-saturation never audited (buckets locked
from SG frequency analysis); `effective_conditioning` value-plausibility per region unvalidated
(the three SG-wrong constants ride every EU tile; `TileLabels` structurally excludes them — T2 —
but wholesale dict readers inherit them).

### F6 — Conditioning delivery. §3: **NO — and the default suite CERTIFIES the failure.**
No check anywhere asserts conditioning VALUES reach the model. Two default-suite tests **LOCK the
value-agnostic constant** (`test_conditioning_prefix_is_the_field_slot_id_block`; flatten-prefix
assertion) — they will go RED on the fix; until then "suite green" is anti-evidence for delivery.
The value-bearing machinery is uniformly tested-but-unwired (4 instances: `conditioning_prefix_ids`
test-only; `build_value_bearing_prefix` data-path unreached; `TrainingShard.tile_conditioning` has
**ZERO readers** — `shard_schema.py`'s "slice reads this" comment is false; 512 embedding rows
capacity-tested, utilization-untested). Generated-side `strata.append(0)` hardcode: only the
slow-deselected smoke reaches `_generate_and_score`. **Id-space hazard: slot ids 1508..1515
literally equal field-0 value-buckets 0..7** (`CONDITIONING_VALUE_BASE == CONDITIONING_ID_BASE`)
— no mutual-exclusivity guard; the concrete consequence is F16's checkpoint flip. One covered
cell: MASK_ID placement (T1, computed from live constants). **The delta-spec §4 PRIOR still
asserts the false "handed D's macro plans" claim uncorrected** — drift recorded everywhere
(known_issues, map) except where decisions read.
NEW instances: "region" name collision (`tile_conditioning["region"]`=admin_region None-for-EU vs
`TrainingShard.region`=city name — a delivery fix wiring by key can grab the wrong one); three-way
seed-semantics divergence; micro_ar prefix-mask tests run at n_cond=8, never the live 512 shape.

### F7 — Planned-but-absent surface. §3: **NO — the silent absences are the exposure.**
LOUD absences (fail at first contact — lower risk): `--config` (argparse exit 2; xfail documents);
Task-10/12 runners (no file); gate-(ii) reader (no file); `configs/experiments/` absent;
mamba raises at construction.
**SILENT absences (execute wrong — the dangerous cells):** `bakeoff_diagnostic.sbatch` **runs
clean today** — SG pre-build, recognized flags, 8 h of GPU on the wrong region, plausible report
(flag-drift test covers flags only, never values/region); obligation (c) wired-with-defaults would
silently assert EU decisions against SG numbers; an improvised Task-12 passes every existing
library guard with a consistent city subset (4-city completeness assertion is prose-only);
`--emergence-floor 1.96` survives any region-only repoint.

### F8 — Checkpoint/resume & orchestration. §3: **NO.**
The across-job resume path has never executed anywhere and cannot (unwired): `test_resume.py` is
T1 for path-selection logic but system-vacuous; `ddp_resume_check` proves within-job
`fit(ckpt_path=…)` resume at 1e-4 (2026-06-02, historical) — NOT USR1 relaunch, NOT across-job,
NOT the production `run_short` path (which passes no ckpt_path). **NEW: the USR1 trap never
forwards the signal to the srun ranks (no checkpoint-on-signal) — the header mechanism is doubly
false; and `trap '… sbatch "$0"; exit 0'` exits 0 even if the resubmit fails** (4th member of the
false-completion family). JOB_DONE markers verify nothing (`set -e` + rc only; the diagnostic's
report-check masks absence with `|| echo`). 30-min checkpoint cadence config is T1-locked (config
drift only). NEW: epoch-mode trainer/scheduler mismatch (`max_epochs` ⇒ trainer max_steps=-1
while cosine anneals against the 2000 default — present on the already-run ddp_resume_check path,
harmless at 16 batches, unguarded as a class).

### F9 — Lock/artifact integrity. §3: **NO.**
COVERED: sealed sub-F vocab (the strongest cell — `tests/data/sub_f/test_vocab.py` pins deployed
YAML content exhaustively: 686 slots, per-family counts, bref 1500..1507, LOCKED status; any edit
shifting `CONDITIONING_ID_BASE`/embedding layout fails the default suite on that checkout; caveat:
test-time only, nothing asserts vocab-sha at the training entry point); bref-id constellation
(T2 transitive).
GAPS: `TRAIN_TOKENS` lock-without-guard (no test references it; ladder boundary tests only
range-pin it accidentally [600M, 2B) — would mask a marker drift while appearing to test it);
`manifest_sha256` freeze-only — **zero reader-side verification** on every training setup and
eval read; no `_EVAL_SET_LOCKED`-presence check on any read; EU marker no-KS-fields/no-readers;
`assert_resolution_sufficient` SG-marker default (dormant; becomes SILENT wrong at Task-12);
1.36/1.358 twins un-cross-guarded (live at Task-12); NEW: `UNSCORED_V1_DIMENSIONS` is consumed by
no src code — the test asserting membership in it guards a frozenset nothing reads; NEW (F9/F16):
report records `cfg.model_dump()` but `eval_cells`/`eval_max_new`/`emergence_floor` are
run_short kwargs — **CLI-only parameters are absent from the reproducibility record** ("config +
commit + snapshot fully determine the run" violated).

### F10 — Dead-twin & stale-doc. §3: **qualified NO** (all twins runtime-confirmed unloaded;
class fires at WIRING time).
**The live plan document (2026-06-09 delta-reconciliation) contains an import snippet for the
DEAD `conditioning_gate` twin** — wiring instruction pointing the wrong way; the twin has no
tombstone, a green default-suite test (misleading liveness signal), and a docstring saying
"injected by the Task-9 diagnostic." Phase-0 `tokenizer.decode` exported+tested, no tombstone.
`deviation_log` T1-unit-tested, zero consumers — its own "enforced by WHEN it is called"
constraint enforced by nothing. Stale docs: `decode_feature` Polygon claim (contradicted in-file;
the trap aims exactly at the unwritten Task-12 consumer); micro_ar n_cond=8 (T1-guarded against
the dangerous direction — code can't regress to match the doc); sub_c 3-type comments (low).

### F11 — Verdict calibration & aggregation. §3: **NO for the wired surface** (gate-(i) verdict
logic is well-toothed: BH/thin-n/UNSUPPORTED all T1); the unwired decision layer's gaps fire at
improvisation time.
COVERED: REC-2 `decision_ks` dict→list regression-lock (T1, verified to bite three ways);
`ks_distance` empty→1.0 mechanism (T1); city-set-equality + empty-mapping guards (T1).
GAPS: `pick_winner`⟷`structural_check_ok` pairing untested (each T1 separately); `pick_winner({})`
bare StopIteration; **`pick_winner` single-entry silent auto-win — live NOW with one buildable
backbone: an improvised one-backbone "bake-off" crowns transformer-ar with zero checks**;
single-backbone IndexError in `binding_city_verdict` (loud, wrong layer); 4-city completeness
prose-only; gate-(i) tile-coverage accounting absent (the one F11 gap on an executed path);
no n asserted before KS is decision-bearing.
**Calibration finding (critic, code-verified):** gate-(i) significance is `p_bh < alpha` ONLY —
the per-pair noise floor is recorded, never decision-bearing; at the artifact's n (up to 169k)
the minimum significant KS was 0.0285 and the gate **cannot PASS on any real city set** at these
sample sizes (gate-must-distinguish-regimes). The FAIL's substance stands on the large practical
effects (max ks−floor 0.554; glasgow/krakow 2× medians, per closeout) — but the verdict mechanism
has no practical-effect-size floor, and the T5 re-derive must address calibration explicitly
rather than inherit `p_bh < alpha` as the reopened gate's criterion.

### F12 — Backbone/scale readiness. §3: **YES for what executes today** (only transformer-ar
constructs; everything else fails loud at construction); conditional exposure when Task-5 lands.
GAPS-with-notes: no pre-submit buildability gate (BackboneNotYetBuilt fires on-node after queue
wait — 1-line CPU preamble dry-run owed); `_generate_and_score` AR-only typing (unreachable until
diffusion exists; no check pins the crash); mamba-lock (Task 10.5) is plan-prose, no code; NEW:
**`maybe_compile` swallow + report records compile INTENT not OUTCOME** — a per-backbone silent
compile failure skews cost/throughput comparability across the 12 runs while every record claims
compile=true.

### F13 — Sample-regime transfer. §3: **NO — weakest class: every instance is a gap; three of
four exist only as prose.**
`DEFAULT_MAX_CELL_TOKENS=5760` (SG P99.9): no EU token-length stats exist anywhere (Task-8 report
carries tile counts only); drop-rate has no threshold and no action contract (the
`feedback_diagnostic_threshold_design` anti-pattern verbatim). Emergence 1.96/7.85: SG-only;
machinery to compute an EU floor exists with zero callers; nothing stops the SG figure riding an
EU repoint. r@90M→larger scales: the sbatch's own RECORDED FLAG says don't — no code could notice
if it did; `feasible_ladder` accepts any r with no provenance/scale field. SG KS numbers
(`ks_resolved_gap_binding` 0.076 etc.): the eval-set-gen carry-forward trigger ("fail-loud if
needed-gap < resolved-gap") is **implemented nowhere**; when Task-12 needs an EU resolved-gap the
only numbers on disk are SG's, unflagged.

### F14 — Distributed/scale execution. §3: **NO — EU-scale and 4-GPU regimes are the map's §4
logged gaps; nothing has executed at 22k-tile scale.**
COVERED: seed-before-model-init (T1, genuinely red-on-divergence); collective-checkpoint shape
(T2 + 2026-06-02 historical).
GAPS: **per-rank EU union rebuild has NO number on record** (≈22k parquet reads + ~624M tokens as
Python ints per rank per setup, repeated on every relaunch; back-of-envelope ~20–25 GB/rank +
unmeasured setup wall-time — the estimate itself is absent from all artifacts, which is the gap);
WorldSizeGuard never observed FIRING (pass-regime evidence only); `--devices 1` under multi-task
srun ⇒ N duplicate trainings, no guard attached, last-writer-wins (the sampler fallback's silent
regime); DistributedSampler padding/duplication semantics untested.

### F15 — Measurement commensurability (NEW). §3: **NO — no check relates any of the lengths.**
The class: a wired, non-vacuous verdict whose two sides are measured on different rulers.
**Core instance (verified):** generation has NO stop condition (`generate.py:49`; padding targets
masked in training so no end-signal is ever learned) → generated cells are degenerate at
`--eval-max-new` (512 default / 2048 diagnostic) while the emergence floor's reference density
(7.85 → 1.96) was computed over FULL-LENGTH real cells (≤5760 tokens, ~250 blocks). Same model,
different `--eval-max-new` ⇒ different §2 verdict; the parameter isn't even in the report (F9).
**Compounding:** the diagnostic trains at `--max-len 2048` — silently DROPPING the densest
training cells (the regime the floor probes) while the floor constant stays full-length-derived.
Three lengths (train max_len / eval_max_new / floor-derivation length) must be commensurate;
nothing relates them. **Forward:** any count/density/cell-level Task-12 metric inherits the bias
(per-feature area/length metrics are safe). Also: the C7 denominator divergence (floor counts all
non-empty cells; gate/eval-set count density-bucketed cells) is the same mode at the denominator
layer. GAP-NO-CHECK on P3/P4/P6/P7.

### F16 — Generation coherence / version-skew / fix-sequencing (NEW). §3: **NO.**
**Checkpoint↔prefix-scheme silent flip (sharpest instance, verified):** the embedding is sized for
both schemes today (n_cond=512), so wiring the F6 fix changes input SEMANTICS with ZERO shape
change — a slot-prefix-trained checkpoint loads cleanly under value-bearing code and is silently
wrong (slot ids 1508..1515 reinterpret as field-0 value buckets). **Neither the checkpoint nor the
report records which prefix scheme the weights were trained under** (`save_hyperparameters(cfg.model_dump())`
has no scheme field). Detected only as unexplained metrics, post-GPU-spend.
**Data-generation skew:** `provenance_sha256` is written into training manifests and **verified by
no reader at load**; the corpus is frozen at one defect level with regen deferred (#16/#17) — the
moment any city regenerates, `build_shards_in_memory` re-reads current disk against a stale
manifest and a mixed-generation union trains silently (the uniform-defect-level trap, hit twice
before, now structurally unguarded on the training path).
**Fix-sequencing:** F5-fix changes value-bucket layouts; F6-fix invalidates checkpoints; both
invalidate Task-8 manifests if the shard schema grows — per-class sign-off would miss that fixing
them in the wrong order is itself a failure path. The enrichment plan must carry an explicit
sequencing/compatibility analysis. GAP-NO-CHECK throughout.

### F17 — Multi-run interference (NEW). §3: **NO.**
The bake-off = 12 runs (3 backbones × ≤4 scales) sharing one cwd. **Torn-manifest silent shrink
(verified):** every `bakeoff_run.sbatch` preamble rewrites the SAME `training_manifest.yaml` via
bare `Path.write_text` (`build_shards.py:284` — no tmp+rename; the project HAS a crash-safe
writer in sub-C, unused here); a concurrent job reading mid-write can get truncated-but-parsable
YAML → passes the holdout audit (fewer tiles can't leak) → trains on a silently smaller set.
**Destructive report collision (verified):** report filename is
`{release}-{region}-loop-closed{suffix}.md` — no backbone/seed/job-id; the three backbones at one
scale OVERWRITE each other's primary evidence. Plus lightning_logs/CSVLogger version-dir races.
GAP-NO-CHECK on P3/P6/P7. Single-run tracing is structurally blind here — no class prompted the
question before the critic.

---

## 4. Additions to the map's §4 logged-gap register (methodology blind spots, adopted)

1. **Allocation lifecycle:** `AIFAC_P02_222` soft-ends 2026-06-11 (tomorrow); the
   resubmit-across-renewal-gap path is exactly the one that is (a) unwired (F8) and (b) about to
   be needed; calendar feasibility of ~375 node-h sits outside every class. Carried to the spec.
2. **Inter-job interference** (now F17) — invisible to single-run tracing.
3. **Generated-token regime fixture-blindness:** all eval-path runtime evidence used real
   round-tripped tokens or a 4-cell untrained smoke; metric-layer behavior on trained-model output
   at 2048 tokens is unobserved (the project's own synthetic-fixture-blind-regime lesson, now at
   the eval layer).
4. **Library-semantics branches:** ModelCheckpoint `monitor="val_loss"` under max_time stops that
   never reach a val epoch; DistributedSampler padding when dataset % world_size ≠ 0; Slurm
   `B:USR1@120` delivery through `srun … & wait`. Historical-run evidence only.

## 5. Consolidated logged-gap register (the spec's input — every gap above, deduplicated, 38 entries)

Highest-leverage clusters (full list = the per-class GAP cells above):
- **G-A. Conditioning (F5+F6+F16):** delivery wiring + expressivity enrichment + scheme tagging +
  sequencing analysis + gate-(i) calibration (effect-size floor) + bref quantification for
  road_length + collapse-localization diagnostic.
- **G-B. EU-training path (F1+F7+F13):** region plumbing (CLI/config), sbatch repoint, EU
  emergence floor, EU token-length stats vs 5760, drop-rate action contract, three-way CRS check.
- **G-C. Run integrity (F8+F15+F17+F9):** resume wiring + signal forwarding, atomic manifest
  writes, report filenames keyed by (backbone, scale, seed), CLI-params into the report,
  commensurability constants related and recorded, reader-side sha/marker verification.
- **G-D. Decision layer (F11+F4+F12):** Task-12 runner with its five paired obligations (basis
  assertion, 4-city guard, promotion, structural-check pairing, n-floors), Point/bref semantics
  pinned, pre-submit buildability gate, compile-outcome recording.
- **G-E. Cheap test debt (one sitting):** slice_eval promotion live-call test; gate-(i)
  extraction-site CRS regression test + tiles-read/skipped counters; conditioning-gate tombstone;
  TRAIN_TOKENS guard; plan-snippet corrections (conditioning_gate import, feature_samples
  unpromoted, delta-spec §4 PRIOR).

## 6. Enumeration completeness statement

With F15–F17 folded in, the class extensions adopted, and §4 of the map extended (items 1–4
above), the adversarial critic **found no remaining mapped surface or reachable path without a
class home**, and every (class × path) cell in §3 carries a named status — no blanks. Per-class §3
answers are NO-with-logged-gaps for F1–F11/F13–F17 and YES-for-today's-executed-surface for F12.
The audit's claim is therefore: **a latent failure on the bake-off's execution surface must now
belong to a named class and land in a named gap** — the criterion's "prove coverage of the
execution surface" is met by the matrix plus this register, not by list length. Residual honest
limits: EU-data/GPU/shell-layer branches rest on artifact-level evidence (map §4); the
generated-token-at-scale regime is unobserved by anything (item 3 above) and is carried as a
first-class gap, not a footnote.
