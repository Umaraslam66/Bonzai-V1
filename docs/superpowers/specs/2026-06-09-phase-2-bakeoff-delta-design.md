# Phase-2 Bake-off — Delta-Reconciliation Design (2026-06-09)

**Status:** design locked via brainstorm 2026-06-09 (T1–T7, topic-by-topic gate discipline, planning-protocol-v3). Awaiting PI review → implementation plan.
**Baseline:** `docs/superpowers/specs/2026-06-02-phase-2-bakeoff-design.md` (the 9-topic bake-off spec). This document is a **DELTA** against that baseline — it does **not** restate it. Where a baseline topic is unchanged, it is named as surviving; where the 112-commit EU/eval-set reality changed it, the change is recorded here.
**Why this exists:** the 2026-06-02 spec was written against a Singapore-only corpus with no eval set. Since then `main` gained 112 commits (multiregion CRS → EU extract → corpus-completion → eval-set-gen, merged `0833ac7`) while `phase-2-bakeoff` (+16) never reconciled. This delta reconciles the locked bake-off spec onto the EU/eval-set reality.
**Scope:** delta-reconciliation, NOT a redesign. The 9 settled topics and ~11 built tasks stand except where named below. Status markers used throughout: **[LOCKED]** (decided), **[PI-CALL]** (framed for Umar, not pre-resolved), **[RULE]** (pre-committed, decide-by-measurement), **[RESIDUAL]** (pre-committed check, not a blocker).

---

## 0. The reconciliation in one paragraph

The EU eval stack (coherence metric, leak guard, §7 coherence power gate) was built in a separate sub-project and assumed a model that emits a macro plan. The bake-off backbones emit **sub-F micro tokens only**, so coherence — a *macro-plan* metric — cannot score them. Phase-2 therefore decides on **KS-realism alone**; coherence and its obligations move to Phase-3. The EU corpus (623.9M train tokens) is far below the 4-scale ladder's Chinchilla demand, so the ladder becomes a **pre-committed function of a measured `r`**, and the decision basis (scaling-curve vs fixed-scale+§13) is itself keyed to how many scales survive. Eval against 4 heterogeneous held-out cities is **net-new design** (the branch eval is single-city); the recommended bar is **worst-case** generalization (Umar's call), coherent **only if** a conditioning-discrimination diagnostic confirms per-city scoring is meaningful, with a pre-committed munich-#21 artifact check. The branch is brought current by **rebase**, opened by a red-before/green-after sequence that makes the EU holdout-API drift the diagnostic.

---

## 1. T1/T2 — Decision axis & the coherence layer-mismatch (Reading A) **[LOCKED]**

**Coherence does not participate in the Phase-2 decision.** `coherence.py` scores the MACRO plan (`list[MacroCoreRow]`, `road_skeleton_class`/`zoning_class` over the 6×6 interior). The bake-off backbones emit **sub-F micro tokens only** — `micro_ar.py:9-10`: *"the model never predicts a conditioning id (conditioning is given, not generated)"*; head → `n_subf_vocab` (`:73`); macro grep across `src/cfm/models/` + `src/cfm/data/training/` is empty. You cannot score coherence on structure the models never generate. **Reading A is forced from source, not chosen.**

- **Phase-2 decision axis = KS-realism (per-feature geometry-realism), sole.** (Baseline §3 unchanged in identity; coherence simply does not join it.)
- **Cross-family validity holds for both metrics** by the baseline §3 criterion (measured on generated *output*, one ruler): coherence's shuffle-gap is *not* NLL-incomparable — it scores the artifact and self-normalizes via the interior permutation. The blocker is a **layer mismatch (macro metric vs micro generation)**, not a ruler mismatch. This is why coherence is *deferrable* (to Phase-3), not *broken*.
- **Coherence + the §7 power gate move to Phase-3** (the macro-planner winner-preview the baseline §7 already named as a deferral).

**Re-mapped obligation ledger** (from the eval-set-gen "fires at first model" set):

| Obligation | Belongs to | Why |
|---|---|---|
| (a) holdout re-point + schema-2.0 flip | **Phase-2 (required)** | the KS reference (`geometry.py`, `feature_resolution.py`), the leak-audit, AND the emergence floor all read the holdout manifest |
| (b) define `model_vs_real_effect` | **Phase-3** | it is the coherence effect; coherence is Phase-3 |
| (c) munich→manchester swap | **Phase-3** | coherence power reserve, owned by `assert_coherence_power_sufficient` (`resolution.py:106`) |
| (c) EU-train-split resolved-gap recompute | **Phase-2 (required)** | feeds `assert_resolution_sufficient`, the SOLE Phase-2 power gate (the EU `_EVAL_SET_LOCKED` carries no KS number) |

---

## 2. T3 — Data/ladder policy (the 623.9M ceiling) **[RULE]**

**Authoritative:** `train_tokens = 623,900,790` (~0.624B), `held_out_tokens = 46,130,102` (`_EVAL_SET_LOCKED`). At Chinchilla 20:1 only **30M** is frontier-feasible (100M 3.2× / 300M 9.6× / 1B 32× short). The EU extraction lifted the corpus 43× (14.4M→624M) but is still below the branch's own ladder demand (27k–896k tiles, `6b3a2cb`). The baseline §4 compute-optimal `D∝N` premise therefore cannot be satisfied as specced.

**Rule 1 — feasible ladder (on-frontier ONLY).** Scale `N` is feasible iff `r·N ≤ 624M·E`, where `r` = geometry-verified tokens/param (`max(Chinchilla, emergence-floor)`, MEASURED by the Task-1 diagnostic) and `E` = epoch factor (≤~4). Failing scales are **dropped, never run data-limited**. Thresholds: `r≈20→{30M}`, `r≈10→{30M}`, `r≈5→{30M,100M}`; `E=4` adds one rung. **1B drops for all admissible (r,E)** (needs `r≤2.5`). `r` is measured **with a CI**; if its CI straddles a rung boundary, **round conservatively** (fewer rungs / higher effective `r`) — never add a rung the data can't clearly support. **∅-case** (even 30M can't emerge on-frontier within 624M) → escalate to more-data (C).

**Rule 2 — decision basis (step function of #feasible points).** The KS value at *every* scale is the **T5 cross-city aggregate** (§4) — so Rule 2 composes with T5: **T5 sets what each point measures** (worst-case-across-cities KS, per the recommendation, or mean if Umar so decides); **Rule 2 sets whether the points form a trustworthy curve.**
- **< 3 points (incl. 2):** the curve is **reported, never decision-bearing** (a 2-point power law has zero fit DoF — unfalsifiable). **One rule for 1 and 2 points:** winner-vs-runner-up KS at the **top feasible scale** > the held-out-powered resolution → KS winner; else **§13 → `transformer-AR`** (baseline §11 cell-collapse tie-break; NOT PRD §13's "hierarchical transformer").
- **= 3 points:** falsifiable curve (1 DoF) → `curve.py` CI-non-overlap at the production target (`curve.py:102`), **but** the extrapolation *distance* is reported and the lever-arm sanity-checked before the verdict is decision-bearing (no bare CI-non-overlap ~6× out).
- **§13 is reachable two ways:** curves fit but CIs overlap, OR too few feasible points to fit a falsifiable curve.

**Epoch-repeat conditions:** `E ≤ ~4` (Muennighoff unique-equivalent regime), **logged as a documented deviation**, **`E` held IDENTICAL across all three backbones** (AR vs diffusion respond differently to repetition — an uneven `E` confounds the cross-family comparison), locked pre-scored-runs.

**Task-1 diagnostic charter (measures; decides nothing):** `r` (with CI) + emergence floor + per-scale eval-cost + **KS-power adequacy** (does the EU held-out set, munich included, resolve winner-vs-runner-up per-feature KS at the feasible scale — discharges the parked obligation; #21 was coherence-specific, does not transfer). Rules decide deterministically off the measurement.

---

## 3. T4 — Branch strategy & the opening sequence **[LOCKED]**

**Rebase `phase-2-bakeoff` onto `main`** (not restart). Conflict surface = **one file** (`scripts/train_scaffold.py`, main +8/branch +89 — and itself an obligation-(a) schema-flip site); the other 34 changed files are net-new or touch files `main` left alone. The branch is **local-only** (not pushed) → rebase rewrites only unshared history; `main` untouched. Restart's only appeal (escaping messy conflicts) doesn't apply; rebase preserves the `6b3a2cb` data-wall provenance that justifies this whole reframe.

**The trap git won't flag (semantic drift):** the branch's eval calls `cfm.eval.holdout.paths` / `assert_resolution_sufficient` as they were at the merge-base; `main` rewrote them for EU. A clean rebase yields a **green-but-lying** suite that breaks at runtime when the holdout API is exercised.

**Opening sequence (the plan's first tasks):**
1. **Rebase** onto main.
2. **Run the full suite immediately, EXPECT RED** on holdout-API drift (Singapore-path imports, `assert_resolution_sufficient` reading the EU marker with no KS number). This RED is the diagnostic — it shows the exact surface (a) must repoint. **Alarm tell:** a GREEN here is *not* relief — either the drift isn't where we think, or the tests don't exercise the live holdout path (vacuous green). Stop and investigate.
3. **Apply obligation (a) atomically** — repoint the branch's eval consumers **and** `train_scaffold.py` to `multiregion_holdout_manifest_path` **and** flip schema 1.0→2.0, together (flip-ahead-of-repoint fail-closes the still-SG manifest).
4. **Re-run the suite, EXPECT GREEN.**

> **[CORRECTION — applied during execution 2026-06-09, Umar-ruled; this clause's literal reading was wrong twice].** Two errors in the sequence above surfaced at execution and are now resolved (the implementer≠reviewer halt-gates caught both before any commit):
> 1. **Step 2 "EXPECT RED" was vacuous-green, not red.** The existing local suite is structurally blind to the EU holdout path (its non-slow tests exercise the consumers only via `region="singapore"` against the still-valid SG manifest; the EU path is slow-deselected / Leonardo-skipped). The red-before is restored by a NEW local test on (a)'s actual surface (the EU manifest is local). Also: Step 2's `assert_resolution_sufficient` cause belongs to obligation **(c)** (it reads `eval_set_locked_marker` = the **SG** marker, not the manifest; needs Leonardo tile data → Task 9), **NOT** (a). The red-before fires on (a)'s manifest-consumer surface ONLY.
> 2. **"repoint the consumers" is REGION-CONDITIONAL, not a blanket swap.** The holdout-manifest readers (`build_shards._holdout_ids`, `geometry.holdout_polygons_per_active_cell`, the region-parameterized scripts) are **dual-region** — reached with `region="singapore"` (local fixtures) AND EU cities (runtime). A blanket repoint fail-closes Singapore (`KeyError: 'singapore'`), violating §5/§6 ("NEVER touches the Singapore set"; "local = Singapore + the frozen holdout manifest"). **Resolution: the `region` arg selects manifest AND schema** — `singapore` → SG manifest/schema-1.0; the 4 EU cities → multiregion manifest/schema-2.0 — via a `holdout_manifest_for_region(release, region)` helper. This reconciles §3 (EU consumers repoint) with §5/§6 (SG survives, as the now-local-only test fixture; EU tile data is Leonardo-only). The schema "1.0→2.0 flip" is likewise region-conditional. **The blanket repoint was the error; do not re-introduce it.**

---

## 4. T5 — Multi-region eval semantics (NET-NEW design)

**Source verdict:** net-new, not a repoint. `realism.py::ks_distance(generated, reference)` takes two **flat** lists (no city dimension); `geometry.py:58` is single-`region`; zero pooling/aggregation/city-loop anywhere. The held-out set is **4 heterogeneous cities**; aggregation must be designed.

**[LOCKED] Per-city KS; pooling rejected** — a pooled reference is a chimeric distribution matching no real city, weighted by each city's cell count (munich's dense-core cells would tilt it, the same subset-hiding that pooling causes). Density is **not** an aggregation axis (all-moderate T1 cut); the live axes are **per-city = per-morphology**.

**[PI-CALL, GATED] The aggregation bar: worst-case vs mean.** *Framed for Umar — not pre-resolved, and **not a coherent rule until the conditioning-discrimination gate below passes.***
> Generalization is a **worst-case property** (the held-out set exists to test it; "a miss must read 'didn't generalize'"). An equal-weight **mean** lets three good cities mask one failure — the aggregate-hides-subsets failure, one level up from pooling. **Recommendation: worst-case (max) KS** — winner-vs-runner-up compared on the *binding* (worst) city; a backbone wins only where it is weakest; **report all 4 per-city KS and which city is binding.** Documented alternative: mean (simpler; not a generalization bar). This changes which architecture can win, so it carries Umar's name.

**[GATE — co-located with the bar; MUST pass before the bar is a decision rule] Conditioning-discrimination gate.** Worst-case (or mean) is a coherent per-city bar **IFF the Task-1 diagnostic confirms per-city KS tracks macro-plan differences** — same-macro-stratum tiles across cities sharing their building-area/road-length distributions — so that a per-city miss reads *"failed to render the handed structure,"* not *"wasn't told the city."* **If the diagnostic shows materially different real feature distributions for same-stratum tiles across cities — equivalently, that the conditioning span doesn't carry enough to distinguish the EU cities — the per-city miss is ambiguous and T5 REOPENS before any scored run.** The diagnostic **GATES** this bar; the bar is **not** locked ahead of it. (One gate, two former framings: this subsumes both the "macro-tracking" check and the "multi-country expressivity" residual — one concern, discharged together.)

**[LOCKED] munich included, not excluded — defect-class reasoning.** Structural exclusion does **not** transfer from coherence: coherence has a shuffle-NULL that munich's density *saturates* (breaks → exclude); KS is a direct two-sample CDF distance with **no null to saturate** (doesn't break → munich's per-city KS is legitimate generalization signal). Same munich fact, *different correct consequence per metric* (the rigorous consistency, not the mechanical one).

**[RULE] Pre-committed #21-artifact check** (set before any numbers; judge munich against *its own* floor, not the moderate-city band):
1. **Binding-city power gate:** the binding city's winner-vs-runner-up gap must exceed *that city's own* KS resolution floor `C/√(n_city_features)` (`feature_resolution.single_region_floor_gap`). munich has the fewest usable tiles (156) → highest floor. If munich is binding but its gap < its own floor, it is **under-powered and cannot decide** → the decision falls to the next-worst resolved city; munich is reported, not gated.
2. **Conditioning-match flag** (reported, not gated v1): report whether munich is conditioned `density=moderate` while geometrically dense-core (#21). Persistent binding + conditioning-mismatch = a conditioning-expressivity gap (out-of-scope v1), flagged for Phase-3 — **never** a reason to switch to mean.
(No "representation floor" gate is claimed — generated and real both pass the same sub-F decode, so decode artifacts cancel; over-claiming it would be a tooth that guards nothing.)

**[PRIOR — source reasoning behind the gate, NOT itself a lock].** Conditioning **does vary** across the 4 held-out cities — via the **handed per-tile macro plan** (`zoning_class`, `road_skeleton_class`, `cell_density_bucket`; `conditioning.py:28-37`) + `coastal_inland_river` — **not** via any city/country/morphology *label* (gated: density all-moderate, `admin_region` None for EU #13, `sub_c_morphology_class` constant #22). This makes the gate above **likely to pass** — "conditioned on city D" means "handed D's macro plans," and KS measures quantities (areas/lengths) the macro plan largely determines — **but it is a prior, not a verdict**: only the Task-1 diagnostic discharges the gate, and a failure reopens T5 before any scored run.

**[LOCKED] GAP-not-DRIFT (plan note):** missing aggregation will **not** turn the rebase suite RED (it doesn't exist to break) — its absence is a GAP, not a DRIFT. The plan must track T5's net-new design separately from the (a) repoint; a passing rebase suite must never be read as "aggregation's fine."

---

## 5. T6 — Train-data pipeline (NET-NEW multi-region) **[LOCKED] build-shape (a)**

**Source verdict:** net-new multi-region build + a small repoint; the corpus is **Leonardo-only** (local = Singapore + the frozen holdout manifest). T6 *designs+unit-tests* locally on synthetic multi-region fixtures; the actual EU shard build is a **Leonardo CPU job**.

Today's pipeline is single-region Singapore: `_holdout_ids` does tile-level exclusion within one region, `build_training_shards(region)` builds one city, the training manifest is per-region `schema 1.0`, `datamodule.py:236` loads one `manifest["region"]`. EU holds out **whole cities**, so:

- **Build shape (a):** per-city training manifests + datamodule **union** (reuse the working single-region `build_training_shards` per train city; add a multi-region driver loop + a union over the train-city list). Smallest correct delta; per-city lineage stays clean; incremental. (Rejected (b) one multi-region manifest — bigger build+schema rewrite for no Phase-2 gain.)
- **Train-city source = the G4 corpus roll-up minus the 4 held-out cities** (`addcities_v1.yaml:2`: shipped corpus = canary_v1 + batch2_v1 − exclusions − paris/lyon/madrid + addcities; the G4 roll-up reads all three). ~26 EU cities → ~22 train. **Not** the target `addcities` config (intent ≠ achievement) — reading the roll-up makes the "frozen-and-incomplete" constraint true by construction: corpus-completion's deferred buckets simply aren't in it. **Recorded constraint, not fixed here.**
- **Held-out exclusion = region-level (whole-city), primary, in the build:** the 4 are absent from the train-city list by construction; the `lineage_audit` city-identity guard (whole_city) is the **backstop**. The tile-level `_holdout_ids` is demoted to a vestigial second backstop.
- **Schema & CRS:** per-city training manifests stay **schema 1.0** (multi-region lives in the datamodule union, not a new schema); the holdout *audit* already defaults to 2.0 (`holdout_guard.py:58`, the eval/leak side — separate). Token layer is **CRS-agnostic** (confirmed by code-inspection: sub-F read returns `token_sequence: list[int]`, `sub_f/io.py:32,46`; no CRS in the read/shard/datamodule path; CRS baked out at encode time). **[RESIDUAL]** real cross-CRS *load* spot-check (a non-z32 city's tokens through the union) = Leonardo task.

**Cross-link [LOCKED]:** city-as-unit on both sides — train *unions* the ~22 cities (equal contributors); eval *worst-cases* the 4 held-out. Consistent.

---

## 6. T7 — Phase-3 parking + lock bookkeeping + budget **[LOCKED]**

- **Phase-3 parking (clean carry-forward, not lost):** coherence metric, `assert_coherence_power_sufficient` (§7 power gate), obligation (b) `model_vs_real_effect`, obligation (c) munich→manchester swap, and the macro-planner / tile-coherence winner-preview gate. Phase-3 opens by building the macro planner + stitching for the winning backbone and running the coherence/tile-coherence preview before committing the full hierarchical build.
- **Comparability-lock additions (into the env/version lock):** the schema-2.0 holdout-audit default, the epoch-`E`-held-constant constraint, and the EU multi-region data snapshot (release `2026-04-15.0`, DERIVATION 1.2) as part of "config + commit + data-snapshot fully determine the run."
- **Budget re-estimate:** against the shrunk ladder (likely 1–3 scales, not 4) + EU holdout eval cells (not Singapore) + a real **core-h→GPU-h conversion** from Leonardo boost-node cores-per-A100 (not core-h≈GPU-h). Order diagnostic/cheapest-first inside the guaranteed pre-11 window; align checkpoints to run boundaries so a refill-gap pause lands on a boundary.

---

## 7. Open decisions with Umar's name on them

1. **T5 aggregation bar: worst-case (recommended) vs mean.** Changes which architecture can win the generalization test.
2. **Allocation state/timing.** `AIFAC_P02_222` soft-ends 2026-06-11 (~36k/40k core-h remain; same-size top-up ~1–2 d post-expiry → worst case a short pause, not lost work). GPU sequencing stays parked until confirmed.
3. **Final merge to `main`** — only on Umar's explicit word, `--no-ff`, after the branch suite is green and a `reports/` summary is written. Never force-push/rewrite `main`.

---

## 8. Carry-forward residuals & pre-committed checks

- **Conditioning-discrimination gate** (T5, §4 — the single gate): the Task-1 diagnostic must confirm per-city KS tracks macro-plan differences **AND** the conditioning span distinguishes the EU cities (one concern, formerly tracked as both "macro-tracking" and "multi-country expressivity"). **Failure reopens T5 before any scored run** — it is a gate on the worst-case bar, not a box-tick.
- **CRS cross-load spot-check** (T6): a non-z32 EU city's tokens through the datamodule union on Leonardo.
- **KS-power adequacy** (T3 diagnostic): does the EU held-out (munich incl.) resolve winner-vs-runner-up per-feature KS at the feasible scale.

---

## 9. What this delta does NOT touch (locked at baseline; do not relitigate)

The baseline §3 ruler argument (NLL incomparable across families); the 3-backbone roster (`transformer-AR`, `mamba-hybrid`, `discrete-diffusion`) and the cell-level pure/hierarchical collapse; the mamba-ssm verify-before-lock ("mamba-lock") + re-lock-all; identity-lock comparability; across-job `$WORK` resume; the §13 `transformer-AR` tie-break. From eval-set-gen: the frozen 4-city held-out set; the all-moderate density cut; density-coherence dropped (perplexity_gap is a correct non-leak); seam coherence never an architecture bar; #13+#22 the bundled hard gate; munich dense-core #21 stay-and-record.
