# Eval-set generation — design spec (Phase 1, stage six)

**Date:** 2026-06-01 · **Branch:** `phase-1-eval-set-generation` · **Status:** design locked, pre-plan.
**Predecessor:** sub-G (cross-artifact validator) merged to main (`59bd00b`); `_PHASE1_VALIDATED` on Singapore (494 tiles). **Successor:** training scaffold (consumes this sub-project's held-out manifest + audit).
**Planning protocol:** v2 (`docs/protocols/sub-project-planning-protocol-v2.md`) — this spec was brainstormed against it; §9 (construction-identity exclusion with regime-distinguishing guard) is the spine.

---

## 0. Purpose (what the PRD actually says)

PRD §9 L119: *"All eval metrics are computed automatically against a **held-out set of real cities**. The held-out set is **locked at the start of the project and never used for training**."* Roadmap L143: *"Generate the eval set."*

The eval set is the **locked, held-out, real-city measurement SUBSTRATE** every §9 metric layer is scored against — across the Phase-2 bake-off (4 architectures × 3 scales → scaling curves) **and** Phase-4 production. It is **DATA, not metric-code**: the model-facing metric harness (`src/cfm/eval/perplexity_gap.py`, `shuffles.py`) is a separate, model-dependent concern and out of scope here (see §A).

---

## 1. The seven locked decisions

### A — Scope boundary (A3)

**In scope** = everything derivable from the held-out REAL tiles alone that the locked measurement substrate must carry:
- held-out tiles (a subset of the 494 validated Singapore tiles);
- conditioning labels (§E);
- the never-train lock over **tiles AND their derivatives** (§F);
- the **raw** and **round-tripped-real** baselines (reference distributions + the round-tripped-real ceilings);
- the bref-placeholder construction-identity predicate (sub-G shared authority); and **R2's real-side baseline + check-definition + the G-D1/G-D2 guards** — the tokenizer-on-**model** execution is deferred (§4.1, §7), so this is *not* "R2 in scope" wholesale.

**Out of scope** (→ eval-harness / training-scaffold sub-project, where `perplexity_gap`/`shuffles` already live): the model-scoring orchestration, simulation-viability execution, and any metric not needed to establish a real-side baseline.

**Boundary principle:** in = anything computable from the held-out real tiles; out = anything needing a trained model or an external simulator. The boundary runs *through* R2/D — see precise-statement obligation #1 (§4).

**Rejected:** A1 (data-only — a tile list without computed baselines isn't a locked substrate; "looks locked but isn't"). A2 (set + full metric suite + model harness + CARLA — over-reach, no model to score, YAGNI).

### B — Holdout structure (B1): v1 in-distribution Singapore holdout; generalization deferred

The PRD generalization metric (L113: train A,B,C → eval D) is structurally a **multi-region-*training* capability** — a single-region-trained model has nothing to generalize *from*. Only Singapore is materialized (a second region is a multi-day pipeline-on-a-never-run-region effort, not a few hours; `known_issues` #1 cold-fetch + #3 Sweden densification prerequisite).

- **v1 eval set = Singapore in-distribution holdout.** It scores every §9 layer **except generalization**.
- **Generalization is UNSCORED in v1 — stated as unscored, never "scored and passing"** (an unscored capability read as a met bar is the same failure shape as a vacuous test passing). It is gated by a **Phase-2 multi-region-training-corpus decision** (which regions enter *training* against the Leonardo budget), not by this sub-project.
- **Extracting one region would NOT unlock it** — it buys only a 1→1 transfer test (the *wrong instrument*: "does a Singapore-only model fail on Sweden?" — known yes, architecture-uninformative), at multi-day cost. Rejected.
- **Forward-compatibility (cost-asymmetry — cheap now, expensive to retrofit):** `region` is a **real first-class partition key** in the holdout manifest, the conditioning labels (§E), and the never-train guard (§F). **Done-right test:** materializing a held-out region D in Phase 2 slots in by *adding a partition*, with **zero change to lock/guard logic**. If adding D forces revisiting §F, region was not first-class.

### C — Ground-truth representation (C3 + cancellation-validity check)

For each §9 layer, the model's output is scored against **two baselines** (H1's core/full, generalized):
- **"core" = round-tripped-real** (real tiles → tokens → decode): cancels the tokenizer's information loss → the **architecture-comparison** number (bake-off scaling curves). Charging architectures for the shared tokenizer ceiling would conflate tokenizer+model.
- **"full" = raw-real** (real tiles' original geometry): includes the tokenizer loss → the **absolute-fidelity** number the AV-sim/defense consumer eats.
- **The gap (full − core) is the tokenizer's own contribution, reported explicitly** (the H1 229m-residual shape).

Assignment is **per-layer** because distributional and intrinsic layers use the baselines differently:

| §9 layer | Kind | core (architecture comparison) | full (absolute) |
|---|---|---|---|
| Geometric validity (angles, connectivity, OGC-valid) | intrinsic | model vs **round-tripped-real ceiling** (`100% − bref-rate`, §D) | model's absolute validity % vs **raw** reality |
| Statistical realism (sizes/lengths/density; Wasserstein/KS) | distributional | `Wasserstein(model, round-tripped-real)` | `Wasserstein(model, raw-real)` |
| Topological (connectivity, intersection density, betweenness) | distributional | model vs **round-tripped-real** | model vs **raw-real** |
| Conditioning compliance | distributional | model-cond-on-X vs **round-tripped-real** X-tiles | raw-real X-tiles (reported) |
| **Simulation viability** | intrinsic | — *raw-primary* | **model output, raw** (does the shipped artifact load + drive?) |
| Generalization | — | **UNSCORED in v1** (B) | — |

- **Sim-viability is the deliberate asymmetry:** it measures the actual shipped artifact, so round-tripping it for scoring is meaningless. Round-tripped-real sim-viability is a **tokenizer-ceiling diagnostic** — where H3's bref-degeneracy surfaces *at the consumer* (real Singapore round-tripping to sim-crashing zero-length roads = a v1 *tokenizer* limitation flagged at the sim boundary, not a model failure). **Contract defined here; execution deferred** (needs CARLA + a model) — obligation #3 (§4).
- **R1 — named assumption (not a fact):** `round-tripped-real` cancels the tokenizer term **only if** tokenizer distortion is distribution-invariant between real and model token streams.
- **R2 — measurable check (rides D's instrument):** compare the tokenizer-on-real vs tokenizer-on-model **bref-placeholder / degenerate rate** (§2). Within tolerance → core is clean; **diverge → core carries a reported bias term, never silently absorbed.**

### D — Known-limitation stance in scoring (D3 + G-D2-at-threshold + stratified-rate)

The v1 micro-tokenizer drops the crossing-vertex position by design (`<bref>` = direction + class, not position; v2-scoped per sub-F spec §1.4). This surfaces as the bref-placeholder collapse (V=2 crossing road → zero-length `[anchor,anchor]`). sub-G report-not-gates it by construction identity. The eval continues that stance, **but the real→model context shift makes it harder**:

- **On real data, `_has_outbound_bref` is contract-grounded** (the encoder emits a bref only where sub-E's boundary contract has a real crossing — the H2-validated bijection). **On model output it is ungrounded** (the model emits bref tokens freely). So `anchor + outbound bref → [anchor,anchor]` is **per-instance ambiguous**: (a) the model faithfully reproducing a real v1-limitation crossing road, or (b) the model failing (a degenerate stub it shouldn't emit). Identical token structure; the construction-identity predicate **cannot distinguish them per-instance**.

**Resolution (D3): per-instance exclude, distribution-level judge.**
- **Per-instance:** EXCLUDE the bref-placeholder shape via sub-G's shared predicate `_is_bref_placeholder_collapse` (Gate 6, **import + identity-lock**, never re-derive) — so a faithful model is not penalized; consistent with sub-G's report-not-gate.
- **Distribution-level:** REPORT the bref-placeholder **rate** on model output vs round-tripped-real; the **excess is a reported model-degeneracy term** (this *is* R2). Per-instance can't separate (a) from (b); the rate can.

**Guards (regime-distinguishing; only G-D1 is inherited):**
- **G-D1** (per-instance, *re-proven on model output*): a model-style token block that decodes to genuine degeneracy **without** an outbound bref MUST count as model-invalid. Same shape as sub-G's `GATE_FIRES_on_degenerate_without_outbound_bref`, re-established against a **model-emitted fixture** — not inherited from sub-G's real-data drill (coverage proven on real data does not carry to the regime the model populates).
- **G-D2** (rate-level, **new** — no ancestor): the over-emission guard. Two requirements:
  - **At the threshold, not at 2×:** the fixture emits *just past* the faithful-reproduction rate (must trip) and *just under* (must pass). The tolerance δ is a **chosen, justified number** — "what rate-excess separates *learned the limitation* from *over-emitting*" — not a round default.
  - **Stratified, not global:** a Singapore-wide rate can match round-tripped-real while the model over-emits in dense strata and under-emits in sparse (the distributional vacuous pass). The fixture includes a **stratified-cancellation case** (global matches, one stratum diverges → must trip). **Strata = E's `cell_density_bucket`** (§E).

**Consequences:**
- **Data-flow:** per-instance classification needs the **token block** (`_has_outbound_bref`), not just decoded GeoJSON → the scoring substrate retains token→geometry provenance (as sub-G's `check_decodability` holds `ftokens + geom`).
- **Ceiling (C∩D):** round-tripped-real itself contains the bref degeneracies → the geometric-validity **ceiling is `100% − bref-placeholder-rate`, reported** (the model is scored against achievable, not perfection — the H1/H3 residual surfacing as the eval ceiling).
- **Shared-authority home (plan-time):** lean **import `_is_bref_placeholder_collapse` from sub-G + identity-lock** (avoids touching sealed sub-F); promote to a shared geometry module only if a third consumer appears. Decide at plan.

### E — Conditioning labels (E1): reuse existing derivations, one source

The conditioning surface is **already materialized** — E aggregates it, it does not invent a scheme.

| Dimension (per-tile) | Source (§3-grounded) | v1 conditioning-compliance scorability |
|---|---|---|
| **region** | region config / tile provenance — **partition key** (B), `=singapore` v1 | **UNSCORED** v1 (constant) |
| **density** | sub-D `tile_population_density` (canonical tile label) + `cell_density_bucket` (D's stratification key); buckets in `configs/macro_plan/v1/macro_plan_vocab.yaml` | **SCORED** (real Singapore variation) |
| **morphology** | sub-D `road_skeleton_class`, `zoning_class` (`io.py:51-53`) | **SCORED** |
| **coastal/inland** | sub-C `conditioning.py` `coastal_inland_river` int8 enum (§11.9) | **UNSCORED** v1 (near-constant: Singapore ~all coastal) |

- **One source / import-don't-re-derive** (Gate 6): reads sub-C conditioning + sub-D macro plan, identity-traceable; exact enums/columns **§3-verified at plan** against `macro_plan/v1/macro_plan_vocab.yaml` + sub-C enums.
- **Two density labels, two consumers:** `tile_population_density` = held-out-unit + §9 conditioning; **`cell_density_bucket` = D's stratification key** (intra-tile heterogeneity lives at cell level; a tile-mean masks the spread → D's G-D2 must stratify at cell granularity). **Relationship §3-verified at plan:** if `tile_population_density` is an *aggregate* of `cell_density_bucket`, it provably masks intra-tile spread and D *must* stratify at cell level; if *independent*, D stratifies on the geometry-derived signal that tracks the degeneracy failure mode.
- **Unscored honesty:** v1 scores conditioning-compliance only on dimensions with real Singapore variation (density, morphology); region + coastal are **UNSCORED-stated**.
- **Forward constraint (→ training scaffold):** the model's conditioning vector MUST consume these same sub-C/sub-D quantities, or conditioning-compliance is apples-to-oranges. Traced to one source.

**Rejected:** E2 (fresh eval-specific conditioning scheme — duplicates sub-C/sub-D, drift risk, YAGNI).

### F — Holdout lock + selection (F-lock-3 + G-F4)

**Core invariant:** *no artifact reachable by the training data loader has a lineage that includes any held-out tile* — tiles **and** every derivative (raw + round-tripped baselines, reference distributions, round-tripped-real ceilings, the tokenizer-on-real side of R2).

**Mechanism = structural separation + fail-loud lineage audit (belt-and-suspenders, justified because the failure is silent AND unrecoverable — a contaminated holdout invalidates every eval number undetectably):**
- **Structural:** held-out tiles + all derivatives live in a `holdout/` partition (**region-keyed**: `holdout/region=…`); the training loader's data root **excludes it by construction**.
- **Lineage audit (the checked guarantee over the structural convention):** asserts no training-reachable artifact's lineage includes a held-out tile; **fail-closed on missing provenance** — absent/untracked lineage on a training-reachable artifact = **FAIL**, not pass.
- **Lock artifact:** a frozen `holdout_manifest` (tile IDs + provenance SHAs + region partition + manifest SHA), written once, never regenerated.
- **Provenance-propagation requirement on the baseline-compute code (A-in-scope):** every baseline / reference-distribution write records its source-tile lineage — without it the fail-closed audit can't bind.

**Guards (must FAIL in the leak regime):**
- **G-F1:** held-out *tile* in training's path → trips.
- **G-F2** (load-bearing — the A3 derivative leak a tile-manifest silently permits): held-out-*derived* artifact in training's path → trips.
- **G-F3:** the tokenizer-on-real R2 baseline referenced from training → trips.
- **G-F4** (completeness twin of G-F2): a held-out-derived artifact with **stripped/absent lineage** in training's path → trips **on the absence**. Without it the guarantee is only "no artifact with *recorded* held-out lineage leaks" — strictly weaker than "no held-out-derived artifact leaks," and the gap is where the untracked derivative hides.
- **Region-scaling test (B):** a synthetic 2-region holdout manifest → the audit logic is **byte-identical** (no per-region special-casing).

**Selection:** held-out tiles are a subset of the 494, **stratified by E's scored labels** (cell + tile density, morphology) so the set spans the strata D's stratified check and conditioning-compliance need — **especially the sparse strata** D's G-D2 depends on. sub-D's Layer-3 selector has the `known_issues` #11 sparse-side bug (under-picks exactly those strata) → **§3-read then fix #11 or write a clean stratified selector; not blind reuse** (blind reuse would silently undermine D's guard). Selection is **G's output, frozen once here** — the manifest freezes what G's co-optimization produces (§4.2), not an independent F decision; N trades against the training residual `494 − N` (§G).

**Scope (A-consistent):** this sub-project ships the holdout partition + frozen manifest + the lineage-audit guard function + its tests. The training loader's *actual* exclusion is consumed later by the training scaffold, which calls **this** manifest/audit (one source).

### G — Eval-set size N (G3): a multi-constraint optimization, not a number

N is determined by floors and ceilings; the binding ones are **per-stratum**, not whole-set. **Output = a procedure + a degradation policy; the literal N emerges from measured per-stratum populations at plan/impl time.**

**Floors (push N up):** whole-set Wasserstein/KS power · R2 rate-detection floor (`n ∝ p(1−p)/δ²`) · **D's stratified floor — the binding one** (every cell-density stratum needs power; the **sparse strata bind**).
**Ceilings (push N down):** training residual `494 − N` (tight — the B single-region reality) · baseline-compute cost (minor).

**The deep tension:** per-stratum power pushes N up, training-residual pushes N down, **both starved by the single-region 494-tile pool** — so feasibility (can a held-out set power every stratum *and* leave a viable training set?) is **a real open question the spec answers by measurement, not assertion.**

**Procedure (G3):** (1) measure real per-stratum cell/feature populations across the 494 tiles; (2) compute per-stratum floors (Wasserstein/KS effect size · R2 δ · D's stratified rate); (3) **co-optimize (N, stratified selection)** — selection and N are not independent (F's pick determines per-stratum populations); (4) where infeasible, **degrade gracefully and state it**.

**Graceful-degradation policy — ORDERED fallback (the three options are NOT the same kind):**
1. **Coarsen strata** — if the coarser stratum is still decision-relevant (honest: changes the granularity claim, stated).
2. **Report UNDERPOWERED** — indicative, CI-bounded, *not* a met bar (the unscored-not-passing discipline).
3. **Relax δ — ONLY within the regime-distinguishing bound, never past it.** δ-relaxation that lets the small-N check pass by no longer separating faithful-from-over-emitting is **weakening-the-assertion-to-pass** (a halt-on-validator-fail violation in a sizing costume; the magnitude-vs-identity move at the sizing layer). Legitimate only if the relaxed δ stays **below** D's regime-distinguishing threshold (still fires on the over-emit failure, less margin). Past it → the honest output is UNDERPOWERED, not "passed at relaxed δ." **"Stated rationale" is necessary but not sufficient — the rationale must show the relaxed δ still catches D's failure mode.**

**Rejected:** G1 (fix N now, e.g. the PRD's "100 tiles" — that is the *pipeline-correctness* gate L143, not an eval size; round-number anti-signal). G2 (single whole-set power calc — masks underpowered strata, the vacuous pass at the sizing layer).

---

## 2. The shared R2 / bref-rate quantity (a one-source obligation)

C's R2 and D's rate-check are **one instrument**, and C's ceiling and D's bref-rate are **one quantity**. This is a real simplification, but it makes a single computation a **shared dependency of two consumers** (C's distributional-core baseline + the round-tripped-real ceiling, AND D's degeneracy-rate judge) **with no independent corroborant** — on real data sub-G's bijection grounded the bref; here nothing cross-checks it. Therefore:

- **Define ONE function** — `bref_placeholder_rate(blocks, geoms, strata)` — that classifies the bref-placeholder shape via sub-G's `_is_bref_placeholder_collapse` and returns the rate, **stratified by E's `cell_density_bucket`**. Computed **once** on round-tripped-real (the real-side baseline, in scope); **imported by both** the C ceiling consumer and the D/R2 consumer.
- **One-source obligation (same as D→sub-G predicate):** state explicitly that this quantity is computed once and imported, **never recomputed** in C's path and D's path separately — the "obvious optimization" of computing it twice resurrects the reimplementation/drift bug class inside the eval, exactly what one-source prevents.
- **The G-D1/G-D2 guards test the SHARED function**, not a separate copy — so the guards actually cover what both consumers read. Because there is no independent corroborant, this guard coverage is the *only* check on the quantity's correctness; it is load-bearing.

---

## 3. Dependency graph (the binding web)

- **Scope (A draws the boundary):** A→F (derivatives = audit surface) · A→D (predicate + ceiling in) · A→R2 (real-side in / model-side out) · A→G (baseline-compute = N ceiling).
- **Instrument unification:** C's **R2 ≡ D's rate-check** · C's **ceiling ≡ D's bref-rate** (= the §2 shared quantity, computed once).
- **Labels/granularity:** D's strata **=** E's labels · E's `cell_density_bucket` = D's stratification key · E's `tile_population_density` = held-out-unit + conditioning.
- **Region/partition (B):** B→F (region-keyed partition + scaling test) · B→E (region dim, UNSCORED v1) · B→G (single-region 494 = residual ceiling + starvation).
- **The F∩D∩G triple:** F's stratified selection must cover **D's sparse strata = G's binding floor**; F's #11 fix is a *precondition* for D's G-D2 to bind.
- **Authority (one-source):** D→sub-G predicate (Gate 6) · E→sub-C/sub-D labels · the §2 shared rate quantity · →training-scaffold (model conditioning same source).
- **Provenance:** F→baseline-compute code (must propagate source-tile lineage, or G-F4 can't bind).
- **Threshold:** G's δ-relaxation bounded by D's regime-distinguishing threshold.

---

## 4. Precise-statement obligations (resolved IF written precisely — verified in self-review)

These four near-conflicts resolve **only in their precise form**; the naive form reintroduces the contradiction.

1. **A's boundary runs *through* R2/D.** PRECISE: "R2 **real-side baseline + check-definition + the G-D1/G-D2 guards** are in scope; the tokenizer-on-**model** execution is deferred to the eval-harness (needs a model)." NAIVE (forbidden): "R2 in scope" — reintroduces the ambiguity the obligation exists to kill.
2. **G→F ordering.** PRECISE: "G's procedure runs → produces (N, stratified selection) → F freezes it into the SHA manifest." Selection is G's output, frozen by F — not two independent decisions.
3. **Sim-viability: contract defined, execution deferred.** PRECISE: "C assigns sim-viability raw-primary + the round-tripped-real tokenizer-ceiling-diagnostic role; **execution (model output + CARLA-like sim) is deferred to the eval-harness**."
4. **E's two density labels, two consumers.** PRECISE: "`tile_population_density` = held-out-unit + conditioning; `cell_density_bucket` = D's stratification key; the **aggregate-vs-independent relationship is §3-verified at plan**, because aggregate ⇒ tile-mean masks intra-tile spread ⇒ D must stratify at cell level."

---

## 5. Per-principle consistency (one principle on five surfaces, not five ad-hoc calls)

| Principle | Surfaces it appears on |
|---|---|
| **Unscored/underpowered-stated-not-passing** (×4) | B (generalization UNSCORED) · E (region + coastal UNSCORED) · G (underpowered strata reported, not passed) · the §9 honesty rule throughout |
| **Shared-authority / one-source** (×3+) | D → sub-G `_is_bref_placeholder_collapse` (identity-lock) · E → sub-C/sub-D labels (import-don't-re-derive) · §2 shared bref-rate (computed once) · → training-scaffold conditioning vector (same sub-C/sub-D source) |
| **Construction-identity exclusion + regime-distinguishing guard** (protocol v2 §9 — the spine) | D (per-instance exclude + rate-judge; G-D1/G-D2) · F (lineage audit + G-F4 fail-closed) · G (δ-relaxation bounded by the regime-distinguishing threshold) — every gate proves it still fires on the defect it guards |

---

## 6. Open items for plan time (flagged, not deciding here)

- **§3 reads:** sub-D `macro_core` + `configs/macro_plan/v1/macro_plan_vocab.yaml` (density/skeleton/zoning enums) · sub-C `conditioning.py` enums · the `tile_population_density` ↔ `cell_density_bucket` relationship (aggregate vs independent) · sub-D Layer-3 selector (`known_issues` #11) before fix-or-rewrite.
- **Measurements (G):** per-stratum cell/feature populations across the 494 tiles; the power floors; the feasible (N, selection); any UNDERPOWERED strata.
- **Chosen numbers to justify (not default):** **D's δ** (the regime-distinguishing rate-excess) — and note **G's δ-relaxation bound (§G option 3) IS this same δ, one number** (one-source applied to the threshold — the §2 move one level down; do NOT carry δ as two separate quantities) · the per-stratum power thresholds · N.
- **Sequencing (for writing-plans) — the F∩D∩G triple has a *temporal* edge, not only logical:** order the plan **(1) fix-or-replace the sub-D `#11` selector → (2) G's per-stratum measurement + (N, selection) co-optimization (which *uses* the selector) → (3) F freezes the manifest.** A buggy `#11` selector under-measures exactly the sparse strata G is trying to power, so the #11 fix must NOT land after G has already measured against it.
- **Shared-authority home (D):** import-from-sub_g vs promote-to-shared-module.

## 7. Deferred (out of scope → successor sub-project)

Model-scoring orchestration · simulation-viability execution (model + CARLA) · the tokenizer-on-model side of R2 · any §9 metric not needed for a real-side baseline → the eval-harness / training-scaffold sub-project (where `perplexity_gap` / `shuffles` already live). The training loader's actual holdout exclusion consumes this sub-project's manifest + audit (one source).
