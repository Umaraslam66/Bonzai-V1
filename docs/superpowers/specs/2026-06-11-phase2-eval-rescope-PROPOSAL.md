# Phase-2 eval re-scope — PROPOSAL (APPROVED 2026-06-11)

**Status: APPROVED by the PI + Umar 2026-06-11, all four §5 knobs LOCKED:** (1) floor =
STRICT `min_T KS(D,T)` (median-over-T context only); (2) Lane-M scope = ALL 38 training
cities; (3) Lane-S aggregate = median + p90 of per-stratum excess; (4) floor-explosion
halt threshold = 0.5. The revision is applied in: readiness-closure spec §8 (+ §4.3/§4.4/§4.5
annotations), bake-off delta-spec §4 discharge note, plan Goal + Tasks 24/25/26 + Phase-8
header, PRD §9. This file is retained as the approved design basis.

**Evidence base:** `reports/2026-06-11-residual-character-recon.{md,yaml}` (anchored, RC=0) +
the three diagnostic runs (`reports/2026-06-10-localization-diagnostic.*`). The controlling
facts: cross-city character at δ=0.15 is real, broad (all 6 pairs, 0.27–0.43), shape-dominated
(buildings 70% beyond-median), and bounded BELOW ~30% significance rate under any bucketed
conditioning (kitchen sink: 781 significant pairs). Same-conditioning ⇒ same-distribution is
empirically false for real cities under any practical conditioning vocabulary.

**The T5 spec anticipated this.** `2026-06-09-phase-2-bakeoff-delta-design.md` §4 (the gate
co-located with the bar): *"If the diagnostic shows materially different real feature
distributions for same-stratum tiles across cities … the per-city miss is ambiguous and T5
REOPENS before any scored run."* That fired. This proposal is the designed answer to the
ambiguity it named: make a per-city miss decomposable into "didn't generalize" vs "city
idiosyncrasy" using the MEASURED floor.

---

## 1. The re-scoped bar (what Phase 2 now proves)

**Claim under test:** *given conditioning for geometry it did not train on, the model produces
plausible, realistic geometry matching that character — by learned grammar, not memorization.*

Two eval lanes, sharing the existing suite (validity/decodability/realism layers unchanged):

### Lane S — SCORED generalization (held-out cities; identity ABLATED)
- Generate on held-out city D's eval-set tiles with `conditioning_ablation="no_city"` (identity
  block zeroed; character carrier live). Identity-memorization cannot satisfy this lane by
  construction — the model is never handed a trained label to look up. (Held-out identities are
  untrained embeddings anyway; explicit ablation makes the claim clean rather than incidental.)
- **Character fidelity, floor-judged:** per (metric, qualifying stratum), compute
  `KS(gen_D, real_D)`. The bar is NOT zero. It is the **measured same-conditioning cross-city
  floor**: `floor_D(metric, stratum) = min over other real cities T of KS(real_D, real_T)`
  (strict variant; median-over-T reported as context — PI knob §5.1). Rationale: the model
  cannot be required to match D more closely than the closest REAL city of identical
  conditioning matches D — the conditioning does not determine the distribution that tightly
  (recon-proven). Score per city: `excess_D = max(0, KS(gen_D, D) − floor_D)` aggregated over
  the stratum profile (median + p90, both reported).
- **T5 aggregation retained as-is:** worst-case (max) `excess` across the 4 held-out cities is
  the binding number; per-city values + binding city reported; the **binding-city power gate**
  (gap vs that city's own `C/√n` resolution floor) and the munich #21 rules carry over verbatim
  — only the quantity changes (excess-over-floor instead of raw KS-vs-zero).

### Lane M — MEMORIZATION DETECTION (the founding anti-overfit tooth, strengthened)
- **Nearest-training-city discriminator.** On the strata where D is measured-distinct from
  training city T (real-real `KS(D,T) ≥ 0.15`, BH-significant — selected from REAL data only,
  no generated-data leakage), require `KS(gen_D, real_D) < KS(gen_D, real_T)` in aggregate
  (median over discriminating strata), **for every training city T**; report all margins.
  A regurgitator emits some training city's distributions; the recon proves those sit
  KS ≈ 0.2 (median) from D exactly where this tooth looks — measured discriminator power,
  not assumed. **FAIL ⇒ memorizer ⇒ no crowning** (hard halt at the decision layer).
- This REPLACES "plausibility alone" as the anti-overfit guarantee: a memorizer passes realism
  metrics by construction (real-trained-city geometry IS realistic); it cannot pass Lane M.
- Cost note: needs real per-stratum distributions for training cities (extraction exists; CPU;
  ~1h for all 38 — v1 may restrict to the k most-confusable per D; PI knob §5.2).

### Lane D — DIAGNOSTIC (seen cities; identity LIVE)
- Per-SEEN-city bars with identity conditioning live — valid by construction ("a miss can't
  mean 'wasn't told the city'"). Uses the per-region emergence-floors artifact (Task 13) as-is.
  Not a generalization claim; never aggregated with Lane S.

### The separation guarantee (identity floor without a memorization shortcut)
Identity exists in training and Lane D only. Lane S zeroes it; Lane M would catch a model that
leaked identity-shaped memorization through the character carrier. The Task-24(a) separability
switch (`conditioning_ablation` zeroing one block without touching the other) is exactly the
instrument — unchanged from the locked plan.

---

## 2. The recalibrated gate's new role: conditioning-floor MEASUREMENT

The gate-(i) machinery (Task 22's δ-floor verdict + bref exclusion + Task 21's counters) is
**demoted from PASS/FAIL halt to the instrument that produces the floor artifact**:

- **Reports:** the per-(city-pair, metric, stratum) real-real KS table over qualifying strata;
  the δ-ladder profile (0.15…0.50 sweep); `floor_D` per held-out city; per-city coverage +
  bref counters. Written as a versioned, sha-stamped artifact (the Lane-S bar input).
- **Still halts (regime-distinguishing, integrity-only):**
  1. all existing extraction-integrity halts (F3 coverage ceiling, zero-tile city, reader-side
     sha/lock, TRAIN_TOKENS guard) — unchanged;
  2. UNSUPPORTED (zero qualifying comparisons) — unchanged;
  3. **floor-collapse sanity:** measured median real-real KS below the SG single-region
     resolution floor (0.049) ⇒ the measurement contradicts every prior run ⇒ halt (fires on a
     broken extraction, never on healthy data);
  4. **floor-explosion sanity:** median real-real KS > 0.5 ⇒ conditioning carries ~nothing and
     Lane S cannot reward conditioning compliance ⇒ halt to PI.
- **Removed:** the "PASS at δ=0.15 ⇒ enrichment worked / FAIL ⇒ T5 stays open" acceptance
  (spec §4.5). Enrichment acceptance moves to the bake-off itself (beat the floor + pass
  Lane M), where the recon says it belongs. Nothing else is removed; the δ ladder remains
  REPORTED so the original question stays answerable from every artifact.

---

## 3. Character carrier (PRETRAINING capability; proposal only — wiring is Task 24, later)

Locked context honored: character conditioning must be RECEIVED during training to learn
character→geometry; not deferrable to a post-training stage. The recon names what the carrier
must transmit (the residual's content):

- **Buildings (shape beyond median):** per-cell building-size distribution sketch — log-median,
  log-IQR, p90/p50 ratio, building count (4 channels; the recon's probes verbatim).
- **Roads (fine location):** per-cell median road length (1 channel).
- **Delivery:** the provisioned-but-empty `macro_tokens` channel, as continuous (scaled-float →
  learned projection) inputs rather than vocabulary buckets — the recon shows bucketing is the
  losing move; continuity is the point. Schema/quantization details are wiring-time design
  (Gate-2 read of the carrier), parameterized like the locked plan's Task-24 character steps.
- Honest expectation, set by the recon: the carrier is for MODEL QUALITY (transmissible shape
  signal); it does not and cannot flip a stratum-measured gate. Derivation is shard-build-time
  from existing sub-C/sub-D artifacts (CPU; wholesale shard rebuild under the
  uniform-defect-level rule; NO sub-C regen).

---

## 4. Blast radius (honest enumeration; nothing edited yet)

| Artifact | Change |
|---|---|
| Spec `2026-06-10-readiness-closure…` §0/§1 | "reopened T5 gate that can render PASS" → "T5 bar valid by construction (floor-judged)"; Mission-B expressivity acceptance rewording. |
| Spec §4.3 | Verdict semantics: PASS/FAIL → measurement + integrity halts (§2 above); δ=0.15 stays as the REPORTING ladder anchor and Lane-M distinctness threshold. |
| Spec §4.4 / PI-call #2 | "Both, asymmetric" SURVIVES; character = continuous distributional carrier via macro_tokens (not a stratum bucket); identity floor unchanged. |
| Spec §4.5 teeth | "PASS = enrichment worked" replaced by: floor artifact + Lane-M tooth + constant-column/all-None guards (which survive verbatim). |
| Bake-off delta-spec §4 (T5) | The co-located gate is DISCHARGED by this re-scope (its REOPEN branch fired and is now answered); worst-case aggregation, binding-city power gate, munich rules survive with quantity = excess-over-floor. |
| Plan Task 24 | 24(a) identity floor + ablation switch + guards: survives nearly verbatim. 24(b) character field: re-parameterized to the macro_tokens carrier (bigger; own mini-spec at dispatch; shard rebuild wholesale). |
| Plan Task 25 | HALT-gate semantics replaced: produce + verify the floor artifact (4 held-out cities; + training-city distributions for Lane M), sanity halts clean ⇒ **T5 closes as re-scoped**. No PASS/FAIL on city-difference. |
| Plan Task 26 | `pick_winner` gains `memorization_check_ok` (Lane M) paired with `structural_check_ok`; decision quantity becomes per-city excess-over-floor; everything else unchanged. |
| PRD §9 (Evaluation) | Generalization metric wording: "match D's distribution when conditioned on D" → "match within the measured same-conditioning cross-city floor; memorization failed by the nearest-training-city discriminator." (Also the plan's §6/§10 PRD references re-checked at edit time.) |
| Eval-set artifacts | UNCHANGED (holdout manifests, freeze, KS-gap sizing all survive). |
| Memories/handoffs | T5-semantics entries updated at revision time (not before approval). |

**Explicitly NOT changed:** Tasks 0–23 deliverables (all shipped teeth survive); the bake-off's
Rule-2 curve discipline; emergence floors; resume/integrity work; the founding anti-overfit
goal — strengthened from "zero cross-city difference" (impossible) to a measured-power
discriminator that a memorizer cannot pass.

---

## 5. PI knobs inside this proposal (carrying Umar's name; none pre-resolved)

1. **Floor statistic:** strict `min_T KS(D,T)` (recommended — hardest defensible bar) vs
   median-over-T (looser). Changes how hard Lane S is.
2. **Lane-M scope:** all 38 training cities (complete; ~1h CPU per re-derive) vs top-k
   most-confusable per held-out city (cheaper, k to pick).
3. **Lane-S aggregate:** median + p90 of per-stratum excess (recommended) vs worst-stratum
   (likely too noisy at stratum granularity — the recon's broad-shallow profile argues against).
4. **Floor-explosion threshold** (§2 halt 4): proposed 0.5 — essentially "conditioning carries
   nothing"; framing wanted.

## 6. Verification obligations the revision must carry (teeth, named now)

- Floor artifact: sha-stamped, write-once beside the holdout manifests; Lane S refuses to run
  against a floor whose sha/lock is absent (mirrors Task-20 reader-side discipline).
- Lane-M tooth must be shown to FIRE: synthetic regurgitator fixture (generated := training
  city T's real samples) must FAIL Lane M; an oracle fixture (generated := D's own held-out
  samples) must PASS both lanes. The gate-must-distinguish-regimes pair, pre-named.
- Floor-collapse/explosion halts: regime fixtures both directions.
- No-leakage pin: discriminating-strata selection provably reads only real data (test asserts
  the selection function's inputs).
