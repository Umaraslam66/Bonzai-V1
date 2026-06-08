# Eval-set-generation design spec — multi-region EU corpus (2026-06-08)

**Status:** design locked (brainstorm complete), awaiting PI review before plan/implementation.
**Scope:** the held-out evaluation set + metrics + de-Singapore generalizations for the
Phase-2 multi-region EU corpus, feeding the architecture bake-off. This is a GENERALIZE job
over the existing Singapore-shaped eval machinery (`src/cfm/eval/holdout/*`,
`src/cfm/eval/perplexity_gap.py`), not a rebuild.
**Source-of-truth discipline:** every load-bearing claim below cites the file (`path:line`)
or the read-only measurement (dated) it traces to. This thread twice leaned on memory and
source corrected it (`morphology_stratum` definition; `perplexity_gap` behaviour) — so no
claim here is asserted from memory. See the §11 source ledger.
**Operating protocol:** `docs/protocols/sub-project-planning-protocol-v3.md` (six gates +
§10.1 write-once sizing / §10.2 relative thresholds / §9 construction-identity exclusion).

---

## 0. Corpus state (the input — frozen, not reopened)

- Validated **42-city / 670,030,892-token / 23,971-tile** EU corpus, every city at sub-F
  **DERIVATION 1.2**, four-part G4 DoD PASS
  (`reports/2026-06-08-phase-2-multiregion-corpus-closeout.md`;
  per-city counts in `reports/2026-06-05-phase-2-g4-corpus-dod.yaml`).
- Lives only on Leonardo `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed`
  (release `2026-04-15.0`), single-copy. **Not reopened by this sub-project.**
- The **train floor is 550M** (`scripts/multiregion/build_g4_rollup.py:58`
  `TARGET_TOKENS = 550_000_000`); its comment (`:56`) records that 550M is the
  **corpus/merge gate** and the **r=20 TRAIN-split floor is a separate concern** — the two
  must not be re-conflated (the "56k" class of error). The r=20 derivation (30M params ×
  20 tokens/param = 600M) is the training-data *ideal*, above the ratified 550M floor
  (`reports/2026-06-06-eval-set-gen-scoping.md`, "FLOOR INTERPRETATION").

---

## 1. Held-out CITY split (T1)

**The set: `glasgow`, `eisenhuttenstadt`, `munich`, `krakow`** — held out whole-city.

Per-city facts (`reports/2026-06-05-phase-2-g4-corpus-dod.yaml`):

| city | morphology | density | country | CRS (UTM) | tokens |
|---|---|---|---|---|---|
| glasgow | planned-grid | moderate | GB | EPSG:25830 (z30) | 14,689,892 |
| eisenhuttenstadt | modernist-sprawl | moderate | DE | EPSG:25833 (z33) | 4,261,436 |
| munich | mixed | moderate | DE | EPSG:25832 (z32) | 10,060,491 |
| krakow | medieval-organic | moderate | PL | EPSG:25834 (z34) | 17,118,283 |

- Held-out = **46,130,102** tokens; **train = 623,900,790** → clears the 550M floor (+73.9M)
  and the r=20 600M ideal (+23.9M). **No add-cities** (the scoping doc's "add 3–4 cities" was
  a stale artifact of the pre-final ~551M corpus; recomputed on source counts).
- All four morphology classes covered; four distinct UTM zones (30/33/32/34) — exercises the
  multi-CRS path Singapore never did.

### 1.1 The admissibility rule (LOCKED)

**Country is part of "type." A held-out city is admissible only if its morphology, density,
country, AND CRS-zone each remain represented in the TRAIN split after holdout.** Verified
city-by-city against the source roster: all four pass on all four axes (e.g. krakow's PL has
lodz/szczecin/tychy/warsaw in train; z34 has budapest/debrecen/umea/… in train).

- **Reason:** the eval's job is that a held-out MISS is *diagnostic* — it must read "didn't
  generalize," nothing else. A held-out city from a country/zone with zero training
  representation makes a miss ambiguous (failed to generalize vs. never saw this national
  geometry style). That ambiguity destroys the metric. v1 tests **within-regime
  generalization**.
- **What it rejected:** `tallinn` (EE-unique) and `lisbon` (PT-unique) — holding either out
  removes its country from train entirely. `krakow` replaced `tallinn` (medieval-organic /
  moderate / PL-in-train / z34-in-train — verified all three).

### 1.2 Density: all-moderate — a RECORDED scope cut, not silent

The four held-out cities are all `moderate` density. The set spans all four morphologies but
**only one density class** → the eval is **silent on dense-core and sparse generalization**.

- **Reason it is acceptable:** (a) PRD names no dense-core-primary deployment target — density
  is one conditioning dimension among ~8 (PRD §8:97), and the held-out test is
  generalization-vs-memorization in general (PRD §9:113); density *compliance* lives in the
  separate conditioning-compliance probe (PRD §9:115). (b) Full per-morphology dense-core
  coverage is **infeasible within the train floor**: cheapest dense-core city per morphology
  sums to ~97.3M held-out → train ~526.6M, **below 550M** (driven by rotterdam, the only
  admissible modernist dense-core city, at 44.6M). So density coverage could only ever be
  partial and lopsided — worse than a clean density-controlled morphology probe.
- **Cost:** dense-core (12 cities) + sparse (4 cities) generalization untested this round.
- **Revisit-trigger:** revisitable at a **phase transition** — a **project-wide locked
  principle (PI ruling, this thread):** a new phase re-anchors the held-out set as a *fresh
  artifact* (consistent with protocol §10.1's write-once-*per-version* — a new phase is a new
  artifact, not a mutation of the frozen one). This is **NOT a literal §10.1 clause** — §10.1
  (`protocol-v3.md:311-313`) is the write-once *sizing* rule. So the all-moderate cut is a
  **scope cut, not a permanent blind spot**. Density *generalization* is a named separate probe
  for a future round, never folded into this v1 aggregate (same discipline as cross-national
  extrapolation, §4.3).

---

## 2. Write-once held-out manifest contract (T2)

Extends the existing region-keyed writer (`src/cfm/eval/holdout/manifest.py`) — already
write-once (`freeze_holdout_manifest` refuses overwrite, `:58-64`), byte-deterministic
(`canonicalize_yaml` + self-excluding `manifest_sha256`, `:52-55`). Schema bumped to `2.0`
(new fields: holdout-kind declaration + achieved properties).

### 2.1 Schema

```yaml
manifest_schema_version: "2.0"
corpus_release: "2026-04-15.0"
derivation_version: "1.2"                 # pins to the gate_d-coherent corpus
held_out_cities: [eisenhuttenstadt, glasgow, krakow, munich]   # sorted
regions:
  <city>:
    partition_path: holdout/region=<city>
    holdout_kind: whole_city              # the declaration the audit city-guard reads (§6)
    morphology: <m>  density: <d>  geography: <cc>  crs: <EPSG>   # stratification labels (NOT scored conditioning)
    n_tiles: <int>                        # frozen-corpus tile count
    n_usable_tiles: <int>                 # >=3 interior road edges, after water filter (coherence power unit; T4 gate(b))
    tokens: <int>
    tiles: [{tile_i, tile_j, provenance_sha256, macro_vocab_sha256}, ...]   # FULL enumeration
totals:
  held_out_tokens: 46130102
  train_tokens: 623900790
manifest_sha256: <self-excluding>
```

**Achieved `n_usable_tiles` (measured read-only 2026-06-08, all four full tile sets):**
glasgow **523/549 (95%)**, eisenhuttenstadt **579/616 (94%)**, munich **156/171 (91%)**,
krakow **601/616 (98%)**; total 1,859 usable of 1,952. munich is inland inner-core (#21) — its
91% is the *opposite* of malmo's 8% water-loss; **measured, not inferred** (the malmo lesson).

### 2.2 The whole-city declaration is correct-by-construction (not asserted)

`holdout_kind: whole_city` drives the §6 city-guard. A faithful guard over a WRONG declaration
still leaks, so the declaration is **proven at build, not claimed**:
- **(a)** assert `held_out_cities ∩ train_cities = ∅` (no city on both sides);
- **(b)** assert each held-out city's enumerated `tiles` **matches the frozen-corpus tile set**
  (count against G4 `n_tiles`; the manifest must not invent or drop tiles).
  - **Wording is load-bearing:** the pass-message reads **"matches frozen-corpus tile set,"
    NEVER "fully enumerated."** munich's 171 is whole-*corpus*-city but inner-core by extent
    (#21) — "fully enumerated" would invite a future reader to mis-verify (b) as *geographic*
    completeness. That is the false-DONE class; one careful word prevents it.

### 2.3 §10.1 (write-once sizing) compliance

- **Sized by** the locked 4-city set (T1 — morphology coverage + country-in-train + train
  floor; today-justifiable), **NOT** by a provisional/expected-to-change parameter.
- **Records the achieved property** (`n_usable_tiles` per stratum, tokens, train_tokens).
- **Defers** the resolution/power-adequacy check to the consumer that can finally evaluate it
  — the first-model power gate (§7), fail-loud. **No provisional resolution number is baked
  in** (this is precisely the 0.076-pinning trap §10.1 exists to prevent).
- **Pin + re-lock:** pinned to DERIVATION 1.2; a future #13/#22 value-bearing re-derive is a
  *new corpus generation* → a deliberate re-lock (consistent with write-once-per-version, not
  a silent move). `_EVAL_SET_LOCKED` marker, force-added to git, verified vs disk.
- **Path (note-back from plan T1/T6):** the EU eval-set freezes to
  **`data/processed/eval_set/2026-04-15.0/multiregion/`** — the frozen SG eval-set already
  owns `eval_set/2026-04-15.0/` (same release, write-once), so the EU set lives in the
  `multiregion/` subdir. `paths.py` gets `multiregion_holdout_manifest_path` /
  `multiregion_eval_set_locked_marker`; the SG set is untouched.

---

## 3. Macro-plan-coherence metric (T4) — the net-new bake-off bar

Scores whether a generated macro plan's cell-level structure is *spatially coherent*. Mirrors
`perplexity_gap`'s matched−null discipline. **Scores the GENERATED macro plan**, not handed
conditioning (see §3.4 — the circularity firewall).

### 3.0 The sub-D substrate (verified from source, not the scoping doc's framing)

`macro_core.parquet` (`src/cfm/data/sub_d/io.py:41-79`): per-cell `zoning_class`,
`cell_density_bucket` (CELL-keyed, 64 cells, null on inactive); `road_skeleton_class`
(EDGE-keyed, **112 internal edges**, `evidence.py:12-19`). `road_skeleton_class` is an
**ordinal bucketed road-crossing-count** (`evidence.py:201-245` emits `road_crossing_count`
per edge; `configs/macro_plan/v1/macro_plan_vocab.yaml:3489-3505` buckets it: token 0 =
`[0,1)` = no crossing, {1,2,3} = `[1,4)/[4,9)/[9,∞)`). The 8×8 lattice's **112 internal edges
both connect adjacent cell-pairs (zoning/density agreement) and carry the skeleton class**
(`src/cfm/data/sub_d/lattice.py:1-119`; axis 0: `(i,j)↔(i+1,j)`, axis 1: `(i,j)↔(i,j+1)`).

### 3.1 The metric — shuffle-gated coherence gap, over the 6×6 INTERIOR

Per held-out tile, per attribute: `gap = score(real arrangement) − score(interior-permuted)`,
over the **6×6 interior** (cells with `i,j ∈ 1..6` of the 0-indexed 8×8 grid; boundary cells
excluded — their roads exit to neighbours, which is the §4.2 cross-tile seam, not an intra-tile
defect). The null **permutes the attribute among interior-interior edge/cell slots only** —
fixing the interior count, isolating *arrangement*.

- **Zoning (CELL-keyed, categorical):** per internal edge **where BOTH incident cells are
  active (non-null zoning)**, agreement = incident cells share a class. Active-inactive and
  inactive-inactive edges are **excluded, NOT scored as disagreement** — the same active-active
  discipline as skeleton's road-touched guard. Without it, every edge on the built-up-area
  perimeter (built cell ↔ empty cell) would score as disagreement, penalizing *geography*, not
  incoherence. (Active-active also ties zoning to its power unit the way skeleton's edge set
  does.) v1 = **same-class only** (no transition table → no permissive-table vacuity risk by
  construction). *If* same-class under-separates (weak teeth-3), add a **TRAIN-derived**
  compatible-transition table (train-split only, no held-out leak) **with non-vacuity teeth**
  (a deliberately-wrong adjacency must score as disagreement; §3.3 teeth-4).
- **Skeleton (EDGE-keyed) — S1, intra-tile, two terms** (interior cells only; the boundary is
  the cross-tile seam = §4.2, deliberately excluded here):
  - **Continuity** = fraction of road-touched interior cells with road-degree ≥ 2 (dead-end
    avoidance);
  - **Fragmentation** = **giant-component fraction** = edges in largest connected component /
    total interior road-edges (catches disconnected islands that continuity alone cannot).
  - Road-carrying edge = `road_skeleton_class ∈ {1,2,3}` (verify the zero-bucket token at
    build against the locked macro vocab).
- **Density (CELL-keyed, ordinal): DROPPED — see §3.4.**

**Threshold** is set AFTER measuring, **relative to the held-out ground-truth tile's own
per-stratum gap** (the model should approach it, not exceed it = over-smoothing, nor fall
short = noise) — never a guessed absolute (protocol §10.2).

### 3.2 Discrimination is real (read-only TRAIN probe, 2026-06-08)

On TRAIN tiles (6 cities, interior-permutation null), both terms separate real from shuffle:
pooled continuity gap +0.120 (Cohen's d≈0.8, 78% of tiles positive), fragmentation gap +0.183
(d≈0.8, 82% positive). **Fragmentation gap > continuity gap** and real giant-frac (~0.96)
collapses harder under shuffle than continuity (~0.80) → the two terms are **not redundant**
(empirical support for the fragmentation term). This validates *metric discrimination*
(real-vs-shuffle); it is NOT the gate-power number (§7).

**Coverage caveat (no overclaim):** the probe's cities cover planned-grid/**dense** (valencia)
and medieval-organic/**dense** (vienna), whereas the held-out set holds out planned-grid/
**moderate** (glasgow) and medieval-organic/**moderate** (krakow). So for those two held-out
morphology×moderate strata, §3.2 is **pooled** evidence that the metric reads arrangement *in
general* — NOT per-held-out-stratum validation at moderate density. Per-held-out-stratum
discrimination is confirmed at the §7 first-model gate, which is where it is backstopped
regardless.

### 3.3 Teeth-proofs (red-before / green-after — required before the metric gates)

1. **Shuffle teeth:** a real tile's gap > margin; its interior-permuted version → gap ≈ 0.
2. **Three-way teeth-2** (the must-distinguish set; uniform/noise alone are insufficient):
   | synthetic | continuity | fragmentation | verdict |
   |---|---|---|---|
   | uniform (every edge a road) | 1.0, gap≈0 | 1.0, gap≈0 | FAIL (shuffles to self) |
   | noise (random edges) | low | low | FAIL |
   | **ten disconnected loops** | **high** | **low** | **FAIL on fragmentation while passing continuity** — proves the term is non-redundant |
3. **Real-vs-permuted separation** on a held-out sample.
4. **(zoning, only if a transition table is added)** a deliberately-disallowed adjacency must
   score as disagreement (non-vacuity); a common transition scores as agreement.

### 3.4 Why density is DROPPED, and the perplexity-gap is NOT (the circularity firewall)

**Stated invariant:** *a matched−shuffled gap defends against MARGINAL/mode conditioning
(the shuffle cancels it) but gives ZERO defense when the per-instance ARRANGEMENT itself is
handed to the model as conditioning.* (Memory `feedback-shuffle-gap-marginal-not-arrangement`.)

The conditioning vector (`src/cfm/data/training/conditioning.py:28-37,54-63`) hands the model:
`zoning_class` and `road_skeleton_class` as **tile-level MODES**
(`morphology_stratum.dominant_zoning_class` / `.modal_road_skeleton_class`,
`eval/holdout/labels.py:46-48,95-98` — `Counter.most_common(1)`), but `cell_density_bucket` as
a **PER-CELL scalar** (`conditioning.py:32,58`; `build_shards.py:121`; `shard_schema.py:31`).

- **Zoning, skeleton — CLEAN:** conditioning fixes only the tile MODE (marginal); the
  shuffle-gap cancels it; the per-cell/per-edge ARRANGEMENT the metric scores is never
  conditioned.
- **Density — CIRCULAR:** the model is handed the per-cell density ARRANGEMENT itself; a model
  echoing its conditioning reproduces the real arrangement → its density gap = the real gap.
  The shuffle-gap cannot save it. → **density coherence term DROPPED in v1.**
  - **Revisit-trigger:** returns in **v2 only if** a macro-plan stage conditioned on
    **TILE-level** density (not per-cell) is pinned, so per-cell density becomes a GENERATED
    output the metric can fairly score.
  - Density **compliance** (PRD §9:115) is a *separate* test, unaffected; density isn't
    dropped from the eval, only from the *coherence* bar.
- **The non-leak twin — do NOT "fix" it:** `src/cfm/eval/perplexity_gap.py:41-114` grades
  **FIXED** held-out micro tokens under matched-vs-shuffled conditioning prefixes
  (`gap = NLL_shuffled − NLL_matched`). It scores tokens the model did not invent; conditioning
  (incl. per-cell density) is the **independent variable** whose USE is tested. Handing it
  per-cell density is the entire point, not a leak. **Same field (`cell_density_bucket`),
  opposite treatment, both correct.** Anyone "fixing" the perplexity-gap's per-cell density is
  breaking the use-test.

### 3.5 Build-time gates (recorded, must all hold before the metric gates)

- **(a)** three-way teeth pass (§3.3).
- **(b)** `n_usable_tiles` measured + recorded for all four held-out cities (done, §2.1) — the
  power unit is *usable* tiles (active, ≥3 interior road edges, after the water filter), NOT
  total tiles (the malmo lesson).
- **(c)** at the first trained model, measure the **model-vs-real** coherence-gap effect size
  per stratum; confirm usable-n resolves *architecture discrimination*, else fire the
  munich→manchester swap (§7). **The §3.2 real-vs-shuffle d≈0.8 is NOT this number.**

---

## 4. Scope locks (T5)

### 4.1 admin_region EXCLUDED from the eval (HARD GATE)

`admin_region` is all-`None` for every EU tile (#13/#14) → no value-bearing signal. The eval
conditioning vector **excludes** it; conditioning-compliance for admin_region is a recorded
v1 blind spot. (Also: `country`/`climate_zone`/`era_class` are SG constants and aren't in
`TileLabels` or the vector at all — see §5 / #22; no SG constant is scored anywhere.)

### 4.2 Cross-tile seam coherence DEFERRED — and never an architecture bar

S1 scores intra-tile connectivity (interior cells). Cross-tile seam (roads continuing across
the tile boundary) is **deferred**.

- **Reason (the decisive one):** cross-tile connectivity in the shipped system is produced by
  the **rules-based STITCHER** (PRD §4:51, "Stitching is a rules-based step, not a model"),
  **identical across every bake-off architecture** → seam coherence can **NEVER discriminate
  architectures**. It **validates a non-learned component**, not a model.
- **Revisit-trigger:** a **one-time correctness check** when sub-E tile-to-tile semantics land
  (the external-edge substrate is `EXTERNAL_DEFERRED` today — `sub_d/lattice.py:162-170`;
  `road_skeleton` evidence is internal-edges-only, `evidence.py:201-245`). **Do NOT let v2
  resurrect seam as an architecture-discriminating bar.**

### 4.3 Within-regime ≠ PRD §9.113 (necessary, not sufficient — recorded explicitly)

PRD §9:113 frames the generalization test as holding out a whole *region D* (cross-region
extrapolation). v1's held-out test is **within-regime** (held-out country + CRS-zone remain in
train, §1.1) — a cleaner, narrower test chosen to keep a miss unambiguous.

- **Therefore:** passing v1's held-out test is **necessary-but-not-sufficient** for PRD
  §9.113's cross-region foundation-model claim. **v1 does NOT close the PRD's headline claim.**
- **Revisit-trigger:** literal region-holdout (cross-region / cross-national extrapolation) is
  a **separate, separately-reported probe** for a future round, never folded into this
  aggregate (same discipline as the density-generalization cut, §1.2).

---

## 5. De-Singapore generalizations (T6) — plumbing + a recorded HARD GATE

- **`src/cfm/eval/holdout/paths.py`:** per-city CRS-label resolution (drop hardcoded
  `EPSG3414`/`singapore`); tile-dir label `tile=EPSG<zone>_i{i}_j{j}` (format confirmed on
  Leonardo, e.g. `tile=EPSG25832_i337_j2662`).
- **`labels.py`: NO morphology change needed.** It already scores the real multi-region signal
  `morphology_stratum` (sub-D `dominant_zoning` + `modal_skeleton`, per-tile) and keeps
  `sub_c_morphology_class` in `UNSCORED_V1_DIMENSIONS` (`labels.py:34-36`). The scoping doc's
  "drop the SG constant, read per-tile values" is moot — the per-tile `morphology_class`
  values ARE the SG constant; `morphology_stratum` is the real signal it already uses.
- **`resolution.py`:** recompute the gap-floor on the **TRAIN split ONLY** — never let held-out
  cities into the floor's own calibration (test-into-threshold leak; same rule as the zoning
  transition table §3.1).
- **Reusable as-is:** `lineage_audit.py` (region-keyed — but see §6), the manifest schema
  (§2), `perplexity_gap.py` (region-agnostic).

### 5.1 #22 — SG-constant conditioning leak, BUNDLED with #13 (one re-derive)

The frozen conditioning data is SG-hardcoded beyond admin_region:
`sub_c/conditioning.py:29` defaults `morphology_class="Asian-megacity"` (+ `era_class`), never
overridden, so every EU tile carries `morphology_class=Asian-megacity`, `country=SG`,
`climate_zone=tropical_rainforest` (verified 2026-06-08 across manchester/munich/krakow/malmo).

- **Inert in v1** (value-AGNOSTIC conditioning — model sees field-slots, not values; #13), so
  not a v1 blocker; not scored anywhere (§4.1).
- **HARD GATE for value-bearing conditioning, BUNDLED with #13:** the corpus re-derive that
  fixes admin_region MUST also fix morphology_class/country/climate_zone — **one re-derive, all
  SG value-bearing leaks, never separately** (the partial-fix trap). Recorded in
  `docs/known_issues.md` #22 (and #13 cross-refs it).
- **`coastal_inland_river` is the exception:** genuinely derived per-tile
  (`conditioning.py:56-72`); confirmed real/non-null/varying across all 42 (2026-06-08: 171
  inland / 18 coastal / 244 riverside / 7 coast+river; coastal codes correctly on coastal
  cities). It is the one real value-bearing EU field — value-bearing-gated for v1.

---

## 6. Holdout-leak guard (T7)

`audit_no_holdout_leak` (`src/cfm/eval/holdout/lineage_audit.py:51-73`) keys the holdout set on
`(region, tile_i, tile_j)` and trips when any training artifact's lineage intersects it;
fail-closed on absent lineage (G-F4). **This keying was CORRECT for SG** — a *within-city tile*
holdout where `region=singapore` sits on both train and holdout sides, so a region-only key
would falsely flag every singapore train tile.

**Our split is whole-city** (no held-out city's tiles in train), so the `(region,tile)` key is
correct **only if the manifest enumerates every tile of each held-out city** — vacuous if the
manifest were city-level or missed a tile.

- **Fix: add a city-identity guard**, scoped to `holdout_kind: whole_city` regions (§2.1): trip
  if any training artifact's lineage touches a wholly-held-out *region*, independent of tile
  enumeration. Keep the `(region,tile)` key + fail-closed underneath. This decouples the guard
  from manifest completeness.
- **Teeth (red-before / green-after) — four cases:**
  1. **A:** inject a held-out-city tile that IS enumerated → trips (tile-key).
  2. **B:** inject a held-out-city tile NOT enumerated (or city-level manifest) → tile-key
     misses it, **city-guard must trip** (proves the city-guard is non-redundant).
  3. **C:** clean train manifest → passes.
  4. **D:** a *synthetic* within-city/partial (`tile_sample`, SG-style) manifest → the
     city-guard must **NOT** fire on that region's train-side tiles (proves correct scoping to
     wholly-held-out regions). **Labelled forward-protection regression** — v1 has no partial
     holdout, so D tests a config that doesn't exist in v1 data.
- **The teeth depend on T2 being locked first:** the guard reads the manifest's whole-city
  *declaration*; a faithful guard over a wrong declaration passes its own teeth and still
  leaks. T2's correct-by-construction verification (§2.2) is what makes the declaration
  trustworthy. So: lock T2 schema → teeth-prove the guard.

**Triggers 2 + 3:** trigger 2 is unified into the §7 power gate (not a separate mechanism).
Trigger 3 (model conditioning = same `_derive_tile_conditioning` source as `labels.py`) is
**satisfied** — the one-source identity is proven (`labels.py:68-79`).

---

## 7. First-model power gate (unified T3 / T4(c) / trigger-2)

There is **ONE** power/resolution gate, evaluated at the **first trained model**, per stratum:

- **Threshold side:** the multi-region resolved-gap recomputed by `resolution.py` on the
  **TRAIN split only** (§5).
- **Power side:** held-out **usable-n** per stratum (§2.1; munich's 156 is the floor).
- **Rule:** measure the **model-vs-real** coherence-gap effect size per stratum; (i) confirm
  usable-n resolves it for **architecture discrimination**, and (ii) if not, fire escalation —
  **munich→manchester / add-a-train-city**, *never* "extract a second region" (moot at 42
  cities). A munich→manchester swap is a **deliberate re-lock** of the write-once set (§2.3).
- **Why unified:** the old 0.076 was a resolution-adequacy floor; the volume escalation is moot
  at 670M but the question changed unit to "can held-out usable-n resolve the coherence gap?"
  — which **is** T4(c). One rule, no second mechanism that can disagree with it.
- **Pre-model status (2026-06-08):** the catastrophic, *pre-model-detectable* trigger is
  CLEARED — munich's usable-n is 156 (91%), healthy, in the band where the probe showed clean
  discrimination. The *definitive* check is inherently at first model (model-vs-real effect
  size cannot exist pre-model); munich→manchester stays in reserve.

---

## 8. v1 bake-off bars (all value-agnostic — they score the macro plan / micro tokens, not
conditioning values)

1. **Loss decreases** — exists.
2. **Macro→micro perplexity-gap** — exists (`perplexity_gap.py`), region-agnostic.
3. **Macro-plan coherence** (§3) — NET-NEW: zoning (same-class) + skeleton (S1 continuity +
   fragmentation), shuffle-gap, threshold relative-to-real, build-time gates §3.5.

Full §9.115 conditioning-compliance and value-bearing conditioning are **DEFERRED** with the
#13/#22 HARD GATE (a corpus re-derive); not a v1 bar.

---

## 9. Deferred items & HARD GATES (one place, with reason + revisit-trigger)

| item | reason | revisit-trigger |
|---|---|---|
| **#13 + #22** admin_region + morphology_class/country/climate SG leak | value-bearing conditioning would train on SG constants / all-None | ONE bundled corpus re-derive before any value-bearing conditioning; never fixed separately |
| Density coherence term | per-cell density conditioning → circular (§3.4) | v2 only if a macro stage conditioned on TILE-level density is pinned |
| Dense-core + sparse generalization | budget can't afford per-morphology coverage; no PRD dense-core-primary target (§1.2) | phase transition (project-wide principle, PI-ratified this thread — NOT a §10.1 clause): a new phase re-anchors the held-out set as a fresh artifact |
| Cross-tile seam coherence | rules-based stitcher, identical across architectures → never an architecture bar (§4.2) | one-time validation when sub-E tile-to-tile semantics land |
| Cross-region (§9.113 literal) generalization | within-regime keeps a miss unambiguous (§4.3) | separate, separately-reported probe |
| munich→manchester swap | held-out usable-n vs first-model effect size (§7) | first trained model; deliberate re-lock if 156 inadequate |
| #15 fallback-bbox over-include; #21 munich inner-core | frozen-corpus scoping facts | next corpus regen if true-extent / full munich needed |

---

## 10. Implementation outline (for the plan — NOT built here)

1. `paths.py` per-city CRS label; `resolution.py` train-split-only recompute.
2. Manifest builder (multi-region merge + `holdout_kind` + achieved-props + §2.2 build-time
   verification); freeze + `_EVAL_SET_LOCKED` + git force-add + verify-vs-disk.
3. Leak-guard city-key (§6) + the four-case teeth.
4. Coherence metric (§3) + the three-way teeth + threshold-after-measuring.
5. Wire the §7 first-model power gate into the eval harness (fail-loud).

---

## 11. Source ledger (every load-bearing claim → the file read this session)

- Corpus totals / per-city counts: `reports/2026-06-05-phase-2-g4-corpus-dod.yaml`,
  `reports/2026-06-08-phase-2-multiregion-corpus-closeout.md`.
- Train floor 550M + corpus-vs-train-floor note: `scripts/multiregion/build_g4_rollup.py:56,58`.
- macro_core schema / cell-vs-edge keying: `src/cfm/data/sub_d/io.py:41-79`,
  `evidence.py:12-19,201-245`; road_skeleton buckets:
  `configs/macro_plan/v1/macro_plan_vocab.yaml:3489-3505`; lattice/axis:
  `src/cfm/data/sub_d/lattice.py:1-119`; enums: `sub_d/enums.py`.
- Conditioning vector + granularity: `src/cfm/data/training/conditioning.py:28-37,54-63`,
  `build_shards.py:71-121`, `shard_schema.py:31`, `datamodule.py:53`;
  `morphology_stratum` definition: `src/cfm/eval/holdout/labels.py:34-36,46-48,50-59,68-122`.
- perplexity-gap behaviour: `src/cfm/eval/perplexity_gap.py:41-114`.
- leak audit: `src/cfm/eval/holdout/lineage_audit.py:51-73`; manifest writer:
  `src/cfm/eval/holdout/manifest.py:1-69`.
- SG-constant leak: `src/cfm/data/sub_c/conditioning.py:24-81`; coastal derivation `:56-72`.
- PRD: `PRD.md:23 (§1), 47/51 (§4), 95-97 (§8), 107-119 (§9)`.
- r=20 / 30M-param / 600M training-data ideal: `reports/2026-06-06-eval-set-gen-scoping.md`
  ("FLOOR INTERPRETATION"); density-coverage infeasibility + held-out token arithmetic computed
  this session from the G4 yaml.
- Protocol: `docs/protocols/sub-project-planning-protocol-v3.md` §9, §10.1 (write-once
  *sizing*, `:311-313`), §10.2. **NB:** the "phase-transition re-anchor" of the held-out set
  (§1.2/§9) is a **project-wide principle PI-ratified in this thread**, NOT a §10.1 clause —
  verified: grep finds no "phase-transition / re-anchor" phrasing in the protocol or repo.
- known_issues: `docs/known_issues.md` #13, #21, #22.
- Read-only measurements (2026-06-08): coherence discrimination probe (pooled d≈0.8);
  usable-n per held-out city (glasgow 523 / eisenhuttenstadt 579 / munich 156 / krakow 601);
  coastal_inland_river across 42 (171/18/244/7, zero nulls).
