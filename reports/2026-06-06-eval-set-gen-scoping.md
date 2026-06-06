# Eval-set-generation scoping (read-only; NOT built) — 2026-06-06

Scoped from the actual eval artifacts (`src/cfm/eval/holdout/*`, `slice_metrics.py`,
`perplexity_gap.py`, `paths.py`, the frozen `holdout_manifest.yaml`), not the workflow doc.
Eval machinery EXISTS but is Singapore-shaped; this is a GENERALIZE job, not a rebuild.
Three deliverables for PI sign-off before any code.

---

## 1. Proposed held-out CITY split (4 cities; distinct morphology + CRS; each held-out type also in train)

Selection rule honored: every held-out morphology is ALSO represented in train, so a held-out
miss reads as "didn't generalize," never "never saw this type." Distinct UTM zones so the eval
exercises the multi-CRS path (the thing Singapore-only never tested). Tokens are the measured
G4 per-city counts.

| | city | morphology | density | CRS (UTM zone) | geography | tokens | also-in-train (same morphology) |
|---|---|---|---|---|---|---|---|
| **HELD-OUT** | tallinn | medieval-organic | moderate | 25835 (z35) | EE | 5.6M | prague, bologna, ljubljana, krakow, edinburgh, toledo, bruges |
| **HELD-OUT** | glasgow | planned-grid | moderate | 25830 (z30) | GB | 14.7M | turin, valencia, helsinki, barcelona, mannheim, karlsruhe, +linz/debrecen* |
| **HELD-OUT** | eisenhuttenstadt | modernist-sprawl | moderate | 25833 (z33) | DE | 4.3M | cergy, tychy, milton_keynes, espoo, +eindhoven/tilburg/wolfsburg/telford* |
| **HELD-OUT** | munich | mixed | moderate | 25832 (z32) | DE | 10.1M | budapest, hamburg, copenhagen, lisbon, manchester, malmo, +szczecin* |

(*add-cities, validation pending — final G4 tomorrow midday.) Held-out zones 35/30/33/32 are
all distinct; geography EE/GB/DE/DE (CRS distinct; DE repeats — acceptable, or swap munich→
**lisbon** PT/z29/21.5M for 4 distinct geographies at +11M token cost).

### FLOOR INTERPRETATION — RESOLVED: train-budget, not corpus-total (don't repeat the 56k conflation)

The spec frames the floor as **CORPUS** — "**Corpus** token-count confirmed ≥ 30M-ceiling need at
r=20" (§25), "assert `total_validated_tokens ≥ 600_000_000` directly" (§233). But the *derivation*
(30M params × r=20 tokens/param) is a **TRAINING-data budget** — tokens the model SEES. The spec
says "corpus" only because it **predates the held-out-CITY decision**: at spec-time the held-out
eval was Singapore-TILES (a separate sub-project), so the multi-region corpus *was* the train set
(corpus = train). The held-out-CITY decision splits them. So two distinct gates, which I had
conflated:

- **Corpus-completion DoD (the MERGE gate, tomorrow):** corpus total. The 4 held-out cities are
  validated corpus members → they **count toward it** → ~551M clears the PI-accepted 550M.
  **Holding out cities does NOT breach the merge gate.** (Coded caveat: `build_g4_rollup.TARGET_TOKENS`
  still literally checks **600M**, so the gate-(a) line prints `False` at 551M; the PI-accepted v1
  floor is **550M + full coverage**, so the merge call is that judgment, not the literal-600M line.
  PI's call: lower `TARGET_TOKENS`→550M to match the ratified floor, or treat gate-(a) as advisory.)
- **Training-data floor (the r=20 INTENT):** the model trains on corpus − held-out ≈ **516M**, below
  the 550M training budget → **add ~3–4 cities so corpus ≥ ~585M → train ≥ 550M after the ~35M
  holdout.** Right reason (train budget), right phase (eval-set-gen / training, post-merge).

**Correction:** my earlier "train dipped under 550M *total*" conflated train and eval (the same
misapplied-number class as the 56k tok/tile estimate). Correct version: **the corpus clears the
merge DoD (held-out cities count toward corpus total); the TRAIN split needs the r=20 floor → add
cities for the training phase.** Add-cities is the cheap lever; firm gap lands tomorrow midday with
the add-city yields. Hold-out city set confirmed CLEAN (all 4 VALIDATED, groups=0, none in the
inflation-excluded/degraded set).

---

## 2. Macro-plan-coherence metric DESIGN (the net-new product bar)

**What it must measure:** is the generated cell-level macro plan (sub-D attributes per cell:
`road_skeleton_class`, `zoning_class`, `density_bucket`) *spatially coherent* on a held-out tile —
i.e. neighbouring cells form realistic structure (districts cluster, density transitions smoothly,
skeletons connect across cell seams) rather than per-cell noise OR uniform mush.

**Proposed metric — a shuffle-gated coherence GAP (mirrors `perplexity_gap`'s NLL_matched−NLL_shuffled):**

Per held-out tile, over the 4-neighbour cell lattice, compute a **neighbour-agreement** score per
macro attribute:
- categorical (zoning, road-skeleton): fraction of adjacent cell-pairs with compatible class
  (same class, or an allowed transition);
- ordinal (density bucket): 1 − mean |Δbucket| / max, i.e. smooth-transition fraction (≈ a discrete
  Moran's-I).

The reported bar is the **coherence gap**: `score(generated, true cell arrangement) −
score(generated, SPATIALLY-SHUFFLED cells)`. A coherent plan must score **strictly above** its own
shuffle (proves the metric responds to spatial ARRANGEMENT, not just the marginal class mix).
Reference target: the held-out GROUND-TRUTH tile's own gap — the model should approach it, not
exceed it (over-smoothing) or fall short (noise).

**Anti-vacuous / teeth (same discipline as the validator relaxes + slice's `n_polygons` guard):**
1. **Shuffle teeth (load-bearing):** real held-out tile's score MUST exceed its spatially-shuffled
   version with margin. If shuffling doesn't drop the score, the metric is measuring the marginal
   distribution, not coherence → vacuous → reject. (Red-before/green-after: a shuffled plan FAILS.)
2. **Two-sided, to block the trivial pass:** a UNIFORM plan (all cells one class) has perfect
   neighbour-agreement but is unreal — so the metric ALSO requires the generated plan's attribute
   entropy/diversity to fall within a band of the real tile's. Uniform fails the diversity side;
   noise fails the agreement side; only real-like-structured passes both. (This is the
   "coherence score every output passes proves nothing" guard you flagged.)
3. **Real-vs-random separation:** real tiles score high, class-permuted tiles score low, with a
   margin — verified on a held-out sample before the metric is ever a gate.

**Scope note:** boundary-contract stitching (skeleton connectivity across cell seams) is the
sub-E/seam piece of "coherence"; v1 can score the in-tile neighbour-agreement + the seam-agreement
on held-out tiles. Threshold/gate is set AFTER the shuffle-teeth + real-vs-random separation are
measured — never a guessed absolute number.

---

## 3. Mechanical generalizations (bundled cheap work — scope confirmed, no design)

- **`paths.py`** — hardcodes `DEFAULT_REGION="singapore"` + `_EPSG_LABEL="EPSG3414"` and builds
  tile dirs as `tile=EPSG3414_i{i}_j{j}`. Generalize to **per-city CRS-label resolution** (each
  region config carries `projected_crs` → `EPSG258xx` → `EPSG258xx` dir label). Pure plumbing.
- **`labels.py`** — the Singapore `morphology_class` CONSTANT + "no real Singapore variation"
  assumption. Multi-region HAS variation (more signal, not less); confirm the conditioning-vector
  extraction reads per-tile sub-C/sub-D values (it does) and drop the single-region constant.
- **`resolution.py`** — mostly mechanical, BUT one care-point: the gap-floor numbers (single-region
  0.049, second-region trigger 0.076 from the frozen-set memo) were Singapore-derived. Recompute on
  the **TRAIN split ONLY** — never include the held-out cities in the floor's own calibration, or
  eval data leaks into the threshold it's later judged against (the calibration must not see the
  test set). Not just "vs this corpus" — specifically the train subset.
- **Reusable AS-IS (no work):** `lineage_audit.py` (region-keyed, one code path — a 2-region
  manifest already exercises identical logic), the manifest schema (regions-keyed), and
  `perplexity_gap.py` (macro→micro, region-agnostic).

---

## Net

Eval-set-gen = (1) generate the held-out-CITY set over the 36-city corpus + resolve the token-floor
tension (likely add a few cities), (2) build + teeth-prove the net-new coherence metric, (3) the
three mechanical de-Singapore fixes. The split-audit, manifest schema, and macro→micro perplexity
gap are already multi-region-ready. Loss-decreases ✓ and macro→micro ✓ exist; coherence is the one
genuinely-new bar. **Held for PI sign-off; nothing built; corpus completion remains the live gate
(final G4 ~tomorrow midday CEST).**
