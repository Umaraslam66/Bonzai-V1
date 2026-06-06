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

### TOKEN-COST TENSION — surfacing, not papering over

- Shipped corpus ≈ **551M** projected (397.7M validated + ~153M from the 7 add-cities, pending).
- Holding out these 4 = **−34.7M** → **training ≈ 516M, UNDER the 550M floor.**
- This is acute because the corpus sits *right at* 550M — there is **no headroom for a held-out
  eval set without dropping train under the floor.** Held-out-CITIES and the 550M training floor
  are in direct competition at the current corpus size.
- **Resolutions (PI's call):** (a) **add ~3–4 more cities** (corpus → ~585M, so train ≥ 550M after
  holdout) — the spec's "add cities, cheap" lever, the cleanest; (b) clarify whether 550M is a
  *training-token* floor (then this binds) vs a *corpus* floor (then it's already met and 516M-train
  is fine); (c) hold out fewer/smaller cities (e.g. drop munich → −24.6M, train ~526M — still under);
  (d) accept ~516M train (the floor is a heuristic, 516M is ~94% of it). **I recommend (a)** — it
  preserves both a real 4-morphology held-out eval AND the training floor; cost is a few more
  evening-launch extractions. **Final add-city yields land tomorrow midday — the exact gap firms then.**

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
- **`resolution.py`** — mostly mechanical, BUT one check: the gap-floor numbers (single-region
  0.049, second-region trigger 0.076 from the frozen-set memo) were Singapore-derived; confirm
  the multi-region held-out gap is computed against THIS corpus, not the carried single-region
  floor. (Tiny, but flag it so it isn't a silent single-region assumption.)
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
