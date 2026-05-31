# sub-G T11 defect cycle 2 — coverage halt: sub-E derives a road class from NON-ROAD crossings

**Date:** 2026-05-31 · **Branch:** `phase-1-sub-G-cross-artifact-validator` · **Status:** characterized; fix needs approval (points UPSTREAM to sealed sub-E + cache regen).

> Surfaced after the N/S encoder fix (defect cycle 1) cleared the symmetry halt. The
> re-derive then halted on the BP7 **coverage** leg — a genuinely different defect.
> Same meta-pattern as [[feedback_loud_false_positive_masks_quiet_defect]].

## TL;DR

The coverage halt is **not** a sub-F over-strictness bug. sub-F's coverage check is
**correct** — it faithfully catches that sub-E marked an internal edge `MINOR_ROAD`
when **no road crosses it**. Root cause is upstream in **sub-E**: it derives a road
boundary-class from **non-road** (building / water-`base`) crossings.

## Root cause: an IMPLEMENTATION bug — the pipeline VIOLATES locked spec §5.1

**Spec §5.1 (`…/2026-05-20-phase-1-sub-E-boundary-contracts-design.md:238-264`) is CORRECT
and explicit:** *"Only road-class features carry a `class_raw`; non-road crossings (water,
coastline, rail) have no `class_raw` and are **excluded from the boundary-class vote
entirely**."* And: *"edge with no road crossings (any class_raw) → **NONE**"* … *"a
water-only or rail-only edge is NONE, not MINOR_ROAD."* The default-bucket
(unmapped→MINOR_ROAD) applies **only to road crossings** with an unknown class; the spec
excludes non-road crossings BEFORE the default bucket. So per spec, every one of these 2016
edges must be **NONE**.

**The implementation does the opposite of what §5.1 says** (`pipeline.py::_derive_tile_rows`,
L304-326): it builds `features_by_id` from road features only (L305-306) — correct so far —
but then iterates **all** crossings and appends `features_by_id.get(c.source_feature_id)`
(L309-311), which is **`None`** for a non-road crossing. It does NOT drop non-road
crossings; it threads them in as `None`. Then `derive_boundary_class` (`derivation.py:84-85`)
maps `None` → **MINOR_ROAD** (the unknown-road-class default). So a water-only edge becomes
MINOR_ROAD — exactly what §5.1 says must not happen.

The bug is the **`None` overload**: `None` should mean only "road crossing, unknown class →
MINOR_ROAD," but the pipeline also produces `None` for "non-road crossing," which §5.1
requires be excluded entirely. The partial road-only filter at L305-306 looks like it
honors the spec, but it only filters the *class lookup*, not the *crossing list* — non-road
crossings still vote (as `None`).

**The L319-324 comment misattributes §5.1.** It says *"Pass all crossings (including None
entries)… Per spec §5.1 … None entries map to the MINOR_ROAD default bucket; filtering them
out would change semantics."* That is backwards — §5.1 says non-road crossings are excluded;
the comment cites the spec as authority for behavior the spec forbids. (A misattributed-
citation tell — `feedback_forced_precedent_tell`.) This is the THIRD time this session that
reading the actual upstream source flipped a conclusion drawn from downstream citations
(cf. the "geographic N=+y vocab" and "fix sub-E cell_to_edge_ids" retractions) — the
`derivation.py` docstring + the pipeline comment together implied §5.1 said "all crossings
count"; §5.1 itself says the opposite. `feedback_decompose_verification_debt_before_inferring`.

> **Correction (honesty note, 2nd on this report).** An earlier draft of THIS report (one
> turn ago) claimed the defect was IN locked spec §5.1 and that the fix required a "spec
> decision." That was wrong — I inferred §5.1's content from `derivation.py`'s function +
> the pipeline's comment without reading §5.1. Reading §5.1:238-264 shows the spec is
> correct and the implementation violates it. This is an implementation-compliance fix, not
> a spec change. Verify-before-lock: the read converted "spec defect" → "impl bug" — opposite
> conclusions, distinguished only by the actual read.

## Drill evidence (read-only, cross-checked; halt tile i10_j10 cell (7,4) edge S)

**READ 1 — feature_class of the 4 interval-crossing IDs on edge (7,4,1):** all
`feature_class=base`, `class_raw='water'`, `geomtype=Polygon`. **Zero road crossings on
the edge.** Yet `contract[(7,4)]["S"] == MINOR_ROAD`.

**READ 2 — geometry:** all four are Polygons whose body-chord lies ON the edge line
(≥2 vertices at the edge y) = **co-linear**, not transversal. (One piece of `428df05f`
sits in cell (7,3) on a different edge; the rest are co-linear on (7,4,1).)

**READ 3 — prevalence across 494 tiles:** **2016** coverage-failure edges over **259/494**
tiles.

## The split — TWO orthogonal classifications (do not conflate)

| Discriminator | Buckets |
|---|---|
| **feature_class of the crossings** (routing) | **building 1478 + base 468 + base&building 70 = 2016 ALL non-road; ROAD = 0** |
| **event_type of the crossings** (geometry shape) | interval-only 2000 + mixed(interval+point) 16 = 2016 |

Both classifications cover the same 2016 edges and **both are 100% non-road**. The "16
mixed" are non-road features (mostly buildings) that produce *both* interval and point
crossing events (e.g. a polygon edge running along the boundary *and* a corner touching
it) — **not** roads.

> **Correction (honesty note).** An earlier draft of this report fabricated a routing
> table claiming "16 edges WITH a road crossing (base,road 12 / road 4)." That was wrong:
> I conflated the event_type split (2000/16) with a road/non-road split and invented the
> road sub-counts. The cross-check by a **second, independent discriminator**
> (feature_class) contradicted it — three reads (`cov_fc`, `cov_fc2`, `sig.json`) all show
> **0 road crossings, 2016 non-road**. A split under discriminator A is not a split under
> discriminator B. Corrected above.

## Fix routing — single finding, all sub-E

| Subset | Count | Finding | Fix home |
|---|---|---|---|
| all coverage failures | 2016 (100%) | sub-E derives `MINOR_ROAD` from non-road co-linear crossings (`None`-overload bug) | **sub-E** (upstream) |

No sub-F or sub-C subset — the coverage check and the clipping are both behaving correctly.

## Recommended fix — IMPLEMENTATION compliance with §5.1 (needs approval; sealed sub-E + cache regen)

The spec is already correct, so this is a code fix bringing the pipeline into compliance —
**not** a spec change. **Exclude non-road crossings from the vote, per §5.1.**

**Code site:** `pipeline.py::_derive_tile_rows` L308-311. Build a road-source-id set (e.g.
`road_ids = {f.source_feature_id for f in features if f.feature_class == _road_class_code}`)
and append a crossing's class only when `c.source_feature_id in road_ids`; skip non-road
crossings entirely (do not append `None` for them). The existing road-only `features_by_id`
then becomes a true filter instead of a `None`-generator. Also **fix the L319-324 comment**,
which currently cites §5.1 for the opposite of what §5.1 says.

Result:
- edge with only non-road crossings → empty list → `derive_boundary_class([]) == NONE`
  (spec-required) → non-emitting edge → sub-F emits nothing → coverage skips it. ✓
- road crossing, unknown/null class → `[None]` → MINOR_ROAD (the intended default, preserved). ✓
- road crossing, known class → that class. ✓

**Active-edge sub-question — RESOLVED (no change needed):** active-ness is `scope_marker`
-driven from sub-D's macro_core (`pipeline.py:288-294, 317`), independent of crossings. The
fix changes only the derived *class* (MINOR_ROAD→NONE) for non-road-only active edges; those
edges stay active with class NONE, which is valid — non-null `boundary_class_enum`=NONE
satisfies the sentinel invariant (non-null iff scope_marker==0), the encoder emits no bref
for NONE, and coverage skips NONE edges.

**Version bump:** spec §5.1:299 + §9/§15 list "non-road crossing handling changes" as a
`boundary_derivation_version` trigger, and the on-disk `boundary_class_enum` labels change
(some MINOR_ROAD→NONE). So bump `boundary_derivation_version` even though this is a bug fix —
the version axis exists precisely to flag "same input → different output labels."

**Lock-and-guards:** add/repair a sub-E test that a non-road-only active edge derives NONE
(an external-source-of-truth test vs §5.1), in the same commit as the fix — this is the test
whose absence let the violation ship. The sub-C-precedent collision aside, this is the same
class as the symmetry/coverage gaps: a check/spec-rule with no test exercising the
failing regime.

**Why verify-before-lock mattered here:** read in isolation, `derivation.py` + the pipeline
comment implied "all crossings count," which would have scoped this as a spec change. Reading
§5.1 itself showed the spec already mandates exclusion — so it's a compliance fix, smaller
and not requiring a spec-semantics decision. Opposite conclusions, distinguished only by the
actual read (`feedback_decompose_verification_debt_before_inferring`).

### Blast radius (larger than the N/S fix)
- **Changes sub-E `boundary_contract.parquet` content** (non-road-only active edges flip
  `MINOR_ROAD`→`NONE`) → **sub-E cache regen required** (the N/S fix left sub-E untouched).
- Then sub-F re-derive; then sub-G validate.
- **Affects sub-G seam-2 semantics** (contract↔tokens bijection reads the corrected contract)
  and sub-E's own `boundary_derivation_version` (likely a version bump per spec §5.1).
- sub-E is merged/sealed → a real sub-E revisit, not a sub-F patch.

## Relationship to the N/S fix (defect cycle 1)
Independent and complementary. The N/S encoder fix (working tree, green, uncommitted)
cleared 2,728 symmetry false-positives and is correct on its own. This sub-E fix is a
separate change with a separate (larger) blast radius. Recommend committing the N/S fix
on its own first (clean diff), then scoping the sub-E fix separately.

## Reproduce
`uv run python /tmp/cov_drill.py` (3 reads) + feature_class cross-check (`/tmp/sig.json`:
`{"building":1478,"base":468,"base,building":70}`, total 2016, road 0). To be promoted to a
committed `scripts/sub_g/` drill if the sub-E fix is scoped.
