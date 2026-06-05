# Batch-2 sub-F BP7 symmetry failures — root-cause investigation

**Date:** 2026-06-05 · **Branch:** `phase-2-multiregion-extract` @ `5225c7a` (the SHA the
corpus actually processed at — confirmed in every job log; the handoff's `e5fd0fe` was stale).
**Author:** investigation driven read-only; NO validator/data changed. **Status:** verdict for PI.

## TL;DR

Batch-2 fanout did **not** complete cleanly: of 35 batch-2 cities, **12 validated, 13 timed out in
sub-C, 9 failed, 1 (welwyn) never submitted.** Of the 9 failures, **8 share one deterministic sub-F
`_check_symmetry` (BP7 Leg 2) error**; the 9th (almere) is a different (sub-G) failure.

**Verdict on the 8: all 14 disagreeing edges are FALSE POSITIVES** — a road terminating at an
internal cell boundary. **The sub-C/D/E data is sound; the bug is in the sub-F validator's symmetry
leg.** Blast radius = sub-F validator layer only. **No re-extraction, no re-fetch, no data regen.**

## How the failure looks

Each of the 8 cities fails its whole validation on **1–2 tiles** (out of ~600), at:
`CrossTileValidationError: BP7 symmetry failure … cell A edge E emits ['MINOR_ROAD'] but paired
neighbour B edge W emits [] — shared-edge views disagree.` The validator raises on the first such
edge, so one edge kills the city.

## Why it is a false positive (mechanism)

1. An internal shared edge is stored **once** in sub-E's `boundary_contract.parquet`; both cells' views
   read the **identical row** (§3.4 byte-identity invariant; `boundary_contract.py` item 8). So the
   contract is **symmetric by construction** — it cannot be the source of an internal-edge asymmetry.
   Verified directly: the bruges failing edge has A.E = MINOR_ROAD **and** B.W = MINOR_ROAD.
2. The asymmetry is in **emission**. The sub-F encoder emits a `<bref>` only where a road LineString's
   **endpoint lies (float-exact) on an active edge** (`encoder._classify_feature_for_bref`).
3. When a road **terminates exactly on an internal cell boundary**, only the cell containing the road
   has an endpoint on that edge → it emits; the neighbour has no road there → it emits nothing.
   sub-C records the endpoint-on-edge as a crossing → sub-E marks the shared edge MINOR_ROAD (both
   sides) → **A emits, B doesn't.**
4. The **coverage leg (Leg 4) already encodes the correct, conditional model** ("we do NOT fire on
   active edges where neither this cell nor its neighbour carries a road feature"). The **symmetry leg
   (Leg 2) contradicts it** by demanding identical emission with no road-presence condition. Leg 2's
   own docstring premise ("a break that survives cross-reference is reachable only via an
   adjacency-mapping bug") is **false** in the presence of legitimate terminations.
5. Singapore (494 tiles) never exercised this regime → sub-F passed in Phase-1. EU cities do, sparsely.
   This is the `feedback_synthetic_fixture_blind_regime_at_validator` pattern (that memory is literally
   about `_check_symmetry` firing first on real data for an unfixtured regime).

## Evidence (read-only probe `scripts/multiregion/diagnostics/symmetry_probe.py`)

14 disagreeing edges across 8 cities, ~4,554 tiles. Road-only classifier (waterways/polygons excluded —
they neither emit brefs nor set the edge class per §5.1):

| City | tiles | disagreeing edges | classification |
|---|---|---|---|
| bruges | 180 | 1 | termination (dry) |
| manchester | 586 | 1 | termination (dry) |
| eisenhuttenstadt | 616 | 2 | termination into ≥80%-water neighbour |
| a_coruna | 695 | 2 | termination (dry) |
| lodz | 609 | 3 | termination (dry) |
| tychy | 616 | 1 | termination (dry) |
| mannheim | 636 | 1 | termination (dry) |
| krakow | 616 | 3 | termination (dry) |
| **total** | **~4554** | **14** | **14/14 false positive, 0 real defect** |

In every case the **emitting road** (track/tertiary/service/residential) has its endpoint exactly on the
edge and is **absent from the neighbour cell**. Two intermediate mislabels were found and corrected in
the probe: (a) a co-located **polygon** spanning both cells, (b) a co-located **waterway** (river/canal/
stream) — both initially looked like "road present both sides" but neither is a road witness.

### Passing-city spot-check (the "symmetric-but-wrong" blast-radius worry)

Audited every active internal edge against sub-C crossings on 4 cities:
`bologna 36,053 · edinburgh 21,829 · malmo 16,931 · munich(canary) 14,014` active edges →
**0 without a real road crossing.** No non-road feature is mislabeled MINOR_ROAD. **The §5.1
contract-mislabel mechanism is NOT occurring; blast radius is not the contract / not all 40 cities.**

## Third-authority source trace (the decisive FP-vs-drop check)

The "termination" classification above is necessary but NOT sufficient: an empty neighbour cell is
*also* the signature of a **sub-C clip-drop** (a road that truly crosses A→B whose B-fragment sub-C
dropped). Inferring termination from the *clipped* sub-C output assumes lossless clipping — the exact
thing sub-G T11 says not to assume. And the encoder fires only on a **float-exact** endpoint on the
250 m grid line, which is the fingerprint of a *clip*, biasing the suspicion toward a dropped crossing.

So each of the 14 edges was traced to the **unclipped Overture source** (`transportation.parquet`,
the third authority independent of the sub-C clip): reproject the source road + the bbox to the tile
CRS, compute `keep = road ∩ bbox` (replicating sub-C's bbox clip — known_issues #15 — independently),
and measure the road's length inside the MISS cell's 250 m square. A dropped crossing leaves source
geometry in the miss cell (`len_in_MISS > 0`); a true termination does not.

**Result — all 14 edges:** `kept == src_len` (none bbox-clipped at the edge), `len_in_EMIT` substantial
(machinery sanity-check passes — the road is in the emit cell), and **`len_in_MISS = 0.0 m` for every
edge.** The unclipped source road has zero geometry beyond the boundary. **No fragment was dropped;
mode B (sub-C clip-drop) is ruled out.** The exact-on-grid endpoint is sub-C snapping a near-boundary
*terminus*, not cutting a through-road. FP confirmed against source; data sound.

Probe mode: `symmetry_probe.py --city <c> --source-trace`.

## Blast radius & fix recommendation

**Data:** sound. No sub-C/D/E regeneration, no re-fetch, no data version bump.

**Fix (sub-F validator):** condition the symmetry leg on road-feature presence, mirroring the coverage
leg — require the neighbour to emit **only if it has a road feature with an endpoint on the shared
edge.** This removes the termination FP.

**The must-distinguish twin needs care — and it canNOT be a symmetry-leg fixture.** A real sub-C
clip-drop makes the neighbour cell look IDENTICAL to a termination (the dropped road is absent from B),
so the relaxed symmetry leg — which keys on "B has no road endpoint on the edge" — would PASS a drop.
The relaxation therefore creates a **drop blind spot** in the symmetry leg by construction. The twin
must live in a check that can see the road *should* be in both cells:

- **Drop-detector (new, the actual twin):** a lossless-clip invariant — for every internal-edge crossing
  recorded in sub-C `crossings.parquet` that implies the road occupies both adjacent cells, both cells
  must carry the road fragment. Fixture: source road crosses A→B + B fragment missing → FAIL. Caveat:
  the crossing record and the clip share sub-C code (correlated-failure risk per
  `feedback_independence_misses_shared_assumptions`), so this is necessary but not fully independent.
- **Independent corpus gate (strongest):** run `symmetry_probe.py --source-trace` (the third-authority
  source trace built here) across all 40 cities as a one-time gate. It does NOT share the clip's code,
  so a `len_in_MISS > 0` anywhere is a genuine drop. It found **0 drops** on the 8 failing cities.

TDD: (a) legit-termination → symmetry PASS; (b) road-present-both asymmetric emission → symmetry still
FAIL; (c) recorded-crossing-with-missing-fragment → drop-detector FAIL.

**Re-run cost:** the sub-F token output is already on disk and unchanged; only **re-validation** is
needed (sub-F validate runs in ~seconds–1 min per city). Recommend re-validating all 40 under the fixed
validator for version uniformity. Trivial compute. (A relaxation-only validator change does not
invalidate the already-passed cities.)

## Separate problems (do NOT conflate)

- **13 sub-C timeouts** (amsterdam, budapest, hamburg, helsinki, lisbon, lyon, madrid, paris, rome,
  rotterdam, valencia, vienna, warsaw): big-metro boxes don't finish sub-C inside boost's **8h wall**.
  A **box-sizing** problem, not a resubmit; longer wall may be unavailable on boost. Parked.
- **almere:** passed sub-F (0 symmetry disagreements); failed in **sub-G** `validate_phase1_region.py`,
  likely tied to its alpha-drop band (**42.7% of buildings dropped** — reclaimed-land, high water).
  Own bucket; needs a separate look.
- **welwyn:** already fetched (cache present), never submitted; parked, no processing.

## Spend (PI asked)

`saldo -b` shows +307 local-h but **lags badly**. Real batch-2 boost consumption from `sacct`
CPUTimeRAW = **~1,411 local-h** (TIMEOUT 850 / COMPLETED 313 / FAILED 273). The **850 local-h spent by
the 13 timeouts produced nothing.** Budget remains ample (~37,600 local-h after batch-2 posts); the
HALT rule is not triggered. Lesson: do not resubmit timeouts on boost with the same boxes/wall.

## Discipline notes

Nothing merged/pushed. Validator untouched. Two classifier bugs in the probe itself were caught and
corrected before drawing the verdict (polygon and waterway both masqueraded as "road present both
sides") — the unanimous-FP result only emerged after the road-witness filter was correct.
