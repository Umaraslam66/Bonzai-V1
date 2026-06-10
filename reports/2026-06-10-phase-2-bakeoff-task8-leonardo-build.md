# Phase-2 Bake-off — Task 8: EU multi-region train-shard build (Leonardo, CPU)

**Date:** 2026-06-10
**Branch / build commit:** `phase-2-bakeoff` @ `ed1138c` (local-only; `main` untouched)
**Where:** Leonardo `/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, job `45436823` on `lrd_all_serial`
**Data snapshot:** release `2026-04-15.0`, G4 roll-up `reports/2026-06-05-phase-2-g4-corpus-dod.yaml`, multiregion holdout manifest `data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml`
**Status:** DONE, independently verified from disk. Phase D remains parked at the Task-9 gate (Umar's word).

## What was done

Built the per-city training manifests for the **38 EU train cities** (the G4 roll-up's 42 validated cities minus the 4 held-out: eisenhuttenstadt, glasgow, krakow, munich). This is spec §5's build-shape (a): a driver loop over `train_cities(...)` writing a per-city `training_manifest.yaml`, then re-reading every manifest to verify end-state. CPU-only, `lrd_all_serial` (budget-free) — **did not touch the 36k core-h GPU allocation** (`saldo` consumed unchanged at 3,960 / 9.9%).

New committed artifacts: `scripts/build_multiregion_train_shards.py` (testable core + CLI), `scripts/build_multiregion_train_shards.sbatch`, and the I1-safe writer `build_train_city_manifest` in `build_shards.py`.

## Two Singapore-only bugs found-and-fixed (small-before-big)

Timing the smallest cities (umea, prague) before the full 38 surfaced two real bugs that **every synthetic test had passed** — both the same class as the Phase A–C design errors: code paths never exercised because everything upstream was Singapore-only.

1. **I1-boundary in the writer (`84edb3b`).** The runner called `build_training_shards` per city, which routes through `_holdout_ids` → `ValueError` for any EU train city (a train city is neither `singapore` nor a held-out city). Spec §5's wording "reuse `build_training_shards` per train city" was itself wrong for the I1 design. Fix: `build_train_city_manifest` (I1-safe writing sibling — all-validated-tiles + write); `_holdout_ids` left byte-identical. Regression test incl. red-on-divergence (single-region writer raises; I1-safe writer does not).

2. **CRS-default in the dir-path layer (`ed1138c`).** `build_shards_in_memory` built per-tile dir names with `tile_dirname(ti,tj)`, which **defaults `epsg_label` to Singapore `EPSG3414`** → EU tiles resolved to a non-existent dir (`FileNotFoundError`). The CRS-agnostic proof from Phase C covered the *token* surface, not the *directory-path-construction* surface. Fix: derive `epsg_label_for_region(region)` per region (helper already existed, just unwired). Verified **three-way CRS consistency** (region-config `projected_crs` == sub-D manifest `region_crs` == on-disk dir label) across all 38 train + 4 held-out cities before fixing. Regression test red-on-divergence.

## Verified end-state (re-read from disk; runner RC NOT trusted)

| Check | Result |
|---|---|
| Train-city manifests present | **38 / 38** |
| `sum(n_training_tiles)` (recomputed from disk) | **22,019** (== expected) |
| Per-city cross-check (`n` == sub-D validated tiles) | pass, all 38 |
| Held-out manifests on disk (must be none) | **none** (the 4 absent) |
| Non-train training dirs | only `singapore` (scaffold leftover; harmless) |
| GPU budget consumed | **unchanged** (3,960 / 9.9%) — build was budget-free |

Per-city tile counts span 36 (umea) to 1,078 (budapest); full breakdown in `reports/2026-06-10-task8-multiregion-train-shards-build.yaml`.

## Tested vs not

- **Tested:** runner build/exclude/verify logic + I1-safe writer + multi-region CRS label — 4 new teeth (red-on-divergence on each), full training test dirs green (78 passed, 1 pre-existing Task-12 xfail). Real-data build verified on Leonardo.
- **Not covered by Task 8 (carried to Task 9):** the eval-side read path.

## Carry-forward → Task 9 PRECONDITION (step 0), recorded in the plan (`bc9e1de`)

The **same CRS-default bug exists on the eval read path** (`geometry.py:89`, `holdout/pipeline.py:189` call `tile_dirname` SG-defaulted). Reclassified (Umar, 2026-06-10) from generic carry-forward to **Task-9 step 0**: gate input (i) is "a DATA property of REAL held-out tiles," read through `geometry.py` — so the diagnostic `FileNotFound`s on the first EU held-out read. Task 9 must fix the eval-side `tile_dirname` (same `epsg_label_for_region` wiring + three-way consistency on the 4 held-out cities + red-before/green-after on a real held-out read) **before measuring any gate input**.

## Next

HOLD for Umar's Task-9 word. No scored runs, no Task-10 fork resolution, no merge (Umar's word + `--no-ff`). When Task 9 starts, step 0 = the eval-side `tile_dirname` fix above, then the diagnostic.
