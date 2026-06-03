# Phase-2 multi-region foundation — CRS parameterization + Berlin pilot (close-out)

**Date:** 2026-06-03
**Branch:** `phase-2-multiregion-crs-utm` (tip `2d5329a`) — 7 commits ahead of `main`, `main` not ahead → clean fast-forward.
**Status:** COMPLETE. De-risking done, foundation green, ready to merge to `main`.

**Scope of this summary.** This closes the **CRS-parameterization + feasibility-pilot** phase that de-risked
multi-region (European) ingestion. It is the **merge-safety record** for landing this branch on `main`. The
**bounded-extract orchestrator** that consumes this foundation (Option 1, decided 2026-06-03) is a *separate
following sub-project* with its own `reports/` summary — deliberately not bundled into this merge, to keep the
proven foundation isolated from the churny new build.

---

## What this delivered

Parameterized the single-region (Singapore SVY21 / `EPSG:3414`) pipeline to **per-city conformal UTM** (one
zone chosen once from the city centroid, EPSG-prefixed tile labels), and proved on real European data
(Berlin `EPSG:25833`) that the full pipeline chain **and** the locked Phase-1 tokenizer transfer with **zero
code changes**.

The load-bearing design fact: the geometry token layer is **CRS-agnostic by construction** (cell-local anchors
+ turn/distance deltas), so the expensive locked artifact — the tokenizer and the embedding head — transfers
for free. Only the *plumbing that gets data into cells* (projection, tile labels) was Singapore-shaped, and it
was a contained, mechanical parameterization the spec already anticipated ("polymorphic per region").

### Commits (7, off `main`)

| SHA | Commit |
|---|---|
| `04de763` | feat(coords): multi-region CRS helpers (centroid→UTM zone, CRS→tile label) |
| `8ef61eb` | feat(overture): required `projected_crs` region field (+ Berlin pilot config) |
| `50ff547` | refactor(coords): region-bound `RegionCoords` transformer (one shared authority, replacing the module singleton) |
| `2a7f140` | feat(sub_c,sub_e): thread region `projected_crs` + tile label end-to-end |
| `16355c2` | perf(overture): skip pre-fetch `COUNT(*)` size estimate when `confirm=True` |
| `c1ae046` | fix(sub_c): derive cross-tile validator tile label from `region_crs` (pilot bug, see below) |
| `c6253ec` | expt(multiregion): Berlin pilot scripts + feasibility/findings record |

---

## Merge-safety evidence — Singapore byte-identity (THE gate)

The CRS refactor must not change Singapore's output by a single byte. **Verified 2026-06-03 on local real
Singapore data at branch tip `2d5329a`** (not asserted from prior handoffs):

- **Fast suite:** `1043 passed, 1 xfailed, 0 failed` (31.8s).
- **Singapore integration / byte-identity (slow, real data):** `11 passed, 0 failed` (12m15s). Covers:
  - sub_c re-extract → **byte-identical parquet** (spec §14.5 determinism contract);
  - sub_c cross-tile validator pass (manifest↔filesystem, schema-version, provenance-SHA chain, outputs-SHA);
  - sub_f end-to-end derive + same-process `cells.parquet` byte-identity.
- Singapore remains `EPSG:3414`, **494 tiles**, `_PHASE1_VALIDATED` lineage unchanged.

**Conclusion: zero risk to existing `main` / Phase-1 artifacts. Singapore is byte-identical post-refactor.**

---

## Berlin pilot — feasibility measured on real EU data

Run on Leonardo (`EPSG:25833`, ETRS89/UTM33N from the Berlin centroid). Resolved every dominant unknown:

| Measurement | Value |
|---|---|
| tiles/city | **465** (≈ Singapore 494) → **~44 cities** for a 30M ladder; ~148 for 100M |
| fetch cost | **13.3 min/city** (with the `COUNT(*)` optimization), NOT the ~8h Singapore cold-fetch |
| CRS code on real EU data | reprojection, large-UTM tile indices (i≈184–208, j≈2900–2918), `EPSG25833` labels — all correct |
| tokenizer chain sub_d→sub_g | **PASSES, zero code changes**; sub_g cross-artifact validator green → locked macro-vocab + sub_f vocab transfer to EU |
| vocab unknown-rate | **Berlin 41.64% < Singapore 54.77%** (better) → minimal-floor regime, NOT EU under-coverage → ship vocab as-is, no re-derivation |

**Interpretation.** Berlin's *lower* unknown-rate than Singapore means the unknown-tag floor is a property of
the minimal locked vocab, not of European under-coverage. The vocabulary ships as-is for the bounded extract;
full EU frequency re-derivation is a Phase-4 concern, not a blocker.

---

## Two fixes found + closed (the pilot's job)

1. **Orchestration.** Full-city sub_c OOM-kills on the Leonardo login node → must run as a 120G/32-cpu **Slurm
   job** (`scripts/berlin_extract.sbatch`). Fetch is fine on the login node (cache-hit, no egress); the
   CPU/memory-heavy *processing* needs a compute node. (Note for the orchestrator: process on a **CPU
   partition**, never `boost_usr_prod` — that bills a 4×A100 node for CPU-only work and would eat the training
   budget.)
2. **Real code bug.** `sub_c/validator_cross_tile.py` hardcoded `tile=EPSG3414_*` in all four checks (missed
   during the threading; the all-Singapore suite was **sample-regime-blind** — every fixture used the one CRS
   the bug agreed with). Fixed to derive the EPSG label from `manifest.region_crs` (`c1ae046`). **Guard:** the
   regression test exercises a *non-Singapore* CRS, so the validator cannot silently re-pass if the hardcode
   returns. (This is the regime-blindness lesson the orchestrator sub-project inherits for the sub_f
   `region_crs` gap.)

---

## Egress (gate #1)

Leonardo login/serial nodes **have S3 egress** (Overture `http 200`; GitHub reachable too — the earlier
"can't fetch GitHub" note was stale). `dcgp_usr_prod` compute-node egress is **untested** — it only matters
past ~148 cities, so it is deferred until/unless the extract grows toward the 100M ceiling.

---

## Carried forward (NOT in this merge)

- **Bounded-extract orchestrator + ~44-city extract** — Option 1, the next sub-project (own branch off `main`).
- **sub_f manifest lacks `region_crs`** — non-blocking (Berlin's sub_g passed), but to be **fixed in-scope** in
  the orchestrator sub-project *with a non-Singapore-CRS test*, per the regime-blindness lesson above.
- **geometry-r / compute-optimal r measurement** — training-measured, belongs to the bake-off, not here.

## References

- `docs/handoffs/2026-06-02-multiregion-feasibility-audit.md` — feasibility audit verdict + bounded-slice re-cost.
- `docs/handoffs/2026-06-03-multiregion-pilot-resolved-decision-pending.md` — pilot resolution + Option-1/3 decision tee-up.
- memory `project_multiregion_feasibility_audit`.
