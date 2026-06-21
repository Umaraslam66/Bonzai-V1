# Lane-S held-out CELL SAMPLER — design spec

**Date:** 2026-06-21 · **Branch:** `phase-2-cell-eos` · **Status:** DESIGN COMPLETE — all gates locked, **awaiting PI review**. NO build, no `writing-plans`, no scored run until approved.

## Decision summary (one line each)
- **Architecture (Gate 1):** per-`(city, 4-tuple)` cell allocator; target = 146 distinct floored 4-tuples (union of the 265 rows); obligation per `(city, metric, stratum)` ≥50 *generated* features; unfloored out of scope. Invariant: target = real-floored, obligation = generated.
- **Pool (Gate 2):** 1,952 manifest tiles (= floor + real reference lineage), not 1,859 usable.
- **Bridge (Gate 3):** `n_cells = ceil(target / real_fpc[binding] × headroom)`; scarce metric (building_area) binds; real density measured from held-out, gen/real ratio the sole proxy.
- **Selection (Gate 4):** blake2b hash-rank over sorted cell keys; take-all-when-capped → ceiling-bound flag; one seed, one manifest for all 6 runs.
- **Feasibility (Gate 5):** §9 split — ceiling-bound short = exclude-and-report + #21 demotion; not-ceiling-bound short = fail loud.
- **N/cost (§6):** `target_features` (default 50) + `headroom` are config knobs; §10.1 — lock records achieved property, never binds provisional N; ~7% grant (transformer) pending `real_fpc` + mamba rate.
- **Contract (§7):** sha-locked write-once `_LANE_S_SAMPLER_LOCKED` manifest + committed regen script; feeds `gen_realism.gen_features_by_city`.

## Consult order
1. `docs/GROUND_TRUTH.md` (canonical — §3 held-out units, §4 bake-off design). If anything here disagrees, GROUND_TRUTH wins.
2. The prior read-only investigation (this branch's first session) + the conditioning floor (`src/cfm/eval/conditioning_floor.py`, locked artifact sha `95abb88`).
3. `docs/protocols/sub-project-planning-protocol-v3.md` (esp. §10.1 write-once sizing, §10.2 relative thresholds, §10.3 power in the correct unit).

## What this is
A **budget-bounded, stratified DOWN-sampler** over the held-out cell pool. It is **not** a selector-from-scratch and **not** a resume of the 2026-06-08 eval-set-gen (per-tile coherence) plan. It feeds the **Lane-S / conditioning-floor** lane: it picks which held-out cells the bake-off backbones generate, so that the generated-side feature distributions clear the floor's `min_n=50` per scored stratum and `decide()` produces a real verdict (winner or `NO_DECISIVE_WINNER`), not an under-sampling artifact.

Output is consumed by `gen_realism.gen_features_by_city(cells: list[DecodedCell])` → `lane_s_excess` → `decide()`.

## Locked facts the design rests on (from source, not up for debate)
- **Sampler picks whole CELLS; the floor scores FEATURES.** A generated cell inherits exactly one 4-tuple `(zoning, road_skeleton, cell_density, coastal)` and dumps *all* its features into that stratum, split across two metrics (`gen_realism.py:79-85`). So the sampler is a per-`(city, 4-tuple)` **cell-count allocator**; the obligation it satisfies is per-`(city, metric, stratum)`.
- **`min_n = 50`, on FEATURES, exclude-and-report, never merge/coarsen** — the floor's locked policy (`conditioning_floor.py:178`; artifact `methodology.min_n: 50`). The sampler MIRRORS this unit (PI-confirmed starting position; no source-level reason it breaks).
- **`decide()` scores only floored strata** and excludes the rest (`bakeoff_decision.py:416-422`); the S13/`NO_DECISIVE_WINNER` resolvability floor is `1.358/√n_reference_features` over REAL features on floored strata (`feature_resolution.py:53-62`, `city_aggregate.py:119-126`). The entire statistical stack is in the FEATURE unit; cells are upstream emitters.
- **Held-out cities + counts:** glasgow / eisenhüttenstadt / krakow / munich. Floored rows = **265** `(city, metric, stratum)` = 119 `building_area_m2` + 146 `road_length_m`. Real cell census = **94,520** distinct conditionable cells over the **1,952** manifest held-out tiles (GROUND_TRUTH §3, job `47473524`).

---

## Gate 1 — Architecture & target set  ✅ LOCKED (2026-06-21)

**Decisions:**
- **Allocation unit:** cells per `(city, 4-tuple stratum)`. A cell is not metric-specific, so allocation cannot be per-metric.
- **Target set:** the distinct `(city, 4-tuple)` strata carrying ≥1 floored metric row — the union of the 265 floored rows = **146 distinct `(city, 4-tuple)` strata**. Unfloored strata are **out of scope** (no speculative re-floor headroom — generate only what `decide()` reads).
- **Obligation:** per floored `(city, metric, stratum)`, **≥50 generated features**.

**Owed-metrics list (recorded now — Gate 3 multiplies against it; cheap now, costly to reconstruct):**

| city | both (B+R) | road_only | total 4-tuples |
|---|---|---|---|
| eisenhüttenstadt | 24 | 8 | 32 |
| glasgow | 36 | 6 | 42 |
| krakow | 32 | 6 | 38 |
| munich | 27 | 7 | 34 |
| **global** | **119** | **27** | **146** |

`building_only = 0` everywhere: **building_area ⊆ road_length** at the stratum level. road_length is owed in all 146 targeted strata; building_area in 119 of them. Where building_area is owed it is the scarce binding metric (Gate 3/M2); the 27 road-only strata are bound by the plentiful metric.

**INVARIANT — real-vs-generated asymmetry (stated now, not first at Gate 4):**
The target set is defined by **REAL-floored** strata (real side ≥50 features by construction). The obligation is on the **GENERATED** side. A real-floored stratum can still under-fill on generation (down-sampling choice × stochastic generation). **target = real-floored; obligation = generated ≥50; the gap between them is exactly what Gate 5 (M4) headroom + a consumer-side fail-loud power check must cover.** No silent reliance on real-side sufficiency.

---

## Gate 2 — Q1: Input pool  ✅ LOCKED (2026-06-21)

**Decision: the input pool is the 1,952 manifest held-out tiles** (conditionable cells = 94,520), **not** the 1,859 "usable" set.

**Reason.** The floor (the bar Lane-S subtracts) and the real reference (the `KS(gen, real)` comparison target) are *both* built on the 1,952 manifest `tiles[]`; the 29 MB `real-features` parquet reproduces the floor exact (265/265 vs sha `95abb88`) on it. Drawing the gen pool from the same set keeps `KS(gen_D, real_D)` a single-population comparison differing only by generated-vs-real — and honors the Gate 1 invariant (real-floored = generated-from, same population).

**Why not 1,859.** "usable" = `≥3 interior road edges, after water filter` (`2026-06-08-eval-set-gen-design.md:114,273`) — the **coherence lane's** power unit (T4 gate(b)), defined for a *different* eval. `n_unreadable = 0` for all four cities, so the 93-tile gap is purpose-filtered low-road/water tiles, not broken tiles. Importing it would (a) make the gen pool a strict subset of the floor's real pool → a real(1,952)-vs-gen(1,859) population mismatch inside the KS, and (b) omit cells the floor *did* score.

**Reproducibility note.** The cell-level conditionable/non-empty filter is already baked into the 94,520 count (missing-cells = 0). The pool is fully determined by the locked holdout manifest — no new tile filter is introduced by this spec.

## Gate 3 — Q2 + M2: cells→features bridge (the crux)  ✅ LOCKED (2026-06-21)

**Sizing formula.** `n_cells(stratum) = ceil(target_features / real_fpc[binding_metric, stratum]) × headroom` (headroom → Gate 5). Derived from the only codebase conversion, `feature_resolution.cells_to_resolve:116-127` (`n_cells = ceil(n_features / features_per_cell)`), specialized to the floor's fixed `min_n` target instead of a KS gap.

**N lives in the obligation unit, not in cells.** The knob is **`target_features` per floored `(metric, stratum)`** (default = `min_n` = 50; §6 keeps it a parameter). Cells fall out of the bridge — a future `min_n` change re-derives cells automatically (protocol §10.3: stay in the unit the verdict reads).

**Per-metric, per-stratum — no global scalar (M2).** Measured rates differ radically (`reports/2026-06-11-token-length-investigation.yaml`, normal cells): `buildings/cell ≈ 0–1`, `roads/cell ≈ 5–15`. So:
- **119 "both" strata → `building_area` is the scarce binding metric** (tens of cells to reach 50 buildings; roads overshoot many ×).
- **27 "road_only" strata → `road_length` binds** (a handful of cells).

**Two `features_per_cell` quantities, kept distinct:**
1. **Real per-stratum density — MEASURED, not a proxy.** The regen script computes `real_fpc[city, metric, stratum]` from the real reference parquet's per-stratum feature counts ÷ the per-stratum cell census. (Neither is local now — parquet on Leonardo; `_heldout_cell_count.py` only *prints* — so the script must read the parquet on Leonardo and the census must be made emit-to-file; **work owed**.) Training-city aggregate fpc **rejected** as the real-side anchor: density varies hugely by stratum and an aggregate mis-sizes dense-vs-sparse.
2. **Gen/real emission ratio — the one training-city-informed PROXY.** Will a *generated* cell emit features at real density? Unknown pre-generation; training-city data informs its magnitude + over-emission risk (`over_cells ≈ 400 feats/cell` vs `≈10` normal). Labeled a proxy, **validated against held-out at first generation**, error absorbed by Gate-5 headroom. Never ground truth.

**Thin-CELL ceiling (the 82/185 census, made precise).** A building-floored stratum that is cell-sparse has building-*dense* cells (≥50 buildings in <50 cells). The risk is the **available-cell ceiling**: `max gen building features ≈ available_cells × real_density × gen_ratio`. If `gen_ratio < 1`, a cell-sparse stratum has **no spare cells to recover with** — `GENERATE_MORE_CELLS` (tier-2) has no room → tier-3 / report-as-underpowered. The feature-level `0/312` hid this because it measured the *real* side (dense cells, full pool), never the down-sampled stochastic gen side. **The sampler must surface this underpowered-risk set; it is NOT the feature-level pass.**

**Cell identity (from source).** A cell's stable key is its grid coordinate `(cell_i, cell_j)` within a tile (`sub_g/readers.py:71`), read-order-independent. Full key = `(city, tile_i, tile_j, cell_i, cell_j)`; the cell carries its density bucket (`_cell_density_by_cell`) and the tile's `(zoning, skeleton, coastal)` → its 4-tuple.

## Gate 4 — M3: selection rule & determinism  ✅ LOCKED (2026-06-21)

**Decision.** Per floored 4-tuple stratum: enumerate available conditionable cells in canonical sorted order by `(city, tile_i, tile_j, cell_i, cell_j)`; select `n_cells` by **blake2b hash-rank** — order cells by `blake2b(f"{seed}:{city}:{tile_i}:{tile_j}:{cell_i}:{cell_j}")`, take the first `n_cells`.

**Reasons.** stdlib `hashlib` is **byte-stable across Python/numpy versions** (a seeded `numpy` shuffle's bit_generator stream can drift across releases → breaks a write-once sha-locked manifest). No RNG state to thread; content-independent ⇒ no positional bias. Deterministic *top-n-by-id* rejected (positional bias toward low coordinates/early tiles).

**Rules.**
- **Take-all when capped:** `available_cells ≤ n_cells` → take all, mark stratum **ceiling-bound** (feeds the Gate-5 underpowered-risk set).
- **Per-stratum independent; uniform within a stratum** (the 4-tuple already fixes zoning/skeleton/density/coastal; uniform cell sampling weights tiles by cell count — default to simplicity).
- **One seed**, recorded in the manifest. **One manifest consumed identically by all 6 matrix runs** (2 backbones × 3 seeds) — comparability + reinforces write-once.

**Real-side weighting confirmed (gen-vs-real single-population).** Both sides use identical flat per-feature accumulation with **no per-tile/per-cell normalization** (`conditioning_discrimination.py:463-465` real; `gen_realism.py:84-85` gen). The real distribution is implicitly cell-weighted; hash-rank keys on cell *identity* not content, so the sampled subset is an **unbiased** sample of that same distribution. Uniform-cell sampling matches the real weighting inside KS.

## Gate 5 — M4: feasibility, headroom & escalation  ✅ LOCKED (2026-06-21)

Closes the Gate-1 real-vs-generated gap (*target = real-floored; obligation = generated ≥50*).

**Headroom.** `n_cells(stratum) = ceil(target_features / real_fpc[binding, stratum] × headroom)`. `headroom` is a single conservative config multiplier, biased **up** (protocol §10.1 asymmetry: under-provisioning silently drops a stratum from the KS aggregate and biases the verdict; over-provisioning only wastes bounded budget). It absorbs the gen/real emission ratio (<1 if the model under-emits), per-cell variance, and degenerate/empty gen cells. It is **provisional** — true `gen_ratio` is unknown pre-generation — so per §10.1 the lock never *binds* to it; the manifest records the achieved property and the real check lives at the consumer.

**Consumer-side power check, in the correct unit (§10.3), on the actual generated set.** After generation, per floored `(city, metric, stratum)`: measure **achieved gen features ≥ `min_n` = 50** — measured on what the model emitted, never assumed from the bridge. This is the fail-loud check §10.1 defers to the consumer.

**§9 construction-identity split (the teeth — distinguish a data limit from a sampler bug by construction identity, never by the thin symptom):**
- **Short AND ceiling-bound** (took all available cells, still short): *data limit* → **exclude-and-report**, mirroring the floor's "report, do NOT coarsen". Surface the labeled count; the stratum drops from Lane-S; if it under-powers a city, the existing `binding_city_verdict` #21 gate **demotes** that city (reused, not rebuilt); a city degraded past use → documented **SECOND_REGION** trigger.
- **Short AND NOT ceiling-bound** (cells were available, sized too few): *sampler bug* → **FAIL LOUD**, re-derive headroom. A sizing error must not hide behind the ceiling exclusion (the regime-distinguishing guard).

**Systematic under-fill** at first generation (`gen_ratio` broadly below the headroom assumption) → re-derive a **new manifest VERSION** (write-once, never in-place edit). `feature_resolution`'s `GENERATE_MORE_CELLS` (`:107-127`) is the per-stratum top-up where cells remain; `SECOND_REGION` where ceiling-bound.

---

## §6 — Q3: N as a parameter, cost model, §10.1 discipline

**N is never hardcoded.** The two knobs are config fields:
- `target_features` per floored `(metric, stratum)` — default = `min_n` = **50** (the floor's locked value; the obligation unit).
- `headroom` — conservative multiplier (Gate 5), default biased up; refined after first generation.

**§10.1 discipline (write-once vs provisional).** The manifest is sha-locked write-once, but `headroom` (and the `gen_ratio` it stands in for) is provisional. So: size by what is justifiable **today** (`real_fpc` measured from held-out + a conservative `headroom`); **record the achieved property** (`n_cells`, expected gen features, ceiling-bound flags) in the lock; and **defer the true power check to the consumer** (Gate 5 fail-loud). The lock never binds to a number known to be provisional.

**Cost model** (re-derived at manifest-build time, not fixed here):
```
total_sampled_cells = Σ_stratum n_cells(stratum)          # building_area binds the 119 "both" strata
total_generations   = 6 × total_sampled_cells             # one manifest, 2 backbones × 3 seeds
gpu_h               = total_generations × per_cell_gen_cost
per_cell_gen_cost  ≈ 0.0045 GPU-h  (transformer, ~600-tok self-terminated, 4-GPU-sharded; GROUND_TRUTH §3)
```
Rough transformer order: ~12k sampled cells → ~72k generations → ~325 GPU-h ≈ **~7% of the 5,000-GPU-h grant** (consistent with the handoff's "N=50/stratum ≈ 7%"). **Contingent on:** (a) `real_fpc[building_area]` per stratum — *computed by the regen script*, not known locally; (b) the **mamba gen-rate — UNVERIFIED at scale, measure at the next GPU smoke** (transformer rate is the only measured one). The old **~1,008 GPU-h / ~20% figure is DEAD** (it assumed 1,859 = cells). The budget is re-derived once (a) and (b) land.

## §7 — Q4: output contract (manifest + regen script)

Mirror the conditioning floor's locked-artifact discipline (`cfm.data.locked_yaml`: `sha256_excluding_field` + `stamp_and_seal`; a `_*_LOCKED` marker beside the file; write-once; reader refuses absent/unsealed/sha-mismatch/schema-skew).

**Artifact:** `data/processed/lane_s_sampler/<release>/sampler-manifest.yaml` (or parquet for the cell list if size warrants) + `_LANE_S_SAMPLER_LOCKED` marker. **Regen script** committed (`scripts/build_lane_s_sampler.py` + sbatch), reproduces the manifest byte-identically from `(holdout manifest, floor artifact, real reference parquet, seed, target_features, headroom)`.

**Manifest schema (v1):**
- `sampler_schema_version`, `release` (`2026-04-15.0`), `floor_sha256` (the `95abb88` floor it targets — lineage lock), `source_holdout_manifest` ref (the 1,952-tile set).
- `methodology`: `min_n`/`target_features`, `headroom`, `seed`, `selection: blake2b_hash_rank`, `sizing: ceil(target/real_fpc[binding]·headroom)`, `binding_rule: scarce_floored_metric`, **proxy disclosure** (`real_fpc` measured from held-out; `gen_ratio` a training-city-informed proxy validated at first generation).
- `held_out_cities`, per-city totals.
- `strata[]`: one row per targeted `(city, 4-tuple)` — `owed_metrics` (both | road_only), `binding_metric`, `real_fpc[binding]` used, `n_cells_target`, `n_cells_selected`, `ceiling_bound: bool`, `expected_gen_features[metric]`.
- `cells[]`: the selected cell keys `(city, tile_i, tile_j, cell_i, cell_j, density_bucket)` — the conditioning specs to generate.
- `sampler_sha256` (over the canonical YAML excluding itself).

**Consumer data flow:**
```
manifest.cells[]  →  generation loop (condition backbone on each cell's context, generate, decode)
                  →  list[DecodedCell]  →  gen_realism.gen_features_by_city(release=…)
                  →  lane_s_excess  →  decide()  (memorization-first → power-gated worst-case → winner | NO_DECISIVE_WINNER)
```
The manifest is the INPUT to the generation loop (which real cells to condition on); `gen_features_by_city` already re-derives the 4-tuple from `read_tile_labels` + density (`gen_realism.py:64-85`), so the manifest's cell key + density bucket is sufficient — no new keying path.

## §8 — Work owed (non-local computations the regen script must perform)
1. **Per-stratum held-out cell census as DATA** — extend `_heldout_cell_count.py` (currently *prints only*) to emit per-`(city, 4-tuple)` distinct-cell counts to a file.
2. **`real_fpc[city, metric, stratum]`** — from the real reference parquet (`reports/phase-2-bakeoff/real-features-2026-04-15.0.parquet`, on Leonardo) feature counts ÷ the per-stratum census. Both run on Leonardo (parquet not local).
3. **Deploy gate (carry from GROUND_TRUTH §5):** Leonardo must be re-deployed to the Mac's committed HEAD (incl. `gen_realism.py`) before any generation.

## §9 — Test plan sketch (for the writing-plans handoff; protocol gates)
- **Selection determinism:** same `(pool, seed)` → byte-identical manifest across cold processes with varied `PYTHONHASHSEED` (memory `feedback_pythonhashseed_dict_iteration_test`).
- **Bridge / scarce-metric binding:** on a fixture stratum, assert `binding_metric = building_area` where building_area floors and `n_cells` is sized by it (not roads, not a blended scalar).
- **§9 regime-distinguishing guard (threshold-pairing §2):** a short stratum that is **ceiling-bound** → exclude-and-report; a short stratum that is **NOT** ceiling-bound (cells available) → FAIL LOUD. The test names the bug: a symptom-keyed ("skip if thin") exclusion would pass the second case; the construction-identity exclusion must fail it.
- **Single-population (Gate 2/4):** gen and real keyed by the identical 4-tuple grammar; a sampled subset's feature distribution is an unbiased sample of the real (cell-weighted) distribution.
- **Lock grammar (§7):** reader refuses absent/unsealed/sha-mismatch/schema-skew; `floor_sha256` lineage matches `95abb88`.
- **External-source-of-truth (protocol Gate 6):** the manifest's 4-tuple + cell keys hand-enumerated against `conditioning_discrimination.py` / `sub_g/readers.py`, not via the sampler's own code.

## §10 — Open questions / risks (for review)
- **R1 — `gen_ratio` unknown until first generation.** Mitigated by conservative `headroom` (§10.1 up-bias) + the Gate-5 consumer check. The first generation is the validation point; a systematic miss → new manifest version.
- **R2 — Over-emission (`over_cells` ≈ 400 feats/cell).** A backbone could emit degenerate dense cells, inflating feature counts (passing min_n with garbage). Cell-EOS self-termination (Tooth-1) bounds runaway length; Lane-S KS still compares distributions, so garbage geometry shows as high excess, not a free pass. Flag for the eval's structural checks.
- **R3 — Ceiling-bound set size unknown locally** (needs the per-stratum census). If many building-floored strata are cell-sparse, more cities risk #21 demotion → `SECOND_REGION`. Quantified when work-owed item 1 lands.
- **R4 — Q3/Q4 not separately topic-gated** — written here from the locked principles (§10.1) + the original agenda. **Please review §6 and §7 with extra attention.**
