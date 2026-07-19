# Multi-region (Europe) data-ingestion FEASIBILITY AUDIT — VERDICT (2026-06-02)

**Read-only audit. No extraction performed.** Branch `phase-2-bakeoff` @ `605d4b8`.
Answers the four questions in `docs/handoffs/2026-06-02-bakeoff-diagnostic-running.md`.
Every claim is anchored to real code (file:line), not optimistic estimation.

---

## STRAIGHT VERDICT

**"Extract all of Europe" is NOT (a) config-and-run. It splits cleanly:**

- **The CODE (CRS + tokenizer + buckets) = (b) moderate adaptation — DAYS.** The expensive,
  locked artifact — the geometry tokenizer and the model's embedding head — is **CRS-agnostic
  by construction** and transfers for free. The Singapore-specific parts are a *contained,
  mechanical* parameterization the spec already anticipated ("polymorphic per region").
- **The DATA (extraction + per-region validation at ladder scale) = (c) its own engineering
  sub-project — WEEKS.** This, not the CRS code, is the **binding constraint**. Serial ~8h/region
  cold-fetch, a 2 GB fetch guard, a five-stage per-region pipeline + validation that has only
  ever been run (and debugged across many sessions) on **one** region, and a per-rung need of
  ~27k–896k tiles (≈50–1,800 cities). **"All of Europe" cannot be validated before June-11 (9 days).**

**Per the handoff's own decision rule** ("days → more-data-now option 1; weeks → re-frame to
data-efficiency option 3"): the data path is **weeks**, so the near-term bake-off should be
**re-framed to the data-limited regime on current Singapore data**. Europe extraction is the
correct **Phase-4-production** move and a legitimate *next* sub-project — but it does **not**
rescue the bake-off inside the June-11 window. **The data-strategy decision remains the PI's**
(this audit deliberately does not pick among options 1/2/3).

---

## The crux, in plain language

A road/building is turned into tokens **relative to its 250 m cell**, not relative to Singapore.
The encoder writes a feature's start point as an offset *inside the cell* (0–250 m, snapped to
0.5 m) and then writes each next vertex as a **turn + distance** ("face 37°, go 4.5 m"). None of
those numbers know or care that the cell is in Singapore. Drop the same street in Berlin, project
it into Berlin's own metre-based grid, and you get tokens **from the exact same vocabulary**.

So the part that was hard and slow to build — the tokenizer and the trained embedding — is
**already region-portable**. What is Singapore-shaped is the *plumbing that gets data into cells*
(which projection, which tile labels) and the *frequency-derived bucket boundaries* — both small,
both extensible. The genuine wall is **getting and validating European data at the scale the
compute-optimal ladder demands**.

---

## Q1 — Coordinate systems (the flagged "likeliest hard blocker"): NOT baked deep

**Finding: CRS is hardcoded but *contained*, and the token layer is CRS-agnostic.**

CRS-agnostic by construction (the load-bearing fact):
- `sub_f/encoder.py:286-313` `_hierarchical_anchor_tokens` — anchors are `coords[0]` quantized at
  0.5 m, sized for a **250 m cell** (500 quantum-cells/axis). `encode_cell(..., cell_origin=(0,0))`
  default → **cell-local** coords (`encoder.py:548-568`).
- `sub_f/encoder.py:319-353` `_direction_magnitude_pair` — vertex-to-vertex **deltas** (1° bins,
  0.5 m quanta). Translation-invariant; identical physical quantities in any metric (TM/UTM) CRS.
- `sub_f/decoder.py:39-68` — inverse uses the same universal constants. Full symmetry.
- `sub_f/vocab.py:191-195` — of **686** on-disk tokens, **531 are geometry/structural/bref
  (universal)**; only **155** are OSM-tag-derived. The geometry vocab is not Singapore-tuned.
- Because anchors are **cell-local**, large European grid coordinates (e.g. UTM eastings ~500k,
  northings ~5–6M) never enter the token stream → **no float-precision loss** at the token layer.

Where 3414/SVY21 IS hardcoded (the complete inventory to parameterize):
- `sub_c/coords.py:19,25-27` — `SVY21_EPSG_CODE = 3414` module constant + a **module-level
  singleton transformer** `_TRANSFORMER_4326_TO_SVY21`. Function names baked: `*_to_svy21`,
  `tile_id_from_svy21`. (~50 `*_svy21` references propagate through `sub_c/pipeline.py`.)
- `sub_c/pipeline.py:418` `region_crs=f"EPSG:{SVY21_EPSG_CODE}"`; `sub_e/pipeline.py:223`
  `region_crs="EPSG:3414"` (a bare string literal).
- Tile dir-name string literal `tile=EPSG3414_i{i}_j{j}` hardcoded in **~10 files**
  (`sub_c/pipeline.py:565`, `sub_c/validator_cross_tile.py` ×4, `sub_d/*`, `sub_e/pipeline.py` ×3,
  `sub_f/pipeline.py:257` parse, `sub_g/validator.py:247`). **`eval/holdout/paths.py:31-33`
  already parameterizes it** (`epsg_label` arg) — the seam exists, call sites just don't use it.
- The 2 km tile grid is **CRS-origin-aligned** (`coords.py:49-58`, `floor(x/2000)` from the CRS
  (0,0)). This is *by design* per-CRS (spec §7.2 "CRS-origin-aligned"), not a Singapore bake — it
  works for any metric CRS; the i,j indices just become larger integers.
- `configs/data/regions/singapore.yaml:12` `crs: "EPSG:4326"` is the *source* CRS of the bbox.
  **There is no `projected_crs` field today** — the 3414 target lives in code, not config.

Spec intent is already multi-region: `coords.py:4` "polymorphic per region"; `coords.py:7-9,71-85`
densification signature "locked for **Sweden** enrollment … without re-opening this code."

**Q1 verdict: (b) moderate.** Add `projected_crs` to the region config; replace the module
singleton with a region-bound transformer factory; derive the tile label from the region CRS;
sweep the ~10 literal sites (helper already exists). Mechanical, testable, **days**.

## Q2 — Tile-grid + tokenizer generalization: generalizes, with a cheap append

- **Geometry vocab: universal** (see Q1). Transfers for free.
- **Semantic vocab (`configs/sub_f/vocab_floor_analysis.yaml:61-69`):** 127 BP1 slots =
  **28 global-OSM-wiki L1 keys + 56 wiki-L2 highway/building pairs + 43 Singapore-empirical
  pairs**. So **84/127 are global-wiki-derived**; the 28 L1 keys (aerialway…healthcare…) are
  universal OSM semantics Europe shares. Only the *frequency filter* that admitted slots used
  Singapore quantiles (`proposed_x_threshold.candidate_a_singapore_elbow`).
- **Graceful degradation:** any European value missing from BP1 buckets to the BP4
  `<unknown_{key}>` family (`encoder.py:253-283`) — no crash. A *new L1 key* (rare; OSM schema is
  global) would be the only thing forcing a slot.
- **Append-only is cheap here:** `subf_vocab_size() = max(token_id)+1 = 1508` (`backbone.py:44-46`;
  max bref id 1507). The embedding **already spans the full 1508-id space**, with ~822 reserved
  ids unused. **Appending European tags needs no embedding resize** — the rows exist; it's a YAML
  add + retrain (you retrain anyway). Not a model-breaking phase-transition.
- **Density/conditioning buckets (`configs/macro_plan/v1/macro_plan_vocab.yaml:3471-3537`):** all
  three numeric namespaces (`cell_density`, `road_skeleton`, `tile_population_density`) are
  **open-topped** (`upper_exclusive: null` on the last bucket) and based on **bounded ratios**
  (building-footprint fraction, p75 proxy). A denser European city shifts toward `bucket_3`
  **gracefully** — no out-of-range, no append. `zoning` = the 4 **universal theme types**
  (building/road/poi/base), not Singapore land-use. Worst case is *statistical* (resolution piles
  up at the top bucket → optional re-binning), not a structural break.

**Q2 verdict: (b) moderate.** Re-run the frequency floor on European data, likely append a handful
of semantic slots (append-safe, no resize), optionally re-bin density. **Days.**

## Q3 — Extraction + storage at scale: THE BINDING CONSTRAINT — (c) weeks

- `overture/loader.py:47-138` `load_region`: **one region, bbox-filtered, serial** (5 themes in a
  `for` loop), each `pq.write_table` holding a whole theme in RAM. `OVERSIZED_THRESHOLD_BYTES = 2 GB`
  (`loader.py:44,243-247`) — country-sized fetches blow past it (need `confirm=True` + memory
  headroom). The ~8h Singapore cold-fetch ([[project_overture_cold_fetch_slow]]) is dominated by the
  DuckDB-httpfs `estimate_size` COUNT(*) scan (`loader.py:223-242`).
- **No multi-region orchestrator exists** (grep: only within-pipeline loops; no region-batch
  runner). Regions are *independent* `load_region` calls, so extraction is **parallelizable in
  principle** by launching N jobs — but you'd build that orchestration, and each fetch is internally
  serial + S3-bottlenecked.
- **Only `singapore.yaml` is enrolled.** The "polymorphic per region" seams have **never been
  exercised** with a second region → unverified code paths ([[feedback_sample_regime_blind_locks]]).
  Only `sub_g` has a CLI; sub_c…sub_f are script-driven → running the full pipeline on a new region
  is itself a manual multi-step orchestration today.
- **Scale math:** 30M rung ≈ 27k tiles ≈ ~50 Singapore-equivalents; full ladder to 1B ≈ 896k tiles
  ≈ ~1,800 cities. Each region needs fetch **and** a five-stage pipeline + validation that, on
  Singapore alone, surfaced 7 regime/contract catches and a multi-cycle sub-G quarantine. European
  data **will** hit new regimes (road topologies, multipolygons, grid-origin edge cases).
- **Timeline:** "all of Europe" validated before June-11 is **infeasible**. A ~50-city slice (30M
  rung) is *conceivable* in ~1–2 weeks **only after** the Q1/Q2 code lands and an orchestrator
  exists — and that estimate is optimistic against this project's "validation surfaces new regimes"
  track record.

**Q3 verdict: (c) own sub-project, weeks. This dominates the overall verdict.**

## Q4 — Overture uniformity across Europe: NOT a blocker

- Overture is a **single global release** with a **uniform GERS schema** across all themes
  (`loader.py:36-42` reads the same 5 themes generically). Europe = same columns/structure as
  Singapore. No per-country schema divergence the pipeline would choke on.
- **Coverage is strong in Europe** (OSM-rich transportation/places; OSM + ML building footprints).
  The known 94%-missing-label issue ([[project_data_state]]) is **global** and already handled
  (`B__UNK__` sentinel → BP4 `<unknown_*>`, `encoder.py:240-241,270-281`). Europe adds no *new*
  schema problem.
- Only mechanical add: per-city/country admin-polygon sourcing via the divisions theme
  (`singapore.yaml:7-10` `level: country|region|locality` already supports it) → one region config
  per enrolled area.

**Q4 verdict: not a blocker.** Schema-uniform; coverage good; per-region config is mechanical.

---

## Decision tee-up (PI call — not made here)

| Question | Verdict | Effort |
|---|---|---|
| Q1 CRS code | (b) moderate — contained, spec-anticipated | days |
| Q2 tokenizer/buckets | (b) moderate — append-safe, no resize | days |
| Q3 extraction + validation @ scale | **(c) own sub-project** | **weeks** |
| Q4 Overture uniformity | not a blocker | mechanical |

**Overall: (c) — weeks — because of DATA, not code.** By the handoff's decision rule this points
to **re-framing the bake-off to data-efficiency on current Singapore data (option 3)** for the
June-11 window, while treating **Europe extraction as the next standalone data-pipeline
sub-project** (it is needed for Phase-4 production regardless). The choice among options 1/2/3
remains the PI's.

**Held pending the PI decision:** all scored bake-off runs + Task 5 (mamba-ssm lock); spec §4
(Topic-2 compute axis) and possibly §11/Topic-1 are a re-open.

---

# ADDENDUM — BOUNDED-SLICE re-cost (PI reframe, 2026-06-02)

PI corrections accepted: (1) June-11 is a **soft** wall — allocation renewal is confident (~couple
days), checkpoints live on `$WORK` (survive expiry), so a data phase can run into the renewed
window; (2) nobody needs all of Europe — the question is the **smallest multi-region extract that
supports a 3-point compute-optimal ladder**, preserving the nine-topic methodology rather than
re-framing to data-efficiency. This addendum re-costs for that bounded target. **The original
"weeks (all Europe)" verdict was scoped to full Europe; re-scoped below.**

## THE LOAD-BEARING CORRECTION: cost is set by the ladder CEILING, and 300M is not bounded

Compute-optimal `D = r·N` (spec §4, r≈20 Chinchilla anchor; spec says geometry-r is *measured*,
could be lower). Tokens/tile = 14.4M/494 = **29,150** (measured). Corpora **nest** — extract once
for the top rung; lower rungs subsample. So the **top rung sets the extract size.**

| Ladder ceiling | Corpus (r=20) | Tiles | ≈cities @494/city (SG-class) | ≈cities @150/city (mid) |
|---|---|---|---|---|
| **30M** | 600M tok | **20,583** | ~42 | ~137 |
| **100M** | 2.00B tok | **68,611** | ~139 | ~457 |
| **300M** | 6.00B tok | **205,832** | ~417 | ~1,372 |

**(a) answer + the correction:** the literal **30M/100M/300M ladder needs the 300M corpus ≈ 206k
tiles ≈ ~400–1,400 cities — NOT a bounded 50–150 slice.** A ~50–150 city slice supports a
compute-optimal ceiling of **~30M** (up to ~100M only at the small-city end). You cannot get a
*300M* compute-optimal point from 150 cities without dropping r — which re-introduces the exact
data-starvation confound the diagnostic exposed (that's option-3-in-disguise, not a real
compute-optimal point).

**Cities band is the dominant uncertainty** (European tiles/city is unmeasured; Singapore=494 is a
dense city-state). A **1-city European pilot calibrates it** and kills the band — do it first.

**Sensitivity (tailwind, do NOT bank):** if measured geometry-r < 20 (plausible — geometry tokens
are more redundant than language), every target scales down proportionally (r=10 → halve all tile
counts). Measure geometry-r on the 30M slice (non-starved) **before** committing to upper rungs.

## The real tradeoff: data-cost vs extrapolation reach (production ≈ 1.75B, spec §4)

| Ceiling | Extract | Eng. timeline* | Extrapolation to ~1.75B |
|---|---|---|---|
| current single-region | 0 (have it) | 0 | **2400×** (hopeless — the diagnostic's wall) |
| **30M (bounded)** | ~40–140 cities | **~2–3 wks** | **~58×** (long but real, methodology intact) |
| 100M | ~140–460 cities | ~3–5 wks | ~17.5× (better) |
| 300M | ~400–1,400 cities | ~6–10+ wks | ~5.8× (best; **no longer bounded** = full data phase) |

\*assumes the 1-city pilot clears and the fetch-egress question (below) resolves favorably.

**Punchline:** a bounded **~40–140 city** extract buys a **30M-ceiling 3-point compute-optimal
ladder** that cuts production extrapolation from **2400× → ~58×** while **preserving the
compute-optimal methodology** (unlike option-3 re-frame) and **without all of Europe** (unlike the
option-1 strawman). This is the middle path, and it is real.

## (b) Q1/Q2 code effort — 3–5 days

- **Q1 CRS parameterization: 3–5 days.** `coords.py` transformer-factory (region-bound, replacing
  the module singleton); add `projected_crs` to region config + thread through sub_c/sub_d/sub_e
  manifests; derive the tile-label from region CRS + sweep the ~10 `tile=EPSG3414_` literal sites
  **and their test/validator guards** (per "lock and guards travel together"); 1-region regression
  proving no behavior change on Singapore. Mechanical but determinism/lock-sensitive — not 1 day.
- **Q2 for the bake-off: ~0–1 day.** Ship with the existing Singapore-derived vocab floor + BP4
  `<unknown_*>` graceful fallback; density buckets are open-topped (no change). Full EU frequency
  re-derivation deferred to Phase-4 (a bake-off compares backbones on one ruler, not vocab
  coverage). Documented caveat: EU-frequent tags degrade to `<unknown_key>`.

## (c) Orchestrator-build — ~5–8 days (one-time; reused by Phase-4)

- Per-region config generation (city bbox + admin level from a curated list / Overture divisions): 1–2 d.
- End-to-end per-region driver chaining the 5 stages (`extract_tiles` → `derive_macro_plan` →
  sub-E → sub-F → `sub_g/cli`) with idempotent resume + per-region `_SUCCESS` gating: 2–3 d.
- Fan-out wrapper + roll-up manifest + "which regions passed validation" gate: 1–2 d.
- **Cold-fetch `COUNT(*)` optimization (8.2h → ~1h): ~1 d, high-leverage** for (d) — the 8.2h is
  the pre-flight estimate (`loader.py:223-242`), not the ~147MB fetch.

## (d) Per-region extraction wall-clock — two parts, two parallelism stories

**FETCH** (S3, needs egress) and **PROCESS** (sub-C..G, CPU, no internet) parallelize differently.
**Established model = fetch LOCALLY (Mac) → rsync to Leonardo** (Leonardo can't even fetch GitHub;
`extract_tiles.py` + the local `data/cache/` confirm this). So:
- **FETCH, optimized to ~1h/region:** local Mac ~4–8 concurrent (bandwidth-bound) → 40–140 cities ≈
  **1–3 days**. (Unoptimized 8.2h → ~2–3 weeks; the `COUNT(*)` cut is the lever.)
- **Faster option:** a **cloud VM co-located with Overture S3 (us-west-2)** → high concurrency →
  <1 day. Recommended if local proves bandwidth-bound.
- **UNVERIFIED, high-leverage:** does Leonardo allow S3 egress from `dcgp_usr_prod`/dmover nodes? If
  yes, fetch fans out as a Slurm array → <1 day. **Verify cheaply (one node, one `load_region`).**
- **PROCESS:** CPU, no egress → Slurm array on Leonardo's **budget-free `lrd_all_serial`** after
  rsync → ~1 day for ~140 regions.
- **Net (d): ~1–3 days** for a 40–140 city slice with the `COUNT(*)` optimization + reasonable
  parallelism. Dominant risk = fetch egress/bandwidth → **the 1-city pilot resolves it.**
- **Storage:** ~147MB raw/city × 140 ≈ ~20GB raw + processed multiple → tens of GB, fine on `$WORK`.
  (The 300M corpus's ~206k tiles → hundreds of GB — another reason to stay bounded.)

## (e) Validation burden — front-loaded, then amortizes

- **The 7 catches do NOT recur** — they were one-time methodology/contract fixes, already in code,
  fully amortized.
- **What recurs is DATA-REGIME validation via the EXISTING automated sub-G cross-artifact
  validator** (not re-engineered per region). Expect the first **2–5 EU cities** to surface new
  regimes — multipolygon buildings ([[project_sub_c_multi_geometry_gap]]), dense medieval cores,
  roundabout/junction topologies, large-UTM tile-index behavior, cross-border admin — → ~2–4 fixes,
  declining per city.
- **After the first handful: automated + occasional structural-exclusion of outliers**
  ([[feedback_structural_exclusion_not_magnitude]]). Near-free.
- **Estimate: ~1–1.5 weeks of triage concentrated in the pilot + first ~5 cities; near-zero
  thereafter.** The 1-city pilot is the gate that front-loads it.

## Recommended STAGED plan (risk-managed; fits the renewed window)

1. **Pilot + code (parallel), ~1 wk:** land Q1 (3–5d) + start orchestrator + **verify Leonardo S3
   egress** + run a **1-city EU pilot** → calibrates tiles/city, proves the CRS code path, surfaces
   first regimes, prices fetch.
2. **30M slice, ~1 wk:** extract ~40–140 cities (parallel), validate (amortizing), run a **30M
   compute-optimal point** and **re-measure geometry-r on non-starved data.**
3. **Decide the ceiling with measured numbers:** re-cost 100M/300M using *measured* tiles/city +
   geometry-r. Extend the ceiling within the renewed allocation **only if the city-count is
   acceptable** — else freeze the bake-off at the 30M-ceiling 3-point ladder (already a real result).

**Net: ~2–3 weeks to a validated, ladder-ready 30M-ceiling bounded slice** — a defensible
compute-optimal bake-off, methodology intact, no all-of-Europe. Extending to a 100M ceiling adds
~1–2 weeks; a literal 300M ceiling is the full data phase (~6–10+ wks), not a bounded slice.
