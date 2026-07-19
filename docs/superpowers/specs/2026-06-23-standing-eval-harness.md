# SPEC — Standing methodology-eval harness (v1)

**Status:** DRAFT for review (design-then-build; build gated on Umar). **Date:** 2026-06-23.
**Branch:** phase-2-cell-eos. **Canon:** `docs/GROUND_TRUTH.md`, `docs/PROJECT_FOCUS.md`.

## 0. Purpose & scope

A **re-runnable** harness that, given any checkpoint, emits three **echo-immune** numbers for
**v1 methodology decisions** (does it train; does conditioning inform it; does it draw valid
geometry). It is run once per checkpoint and an aggregator combines the 6 existing checkpoints
(`transformer-ar-53M`, `mamba-hybrid-54M`) × (`krakow-seed7/13/23`).

**This is NOT** the parked Lane-S realism crown. It contains **no** `decide()`, no floor / floor
re-pin, no Lane-S / Lane-M KS, no `gen_realism`, no `NO_DECISIVE_WINNER` verdict object, no merge.
It does **not** touch or "fix" the char_stats↔KS echo (PROJECT_FOCUS). The perplexity-gap is the
**§3.4 non-leak twin** (NLL, not generated-feature KS) and is the only architecture-*ranking*
number; ranking ≠ crowning (it reports the gap and the seed-noise; it does not declare a winner or
gate anything).

## 1. Inputs (exact)

| input | source | notes |
|---|---|---|
| Checkpoints (6) | `/leonardo_work/AIFAC_P02_548/.../checkpoints/bakeoff/{transformer-ar-53M,mamba-hybrid-54M}/krakow-seed{7,13,23}/last.ckpt` | one `last.ckpt` each; backbone/d_model/n_layers read from `ck["hyper_parameters"]` |
| Held-out cells (metric 1) | multiregion holdout manifest (glasgow, eisenhüttenstadt, munich, krakow) sub-F `cells.parquet` on Leonardo `$WORK` | via `read_sub_f_cells` + `holdout_manifest_for_region` + `sub_f_region_dir`/`tile_dirname`/`epsg_label_for_region`; real per-cell conditioning via `read_tile_labels`/`_derive_tile_conditioning`; char_stats via `derive_character_stats` |
| Training CSVs (metric 2) | `reports/logs/training-scaffold/version_N/metrics.csv` (Leonardo 222 tree) | `version_N`→(backbone,seed) resolved from `version_N/hparams.yaml` (see OPEN DECISION D4) |
| Fixed probe set (metric 3) | `scripts/_eyeball_gen_probe.py::CONTEXTS` — 3 contexts × 7 cells, gen seeds 1000–1006, probe cap 1536 | the SAME fixed probe as the eyeball/backbone-compare work; reused verbatim |

All inputs are read-only. `reports/` is never written by the harness (it writes under a new
`reports/_standing_eval/<run_id>/`, append-only, like the probe dirs).

## 2. Metric 1 — PERPLEXITY-GAP (conditional NLL on held-out cells)

**Definition.** For each held-out cell: teacher-forced NLL of the cell body under (a) its **matched**
conditioning and (b) a **shuffled** (donor) conditioning. `gap = NLL_shuffled − NLL_matched`
(nats/token). Positive gap = conditioning informs the body. **No generation.**

**Two columns (D1 RESOLVED):**
- **MACRO-ONLY (PRIMARY).** Shuffle only the macro buckets (zoning, road_skeleton, density,
  coastal); keep char_stats **matched on both sides**. The char_stats contribution cancels, so the
  gap isolates the model's use of the **macro conditioning** — the honest "what structure did it
  learn" read, with the echo channel removed. This is the architecture-ranking number.
- **FULL (SECONDARY).** Shuffle the whole conditioning (macro buckets + char_stats from the donor).
  Reported alongside as a sanity column; its gap also reflects char_stats informativeness.

**Computation (reuse `compute_perplexity_gap`).**
- Build, per held-out cell: `micro_tokens` = the cell body token list; `matched_conditioning_prefix`
  = `build_value_bearing_prefix(...)` from the cell's REAL labels + `[CHAR_PLACEHOLDER]`, paired with
  the cell's REAL char_stats; `shuffled_conditioning_prefix` = a donor cell's conditioning (see D1/D3).
- Inject `model_forward(*, micro_tokens, conditioning_prefix) -> float` = a thin wrapper that runs ONE
  teacher-forced forward of `[prefix + body]` through the loaded model and returns **mean NLL in
  nats/token over body positions only**, reusing the training masked-CE path (`lit_module._loss` /
  the model `forward(..., prefix_len=, char_stats=)`), so eval NLL == training NLL by construction.
- `compute_perplexity_gap(cells, model_forward, p_threshold)` returns `gap_nats_per_token`,
  `mean_gap_nats_per_cell`, `fraction_positive`, and the exact-binomial sign test.

**Aggregation across seeds (the ranking read).** Per backbone, the 3 seeds give mean gap + **std
across seeds (seed-noise)**, on the **MACRO-ONLY (primary)** number. Report transformer vs mamba side
by side with seed-noise visible. **If |mean_gap_T − mean_gap_M| < (seed-noise std), the harness labels
the row `NO_DECISIVE` (descriptive text, not a `decide()` verdict object).** FULL is shown in an
adjacent column for sanity, not used for the ranking flag.

**Units:** nats/token (primary), nats/cell (secondary). Sign-test p-value dimensionless.

**CAN conclude:** which architecture extracts more predictive information from conditioning on REAL
held-out cells (conditional-likelihood quality); echo-immune (NLL on real data, not generated-feature
KS — it cannot be gamed by char_stats being copied into the output); whether the T-vs-M difference
exceeds seed-noise.

**CANNOT conclude:** NOT a realism verdict, NOT the Lane-S crown, NOT geometry validity. Lower NLL ≠
more plausible geometry. Valid only because both models share identical tokenizer/vocab/data (they
do). The **PRIMARY (macro-only)** gap excludes char_stats by construction (matched both sides), so it
is the echo-free "did it learn macro structure" read. The **FULL** gap additionally includes
char_stats informativeness (channels 0–4 reduce NLL on size/length tokens) — reported only as a sanity
column, not the ranking number.

## 3. Metric 2 — SATURATION (loss-vs-steps, read-only)

**Definition.** From each run's `metrics.csv`, the `(step, train_loss, val_loss)` series; answer "was
loss still descending at ~112k steps?" → "would more training plausibly help?"

**Computation.** Parse CSV (pure, no GPU). Report: final logged step; loss at final step; loss at
80%/90% of training; **final-window slope** = Δloss over the last ~10k steps, in **nats/token per 1k
steps**; classify `DESCENDING` vs `PLATEAUED` by |slope| vs a stated threshold (D5). Plot-free; the
numbers are the artifact (a PNG of the 6 curves is an optional extra, not load-bearing).

**Units:** loss nats/token; slope nats/token per 1k steps; steps integer.

**CAN conclude:** whether each run's optimization had flattened by 112k (a "more steps" signal),
read directly from logs.

**CANNOT conclude:** that more steps would improve **geometry/generalization** (train/val loss ≠
downstream quality); NOT an architecture verdict — do **not** compare T vs M final loss as a quality
ranking (different inductive biases; lower loss ≠ better model). Descriptive per-run only.

## 4. Metric 3 — GEOMETRY-VALIDITY (echo-immune, on the fixed probe set)

**Definition.** Generate the fixed probe set from the checkpoint, decode, and report only the
construction-identity metrics that char_stats does **NOT** hand the model:
- **building closure-gap distribution** — for building-class blocks, gap = |first−last|/bbox-diag;
  report median, p90, % within 2% / 5% (NOT just the exact-equality rate — defect (a): the exact
  check is over-strict). Per context.
- **road fragmentation** — on the TRUE road set (construction-identity reclassified, so near-closed
  buildings are NOT counted as roads): components÷segments, largest-CC fraction, dangling-endpoint
  fraction, at the stated tolerance τ. Per context. (Interpretation: defect (c) — v1 representation,
  not a model bug.)
- **self-term %** — fraction of probe cells emitting `<cell_end>`(260) and not hitting the cap.
- **decode %** — fraction of feature blocks that decode.

**Computation (reuse).** `generate_cell_tokens` (probe gen) → `split_cell_into_features` +
`try_decode_block` + `_is_building_block`/`_is_closed_ring` + `promote_building_rings` (decode &
construction-identity) + the connectivity math from `scripts/_road_connectivity_diag.py` (lifted into
a reusable `src/cfm/eval` module). Closure-gap from the viz `build_data` classifier logic.

**Units:** closure-gap fraction (dimensionless); fragmentation ratios (dimensionless); τ in cell units
(= metres, quantum 0.5); self-term/decode in %.

**CAN conclude:** echo-immune structural validity of generated geometry per checkpoint per context
(does it seal rings; how connected are roads; does it self-terminate; does it decode).

**CANNOT conclude:** realism (probe conditioning is hand-built synthetic, in-distribution but not real
cells — no real-vs-gen claim); road fragmentation is a **v1 representation property** (defect (c)),
not a per-checkpoint quality defect; closure must be read as the **gap distribution**, not exact-rate
(defect (a)). **Counts** (n_buildings/n_roads) are **echo-tainted** (char_stats count/presence) →
reported descriptively only, excluded from the validity claim.

## 5. Entry point, outputs, table

- **Primary entry point (per checkpoint):** `scripts/eval_checkpoint.py --ckpt PATH --out DIR`
  → runs metrics 1 (GPU, NLL), 3 (GPU, generate+decode), 2 (CPU, CSV parse) and writes
  `DIR/<ckpt_id>.json` + a human `DIR/<ckpt_id>.md` table. Leonardo wrapper:
  `scripts/eval_checkpoint.sbatch` (1 GPU; same module/`LD_PRELOAD` env as the probe sbatch).
- **Aggregator (6-way):** `scripts/eval_aggregate.py --in DIR` → the transformer-vs-mamba table with
  per-metric mean ± seed-noise (std across seeds), incl. the metric-1 `NO_DECISIVE` flag.
- **Run manifest:** each run writes `manifest.json` = {code git sha, per-checkpoint sha256, held-out
  manifest sha, probe-set hash, all seeds, N} — reproducibility = config + code hash + data snapshot.

## 6. Byte-determinism

- **Metric 1:** teacher-forced (no sampling). Deterministic given (checkpoint, held-out cell set,
  shuffle seed, torch deterministic flags), single-GPU (rank 0). **Bit-identical on the same hardware;
  functional-identical at float tolerance across hardware** (the documented DDP/NCCL/float-ε reality —
  not bit-portable). The held-out cell set and donor assignment are fixed by a seed.
- **Metric 2:** pure CSV read → fully deterministic / portable.
- **Metric 3:** generation seeded (1000–1006) → same determinism contract as metric 1; decode is
  pure-deterministic.
- Outputs JSON with `sort_keys=True`; floats rounded to a fixed precision in the human table only
  (full precision in JSON).

## 7. Reuse map / new code

**Reuse (do not reinvent):** `eval/perplexity_gap.py` (shell); `data/training/conditioning.build_value_bearing_prefix`;
`data/training/build_shards.character_stats_for_cell` + `derive_character_stats`;
`data/sub_g/readers.read_sub_f_cells`; `eval/holdout/{paths,labels}`; `models/backbone.build_backbone`
+ `training/config.ScaffoldConfig` + `training/lit_module` loss path; `inference/generate.generate_cell_tokens`;
`data/sub_g/seam_decodability.split_cell_into_features`; `inference/generate.try_decode_block`;
`eval/geometry.{_is_building_block,_is_closed_ring,promote_building_rings}`.
**New:** `src/cfm/eval/standing/` (orchestration: NLL wrapper, current-schema shuffle builder,
connectivity+closure metrics lifted from the diag script, saturation parser, table assembler) +
the two scripts + sbatch. **NOT reused:** `eval/shuffles.py` (stale country/climate/era schema; the
harness builds the value-char-v1 shuffle inline).

## 8. Out of scope (guardrails)

No `decide()` / floor / floor re-pin / Lane-S / Lane-M / `gen_realism` / KS-on-generated-features / echo
fix / crown / merge / sampler-manifest rebuild. Metric 1 ranks and shows seed-noise; it does not crown.

## 9. Test plan

- `compute_perplexity_gap`: already shipped — add a wrapper test that NLL_matched on a tiny synthetic
  model + body equals the training-loss path (consistency tooth).
- shuffle builder: determinism under fixed seed; donor ≠ self (derangement); PYTHONHASHSEED-robust.
- saturation parser: synthetic CSV → known slope/plateau classification; threshold fires in the
  descending regime and not in the plateau regime (regime-distinguishing gate).
- geometry metrics: pin counts on the existing probe tokens (reuse `tests/test_viz_build_data` fixtures);
  connectivity matches `scripts/_road_connectivity_diag.py` on the same input.
- entry point: a `--smoke` path on 1 checkpoint × tiny N completes and writes a well-formed table.

## 10. DECISIONS — RESOLVED (Umar, 2026-06-23)

- **D1 — shuffle scope:** implement **BOTH** columns. **MACRO-ONLY is PRIMARY** (char_stats matched
  both sides → cancels → measures use of macro conditioning, echo removed). **FULL is SECONDARY**
  (sanity). (Reversed from the draft rec — macro-only is the honest "what did it learn" read.)
- **D2 — held-out N:** fixed seeded sample, **2,000/city**, `--n-per-city` tunable.
- **D3 — shuffle pool:** **within-city derangement** (donor ≠ self, donor from same city), fixed seed.
- **D4 — `version_N`→(backbone,seed):** resolve from `version_N/hparams.yaml`. **If the seed is not
  unambiguous, FAIL LOUD** and use a committed lookup table — **do not guess.**
- **D5 — saturation plateau threshold:** **derive from observed final-window noise** (not a magic
  number); **print the derived value** in the output.
- **D6 — run location:** **one Leonardo GPU job per checkpoint** (NLL + generate + CSV-parse + table) +
  a local `--decode-only` mode (re-decode dumped tokens off-GPU).

## 11. Build order (TDD, §9 teeth red-before-green)

CPU cores first (locally testable): saturation parser (regime-distinguishing test red→green) →
shuffle builder → geometry-validity (pin on existing probe tokens) → NLL≡training-loss consistency
tooth (tiny synthetic model). Then orchestration + entry point + sbatch. **Smoke on ONE checkpoint,
show the table, HOLD before the full 6.** No merge.
