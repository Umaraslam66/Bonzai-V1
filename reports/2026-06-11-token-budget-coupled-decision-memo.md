# Token-budget coupled decision — DECISION MEMO (2026-06-11)

**Status: LOCKED by Umar, 2026-06-11.** The locked quadruple:
`DEFAULT_MAX_CELL_TOKENS = max_len = 13,312`; `eval_max_new >= 13,312` at scored
runs; `MAX_TOO_LONG_DROP_RATE = 0.005` unchanged; commensurability = one
authoritative number (2048 opt-downs legal but mechanically unscoreable;
entrypoint assert at execution). The lock was CONTINGENT on the flash-attention
probe (execution obligation #2) — **DISCHARGED: PASS**
(`reports/2026-06-11-sdpa-window-probe.yaml`, Leonardo jobs 45898653
measurement / 45899171 artifact, A100-SXM-64GB, torch matches the bake-off
lock): default SDPA path engages `pytorch_flash::flash_fwd_kernel` at T=13,322
(peak 5.909 GiB at toy b8, identical to flash-forced; math contrast 45.689 GiB,
math b8 OOM as expected); 300M-class b2 fwd+bwd peak 19.959 GiB. The window is
runnable as-is at both ladder ends.

**The constant change still does NOT land here — DEFERRED TO EXECUTION,
deliberately, not forgotten** (recorded on Umar's word, 2026-06-11). It ships at
scored-run-planning execution as ONE commit: the 13,312 constants + the §4
entrypoint commensurability assert (`eval_max_new >= budget`) + the
lock-and-guards test sweep (grep tests for assertions pinned to 5760; update in
the same commit).

**UPDATE 2026-06-12 — EXECUTED as W1 of the GPU-wait CPU drain (pulled forward
on Umar's word with the approved worklist).** One commit on branch
`phase-2-w1-budget-13312`: `DEFAULT_MAX_CELL_TOKENS = 13_312`,
`ScaffoldConfig.max_len = 13_312`, the `--scored-run` entrypoint gate
(`assert_scored_commensurate`: max_len == the lock AND eval_max_new >= it;
wired into `bakeoff_run.sbatch`), `--budget` flag on the measure script, the
lock-and-guards sweep (test pins updated; the sub-F alpha-drop band's 5760 is
anchored to the sub-F PADDED budget 6016-256 — a different, unmoved lock — and
deliberately untouched), obligation #4's regime-pair test (a 10,000-token cell
flows at the lock, drops at 5760), and the recorded per-city tail-drop
measurement at 13,312 (obligation #1; first real post-lapse CPU job).

This is backlog #1 / resume-gate #3: the decision deferred from gated step 15.5
("resolved-direction, number deferred-by-design"). Direction already recorded by Umar
(2026-06-11): de-densify is RULED OUT with evidence; raise the budget toward ~p99.9 of
the dense cities, with recorded per-city tail-drop for the extreme tail.

---

## 0. Why this cannot wait for GPU

Recomputed from `reports/2026-06-11-cell-token-lengths-38cities.yaml` (sum of
`frac_over_budget x n_cells` over all 38 cities): the EU training union has
**1,046,162 non-empty cells, of which 9,590 (0.92%) exceed 5760**. The F13 action
contract halts when the union over-length drop rate exceeds
`MAX_TOO_LONG_DROP_RATE = 0.005` (`datamodule.py`, `CellDataModule.setup`,
"accumulate-then-check over the WHOLE union"). 0.92% > 0.5%: **a scored
`eu-train-union` run at today's default raises `DropRateExceeded` and refuses to
start.** The first scored GPU-hour is blocked on this number regardless of when the
allocation renews.

## 1. The coupling — why this is ONE decision

Four knobs, one mechanism. Setting any one alone silently re-decides the others:

1. **`DEFAULT_MAX_CELL_TOKENS`** (`src/cfm/data/training/datamodule.py`, currently
   5760) — the data-side drop threshold: any cell longer than this is dropped from
   training entirely (dropped, not truncated — truncation would cut a feature
   mid-grammar). It is also the **floor regime**: `train_scaffold.py` passes
   `floor_regime_cell_length=DEFAULT_MAX_CELL_TOKENS` into the eval ("the floor
   regime is a property of the DATA REGIME ... NOT of this run's cfg.max_len").
2. **`ScaffoldConfig.max_len`** (`src/cfm/training/config.py`, currently 5760) — the
   model-side context window. It sizes the learned positional-embedding table
   (`MicroAR`: `nn.Embedding(cfg.max_len + 10, d_model)` via `backbone.py`) and sets
   the worst-case attention cost. A cell the data side keeps but the model side
   cannot hold has nowhere to go, so these two must be the same number.
3. **The 2048-vs-5760 commensurability question** — three Leonardo entry points
   (`bakeoff_diagnostic.sbatch`, `scaffold_ddp_short.sbatch`,
   `scaffold_scaleup_probe.sbatch`) deliberately pass `--max-len 2048` for memory
   headroom/smoke. The question parked at 15.5: what makes a run at an opted-down
   budget incomparable with floors derived at the full budget, and what enforces
   that? (Resolved in §4, option-independent.)
4. **`ScaffoldConfig.eval_max_new`** (currently 512) — discovered while verifying
   the F15 seam for this memo, and it belongs in the same lock: the §2 emergence
   verdict is REFUSED as `INCOMMENSURATE` whenever
   `generated_length_cap < floor_regime_cell_length`
   (`src/cfm/eval/slice_metrics.py`, F15 refuse rule: "a capped generation never got
   the budget to emit what the floor counts"). **Today's default 512 < 5760 would
   refuse the verdict on every scored run at ANY budget choice.** Scored-run configs
   must set `eval_max_new >= the locked budget`.

Derived, not independent: **`MAX_TOO_LONG_DROP_RATE` (0.005)** was calibrated as "5x
the per-city P99.9 design point" when 5760 was the Singapore p99.9. If the new budget
is again a p99.9-class number (per-city drop <= 0.1%), the same 0.005 keeps its
original meaning unchanged — recommendation below keeps it.

**One thing that does NOT couple (verified):** the frozen conditioning-floor artifact
(`reports/conditioning_floor/2026-04-15.0/`, Lane S/M floors) is token-budget-
independent. Its real-side features come from `read_sub_f_cells`
(`src/cfm/data/sub_g/readers.py`), which reads every cell with **no length filter** —
neither `conditioning_floor.py` nor `conditioning_discrimination.py` references the
budget. Raising the budget needs **no floor regen**. (Same for the emergence floors:
their `derivation_regime.cell_length: "full"` is the holdout derivation, not a model
window.) Corollary: at 5760 the model never trains on the densest ~1% of cells, yet
Lane S scores it against real distributions that *include* them — the current budget
biases the model against exactly the cells the eval counts.

## 2. The evidence (from the two artifact YAMLs; every number recomputed locally)

Sources: `reports/2026-06-11-token-length-investigation.yaml` and
`reports/2026-06-11-cell-token-lengths-38cities.yaml` (both derived at `8f9fc45`,
release 2026-04-15.0, budget 5760; lengths over non-empty cells, strict `len >
budget` matching the training drop).

**Why the gate fired — genuine density, not redundancy (15.5 investigation):**

- Tokens-per-building is flat ~15 everywhere: over-length cells 15.2–15.7 vs normal
  cells 15.0 across all investigated cities. The encoding is not bloating dense cells.
- NL source geometry is LEAN: over-cell vertices-per-building ~6.7–7.1 in
  almere/rotterdam/amsterdam vs 11.7 (warsaw) / 14.0 (barcelona) controls.
- The driver is **feature count**: median over-length rotterdam cell = 406 real
  features (301 roads, 88 buildings) vs 15 in a normal cell; roads carry ~80% of
  over-cell tokens (building token share ~0.19–0.21).
- De-densify is ruled out with evidence: a 0.5 m shape-preserving simplification
  leaves projected drop rates **2.1x–7.8x over the 0.005 gate** (recomputed:
  rotterdam 0.039, amsterdam 0.035, almere 0.026, tilburg 0.013, eindhoven 0.012,
  barcelona 0.011).

**What 5760 costs in data:** 11/38 cities over the 0.5% alarm — barcelona 7.99%,
rotterdam 6.26%, amsterdam 5.79%, almere 4.46%, tilburg 2.41%, eindhoven 2.31%,
cergy 1.54%, valencia 1.40%, bruges 1.09%, lisbon 0.66%, manchester 0.63%. Because
the dropped cells are the long ones, they carry an outsized token share:
**in almere/rotterdam/amsterdam, the ~4.5–6.3% of cells dropped carry ~51–52% of the
city's cell tokens** (median-based estimate: n_over x median_len(over) vs rest;
means would push this higher). These are the dense urban cores — exactly the
"character" the conditioning bet is about.

**The length landscape across 38 cities:** worst p99 = barcelona 9,869; worst p99.9
= valencia 13,268 (next: barcelona 12,156, amsterdam 11,688); global max = valencia
19,372 (an outlier — next max is amsterdam 14,952). Full valencia coverage is
unreasonable (already recorded at 15.5).

## 3. The cost model — what raising the window buys and costs

**Pad or pack? VERIFIED: neither flat-pad nor packing — pad-to-batch-max.**
`collate_cells` (`src/cfm/data/training/datamodule.py`: "Right-pad `ids` to the batch
max") pads each batch only to the longest sequence *in that batch*, and
`training_loss` masks padding targets via `seq_len`. So `max_len` is a **ceiling, not
a flat per-cell tax**: a batch of short cells stays cheap no matter what the ceiling
is. The cost of raising the budget is paid only when a long cell is actually sampled
— but when one is, it widens its whole batch (one 13k cell makes all 8 rows 13k wide).
With union P(cell > 5760) = 0.92%, **~7.1% of shuffled batches (batch_size 8) will
contain at least one newly admitted cell**; the other ~93% are untouched.

**Worst-case batch multiplier (memory sizing + compile must use this).** Per-layer
forward FLOPs ~ 24·T·d² (linears) + 4·T²·d (attention); multiplier vs a full 5760
window:

| window | d=256 (toy) | d=1024 (~300M-class) | attn share @d=256 | linear factors (KV cache, pos rows, activations) |
|--------|-------------|----------------------|--------------------|-----------------|
| 5760   | 1.00x       | 1.00x                | 79%                | 1.00x |
| 8192   | 1.90x       | 1.71x                | 84%                | 1.42x |
| 10240  | 2.87x       | 2.45x                | 87%                | 1.78x |
| 13312  | 4.70x       | 3.78x                | 90%                | 2.31x |
| 16384  | 6.99x       | 5.38x                | 91%                | 2.84x |

In plain terms: attention compares every token with every earlier token, so doubling
the window roughly quadruples that part of the work; everything else (memory for
generation state, the position table, activations) grows in straight proportion.

**Expected (not worst-case) training cost rises far less than the table** — most
batches never see a long cell — but the *worst* batch is what must fit in GPU memory
and what a compiled kernel must handle. Three concrete cautions:

1. **Attention-memory cliff (must verify on hardware).** `MicroAR` runs
   `nn.TransformerEncoder` with an explicit causal mask + `is_causal=True`. If
   PyTorch's flash/memory-efficient attention kernel engages, attention memory is
   ~linear in T and 13k is comfortable. If it falls back to the math path, the score
   matrix alone at T=13.3k is ~45 GB (batch 8, 8 heads, fp32) — instant OOM on a
   64 GB A100. The 2048 opt-downs existing "for memory headroom" says memory was
   already a concern at 5760. **Verification obligation: a short non-scored probe at
   the locked window (existing `scaffold_scaleup_probe.sbatch` path) before any
   scored run; fallback is batch_size down + `grad_accum` up (the comparability
   mechanism already in `ScaffoldConfig`).**
2. **Eval generation cost scales too.** §1.4 forces `eval_max_new >= budget` at
   scored runs; AR generation of a 13k-token cell is ~5x a 5760 one worst-case. The
   scaffold already prices eval explicitly ("autoregressive generation is the
   binding bake-off cost"); with `eval_cells = 64` this is bounded, and most cells
   stop at EOS long before the cap — the cap costs only what long cells actually use.
3. **Under-training of long positions (real, bounded, shared).** The positional table
   is learned per-position; rows above 5760 receive gradient only from the 9,590
   admitted long cells, rows above ~10k from roughly the union p99.9 tail
   (~1,300 cells), rows near 13k from a few hundred. Those rows will be undertrained
   relative to row 100. This is an accepted cost of covering the tail (it argues
   against windows beyond p99.9-cover, where rows would be *never* trained), and it
   is a genuine architecture property the bake-off is allowed to see: mamba-hybrid
   has no quadratic window and discrete-diffusion pays per denoise step — the same
   budget prices the three backbones differently, which is part of what we are
   measuring.

## 4. The commensurability resolution (knob 3 — option-independent)

The 2048 opt-downs are **legal but never scoreable**, enforced mechanically, with one
authoritative number:

- `DEFAULT_MAX_CELL_TOKENS == ScaffoldConfig.max_len default == B` (the locked
  number). One source; `floor_regime_cell_length` already reads the constant, so the
  floor regime follows automatically.
- Diagnostic/smoke entry points may keep passing `--max-len 2048`: the F13 halt
  already deliberately enforces only at/above the default ("in that regime the rate
  measures the opt-down, not a corpus defect"), with the unconditional INFO log for
  visibility. Unchanged.
- The teeth that make an opt-down unscoreable: the F15 refuse rule yields
  `INCOMMENSURATE` (never a verdict) whenever the generation cap is below the floor
  regime, and a model built at `max_len` 2048 physically cannot emit past its
  positional table (loud index error, not silent truncation).
- **One added tooth at decision-execution time** (not now): scored bake-off runs
  assert `cfg.max_len == DEFAULT_MAX_CELL_TOKENS` at the entrypoint, closing the
  residual seam where a scored run could train short but evaluate long. Carries the
  existing comment's trigger ("Revisit if a sub-design budget ever appears in a
  SCORED bake-off run") into code.

## 5. Options and recommendation

All coverage counts recomputed from the 38-city YAML. "Union drop UB" = upper bound
on the union drop rate from per-city quantile bounds (drop <= 0.001 where
p99.9 <= B, <= 0.01 where p99 <= B, 0 where max <= B).

| option | window B | covers | cities still dropping anything | union drop UB | worst-case cost (d=256 / d=1024) |
|--------|---------|--------|-------------------------------|---------------|---------------------------|
| A | 10,240 | every city's **p99** (worst: barcelona 9,869) | 13 (4 above p99.9: valencia, barcelona, amsterdam, rotterdam) | 0.0013 | 2.9x / 2.5x |
| B (recommended) | 13,312 | every city's **p99.9** (worst: valencia 13,268) | 5 (almere, amsterdam, rotterdam, tilburg, valencia — max-tail only) | 0.0002 (~200 cells of 1.05M) | 4.7x / 3.8x |
| C | 16,384 | everything except valencia's max (19,372) | 1 (valencia) | ~0.0000 | 7.0x / 5.4x |

**Recommendation — the triple (Umar's to lock, not mine):**

- **`DEFAULT_MAX_CELL_TOKENS = ScaffoldConfig.max_len = 13,312`** (option B,
  p99.9-cover). This is the recorded 15.5 direction ("~p99.9 of the dense cities";
  the input range's upper end ~12–13k) made exact: the smallest window that puts
  every city's drop at or below the 0.1% tail-trim regime the 5760 budget was
  *designed* to be (it was the SG p99.9 — same semantics, now held EU-wide). The
  union drop falls from 0.92% (halt) to ~0.02%; the recorded per-city tail-drop is
  confined to five cities' extreme max-tails. Option A is materially cheaper (2.9x
  vs 4.7x worst-case) but leaves four cities dropping >0.1% — including barcelona
  and valencia, two of the most character-distinct cities in the union — and
  re-opens the "budget biases against dense cores" eval-validity argument this
  decision exists to close. Option C buys 49% more window for ~200 cells and trains
  its top positions on almost nothing.
- **`eval_max_new >= 13,312` in scored-run configs** (else the emergence verdict is
  refused as INCOMMENSURATE by construction; today's 512 default stays fine for
  smoke).
- **Commensurability resolved as §4** (one authoritative number; opt-downs legal,
  never scoreable; entrypoint assert added at execution).
- **`MAX_TOO_LONG_DROP_RATE = 0.005` unchanged** — its calibration ("5x the per-city
  p99.9 design point") carries over verbatim because B is again a p99.9-class budget.

**Execution obligations when the lock lands (recorded now, done at scored-run
planning execution — none started before Umar's word):**

1. Re-run `scripts/measure_cell_token_lengths.py` at B=13,312 (CPU, `lrd_all_serial`)
   to record the EXACT per-city tail-drop — the quantile bounds above are bounds, not
   the recorded numbers the 15.5 resolution requires.
2. ~~Memory probe at B on a real node before any scored run (§3 caution 1)~~ —
   **DONE 2026-06-11** (see Status block): flash engages, both ladder-end shapes fit.
3. The entrypoint assert (§4) + the lock-and-guards sweep: grep tests for assertions
   pinned to 5760 and update in the SAME commit as the constant change.
4. Shard rebuild question: **no shard rebuild is required** — verified at the code
   layer (`build_shards.py` never references the budget; the drop lives only in
   `flatten_shards_to_cells`, read-time, `if n > max_cell_tokens`). Residual runtime
   spot-check at execution: confirm one known >5760 cell flows from a stored shard
   into the example set at the new budget.

## 6. What this memo does NOT decide

GPU spend, run schedule, model scale ladder, and the shard-caching design (backlog
#2, separate memo). No code, config, or constant changes ship with this memo; the
tree still reads 5760/0.005/512 everywhere.
