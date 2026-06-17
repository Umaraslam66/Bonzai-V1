# T9 diagnostic RUNNING — handoff (2026-06-17)

The bake-off backbone build (T1–T8) is DONE on branch `phase-2-bakeoff-2backbone` (unmerged).
The **T9 diagnostic is RUNNING** on Leonardo; this handoff carries the scale reconciliation
and the budget-gate requirements so T9→T10 can't drift.

## STATE
- **T9 diagnostic: job `47143523`** (4×A100, `boost_qos_lprod`, account 548), commit `75bfb1d`,
  `--train-set eu-train-union` (the corpus the scored runs train on), `REGION=krakow` (held-out
  EU floor), `--no-compile`, `--max-time 00:07:30:00` (graceful stop → the eval RUNS → captures
  `gen_seconds_per_token` + the report; r is read from the loss CSV up to the stop). DO NOT kill it.
- Bake-off data is EU multiregion (`eu-train-union`), NOT singapore — see
  `docs/handoffs/2026-06-09-start-of-bakeoff.md` + the memory `project_bakeoff_data_is_eu_multiregion`.

## ⚠️ SCALE RECONCILIATION — r is measured at the 100M scale (read before T10)
geometry-`r` is **scale-dependent**, so r only sizes the budget validly at the scored scale.
The scored T10 scales are the param-matched table `{30/100/300M/1B}` (`src/cfm/models/bakeoff_scales.py`).

- **The diagnostic's mixer `d_model 768 / 12L / 12H` is IDENTICAL to the table's `100M`
  transformer-ar config** (`_TRANSFORMER_AR["100M"] = {768,12,12}`). The "90M" label is the
  diagnostic at `--max-len 2048` (pos-embedding-light: ~90M vs the scored 100M's 98M at
  `max_len 13322`; the ~8M delta is purely the larger positional table, NOT the mixer). r
  depends on the MIXER → **r is measured AT the 100M scored scale** (outcome 1 for 100M).
- **For 30M / 300M / 1B: r is applied as scale-invariant** (Chinchilla tokens-per-param
  ~constant). This is the spec's **RECORDED FLAG 1** (in `bakeoff_diagnostic.sbatch`):
  *"geometry-r is MEASURED-AT-90M; applying it to size the 1B budget is an ASSUMPTION, not a
  measured fact… a 300M check would bound the scale-dependence… do NOT let the 90M r silently
  become the 1B r."* **1B-r is the least-certain extrapolation.**
- **Verdict: NOT a silent mismatch.** r is valid for 100M directly; an explicit, spec-recorded
  assumption for the other scales. The T9 budget gate MUST carry this caveat into T10 (and a
  300M r-check is the budgeted way to bound it if the ladder leans on 300M/1B).

## T9 BUDGET GATE — what it must SHOW before ANY T10 commit (Umar's word)
1. **Budget arithmetic with REAL inputs:** region-agnostic per-token unit cost
   (`gen_seconds_per_token` from the diagnostic eval) × **floor-regime `eval_max_new` ≈ 13,312**
   (NOT the diagnostic's 2048, NOT the smoke's 64) × **summed REAL per-city held-out workload
   over ALL 4 cities** (usable_n 523/579/156/601 — glasgow/eisenhuttenstadt/munich/krakow; munich
   included, NOT krakow×4) × 2 backbones × feasible scales × seed-repeats. Show the numbers.
2. **geometry-`r` read from the UNION loss curve** (the diagnostic CSV). If loss hasn't flattened
   by the graceful stop → **ladder-collapse → `FIXED_SCALE_PLUS_S13`**, emitted as a verdict,
   never a silent garbage-r. r is valid for 100M; scale-invariance caveat (above) for the rest.
3. **STOP with the projected budget vs the 5000 node-h allocation.** Expect a tighter feasible
   ladder than the 2048-sized estimate (eval dominates, §6); the joint `feasible_ladder` ∩
   eval-budget bound may drop 1B / the top rung.

## CARRY-FORWARDS to T10 (the scored matrix)
- **`--no-compile` for scored runs** (T7 §5 verdict: recompiles 22>10, no plateau, overhead 62.6%;
  torch.compile graph-breaks on selective_scan/causal_conv1d). Add `--no-compile` to
  `bakeoff_run.sbatch`. Both backbones eager for commensurability.
- **`bakeoff_run.sbatch` buildability dry-run must inject `'region': '${REGION}'`** (mirroring the
  `'backbone': '${BACKBONE}'` override) — item-3 made `ScaffoldConfig.region` REQUIRED, so the
  preamble `ScaffoldConfig(**loaded)` dry-run fails unless the per-run YAML carries region or the
  env region is injected. Do this when adding `--no-compile`.
- **Obligations (b)/(c)** (start-of-bakeoff handoff §1): define `model_vs_real_effect` (with its own
  teeth: a conditioning-echoing model must FAIL it) + the munich→manchester power-gate swap reserve
  + the EU-train-split resolved-gap recompute. These fire at the scored/decision stage (T10).
- Per-run YAMLs `configs/experiments/bakeoff-{backbone}-{scale}.yaml` don't exist yet — emit them at
  T10 from the param-match table + the feasible ladder; each MUST carry `region`.

## DONE / VERIFIED (this session, branch `phase-2-bakeoff-2backbone`)
- T1 sbatch gcc; T2 golden; T3 ScaffoldBackbone (bit-identical); T4 MambaHybrid 7:1 (+pre-norm
  residual wrapper — raw Mamba has none); T5 param-match ≤2% (re-derived independently); T6
  build_backbone mamba branch; T7 mamba smoke PASS + `--no-compile` verdict; T8 per-token gen cost.
- Unified Leonardo `.venv`: torch 2.5.1+cu121 (untouched) + mamba-ssm 2.3.1 + causal-conv1d
  1.6.2.post1 + einops/hf_hub/transformers/ninja (`--no-deps`); env-lock passes.
- item-3: `ScaffoldConfig.region` REQUIRED (fail-closed; `dbdf3d5`) — full-suite green verification
  in flight (job-less login-node pytest).
- Deploy: git bundle to the SHARED `/leonardo_work/AIFAC_P02_222/<name>.bundle` (NOT `/tmp` —
  per-login-node, dies on socket/node switch). Compute=548, repo/data/venv on the 222 tree (RW),
  `/leonardo_work/AIFAC_P02_548/Bonzai-OSM` is an empty stub.
