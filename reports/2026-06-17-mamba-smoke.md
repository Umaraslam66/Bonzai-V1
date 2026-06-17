# mamba-hybrid non-scored GPU smoke — verdict (2026-06-17, Task 7)

**Job 47140784** (4×A100, `boost_qos_dbg`, account `AIFAC_P02_548`), commit `e7bc2581`,
elapsed 19m35s, exit 0:0. Data: **EU multiregion `eu-train-union`** (sealed cache, 38 train
cities, held-out excluded), `REGION=krakow` (held-out EU city, the emergence-floor tag).
30M-param-matched mamba-hybrid shape (d_model 512 / 12 layers / 8 heads = 24.7M),
**compiled**, `--max-len 2048 --max-steps 100 --eval-cells 4 --eval-max-new 64` (non-scored).

This is the pre-scored gate (spec §3.5): mamba-hybrid must train, eval cleanly (the first
real check of the gcc/compile eval-path on the mamba backbone), and the §5 compile-stability
decision must be made from data.

## The three teeth

### 1. Trains — ✅ PASS
100 steps completed (`fit_seconds=472.9`, `steps_per_sec=0.211`, finite grads, loss
decreasing). The mamba-hybrid backbone trains under DDP on 4×A100 on the EU union, compiled.

### 2. Evals (generate → decode → score) — ✅ PASS
Post-train generation on the mamba backbone completed cleanly **on the GPU**:
`decodability_rate=1.0` (14/14 blocks decoded), `ogc_valid_rate=0.917`, building tokens
emitted in 2/4 cells, `gen_seconds_per_token=0.181`. `emergence_verdict=INCOMMENSURATE` —
HONEST by construction (smoke `eval_max_new=64` « the 13,312 floor regime; not a defect).

This verifies the **eval-path fix on mamba**. Two bugs the smoke caught and that are now
fixed (commits in this branch):
- the EU-data re-point (singapore was the stale `ScaffoldConfig` default; bake-off data is
  the EU multiregion `eu-train-union`);
- **post-fit eval ran on CPU** (Lightning DDP teardown moves the model to CPU; transformer-ar
  tolerated CPU generation, mamba's `causal_conv1d`/`selective_scan` kernels are CUDA-only and
  raised). Fix: re-place the model on the GPU before the rank-0 eval.

### 3. Compile stability (§5) — measured → **VERDICT: `--no-compile` for scored runs**
`reports/2026-06-17-mamba-smoke-compile.json` (250-step window over the real EU
variable-length cell batches):

| metric | value | §5 threshold | pass? |
|---|---|---|---|
| recompiles | **22** | ≤ ~10 | ✗ |
| recompiles plateaued (none in 2nd half) | **false** | true | ✗ |
| compile overhead fraction | **0.626** | < 0.10 | ✗ |
| dynamo unique graphs | 19 | — | — |

**Both teeth fail decisively.** Root cause (from the run's stderr): `torch.compile`
**graph-breaks on `selective_scan_cuda` / `causal_conv1d`** — the mamba kernels are
opaque C++ extensions dynamo cannot trace — and the per-batch-max variable cell lengths
cause recompilation that never plateaus (62.6% of wall-clock is compile). Compile gives
mamba little benefit (it breaks around exactly the expensive kernels) and large cost.

## Decision (carry-forward to Task 10 — scored runs)
**Scored runs use `--no-compile`** (the §5 pre-stated rule: keep compile ON iff recompiles
plateau ≤~10 AND overhead <10%; both fail). For bake-off **commensurability** (architecture
the only variable), run BOTH backbones `--no-compile` rather than compile-one/eager-other.
`bakeoff_run.sbatch` must add `--no-compile` at Task 10. (The §5 measurement is on mamba;
transformer-ar compile was not separately measured — moot under the commensurability call.)

The diagnostic (Task 9, transformer-ar, already `--no-compile`) is unaffected.
