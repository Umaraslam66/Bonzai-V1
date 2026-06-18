# LIVE BOOT DOC — Phase-2 bake-off: 50M locked, decision+sharding+sbatch DONE, GPU work `$WORK`-blocked (2026-06-18)

This is the current boot doc for the Phase-2 bake-off. It supersedes the archived
`docs/handoffs/archive/2026-06-17-t9-diagnostic-running.md` (+ the older bake-off start/resume
chain in `archive/`). Branch **`phase-2-bakeoff-2backbone`**, UNMERGED. The substantive session
work ends at commit **`350f373`** (§4); the branch **tip is this doc's own refresh commit** — the
latest commit per `git log --oneline -1` is a docs/handoff update, NOT a code change, so a tip
that is a `docs(handoff)` commit on top of `350f373` is expected, not drift.
**Do NOT merge and do NOT submit scored runs without Umar's word.**

## 0. CONSULT FIRST (in this order)
1. **`docs/GROUND_TRUTH.md`** — the canonical facts (compute units, account, corpus, locked
   design), each with a source pointer. **If anything below disagrees with GROUND_TRUTH, that
   file wins.** Always state the unit (node-h vs GPU-h — they differ by 4×).
2. **Your memory** (`MEMORY.md` is loaded each session; line 1 points here). Recall the
   `feedback_*` lessons before acting — esp. `feedback_no_marker_without_endstate_verify`,
   `feedback_tool_output_trustworthiness_layer`, `feedback_verify_before_lock_not_after`,
   `feedback_lock_and_guards_travel_together`. The `project_bakeoff_50m_locked` entry is the
   one-line state.
3. **Spec** `docs/superpowers/specs/2026-06-17-phase-2-bakeoff-2backbone-delta-design.md` §1A
   (locked decision), §9 (seeds/two-floor rule). **Plan** `docs/superpowers/plans/2026-06-17-phase-2-bakeoff-2backbone.md`
   Task 9 (gate, DONE), Task 10 (scored matrix, GATED), Task 11 (eval-sharding).
4. **`docs/protocols/sub-project-planning-protocol-v3.md`** before any new brainstorm.

## 1. THE LOCKED DECISION (Umar's word, 2026-06-18)
Phase 2 is a **SINGLE fixed-scale bake-off, NOT a scaling curve.** One rung **N = 50M**, two
backbones **`transformer-ar` vs `mamba-hybrid`** (7:1 Jamba), **`--no-compile`** both, **3 seeds**
per backbone, **4-GPU eval-sharding**. No ladder; decision basis = fixed-scale by choice
(`FIXED_SCALE_PLUS_S13` family). Matrix = **2 bb × {50M} × 3 seeds = 6 runs**.
**Why 50M:** Chinchilla ≈20 tok/param → optimum ≈30M for ~624M unique EU tokens; 100M was the
measured diagnostic rung; 50M = the chosen middle (r≈20 → ~1B tokens → ~1.6× reuse, inside the
safe ≤~4× band → nearly fully-cooked, more meaningful than 30M).

## 2. T9 DIAGNOSTIC — VERIFIED CLEAN from artifacts (job `47143523`, COMPLETED 0:0, 3:15:45)
- `.err`: `Trainer.fit stopped: max_steps=110000 reached` → full horizon, not an early stop;
  fit 8082.8s + eval 3510.3s ≈ elapsed. **Eval RAN** (the watched failure did not happen).
- Report `reports/phase-1-training-scaffold/2026-04-15.0-krakow-transformer-ar-89M-seed7-loop-closed.md`:
  **`gen_seconds_per_token = 0.026779`** (single-GPU, @100M mixer), decodability 0.983,
  OGC-valid 0.989, right-angle 0.9985. `emergence_verdict: INCOMMENSURATE` = the expected
  2048<13312 gap (F15), not a failure.
- Loss `reports/logs/training-scaffold/version_21/metrics.csv`: mean train_loss 2.60→2.094;
  last 30k steps ~0.026 nats → **FLATTENED** at r≈40 (via ~5.8× reuse). val_loss not logged.

## 3. BUDGET (units canon = GROUND_TRUTH §1; grant = 5,000 GPU-h = 1,250 node-h = 40,000 core-h)
Single-GPU 0.026779 s/tok × 13,312 full-cap × Σ 1,859 held-out cells (523/579/156/601 =
glasgow/eisenhüttenstadt/munich/krakow) × 2 backbones. Per-seed eval ≈ 336 node-h wall-clock
(tf 233 + mamba 103, incl. tf full-ctx ×2.7 / mamba ×1.2 correction):

| seeds | AS-IS rank-0 eval | 4-GPU eval-sharding |
|---|---|---|
| 1 | 336 node-h = 1,344 GPU-h (27%) | 84 node-h = 336 GPU-h (7%) |
| 2 | 672 node-h = 2,688 GPU-h (54%) | 168 node-h = 672 GPU-h (13%) |
| **3 (LOCKED)** | 1,008 node-h = 4,032 GPU-h (**81%**) | 252 node-h = **1,008 GPU-h (20%)** |

The full 4-rung ladder was ~22,000 GPU-h (4.4× over) → drove the single-scale lock. AS-IS 3 seeds
= 81% of the whole grant; **sharding (~4×) makes it 20%** → that is why sharding is locked, not
optional. Training (r≈20 ≈1B tokens) is ~<15 node-h (DDP, no waste).

## 4. DONE THIS SESSION (committed on `phase-2-bakeoff-2backbone`; base was `9a450f4`)
- **`a8051df`** feat: eval-sharding CORE — `src/cfm/eval/shard.py` (torch-free: `partition_indices`
  ragged-safe round-robin, `assert_conservation`, `gather_in_order` canonical/order-independent)
  + `tests/eval/test_shard.py` (51 local tests, mutation-verified).
- **`bfe5d34`** docs: locked 50M + 3-seed/sharding/two-floor; added **`docs/GROUND_TRUTH.md`**;
  archived 6 superseded bake-off handoffs to `docs/handoffs/archive/`; spec §1A/§9 + plan T10/T11.
- **`fb2b4c7`** feat: **NO_DECISIVE_WINNER two-floor verdict IMPLEMENTED** (was only documented).
  `city_aggregate.binding_city_verdict` now returns `BindingVerdict | NoDecisiveWinner`; a city is
  DECISIVE only if gap > `max(C/√n resolution, seed-noise reproducibility)`; else the named
  `NoDecisiveWinner` (S13, `basis=FIXED_SCALE_PLUS_S13`). `PerCityKS.seed_sem` (default 0 → legacy
  C/√n). `pick_winner` raises the named `NoDecisiveWinnerRefusal` (→ §13). 3-band tests
  (DECISIVE/LUCK/MIDDLE) + consumer refusal; mutation-verified.
- **`11374c7`** feat: `shard.sharded_eval` (the `all_gather_object` wrapper; lazy torch.distributed,
  module stays torch-free) + `indices_for_rank`; **the GPU equivalence golden authored**
  (`scripts/eval_sharding_golden.py` + `.sbatch`), execution-deferred.
- **`350f373`** chore: `bakeoff_run.sbatch` carry-forwards — **`--no-compile` on the scored srun**
  (both backbones) + **`'region': '${REGION}'` injected into the dry-run** with a symmetric
  `CONFIG_REGION_MISMATCH` refusal; new text-only contract test locks it.

Local verification (no torch on the Mac): `tests/eval/test_{shard,city_aggregate,bakeoff_decision}.py`
= 93 passed; ruff clean; sbatch `bash -n` OK; golden `py_compile` OK.

## 5. DEFERRED — `$WORK`-blocked, but now CODE-COMPLETE (just submit when `$WORK` heals)
Two items remain before any scored run; both gated on `$WORK` heavy-I/O recovery. Bundle them.
1. **50M param-match on ACTUAL built counts** (full T5, never eyeball): extend
   `scripts/tune_bakeoff_scales.py` with a 50M seed (provisional: tf `d640/8L/10H`≈50,219,748
   +0.44% HIGH-confidence; mamba `d640/~14L/every7` SEED, search n∈{13,14,15}), run on the unified
   Leonardo env (construction is CPU-safe — login node OK once FS healthy), confirm tf vs mamba
   **≤2%**, **append** the verified pair to `src/cfm/models/bakeoff_scales.py` (append-only;
   existing {30/100/300M/1B} rungs stay), pass `tests/models/test_bakeoff_param_match.py` at 50M.
   **Do NOT lock the provisional analytic numbers.** Scratch helper `scripts/_derive_50m.sh`
   (untracked, on Mac + Leonardo) shows the exact env+approach — delete after.
2. **Eval-sharding GPU equivalence golden**: `sbatch scripts/eval_sharding_golden.sbatch`
   (torchrun, 4 ranks). Two teeth: (1) per-cell tokens 4-GPU-sharded == rank-0 baseline
   bit-identical; (2) ragged-city (523) count-conservation. Writes
   `reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS` only if green. Must PASS before scored runs.
3. **`--test-only` on `bakeoff_run.sbatch`** (deferred only because it pages `$WORK`/torch):
   `sbatch --test-only scripts/bakeoff_run.sbatch` once healthy; also the torch-needing
   `tests/training/test_cli_contract.py` tests (the ones that import `train_scaffold`) run there.

Then **STOP for Umar** before T10 (emit `configs/experiments/bakeoff-{transformer-ar,mamba-hybrid}-50M.yaml`
from the verified table — each carrying `region` + r-derived `max_steps` + `seeds:[…3…]`) and
before any merge.

## 6. `$WORK` STATUS (CINECA storage incident, 2026-06-18)
CINECA reported a `$WORK` filesystem issue (slow/unresponsive I/O; some compute nodes removed;
"avoid `$WORK` until further notice"). As of **~14:20 GMT+2**: light I/O RECOVERED (`ls`,
small write/read OK on the 222 tree) but **heavy `torch` import (~1 GB paging) still did NOT
complete within a 120s cap → heavy `$WORK` ops are STILL DEGRADED.** Do NOT run the deferred
GPU/param-match items yet. **Re-probe before assuming heavy ops work:**
```
ssh leonardo 'cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM && timeout 120 bash -lc \
  "module load python/3.11.7; source .venv/bin/activate; python -c \"import torch; print(torch.__version__)\""'
```
TORCH_OK in well under 120s → heavy ops are back. Hang/timeout → still degraded; do `$WORK`-free
work only. (Diagnostic artifacts already on disk are intact; do not pound the FS during the incident.)

## 7. CAVEATS / RECURRING TRAPS (do not relearn these the hard way)
- **node-h vs GPU-h**: 1 node = 4 GPU = 32 core. The "5,000" ceiling is GPU-h. Always label.
- **Account/storage split**: compute = `AIFAC_P02_548`; repo+data+venv on the **222 tree**
  (`/leonardo_work/AIFAC_P02_222`); 222 compute expired 2026-06-11 (FS durability risk); 548 tree
  is an empty stub. Deploy = git bundle to the shared 222 path (no GitHub creds on login).
- **singapore is HISTORICAL**; bake-off data is `eu-train-union`; `ScaffoldConfig.region` is
  REQUIRED (fail-closed, no default).
- **`bakeoff_scales.py` is the lock surface** — never write provisional/eyeballed param counts
  into it; only actual-built-count-verified numbers.
- **Markers must follow proven end-state** (F8): `_SHARDING_GOLDEN_PASS` / `JOB_DONE` only after
  on-disk verification.
- **Do NOT solo `model_vs_real_effect` (§7)** — it's a brainstorm with Umar (a conditioning-echoing
  model must FAIL it). Obligations (b)/(c): munich→manchester power-gate reserve + EU-train-split
  resolved-gap recompute fire at the scored/decision stage.
