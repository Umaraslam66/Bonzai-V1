# Corpus scale-up scoping (42 cities / ~624M tokens → ~10B tokens)

Plan + estimate only — **not** authorization to build. Every number is sourced to
`docs/GROUND_TRUTH.md` (GT), code, or a measurement noted inline. Date: 2026-06-23.

## 1. Pipeline to add cities (scripts, order, automation)

The existing multiregion pipeline (it built the current 42-city corpus) has five stages:

| # | Stage | Entry point | Where | Automated? |
|---|---|---|---|---|
| 1 | **Config gen** (bbox/country/morphology per city) | `scripts/multiregion/build_*_configs.py` → `configs/multiregion/*.yaml` | local | **manual** (hand-curated city list) |
| 2 | **Fetch** Overture → cache | `load_region` loop in tmux → `data/cache/overture/` | **login node** (S3 egress) | semi-manual (tmux loop) |
| 3 | **Process**: sub-C features → sub-D macro → **sub-F encode** → sub-G validate | `scripts/extract_region_batch.py --cities <city>` via `scripts/multiregion_process.sbatch` | `lrd_all_serial` (8 cpu / 30 G / **4 h cap**) | automated per city, **serialised** (per-user cpu=8 cap → one city at a time); fanned out by `gated_addcities.sh` / `fanout_remaining_addcities.sh` |
| 4 | **Build training shards** (union per-city tiles) | `scripts/build_multiregion_train_shards.py` (+ `.sbatch`) | Slurm | automated |
| 5 | **Build shard cache** | `scripts/build_shard_cache.py` (+ `build_shard_cache_v2.sbatch`) | serial Slurm (**never login** — SIGKILL@26 min) | automated |

Bottlenecks: **config gen is manual per batch**; **fetch** is semi-manual on the login node;
process/shards/cache are scripted but **gated by the serial-partition cap**.

## 2. How many cities / tiles ≈ 10B tokens

Derived from GT §3 (`TRAIN_TOKENS = 623_900_790`; 42 corpus cities per `rederive_project.py`;
measured avg **cell-body ≈ 488 tok**; held-out **94,520 cells / 1,952 tiles** → ~48 cells/tile):

| quantity | current (42 cities) | per city | **≈10B target** |
|---|---|---|---|
| tokens | 623.9 M | 14.86 M | 10,000 M (**×16.0**) |
| cells | ~1.28 M (÷488) | ~30,400 | **~20.5 M** |
| tiles | ~26,600 (~48 cells/tile) | ~633 | **~423,000** |
| cities | 42 | — | **~673 total (~631 new)** |

So **~16×** = **~631 new EU cities**, **~400k new tiles**, **~20M cells**. (Units: tokens / cells /
tiles / cities are distinct.)

## 3. Wall-clock + compute + storage for the data build

- **Fetch (Overture):** cold fetch is **hours per city** (singapore ~8 h cold, EU smaller — memory
  `project_overture_cold_fetch_slow`), login-node only, serial-ish. 631 cities → **hundreds–low
  thousands of login-node hours + large S3 egress**. Hard to parallelize (no creds on compute nodes).
- **Process (sub-C→sub-F→validate):** ~91 min / 253 tiles @ 8 cpu (prague canary). Avg city ~633
  tiles → **~2–4 h/city** (large cities brush the **4 h serial cap** — a splitting landmine). Two paths:
  - `lrd_all_serial` (**budget-free, serialised**): 631 × ~2.5 h ≈ **~1,600 h ≈ ~9 weeks** wall-clock.
  - `dcgp_usr_prod` (**parallel, billed**): ≈ 631 × 2.5 h × 8 core ≈ **~12,600 core-h ≈ ~31% of the
    40,000 core-h grant** (GT §1) — and that grant is meant for *training*.
- **Shard cache build:** ~50 min cold @624M (memory `leonardo_shard_cache_rebuild_ops`) → **~13 h
  @10B** in one job → **exceeds the 4 h serial cap**; needs splitting / incremental cache.
- **Storage (measured on Leonardo, read-only `du`, 2026-06-23):**

  | component | current (42 cities) | per city | **≈10B (×16)** |
  |---|---|---|---|
  | `data/processed` total | **11 GB** | 0.26 GB | ~176 GB |
  | ├ `sub_c` (feature WKB geometry — bulk) | 6.6 GB | | ~106 GB |
  | ├ `sub_f` (tokens) | 1.3 GB | | ~21 GB |
  | ├ `sub_d` (macro plan) | 667 MB | | ~11 GB |
  | └ `training_cache` (shard cache) | 1.5 GB | | ~24 GB |
  | `data/cache/overture` (raw Overture) | 7.9 GB | 0.19 GB | ~126 GB |
  | **total `data/`** | **19 GB** | **0.45 GB** | **~300 GB** |

## 4. Landmines at ~16× scale

1. **222-tree durability (most serious).** GT §2: the 222 *compute* allocation **expired 2026-06-11**;
   the FS tree is "**at risk post-expiry**." This work also hit **transient Lustre EIO** on the 222
   tree (it killed two eval jobs on 2026-06-23). ~300 GB on an at-risk, expired-allocation
   filesystem is a real data-loss exposure — **pick a durable home (548 tree / object storage) before
   building.**
2. **Cache-build time vs the 4 h serial cap.** ~13 h cache rebuild vs a 4 h cap; the cache is
   write-once-seal → a full rebuild. Needs an incremental / sharded cache design.
3. **Build wall-clock vs budget tradeoff.** ~9 weeks free-serial **or** ~31% of the grant on dcgp.
   Neither is free; the serial default is *weeks*.
4. **Dedup / quality at scale.** More degraded-source cities (rotterdam/warsaw class, known-issue
   #20); **no cross-city dedup** today (overlapping/near-duplicate generic geometry inflates tokens
   without information); the sub-G validator must clear **~423k tiles** (16× the gate volume). Quality
   control at this scale is unproven.
5. **(Implication, not data-build) model + training cost.** 10B tokens at Chinchilla ~20 tok/param
   implies a **~500M-param model** (~10× the 53M bake-off); training on 10B tokens would **far exceed
   the 5,000 GPU-h grant** (GT §1). The data build is the cheap half — flagged so the corpus target
   is not scoped without the training cost.
