# Shard-derivation cache — DESIGN MEMO (2026-06-12)

**Status: DESIGN FOR UMAR'S REVIEW. No code change ships with this memo.**
Backlog #2 / resume-gate #4. Likely lands at scored-run-planning execution,
alongside (but in a separate commit from) the 13,312 budget-constant change.

## 1. The problem, precisely

Every training job start derives the full training corpus in memory:
`CellDataModule.setup` calls `build_shards_in_memory` per city, which walks every
validated tile reading FOUR sources — sub-D tile labels, sub-D
`macro_core.parquet` (cell-density buckets), sub-F `cells.parquet` (token
sequences), and sub-C features (building areas + road lengths, WKB-parsed, for
the Task-24b `character_stats`). Measured ~0.14 s/tile ⇒ **~40+ min for the
38-city EU union**. Three aggravations:

1. `setup` "Runs on ALL ranks" (`datamodule.py`) — a 4-GPU DDP job runs the walk
   4x concurrently. Wall-clock stays ~40 min (likely worse under I/O contention);
   the billed node idles its 4 A100s the whole time ⇒ **~2.7 GPU-h burned per job
   START, before step 0**.
2. F8 resume discipline means every resubmit (checkpoint resume, walltime slice,
   preemption) is a fresh start — each re-pays the walk.
3. The 40 min also sits in front of every FAILURE: a job that dies at step 1
   still costs 40 min of wall-clock before it can die.

**Value as a function of job-start count N (set at scored-run planning, so this
is a curve, not a number).** Cost without cache ≈ N x 2.7 GPU-h + N x 40 min
wall-clock serialized into iteration latency:

| N starts (runs + resumes + restarts) | GPU-h burned | vs ~1,250 GPU-h envelope |
|---|---|---|
| 10 (bare 3-backbone x 3-scale, no restarts) | ~27 | ~2% |
| 20 (one resume/restart each) | ~53 | ~4% |
| 40 (realistic with diagnostics + short jobs) | ~107 | ~9% |

The cache converts this to a ONE-TIME CPU build (`lrd_all_serial` Slurm job — no
GPU node idle) plus a seconds-class read at every job start.

## 2. What gets cached (and the layer boundary)

**Cached: the output of `build_shards_in_memory`, per city** — the
`TrainingShard` list: per-tile lineage + tile_conditioning, and per-cell
`(cell_i, cell_j, token_sequence, cell_density_bucket, character_stats[7])`.
This layer is deliberately upstream of every per-run knob:

**NOT cached — stays read-time at every setup (all cheap, all teeth stay live):**
- the **holdout audit** (`run_holdout_audit` + manifest sha/lock verification) —
  an audit is never cached, it runs live against the union every job;
- the **budget drop** (`flatten_shards_to_cells`, `len > max_cell_tokens`) — so
  the cache is **invariant to the 13,312 change** (shards carry full unfiltered
  cells; the locked budget decision does not invalidate it, and the 2048
  opt-downs read the same cache);
- prefix-id building (registry encoding, seed, `conditioning_ablation`) and the
  train/val split — per-run by design;
- the **city-identity and character-stats wiring guards** — re-run at READ on
  the loaded cells (they take the same inputs whether cells came from a walk or
  a cache; caching must not retire a tooth).

## 3. The cache key — every input that determines shard content

The real risk is a stale cache silently feeding a wrong-derivation training run.
The key must cover everything `build_shards_in_memory` reads or encodes:

1. **`release`** (e.g. 2026-04-15.0).
2. **Sorted city list** of the union.
3. **Per-city `training_manifest.yaml` sha256** — carries `tiles[]` (the
   authoritative inventory) and per-tile `provenance_sha256` (sub-D lineage).
4. **Per-tile source-file identity for EVERY file the walk reads**: sub-D labels,
   `macro_core.parquet`, sub-F `cells.parquet`, sub-C features parquet —
   recorded as (relative path, size, sha256) at cache-build time. The manifest
   sha (#3) does NOT transitively pin sub-C/sub-F bytes; these must be recorded
   directly (decompose-verification-debt: "the manifest covers it" is an
   unread-source assumption until verified, so the cache records its own).
5. **Derivation-version constant** `SHARD_CACHE_DERIVATION_VERSION` — bumped
   whenever derivation CODE changes meaning: `character_stats_for_cell` (the 7
   channels, `_CHAR_LOG10_CLIP`, presence-flag indices), `_tile_conditioning_dict`
   field mapping, `_cell_density_by_cell`, `CellPayload`/`TrainingShard` schema.
   Teeth in §5 make a silent un-bumped edit fail red.
6. **Conditioning scheme tag** (`value-char-v1`) — the payload shape is scheme-
   dependent (character_stats exists because of 24b).
7. **Cache format version** `SHARD_CACHE_SCHEMA_VERSION` (the serialization
   itself), separate from #5 (a format migration is not a derivation change).

Explicitly NOT key inputs (per §2): budget/max_len, seed, val_fraction,
ablation, registry sha, holdout manifest (audited live), batch/devices.

## 4. Format options + recommendation

- **Option A — no cache (status quo).** Zero new risk surface; pays §1 forever.
- **Option B — per-city parquet + union cache-manifest YAML (RECOMMENDED).**
  One `cells.parquet` per city (columns as §2; `character_stats` as
  fixed-size-list<float,7>; token_sequence as list<int>) + one small
  `tiles.parquet` (lineage + tile_conditioning), under
  `$WORK/.../training_cache/<release>/<city>/`; a union-level
  `cache_manifest.yaml` holding the FULL §3 key + per-file shas of the cache
  files themselves + `_SHARD_CACHE_VALID` marker. Parquet matches every existing
  reader idiom in the repo (pq.ParquetFile, byte-deterministic writers, no
  pickle). Per-city files mean a single-city invalidation doesn't force a
  38-city rebuild.
- **Option C — cache at the flattened-example layer (post-budget/prefix).**
  Faster still at read, but the key explodes to include budget, seed, ablation,
  registry — exactly the knobs that change between runs. Rejected.
- **Option D — pickle/torch.save of the shard list.** Simple, but no schema
  discipline, no partial verification, unsafe load. Rejected.

Storage: **$WORK, not $SCRATCH** — scratch is subject to periodic cleanup wipes;
a cache that can vanish mid-campaign re-introduces the 40-min tax at random.

## 5. Teeth (sha-on-read refusal, mirroring the floor-artifact/registry grammar)

1. **Verified-read-only construction**: `load_verified_shard_cache()` is the ONLY
   path to cached shards (proof-token grammar, mirroring
   `VerifiedFloorArtifact`). It recomputes the §3 key from LIVE inputs (re-reads
   manifest shas; re-hashes source files per the tier below), compares to
   `cache_manifest.yaml`, and on ANY mismatch raises `ShardCacheStale` **naming
   the differing component** — fail-closed; no silent rebuild, no silent
   fallback to the walk. Rebuild only via an explicit `--rebuild-shard-cache`.
2. **No marker without end-state verification**: the build writes files, then
   RE-READS them and compares against the in-memory derivation, then re-derives
   a seeded sample of k cells per city from SOURCE and compares exactly — only
   then writes `_SHARD_CACHE_VALID`. (The false-DONE class: a marker that trusts
   control flow poisons every later session.)
3. **Source-verification tier at read — MEASURED AND RESOLVED (2026-06-12,
   tier b′)**: the one-off measurements on the real 38-city union (88,076 files
   / 22,019 tiles x 4 / 5.88 GiB) settled this with two numbers:
   tier (a) full re-hash = **26.8 min** (job 46050613;
   `reports/2026-06-12-shard-cache-tier-a-cost.yaml`) — per-file open latency
   ~18 ms on Lustre dominates, not bytes — REJECTED; the originally proposed
   tier (b) "fully re-hash all small files" would open 71,355 of the 88,076
   files (~20 min by the same latency math) — ALSO REJECTED by the same
   measurement. **Tier (b′)**: stat (existence+size) on ALL files = **59 s**
   (job 46055141; `reports/2026-06-12-shard-cache-stat-walk-cost.yaml`) + a
   seeded sample of 8 files per city fully re-hashed (~seconds) ≈ **~1 min per
   job start**. Residual stated honestly in the code: a same-size single-file
   content edit is caught only when sampled; every realistic regen moves many
   files/sizes/manifests (caught by stat-all + manifest shas).
4. **Regime-distinguishing mutation tests** (a gate must distinguish regimes):
   one red test per key component — flip a manifest byte, bump a source file,
   edit `_CHAR_LOG10_CLIP` without bumping the derivation version (caught by a
   golden-fixture test: known inputs -> pinned cached bytes), change the scheme
   tag — each must produce a NAMED `ShardCacheStale`/red, not a pass.
5. **Lock-and-guards travel together**: the derivation-version bump and its
   golden-fixture update land in the SAME commit, enforced by the golden test
   being derivation-pinned.
6. **Byte-determinism**: build twice -> identical bytes (no timestamps; same
   discipline as the training manifests), so cache files are themselves
   sha-comparable across rebuilds and machines.
7. **Guards stay live** (§2): identity/character guards run on loaded cells at
   every read.

## 6. PI-CALLs (Umar's, at review)

1. **Verification tier at read** (§5.3): tier (b) now + measure tier (a), or
   insist on tier (a) regardless of cost?
2. **DDP read pattern**: all 4 ranks read the cache independently (simplest;
   seconds each), or rank-0-reads + broadcast? Provisional: all-ranks read —
   no new collective, no rank-0 special-casing (the save_checkpoint deadlock
   lesson argues against asymmetric rank behavior).
3. **Where the build lives**: a standalone `scripts/build_shard_cache.py`
   (lrd_all_serial, one job, per-city resumable) vs folding into the Task-8
   manifest builder. Provisional: standalone — the Task-8 builder is sealed and
   its scope is manifests, not payloads.
4. **Timing**: implement at scored-run-planning execution (with the budget
   constants, separate commits) vs earlier as its own small sub-project.

## 7. What this memo does NOT do

No code, no constants, no cache files exist after this memo. The 40-min walk
remains the live behavior until the design is approved and implemented behind
the usual TDD/red-first discipline.
