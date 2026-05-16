# Known issues

A short, in-tree list of accepted-but-not-yet-fixed issues. Each entry says: where the issue is, why we accepted it, what blocks fixing, and when we have to fix it.

Add new entries on top. Remove entries when they're fixed.

---

## #1 — Cold-fetch of a fresh region takes ~8 hours

- **Filed:** 2026-05-16 (Phase 1 sub-A shipping checklist)
- **Severity:** medium (perf, not correctness)
- **Status:** deferred — **fix before adding Sweden as a region**
- **Affects:** `cfm.data.overture.load_region` cold path. Cache-hit path is unaffected (~1 s).

### Symptom

Calling `load_region("singapore")` against an empty cache against the pinned Overture release `2026-04-15.0` took **29,479.8 s (≈ 8.2 hours)** end to end on a normal home connection (2026-05-16 run). All five themes downloaded correctly; the manifest is valid; subsequent calls hit cache in ≈ 1 s.

### Root cause

`cfm.data.overture.loader._check_total_size` runs a `COUNT(*)` query against every theme via `S3DuckDBBackend.build_count_query` **before any read_theme call**. Each `COUNT(*)` scans the metadata of every parquet in the theme's S3 prefix (`s3://overturemaps-us-west-2/release/<release>/theme=<theme>/type=<type>/*`). For buildings/places/transportation that is hundreds of partitioned parquet files distributed globally. DuckDB has to open each one to read its row-group bbox stats before it can prove the file is outside Singapore. With httpfs latency this is the slow path.

The actual data reads (the `read_theme` calls after the COUNT phase) are the smaller portion of total time.

### Planned fix

Push the Singapore bbox into Overture's partition selection so that DuckDB only opens parquets that geographically cover Singapore, not the whole world. Overture's theme directories use coarse spatial partitioning (Hilbert-style); the right glob or a manual partition prune should reduce the metadata-scan workload by 1–2 orders of magnitude.

Concretely, three candidates worth trying in order:

1. **Skip or stub the COUNT pre-estimate.** Use a static heuristic per theme + bbox area for the `OversizedFetch` guard. Cheapest change; loses the precise size print but keeps the safety threshold.
2. **Glob the partition layer directly.** Replace `theme=<theme>/type=<type>/*` with a path that limits to relevant geographic partitions. Requires inspecting Overture's actual partition layout for the pinned release.
3. **Stream-and-write batches.** Skip materialising a `pyarrow.Table` per theme; stream `pq.write_table` from the DuckDB record-batch reader so we never hold a full theme in RAM. Orthogonal to the COUNT issue but worth doing while we're in there.

### Effort estimate

Half a day of work + verification (re-run a real cold fetch against Singapore and confirm wall-clock drops below an hour). Not a multi-day fix.

### Why we're not fixing it now

Phase 1 sub-A's contract is verified end-to-end. Phase 1 sub-projects B1–G read from the cache, never the cold path. The next time the cold path matters is when we add Sweden as a second region — at that point fixing this is a hard prerequisite, not optional.

### Tracking

- Source: `src/cfm/data/overture/loader.py::_check_total_size` and `src/cfm/data/overture/backend.py::S3DuckDBBackend.build_count_query`.
- Project memory: `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/project_overture_cold_fetch_slow.md`.
- Pinning policy reminder (`docs/data/overture_pinning_policy.md`) says re-pinning invalidates caches — re-pinning Singapore today would re-incur this 8-hour cost.
