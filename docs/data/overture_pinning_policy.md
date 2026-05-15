# Overture release pinning policy

The Overture Maps S3 data is released on a roughly monthly cadence. Every Bonzai-OSM artifact (cache manifest, vocabulary, trained model checkpoint) must be traceable to one specific Overture release. This document is the rulebook.

## Source of truth

`configs/data/overture_release.yaml` is the **single source of truth** for which Overture release every data-pipeline run uses. The file pins:

- `release`: the release identifier, e.g. `"2026-04-15.0"`.
- `overture_schema_version`: e.g. `"v1.16.0"`.
- `release_date` and `release_subversion`: split for easy querying.

Currently pinned: `2026-04-15.0` (set 2026-05-16).

## Update cadence

**Once per phase. Never mid-phase.** A re-pin is a single commit, reviewed, merged. The cost of a mid-phase re-pin is: every cached region invalidates, every fixture parquet has to be regenerated, B1's frequency-analysis numbers shift. Don't.

## Re-pinning procedure

1. **Update the pin.** Change the four fields in `configs/data/overture_release.yaml` to the new release.
2. **Regenerate fixtures.**
   ```bash
   uv run python scripts/snapshot_overture_fixtures.py --mode s3
   ```
   This fetches a tiny real bbox in central Singapore from the newly-pinned release and overwrites `tests/fixtures/overture_mini/*.parquet`. If the schema changed, the diff will be visible in the new parquets.
3. **Update `schema.py` if columns changed.** The slow opt-in S3 test (`tests/slow/test_real_s3_opt_in.py`) is the canonical detector of column drift; if it fails after a re-pin, your task is to update `src/cfm/data/overture/schema.py` to match reality. Don't suppress the failure.
4. **Invalidate cached regions** if you want the next `load_region` call to fetch from the new release rather than rely on the silent-refetch path:
   ```bash
   uv run python scripts/cfm_data_invalidate.py singapore
   ```
5. **Commit.** Conventional commit prefix: `data:`. Message format:
   ```
   data: re-pin Overture to <release>
   ```
   Body should record what schema changes were observed (if any), what tests broke and were fixed.

## When the pin is wrong

If we discover (mid-implementation, mid-training, whenever) that the pinned release has a problem — corrupt geometry, missing themes, schema bug — re-pin once. Don't accumulate workarounds; the pin is cheap to move and the data is canonical.

## When Overture changes their S3 layout or auth

`s3://overturemaps-us-west-2/` is currently public-read with no credentials required (per https://docs.overturemaps.org/getting-data/). If Overture migrates to authenticated access or a different bucket, the `S3DuckDBBackend` in `src/cfm/data/overture/backend.py` must be updated and this document amended. The `OvertureUnreachable` exception is the most likely first symptom.

## When we want to read multiple releases simultaneously

We don't. One pin, one cache, one source of truth. If you need to compare releases, run on a branch with a different pin.
