# Phase 1 sub-project A — Overture loader design

- **Date:** 2026-05-16
- **Phase:** 1, sub-project A (data infrastructure)
- **Status:** Draft, pending user review
- **Owner:** umar

## 1. Goal

Stand up `cfm.data.overture` — a focused Python module that loads pinned Overture Maps release themes from public S3, scoped to a named region (Singapore for Phase 1), with cache management, schema validation, and a backend protocol that lets tests run without network. After A lands, downstream sub-projects (B1, C, ...) consume a `Region` object and never touch S3 or DuckDB directly.

**S3 access prerequisite:** the Overture S3 bucket `s3://overturemaps-us-west-2/` is **public-read with no credentials required** (per https://docs.overturemaps.org/getting-data/). The backend uses anonymous access — no AWS account or IAM role needed for CI or contributor machines. If Overture ever switches to authenticated access (extremely unlikely but possible), `S3DuckDBBackend` would need to accept credentials via env vars and the pinning policy doc would have to be updated to document the auth requirement. The slow opt-in test would surface this immediately by failing on `OvertureUnreachable`.

## 2. Scope (in / out)

**In scope for A:**

- `cfm.data.overture` package with a public API of one function (`load_region`) and one dataclass (`Region`).
- `OvertureBackend` protocol + two concrete implementations: `S3DuckDBBackend` (real) and `LocalFixtureBackend` (tests).
- Pinned release configuration at `configs/data/overture_release.yaml`.
- Region configuration at `configs/data/regions/singapore.yaml` (admin-polygon-based scoping with bbox fallback).
- Cache management with rich `manifest.yaml` per cached region.
- Pinning policy doc at `docs/data/overture_pinning_policy.md`.
- Fixture-generation script at `scripts/snapshot_overture_fixtures.py` (regenerates `tests/fixtures/overture_mini/` from a tiny real S3 fetch when re-pinning).
- Cache-invalidation CLI at `scripts/cfm_data_invalidate.py`.
- Tests run on CI without network using `LocalFixtureBackend`.

**Out of scope for A** (deferred to later sub-projects):

- Reprojection from EPSG:4326 to a local metric frame — that's C-stage (tile extraction).
- Tile partitioning, cell-clipping, boundary-contract derivation.
- Tokenisation.
- Frequency analysis on categorical fields — that's B1.
- Vocabulary derivation — that's B2.
- Sea masking — if admin-polygon scoping turns out impractical, sea is masked at C-stage using Overture's `base.water` theme.
- Multi-region orchestration. Singapore is enough for Phase 1; Sweden gets added in a small follow-up once Singapore works end to end.

## 3. Load-bearing decisions

1. **Admin-polygon scoping, not bbox.** Singapore's bbox includes a lot of open sea that would dilute downstream training signal. Loader fetches Singapore's admin boundary once from Overture's `divisions` theme, then uses it as a spatial filter for the other four themes (`buildings`, `places`, `transportation`, `base`). **Fallback:** if admin-polygon scoping proves harder than expected during implementation, fall back to bbox loading and explicitly sea-mask via `base.water` at C-stage. Either way, raw open sea does not enter the training set.
2. **Five themes, not four.** Phase 1 loads `divisions` in addition to the originally-proposed four (`buildings`, `places`, `transportation`, `base`). This is a direct consequence of decision 1: admin-polygon scoping requires the boundary geometry, which lives in `divisions`. Downstream sub-projects (B1, C) must expect a five-theme `Region` object; their schemas, manifests, and tests all account for this. If decision 1 is reversed (bbox fallback), `divisions` could in principle be dropped — but we keep it anyway because it carries country/locality labels useful for conditioning later.
3. **Pinned release: `2026-04-15.0`** (schema v1.16.0), confirmed via https://docs.overturemaps.org/release-calendar/ on 2026-05-16. We update the pin once per phase and never mid-phase. If frequency analysis (B1) reveals that 2026-04-15 has problems for our use case, we re-pin in one commit. The pin lives in `configs/data/overture_release.yaml` so every data-pipeline run is traceable to a commit + a release.
4. **Backend protocol for testability.** A `Protocol` interface with one method, `read_theme(theme, region, release) -> pyarrow.Table`. Real impl talks to `s3://overturemaps-us-west-2/release/<release>/theme=<theme>/`. Test impl reads from `tests/fixtures/overture_mini/<theme>.parquet`. The loader takes a backend instance via dependency injection; default factory returns the real backend.
5. **Cache layout under `data/cache/overture/`** (repo-local, gitignored). Per-release, per-region subdirectories. Each region has a rich `manifest.yaml` for full audit traceability.
6. **Fixture-generation snapshot script is the maintenance mechanism.** `scripts/snapshot_overture_fixtures.py` regenerates `tests/fixtures/overture_mini/` from a tiny real S3 fetch (e.g., a 0.01° × 0.01° bbox in central Singapore). The pinning policy doc instructs maintainers to re-run the script whenever they re-pin. A live schema-drift slow test against real S3 (`tests/slow/test_real_s3_opt_in.py`, §13) is an additional, opt-in verification — useful when investigating a suspected drift, but not part of the regular workflow.

## 4. Public API

```python
from cfm.data.overture import Region, load_region

singapore: Region = load_region("singapore")
# singapore.name             -> "singapore"
# singapore.release          -> "2026-04-15.0"
# singapore.themes           -> dict[str, pyarrow.Table]
#                               keys: "buildings", "places", "transportation", "base", "divisions"
# singapore.admin_polygon    -> shapely.Polygon  (the boundary used for clipping)
# singapore.manifest_path    -> Path to the cached manifest.yaml
```

`load_region` signature:

```python
def load_region(
    name: str,
    *,
    backend: OvertureBackend | None = None,   # default: S3DuckDBBackend()
    refresh: bool = False,                     # force re-fetch even if cache is valid
    confirm: bool = False,                     # required when pre-fetch estimate > 2 GB
) -> Region: ...
```

`confirm=False` is the default; small fetches proceed without it. `confirm=True` is required only when the §6 size estimator predicts more than 2 GB; otherwise `OversizedFetch` is raised. Singapore fits comfortably under the threshold, so day-to-day callers never need to pass `confirm`.

`Region` is a frozen dataclass; mutating it raises.

## 5. Backend protocol

```python
class OvertureBackend(Protocol):
    def read_theme(
        self,
        theme: str,
        scope: SpatialScope,       # admin polygon + optional bbox hint
        release: str,
    ) -> pyarrow.Table: ...

    def estimate_size(
        self,
        theme: str,
        scope: SpatialScope,
        release: str,
    ) -> SizeEstimate: ...          # row count + approx bytes; used for pre-fetch log
```

`SpatialScope` carries the admin polygon (preferred filter) and a bbox (used as a coarse DuckDB pre-filter for speed, then refined polygon-clipped in memory).

`S3DuckDBBackend` runs DuckDB queries against `s3://overturemaps-us-west-2/release/<release>/theme=<theme>/`. Query template per theme:

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SELECT *
FROM read_parquet('s3://.../release/<release>/theme=<theme>/*.parquet')
WHERE bbox.xmin <= ? AND bbox.xmax >= ?
  AND bbox.ymin <= ? AND bbox.ymax >= ?
  AND ST_Intersects(geometry, ST_GeomFromText(?));   -- admin polygon WKT
```

`LocalFixtureBackend` reads from `tests/fixtures/overture_mini/<theme>.parquet`, ignores `scope`, ignores `release`. Tests provide pre-shaped fixtures matching the expected schema.

## 6. Pre-fetch size estimation

Before fetching any theme, the loader calls `backend.estimate_size(...)`, which runs a cheap `COUNT(*)` query plus a small-sample average-row-size measurement. The loader logs:

```
[overture] estimated fetch: theme=buildings      rows≈340,000 size≈45 MB
[overture] estimated fetch: theme=places         rows≈12,400  size≈3.1 MB
[overture] estimated fetch: theme=transportation rows≈85,000  size≈22 MB
[overture] estimated fetch: theme=base           rows≈4,200   size≈1.8 MB
[overture] estimated fetch: theme=divisions      rows≈18      size≈64 KB
[overture] estimated total: 5 themes, ~72 MB
```

If the total exceeds **2 GB**, the loader prints a warning and requires `confirm=True` to proceed (the public `load_region` accepts a `confirm` parameter; default `False` means small fetches go through, large fetches abort with a clear message). For Singapore the total will be well under 2 GB.

The rationale is the user's: "Prevents accidental large fetches when someone runs `load_region('germany')` thinking it'll be quick."

## 7. Manifest contents

`data/cache/overture/<release>/<region>/manifest.yaml`:

```yaml
schema_version: 1                     # of this manifest format, not Overture's
release: "2026-04-15.0"
release_date: "2026-04-15"
release_subversion: 0
overture_schema_version: "v1.16.0"
region: singapore
scope:
  admin_polygon_source: "overture://divisions:country:SG"
  bbox: [103.6, 1.16, 104.05, 1.48]   # bounding box of the admin polygon, recorded for audit
backend: "S3DuckDBBackend"            # class name; helps debug fixture-vs-real divergence
fetched_at: "2026-05-16T14:32:11Z"    # ISO 8601 UTC
themes:
  buildings:
    s3_url: "s3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/"
    rows: 342156
    bytes: 47298304
    sha256: "f3a9b...c0e1"             # of the cached parquet file
    parquet_filename: "buildings.parquet"
  places:
    ...
  transportation:
    ...
  base:
    ...
  divisions:
    ...
```

Six fields per theme (s3_url, rows, bytes, sha256, parquet_filename) + per-region metadata (release with date and subversion split, scope, backend, fetched_at) — invaluable for debugging and reproducibility.

## 8. Cache layout

```
data/                                  # gitignored (already in .gitignore)
└── cache/
    └── overture/
        └── 2026-04-15.0/             # release subdir
            └── singapore/             # region subdir
                ├── manifest.yaml
                ├── buildings.parquet
                ├── places.parquet
                ├── transportation.parquet
                ├── base.parquet
                └── divisions.parquet
```

Cache-hit rule (deterministic, three cases):

1. **No manifest or release mismatch:** if `manifest.yaml` does not exist, **or** its `release` field does not match the currently-pinned release in `configs/data/overture_release.yaml`, the loader silently re-fetches all themes for the region. Rationale: the pin is the source of truth; a stale-release cache is exactly what we expect to find right after a re-pin, and silently fetching the correct release is the right behaviour.
2. **Release matches, sha256 mismatch:** if `manifest.yaml` exists with the right release, but any cached parquet's actual sha256 differs from what the manifest records, the loader raises `CacheCorrupt`. Rationale: the cache has been tampered with or partially written; silently re-fetching could mask a real problem.
3. **Release matches, sha256 matches:** cache hit. Return cached themes without touching the network.

`refresh=True` short-circuits this and unconditionally re-fetches (still writes a fresh manifest). The `refresh=False` default does not "fail loudly" on a missing/stale-release manifest; case 1 covers that.

## 9. Module layout

```
src/cfm/data/
├── __init__.py
└── overture/
    ├── __init__.py              # public API: load_region, Region
    ├── backend.py               # OvertureBackend protocol + S3DuckDBBackend + LocalFixtureBackend
    ├── loader.py                # high-level load_region(); cache logic; pre-fetch size log
    ├── region.py                # Region frozen dataclass; SpatialScope dataclass
    ├── manifest.py              # CacheManifest read/write; sha256 helpers
    ├── schema.py                # per-theme expected columns; validate_schema()
    └── errors.py                # OvertureUnreachable, OvertureSchemaMismatch,
                                  #   RegionNotFound, ReleaseNotConfigured, OversizedFetch

configs/data/
├── overture_release.yaml         # one key: release: "2026-04-15.0"
└── regions/
    └── singapore.yaml            # name, admin_source, fallback_bbox

docs/data/
└── overture_pinning_policy.md    # update once per phase; fixture regeneration procedure

scripts/
├── snapshot_overture_fixtures.py # regenerates tests/fixtures/overture_mini/
└── cfm_data_invalidate.py        # deletes a region's cache; forces re-fetch

tests/fixtures/overture_mini/     # ~10-row synthetic-but-real-schema parquets
├── buildings.parquet
├── places.parquet
├── transportation.parquet
├── base.parquet
└── divisions.parquet

tests/data/overture/
├── conftest.py                   # shared fixtures: fake backend, manifest paths
├── test_backend.py               # LocalFixtureBackend behaviour
├── test_loader.py                # cache logic, refresh, error paths
├── test_region.py                # Region frozen, SpatialScope construction
├── test_schema.py                # validate_schema raises on column drift
└── test_manifest.py              # round-trip, sha256, schema_version

tests/slow/
└── test_real_s3_opt_in.py        # @pytest.mark.slow; real S3 fetch of a tiny bbox
```

## 10. Errors

All in `cfm.data.overture.errors`. Single base class:

```python
class OvertureError(RuntimeError): ...

class OvertureUnreachable(OvertureError):
    """Backend cannot reach the data source."""

class OvertureSchemaMismatch(OvertureError):
    """A theme's columns do not match cfm.data.overture.schema expectations.
    Usually means Overture changed their schema between releases; bump the
    pin, re-snapshot fixtures, and update schema.py."""

class RegionNotFound(OvertureError):
    """No configs/data/regions/<name>.yaml found."""

class ReleaseNotConfigured(OvertureError):
    """configs/data/overture_release.yaml missing or malformed."""

class OversizedFetch(OvertureError):
    """Estimated download exceeds the 2 GB threshold and confirm=False."""

class CacheCorrupt(OvertureError):
    """Manifest exists but sha256 of a parquet doesn't match what manifest records."""
```

## 11. Cache-invalidation CLI

`scripts/cfm_data_invalidate.py`:

```
$ uv run python scripts/cfm_data_invalidate.py singapore
[overture] Removing data/cache/overture/2026-04-15.0/singapore/
[overture] Removed 5 parquet files + manifest.yaml. Total reclaimed: 47.3 MB.
```

Accepts optional `--release <version>` to invalidate a specific release rather than the currently pinned one. Refuses to delete anything outside `data/cache/overture/`. Uses `pathlib`; no shell calls.

## 12. Pinning policy (excerpt)

The full doc lands at `docs/data/overture_pinning_policy.md`. Key rules:

- The pin in `configs/data/overture_release.yaml` is the **single source of truth** for which Overture release every artifact derives from.
- We update the pin **once per phase**, never mid-phase. A re-pin is a single commit on a branch, reviewed, merged, and triggers fixture regeneration.
- Re-pinning procedure:
  1. Update `release` field in `configs/data/overture_release.yaml`.
  2. Run `uv run python scripts/snapshot_overture_fixtures.py` to refresh `tests/fixtures/overture_mini/`. Inspect the diff — schema changes in Overture will appear as column additions/removals here.
  3. If column changes are observed, update `src/cfm/data/overture/schema.py` to match, fixing affected tests.
  4. Optionally run `uv run python scripts/cfm_data_invalidate.py singapore` to force a re-fetch on next load.
  5. Commit the pin update + fixtures + schema in one logical change.
- The current pin is `2026-04-15.0` (Phase 1 starting pin, set 2026-05-16).

## 13. Tests

- **Fast suite (`uv run pytest`):** all `tests/data/overture/` tests use `LocalFixtureBackend`. No network. Covers: manifest round-trip, schema validation passes/fails, region loading, cache hit/miss, refresh, error paths, oversized-fetch guard.
- **Slow opt-in (`uv run pytest -m slow`):** `tests/slow/test_real_s3_opt_in.py` fetches a tiny real bbox from S3 (under 1 MB), asserts schema matches `schema.py`. Skipped by default.

## 14. Done criteria

A is done when:

- `uv run pytest` passes the data/overture tests using fixtures, **no network calls**.
- `uv run pytest -m slow tests/slow/test_real_s3_opt_in.py` passes on a machine with internet (run manually before sign-off, not on every commit).
- `uv run python -c "from cfm.data.overture import load_region; r = load_region('singapore'); print(r.release, list(r.themes), len(r.themes['buildings']))"` works after the first real fetch (manifest cached, sha256s recorded).
- `data/cache/overture/2026-04-15.0/singapore/manifest.yaml` matches §7 in form and content.
- The pinning policy is committed; a contributor can read it and re-pin without asking questions.
- All six new error classes (§10) have at least one test.

## 15. Risks specific to A

- **Overture changes their S3 path layout between releases.** Mitigation: backend URL is templated by release version; if 2026-04-15 uses a different path structure than the fixture-generation testing assumed, the slow-marked S3 test will catch it during first run.
- **Admin polygon for Singapore is multi-polygon (main island + offshore islands).** Mitigation: SpatialScope holds a `shapely.MultiPolygon | Polygon`; ST_Intersects handles either.
- **DuckDB spatial extension version drift.** Mitigation: pin `duckdb == 1.5.2` exactly in `pyproject.toml` (latest stable as of 2026-05-16 per pypi.org); also record the resolved version string in the manifest's `backend` field for full traceability. Re-pin DuckDB only on the same cadence as Overture re-pins.
- **Fixture parquet files become outdated when we re-pin.** Mitigation: §12 step 2 explicitly requires running the snapshot script on every re-pin. The schema-drift live test is the future-work backstop.

## 16. Out-of-scope items deferred to later sub-projects

- **B1** consumes Singapore's themes (especially `buildings.class`, `places.categories`, `transportation.class`, `base.subtype`) and produces the frequency-analysis report.
- **B2** turns B1's reviewed report into `configs/tokenizer/vocab_phase1.yaml`.
- **C** consumes the `Region` object, reprojects to a local metric frame, partitions into 2 km tiles + 250 m cells, clips features at cell boundaries.
- **D, E** build on C.
- **F** stitches per-cell output back into a tile.
- **G** is the end-to-end pipeline + validator.
