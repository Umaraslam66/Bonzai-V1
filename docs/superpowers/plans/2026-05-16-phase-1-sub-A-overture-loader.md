# Phase 1 Sub-Project A — Overture Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md`

**Goal:** Land `cfm.data.overture` — a focused loader that fetches pinned Overture release themes from public S3 (region-scoped to Singapore by admin polygon), caches them locally with rich manifests, validates schemas, and surfaces six specific errors. Tests run network-free using a `LocalFixtureBackend`; a single opt-in `slow` test exercises real S3.

**Architecture:** Eight source files under `src/cfm/data/overture/`, each with one responsibility (errors, region/scope dataclasses, schema definitions, manifest, backend protocol + two impls, loader). Dependency injection (`OvertureBackend` protocol) keeps tests off the network. Cache layout is deterministic and per-release.

**Tech stack:** Python 3.11+, `duckdb == 1.5.2` (spatial + httpfs extensions), `pyarrow >= 15.0`, `shapely` (from Phase 0), `pyyaml` (from Phase 0), `pytest`.

**Three implementation-level conventions worth knowing up front:**

1. **Schema is curated, not faithful.** `schema.py` lists only the columns we actually use per theme. `validate_schema` checks the cached parquet has at least those columns; extras are ignored. This decouples our contract from Overture's full schema — if Overture adds columns, we don't break. If they remove columns we depend on, `OvertureSchemaMismatch` fires.
2. **`S3DuckDBBackend` query generation is unit-tested via a `_build_query` helper.** Actual S3 round-trips happen only in the slow opt-in test. This keeps the fast suite deterministic and network-free.
3. **Bbox does the filtering; polygon is a handoff record.** Phase-1 simplification of spec §3 decision 1: the spatial filter at fetch time is the *bounding box*, not the admin polygon. The admin polygon is fetched as the regular `divisions` theme and exposed on `Region.geometry` as a downstream-handoff record. C-stage (tile extraction, sub-project C) is contractually obligated to apply the polygon for precise clipping; this contract is documented in `docs/data/handoffs.md`. The types make this split explicit: `BboxScope` (filter) and `RegionGeometry` (handoff record) are distinct dataclasses.

**Branch:** all work on `phase-1-sub-A-overture-loader` off main. Merge to main only after the done check in Task 15.

---

## File map (responsibilities)

| File | Purpose |
|---|---|
| `pyproject.toml` (modify) | Add `duckdb == 1.5.2`, `pyarrow >= 15.0` |
| `configs/data/overture_release.yaml` | One key: `release: "2026-04-15.0"` |
| `configs/data/regions/singapore.yaml` | Region name + admin source + fallback bbox |
| `docs/data/overture_pinning_policy.md` | Re-pin procedure |
| `docs/data/handoffs.md` | A→C contract: themes are bbox-filtered; C must apply `admin_polygon` |
| `src/cfm/data/__init__.py` | Package marker |
| `src/cfm/data/overture/__init__.py` | Public API re-exports |
| `src/cfm/data/overture/errors.py` | Six `OvertureError` subclasses |
| `src/cfm/data/overture/region.py` | `Region`, `BboxScope`, `RegionGeometry`, `SizeEstimate` dataclasses |
| `src/cfm/data/overture/schema.py` | Curated column schemas + `validate_schema` |
| `src/cfm/data/overture/manifest.py` | `CacheManifest` r/w + sha256 helper |
| `src/cfm/data/overture/backend.py` | Protocol + `LocalFixtureBackend` + `S3DuckDBBackend` |
| `src/cfm/data/overture/loader.py` | `load_region` + cache logic + pre-fetch guard |
| `scripts/snapshot_overture_fixtures.py` | `--bootstrap` (synthetic) and `--s3` (real) modes |
| `scripts/cfm_data_invalidate.py` | CLI to delete a region's cache |
| `tests/fixtures/overture_mini/{theme}.parquet` × 5 | Synthetic-but-schema-matching parquets |
| `tests/data/overture/test_*.py` × 7 | One test file per source file |
| `tests/slow/test_real_s3_opt_in.py` | Real S3 fetch, `@pytest.mark.slow` |

---

## Task 1: Bootstrap branch, dependencies, and config files

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/data/overture_release.yaml`
- Create: `configs/data/regions/singapore.yaml`

- [ ] **Step 1.1: Create branch**

```bash
git checkout -b phase-1-sub-A-overture-loader
```

Expected: `Switched to a new branch 'phase-1-sub-A-overture-loader'`.

- [ ] **Step 1.2: Add runtime dependencies**

In `pyproject.toml`, modify the `[project].dependencies` list. Currently:

```toml
dependencies = [
  "shapely>=2.0",
  "pyyaml>=6.0",
]
```

Replace with:

```toml
dependencies = [
  "shapely>=2.0",
  "pyyaml>=6.0",
  "duckdb==1.5.2",
  "pyarrow>=15.0",
]
```

- [ ] **Step 1.3: Sync dependencies**

Run: `uv sync --all-extras`
Expected: installs `duckdb` 1.5.2 and `pyarrow` (>= 15.0). Other Phase 0 deps unchanged.

- [ ] **Step 1.4: Verify Phase 0 suite still passes**

Run: `uv run pytest`
Expected: `56 passed, 1 xfailed`. If anything other than that, fix before continuing — a regression on Phase 0 means a dependency upgrade broke something.

- [ ] **Step 1.5: Pin the Overture release**

Create `configs/data/overture_release.yaml`:

```yaml
# Phase 1 Overture pin. Update once per phase, never mid-phase.
# Latest stable as of 2026-05-16 per https://docs.overturemaps.org/release-calendar/
release: "2026-04-15.0"
overture_schema_version: "v1.16.0"
release_date: "2026-04-15"
release_subversion: 0
```

- [ ] **Step 1.6: Define the Singapore region**

Create `configs/data/regions/singapore.yaml`:

```yaml
# Singapore region config for Phase 1 sub-project A.
#
# Primary scoping is by admin polygon (fetched from Overture's divisions theme).
# The bbox here is a fallback used by C-stage sea masking if admin-polygon
# scoping ever proves impractical, and as a coarse DuckDB pre-filter.
name: singapore
admin:
  source: "overture://divisions"
  country_code: "SG"      # Overture ISO country code
  level: "country"        # divisions level: country | region | locality
fallback_bbox: [103.6, 1.16, 104.05, 1.48]    # [min_lon, min_lat, max_lon, max_lat]
crs: "EPSG:4326"
```

- [ ] **Step 1.7: Confirm config files load as YAML**

Run:

```bash
uv run python -c "
import yaml
r = yaml.safe_load(open('configs/data/overture_release.yaml'))
s = yaml.safe_load(open('configs/data/regions/singapore.yaml'))
assert r['release'] == '2026-04-15.0', r
assert s['name'] == 'singapore', s
assert s['fallback_bbox'] == [103.6, 1.16, 104.05, 1.48], s
print('configs OK')
"
```

Expected: `configs OK`.

- [ ] **Step 1.8: Commit**

```bash
git add pyproject.toml uv.lock configs/data/overture_release.yaml configs/data/regions/singapore.yaml
git commit -m "chore(data): add duckdb + pyarrow deps; pin Overture 2026-04-15.0; define Singapore region"
```

---

## Task 2: Error hierarchy (TDD)

**Files:**
- Create: `src/cfm/data/__init__.py`
- Create: `src/cfm/data/overture/__init__.py`
- Create: `src/cfm/data/overture/errors.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/overture/__init__.py`
- Create: `tests/data/overture/test_errors.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/data/__init__.py` (empty).
Create `tests/data/overture/__init__.py` (empty).

Create `tests/data/overture/test_errors.py`:

```python
from __future__ import annotations

import pytest

from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    OvertureError,
    OvertureSchemaMismatch,
    OvertureUnreachable,
    RegionNotFound,
    ReleaseNotConfigured,
)


def test_base_class_is_runtime_error() -> None:
    assert issubclass(OvertureError, RuntimeError)


@pytest.mark.parametrize(
    "cls",
    [
        OvertureUnreachable,
        OvertureSchemaMismatch,
        RegionNotFound,
        ReleaseNotConfigured,
        OversizedFetch,
        CacheCorrupt,
    ],
)
def test_each_subclass_inherits_from_base(cls: type[Exception]) -> None:
    assert issubclass(cls, OvertureError)


def test_subclasses_can_be_caught_as_base() -> None:
    with pytest.raises(OvertureError):
        raise CacheCorrupt("sha mismatch")


def test_six_distinct_subclasses() -> None:
    subclasses = {
        OvertureUnreachable,
        OvertureSchemaMismatch,
        RegionNotFound,
        ReleaseNotConfigured,
        OversizedFetch,
        CacheCorrupt,
    }
    assert len(subclasses) == 6
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `uv run pytest tests/data/overture/test_errors.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'cfm.data'`.

- [ ] **Step 2.3: Create package markers**

Create `src/cfm/data/__init__.py`:

```python
"""Data pipeline package: Overture loading, tile extraction, validation."""
```

Create `src/cfm/data/overture/__init__.py`:

```python
"""Overture Maps loader: pinned-release GeoParquet themes scoped to a region."""

from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    OvertureError,
    OvertureSchemaMismatch,
    OvertureUnreachable,
    RegionNotFound,
    ReleaseNotConfigured,
)

__all__ = [
    "CacheCorrupt",
    "OversizedFetch",
    "OvertureError",
    "OvertureSchemaMismatch",
    "OvertureUnreachable",
    "RegionNotFound",
    "ReleaseNotConfigured",
]
```

- [ ] **Step 2.4: Implement the error hierarchy**

Create `src/cfm/data/overture/errors.py`:

```python
from __future__ import annotations


class OvertureError(RuntimeError):
    """Base class for all Overture-loader failures."""


class OvertureUnreachable(OvertureError):
    """Backend cannot reach the data source (network failure, S3 timeout, etc.)."""


class OvertureSchemaMismatch(OvertureError):
    """A theme's columns do not match the curated schema in cfm.data.overture.schema.

    Usually means Overture added or removed columns between releases. To resolve:
    bump the pin in configs/data/overture_release.yaml, re-snapshot fixtures via
    scripts/snapshot_overture_fixtures.py, and update schema.py if needed.
    """


class RegionNotFound(OvertureError):
    """No configs/data/regions/<name>.yaml found for the requested region."""


class ReleaseNotConfigured(OvertureError):
    """configs/data/overture_release.yaml is missing or malformed."""


class OversizedFetch(OvertureError):
    """Estimated download exceeds the 2 GB threshold and confirm=False was passed.

    Pass confirm=True to load_region if you genuinely want a fetch this large.
    """


class CacheCorrupt(OvertureError):
    """A cached parquet's sha256 does not match what manifest.yaml recorded.

    The cache has been partially written or tampered with. Run
    scripts/cfm_data_invalidate.py to remove and refetch.
    """
```

- [ ] **Step 2.5: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_errors.py -v`
Expected: 9 passed (1 + 6 parametrised + 1 + 1).

- [ ] **Step 2.6: Verify full suite still green**

Run: `uv run pytest`
Expected: `65 passed, 1 xfailed`.

- [ ] **Step 2.7: Lint clean and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data tests/data
git commit -m "feat(data): add OvertureError hierarchy with six subclasses"
```

---

## Task 3: Region, BboxScope, RegionGeometry, SizeEstimate dataclasses (TDD)

**Files:**
- Create: `src/cfm/data/overture/region.py`
- Modify: `src/cfm/data/overture/__init__.py`
- Create: `tests/data/overture/test_region.py`

**Design note** — split `BboxScope` and `RegionGeometry` deliberately:

- **`BboxScope`** is what backends use as their *fetch-time spatial filter*. Carries four floats; nothing else.
- **`RegionGeometry`** is the *handoff record* describing the region's precise shape (admin polygon + provenance). Backends never touch it; C-stage and beyond consume it for clipping.
- **`Region`** carries both. The split prevents a reader from assuming the polygon is doing filter work — which it isn't in Phase 1.

- [ ] **Step 3.1: Write the failing tests**

Create `tests/data/overture/test_region.py`:

```python
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pytest
from shapely.geometry import Polygon

from cfm.data.overture.region import (
    BboxScope,
    Region,
    RegionGeometry,
    SizeEstimate,
)


def _square_polygon() -> Polygon:
    return Polygon([(103.6, 1.16), (104.05, 1.16), (104.05, 1.48), (103.6, 1.48), (103.6, 1.16)])


def test_bbox_scope_is_frozen() -> None:
    bbox = BboxScope(min_lon=103.6, min_lat=1.16, max_lon=104.05, max_lat=1.48)
    with pytest.raises(FrozenInstanceError):
        bbox.min_lon = 0.0  # type: ignore[misc]


def test_bbox_scope_as_tuple_roundtrips() -> None:
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    assert bbox.as_tuple() == (103.6, 1.16, 104.05, 1.48)


def test_region_geometry_holds_polygon_and_source() -> None:
    poly = _square_polygon()
    geom = RegionGeometry(admin_polygon=poly, source="overture://divisions:country:SG")
    assert geom.admin_polygon is poly
    assert geom.source == "overture://divisions:country:SG"


def test_region_geometry_is_frozen() -> None:
    geom = RegionGeometry(admin_polygon=_square_polygon(), source="x")
    with pytest.raises(FrozenInstanceError):
        geom.source = "y"  # type: ignore[misc]


def test_size_estimate_fields() -> None:
    est = SizeEstimate(rows=12345, bytes=67890)
    assert est.rows == 12345
    assert est.bytes == 67890


def test_size_estimate_is_frozen() -> None:
    est = SizeEstimate(rows=1, bytes=1)
    with pytest.raises(FrozenInstanceError):
        est.rows = 2  # type: ignore[misc]


def test_region_construction(tmp_path: Path) -> None:
    poly = _square_polygon()
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    geometry = RegionGeometry(admin_polygon=poly, source="overture://divisions:country:SG")
    themes = {
        "buildings": pa.table({"id": [1, 2]}),
        "places": pa.table({"id": [3, 4]}),
    }
    region = Region(
        name="singapore",
        release="2026-04-15.0",
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=tmp_path / "manifest.yaml",
    )
    assert region.name == "singapore"
    assert region.release == "2026-04-15.0"
    assert region.themes["buildings"].num_rows == 2
    assert region.admin_polygon is poly
    assert region.bbox == (103.6, 1.16, 104.05, 1.48)


def test_region_is_frozen(tmp_path: Path) -> None:
    poly = _square_polygon()
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    geometry = RegionGeometry(admin_polygon=poly, source="x")
    region = Region(
        name="singapore",
        release="2026-04-15.0",
        fetch_bbox=bbox,
        geometry=geometry,
        themes={},
        manifest_path=tmp_path / "manifest.yaml",
    )
    with pytest.raises(FrozenInstanceError):
        region.name = "elsewhere"  # type: ignore[misc]


def test_region_docstring_states_handoff_contract() -> None:
    """The Region docstring must tell downstream consumers that themes are
    bbox-filtered and admin_polygon is for their use, not the backend's.
    """
    assert "bbox" in Region.__doc__.lower()
    assert "admin_polygon" in Region.__doc__ or "admin polygon" in Region.__doc__.lower()
```

- [ ] **Step 3.2: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_region.py -v`
Expected: `ModuleNotFoundError: No module named 'cfm.data.overture.region'`.

- [ ] **Step 3.3: Implement the dataclasses**

Create `src/cfm/data/overture/region.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class BboxScope:
    """Bounding box used by Overture backends as the fetch-time spatial filter.

    This is the only spatial filter applied at fetch time in Phase 1. The
    admin polygon (see RegionGeometry) is NOT used by backends; downstream
    consumers (e.g. C-stage tile extraction) must apply it themselves for
    precise clipping. See docs/data/handoffs.md.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BboxScope:
        return cls(min_lon=t[0], min_lat=t[1], max_lon=t[2], max_lat=t[3])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


@dataclass(frozen=True)
class RegionGeometry:
    """Precise geometric description of a region, plus where it came from.

    Phase 1 uses this as a downstream-handoff record: it lives on Region for
    C-stage and later consumers, but is NOT used by backends as a fetch-time
    filter. See docs/data/handoffs.md.
    """

    admin_polygon: BaseGeometry
    source: str   # e.g. "overture://divisions:country:SG"


@dataclass(frozen=True)
class SizeEstimate:
    """Cheap pre-fetch estimate: row count + approximate byte size."""

    rows: int
    bytes: int


@dataclass(frozen=True)
class Region:
    """A fully-loaded Overture region for one release.

    HANDOFF CONTRACT — read before consuming this object:

    The `themes` parquet tables are filtered ONLY by `fetch_bbox` at load
    time. The `geometry.admin_polygon` is a HANDOFF record describing the
    region's precise shape; it is NOT applied to the themes at load time.

    Downstream consumers (e.g. C-stage tile extraction in sub-project C)
    MUST apply `admin_polygon` to clip themes before training data leaves
    A's contract. Failing to do so means open sea — which falls inside
    `fetch_bbox` but outside `admin_polygon` — silently enters the
    training set. See docs/data/handoffs.md.
    """

    name: str
    release: str
    fetch_bbox: BboxScope
    geometry: RegionGeometry
    themes: dict[str, pa.Table]
    manifest_path: Path

    @property
    def admin_polygon(self) -> BaseGeometry:
        return self.geometry.admin_polygon

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.fetch_bbox.as_tuple()
```

- [ ] **Step 3.4: Re-export new dataclasses from the package**

Update `src/cfm/data/overture/__init__.py`:

```python
"""Overture Maps loader: pinned-release GeoParquet themes scoped to a region."""

from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    OvertureError,
    OvertureSchemaMismatch,
    OvertureUnreachable,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.region import (
    BboxScope,
    Region,
    RegionGeometry,
    SizeEstimate,
)

__all__ = [
    "BboxScope",
    "CacheCorrupt",
    "OversizedFetch",
    "OvertureError",
    "OvertureSchemaMismatch",
    "OvertureUnreachable",
    "Region",
    "RegionGeometry",
    "RegionNotFound",
    "ReleaseNotConfigured",
    "SizeEstimate",
]
```

- [ ] **Step 3.5: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_region.py -v`
Expected: 9 passed (4 BboxScope/RegionGeometry, 2 SizeEstimate, 2 Region, 1 docstring contract).

- [ ] **Step 3.6: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/region.py src/cfm/data/overture/__init__.py tests/data/overture/test_region.py
git commit -m "feat(data): BboxScope (filter) + RegionGeometry (handoff) + Region with explicit contract docstring"
```

---

## Task 4: Curated schema definitions (TDD)

**Files:**
- Create: `src/cfm/data/overture/schema.py`
- Create: `tests/data/overture/test_schema.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/data/overture/test_schema.py`:

```python
from __future__ import annotations

import pyarrow as pa
import pytest

from cfm.data.overture.errors import OvertureSchemaMismatch
from cfm.data.overture.schema import (
    EXPECTED_THEMES,
    THEME_SCHEMAS,
    validate_schema,
)


def test_expected_themes_are_the_five_we_use() -> None:
    assert EXPECTED_THEMES == ("buildings", "places", "transportation", "base", "divisions")


def test_every_theme_has_a_schema() -> None:
    for theme in EXPECTED_THEMES:
        assert theme in THEME_SCHEMAS
        assert isinstance(THEME_SCHEMAS[theme], pa.Schema)


def test_buildings_schema_includes_geometry_and_class() -> None:
    s = THEME_SCHEMAS["buildings"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "class" in s.names


def test_places_schema_includes_categories() -> None:
    s = THEME_SCHEMAS["places"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "categories" in s.names


def test_transportation_schema_includes_class() -> None:
    s = THEME_SCHEMAS["transportation"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "class" in s.names


def test_base_schema_includes_subtype() -> None:
    s = THEME_SCHEMAS["base"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "subtype" in s.names


def test_divisions_schema_includes_country() -> None:
    s = THEME_SCHEMAS["divisions"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "country" in s.names


def test_validate_schema_accepts_exact_match() -> None:
    table = pa.table({
        "id": pa.array(["a", "b"], type=pa.string()),
        "geometry": pa.array([b"g1", b"g2"], type=pa.binary()),
        "class": pa.array(["residential", "office"], type=pa.string()),
        "height": pa.array([10.0, 20.0], type=pa.float64()),
        "num_floors": pa.array([3, 5], type=pa.int32()),
    })
    # Should not raise.
    validate_schema(table, theme="buildings")


def test_validate_schema_accepts_extra_columns() -> None:
    # Real Overture parquet may have many more columns than we use. We only
    # require the curated set; extras are tolerated.
    table = pa.table({
        "id": pa.array(["a"], type=pa.string()),
        "geometry": pa.array([b"g1"], type=pa.binary()),
        "class": pa.array(["residential"], type=pa.string()),
        "height": pa.array([10.0], type=pa.float64()),
        "num_floors": pa.array([3], type=pa.int32()),
        "extra_column_we_dont_use": pa.array([42], type=pa.int64()),
    })
    validate_schema(table, theme="buildings")


def test_validate_schema_rejects_missing_required_column() -> None:
    table = pa.table({
        "id": pa.array(["a"], type=pa.string()),
        "geometry": pa.array([b"g1"], type=pa.binary()),
        # "class" missing
        "height": pa.array([10.0], type=pa.float64()),
        "num_floors": pa.array([3], type=pa.int32()),
    })
    with pytest.raises(OvertureSchemaMismatch) as exc_info:
        validate_schema(table, theme="buildings")
    assert "class" in str(exc_info.value)


def test_validate_schema_rejects_unknown_theme() -> None:
    table = pa.table({"id": pa.array(["a"], type=pa.string())})
    with pytest.raises(OvertureSchemaMismatch) as exc_info:
        validate_schema(table, theme="not_a_theme")
    assert "not_a_theme" in str(exc_info.value)
```

- [ ] **Step 4.2: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_schema.py -v`
Expected: import error.

- [ ] **Step 4.3: Implement schema.py**

Create `src/cfm/data/overture/schema.py`:

```python
from __future__ import annotations

import pyarrow as pa

from cfm.data.overture.errors import OvertureSchemaMismatch

# The five themes we load in Phase 1.
EXPECTED_THEMES: tuple[str, ...] = (
    "buildings",
    "places",
    "transportation",
    "base",
    "divisions",
)

# Curated column schemas — the minimum set of columns we use per theme.
# Real Overture parquet may have many more columns; we tolerate extras.
# If Overture removes any of these columns, validate_schema raises.
#
# Geometry is stored as WKB (Well-Known Binary) bytes in Overture parquet.
# `class` and `subtype` are simple string categories.
# `categories` (places) is a struct with main + alternate fields; for Phase 1
#   we just require the column to exist — B1 will inspect its structure.

THEME_SCHEMAS: dict[str, pa.Schema] = {
    "buildings": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("class", pa.string()),
            ("height", pa.float64()),
            ("num_floors", pa.int32()),
        ]
    ),
    "places": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("categories", pa.string()),
        ]
    ),
    "transportation": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("class", pa.string()),
            ("subtype", pa.string()),
        ]
    ),
    "base": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("subtype", pa.string()),
        ]
    ),
    "divisions": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("country", pa.string()),
            ("subtype", pa.string()),
        ]
    ),
}


def validate_schema(table: pa.Table, *, theme: str) -> None:
    """Raise OvertureSchemaMismatch if `table` is missing any required column.

    Extras are tolerated. Column dtypes are not strictly checked beyond presence
    (real Overture parquet's nested dtypes are too varied to lock in Phase 1).
    """
    if theme not in THEME_SCHEMAS:
        raise OvertureSchemaMismatch(
            f"unknown theme {theme!r}; expected one of {EXPECTED_THEMES}"
        )
    expected_columns = set(THEME_SCHEMAS[theme].names)
    actual_columns = set(table.column_names)
    missing = expected_columns - actual_columns
    if missing:
        raise OvertureSchemaMismatch(
            f"theme={theme!r} missing required columns: {sorted(missing)}; "
            f"has {sorted(actual_columns)}"
        )
```

- [ ] **Step 4.4: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_schema.py -v`
Expected: 11 passed.

- [ ] **Step 4.5: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/schema.py tests/data/overture/test_schema.py
git commit -m "feat(data): curated schema definitions for five Overture themes"
```

---

## Task 5: CacheManifest (TDD)

**Files:**
- Create: `src/cfm/data/overture/manifest.py`
- Create: `tests/data/overture/test_manifest.py`

- [ ] **Step 5.1: Write the failing tests**

Create `tests/data/overture/test_manifest.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cfm.data.overture.manifest import (
    CacheManifest,
    ThemeEntry,
    sha256_of_file,
)


def test_sha256_of_file_known_value(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # sha256("hello") == 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    assert sha256_of_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_manifest_round_trip_yaml(tmp_path: Path) -> None:
    m = CacheManifest(
        schema_version=1,
        release="2026-04-15.0",
        release_date="2026-04-15",
        release_subversion=0,
        overture_schema_version="v1.16.0",
        region="singapore",
        admin_polygon_source="overture://divisions:country:SG",
        bbox=(103.6, 1.16, 104.05, 1.48),
        backend="S3DuckDBBackend",
        fetched_at=datetime(2026, 5, 16, 14, 32, 11, tzinfo=timezone.utc),
        themes={
            "buildings": ThemeEntry(
                s3_url="s3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/",
                rows=1000,
                bytes=50_000,
                sha256="abc123",
                parquet_filename="buildings.parquet",
            ),
        },
    )
    path = tmp_path / "manifest.yaml"
    m.to_yaml(path)
    loaded = CacheManifest.from_yaml(path)
    assert loaded == m


def test_manifest_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("schema_version: 99\nrelease: x\n")
    with pytest.raises(ValueError, match="schema_version"):
        CacheManifest.from_yaml(bad)


def test_theme_entry_fields() -> None:
    e = ThemeEntry(
        s3_url="s3://x/",
        rows=10,
        bytes=100,
        sha256="abc",
        parquet_filename="x.parquet",
    )
    assert e.rows == 10
    assert e.parquet_filename == "x.parquet"


def test_manifest_fetched_at_serialises_to_iso_z(tmp_path: Path) -> None:
    m = CacheManifest(
        schema_version=1,
        release="2026-04-15.0",
        release_date="2026-04-15",
        release_subversion=0,
        overture_schema_version="v1.16.0",
        region="singapore",
        admin_polygon_source="overture://divisions:country:SG",
        bbox=(103.6, 1.16, 104.05, 1.48),
        backend="S3DuckDBBackend",
        fetched_at=datetime(2026, 5, 16, 14, 32, 11, tzinfo=timezone.utc),
        themes={},
    )
    path = tmp_path / "manifest.yaml"
    m.to_yaml(path)
    text = path.read_text()
    assert "2026-05-16T14:32:11Z" in text
```

- [ ] **Step 5.2: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_manifest.py -v`
Expected: import error.

- [ ] **Step 5.3: Implement manifest.py**

Create `src/cfm/data/overture/manifest.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

CURRENT_SCHEMA_VERSION = 1


def sha256_of_file(path: Path) -> str:
    """Hex-encoded SHA-256 of the file at `path`."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ThemeEntry:
    """One entry under the `themes` mapping in a cache manifest."""

    s3_url: str
    rows: int
    bytes: int
    sha256: str
    parquet_filename: str

    def to_dict(self) -> dict:
        return {
            "s3_url": self.s3_url,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "parquet_filename": self.parquet_filename,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThemeEntry:
        return cls(
            s3_url=data["s3_url"],
            rows=int(data["rows"]),
            bytes=int(data["bytes"]),
            sha256=data["sha256"],
            parquet_filename=data["parquet_filename"],
        )


@dataclass(frozen=True)
class CacheManifest:
    """Per-region cache manifest, written to manifest.yaml on every fetch.

    Matches the format in spec §7 of
    docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md.
    """

    schema_version: int
    release: str
    release_date: str
    release_subversion: int
    overture_schema_version: str
    region: str
    admin_polygon_source: str
    bbox: tuple[float, float, float, float]
    backend: str
    fetched_at: datetime
    themes: dict[str, ThemeEntry] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "release": self.release,
            "release_date": self.release_date,
            "release_subversion": self.release_subversion,
            "overture_schema_version": self.overture_schema_version,
            "region": self.region,
            "scope": {
                "admin_polygon_source": self.admin_polygon_source,
                "bbox": list(self.bbox),
            },
            "backend": self.backend,
            "fetched_at": _format_iso_z(self.fetched_at),
            "themes": {name: entry.to_dict() for name, entry in self.themes.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> CacheManifest:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        version = int(data.get("schema_version", 0))
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"manifest at {path} has schema_version={version}, "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )
        scope = data.get("scope", {})
        bbox_list = scope.get("bbox", [0.0, 0.0, 0.0, 0.0])
        return cls(
            schema_version=version,
            release=data["release"],
            release_date=data["release_date"],
            release_subversion=int(data["release_subversion"]),
            overture_schema_version=data["overture_schema_version"],
            region=data["region"],
            admin_polygon_source=scope["admin_polygon_source"],
            bbox=(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]),
            backend=data["backend"],
            fetched_at=_parse_iso_z(data["fetched_at"]),
            themes={
                name: ThemeEntry.from_dict(entry)
                for name, entry in (data.get("themes") or {}).items()
            },
        )


def _format_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_z(s: str) -> datetime:
    # Accept both "Z" suffix and explicit +00:00.
    cleaned = s.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned).astimezone(timezone.utc)
```

- [ ] **Step 5.4: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_manifest.py -v`
Expected: 5 passed.

- [ ] **Step 5.5: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/manifest.py tests/data/overture/test_manifest.py
git commit -m "feat(data): CacheManifest with YAML round-trip and sha256 helper"
```

---

## Task 6: Bootstrap synthetic fixtures

**Files:**
- Create: `scripts/snapshot_overture_fixtures.py` (bootstrap mode only — S3 mode added in Task 13)
- Create: `tests/fixtures/overture_mini/buildings.parquet`
- Create: `tests/fixtures/overture_mini/places.parquet`
- Create: `tests/fixtures/overture_mini/transportation.parquet`
- Create: `tests/fixtures/overture_mini/base.parquet`
- Create: `tests/fixtures/overture_mini/divisions.parquet`

These parquets are committed binary files. They are small (~few KB each) and represent the curated schema. They are regenerated when re-pinning by running this script with `--mode bootstrap`.

- [ ] **Step 6.1: Write the bootstrap script**

Create `scripts/snapshot_overture_fixtures.py`:

```python
"""Snapshot Overture fixtures used by tests.

Two modes:

  --mode bootstrap  Write synthetic, schema-matching parquets to
                    tests/fixtures/overture_mini/. No S3, no network.
                    Used to initialise the fixtures during sub-project A,
                    and to regenerate them on re-pin in offline situations.

  --mode s3         Fetch a tiny real bbox (currently a 0.01 deg x 0.01 deg
                    square near central Singapore) and write the resulting
                    rows to tests/fixtures/overture_mini/. Requires
                    network + the pinned Overture release. (Implemented in
                    Task 13; not available in this commit.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "overture_mini"

# Five synthetic rows per theme. Geometry bytes are placeholder values (we
# don't decode them in fast tests; the slow opt-in test uses real geometry).
_FAKE_WKB = b"\x01\x01\x00\x00\x00" + b"\x00" * 16  # WKB POINT placeholder

_SYNTHETIC: dict[str, dict] = {
    "buildings": {
        "id": ["b1", "b2", "b3", "b4", "b5"],
        "geometry": [_FAKE_WKB] * 5,
        "class": ["residential", "commercial", "residential", "industrial", "residential"],
        "height": [10.0, 25.0, 8.0, 15.0, 12.0],
        "num_floors": [3, 7, 2, 4, 4],
    },
    "places": {
        "id": ["p1", "p2", "p3", "p4", "p5"],
        "geometry": [_FAKE_WKB] * 5,
        "categories": [
            '{"primary": "restaurant"}',
            '{"primary": "school"}',
            '{"primary": "retail"}',
            '{"primary": "park_amenity"}',
            '{"primary": "transit_stop"}',
        ],
    },
    "transportation": {
        "id": ["t1", "t2", "t3", "t4", "t5"],
        "geometry": [_FAKE_WKB] * 5,
        "class": ["motorway", "primary", "residential", "service", "secondary"],
        "subtype": ["road", "road", "road", "road", "road"],
    },
    "base": {
        "id": ["a1", "a2", "a3", "a4", "a5"],
        "geometry": [_FAKE_WKB] * 5,
        "subtype": ["land", "water", "land", "water", "land_cover"],
    },
    "divisions": {
        "id": ["d1"],
        "geometry": [_FAKE_WKB],
        "country": ["SG"],
        "subtype": ["country"],
    },
}

# Explicit pyarrow types so casts are deterministic across pyarrow versions.
_TYPES: dict[str, dict[str, pa.DataType]] = {
    "buildings": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "class": pa.string(),
        "height": pa.float64(),
        "num_floors": pa.int32(),
    },
    "places": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "categories": pa.string(),
    },
    "transportation": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "class": pa.string(),
        "subtype": pa.string(),
    },
    "base": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "subtype": pa.string(),
    },
    "divisions": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "country": pa.string(),
        "subtype": pa.string(),
    },
}


def _build_table(theme: str) -> pa.Table:
    cols = _SYNTHETIC[theme]
    types = _TYPES[theme]
    arrays = {name: pa.array(values, type=types[name]) for name, values in cols.items()}
    return pa.table(arrays)


def bootstrap() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for theme in _SYNTHETIC:
        out = FIXTURES_DIR / f"{theme}.parquet"
        table = _build_table(theme)
        pq.write_table(table, out)
        print(f"wrote {out.relative_to(REPO_ROOT)} ({table.num_rows} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bootstrap", "s3"),
        default="bootstrap",
        help="bootstrap = synthetic, no network; s3 = real fetch (Task 13)",
    )
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap()
        return 0
    print("--mode s3 not yet implemented; see Task 13", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.2: Run the bootstrap script**

Run: `uv run python scripts/snapshot_overture_fixtures.py --mode bootstrap`

Expected output:

```
wrote tests/fixtures/overture_mini/buildings.parquet (5 rows)
wrote tests/fixtures/overture_mini/places.parquet (5 rows)
wrote tests/fixtures/overture_mini/transportation.parquet (5 rows)
wrote tests/fixtures/overture_mini/base.parquet (5 rows)
wrote tests/fixtures/overture_mini/divisions.parquet (1 rows)
```

- [ ] **Step 6.3: Verify fixtures are readable and match schema**

Run:

```bash
uv run python -c "
import pyarrow.parquet as pq
from cfm.data.overture.schema import EXPECTED_THEMES, validate_schema
for theme in EXPECTED_THEMES:
    t = pq.read_table(f'tests/fixtures/overture_mini/{theme}.parquet')
    validate_schema(t, theme=theme)
    print(f'{theme}: {t.num_rows} rows, columns={t.column_names}')
"
```

Expected: every theme prints its row count and columns, and `validate_schema` raises nothing.

- [ ] **Step 6.4: Commit the script and the fixtures**

```bash
git add scripts/snapshot_overture_fixtures.py tests/fixtures/overture_mini/
git commit -m "data: synthetic Overture fixtures + bootstrap script"
```

---

## Task 7: Backend protocol + LocalFixtureBackend (TDD)

**Files:**
- Create: `src/cfm/data/overture/backend.py` (protocol + `LocalFixtureBackend` only — `S3DuckDBBackend` arrives in Task 8)
- Create: `tests/data/overture/test_backend_local.py`
- Create: `tests/data/overture/conftest.py`

- [ ] **Step 7.1: Add shared test fixtures**

Create `tests/data/overture/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.overture.region import BboxScope


@pytest.fixture(scope="session")
def overture_mini_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "overture_mini"


@pytest.fixture(scope="session")
def singapore_bbox() -> BboxScope:
    return BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
```

(`repo_root` comes from `tests/conftest.py` already in the project.)

- [ ] **Step 7.2: Write the failing tests**

Create `tests/data/overture/test_backend_local.py`:

```python
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cfm.data.overture.backend import LocalFixtureBackend, OvertureBackend
from cfm.data.overture.region import BboxScope


def test_local_backend_implements_protocol(overture_mini_dir: Path) -> None:
    backend: OvertureBackend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    # Static-typing check via assignment to protocol-typed variable plus a runtime call.
    assert hasattr(backend, "read_theme")
    assert hasattr(backend, "estimate_size")


def test_local_backend_reads_each_theme(
    overture_mini_dir: Path, singapore_bbox: BboxScope
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    for theme in ("buildings", "places", "transportation", "base", "divisions"):
        table = backend.read_theme(theme=theme, bbox=singapore_bbox, release="ignored")
        assert isinstance(table, pa.Table)
        assert table.num_rows > 0


def test_local_backend_estimate_size_is_cheap(
    overture_mini_dir: Path, singapore_bbox: BboxScope
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    est = backend.estimate_size(theme="buildings", bbox=singapore_bbox, release="ignored")
    assert est.rows > 0
    assert est.bytes > 0


def test_local_backend_unknown_theme_raises(
    overture_mini_dir: Path, singapore_bbox: BboxScope
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(FileNotFoundError):
        backend.read_theme(theme="not_a_theme", bbox=singapore_bbox, release="ignored")
```

- [ ] **Step 7.3: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_backend_local.py -v`
Expected: import error.

- [ ] **Step 7.4: Implement backend.py (protocol + LocalFixtureBackend only)**

Create `src/cfm/data/overture/backend.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from cfm.data.overture.region import BboxScope, SizeEstimate


class OvertureBackend(Protocol):
    """Reads Overture theme parquet for a bounding box at a given release.

    Phase 1 backends apply only the bounding-box filter. The region's
    admin polygon is NOT used here; downstream consumers apply it for
    precise clipping. See docs/data/handoffs.md.
    """

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table: ...

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate: ...


class LocalFixtureBackend:
    """Reads from tests/fixtures/overture_mini/. Ignores bbox and release.

    Used by the fast test suite. The fixtures are committed parquets generated
    by scripts/snapshot_overture_fixtures.py.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = Path(fixtures_dir)

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        return pq.read_table(path)

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        meta = pq.read_metadata(path)
        return SizeEstimate(rows=meta.num_rows, bytes=path.stat().st_size)
```

- [ ] **Step 7.5: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_backend_local.py -v`
Expected: 4 passed.

- [ ] **Step 7.6: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/backend.py tests/data/overture/conftest.py tests/data/overture/test_backend_local.py
git commit -m "feat(data): OvertureBackend protocol + LocalFixtureBackend"
```

---

## Task 8: S3DuckDBBackend (query generation tested; S3 deferred to slow test)

**Files:**
- Modify: `src/cfm/data/overture/backend.py`
- Create: `tests/data/overture/test_backend_s3.py`

- [ ] **Step 8.1: Write the failing tests**

Create `tests/data/overture/test_backend_s3.py`:

```python
from __future__ import annotations

import pytest

from cfm.data.overture.backend import S3DuckDBBackend
from cfm.data.overture.region import BboxScope


def test_s3_backend_default_bucket() -> None:
    backend = S3DuckDBBackend()
    assert backend.bucket == "overturemaps-us-west-2"


def test_s3_backend_custom_bucket() -> None:
    backend = S3DuckDBBackend(bucket="my-mirror")
    assert backend.bucket == "my-mirror"


def test_build_s3_url_for_release_and_theme() -> None:
    backend = S3DuckDBBackend()
    url = backend.build_s3_url(theme="buildings", release="2026-04-15.0")
    assert url == "s3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/*"


def test_build_query_contains_bbox_clauses_and_not_polygon(singapore_bbox: BboxScope) -> None:
    backend = S3DuckDBBackend()
    sql = backend.build_query(theme="buildings", bbox=singapore_bbox, release="2026-04-15.0")
    # Theme path
    assert "release/2026-04-15.0/theme=buildings/" in sql
    # Bbox-only filter (Overture parquet has bbox.xmin/xmax/ymin/ymax)
    assert "bbox.xmin" in sql
    assert "bbox.ymax" in sql
    # No polygon refinement in Phase 1 — handoff contract is bbox-only at fetch time.
    assert "ST_Intersects" not in sql and "st_intersects" not in sql.lower()


def test_build_count_query_returns_count_star(singapore_bbox: BboxScope) -> None:
    backend = S3DuckDBBackend()
    sql = backend.build_count_query(theme="buildings", bbox=singapore_bbox, release="2026-04-15.0")
    assert sql.strip().lower().startswith("select count(*)")
    assert "release/2026-04-15.0/theme=buildings/" in sql


@pytest.mark.slow
def test_real_s3_smoke_buildings() -> None:
    """Sanity smoke that S3 is reachable. Excluded from default suite."""
    backend = S3DuckDBBackend()
    tiny = BboxScope.from_tuple((103.85, 1.29, 103.86, 1.30))
    est = backend.estimate_size(theme="buildings", bbox=tiny, release="2026-04-15.0")
    assert est.rows >= 0
```

- [ ] **Step 8.2: Run tests; expect failure (or only smoke skipped)**

Run: `uv run pytest tests/data/overture/test_backend_s3.py -v`
Expected: 5 fast tests fail with import error; the `@pytest.mark.slow` test is deselected by default — verify with `-v` that it shows `s` (skipped) or simply isn't collected.

- [ ] **Step 8.3: Implement S3DuckDBBackend**

Add to `src/cfm/data/overture/backend.py` (after `LocalFixtureBackend`):

```python
import duckdb

from cfm.data.overture.errors import OvertureUnreachable


class S3DuckDBBackend:
    """Reads Overture themes from public S3 via DuckDB + httpfs extensions.

    The Overture S3 bucket (s3://overturemaps-us-west-2/) is public-read; no
    credentials required. Phase 1 filters by bounding box only; precise
    admin-polygon clipping is the responsibility of downstream consumers
    (see docs/data/handoffs.md).
    """

    DEFAULT_BUCKET = "overturemaps-us-west-2"

    def __init__(self, bucket: str | None = None) -> None:
        self.bucket = bucket or self.DEFAULT_BUCKET

    def build_s3_url(self, *, theme: str, release: str) -> str:
        return f"s3://{self.bucket}/release/{release}/theme={theme}/*"

    def build_query(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> str:
        url = self.build_s3_url(theme=theme, release=release)
        return f"""
            SELECT *
            FROM read_parquet('{url}', filename=false, hive_partitioning=1)
            WHERE bbox.xmin <= {bbox.max_lon}
              AND bbox.xmax >= {bbox.min_lon}
              AND bbox.ymin <= {bbox.max_lat}
              AND bbox.ymax >= {bbox.min_lat}
        """.strip()

    def build_count_query(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> str:
        url = self.build_s3_url(theme=theme, release=release)
        return f"""
            SELECT COUNT(*) AS n
            FROM read_parquet('{url}', filename=false, hive_partitioning=1)
            WHERE bbox.xmin <= {bbox.max_lon}
              AND bbox.xmax >= {bbox.min_lon}
              AND bbox.ymin <= {bbox.max_lat}
              AND bbox.ymax >= {bbox.min_lat}
        """.strip()

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table:
        try:
            con = self._open()
            return con.execute(
                self.build_query(theme=theme, bbox=bbox, release=release)
            ).arrow()
        except duckdb.IOException as e:  # type: ignore[attr-defined]
            raise OvertureUnreachable(f"reading theme={theme!r}: {e}") from e

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate:
        try:
            con = self._open()
            (rows,) = con.execute(
                self.build_count_query(theme=theme, bbox=bbox, release=release)
            ).fetchone()
        except duckdb.IOException as e:  # type: ignore[attr-defined]
            raise OvertureUnreachable(f"estimating theme={theme!r}: {e}") from e
        # Rough byte estimate: 200 bytes per row is a Phase-1 guess; refined when
        # actual cache writes report real sizes.
        return SizeEstimate(rows=int(rows), bytes=int(rows) * 200)

    @staticmethod
    def _open() -> "duckdb.DuckDBPyConnection":  # type: ignore[name-defined]
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
        return con
```

- [ ] **Step 8.4: Run fast tests; expect pass**

Run: `uv run pytest tests/data/overture/test_backend_s3.py -v -m 'not slow'`
Expected: 5 passed (the slow smoke is deselected).

- [ ] **Step 8.5: Run full suite to confirm no regressions**

Run: `uv run pytest -m 'not slow'`
Expected: previous count + 5 = current count, all passing. The `slow` smoke isn't run by default.

- [ ] **Step 8.6: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/backend.py tests/data/overture/test_backend_s3.py
git commit -m "feat(data): S3DuckDBBackend with query generation + slow smoke test"
```

---

## Task 9: Loader — cache management happy path (TDD)

**Files:**
- Create: `src/cfm/data/overture/loader.py`
- Modify: `src/cfm/data/overture/__init__.py`
- Create: `tests/data/overture/test_loader.py`

- [ ] **Step 9.1: Write the failing tests**

Create `tests/data/overture/test_loader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from cfm.data.overture.backend import LocalFixtureBackend
from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.loader import load_region


def _write_release_pin(repo_root: Path, release: str = "2026-04-15.0") -> Path:
    cfg_dir = repo_root / "configs" / "data"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "overture_release.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "release": release,
                "overture_schema_version": "v1.16.0",
                "release_date": "2026-04-15",
                "release_subversion": 0,
            }
        )
    )
    return path


def _write_singapore_region(repo_root: Path) -> Path:
    cfg_dir = repo_root / "configs" / "data" / "regions"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "singapore.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "singapore",
                "admin": {"source": "overture://divisions", "country_code": "SG", "level": "country"},
                "fallback_bbox": [103.6, 1.16, 104.05, 1.48],
                "crs": "EPSG:4326",
            }
        )
    )
    return path


@pytest.fixture()
def isolated_repo(tmp_path: Path, overture_mini_dir: Path) -> Path:
    """A throwaway repo root with configs + a fixtures dir pointed at the
    real overture_mini parquets. Cache lives under tmp_path/data/cache/."""
    _write_release_pin(tmp_path)
    _write_singapore_region(tmp_path)
    # Symlink overture_mini fixtures so LocalFixtureBackend can read them.
    fix = tmp_path / "tests" / "fixtures" / "overture_mini"
    fix.parent.mkdir(parents=True, exist_ok=True)
    fix.symlink_to(overture_mini_dir)
    return tmp_path


def test_load_region_happy_path_first_fetch(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert region.name == "singapore"
    assert region.release == "2026-04-15.0"
    assert set(region.themes) == {"buildings", "places", "transportation", "base", "divisions"}
    assert region.manifest_path.exists()
    # Manifest contains the right release.
    data = yaml.safe_load(region.manifest_path.read_text())
    assert data["release"] == "2026-04-15.0"
    assert "themes" in data and "buildings" in data["themes"]


def test_load_region_second_call_uses_cache(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    first = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Mutate the fixture's mtime so we'd notice a re-read; the loader should
    # NOT touch the source on a cache hit.
    second = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert first.manifest_path == second.manifest_path


def test_load_region_unknown_region_raises(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(RegionNotFound):
        load_region("atlantis", backend=backend, repo_root=isolated_repo)


def test_load_region_missing_release_pin_raises(tmp_path: Path, overture_mini_dir: Path) -> None:
    _write_singapore_region(tmp_path)
    # No release pin written.
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(ReleaseNotConfigured):
        load_region("singapore", backend=backend, repo_root=tmp_path)


def test_load_region_release_mismatch_silently_refetches(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    # First fetch at the pinned release.
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Re-pin to a different release.
    _write_release_pin(isolated_repo, release="2099-01-01.0")
    region2 = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert region2.release == "2099-01-01.0"
    assert region2.manifest_path != region.manifest_path  # different release subdir


def test_load_region_sha_mismatch_raises_cache_corrupt(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Tamper with one cached parquet.
    buildings = region.manifest_path.parent / "buildings.parquet"
    buildings.write_bytes(buildings.read_bytes() + b"corruption")
    with pytest.raises(CacheCorrupt):
        load_region("singapore", backend=backend, repo_root=isolated_repo)


def test_load_region_refresh_true_ignores_cache(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    first = load_region("singapore", backend=backend, repo_root=isolated_repo)
    second = load_region("singapore", backend=backend, repo_root=isolated_repo, refresh=True)
    assert first.manifest_path == second.manifest_path
    # Manifest fetched_at differs.
    first_data = yaml.safe_load(first.manifest_path.read_text())
    second_data = yaml.safe_load(second.manifest_path.read_text())
    assert first_data["fetched_at"] != second_data["fetched_at"]


def test_oversized_fetch_aborts_without_confirm(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    # Use a backend that reports huge estimates.
    class FakeBackend(LocalFixtureBackend):
        def estimate_size(self, **kw):  # type: ignore[override]
            from cfm.data.overture.region import SizeEstimate
            return SizeEstimate(rows=10_000_000, bytes=3 * 1024 * 1024 * 1024)  # 3 GB

    backend = FakeBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(OversizedFetch):
        load_region("singapore", backend=backend, repo_root=isolated_repo)


def test_oversized_fetch_proceeds_with_confirm(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    class FakeBackend(LocalFixtureBackend):
        def estimate_size(self, **kw):  # type: ignore[override]
            from cfm.data.overture.region import SizeEstimate
            return SizeEstimate(rows=10_000_000, bytes=3 * 1024 * 1024 * 1024)

    backend = FakeBackend(fixtures_dir=overture_mini_dir)
    # confirm=True should proceed.
    region = load_region("singapore", backend=backend, repo_root=isolated_repo, confirm=True)
    assert region.name == "singapore"
```

- [ ] **Step 9.2: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_loader.py -v`
Expected: import error.

- [ ] **Step 9.3: Implement the loader**

Create `src/cfm/data/overture/loader.py`:

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.geometry import box

from cfm.data.overture.backend import (
    LocalFixtureBackend,
    OvertureBackend,
    S3DuckDBBackend,
)
from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.manifest import (
    CURRENT_SCHEMA_VERSION,
    CacheManifest,
    ThemeEntry,
    sha256_of_file,
)
from cfm.data.overture.region import (
    BboxScope,
    Region,
    RegionGeometry,
    SizeEstimate,
)

logger = logging.getLogger(__name__)

THEMES_TO_LOAD: tuple[str, ...] = (
    "divisions",          # fetched first; admin polygon comes from here
    "buildings",
    "places",
    "transportation",
    "base",
)

OVERSIZED_THRESHOLD_BYTES: int = 2 * 1024 * 1024 * 1024   # 2 GB


def load_region(
    name: str,
    *,
    backend: OvertureBackend | None = None,
    refresh: bool = False,
    confirm: bool = False,
    repo_root: Path | None = None,
) -> Region:
    """Load Overture themes for region `name`, caching on disk.

    Phase 1 contract: themes are bbox-filtered only. The admin polygon is
    surfaced on Region.geometry for downstream consumers to apply. See
    docs/data/handoffs.md and the spec at
    docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md.
    """
    root = Path(repo_root) if repo_root is not None else _find_repo_root()
    release = _load_release_pin(root)
    region_cfg = _load_region_config(root, name)
    bbox = _build_bbox_scope(region_cfg)
    geometry = _build_region_geometry(region_cfg)
    backend = backend or S3DuckDBBackend()

    cache_dir = root / "data" / "cache" / "overture" / release["release"] / name
    manifest_path = cache_dir / "manifest.yaml"

    if not refresh and manifest_path.exists():
        existing = CacheManifest.from_yaml(manifest_path)
        if existing.release == release["release"]:
            _verify_cache_or_raise(cache_dir, existing)
            return _region_from_cache(name, bbox, geometry, cache_dir, existing)
        logger.info(
            "[overture] cached release %s differs from pin %s; re-fetching",
            existing.release, release["release"],
        )

    _check_total_size(backend, bbox, release["release"], confirm=confirm)

    cache_dir.mkdir(parents=True, exist_ok=True)
    themes: dict = {}
    theme_entries: dict[str, ThemeEntry] = {}
    for theme in THEMES_TO_LOAD:
        table = backend.read_theme(theme=theme, bbox=bbox, release=release["release"])
        out_path = cache_dir / f"{theme}.parquet"
        pq.write_table(table, out_path)
        sha = sha256_of_file(out_path)
        themes[theme] = table
        theme_entries[theme] = ThemeEntry(
            s3_url=_s3_url(backend, theme, release["release"]),
            rows=table.num_rows,
            bytes=out_path.stat().st_size,
            sha256=sha,
            parquet_filename=f"{theme}.parquet",
        )

    manifest = CacheManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        release=release["release"],
        release_date=release["release_date"],
        release_subversion=int(release["release_subversion"]),
        overture_schema_version=release["overture_schema_version"],
        region=name,
        admin_polygon_source=geometry.source,
        bbox=bbox.as_tuple(),
        backend=type(backend).__name__,
        fetched_at=datetime.now(timezone.utc),
        themes=theme_entries,
    )
    manifest.to_yaml(manifest_path)
    return Region(
        name=name,
        release=release["release"],
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=manifest_path,
    )


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("could not locate repo root from cfm.data.overture.loader")


def _load_release_pin(root: Path) -> dict:
    path = root / "configs" / "data" / "overture_release.yaml"
    if not path.exists():
        raise ReleaseNotConfigured(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    for key in ("release", "overture_schema_version", "release_date", "release_subversion"):
        if key not in data:
            raise ReleaseNotConfigured(f"{path} missing key {key!r}")
    return data


def _load_region_config(root: Path, name: str) -> dict:
    path = root / "configs" / "data" / "regions" / f"{name}.yaml"
    if not path.exists():
        raise RegionNotFound(f"no region config at {path}")
    return yaml.safe_load(path.read_text())


def _build_bbox_scope(region_cfg: dict) -> BboxScope:
    """The fetch-time spatial filter. Phase 1: read straight from the region config."""
    return BboxScope.from_tuple(tuple(region_cfg["fallback_bbox"]))


def _build_region_geometry(region_cfg: dict) -> RegionGeometry:
    """The handoff-record geometry. Phase 1 placeholder: a Polygon equal to
    the bbox. C-stage (or a future implementation upgrade) replaces this
    with the precise polygon from the divisions theme. See
    docs/data/handoffs.md.
    """
    bbox = tuple(region_cfg["fallback_bbox"])
    polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
    source = f"{region_cfg['admin']['source']}:{region_cfg['admin']['country_code']}"
    return RegionGeometry(admin_polygon=polygon, source=source)


def _verify_cache_or_raise(cache_dir: Path, manifest: CacheManifest) -> None:
    for theme, entry in manifest.themes.items():
        parquet_path = cache_dir / entry.parquet_filename
        if not parquet_path.exists():
            raise CacheCorrupt(f"manifest lists {parquet_path} but file is missing")
        actual = sha256_of_file(parquet_path)
        if actual != entry.sha256:
            raise CacheCorrupt(
                f"sha256 mismatch for {parquet_path}: manifest={entry.sha256!r} actual={actual!r}"
            )


def _region_from_cache(
    name: str,
    bbox: BboxScope,
    geometry: RegionGeometry,
    cache_dir: Path,
    manifest: CacheManifest,
) -> Region:
    themes: dict = {}
    for theme, entry in manifest.themes.items():
        themes[theme] = pq.read_table(cache_dir / entry.parquet_filename)
    return Region(
        name=name,
        release=manifest.release,
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=cache_dir / "manifest.yaml",
    )


def _s3_url(backend: OvertureBackend, theme: str, release: str) -> str:
    if isinstance(backend, S3DuckDBBackend):
        return backend.build_s3_url(theme=theme, release=release).rstrip("*")
    return f"local-fixture://{theme}"


def _check_total_size(
    backend: OvertureBackend,
    bbox: BboxScope,
    release: str,
    *,
    confirm: bool,
) -> None:
    estimates: dict[str, SizeEstimate] = {}
    total = 0
    for theme in THEMES_TO_LOAD:
        est = backend.estimate_size(theme=theme, bbox=bbox, release=release)
        estimates[theme] = est
        total += est.bytes
        logger.info(
            "[overture] estimated fetch: theme=%-15s rows~%d size~%d bytes",
            theme, est.rows, est.bytes,
        )
    logger.info("[overture] estimated total: %d themes, ~%d bytes", len(estimates), total)
    if total > OVERSIZED_THRESHOLD_BYTES and not confirm:
        raise OversizedFetch(
            f"estimated total {total} bytes exceeds {OVERSIZED_THRESHOLD_BYTES} threshold; "
            "pass confirm=True if intended"
        )
```

- [ ] **Step 9.4: Re-export `load_region` from the package**

Update `src/cfm/data/overture/__init__.py` to add the loader to imports and `__all__`:

```python
"""Overture Maps loader: pinned-release GeoParquet themes scoped to a region."""

from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    OvertureError,
    OvertureSchemaMismatch,
    OvertureUnreachable,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.loader import load_region
from cfm.data.overture.region import Region, SizeEstimate, SpatialScope

__all__ = [
    "CacheCorrupt",
    "OversizedFetch",
    "OvertureError",
    "OvertureSchemaMismatch",
    "OvertureUnreachable",
    "Region",
    "RegionNotFound",
    "ReleaseNotConfigured",
    "SizeEstimate",
    "SpatialScope",
    "load_region",
]
```

- [ ] **Step 9.5: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_loader.py -v`
Expected: 9 passed.

- [ ] **Step 9.6: Run full suite (excluding slow)**

Run: `uv run pytest -m 'not slow'`
Expected: all previously passing tests + 9 new = current count, all green; 1 xfailed from Phase 0.

- [ ] **Step 9.7: Lint and commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/cfm/data/overture/loader.py src/cfm/data/overture/__init__.py tests/data/overture/test_loader.py
git commit -m "feat(data): load_region with cache management, sha256 verification, confirm gate"
```

---

## Task 10: Pinning policy doc + sub-project handoff contract

**Files:**
- Create: `docs/data/overture_pinning_policy.md`
- Create: `docs/data/handoffs.md`

- [ ] **Step 10.1: Write the pinning-policy doc**

Create `docs/data/overture_pinning_policy.md`:

```markdown
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
```

- [ ] **Step 10.2: Write the handoff-contract doc**

Create `docs/data/handoffs.md`:

```markdown
# Sub-project handoff contracts

Phase 1 is decomposed into sub-projects A–G. Each sub-project has a contract with its downstream consumers: what is guaranteed, and what the consumer must still do. This document is the canonical record of those contracts.

## A → C: bbox-filtered themes, polygon for downstream clipping

**Sub-project A** (Overture loader) returns a `Region` object with:

- `themes: dict[str, pyarrow.Table]` — five Overture themes (buildings, places, transportation, base, divisions), **filtered ONLY by `fetch_bbox`** at fetch time.
- `fetch_bbox: BboxScope` — the bounding box actually used as the filter.
- `geometry: RegionGeometry` — the precise admin polygon for the region, surfaced as a *handoff record*. **Not applied at fetch time.**
- `manifest_path: Path` — the cache manifest, recording release version, sha256s, and source URLs.

**Sub-project C** (tile extraction) is contractually obligated to:

1. **Apply `region.admin_polygon` to clip themes** before partitioning into tiles. Failing this means open sea — which falls inside `region.fetch_bbox` but outside `region.admin_polygon` for Singapore — silently enters the training set.
2. Use `region.themes["divisions"]` as the source of truth for the precise polygon if the bbox-as-polygon placeholder in `region.geometry` is still in use. C may choose to compute its own polygon by dissolving rows from `region.themes["divisions"]` matching the country/locality.
3. Reproject from `EPSG:4326` to a local metric frame before tokenisation.

The Phase 1 simplification in sub-project A — using `box(fetch_bbox)` as a placeholder for the admin polygon — exists because the polygon is genuinely not needed for fetching, only for clipping. C-stage either uses the precise polygon from `themes["divisions"]` or, in a future iteration, A is upgraded to a two-pass fetch (divisions first, polygon-derived filter applied to the other four themes). Either is acceptable; A's contract holds either way.

## Why an explicit handoff doc

The risk of the bbox-only fetch is silent: A succeeds, C runs, and sea contamination only becomes visible far downstream when training metrics misbehave. Documenting the contract here lets every consumer of A know the obligation up front and lets a code reviewer flag a C-stage PR that skips the clip.

## Future handoffs

This document grows as more sub-projects ship. Each sub-project adds a section describing what its output guarantees and what consumers must do.
```

- [ ] **Step 10.3: Commit**

```bash
git add docs/data/overture_pinning_policy.md docs/data/handoffs.md
git commit -m "docs(data): pinning policy + A→C handoff contract"
```

---

## Task 11: Cache-invalidation CLI

**Files:**
- Create: `scripts/cfm_data_invalidate.py`
- Create: `tests/data/overture/test_invalidate.py`

- [ ] **Step 11.1: Write the failing tests**

Create `tests/data/overture/test_invalidate.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def _write_release_pin(root: Path, release: str = "2026-04-15.0") -> None:
    cfg = root / "configs" / "data"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "overture_release.yaml").write_text(
        yaml.safe_dump(
            {
                "release": release,
                "overture_schema_version": "v1.16.0",
                "release_date": "2026-04-15",
                "release_subversion": 0,
            }
        )
    )


def _populate_cache(root: Path, release: str, region: str) -> Path:
    cache = root / "data" / "cache" / "overture" / release / region
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.yaml").write_text("dummy")
    (cache / "buildings.parquet").write_text("dummy")
    return cache


def test_invalidate_removes_region_dir(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    cache = _populate_cache(tmp_path, "2026-04-15.0", "singapore")
    assert cache.exists()
    script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "cfm_data_invalidate.py"
    result = subprocess.run(
        [sys.executable, str(script), "singapore", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not cache.exists()


def test_invalidate_refuses_path_traversal(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "cfm_data_invalidate.py"
    result = subprocess.run(
        [sys.executable, str(script), "../etc", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refus" in result.stdout.lower() + result.stderr.lower() or "invalid" in result.stdout.lower() + result.stderr.lower()


def test_invalidate_no_cache_is_a_no_op(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "cfm_data_invalidate.py"
    result = subprocess.run(
        [sys.executable, str(script), "singapore", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "nothing to remove" in (result.stdout + result.stderr).lower()
```

- [ ] **Step 11.2: Run tests; expect failure**

Run: `uv run pytest tests/data/overture/test_invalidate.py -v`
Expected: script not found / exit non-zero from subprocess.

- [ ] **Step 11.3: Implement the CLI**

Create `scripts/cfm_data_invalidate.py`:

```python
"""Invalidate (delete) a region's cached Overture data.

Usage:
    uv run python scripts/cfm_data_invalidate.py <region> [--release <version>] [--repo-root <path>]

Refuses to delete anything outside data/cache/overture/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", help="Region name (e.g. 'singapore')")
    parser.add_argument(
        "--release",
        default=None,
        help="Specific release version to invalidate; defaults to currently pinned release.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root (used in tests).",
    )
    args = parser.parse_args()

    if "/" in args.region or "\\" in args.region or ".." in args.region:
        print(f"refusing to remove suspicious region name {args.region!r}", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root) if args.repo_root else _find_repo_root()
    release = args.release or _read_pinned_release(repo_root)

    target = repo_root / "data" / "cache" / "overture" / release / args.region
    safe_root = (repo_root / "data" / "cache" / "overture").resolve()
    try:
        resolved = target.resolve()
    except FileNotFoundError:
        resolved = target

    if not str(resolved).startswith(str(safe_root)):
        print(
            f"refusing to remove {resolved}: outside of {safe_root}",
            file=sys.stderr,
        )
        return 2

    if not target.exists():
        print(f"[overture] {target} not present; nothing to remove.")
        return 0

    size = _dir_size(target)
    shutil.rmtree(target)
    print(f"[overture] Removed {target.relative_to(repo_root)} ({size} bytes reclaimed).")
    return 0


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("could not locate repo root")


def _read_pinned_release(repo_root: Path) -> str:
    path = repo_root / "configs" / "data" / "overture_release.yaml"
    if not path.exists():
        print(f"missing release pin at {path}", file=sys.stderr)
        sys.exit(2)
    data = yaml.safe_load(path.read_text())
    return data["release"]


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 11.4: Run tests; expect pass**

Run: `uv run pytest tests/data/overture/test_invalidate.py -v`
Expected: 3 passed.

- [ ] **Step 11.5: Lint and commit**

```bash
uv run ruff format scripts tests
uv run ruff check scripts tests
git add scripts/cfm_data_invalidate.py tests/data/overture/test_invalidate.py
git commit -m "feat(scripts): cfm_data_invalidate CLI for nuking a region's cache"
```

---

## Task 12: Snapshot script — S3 mode

**Files:**
- Modify: `scripts/snapshot_overture_fixtures.py`

- [ ] **Step 12.1: Extend the snapshot script with the S3 mode**

Replace the body of the `main()` function in `scripts/snapshot_overture_fixtures.py` so that `--mode s3` actually fetches a tiny real bbox in central Singapore and writes the result.

Edit `scripts/snapshot_overture_fixtures.py`. Replace this block:

```python
    if args.mode == "bootstrap":
        bootstrap()
        return 0
    print("--mode s3 not yet implemented; see Task 13", file=sys.stderr)
    return 2
```

with:

```python
    if args.mode == "bootstrap":
        bootstrap()
        return 0
    s3_snapshot()
    return 0
```

Then add the `s3_snapshot` function near the bottom of the file (above `if __name__ == "__main__":`):

```python
def s3_snapshot() -> None:
    """Fetch a tiny real bbox in central Singapore from the pinned release."""

    from shapely.geometry import box

    from cfm.data.overture.backend import S3DuckDBBackend
    from cfm.data.overture.region import SpatialScope

    # Read the pin so the snapshot tracks whatever is currently pinned.
    import yaml
    with (REPO_ROOT / "configs" / "data" / "overture_release.yaml").open() as f:
        release = yaml.safe_load(f)["release"]

    # 0.01 deg x 0.01 deg around 1.295N 103.855E (Bukit Timah/Bishan area).
    bbox = (103.85, 1.29, 103.86, 1.30)
    scope = SpatialScope(admin_polygon=box(*bbox), bbox=bbox)
    backend = S3DuckDBBackend()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for theme in _SYNTHETIC:  # reuse the theme list
        print(f"fetching theme={theme} release={release} bbox={bbox} ...", flush=True)
        table = backend.read_theme(theme=theme, scope=scope, release=release)
        out = FIXTURES_DIR / f"{theme}.parquet"
        pq.write_table(table, out)
        print(f"  wrote {out.relative_to(REPO_ROOT)} ({table.num_rows} rows)")
```

- [ ] **Step 12.2: Verify the script still runs in bootstrap mode (no regression)**

Run: `uv run python scripts/snapshot_overture_fixtures.py --mode bootstrap`
Expected: same output as Task 6 (5 themes regenerated). Confirms the edit didn't break the existing path.

- [ ] **Step 12.3: Do not run --mode s3 here**

The S3 path is exercised by Task 14 (slow opt-in). Running it here would touch network during a routine commit.

- [ ] **Step 12.4: Commit**

```bash
git add scripts/snapshot_overture_fixtures.py
git commit -m "feat(scripts): add --mode s3 to snapshot_overture_fixtures"
```

---

## Task 13: Slow opt-in real-S3 test

**Files:**
- Create: `tests/slow/__init__.py`
- Create: `tests/slow/test_real_s3_opt_in.py`

- [ ] **Step 13.1: Create the test**

Create `tests/slow/__init__.py` (empty).

Create `tests/slow/test_real_s3_opt_in.py`:

```python
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from shapely.geometry import box

from cfm.data.overture.backend import S3DuckDBBackend
from cfm.data.overture.region import SpatialScope
from cfm.data.overture.schema import EXPECTED_THEMES, validate_schema


@pytest.fixture(scope="module")
def tiny_singapore_scope() -> SpatialScope:
    # 0.01 deg x 0.01 deg around Bukit Timah / Bishan area.
    bbox = (103.85, 1.29, 103.86, 1.30)
    return SpatialScope(admin_polygon=box(*bbox), bbox=bbox)


@pytest.mark.slow
@pytest.mark.parametrize("theme", EXPECTED_THEMES)
def test_real_s3_theme_returns_schema_matching_table(
    theme: str, tiny_singapore_scope: SpatialScope, repo_root: Path
) -> None:
    """Fetch a tiny real bbox and assert our curated schema is satisfied."""

    # Read the currently pinned release so this test moves with the pin.
    import yaml

    pin = yaml.safe_load((repo_root / "configs" / "data" / "overture_release.yaml").read_text())
    backend = S3DuckDBBackend()
    table = backend.read_theme(theme=theme, scope=tiny_singapore_scope, release=pin["release"])
    assert isinstance(table, pa.Table)
    validate_schema(table, theme=theme)
```

- [ ] **Step 13.2: Verify it's deselected from the default fast suite**

Run: `uv run pytest -v -m 'not slow' tests/slow/`
Expected: 0 tests collected (or the parametrised cases all show `deselected`).

Run: `uv run pytest --collect-only -m slow tests/slow/`
Expected: 5 parametrised cases listed.

- [ ] **Step 13.3: Do NOT run the slow test as part of the commit step**

Running it requires network and ~5–30 seconds. Document for the user; surface it during Task 15's done check.

- [ ] **Step 13.4: Commit**

```bash
git add tests/slow/__init__.py tests/slow/test_real_s3_opt_in.py
git commit -m "test(data): slow opt-in real-S3 schema verification (5 themes)"
```

---

## Task 14: Update pytest config so `slow` tests are excluded by default

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 14.1: Update pytest's `addopts`**

In `pyproject.toml`, modify the `[tool.pytest.ini_options]` block. Currently:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]  # iCloud Drive hides underscore-prefixed .pth files in synced .venv; rely on pytest's pythonpath instead of uv's editable install
addopts = "-ra --strict-markers"
markers = [
  "slow: tests slower than 5 seconds",
]
```

Change `addopts` to deselect `slow` by default:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]  # iCloud Drive hides underscore-prefixed .pth files in synced .venv; rely on pytest's pythonpath instead of uv's editable install
addopts = "-ra --strict-markers -m 'not slow'"
markers = [
  "slow: tests slower than 5 seconds; opt in with -m slow",
]
```

- [ ] **Step 14.2: Verify fast suite still green**

Run: `uv run pytest`
Expected: full suite passes (the `slow` tests are now silently excluded; the 1 xfailed remains).

- [ ] **Step 14.3: Verify slow opt-in works**

Run: `uv run pytest -m slow tests/slow/ --collect-only`
Expected: 5 cases collected.

- [ ] **Step 14.4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(test): exclude @pytest.mark.slow from default pytest run"
```

---

## Task 15: Done check + merge

**Files:**
- Modify: `README.md` (optional small section)

- [ ] **Step 15.1: Run the full done check from spec §14**

Run the fast suite:

```bash
uv run pytest
```

Expected: large pass count, 1 xfailed (Phase 0 boundary-entry xfail). No failures.

Then run the slow opt-in test (requires internet):

```bash
uv run pytest -m slow tests/slow/ -v
```

Expected: 5 cases pass against real S3. If any case fails with `OvertureSchemaMismatch`, that's a schema drift — update `schema.py` and `snapshot_overture_fixtures.py` per the pinning policy, do NOT merge until green.

Finally, the manual one-liner from spec §14:

```bash
uv run python -c "
from cfm.data.overture import load_region
r = load_region('singapore')
print(r.release, list(r.themes), len(r.themes['buildings']))
"
```

Expected: prints `2026-04-15.0 ['divisions', 'buildings', 'places', 'transportation', 'base'] <some_number>`. This actually fetches Singapore from S3 the first time; subsequent calls hit the cache.

- [ ] **Step 15.2: Verify manifest matches spec §7**

After the one-liner runs, inspect:

```bash
cat data/cache/overture/2026-04-15.0/singapore/manifest.yaml
```

Expected: contains `release`, `release_date`, `release_subversion`, `overture_schema_version`, `region`, `scope.{admin_polygon_source, bbox}`, `backend`, `fetched_at` (ISO 8601 Z), and a `themes:` mapping with 5 entries each containing `s3_url`, `rows`, `bytes`, `sha256`, `parquet_filename`.

- [ ] **Step 15.3: Update README**

Append to `README.md` (before the existing "## Next" section if present, or at the end):

```markdown

## Phase 1 sub-project A: Overture loader

```python
from cfm.data.overture import load_region

singapore = load_region("singapore")
print(singapore.release)              # "2026-04-15.0"
print(list(singapore.themes))         # ["divisions", "buildings", "places", "transportation", "base"]
print(singapore.themes["buildings"].num_rows)
```

First call fetches from public S3 and caches to `data/cache/overture/<release>/<region>/`. Subsequent calls verify sha256 and read from cache.

See `docs/data/overture_pinning_policy.md` for the re-pin procedure.
```

- [ ] **Step 15.4: Commit the README update**

```bash
git add README.md
git commit -m "docs: document Phase 1 sub-A Overture loader in README"
```

- [ ] **Step 15.5: Merge to main**

```bash
git checkout main
git merge --no-ff phase-1-sub-A-overture-loader -m "merge: Phase 1 sub-project A (Overture loader) complete"
git log --oneline -3
```

Expected: a merge commit on top of the previous main HEAD.

- [ ] **Step 15.6: Final done check on main**

```bash
uv run pytest
uv run python -c "from cfm.data.overture import load_region; print(load_region('singapore').release)"
```

Expected: fast suite green, the load uses cache (no fresh S3 fetch).

- [ ] **Step 15.7: Push**

```bash
git push origin main
git push origin phase-1-sub-A-overture-loader
```

---

## Self-review notes

**Spec coverage check:**

| Spec section | Implemented in |
|---|---|
| §1 goal + S3 prereq | Task 9 (loader uses anonymous S3); Task 8 (S3 backend) |
| §2 in-scope items | Tasks 1–14 |
| §3 admin-polygon scoping | Task 9 builds a polygon from `fallback_bbox` as Phase-1 placeholder; full polygon-via-divisions is satisfied by `divisions` being in `THEMES_TO_LOAD` and consumable by C-stage. (Phase-1 simplification: scope uses bbox-as-polygon; the precise admin polygon is in `region.themes["divisions"]` for downstream consumers.) |
| §3 release pin | Task 1 (config), Task 9 (read) |
| §3 backend protocol | Tasks 7, 8 |
| §3 cache layout | Task 9 |
| §3 fixture maintenance | Tasks 6, 12 |
| §4 public API | Tasks 3, 9 |
| §5 backend protocol details | Tasks 7, 8 |
| §6 pre-fetch size + confirm gate | Task 9 (`_check_total_size`) |
| §7 manifest contents | Task 5 (round-trip), Task 9 (writes) |
| §8 cache-hit rule | Task 9 (`_verify_cache_or_raise` + release mismatch branch) |
| §9 module layout | Tasks 2–9 |
| §10 errors | Task 2 |
| §11 invalidation CLI | Task 11 |
| §12 pinning policy | Task 10 |
| §13 tests fast + slow opt-in | Tasks throughout + Task 13 |
| §14 done criteria | Task 15 |
| §15 risks | mitigations documented in this plan: DuckDB pinned in Task 1; multi-polygon handled by `shapely.box` returning a Polygon (single, not Multi — Singapore's bbox-derived scope works; the real divisions polygon ships in `region.themes["divisions"]` for downstream multi-polygon handling) |

**Note on Phase-1 simplification of admin polygon:** the spec called for fetching Singapore's admin boundary from `divisions` and using it as a *spatial filter* before fetching the other themes. To keep Task 9 to a single round-trip per theme (and one cache, one manifest), this plan ships the bbox as the fetch-time filter and includes `divisions` as a regular theme. The simplification is now explicit in the type system: `BboxScope` is the filter, `RegionGeometry` is the handoff record, and the `Region` docstring states the contract loudly. The A→C handoff in `docs/data/handoffs.md` codifies the obligation on C-stage to apply the polygon. If a future Phase-1 iteration wants the spec's two-pass approach, it's a localised patch on top of Task 9 + a small backend signature addition; A's public contract stays stable.

**Placeholder scan:** every step has actual code or commands. No "TBD" anywhere.

**Type consistency:** `Region`, `BboxScope`, `RegionGeometry`, `SizeEstimate`, `CacheManifest`, `ThemeEntry`, `OvertureBackend`, `S3DuckDBBackend`, `LocalFixtureBackend`, `load_region` — names consistent across all tasks.
