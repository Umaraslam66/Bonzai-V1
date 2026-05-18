# Phase 1 sub-C Tile Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-tile structured extraction pipeline that turns sub-A's cached Overture themes into `cells.parquet` + `features.parquet` + `crossings.parquet` + `meta.yaml` + `provenance.yaml` per tile, under a region-level `manifest.yaml` + `_SUCCESS` integrity chain.

**Architecture:** Pure-function library in `cfm.data.sub_c` covers each pipeline stage (coords, geom, sea_mask, policy, conditioning, io, determinism, validator_inline, validator_cross_tile, manifest). A `pipeline.py` orchestrator composes them; a `multiprocessing.Pool` model parallelizes across tiles with shared once-per-region inputs (densified admin polygon, derived sea polygons). Two CLI scripts (`extract_tiles.py`, `validate_extraction.py`) drive extraction and the cross-tile-validator gate.

**Tech Stack:** Python 3.11+, pyarrow, shapely, pyproj, PyYAML, pytest, uv. Reuses sub-A's `cfm.data.overture.load_region` for cache-hit reads + B2's `cfm.data.vocab_derivation.canonicalize_yaml` for YAML determinism.

**Spec reference:** `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md` (committed at sub-C-spec commit).

---

## PREREQUISITE — B2 follow-up (NOT included in this plan)

Sub-C implementation **cannot start** until the B2 follow-up has landed. Per spec §3, B2's `_LOCKED_MISSING_POLICIES` dict (at `src/cfm/data/vocab_derivation.py:326`) must extend from `{field: (type, rationale, is_provisional)}` to `{field: {missing_value: (...), not_in_vocab: (...)}}`, with the four-case `not_in_vocab` values from spec §10.2. `configs/data/missing_value_policy.yaml` regenerated. B2 spec §8 + §10 updated. B2 tests updated.

**This is a B2-scoped half-day follow-up; it should have its own spec/plan, NOT be bundled here.** Task 0 of this plan verifies the prerequisite has landed before any sub-C work starts.

---

## Branch discipline (every task)

- All work develops on branch `phase-1-sub-C-tile-extraction` (created in Task 0 from `main`).
- Task-by-task commits with conventional-commit prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `data:`, `expt:`).
- **Local-first; no PR flow; no push to remote.** Merge to `main` via `git merge --no-ff phase-1-sub-C-tile-extraction` at sub-C end (Task 25), matching the sub-A/B1/B2 pattern.
- **Every implementer-subagent dispatch prompt MUST explicitly state:** "Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-C-tile-extraction` branch via the user's git config." The Task 3 incident from sub-B2 (subagent created its own branch and opened a PR) must not recur.

## Test discipline (every task)

If a test fails because real Singapore data violates an assumed invariant, **STOP and escalate**. Do not modify the assertion. The sub-C-specific case to watch: the kept-cell rule (`NOT (sea_water_fraction >= 1.0 - EPS_RATIO AND zero non-sea features)`) — if a real tile triggers this, the 2.5a drop-rule wasn't applied. Fix the code, not the test. Per auto-memory `feedback_test_weakening_to_pass.md`.

---

## File map

**Create (under `src/cfm/data/sub_c/`):**
- `__init__.py` — public exports per spec §15.1
- `epsilon.py` — `EPS_RATIO`, `EPS_COORD_M`, `EPS_AREA_M2`, `EPS_LENGTH_M` constants
- `enums.py` — int8 enum mappings (GEOMETRY_TYPE, FEATURE_CLASS, AXIS, EVENT_TYPE, COASTAL_RIVER)
- `errors.py` — `PolicyError`, `TileValidationError` with structured payload
- `coords.py` — `reproject_to_local_metric`, tile-ID derivation, `densify_polygon`, `clip_to_admin_polygon`, `partition_into_tiles`
- `geom.py` — `partition_into_cells`, split-at-boundaries, crossing-record derivation, `apply_sliver_drop`
- `sea_mask.py` — `derive_sea_polygons`, `apply_sea_mask`, `compute_sea_overlap_fraction`
- `policy.py` — `apply_missing_value_policy` with closed handler-map
- `conditioning.py` — `compute_conditioning_per_tile`
- `io.py` — `_PARQUET_WRITE_KWARGS`, write helpers, WKB byte-order pin, canonicalize_yaml re-export
- `determinism.py` — `EXCLUDED_FROM_SHA`, sha helpers
- `validator_inline.py` — per-tile inline validator (8 named invariants)
- `validator_cross_tile.py` — cross-tile validator logic (called by script)
- `manifest.py` — `RegionManifest` dataclass, `write_manifest`, `_SUCCESS` write protocol
- `pipeline.py` — orchestrator with `multiprocessing.Pool` worker model

**Create (scripts):**
- `scripts/extract_tiles.py` — CLI per spec §15.2
- `scripts/validate_extraction.py` — CLI per spec §15.2

**Create (tests):**
- `tests/data/sub_c/test_epsilon_enums.py` — Layer 1 constants tests
- `tests/data/sub_c/test_coords.py` — Layer 1 (~8 tests for reproject/tile-ID/half-open/densify/clip)
- `tests/data/sub_c/test_geom.py` — Layer 1 (~10 tests for split/crossing/sliver/edge-cases)
- `tests/data/sub_c/test_sea_mask.py` — Layer 1 (~8 tests for sea def/cell-mask/feature-overlap)
- `tests/data/sub_c/test_policy.py` — Layer 1 (~7 tests for 4-case policy + signature)
- `tests/data/sub_c/test_conditioning.py` — Layer 1 (~5 tests for derive_per_tile + 500m threshold)
- `tests/data/sub_c/test_io_determinism.py` — Layer 1 (~6 tests for parquet config/WKB/YAML/EXCLUDED_FROM_SHA)
- `tests/data/sub_c/test_validator_inline.py` — Layer 1 + Layer 2 (~10 tests for 8 invariants + structured payload)
- `tests/data/sub_c/test_pipeline_torture_tile.py` — Layer 2 (~6 tests on torture-test fixture)
- `tests/data/sub_c/test_cross_tile_validator.py` — Layer 2 (~5 tests on cross-tile micro-fixture)
- `tests/data/sub_c/test_determinism.py` — Layer 2 (~5 tests for pool-size + sha stability)
- `tests/data/sub_c/test_singapore_integration.py` — Layer 3 (~4 tests on cached Singapore)

**Create (fixtures):**
- `tests/fixtures/sub_c/__init__.py`
- `tests/fixtures/sub_c/build_torture_tile.py` — declarative synthetic Overture-shaped tile
- `tests/fixtures/sub_c/build_cross_tile_fixture.py` — 2-tile minimal fixture for cross-tile-validator failure tests

**Create (lint):**
- `.pre-commit-config.yaml` (modify if exists; create if not) — add `no-pandas-in-write-path` hook
- `scripts/lint/no_pandas_in_write_path.py` — the hook implementation

**Modify:**
- `src/cfm/data/__init__.py` — add `sub_c` re-exports
- `docs/known_issues.md` — add 2 new entries (Sweden densification revisit; tokenizer enhancement training-path dependency)

---

## Phase 0 — Prerequisites + branch

### Task 0: Verify B2 follow-up gate + create branch

**Files:** none modified (verification only)

**Dependencies:** none

**Complexity:** trivial

- [ ] **Step 1: Verify B2 follow-up has landed**

Run:
```bash
uv run python -c "
from cfm.data.vocab_derivation import _LOCKED_MISSING_POLICIES
sample = _LOCKED_MISSING_POLICIES['buildings.class']
# After B2 follow-up: sample is {'missing_value': (type, rationale, prov), 'not_in_vocab': (...)}
# Before B2 follow-up: sample is (type, rationale, prov)
assert isinstance(sample, dict), 'B2 follow-up NOT LANDED — missing_value_policy still uses 3-tuple shape; abort sub-C'
assert 'missing_value' in sample and 'not_in_vocab' in sample, \
    'B2 follow-up incomplete — missing not_in_vocab axis; abort sub-C'
print('B2 follow-up OK: four-case schema present')
"
```
Expected: prints `B2 follow-up OK: four-case schema present`. If AssertionError, the B2 follow-up has not landed; stop and complete it before any sub-C work.

- [ ] **Step 2: Confirm `configs/data/missing_value_policy.yaml` carries the not_in_vocab axis**

Run:
```bash
uv run python -c "
import yaml
with open('configs/data/missing_value_policy.yaml') as f:
    policy = yaml.safe_load(f)
buildings = policy['fields']['buildings.class']['policies']
assert 'missing_value' in buildings and 'not_in_vocab' in buildings, \
    'YAML missing not_in_vocab axis; regenerate via scripts/derive_phase1_vocab.py'
print('YAML OK: not_in_vocab axis present')
"
```
Expected: prints `YAML OK: not_in_vocab axis present`.

- [ ] **Step 3: Verify clean working tree on `main`**

Run:
```bash
git status
git rev-parse --abbrev-ref HEAD
```
Expected: branch is `main`; no uncommitted changes. If anything else, abort and reconcile.

- [ ] **Step 4: Create sub-C feature branch**

Run:
```bash
git checkout -b phase-1-sub-C-tile-extraction
git rev-parse --abbrev-ref HEAD
```
Expected: branch is `phase-1-sub-C-tile-extraction`.

- [ ] **Step 5: Confirm fast-suite passes on baseline**

Run: `uv run pytest -q`
Expected: `187 passed, 6 deselected, 1 xfailed` (or whatever B2 follow-up bumped this to — should still be all-pass).

No commit needed (no files modified).

---

## Phase 1 — Module skeleton + foundations

### Task 1: Create `cfm.data.sub_c` package + epsilon/enums/errors

**Files:**
- Create: `src/cfm/data/sub_c/__init__.py`
- Create: `src/cfm/data/sub_c/epsilon.py`
- Create: `src/cfm/data/sub_c/enums.py`
- Create: `src/cfm/data/sub_c/errors.py`
- Create: `tests/data/sub_c/__init__.py`
- Create: `tests/data/sub_c/test_epsilon_enums.py`

**Spec sections:** §4.3, §14.3, §14.4, §17

**Determinism categories satisfied:** C (EPSILON table), E (int8 enums)

**Named tests:**
- `test_epsilon_values_match_spec_table`
- `test_int8_enum_mappings_match_spec`
- `test_geometry_type_enum_round_trip`
- `test_feature_class_enum_round_trip`
- `test_axis_enum_round_trip`
- `test_event_type_enum_round_trip`
- `test_coastal_river_enum_round_trip`
- `test_policy_error_subclass_of_value_error`
- `test_tile_validation_error_payload_structure`

**Dependencies:** Task 0

**Complexity:** small

- [ ] **Step 1: Write the failing tests for epsilon constants**

Create `tests/data/sub_c/test_epsilon_enums.py`:

```python
from __future__ import annotations

import pytest

from cfm.data.sub_c.epsilon import (
    EPS_RATIO,
    EPS_COORD_M,
    EPS_AREA_M2,
    EPS_LENGTH_M,
)
from cfm.data.sub_c.enums import (
    GEOMETRY_TYPE,
    FEATURE_CLASS,
    AXIS,
    EVENT_TYPE,
    COASTAL_RIVER,
    encode_enum,
    decode_enum,
)
from cfm.data.sub_c.errors import PolicyError, TileValidationError


def test_epsilon_values_match_spec_table():
    assert EPS_RATIO == 1e-9
    assert EPS_COORD_M == 1e-6
    assert EPS_AREA_M2 == 1e-6
    assert EPS_LENGTH_M == 1e-6


def test_int8_enum_mappings_match_spec():
    assert GEOMETRY_TYPE == {0: "Point", 1: "LineString", 2: "Polygon"}
    assert FEATURE_CLASS == {0: "road", 1: "building", 2: "poi", 3: "base"}
    assert AXIS == {0: "x", 1: "y"}
    assert EVENT_TYPE == {0: "enter", 1: "exit", 2: "interval"}
    assert COASTAL_RIVER == {0: "inland", 1: "coastal", 2: "riverside", 3: "coastal_riverside"}


@pytest.mark.parametrize("mapping", [GEOMETRY_TYPE, FEATURE_CLASS, AXIS, EVENT_TYPE, COASTAL_RIVER])
def test_enum_round_trip(mapping):
    for code, label in mapping.items():
        assert encode_enum(mapping, label) == code
        assert decode_enum(mapping, code) == label


def test_geometry_type_enum_round_trip():
    assert encode_enum(GEOMETRY_TYPE, "Polygon") == 2
    assert decode_enum(GEOMETRY_TYPE, 1) == "LineString"


def test_feature_class_enum_round_trip():
    assert encode_enum(FEATURE_CLASS, "poi") == 2


def test_axis_enum_round_trip():
    assert encode_enum(AXIS, "y") == 1


def test_event_type_enum_round_trip():
    assert encode_enum(EVENT_TYPE, "interval") == 2


def test_coastal_river_enum_round_trip():
    assert encode_enum(COASTAL_RIVER, "coastal_riverside") == 3


def test_policy_error_subclass_of_value_error():
    assert issubclass(PolicyError, ValueError)


def test_tile_validation_error_payload_structure():
    err = TileValidationError(
        tile="tile=EPSG3414_i12_j17",
        invariant="bbox_matches_wkb",
        failed_row={"source_feature_id": "abc", "row_index": 341},
        detail={"stored": (0.0, 0.0, 1.0, 1.0), "actual": (0.0, 0.1, 1.0, 1.0)},
    )
    assert err.tile == "tile=EPSG3414_i12_j17"
    assert err.invariant == "bbox_matches_wkb"
    assert err.failed_row == {"source_feature_id": "abc", "row_index": 341}
    assert "bbox_matches_wkb" in str(err)
```

- [ ] **Step 2: Run tests, confirm ImportError**

Run: `uv run pytest tests/data/sub_c/test_epsilon_enums.py -v`
Expected: `ImportError` on `cfm.data.sub_c.epsilon` (module doesn't exist).

- [ ] **Step 3: Create the package skeleton**

Create `src/cfm/data/sub_c/__init__.py`:

```python
"""Sub-C tile extraction pipeline.

See docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md
"""

from cfm.data.sub_c.epsilon import EPS_RATIO, EPS_COORD_M, EPS_AREA_M2, EPS_LENGTH_M
from cfm.data.sub_c.enums import (
    GEOMETRY_TYPE,
    FEATURE_CLASS,
    AXIS,
    EVENT_TYPE,
    COASTAL_RIVER,
    encode_enum,
    decode_enum,
)
from cfm.data.sub_c.errors import PolicyError, TileValidationError

__all__ = [
    "EPS_RATIO",
    "EPS_COORD_M",
    "EPS_AREA_M2",
    "EPS_LENGTH_M",
    "GEOMETRY_TYPE",
    "FEATURE_CLASS",
    "AXIS",
    "EVENT_TYPE",
    "COASTAL_RIVER",
    "encode_enum",
    "decode_enum",
    "PolicyError",
    "TileValidationError",
]
```

Create `src/cfm/data/sub_c/epsilon.py`:

```python
"""Per-quantity-type EPSILON constants for structural-boundary comparisons.

Per spec §4.3 / §14.4:
- Apply EPSILON at STRUCTURAL boundaries (0, 1, computed-value equality).
- Do NOT apply EPSILON at USER thresholds (500m, 0.01m²); use strict comparison.

See auto-memory feedback_epsilon_structural_vs_user_threshold.md.
"""

from __future__ import annotations

EPS_RATIO: float = 1e-9
"""For [0, 1] ratio comparisons: sea_water_fraction, water_fraction, sea_overlap_fraction."""

EPS_COORD_M: float = 1e-6
"""For SVY21 meter coordinate equality: bbox match validator, cross-run coord comparisons."""

EPS_AREA_M2: float = 1e-6
"""For m² area equality: area-weighted-mean validator, cell_area_admin_clipped_m2 > 0 check."""

EPS_LENGTH_M: float = 1e-6
"""For meter length equality (NOT for the 500m user threshold)."""
```

Create `src/cfm/data/sub_c/enums.py`:

```python
"""int8 enum mappings for sub-C output columns.

Per spec §14.3: small enums are int8 (not strings) for byte-determinism
across PyArrow versions. Open string columns (class_raw, subtype_raw,
categories_primary, categories_alternate, admin_region, morphology_class,
era_class) stay as strings — Overture-driven unbounded domain.

Adding a value to a closed enum = append-only-within-phase; triggers
sub_c_schema_version bump per spec §14.9.
"""

from __future__ import annotations

GEOMETRY_TYPE: dict[int, str] = {0: "Point", 1: "LineString", 2: "Polygon"}
FEATURE_CLASS: dict[int, str] = {0: "road", 1: "building", 2: "poi", 3: "base"}
AXIS: dict[int, str] = {0: "x", 1: "y"}
EVENT_TYPE: dict[int, str] = {0: "enter", 1: "exit", 2: "interval"}
COASTAL_RIVER: dict[int, str] = {
    0: "inland",
    1: "coastal",
    2: "riverside",
    3: "coastal_riverside",
}


def encode_enum(mapping: dict[int, str], label: str) -> int:
    """Reverse lookup: label → int8 code. Raises KeyError if label unknown."""
    for code, value in mapping.items():
        if value == label:
            return code
    raise KeyError(f"label {label!r} not in enum mapping {mapping!r}")


def decode_enum(mapping: dict[int, str], code: int) -> str:
    """Forward lookup: int8 code → label. Raises KeyError if code unknown."""
    return mapping[code]
```

Create `src/cfm/data/sub_c/errors.py`:

```python
"""Sub-C exception types.

PolicyError: unknown missing-value policy type encountered in YAML.
TileValidationError: inline + cross-tile validator failures, with structured
payload for diagnostic determinism (spec §12.4 + §13.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PolicyError(ValueError):
    """Raised by apply_missing_value_policy when an unknown policy type is encountered."""


@dataclass
class TileValidationError(Exception):
    """Structured payload for inline + cross-tile validator failures.

    The fields (tile, invariant, failed_row, detail) form the canonical
    diagnostic payload that must be byte-deterministic across runs given
    identical input — tested by test_validator_diagnostic_payloads_byte_deterministic.
    """

    tile: str
    invariant: str
    failed_row: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(
            f"TileValidationError on {self.tile}: invariant={self.invariant}, "
            f"row={self.failed_row}, detail={self.detail}"
        )
```

Create `tests/data/sub_c/__init__.py` (empty file).

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/sub_c/test_epsilon_enums.py -v`
Expected: all tests pass.

- [ ] **Step 5: Add sub_c to cfm.data re-exports**

Modify `src/cfm/data/__init__.py` (append):

```python
# Sub-C tile extraction (Phase 1)
from cfm.data import sub_c  # noqa: F401
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_c/ tests/data/sub_c/__init__.py tests/data/sub_c/test_epsilon_enums.py src/cfm/data/__init__.py
git commit -m "$(cat <<'EOF'
feat(sub_c): package skeleton with EPSILON constants, int8 enums, errors

Establishes cfm.data.sub_c with the foundational primitives every later
module depends on: EPS_* constants (structural-boundary float comparisons
per spec §4.3 / §14.4), int8 enum mappings (geometry_type / feature_class /
axis / event_type / coastal_river per spec §14.3), and the structured
TileValidationError payload (spec §12.4).
EOF
)"
```

---

## Phase 2 — Coordinate handling (spec §7)

### Task 2: Reprojection + tile-ID derivation

**Files:**
- Create: `src/cfm/data/sub_c/coords.py`
- Create: `tests/data/sub_c/test_coords.py`

**Spec sections:** §7.1 (CRS), §7.2 (tile origin / half-open grid), §6 (pipeline ordering)

**Determinism categories satisfied:** A (EPSG:3414 integer code), B (floor-division determinism, half-open interval)

**Named tests:**
- `test_reproject_lonlat_to_svy21_byte_deterministic`
- `test_tile_id_from_svy21_point_basic`
- `test_tile_id_half_open_boundary_at_exact_x_equals_2000`
- `test_tile_id_half_open_boundary_at_exact_y_equals_2000`
- `test_co_linear_feature_attaches_to_higher_ij_cell` (uses 250m cell grid; tested again in Task 3 for cell level)

**Dependencies:** Task 1

**Complexity:** small–medium

- [ ] **Step 1: Write failing tests**

Create `tests/data/sub_c/test_coords.py`:

```python
from __future__ import annotations

import pyproj
import pytest
from shapely.geometry import Point

from cfm.data.sub_c.coords import (
    SVY21_EPSG_CODE,
    reproject_lonlat_to_svy21,
    tile_id_from_svy21,
    TILE_SIZE_M,
)


def test_svy21_epsg_code_is_3414():
    assert SVY21_EPSG_CODE == 3414


def test_reproject_lonlat_to_svy21_byte_deterministic():
    # Marina Bay, Singapore: ~103.8587°E, 1.2839°N → SVY21 ~30000, 29000 m
    lon, lat = 103.8587, 1.2839
    x1, y1 = reproject_lonlat_to_svy21(lon, lat)
    x2, y2 = reproject_lonlat_to_svy21(lon, lat)
    assert x1 == x2 and y1 == y2  # bit-identical
    assert 25000 < x1 < 35000
    assert 25000 < y1 < 35000


def test_tile_id_from_svy21_point_basic():
    # SVY21 (3000, 9000) → tile (1, 4) under 2km grid
    assert tile_id_from_svy21(3000.0, 9000.0) == (1, 4)


def test_tile_id_half_open_boundary_at_exact_x_equals_2000():
    # x = 2000.0 belongs to tile i=1 (NOT i=0) per half-open [i*2000, (i+1)*2000)
    assert tile_id_from_svy21(2000.0, 5000.0) == (1, 2)
    # x just below 2000.0 belongs to tile i=0
    assert tile_id_from_svy21(1999.999999, 5000.0) == (0, 2)


def test_tile_id_half_open_boundary_at_exact_y_equals_2000():
    assert tile_id_from_svy21(5000.0, 2000.0) == (2, 1)
    assert tile_id_from_svy21(5000.0, 1999.999999) == (2, 0)


def test_co_linear_feature_attaches_to_higher_ij_cell():
    # A point on the boundary x=4000.0 lives in tile i=2, not i=1
    assert tile_id_from_svy21(4000.0, 4000.0) == (2, 2)


def test_tile_size_constant():
    assert TILE_SIZE_M == 2000
```

- [ ] **Step 2: Run tests, confirm ImportError**

Run: `uv run pytest tests/data/sub_c/test_coords.py -v`
Expected: `ImportError: cannot import name 'SVY21_EPSG_CODE' from 'cfm.data.sub_c.coords'`.

- [ ] **Step 3: Implement coords.py (partial — reprojection + tile-ID only)**

Create `src/cfm/data/sub_c/coords.py`:

```python
"""Coordinate handling for sub-C: reprojection, tile/cell partitioning, densification, clipping.

Per spec §7:
- §7.1 EPSG:3414 (SVY21) for Singapore; polymorphic per region.
- §7.2 CRS-origin-aligned 2km tile grid; half-open intervals.
- §7.3 Reproject everything to SVY21 first, then clip in SVY21.
- §7.4 Polygon densification: no-op for Singapore (max edge 775m < cell size); function
  signature locked for Sweden enrollment to pass max_edge_length_m without re-opening this code.
"""

from __future__ import annotations

import math

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

SVY21_EPSG_CODE: int = 3414  # Singapore SVY21 / Singapore TM
TILE_SIZE_M: int = 2000  # per spec §7.2
CELL_SIZE_M: int = 250  # per spec §7.2 (8×8 grid per tile)


# Reusable transformer; constructed once at module import for determinism.
_TRANSFORMER_4326_TO_SVY21 = pyproj.Transformer.from_crs(
    "EPSG:4326", f"EPSG:{SVY21_EPSG_CODE}", always_xy=True
)


def reproject_lonlat_to_svy21(lon: float, lat: float) -> tuple[float, float]:
    """Project (lon, lat) in EPSG:4326 to (easting, northing) in EPSG:3414 SVY21.

    Determinism: pyproj.Transformer is constructed once per module load; the
    transformation is a Transverse Mercator formula (no datum grid for SVY21);
    output is byte-deterministic given fixed input. See spec §14.1.
    """
    x, y = _TRANSFORMER_4326_TO_SVY21.transform(lon, lat)
    return float(x), float(y)


def reproject_geometry_to_svy21(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a shapely geometry from EPSG:4326 to EPSG:3414 SVY21."""

    def _xy(x: float, y: float, z: float | None = None) -> tuple[float, float]:
        nx, ny = _TRANSFORMER_4326_TO_SVY21.transform(x, y)
        return float(nx), float(ny)

    return shapely_transform(_xy, geom)


def tile_id_from_svy21(x: float, y: float) -> tuple[int, int]:
    """Map an SVY21 (easting, northing) point to its (tile_i, tile_j).

    Half-open convention per spec §7.2: tile (i, j) covers
    [i*TILE_SIZE_M, (i+1)*TILE_SIZE_M) × [j*TILE_SIZE_M, (j+1)*TILE_SIZE_M).
    A point at exactly x = i*TILE_SIZE_M lands in tile i (not i-1).
    """
    i = int(math.floor(x / TILE_SIZE_M))
    j = int(math.floor(y / TILE_SIZE_M))
    return i, j
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/sub_c/test_coords.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/coords.py tests/data/sub_c/test_coords.py
git commit -m "$(cat <<'EOF'
feat(sub_c/coords): reprojection 4326→SVY21 + tile-ID derivation

Locks EPSG:3414 as the SVY21 code (spec §7.1), provides reproject helpers,
and tile_id_from_svy21 with the half-open boundary convention (spec §7.2):
a point at x = i*2000 lands in tile i, not tile i-1. Determinism categories
A (integer EPSG) + B (floor-division half-open) satisfied.

Reuses a module-scoped pyproj.Transformer built once per import for
byte-deterministic reprojection (spec §14.1).
EOF
)"
```

### Task 3: Cell partitioning + densify_polygon

**Files:**
- Modify: `src/cfm/data/sub_c/coords.py` (add cell-grid + densify_polygon + clip_to_admin_polygon + partition_into_tiles)
- Modify: `tests/data/sub_c/test_coords.py` (add cell + densify + clip tests)

**Spec sections:** §7.2 (cell grid), §7.3 (clipping order), §7.4 (densification — no-op for SG)

**Determinism categories satisfied:** B (half-open at cell level), A (densification placement locked)

**Named tests added:**
- `test_cell_id_within_tile_half_open_at_exact_x_equals_250`
- `test_cell_id_within_tile_half_open_at_exact_y_equals_250`
- `test_partition_into_tiles_emits_inventory_sorted_by_ij`
- `test_densify_polygon_with_none_returns_unchanged`
- `test_densify_polygon_with_real_threshold_inserts_vertices_on_long_edges`
- `test_clip_to_admin_polygon_clips_in_svy21`

**Dependencies:** Task 2

**Complexity:** small–medium

- [ ] **Step 1: Write failing tests** (append to `tests/data/sub_c/test_coords.py`):

```python
from shapely.geometry import Polygon, LineString
import pyarrow as pa

from cfm.data.sub_c.coords import (
    CELL_SIZE_M,
    cell_id_within_tile,
    densify_polygon,
    clip_to_admin_polygon,
    partition_into_tiles,
)


def test_cell_size_constant():
    assert CELL_SIZE_M == 250


def test_cell_id_within_tile_half_open_at_exact_x_equals_250():
    # x_in_tile=250 → cell ci=1 (NOT 0)
    assert cell_id_within_tile(250.0, 500.0) == (1, 2)
    assert cell_id_within_tile(249.999999, 500.0) == (0, 2)


def test_cell_id_within_tile_half_open_at_exact_y_equals_250():
    assert cell_id_within_tile(500.0, 250.0) == (2, 1)
    assert cell_id_within_tile(500.0, 249.999999) == (2, 0)


def test_densify_polygon_with_none_returns_unchanged():
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    out = densify_polygon(poly, max_edge_length_m=None)
    assert out.equals(poly)
    # Same vertex count
    assert len(list(out.exterior.coords)) == len(list(poly.exterior.coords))


def test_densify_polygon_with_real_threshold_inserts_vertices_on_long_edges():
    # 4-vertex square 10km on each side; with 1000m threshold should densify
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    out = densify_polygon(poly, max_edge_length_m=1000.0)
    out_n = len(list(out.exterior.coords))
    assert out_n > len(list(poly.exterior.coords))
    # Every edge now ≤ 1000m
    coords = list(out.exterior.coords)
    for a, b in zip(coords[:-1], coords[1:]):
        edge_len = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        assert edge_len <= 1000.0 + 1e-6


def test_clip_to_admin_polygon_clips_in_svy21():
    # admin polygon = unit-square 1km×1km at origin in SVY21
    admin = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    # a feature linestring extending past admin
    feature = LineString([(500, 500), (1500, 500)])
    clipped = clip_to_admin_polygon([feature], admin)
    assert len(clipped) == 1
    assert clipped[0].length == pytest.approx(500.0)  # clipped at x=1000


def test_partition_into_tiles_emits_inventory_sorted_by_ij():
    # admin polygon covering tiles (0,0), (0,1), (1,0), (1,1)
    admin = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])
    inventory = partition_into_tiles(admin)
    keys = list(inventory.keys())
    assert keys == sorted(keys)  # lexicographic sort by (i, j)
    assert (0, 0) in keys
    assert (1, 1) in keys
```

- [ ] **Step 2: Run tests, confirm failures**

Run: `uv run pytest tests/data/sub_c/test_coords.py -v`
Expected: failures on the new imports (`cell_id_within_tile`, `densify_polygon`, `clip_to_admin_polygon`, `partition_into_tiles`).

- [ ] **Step 3: Append implementations to `coords.py`**

Append to `src/cfm/data/sub_c/coords.py`:

```python
def cell_id_within_tile(x_in_tile: float, y_in_tile: float) -> tuple[int, int]:
    """Map an in-tile metric coordinate (0 ≤ x_in_tile < TILE_SIZE_M)
    to its (cell_i, cell_j) within the 8×8 grid. Half-open at cell boundaries
    per spec §7.2; a point at x = c*CELL_SIZE_M lands in cell c (not c-1).
    """
    ci = int(math.floor(x_in_tile / CELL_SIZE_M))
    cj = int(math.floor(y_in_tile / CELL_SIZE_M))
    return ci, cj


def densify_polygon(
    polygon: BaseGeometry,
    max_edge_length_m: float | None,
) -> BaseGeometry:
    """If max_edge_length_m is None, return polygon unchanged (Singapore no-op
    per spec §7.4 — max edge 775m < cell quantization scale).

    Otherwise insert vertices on every edge longer than max_edge_length_m
    so the densified polygon has no edge exceeding the threshold. Sweden
    enrollment passes a real value without re-opening sub-C code.
    """
    if max_edge_length_m is None:
        return polygon

    # Use shapely's segmentize (available in shapely 2.x)
    return polygon.segmentize(max_segment_length=max_edge_length_m)


def clip_to_admin_polygon(
    features: list[BaseGeometry],
    admin_polygon: BaseGeometry,
) -> list[BaseGeometry]:
    """Intersect each feature with admin_polygon (both in SVY21 per spec §7.3).

    Returns the kept sub-geometries (in input order). Empty intersections
    are dropped. Order is preserved so callers can re-associate with
    feature attributes by index.
    """
    out: list[BaseGeometry] = []
    for f in features:
        clipped = f.intersection(admin_polygon)
        if not clipped.is_empty:
            out.append(clipped)
    return out


def partition_into_tiles(
    admin_polygon: BaseGeometry,
) -> dict[tuple[int, int], BaseGeometry]:
    """For each 2km × 2km tile that intersects admin_polygon, return the
    intersection of the tile box with the admin polygon as the tile's
    admin-clipped footprint. Result is sorted by (tile_i, tile_j) for
    byte-determinism (spec §11.7 manifest tiles[] sort).
    """
    from shapely.geometry import box as shapely_box

    min_x, min_y, max_x, max_y = admin_polygon.bounds
    min_i = int(math.floor(min_x / TILE_SIZE_M))
    min_j = int(math.floor(min_y / TILE_SIZE_M))
    max_i = int(math.floor((max_x - 1e-9) / TILE_SIZE_M))
    max_j = int(math.floor((max_y - 1e-9) / TILE_SIZE_M))

    inventory: dict[tuple[int, int], BaseGeometry] = {}
    for i in range(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            tile_box = shapely_box(
                i * TILE_SIZE_M,
                j * TILE_SIZE_M,
                (i + 1) * TILE_SIZE_M,
                (j + 1) * TILE_SIZE_M,
            )
            intersection = tile_box.intersection(admin_polygon)
            if not intersection.is_empty:
                inventory[(i, j)] = intersection

    # Sort by (i, j) for byte-determinism
    return dict(sorted(inventory.items()))
```

- [ ] **Step 4: Run tests, confirm all pass**

Run: `uv run pytest tests/data/sub_c/test_coords.py -v`
Expected: 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/coords.py tests/data/sub_c/test_coords.py
git commit -m "$(cat <<'EOF'
feat(sub_c/coords): cell partitioning, densification, clipping, tile inventory

Adds cell_id_within_tile with the same half-open convention as tiles (spec §7.2),
densify_polygon with signature locked even for Singapore's no-op case (spec §7.4),
clip_to_admin_polygon for the reproject-first-then-clip ordering (spec §7.3), and
partition_into_tiles emitting a (i,j)-sorted inventory for byte-deterministic
manifest assembly (spec §11.7).
EOF
)"
```

---

## Phase 3 — Cell extraction (spec §8)

### Task 4: Split-at-boundaries + crossing-record schema

**Files:**
- Create: `src/cfm/data/sub_c/geom.py`
- Create: `tests/data/sub_c/test_geom.py`

**Spec sections:** §8.1 (split-at-boundaries), §8.2 (8-column schema + sort key), §8.3 (edge cases)

**Determinism categories satisfied:** A (shapely cut-point byte-stability), D (crossings sort key with source_feature_id tie-break), E (int8 enums for axis/event_type)

**Named tests:**
- `test_split_at_boundaries_single_cell_feature_emits_one_subfeature_zero_crossings`
- `test_split_at_boundaries_multi_cell_road_emits_n_subfeatures_n_minus_one_crossings`
- `test_corner_crossing_emits_two_records_one_per_axis`
- `test_polygon_interior_ring_emits_multiple_records_per_source_feature`
- `test_co_linear_entirety_emits_zero_records_attaches_to_higher_ij`
- `test_touch_but_not_cross_emits_zero_records`
- `test_partial_co_linearity_emits_interval_event_with_extent`
- `test_multi_crossing_same_edge_emits_alternating_enter_exit_sorted_by_position`
- `test_crossings_sort_key_canonical`
- `test_apply_sliver_drop_removes_below_threshold_features`

**Dependencies:** Task 3

**Complexity:** large (geometric edge cases are the meat of this sub-project)

- [ ] **Step 1: Write failing tests for the dataclass shapes + simple single-cell case**

Create `tests/data/sub_c/test_geom.py`:

```python
from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon, box as shapely_box

from cfm.data.sub_c.coords import CELL_SIZE_M, TILE_SIZE_M
from cfm.data.sub_c.enums import AXIS, EVENT_TYPE, FEATURE_CLASS, GEOMETRY_TYPE, encode_enum
from cfm.data.sub_c.geom import (
    CellSubFeature,
    CrossingRecord,
    partition_into_cells,
    apply_sliver_drop,
)


def _tile_box(tile_i: int = 0, tile_j: int = 0):
    return shapely_box(
        tile_i * TILE_SIZE_M,
        tile_j * TILE_SIZE_M,
        (tile_i + 1) * TILE_SIZE_M,
        (tile_j + 1) * TILE_SIZE_M,
    )


def test_split_at_boundaries_single_cell_feature_emits_one_subfeature_zero_crossings():
    # A small road entirely inside cell (0, 0) of tile (0, 0)
    feature = LineString([(50, 50), (200, 200)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_001", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 1
    assert subfeatures[0].cell_i == 0
    assert subfeatures[0].cell_j == 0
    assert subfeatures[0].source_feature_id == "road_001"
    assert len(crossings) == 0


def test_split_at_boundaries_multi_cell_road_emits_n_subfeatures_n_minus_one_crossings():
    # Road crossing 3 cells: (0,0) → (1,0) → (2,0) along y=125 from x=100 to x=600
    feature = LineString([(100, 125), (600, 125)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_002", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 3
    assert {s.cell_i for s in subfeatures} == {0, 1, 2}
    assert all(s.cell_j == 0 for s in subfeatures)
    assert len(crossings) == 2
    # All share source_feature_id
    assert all(c.source_feature_id == "road_002" for c in crossings)
    # Both are axis=x crossings (vertical edges between (i,j) and (i+1,j))
    assert all(c.axis == encode_enum(AXIS, "x") for c in crossings)


def test_corner_crossing_emits_two_records_one_per_axis():
    # Road passes through exact corner (250, 250) — cell-boundary corner
    feature = LineString([(125, 125), (375, 375)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_003", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Two sub-features: (0,0) and (1,1)
    assert len(subfeatures) == 2
    # Two crossing records: one x-axis edge, one y-axis edge
    axis_codes = sorted(c.axis for c in crossings)
    assert axis_codes == [encode_enum(AXIS, "x"), encode_enum(AXIS, "y")]
    # Both share source_feature_id
    assert all(c.source_feature_id == "road_003" for c in crossings)


def test_polygon_interior_ring_emits_multiple_records_per_source_feature():
    # Polygon with a hole crossing a cell boundary
    shell = Polygon(
        [(100, 100), (700, 100), (700, 700), (100, 700)],
        holes=[[(200, 200), (600, 200), (600, 600), (200, 600)]],
    )
    subfeatures, crossings = partition_into_cells(
        features=[(shell, "building_004", "building")],
        tile_i=0,
        tile_j=0,
    )
    # Many crossings expected; at least one should have ring_index >= 1 (interior ring)
    assert any(c.ring_index >= 1 for c in crossings)


def test_co_linear_entirety_emits_zero_records_attaches_to_higher_ij():
    # Road lying exactly on cell-boundary y=250
    feature = LineString([(100, 250), (200, 250)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_005", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 1
    # Half-open: y=250 attaches to cell j=1 (the higher-j side)
    assert subfeatures[0].cell_j == 1
    assert len(crossings) == 0


def test_touch_but_not_cross_emits_zero_records():
    # Road ending exactly at boundary x=250
    feature = LineString([(100, 100), (250, 100)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_006", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Per spec §8.3: touch-but-not-cross means feature wholly in one cell, no crossing record
    assert len(subfeatures) == 1
    assert len(crossings) == 0


def test_partial_co_linearity_emits_interval_event_with_extent():
    # Polygon with one shell segment lying along a cell boundary;
    # body spans both adjacent cells
    poly = Polygon([(100, 100), (400, 100), (400, 400), (100, 400)])
    # Cell boundary at x=250 cuts the polygon
    subfeatures, crossings = partition_into_cells(
        features=[(poly, "building_007", "building")],
        tile_i=0,
        tile_j=0,
    )
    # Crossings include intervals on the x=250 edge
    interval_crossings = [c for c in crossings if c.event_type == encode_enum(EVENT_TYPE, "interval")]
    assert len(interval_crossings) >= 1
    assert all(c.edge_extent_length_m > 0 for c in interval_crossings)


def test_multi_crossing_same_edge_emits_alternating_enter_exit_sorted_by_position():
    # Zigzag road crossing x=250 three times
    feature = LineString([
        (100, 100),
        (350, 200),
        (200, 300),
        (350, 400),
    ])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_008", "road")],
        tile_i=0,
        tile_j=0,
    )
    # All crossings on edge between cells (0, *) and (1, *) at axis=x
    x_edge_crossings = [c for c in crossings if c.lower_cell_i == 0 and c.axis == encode_enum(AXIS, "x")]
    # Sort by edge_position_m
    sorted_crossings = sorted(x_edge_crossings, key=lambda c: c.edge_position_m)
    # event_types should alternate enter / exit
    types = [c.event_type for c in sorted_crossings]
    assert len(types) >= 3
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], "enter/exit must alternate"


def test_crossings_sort_key_canonical():
    # Mix of axes, source_feature_ids, ring_indices, event_types, positions
    poly = Polygon([(100, 100), (400, 100), (400, 400), (100, 400)])
    road = LineString([(100, 100), (300, 300)])
    subfeatures, crossings = partition_into_cells(
        features=[(poly, "polygon_009", "building"), (road, "road_010", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Per spec §8.2: sort key (lower_cell_i, lower_cell_j, axis, source_feature_id, ring_index, edge_position_m, event_type)
    sort_keys = [
        (c.lower_cell_i, c.lower_cell_j, c.axis, c.source_feature_id, c.ring_index, c.edge_position_m, c.event_type)
        for c in crossings
    ]
    assert sort_keys == sorted(sort_keys)


def test_apply_sliver_drop_removes_below_threshold_features():
    # A normal feature + a tiny sliver
    normal = LineString([(0, 0), (100, 0)])
    sliver_line = LineString([(0, 0), (0.005, 0)])  # 5 mm
    normal_sub = CellSubFeature(
        cell_i=0, cell_j=0, source_feature_id="n", feature_class="road",
        geometry=normal, geometry_type="LineString",
    )
    sliver_sub = CellSubFeature(
        cell_i=0, cell_j=0, source_feature_id="s", feature_class="road",
        geometry=sliver_line, geometry_type="LineString",
    )
    kept = apply_sliver_drop(
        [normal_sub, sliver_sub],
        area_threshold_m2=0.01,
        length_threshold_m=0.01,
    )
    assert len(kept) == 1
    assert kept[0].source_feature_id == "n"
```

- [ ] **Step 2: Run tests, confirm ImportError**

Run: `uv run pytest tests/data/sub_c/test_geom.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement geom.py**

Create `src/cfm/data/sub_c/geom.py`:

```python
"""Cell partitioning, split-at-boundaries, crossing-record derivation, sliver drop.

Per spec §8 + §8.3 edge cases. Tokenizer (encode.py:_require_in_bounds) requires
per-cell features to fit in cell-local coordinates; split-at-boundaries (§8.1) is
how multi-cell features are made tokenizable.

Crossing records (§8.2) are the raw input from which sub-E derives PRD-§5
boundary contracts; the 8-column schema + canonical sort key are locked here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box as shapely_box
from shapely.geometry.base import BaseGeometry

from cfm.data.sub_c.coords import CELL_SIZE_M, TILE_SIZE_M, cell_id_within_tile
from cfm.data.sub_c.enums import AXIS, EVENT_TYPE, encode_enum


@dataclass(frozen=True)
class CellSubFeature:
    """A piece of a source feature that fits in a single cell.

    Per spec §11.2 features.parquet row corresponds to one CellSubFeature.
    """

    cell_i: int
    cell_j: int
    source_feature_id: str
    feature_class: str  # "road" | "building" | "poi" | "base"
    geometry: BaseGeometry  # CELL-LOCAL coordinates (translated by cell origin)
    geometry_type: str  # "Point" | "LineString" | "Polygon"


@dataclass(frozen=True)
class CrossingRecord:
    """Per spec §8.2 — 8-column schema (column types int8 / string / float64 / int16).

    Canonical sort key (spec §8.2 + §14.2):
    (lower_cell_i, lower_cell_j, axis, source_feature_id, ring_index,
     edge_position_m, event_type).
    """

    source_feature_id: str
    lower_cell_i: int
    lower_cell_j: int
    axis: int  # int8 enum: 0=x, 1=y
    ring_index: int  # 0=exterior shell; ≥1=interior rings
    event_type: int  # int8 enum: 0=enter, 1=exit, 2=interval
    edge_position_m: float  # raw SVY21 meter along edge
    edge_extent_length_m: float  # 0 for point crossings; >0 for polygon edge-intervals


def partition_into_cells(
    features: Iterable[tuple[BaseGeometry, str, str]],
    tile_i: int,
    tile_j: int,
) -> tuple[list[CellSubFeature], list[CrossingRecord]]:
    """Partition features (in SVY21, already clipped to tile bounds) into per-cell
    sub-features and crossing records.

    Args:
      features: iterable of (geometry, source_feature_id, feature_class) tuples
        in SVY21 absolute coordinates; expected to lie within this tile.
      tile_i, tile_j: tile coordinates.

    Returns: (sub_features, crossings).

    Algorithm per spec §8.1 + §8.3 edge cases:
      - For each feature, intersect with each of the 64 cells in the tile.
      - Non-empty intersections become CellSubFeatures (geometry translated to
        cell-local coords [0, CELL_SIZE_M]).
      - For features that produce sub-features in ≥2 cells, derive crossing
        records by examining cell-boundary intersection points.
      - Co-linear-entirety, touch-but-not-cross, polygon interior rings,
        multi-crossing, partial co-linearity all handled.
    """
    tile_origin_x = tile_i * TILE_SIZE_M
    tile_origin_y = tile_j * TILE_SIZE_M

    sub_features: list[CellSubFeature] = []
    crossings: list[CrossingRecord] = []

    for geom, source_id, fclass in features:
        # Determine bbox of geometry in cell-grid terms (relative to tile)
        rel_geom = _translate(geom, -tile_origin_x, -tile_origin_y)
        if rel_geom.is_empty:
            continue
        min_x, min_y, max_x, max_y = rel_geom.bounds
        min_ci = max(0, int(math.floor(min_x / CELL_SIZE_M)))
        min_cj = max(0, int(math.floor(min_y / CELL_SIZE_M)))
        max_ci = min(7, int(math.floor((max_x - 1e-9) / CELL_SIZE_M)))
        max_cj = min(7, int(math.floor((max_y - 1e-9) / CELL_SIZE_M)))

        per_cell_pieces: dict[tuple[int, int], BaseGeometry] = {}
        for ci in range(min_ci, max_ci + 1):
            for cj in range(min_cj, max_cj + 1):
                cell_box_relative = shapely_box(
                    ci * CELL_SIZE_M,
                    cj * CELL_SIZE_M,
                    (ci + 1) * CELL_SIZE_M,
                    (cj + 1) * CELL_SIZE_M,
                )
                piece = rel_geom.intersection(cell_box_relative)
                if piece.is_empty:
                    continue
                # Co-linear/touch-but-not-cross: a degenerate intersection
                # whose area==0 AND length==0 (e.g., a 0-d point on boundary)
                # is not a real "this feature is in this cell" event.
                if isinstance(piece, Point) and not isinstance(geom, Point):
                    # geometry was a line/polygon but intersection collapsed to a point
                    # — that's touch-but-not-cross; skip
                    continue
                per_cell_pieces[(ci, cj)] = piece

        # Co-linear-entirety handling (half-open tie-break): if the source geometry
        # lies entirely on a cell boundary, attach to the higher-ij cell only.
        # Detection: only one or two pieces exist that are essentially zero-area /
        # zero-length except via boundary attachment.
        # For simplicity in this version, the bbox-based loop already produces
        # ≥1 piece; the half-open via floor already handles edges. Co-linear on
        # boundary y=cj*250 will produce piece in cell j and cell j-1 if both
        # are within bbox; we keep only the higher (cell j).
        # We detect by length comparison: any piece with length < EPS_LENGTH whose
        # bounding-line lies exactly on a boundary → drop the lower-side piece.
        # (Real implementation tightens this; the test fixture exercises the
        # canonical case.)

        for (ci, cj), piece in per_cell_pieces.items():
            cell_origin_local_x = ci * CELL_SIZE_M
            cell_origin_local_y = cj * CELL_SIZE_M
            cell_local_geom = _translate(piece, -cell_origin_local_x, -cell_origin_local_y)
            sub_features.append(
                CellSubFeature(
                    cell_i=ci,
                    cell_j=cj,
                    source_feature_id=source_id,
                    feature_class=fclass,
                    geometry=cell_local_geom,
                    geometry_type=cell_local_geom.geom_type,
                )
            )

        # Derive crossing records: for each pair of adjacent cells that both have
        # pieces of this feature, examine the shared edge.
        if len(per_cell_pieces) < 2:
            continue
        for (ci, cj) in per_cell_pieces:
            # x-axis edge: between (ci, cj) and (ci+1, cj)
            if (ci + 1, cj) in per_cell_pieces:
                crossings.extend(
                    _derive_crossings_on_edge(
                        rel_geom,
                        lower_ci=ci,
                        lower_cj=cj,
                        axis_name="x",
                        edge_x=(ci + 1) * CELL_SIZE_M,
                        edge_y_range=(cj * CELL_SIZE_M, (cj + 1) * CELL_SIZE_M),
                        tile_origin_x=tile_origin_x,
                        tile_origin_y=tile_origin_y,
                        source_feature_id=source_id,
                    )
                )
            # y-axis edge: between (ci, cj) and (ci, cj+1)
            if (ci, cj + 1) in per_cell_pieces:
                crossings.extend(
                    _derive_crossings_on_edge(
                        rel_geom,
                        lower_ci=ci,
                        lower_cj=cj,
                        axis_name="y",
                        edge_y=(cj + 1) * CELL_SIZE_M,
                        edge_x_range=(ci * CELL_SIZE_M, (ci + 1) * CELL_SIZE_M),
                        tile_origin_x=tile_origin_x,
                        tile_origin_y=tile_origin_y,
                        source_feature_id=source_id,
                    )
                )

    # Canonical sort per spec §8.2
    crossings.sort(key=lambda c: (
        c.lower_cell_i,
        c.lower_cell_j,
        c.axis,
        c.source_feature_id,
        c.ring_index,
        c.edge_position_m,
        c.event_type,
    ))
    sub_features.sort(key=lambda s: (s.cell_i, s.cell_j, s.feature_class, s.source_feature_id))

    return sub_features, crossings


def _translate(geom: BaseGeometry, dx: float, dy: float) -> BaseGeometry:
    """Translate a shapely geometry by (dx, dy). Avoids the
    affine_transformations submodule overhead."""
    from shapely.affinity import translate
    return translate(geom, xoff=dx, yoff=dy)


def _derive_crossings_on_edge(
    rel_geom: BaseGeometry,
    *,
    lower_ci: int,
    lower_cj: int,
    axis_name: str,  # "x" or "y"
    source_feature_id: str,
    tile_origin_x: float,
    tile_origin_y: float,
    edge_x: float | None = None,
    edge_y: float | None = None,
    edge_x_range: tuple[float, float] | None = None,
    edge_y_range: tuple[float, float] | None = None,
) -> list[CrossingRecord]:
    """Intersect rel_geom with the specified cell-shared edge and emit
    enter/exit/interval CrossingRecord rows.

    edge_position_m: RAW SVY21 meter (absolute, not cell-local) along the
    perpendicular axis (for x-axis edge: position is the y-coordinate;
    for y-axis edge: position is the x-coordinate).
    """
    if axis_name == "x":
        edge_line = LineString([(edge_x, edge_y_range[0]), (edge_x, edge_y_range[1])])
        axis_code = encode_enum(AXIS, "x")
    else:
        edge_line = LineString([(edge_x_range[0], edge_y), (edge_x_range[1], edge_y)])
        axis_code = encode_enum(AXIS, "y")

    intersection = rel_geom.intersection(edge_line)
    if intersection.is_empty:
        return []

    records: list[CrossingRecord] = []
    pieces = _flatten_intersection(intersection)
    for ring_index, piece in pieces:
        # Determine event_type + edge_position_m
        if isinstance(piece, Point):
            pos_local = piece.y if axis_name == "x" else piece.x
            # Position in SVY21 absolute meters: cell-local + tile origin
            pos_absolute = pos_local + (tile_origin_y if axis_name == "x" else tile_origin_x)
            # For a single Point intersection, decide enter or exit by
            # comparing geometry on either side of the edge.
            # For roads (LineString sources), one Point = one crossing event.
            event_type_label = "enter"  # default; caller sorts and alternates
            records.append(
                CrossingRecord(
                    source_feature_id=source_feature_id,
                    lower_cell_i=lower_ci,
                    lower_cell_j=lower_cj,
                    axis=axis_code,
                    ring_index=ring_index,
                    event_type=encode_enum(EVENT_TYPE, event_type_label),
                    edge_position_m=float(pos_absolute),
                    edge_extent_length_m=0.0,
                )
            )
        elif isinstance(piece, LineString):
            # Interval crossing (partial co-linearity per spec §8.3)
            pos_local_start = piece.coords[0][1] if axis_name == "x" else piece.coords[0][0]
            pos_local_end = piece.coords[-1][1] if axis_name == "x" else piece.coords[-1][0]
            extent_local = abs(pos_local_end - pos_local_start)
            pos_absolute = min(pos_local_start, pos_local_end) + (
                tile_origin_y if axis_name == "x" else tile_origin_x
            )
            records.append(
                CrossingRecord(
                    source_feature_id=source_feature_id,
                    lower_cell_i=lower_ci,
                    lower_cell_j=lower_cj,
                    axis=axis_code,
                    ring_index=ring_index,
                    event_type=encode_enum(EVENT_TYPE, "interval"),
                    edge_position_m=float(pos_absolute),
                    edge_extent_length_m=float(extent_local),
                )
            )
    # For zigzag multi-crossing: sort by position then alternate enter/exit
    # (real implementation is more careful; placeholder for the test pattern)
    point_records = [r for r in records if r.edge_extent_length_m == 0.0]
    point_records.sort(key=lambda r: r.edge_position_m)
    for idx, rec in enumerate(point_records):
        # Alternate enter (0) / exit (1) based on position order
        new_event = encode_enum(EVENT_TYPE, "enter" if idx % 2 == 0 else "exit")
        # Replace in records list
        rec_idx = records.index(rec)
        records[rec_idx] = CrossingRecord(
            source_feature_id=rec.source_feature_id,
            lower_cell_i=rec.lower_cell_i,
            lower_cell_j=rec.lower_cell_j,
            axis=rec.axis,
            ring_index=rec.ring_index,
            event_type=new_event,
            edge_position_m=rec.edge_position_m,
            edge_extent_length_m=rec.edge_extent_length_m,
        )
    return records


def _flatten_intersection(intersection: BaseGeometry) -> list[tuple[int, BaseGeometry]]:
    """Return list of (ring_index, geometry_piece) for an intersection result.
    ring_index: 0 for exterior; ≥1 for interior rings (placeholder — real
    implementation distinguishes by source-polygon ring origin).
    """
    if hasattr(intersection, "geoms"):
        return [(0, g) for g in intersection.geoms]
    return [(0, intersection)]


def apply_sliver_drop(
    sub_features: Sequence[CellSubFeature],
    *,
    area_threshold_m2: float = 0.01,
    length_threshold_m: float = 0.01,
) -> list[CellSubFeature]:
    """Drop features whose geometry has area < area_threshold_m2 OR length < length_threshold_m.

    Per spec §11.5: strict comparison (β user-threshold; no EPSILON per §4.3).
    """
    kept: list[CellSubFeature] = []
    for sf in sub_features:
        g = sf.geometry
        if g.geom_type in ("Polygon", "MultiPolygon"):
            if g.area < area_threshold_m2:
                continue
        elif g.geom_type in ("LineString", "MultiLineString"):
            if g.length < length_threshold_m:
                continue
        # Points are never slivers under area/length thresholds
        kept.append(sf)
    return kept
```

> **Implementation note for the engineer:** the `_derive_crossings_on_edge`
> implementation above is a working approximation; the corner-crossing case
> (intersection collapses to single Point on a shared boundary corner)
> needs special handling to emit TWO records (one per axis) — see spec
> §8.3 corner case. The test `test_corner_crossing_emits_two_records_one_per_axis`
> drives the refinement. If the test fails on the simple intersection
> approach above, the geometric logic must detect when a Point sits exactly
> on a corner (4 cells meet) and emit per-axis records for both adjacent
> edges.

- [ ] **Step 4: Run tests, iterate until all pass**

Run: `uv run pytest tests/data/sub_c/test_geom.py -v`
Expected: 10 tests pass. If corner-crossing or interior-ring tests fail, refine `_derive_crossings_on_edge` to handle corners + multi-ring polygons. STOP and escalate per `feedback_test_weakening_to_pass.md` if the test failure looks like a real-data invariant violation rather than a logic gap.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/geom.py tests/data/sub_c/test_geom.py
git commit -m "$(cat <<'EOF'
feat(sub_c/geom): split-at-boundaries + crossing-record derivation

Implements partition_into_cells (spec §8.1) with the 8-column CrossingRecord
schema (spec §8.2) and canonical sort key, handling six edge cases (spec §8.3):
single-cell features, multi-cell roads, corner-crossings (two records per
axis), polygon interior rings (multiple records per source_feature_id),
co-linear-entirety (zero records; higher-ij attachment via half-open),
touch-but-not-cross (zero records), partial co-linearity (interval events),
and multi-crossing zigzag (alternating enter/exit). Adds apply_sliver_drop
with strict β user-threshold comparison (spec §4.3 / §11.5).

Determinism categories satisfied: A (shapely cut-point byte-stability via
fixed-input intersection), D (canonical sort key with source_feature_id
tie-break), E (int8 enum values for axis/event_type).
EOF
)"
```

---

## Phase 4 — Sea masking (spec §9)

### Task 5: derive_sea_polygons + cell-level + feature-level sea logic

**Files:**
- Create: `src/cfm/data/sub_c/sea_mask.py`
- Create: `tests/data/sub_c/test_sea_mask.py`

**Spec sections:** §9.1 (sea definition + derive_sea_polygons as pre-policy view), §9.2 (cell-mask rule), §9.3 (feature-overlap)

**Determinism categories satisfied:** A (cell-local sea geometry cached once per cell — deterministic cache), C (EPS_RATIO at structural boundary)

**Named tests:**
- `test_derive_sea_polygons_filters_class_in_ocean_strait_bay`
- `test_derive_sea_polygons_filters_subtype_ocean`
- `test_derive_sea_polygons_returns_multipolygon_union`
- `test_derive_sea_polygons_runs_against_raw_base_not_policied_themes` (verifies pipeline ordering — spec self-review fix)
- `test_apply_sea_mask_drops_pure_sea_cell_with_zero_non_sea_features`
- `test_apply_sea_mask_keeps_coastal_cell_with_bridge`
- `test_apply_sea_mask_keeps_inland_water_macritchie_like_cell`
- `test_apply_sea_mask_uses_admin_clipped_denominator`
- `test_sea_overlap_fraction_uses_intersects_predicate_for_points`
- `test_sea_overlap_fraction_zero_when_cell_sea_water_fraction_zero` (fast-path)
- `test_sea_overlap_fraction_caches_cell_local_sea_geometry`

**Dependencies:** Tasks 2, 3, 4

**Complexity:** medium–large

- [ ] **Step 1: Write failing tests** — see spec §9 for input shapes. Create `tests/data/sub_c/test_sea_mask.py` with the 11 named tests; use synthetic shapely-based base-theme rows (no Overture cache needed for Layer 1).

- [ ] **Step 2: Run, confirm ImportError**

- [ ] **Step 3: Implement `sea_mask.py`**

Create `src/cfm/data/sub_c/sea_mask.py`:

```python
"""Sea masking: derivation of sea polygons (pre-policy view), cell-level drop
rule, feature-level sea_overlap_fraction.

CRITICAL ORDERING (spec §6 + §9.1 + §5 cross-decision dependency):
derive_sea_polygons MUST run against RAW base theme BEFORE
apply_missing_value_policy. The base.class not-in-vocab drop_row policy
(spec §10.2) would otherwise eliminate ocean/strait/bay rows (35 SG rows
below Strict-300 floor), leaving sea-mask with no polygons to work with.
Sea polygons are masks, not features — features.parquet does NOT contain
sea polygons; the policied themes correctly drop them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from cfm.data.sub_c.epsilon import EPS_RATIO


SEA_CLASS_VALUES = frozenset({"ocean", "strait", "bay"})
SEA_SUBTYPE_VALUES = frozenset({"ocean"})


def derive_sea_polygons(base_theme: pa.Table) -> BaseGeometry:
    """Extract sea-defining polygons from the RAW base theme.

    Filter: class IN {ocean, strait, bay} OR subtype = ocean.
    Returns: unioned MultiPolygon (or empty GeometryCollection) for efficient
    per-cell intersection downstream. Sha256 of WKB bytes is recorded in
    manifest.sea_polygons_sha256 per spec §11.7.

    Per spec §9.1: this MUST run on raw themes BEFORE apply_missing_value_policy;
    sea polygons are masks (not features) so policy-drop correctly removes them
    from feature emission.
    """
    class_col = base_theme.column("class")
    subtype_col = base_theme.column("subtype")
    in_sea_class = pc.is_in(class_col, value_set=pa.array(list(SEA_CLASS_VALUES)))
    in_sea_subtype = pc.is_in(subtype_col, value_set=pa.array(list(SEA_SUBTYPE_VALUES)))
    mask = pc.or_(in_sea_class, in_sea_subtype)
    sea_rows = base_theme.filter(mask)
    if len(sea_rows) == 0:
        return MultiPolygon()
    geometries = [wkb.loads(g) for g in sea_rows.column("geometry").to_pylist()]
    return unary_union(geometries)


def apply_sea_mask(
    *,
    cell_box_admin_clipped: BaseGeometry,
    cell_features: list,  # list[CellSubFeature]; circular-import-avoided typing
    sea_polygons_svy21: BaseGeometry,
) -> tuple[float, float, bool]:
    """For a single cell, compute (sea_water_fraction, water_fraction, drop_flag).

    sea_water_fraction = area(cell ∩ admin ∩ sea_polygons) / area(cell ∩ admin)
    drop_flag = (sea_water_fraction >= 1.0 - EPS_RATIO) AND (zero non-sea features)

    Per spec §9.2 + §4.3 α structural-boundary EPSILON.
    """
    cell_admin_area = cell_box_admin_clipped.area
    if cell_admin_area <= 0:
        # Degenerate cell — should have been filtered upstream
        return 0.0, 0.0, True

    sea_overlap = cell_box_admin_clipped.intersection(sea_polygons_svy21)
    sea_water_fraction = sea_overlap.area / cell_admin_area
    # water_fraction is set by the caller based on inland water; here we
    # report sea only. The pipeline orchestrator combines with inland.
    water_fraction = sea_water_fraction  # placeholder; orchestrator overrides

    # "Zero non-sea features" — under the pipeline order (sea polygons removed
    # from policied themes already), every feature in cell_features is non-sea.
    non_sea_count = len(cell_features)
    drop_flag = sea_water_fraction >= (1.0 - EPS_RATIO) and non_sea_count == 0
    return sea_water_fraction, water_fraction, drop_flag


def compute_sea_overlap_fraction(
    *,
    feature_geom: BaseGeometry,
    feature_type: str,
    cell_local_sea_geometry: BaseGeometry | None,
) -> float:
    """Per spec §9.3: sea_overlap_fraction per feature, using cached
    cell-local sea geometry.

    Fast-path: if cell_local_sea_geometry is None (no sea in cell),
    return 0.0 immediately (no per-feature compute).

    For points: uses INTERSECTS predicate (NOT contains) — coastline POIs
    count as sea-adjacent (spec §9.3 precision item 1).

    For lines: length(geometry ∩ sea) / length(geometry).
    For polygons: area(geometry ∩ sea) / area(geometry).
    """
    if cell_local_sea_geometry is None:
        return 0.0

    if feature_type == "Point":
        return 1.0 if feature_geom.intersects(cell_local_sea_geometry) else 0.0
    if feature_type == "LineString":
        total = feature_geom.length
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea_geometry).length / total)
    if feature_type == "Polygon":
        total = feature_geom.area
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea_geometry).area / total)
    return 0.0
```

- [ ] **Step 4: Run tests, iterate to pass**

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/sea_mask.py tests/data/sub_c/test_sea_mask.py
git commit -m "$(cat <<'EOF'
feat(sub_c/sea_mask): derive_sea_polygons (pre-policy) + cell + feature sea logic

Sea polygons (base.class IN {ocean,strait,bay} OR subtype=ocean) are masks,
not features. derive_sea_polygons extracts them from RAW base theme BEFORE
the policy step (spec §6 + §9.1 cross-decision dependency); otherwise the
base.class not-in-vocab drop_row policy would eliminate them.

apply_sea_mask uses α structural-boundary EPSILON (EPS_RATIO at the 1.0
boundary; spec §4.3 + §9.2) with admin-clipped denominator.
compute_sea_overlap_fraction uses GEOS intersects (NOT contains) for points
so coastline POIs count as sea-adjacent (spec §9.3).

Determinism categories: A (cell-local sea geometry deterministic), C
(EPS_RATIO at structural 1.0 boundary).
EOF
)"
```

---

## Phase 5 — Missing-value policy + conditioning (spec §10, §11.9)

### Task 6: apply_missing_value_policy with closed handler-map

**Files:**
- Create: `src/cfm/data/sub_c/policy.py`
- Create: `tests/data/sub_c/test_policy.py`

**Spec sections:** §10.1, §10.2 (four-case schema), §3 (B2 prereq)

**Determinism categories:** (none new; reads policy YAML deterministically)

**Named tests:**
- `test_apply_missing_value_policy_returns_new_themes_dict_signature_enforced_non_mutation`
- `test_apply_missing_value_policy_drops_transportation_null_class_rows`
- `test_apply_missing_value_policy_assigns_b_unk_to_null_buildings_class`
- `test_apply_missing_value_policy_assigns_poi_unk_to_null_places_primary`
- `test_apply_missing_value_policy_drops_sea_defining_base_rows_from_features`
- `test_not_in_vocab_buildings_class_stored_as_class_raw`
- `test_not_in_vocab_transportation_class_dropped_symmetric_extension`
- `test_not_in_vocab_base_class_dropped_per_strict_decision`
- `test_apply_missing_value_policy_raises_policy_error_on_unknown_policy_type`

**Dependencies:** Task 1, B2 prereq (Task 0)

**Complexity:** medium

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Confirm ImportError**
- [ ] **Step 3: Implement policy.py**

Create `src/cfm/data/sub_c/policy.py`:

```python
"""apply_missing_value_policy: applies the four-case (missing_value, not_in_vocab)
rule from configs/data/missing_value_policy.yaml + configs/tokenizer/vocab_phase1.yaml
to raw Overture themes.

Per spec §10.1: pure function; signature enforces non-mutation (returns NEW
themes dict). Closed handler-map registry; unknown policy types raise PolicyError.

Per spec §10.2 four-case table:
  emit_unknown_token + scalar (buildings.class, places.categories.primary):
    NULL → <prefix>__UNK__; not-in-vocab → store raw (tokenizer handles at encode)
  drop_row + scalar (transportation.class):
    NULL OR not-in-vocab → drop row
  n_a + scalar (base.class):
    NULL doesn't occur; not-in-vocab → drop row (Strict floor explicit decision)
  n_a + list (places.categories.alternate):
    Store full list raw; tokenizer filters not-in-vocab elements at encode
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from cfm.data.sub_c.errors import PolicyError


# Vocab → kept-token set (loaded once per call)
def _load_vocab_kept_sets(vocab_yaml_path: Path) -> dict[str, set[str]]:
    """Returns map: field_name → set of kept token names (with prefix stripped).
    e.g., "buildings.class" → {"residential", "house", ...} (no B_ prefix).
    """
    with open(vocab_yaml_path) as f:
        vocab = yaml.safe_load(f)
    sections = vocab["feature_class"]
    result: dict[str, set[str]] = {}
    field_map = {
        "buildings.class": ("building", "B_"),
        "transportation.class": ("road", "R_"),
        "places.categories.primary": ("poi", "POI_"),
        "base.class": ("base", "BASE_"),
    }
    for field, (section_name, prefix) in field_map.items():
        tokens = sections[section_name]["tokens"]
        # Strip prefix; exclude __UNK__ placeholders
        kept = set()
        for tok in tokens:
            if "__UNK__" in tok:
                continue
            assert tok.startswith(prefix), f"token {tok} missing expected prefix {prefix}"
            kept.add(tok[len(prefix):])
        result[field] = kept
    return result


def apply_missing_value_policy(
    themes: dict[str, pa.Table],
    policy_yaml_path: Path,
    *,
    vocab_yaml_path: Path | None = None,
) -> dict[str, pa.Table]:
    """Returns a NEW themes dict; signature enforces non-mutation.
    Sub-A's Region object is untouched.

    The base.class not-in-vocab drop_row step removes ocean/strait/bay rows
    from the returned policied_themes — correct, because sea polygons are
    masks (not features). Sea-masking sources its sea-polygon set from the
    pre-policy derive_sea_polygons view (spec §6 + §9.1).

    Per spec §10.1 closed-set handler-map; unknown policy types raise PolicyError.
    """
    with open(policy_yaml_path) as f:
        policy = yaml.safe_load(f)

    vocab_kept = _load_vocab_kept_sets(vocab_yaml_path) if vocab_yaml_path else {}

    new_themes: dict[str, pa.Table] = {}
    for theme_name, table in themes.items():
        new_themes[theme_name] = table  # default unchanged

    # Process each field per its policy
    for field_path, entry in policy["fields"].items():
        theme_name, _, col = field_path.partition(".")
        if theme_name == "places":
            # places.categories.primary / places.categories.alternate
            new_themes["places"] = _apply_places_policy(
                new_themes["places"], field_path, entry, vocab_kept,
            )
        elif theme_name == "buildings":
            new_themes["buildings"] = _apply_scalar_policy(
                new_themes["buildings"], col, entry, vocab_kept.get(field_path),
                unknown_token="B__UNK__",
            )
        elif theme_name == "transportation":
            new_themes["transportation"] = _apply_scalar_policy(
                new_themes["transportation"], col, entry, vocab_kept.get(field_path),
                unknown_token=None,  # drop_row policy
            )
        elif theme_name == "base":
            new_themes["base"] = _apply_scalar_policy(
                new_themes["base"], col, entry, vocab_kept.get(field_path),
                unknown_token=None,  # n_a + not-in-vocab drop_row
            )
        else:
            raise PolicyError(f"unknown theme {theme_name!r} in policy YAML")

    return new_themes


def _apply_scalar_policy(
    table: pa.Table,
    column: str,
    entry: dict,
    kept_values: set[str] | None,
    *,
    unknown_token: str | None,
) -> pa.Table:
    """Apply (missing_value, not_in_vocab) policies for a scalar column."""
    mv_type = entry["policies"]["missing_value"]["type"]
    niv_type = entry["policies"].get("not_in_vocab", {}).get("type")

    col = table.column(column)

    # Step 1: missing_value handling
    if mv_type == "emit_unknown_token":
        if unknown_token is None:
            raise PolicyError(f"emit_unknown_token policy requires unknown_token; column={column}")
        col = pc.coalesce(col, pa.scalar(unknown_token))
        table = table.set_column(table.schema.get_field_index(column), column, col)
    elif mv_type == "drop_row":
        not_null_mask = pc.is_valid(col)
        table = table.filter(not_null_mask)
        col = table.column(column)
    elif mv_type == "n_a":
        # 100% non-null; nothing to do
        pass
    else:
        raise PolicyError(f"unknown missing_value type {mv_type!r}")

    # Step 2: not_in_vocab handling
    if niv_type is None or niv_type == "n_a":
        pass
    elif niv_type == "emit_unknown_token":
        # Sub-C stores raw; tokenizer handles fall-through at encode time
        # (spec §10.2 — Option A). No mutation at sub-C.
        pass
    elif niv_type == "drop_row":
        if kept_values is None:
            raise PolicyError(f"drop_row not_in_vocab requires kept_values; column={column}")
        # Keep rows where value IS in kept_values OR is the unknown_token placeholder
        valid_set = set(kept_values)
        if unknown_token:
            valid_set.add(unknown_token)
        is_kept = pc.is_in(table.column(column), value_set=pa.array(list(valid_set)))
        table = table.filter(is_kept)
    elif niv_type == "drop_element":
        raise PolicyError(
            f"drop_element policy is for list fields, not scalar columns; got column={column}"
        )
    else:
        raise PolicyError(f"unknown not_in_vocab type {niv_type!r}")

    return table


def _apply_places_policy(
    table: pa.Table,
    field_path: str,
    entry: dict,
    vocab_kept: dict[str, set[str]],
) -> pa.Table:
    """places.categories is struct{primary, alternate}; apply policies on
    each sub-field. primary is scalar (4-case scalar); alternate is list
    (stored full at sub-C; tokenizer filters at encode per spec §10.2)."""
    # For sub-C: primary policy maps NULL → POI__UNK__; alternate is preserved unchanged.
    # Tokenizer at encode time handles not-in-vocab for both. Sub-C stores raw.
    if field_path == "places.categories.primary":
        mv_type = entry["policies"]["missing_value"]["type"]
        if mv_type == "emit_unknown_token":
            cats = table.column("categories")
            primary = pc.struct_field(cats, "primary")
            primary_filled = pc.coalesce(primary, pa.scalar("POI__UNK__"))
            alternate = pc.struct_field(cats, "alternate")
            new_categories = pa.StructArray.from_arrays(
                [primary_filled.combine_chunks(), alternate.combine_chunks()],
                names=["primary", "alternate"],
            )
            return table.set_column(
                table.schema.get_field_index("categories"), "categories", new_categories
            )
    # places.categories.alternate: no mutation at sub-C
    return table
```

- [ ] **Step 4: Run tests, iterate**
- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/policy.py tests/data/sub_c/test_policy.py
git commit -m "$(cat <<'EOF'
feat(sub_c/policy): four-case missing_value + not_in_vocab application

apply_missing_value_policy returns a NEW themes dict (signature enforces
non-mutation per spec §10.1) and applies the four-case schema (spec §10.2):
emit_unknown_token + scalar → fill NULL with prefix__UNK__;
drop_row + scalar → drop NULL or not-in-vocab rows;
n_a + scalar → drop only not-in-vocab (Strict-floor explicit decision);
n_a + list → preserve full list raw at sub-C (tokenizer filters at encode).

Closed handler-map raises PolicyError on unknown policy types; sub-C reads
both YAML axes from B2's regenerated missing_value_policy.yaml (PREREQUISITE
verified in Task 0).
EOF
)"
```

### Task 7: compute_conditioning_per_tile

**Files:**
- Create: `src/cfm/data/sub_c/conditioning.py`
- Create: `tests/data/sub_c/test_conditioning.py`

**Spec sections:** §11.9 (conditioning vector schema + derivation rule)

**Determinism categories:** C (500m β strict threshold; NO EPSILON), E (int8 enum for COASTAL_RIVER)

**Named tests:**
- `test_compute_conditioning_inland_cell_classified_inland`
- `test_compute_conditioning_coastal_cell_classified_coastal`
- `test_compute_conditioning_riverside_requires_river_length_at_least_500m_strict`
- `test_compute_conditioning_coastal_plus_river_classified_coastal_riverside`
- `test_compute_conditioning_morphology_class_per_tile_from_day_one`
- `test_compute_conditioning_population_density_bucket_null_with_owner_tag`

**Dependencies:** Task 4 (CellSubFeature)

**Complexity:** small

- [ ] **Step 1–5: Standard TDD pattern.** Implementation reads tile-level cells + features, computes coastal_inland_river via the 500m β-strict rule (spec §11.9), populates morphology_class / era_class / admin_region per-tile from day one, leaves population_density_bucket null with `_owner: "sub-D"` tag. Commit message matches the spec section. See spec §11.9 for the locked enum mapping + rule.

---

## Phase 6 — I/O + encoding determinism (spec §14.3, §14.6)

### Task 8: io.py + determinism.py

**Files:**
- Create: `src/cfm/data/sub_c/io.py`
- Create: `src/cfm/data/sub_c/determinism.py`
- Create: `tests/data/sub_c/test_io_determinism.py`

**Spec sections:** §14.3 (parquet config, WKB byte-order, YAML canonicalization, int8 enums, NaN policy), §14.6 (EXCLUDED_FROM_SHA), §11.2/11.3/11.4 (parquet schemas)

**Determinism categories satisfied:** E (parquet config + WKB byte order + YAML canonical + int8 enums), F (sha computation with exclusions), H (excluded-from-determinism fields)

**Named tests:**
- `test_parquet_write_kwargs_match_spec_table`
- `test_canonicalize_yaml_sorted_keys_byte_deterministic`
- `test_wkb_byte_order_explicit_little_endian`
- `test_excluded_from_sha_includes_timestamps_excludes_rerun_reason`
- `test_excluded_from_sha_wildcard_suffix_match_on_final_dotted_segment`
- `test_compute_sha256_excludes_listed_fields`

**Dependencies:** Task 1

**Complexity:** medium

- [ ] **Step 1: Write failing tests**

Key test code (`tests/data/sub_c/test_io_determinism.py`):

```python
from cfm.data.sub_c.io import _PARQUET_WRITE_KWARGS, canonicalize_yaml, dump_wkb
from cfm.data.sub_c.determinism import EXCLUDED_FROM_SHA, compute_sha256


def test_parquet_write_kwargs_match_spec_table():
    expected = {
        "compression": "snappy",
        "row_group_size": 50_000,
        "data_page_size": 1_048_576,
        "write_batch_size": 10_000,
        "use_dictionary": True,
        "write_statistics": True,
        "use_compliant_nested_type": True,
        "version": "2.6",
    }
    for k, v in expected.items():
        assert _PARQUET_WRITE_KWARGS[k] == v, f"parquet kwarg {k}: expected {v}, got {_PARQUET_WRITE_KWARGS[k]!r}"


def test_canonicalize_yaml_sorted_keys_byte_deterministic():
    data = {"z": 1, "a": {"y": 2, "b": 3}}
    out1 = canonicalize_yaml(data)
    out2 = canonicalize_yaml(data)
    assert out1 == out2
    assert out1.index("a:") < out1.index("z:")  # sorted


def test_wkb_byte_order_explicit_little_endian():
    from shapely.geometry import Point
    blob = dump_wkb(Point(1, 2))
    # NDR = little-endian; first byte is 0x01
    assert blob[0] == 1


def test_excluded_from_sha_includes_timestamps_excludes_rerun_reason():
    assert "extraction.extracted_utc" in EXCLUDED_FROM_SHA["provenance.yaml"]
    # rerun_reason NOT excluded — audit trail purpose
    assert "extraction.rerun_reason" not in EXCLUDED_FROM_SHA["provenance.yaml"]


def test_excluded_from_sha_wildcard_suffix_match_on_final_dotted_segment():
    from cfm.data.sub_c.determinism import _field_matches_excluded
    # "*_sha256" matches any field whose final segment ends with _sha256
    assert _field_matches_excluded("vocab_sha256", ["*_sha256"])
    assert _field_matches_excluded("tiles[3].provenance_sha256", ["*_sha256"])
    assert _field_matches_excluded("outputs.cells_parquet_sha256", ["*_sha256"])
    assert not _field_matches_excluded("sha256_input", ["*_sha256"])  # suffix only


def test_compute_sha256_excludes_listed_fields():
    content_a = {"keep": 1, "extracted_utc": "2026-01-01T00:00:00Z", "rerun_reason": "test"}
    content_b = {"keep": 1, "extracted_utc": "2026-01-02T00:00:00Z", "rerun_reason": "test"}
    # Same after stripping extracted_utc → same sha
    sha_a = compute_sha256(content_a, file_kind="provenance.yaml")
    sha_b = compute_sha256(content_b, file_kind="provenance.yaml")
    assert sha_a == sha_b
    # Different rerun_reason → different sha (NOT excluded per F2 fix)
    content_c = {"keep": 1, "extracted_utc": "2026-01-01T00:00:00Z", "rerun_reason": "other"}
    sha_c = compute_sha256(content_c, file_kind="provenance.yaml")
    assert sha_a != sha_c
```

- [ ] **Step 2: Confirm ImportError**

- [ ] **Step 3: Implement `io.py` and `determinism.py`**

Create `src/cfm/data/sub_c/io.py`:

```python
"""Sub-C I/O helpers: parquet writer config, WKB byte-order, YAML canonicalize.

Per spec §14.3. canonicalize_yaml is re-exported from B2 for shared use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from shapely import wkb
from shapely.geometry.base import BaseGeometry


_PARQUET_WRITE_KWARGS = {
    "compression": "snappy",
    "row_group_size": 50_000,
    "data_page_size": 1_048_576,
    "write_batch_size": 10_000,
    "use_dictionary": True,
    "write_statistics": True,
    "use_compliant_nested_type": True,
    "version": "2.6",
}


def write_parquet(table, path: Path) -> None:
    """Write table with pinned config (spec §14.3)."""
    pq.write_table(table, path, **_PARQUET_WRITE_KWARGS)


def dump_wkb(geom: BaseGeometry) -> bytes:
    """Serialize geometry as WKB with explicit little-endian byte order.

    Per spec §14.3: NDR (byteorder=1) is pinned to kill shapely-version drift.
    """
    return wkb.dumps(geom, hex=False, byteorder=1)


def canonicalize_yaml(data: dict[str, Any]) -> str:
    """Deterministic YAML emit (spec §14.3 + inherited from B1/B2).

    Settings: sorted keys, block style, allow_unicode, SafeDumper, no comments.
    """
    return yaml.dump(
        data,
        Dumper=yaml.SafeDumper,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write canonicalized YAML to disk."""
    path.write_text(canonicalize_yaml(data))
```

Create `src/cfm/data/sub_c/determinism.py`:

```python
"""EXCLUDED_FROM_SHA + sha computation helpers.

Per spec §14.6 / E1: extends B2's sha-exclusion pattern (sha field excluded
from canonicalization) with timestamp + rerun-bookkeeping exclusions.
rerun_reason is NOT excluded (per F2: audit-trail purpose).
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from cfm.data.sub_c.io import canonicalize_yaml


# Wildcard semantics (NOT glob):
# - file key "*" matches any YAML file.
# - field-path "*_sha256" matches any field whose FINAL dotted-path segment
#   ends with the suffix "_sha256" (string endswith on last segment).
EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
    "provenance.yaml": [
        "extraction.extracted_utc",
        # rerun_reason NOT excluded (per F2: audit-trail purpose)
    ],
}

EXCLUDED_FROM_TEST_COMPARE = EXCLUDED_FROM_SHA  # one source of truth


def _field_matches_excluded(dotted_path: str, patterns: list[str]) -> bool:
    """Apply the wildcard semantics. dotted_path examples:
    "vocab_sha256", "tiles[3].provenance_sha256", "outputs.cells_parquet_sha256".
    """
    final_segment = dotted_path.rsplit(".", 1)[-1]
    # Strip array indices like "[3]"
    if "]" in final_segment:
        final_segment = final_segment.rsplit("]", 1)[-1] or final_segment.split("[", 1)[0]
    for p in patterns:
        if p.startswith("*_"):
            suffix = p[1:]  # "_sha256"
            if final_segment.endswith(suffix):
                return True
        else:
            if dotted_path == p:
                return True
    return False


def _strip_excluded(content: Any, exclude_paths: list[str], prefix: str = "") -> Any:
    """Return a deep-copy of content with EXCLUDED fields removed."""
    if isinstance(content, dict):
        out = {}
        for k, v in content.items():
            path = f"{prefix}.{k}" if prefix else k
            if _field_matches_excluded(path, exclude_paths):
                continue
            out[k] = _strip_excluded(v, exclude_paths, prefix=path)
        return out
    if isinstance(content, list):
        return [_strip_excluded(item, exclude_paths, prefix=prefix) for item in content]
    return content


def compute_sha256(content: dict, *, file_kind: str) -> str:
    """sha256 over canonicalized YAML content with EXCLUDED_FROM_SHA fields removed.

    file_kind: one of "manifest.yaml", "provenance.yaml", "meta.yaml" — selects the
    per-file exclusion list (in addition to "*" universal exclusions).
    """
    exclude = list(EXCLUDED_FROM_SHA.get("*", []))
    exclude.extend(EXCLUDED_FROM_SHA.get(file_kind, []))
    stripped = _strip_excluded(copy.deepcopy(content), exclude)
    return hashlib.sha256(canonicalize_yaml(stripped).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests, iterate**
- [ ] **Step 5: Commit**

---

## Phase 7 — Storage layout (spec §11)

### Task 9: Per-tile write helpers (cells, features, crossings, meta, provenance)

**Files:**
- Modify: `src/cfm/data/sub_c/io.py` (add per-tile dataclass writes)
- Create: `tests/data/sub_c/test_io_tile_writes.py` (or extend existing)

**Spec sections:** §11.1 (directory layout), §11.2 (features.parquet), §11.3 (cells.parquet), §11.4 (crossings.parquet), §11.5 (meta.yaml), §11.6 (provenance.yaml)

**Determinism categories:** D (sort keys per file), E (parquet config + WKB), F (provenance.outputs digests)

**Named tests:**
- `test_write_features_parquet_schema_and_sort_key`
- `test_write_cells_parquet_schema_and_sort_key`
- `test_write_crossings_parquet_schema_and_sort_key`
- `test_write_meta_yaml_includes_aggregates_and_conditioning`
- `test_write_provenance_yaml_inputs_and_outputs_digests`
- `test_features_parquet_bbox_columns_match_wkb_per_row`
- `test_features_parquet_geometry_type_int8_matches_wkb_header_per_row`
- `test_per_tile_directory_naming_includes_crs_and_named_ij`

**Dependencies:** Tasks 4, 5, 7, 8

**Complexity:** medium–large (multiple parquet schemas + yaml shapes)

- [ ] **Step 1–5:** Implement `write_cells_parquet`, `write_features_parquet`, `write_crossings_parquet`, `write_meta_yaml`, `write_provenance_yaml` in `io.py`. Each takes the structured dataclass instances from earlier tasks and writes per the spec schemas. The tests verify schemas, sort keys, denormalization-consistency (bbox + geometry_type from WKB), and tile-directory naming. Commit when all pass.

### Task 10: manifest.py + _SUCCESS write protocol

**Files:**
- Create: `src/cfm/data/sub_c/manifest.py`
- Modify: `tests/data/sub_c/test_io_tile_writes.py` (extend; or create test_manifest.py)

**Spec sections:** §11.7 (manifest.yaml), §11.8 (write order + _SUCCESS semantics), §14.6 (digest chain)

**Determinism categories:** F (digest chain), G (sub_c_schema_version), J (write order)

**Named tests:**
- `test_write_manifest_yaml_schema_and_sorted_tiles`
- `test_manifest_tiles_carries_provenance_sha256_per_tile`
- `test_manifest_initial_extraction_block_frozen_after_first_run`
- `test_write_success_marker_only_after_cross_tile_validator_passes` (deferred; tested in Task 14)
- `test_single_tile_re_extraction_updates_only_target_provenance_sha256_in_manifest`

**Dependencies:** Tasks 8, 9

**Complexity:** medium

- [ ] **Step 1–5:** TDD implementation. The manifest writer takes a list of per-tile (tile_i, tile_j, provenance_sha256) entries, the region-level constants from `config:` + `conditioning_defaults:`, and assembles the canonicalized YAML. Single-tile re-extraction updates only the target tile's `provenance_sha256` in `tiles[]`; the `initial_extraction` block is frozen.

---

## Phase 8 — Inline validator (spec §12.1)

### Task 11: validator_inline.py with 10 named invariants + structured TileValidationError

**Files:**
- Create: `src/cfm/data/sub_c/validator_inline.py`
- Create: `tests/data/sub_c/test_validator_inline.py`

**Spec sections:** §12.1 (10 invariants), §12.4 (structured diagnostic format), §14.3 (NaN combined-pass)

**Determinism categories satisfied:** K (structured diagnostic format deterministic), C (EPSILON application in invariants 2, 5, 6, 8, 9), F (geometry_type / bbox consistency tests are part of integrity chain)

**Named tests** (one per spec-§12.1 invariant; numbering matches spec):
- `test_inline_validator_passes_on_clean_torture_tile_output` (Layer 2 — deferred to Task 16)
- `test_inline_invariant_1_schema_correctness_diagnostic`
- `test_inline_invariant_2_bbox_matches_wkb_diagnostic`
- `test_inline_invariant_3_geometry_type_matches_wkb_diagnostic`
- `test_inline_invariant_4_crossings_features_source_feature_id_diagnostic`
- `test_inline_invariant_5_water_fraction_bounds_combined_with_nan_check_single_pass`
- `test_inline_invariant_6_kept_cell_rule_diagnostic`
- `test_inline_invariant_7_kept_features_count_matches_features_parquet_row_count`
- `test_inline_invariant_8_mean_water_fraction_area_weighted_formula_match`
- `test_inline_invariant_9_cell_area_admin_clipped_alpha_structural_boundary`
- `test_inline_invariant_10_nan_free_on_non_water_fraction_numeric_columns` (standalone NaN check for `edge_position_m`, `edge_extent_length_m`, `bbox_*`, `cell_area_admin_clipped_m2`, `sea_overlap_fraction`; the water-fraction half of #10 is folded into #5's combined-pass per spec §14.3 efficiency lock)

**Dependencies:** Tasks 5, 8, 9

**Complexity:** medium — **implementation is spec-driven; spec §12.1 is implementation-complete.** The 10 invariants are exhaustively defined there with their exact assertions, EPSILON usage, and structured-payload shape; this task wires them into Python without designing new logic. The wiring pattern is shown for one invariant below; the other 9 are mechanical translations.

- [ ] **Step 1: Write the failing tests for all 10 invariants + the structured-payload contract**

Create `tests/data/sub_c/test_validator_inline.py`. The test pattern is the same for all 10: start from a known-good tile_dir fixture, corrupt one artifact in a temp copy, call `validate_tile_inline(tmp_tile_dir)`, assert the `TileValidationError` has `invariant=<the_name>` and the right `failed_row` / `detail` shape.

Skeleton for one invariant (the other 9 follow the same pattern, swapping the corruption + assertions per spec §12.1):

```python
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cfm.data.sub_c.errors import TileValidationError
from cfm.data.sub_c.validator_inline import validate_tile_inline


@pytest.fixture
def clean_tile_dir(tmp_path):
    """Copy a known-good single-tile fixture into tmp_path; tests corrupt
    inside the copy without touching the canonical fixture. The clean tile
    fixture is built by tests/fixtures/sub_c/build_torture_tile.py (Task 15)
    and persisted under tests/fixtures/sub_c/torture_tile/.
    """
    src = Path(__file__).parent.parent.parent / "fixtures" / "sub_c" / "torture_tile" / "tile=EPSG3414_i0_j0"
    dst = tmp_path / "tile=EPSG3414_i0_j0"
    shutil.copytree(src, dst)
    return dst


def test_inline_invariant_2_bbox_matches_wkb_diagnostic(clean_tile_dir):
    # Corrupt: bbox_min_x off by 100m on row 0
    features_path = clean_tile_dir / "features.parquet"
    table = pq.read_table(features_path)
    bbox = table.column("bbox_min_x").to_pylist()
    bbox[0] += 100.0
    table = table.set_column(
        table.schema.get_field_index("bbox_min_x"), "bbox_min_x", [bbox],
    )
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc:
        validate_tile_inline(clean_tile_dir)
    err = exc.value
    assert err.invariant == "bbox_matches_wkb"
    assert err.tile == "tile=EPSG3414_i0_j0"
    assert err.failed_row.get("row_index") == 0
    assert "stored_bbox" in err.detail and "wkb_bbox" in err.detail


# Repeat the pattern for invariants 1, 3, 4, 5, 6, 7, 8, 9, 10 per spec §12.1.
# Examples of per-invariant corruption strategy:
#   #5: corrupt water_fraction=1.5 AND sea_water_fraction=NaN on same cell row
#       — single combined-pass test verifies both bounds + NaN caught.
#   #6: corrupt cells.parquet to include a cell where sea_water_fraction=1.0
#       AND kept_features_count=0 (the drop rule was bypassed).
#   #10: corrupt edge_position_m=float('nan') on a crossings.parquet row;
#        assert invariant fires with the right column name in detail.
```

- [ ] **Step 2: Run tests, confirm ImportError**

Run: `uv run pytest tests/data/sub_c/test_validator_inline.py -v`
Expected: `ImportError` on `cfm.data.sub_c.validator_inline`.

- [ ] **Step 3: Implement `validator_inline.py`**

Create `src/cfm/data/sub_c/validator_inline.py`. Public entry point: `validate_tile_inline(tile_dir: Path) -> None`. Loads `cells.parquet`, `features.parquet`, `crossings.parquet`, `meta.yaml`; runs the 10 invariants in spec-§12.1 order; raises `TileValidationError` with structured payload on first failure.

Skeleton showing the dispatcher + one invariant helper (the other 9 helpers follow the same shape — read spec §12.1 row-by-row for each):

```python
"""Per-tile inline validator. Runs after parquet + meta.yaml writes, BEFORE
provenance.yaml. Failure halts (no provenance.yaml = tile not complete).

Per spec §12.1 + §12.4: 10 named invariants; structured TileValidationError
payload (tile, invariant, failed_row, detail). Diagnostic-payload determinism
is verified end-to-end by test_validator_diagnostic_payloads_byte_deterministic
in Task 16.

Failures are bugs-in-sub-C, NOT data-issues — per auto-memory
feedback_test_weakening_to_pass.md. If an invariant fires on real Singapore
data, STOP and escalate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely import wkb

from cfm.data.sub_c.enums import GEOMETRY_TYPE, decode_enum
from cfm.data.sub_c.epsilon import EPS_AREA_M2, EPS_COORD_M, EPS_RATIO
from cfm.data.sub_c.errors import TileValidationError


def validate_tile_inline(tile_dir: Path) -> None:
    """Run all 10 spec-§12.1 invariants on tile_dir. First failure raises
    TileValidationError with structured payload."""
    tile_name = tile_dir.name
    cells = pq.read_table(tile_dir / "cells.parquet")
    features = pq.read_table(tile_dir / "features.parquet")
    crossings = pq.read_table(tile_dir / "crossings.parquet")
    meta = yaml.safe_load((tile_dir / "meta.yaml").read_text())

    _invariant_1_schema_correctness(tile_name, cells, features, crossings)
    _invariant_2_bbox_matches_wkb(tile_name, features)
    _invariant_3_geometry_type_matches_wkb(tile_name, features)
    _invariant_4_crossings_features_source_feature_id(tile_name, features, crossings)
    _invariant_5_water_fraction_bounds_combined_nan(tile_name, cells)
    _invariant_6_kept_cell_rule(tile_name, cells)
    _invariant_7_kept_features_count(tile_name, cells, features)
    _invariant_8_mean_water_fraction_area_weighted(tile_name, cells, meta)
    _invariant_9_cell_area_admin_clipped(tile_name, cells)
    _invariant_10_nan_free_non_water_fraction_columns(tile_name, cells, features, crossings)


def _invariant_2_bbox_matches_wkb(tile_name: str, features) -> None:
    """For every feature row, bbox_* columns match WKB-derived bbox within EPS_COORD_M."""
    for idx in range(features.num_rows):
        geom_wkb = features.column("geometry")[idx].as_py()
        geom = wkb.loads(geom_wkb)
        actual = geom.bounds
        stored = (
            features.column("bbox_min_x")[idx].as_py(),
            features.column("bbox_min_y")[idx].as_py(),
            features.column("bbox_max_x")[idx].as_py(),
            features.column("bbox_max_y")[idx].as_py(),
        )
        if any(abs(a - b) > EPS_COORD_M for a, b in zip(actual, stored)):
            raise TileValidationError(
                tile=tile_name,
                invariant="bbox_matches_wkb",
                failed_row={
                    "source_feature_id": features.column("source_feature_id")[idx].as_py(),
                    "row_index": idx,
                },
                detail={"stored_bbox": stored, "wkb_bbox": actual},
            )


# Helpers _invariant_1, 3, 4, 5, 6, 7, 8, 9, 10 follow the same shape.
# Translate spec §12.1 row-by-row. Diagnostic-payload determinism note:
# use sorted dict keys and explicit tuple ordering (no Python set iteration)
# so the payload bytes reproduce across runs given identical input.
```

The 9 remaining `_invariant_N_*` helpers are mechanical translations of the spec §12.1 table. The engineer reads each row of the spec table and writes the corresponding helper following the bbox pattern. **Diagnostic-payload determinism note for the implementer:** use sorted dict keys and explicit tuple ordering (no Python `set` iteration order leaking into the payload), so the bytes reproduce across runs given identical input — caught by `test_validator_diagnostic_payloads_byte_deterministic` in Task 16.

- [ ] **Step 4: Run tests, iterate until all 11 pass**

Run: `uv run pytest tests/data/sub_c/test_validator_inline.py -v`
Expected: 11 named tests pass (1 Layer-2-deferred + 10 invariant tests).

If any test fails because the clean-fixture data violates the invariant (rather than the corruption catching it), STOP and escalate. The most likely candidate is #6 (kept-cell rule) — that would indicate a 2.5a drop-rule bug per `feedback_test_weakening_to_pass.md`.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/sub_c/validator_inline.py tests/data/sub_c/test_validator_inline.py
git commit -m "$(cat <<'EOF'
feat(sub_c/validator_inline): 10 inline invariants with structured TileValidationError

Implements validate_tile_inline per spec §12.1 (10 invariants exhaustively
defined there): schema correctness, bbox↔WKB consistency, geometry_type↔WKB
consistency, crossings↔features source_feature_id linkage, water-fraction
bounds combined with NaN check in a single column traversal (spec §14.3
efficiency lock), kept-cell rule (catches "drop rule wasn't applied" bugs
per feedback_test_weakening_to_pass.md), kept_features_count match,
mean_water_fraction area-weighted formula match, cell_area_admin_clipped
α structural-boundary check, and standalone NaN-free assertion on
non-water-fraction numeric columns (edge_position_m, edge_extent_length_m,
bbox_*, cell_area_admin_clipped_m2, sea_overlap_fraction).

Each invariant emits TileValidationError with the structured (tile, invariant,
failed_row, detail) payload per spec §12.4; payload determinism is verified
end-to-end by Task 16's test_validator_diagnostic_payloads_byte_deterministic.
EOF
)"
```

---

## Phase 9 — Pipeline orchestrator (spec §6)

### Task 12: Pipeline orchestrator (sequential first; parallelism in next task)

**Files:**
- Create: `src/cfm/data/sub_c/pipeline.py`
- Create: `tests/data/sub_c/test_pipeline_sequential.py`

**Spec sections:** §6 (pipeline overview), §10.1 (apply_missing_value_policy at raw level), §9.1 (derive_sea_polygons before policy), §7.3 (reproject-first-then-clip)

**Determinism categories:** J (pipeline order)

**Named tests:**
- `test_pipeline_runs_derive_sea_polygons_before_apply_missing_value_policy`
- `test_pipeline_reproject_runs_before_clip`
- `test_pipeline_clip_runs_before_partition_into_tiles`
- `test_pipeline_sliver_drop_runs_before_sea_mask`
- `test_pipeline_extract_tile_produces_complete_directory_artifacts`

**Dependencies:** Tasks 4, 5, 6, 7, 8, 9, 10, 11

**Complexity:** medium

- [ ] **Step 1–5:** Implement `extract_region(region_name, output_dir, ...) -> RegionManifest` in `pipeline.py` that composes all sub-C stages in the locked order. Sequential first; Task 13 adds the process-pool variant. Tests use a minimal Region object (synthetic; not the cached Singapore) to verify the per-tile output directory contains all expected files.

### Task 13: Process-pool extraction with shared once-per-region inputs

**Files:**
- Modify: `src/cfm/data/sub_c/pipeline.py` (add `extract_region_parallel`)
- Create: `tests/data/sub_c/test_pipeline_parallel.py`

**Spec sections:** §14.5 (parallelization safety)

**Determinism categories:** J (parallelization safety; pool-size independence)

**Named tests:**
- `test_extraction_pool_size_independence` — pool_size=1 vs pool_size=4 byte-identical (after timestamp strip)
- `test_extraction_pool_size_independence_more_workers_than_tiles` — pool_size=1 vs pool_size=N>tile_count
- `test_workers_receive_shared_densified_polygon_and_sea_polygons`

**Dependencies:** Task 12

**Complexity:** medium

- [ ] **Step 1–5:** Add `extract_region_parallel(region_name, output_dir, pool_size)` using `multiprocessing.Pool` with dynamic queue. Densified admin polygon and sea polygons computed once in main process and serialized to workers via WKB. Tests verify byte-output is invariant under pool_size; the `_more_workers_than_tiles` test catches empty-queue worker shutdown bugs.

---

## Phase 10 — CLI + cross-tile validator (spec §15.2, §12.2)

### Task 14: scripts/extract_tiles.py + scripts/validate_extraction.py

**Files:**
- Create: `scripts/extract_tiles.py`
- Create: `scripts/validate_extraction.py`
- Create: `src/cfm/data/sub_c/validator_cross_tile.py`
- Create: `tests/data/sub_c/test_cli.py`

**Spec sections:** §12.2 (cross-tile validator), §15.2 (CLI signatures), §11.8 (write order)

**Determinism categories:** F (digest chain enforcement), J (cross-tile validator as _SUCCESS gate)

**Named tests:** (Layer 2 negative-fixture tests come in Task 16; these tests verify the CLIs as scripts)
- `test_extract_tiles_script_writes_manifest_and_success`
- `test_validate_extraction_script_exits_zero_on_clean_extraction`
- `test_validate_extraction_script_exits_nonzero_on_orphan_tile_dir`

**Dependencies:** Tasks 12, 13

**Complexity:** medium

- [ ] **Step 1–5:** Implement both scripts. `extract_tiles.py` parses args (`--region`, `--release`, `--output-dir`, `--pool-size`, `--rerun`, `--rerun-reason`) and calls `extract_region(_parallel)`. `validate_extraction.py` calls the cross-tile validator and writes `_SUCCESS` if it passes (or removes it if it doesn't). Cross-tile validator logic in `validator_cross_tile.py` checks all 4 invariants from spec §12.2.

---

## Phase 11 — Test fixtures + Layer 2 tests (spec §13.2)

### Task 15: build_torture_tile.py + build_cross_tile_fixture.py

**Files:**
- Create: `tests/fixtures/sub_c/__init__.py`
- Create: `tests/fixtures/sub_c/build_torture_tile.py`
- Create: `tests/fixtures/sub_c/build_cross_tile_fixture.py`

**Spec sections:** §13.2 (fixture design; declarative element list tagged per topic decision)

**Determinism categories:** (fixtures are byte-deterministic by construction)

**Named tests:** (these are fixture builders, not tests themselves; tested transitively in Tasks 16+)

**Dependencies:** Tasks 1–11

**Complexity:** medium

- [ ] **Step 1–5:** Build the torture-test fixture as a declarative list of feature definitions, each tagged with a comment naming the topic decision it exercises (e.g., `# exercises 2b corner-crossing`). 4×4 cell synthetic tile (smaller than production 8×8 to keep fixture tractable); 14+ features covering every code path enumerated in spec §13.2. Generated Overture-style parquets are byte-deterministic.

Build cross-tile micro-fixture: 2 tiles, each 1 cell + 1 feature, used only for cross-tile-validator failure-mode tests.

### Task 16: Layer 2 tests (torture-tile pipeline + cross-tile validator failure modes + determinism)

**Files:**
- Create: `tests/data/sub_c/test_pipeline_torture_tile.py` (uses session-scoped fixture)
- Create: `tests/data/sub_c/test_cross_tile_validator.py`
- Create: `tests/data/sub_c/test_determinism.py`

**Spec sections:** §13.2 (all Layer 2 tests)

**Determinism categories:** F (sha stability), J (pool-size + write-order), K (diagnostic payload determinism)

**Named tests:** (per spec §13.2)
- `test_torture_tile_extraction_succeeds`
- `test_torture_tile_inline_validator_passes_on_clean_output`
- `test_torture_tile_reextract_byte_identical_modulo_excluded_fields`
- 8 per-invariant diagnostic payload tests (per Task 11's list; verified end-to-end here)
- `test_validator_diagnostic_payloads_byte_deterministic`
- `test_provenance_sha256_byte_deterministic_across_runs`
- `test_cross_tile_validator_detects_orphan_tile_dir`
- `test_cross_tile_validator_detects_missing_tile_dir`
- `test_cross_tile_validator_detects_provenance_sha256_mismatch`
- `test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun`
- `test_extraction_pool_size_independence` (re-verified at Layer 2 against torture-tile)
- `test_extraction_pool_size_independence_more_workers_than_tiles`
- `test_pyarrow_version_2_6_parquet_format_correct` (verify-at-impl)
- `test_pyproj_uses_formula_path_for_svy21` (verify-at-impl)

**Dependencies:** Task 15

**Complexity:** large (the most test-density of any single task)

- [ ] **Step 1–5:** Implement session-scoped pytest fixture for torture-tile extraction (extract ONCE per pytest session; tests assert on shared output unless they require corruption, in which case they copy-and-modify per spec §13.2 P5). All ~16 Layer 2 named tests pass. STOP and escalate on any unexpected failure per `feedback_test_weakening_to_pass.md`.

---

## Phase 12 — Layer 3 + finishing

### Task 17: Layer 3 cached-Singapore integration tests

**Files:**
- Create: `tests/data/sub_c/test_singapore_integration.py`

**Spec sections:** §13.3 (Layer 3)

**Named tests:**
- `test_singapore_two_tile_extraction_shape` (pick specific (tile_i, tile_j) for Marina Bay + central reservoir representatives; document choice in test docstring)
- `test_singapore_tile_reextract_byte_identical_modulo_excluded_fields`
- `test_singapore_two_tile_cross_tile_validator_pass`
- `test_singapore_manifest_sub_c_schema_version_consistency`

**Dependencies:** Tasks 14, 15, 16

**Complexity:** small–medium (uses cached Singapore, ~1s per extraction)

- [ ] **Step 1–5:** Use `load_region("singapore")` (cache-hit). Pick specific (tile_i, tile_j) values for tests based on inspecting which Singapore SVY21 tile contains Marina Bay (`reproject_lonlat_to_svy21(103.8587, 1.2839)` → `tile_id_from_svy21(...)` gives the exact (i, j)). Same for a central-reservoir tile. Document the choice in each test's docstring. STOP and escalate on any unexpected failure.

### Task 18: Pre-commit lint rule for pandas-in-write-path

**Files:**
- Create: `scripts/lint/no_pandas_in_write_path.py`
- Modify (or create): `.pre-commit-config.yaml`
- Create: `tests/lint/test_no_pandas_in_write_path.py`

**Spec sections:** §14.7 (pandas forbidden in write path)

**Determinism categories:** I (library pinning enforcement)

**Named tests:**
- `test_lint_rule_passes_on_clean_write_path`
- `test_lint_rule_fails_on_synthetic_offending_import`

**Dependencies:** Tasks 1–14 (lint targets src/cfm/data/sub_c/)

**Complexity:** small

- [ ] **Step 1–5:** Implement the grep-based lint rule that scans `src/cfm/data/sub_c/` (write-path modules) for `import pandas` or `from pandas`, fails the commit if found. The lint rule's test creates a synthetic offending file in a temp location and verifies the rule rejects it. The rule whitelists test fixtures, analysis scripts, exploratory notebooks (not under src/cfm/data/sub_c/).

### Task 19: docs/known_issues.md entries + final verification + merge to main

**Files:**
- Modify: `docs/known_issues.md` (add 2 entries)
- None else (verification only)

**Spec sections:** §3 (B2 prereq), §7.4 (Sweden densification revisit), §17 (tokenizer enhancement training-path dependency)

**Dependencies:** All prior tasks

**Complexity:** small

- [ ] **Step 1: Add `docs/known_issues.md` entry: Sweden densification revisit**

Prepend to `docs/known_issues.md` (between the existing header and `## #2`):

```markdown
## #3 — Polygon densification: Singapore measured no-op; Sweden requires re-measurement

- **Filed:** 2026-05-18 (Phase 1 sub-C ship)
- **Severity:** low (per-region prerequisite, not a bug)
- **Status:** open — **Sweden enrollment sub-project MUST measure Sweden's coastline edge-length distribution before reaching the same conclusion as Singapore**

### Context

Sub-C's `densify_polygon(polygon, max_edge_length_m)` function signature is locked at spec §7.4. For Singapore it is invoked with `max_edge_length_m=None` (no-op) per the empirical measurement (cached 2026-04-15.0 Singapore divisions parquet: max edge 775m, 99% < 500m, median 57m; at 1.3°N latitude, 4326-Cartesian distortion on the longest edge is ~23cm, well below the 250m cell quantization scale).

### Why this needs revisit for Sweden

Higher latitudes amplify the 4326-Cartesian distortion (cos(lat) effect: 0.9997 at 1.3°N → ~0.515 at 59°N). Sweden's archipelago coastlines may also contain much longer edges than Singapore's. The same no-densification conclusion CANNOT be assumed without re-measurement.

### What to do

When Sweden enrollment ships, before extracting Swedish tiles: measure Sweden's admin-polygon edge-length distribution; pick a `max_edge_length_m` value (suggestion: 100m, or whatever keeps post-densification edge distortion below the cell quantization scale at Sweden's latitude); update the Sweden region's `--densify-max-edge-m` CLI argument (Sweden enrollment will add this) accordingly.

### Tracking

- Sub-C spec §7.4: `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md`
- Function signature: `src/cfm/data/sub_c/coords.py::densify_polygon`

---
```

(also add a known_issues entry for the tokenizer enhancement training-path dependency).

- [ ] **Step 2: Add `docs/known_issues.md` entry: Tokenizer enhancement for emit_unknown_token fall-through**

Prepend (above the just-added #3):

```markdown
## #4 — Tokenizer hard-raises on not-in-vocab classes; sub-C output requires enhancement before training

- **Filed:** 2026-05-18 (Phase 1 sub-C ship)
- **Severity:** medium (training-critical-path; not a sub-D/E/F/G blocker)
- **Status:** open — separate sub-project required before training is possible

### Context

Sub-C ships per-cell rows with raw `class_raw` values (mix of Phase 1 vocab values + originally-rare Overture values + `B__UNK__` / `POI__UNK__` from NULL policy). Today's encoder at `src/cfm/tokenizer/encode.py:59-60` hard-raises `UnsupportedFeatureClass` on any class not in `vocab.token_to_id`.

### Why this is OK for now

- Sub-D consumes per-cell rows (NOT tokens).
- Sub-E consumes crossing records (NOT tokens).
- Sub-F consumes per-cell + crossing data (NOT tokens).
- Sub-G consumes geometry + macro plan (NOT tokens directly).

Only training itself requires tokenization. The dependency is on the training critical path, not on sub-D/E/F/G progress.

### Planned fix

A small (~10 line) tokenizer enhancement adds: for any class not in vocab where the corresponding `missing_value_policy.yaml` says `emit_unknown_token`, fall through to `f"{prefix}__UNK__"`. Otherwise raise (current behavior).

This is a separate sub-project with its own brainstorm-spec-test cycle. The motivating benefit is Phase 1.1 Sweden vocab expansion: Singapore's originally-rare "barn" rows correctly tokenize as `B_barn` AFTER Phase 1.1 promotion without re-extracting any Singapore tile.

### Tracking

- Sub-C spec §3, §10.2 four-case rule
- Encoder: `src/cfm/tokenizer/encode.py:59-60`

---
```

- [ ] **Step 3: Run full fast suite**

Run: `uv run pytest -q`
Expected: ~187 (pre-sub-C) + ~60 (sub-C) = ~247 passed, 1 xfailed (Phase 0 boundary-entry-marker), 6 deselected (slow sub-A tests).

- [ ] **Step 4: Run lint + format check**

Run: `uv run ruff check src/cfm/data/sub_c/ scripts/extract_tiles.py scripts/validate_extraction.py`
Expected: all clean.

Run: `uv run ruff format --check src/cfm/data/sub_c/ scripts/extract_tiles.py scripts/validate_extraction.py`
Expected: already-formatted.

- [ ] **Step 5: Run extraction against cached Singapore + cross-tile validator**

Run:
```bash
uv run python scripts/extract_tiles.py --region singapore --release 2026-04-15.0
uv run python scripts/validate_extraction.py --region singapore --release 2026-04-15.0
ls data/processed/sub_c/2026-04-15.0/singapore/_SUCCESS
```
Expected: extraction completes; validator exits 0; `_SUCCESS` present.

- [ ] **Step 6: Run extraction a second time and confirm byte-identical artifacts**

Run:
```bash
mkdir -p /tmp/sub-c-determinism-check
uv run python scripts/extract_tiles.py --region singapore --release 2026-04-15.0 --output-dir /tmp/sub-c-determinism-check
# Compare with the excluded-fields stripped
uv run python -c "
from cfm.data.sub_c.determinism import compute_sha256
import yaml
m1 = yaml.safe_load(open('data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml'))
m2 = yaml.safe_load(open('/tmp/sub-c-determinism-check/2026-04-15.0/singapore/manifest.yaml'))
assert compute_sha256(m1, file_kind='manifest.yaml') == compute_sha256(m2, file_kind='manifest.yaml'), 'manifest sha drift'
print('byte-identical modulo excluded fields')
"
```
Expected: `byte-identical modulo excluded fields`.

- [ ] **Step 7: Commit final docs + verification artifacts**

```bash
git add docs/known_issues.md
git commit -m "$(cat <<'EOF'
docs(sub_c): known_issues entries for Sweden densification + tokenizer enhancement

Sub-C ships against Singapore only; #3 records that Sweden enrollment must
re-measure coastline edge-length distribution before reaching the same
no-densification conclusion (higher latitudes amplify cos(lat) distortion;
archipelago coastlines have long edges). #4 records the tokenizer
enhancement for emit_unknown_token fall-through as a training-critical-path
dependency (NOT a sub-D/E/F/G blocker — those consume rows, not tokens).
EOF
)"
```

- [ ] **Step 8: Merge sub-C to main**

Run:
```bash
git checkout main
git merge --no-ff phase-1-sub-C-tile-extraction -m "merge: Phase 1 sub-project C (multi-cell tile extraction) complete"
git log --oneline -5
```
Expected: merge commit at HEAD; sub-C branch merged.

- [ ] **Step 9: Print done summary**

Run:
```bash
echo "=== Sub-C shipped ==="
echo "Spec: docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md"
echo "Plan: docs/superpowers/plans/2026-05-17-phase-1-sub-C-tile-extraction.md"
echo "Library: src/cfm/data/sub_c/"
echo "Scripts: scripts/extract_tiles.py, scripts/validate_extraction.py"
echo "Output: data/processed/sub_c/<release>/<region>/"
echo
echo "Test counts:"
uv run pytest -q --collect-only 2>/dev/null | tail -3
echo
echo "Phase 1 progress:"
echo "  sub-A (Overture loader)             DONE"
echo "  sub-B1 (frequency analysis)         DONE"
echo "  sub-B2 (vocab YAML derivation)      DONE"
echo "  sub-C (multi-cell tile extraction)  DONE"
echo "  sub-D (macro plan derivation)        NEXT"
echo "  sub-E (boundary contracts)"
echo "  sub-F (deterministic stitcher)"
echo "  sub-G (end-to-end pipeline + validator)"
```

---

## Self-review checklist (completed by plan author)

**Spec coverage** — every spec section maps to a task:
- §1 Goal → Tasks 1–19 collectively
- §2 Scope → Task 0 (PREREQUISITE gate); Task 19 (known_issues)
- §3 PREREQUISITE B2 follow-up → Task 0 verification gate
- §4 Design principles → embedded in per-task spec references; epsilon/enums/errors capture the formal primitives in Task 1
- §5 Cross-decision dependencies → enforced by per-task ordering + Task 5's `test_derive_sea_polygons_runs_against_raw_base_not_policied_themes`
- §6 Pipeline → Tasks 12 + 13 orchestrator; per-stage tasks
- §7 Coords → Tasks 2, 3
- §8 Cell extraction → Task 4
- §9 Sea masking → Task 5
- §10 Policy → Task 6
- §11 Storage layout → Tasks 9, 10
- §11.9 Conditioning → Task 7
- §12 Validators → Tasks 11 (inline), 14 (cross-tile)
- §13 Tests → Layer 1 distributed; Layer 2 in Tasks 15, 16; Layer 3 in Task 17
- §14 Determinism contract → Tasks 8 (io/determinism), 13 (parallelization), 16 (test surface)
- §15 Public API → exports added per task; CLI in Task 14
- §16 Module layout → File map in plan header
- §17 Errors → Task 1 (errors.py); Task 6 (PolicyError); Task 11 (TileValidationError end-to-end)
- §18 Done criteria → Task 19 verification steps
- §19 Risks → addressed by inline validator (Task 11), pre-commit lint (Task 18), known_issues (Task 19)
- §20 Out-of-scope → Task 19 known_issues + the PREREQUISITE framing
- §21 Implementation order → matches Tasks 0 → 19
- §22 References → in plan header + per-task spec citations

**Placeholder scan** — no `TBD` / `TODO` / vague references. Task 7 ("Standard TDD pattern") references the spec §11.9 for exact rule; that's acceptable because §11.9 fully specifies the rule (enum mapping, 500m threshold, derivation logic). Task 9 ("multiple parquet schemas + yaml shapes") is a complexity flag, not a placeholder — schemas are exhaustively defined in spec §11.2–§11.6.

**Type consistency:**
- `CellSubFeature` shape used identically across Tasks 4, 5, 6, 7, 9, 11.
- `CrossingRecord` shape used identically across Tasks 4, 9, 11.
- `_PARQUET_WRITE_KWARGS` referenced as single source from Tasks 8, 9, 12.
- `EXCLUDED_FROM_SHA` referenced from Tasks 8, 10, 16.
- `EPS_RATIO` / `EPS_AREA_M2` / etc. referenced from Tasks 1, 5, 7, 11.
- `derive_sea_polygons` (Task 5) consumed by Task 12 (sequential pipeline) and Task 13 (parallel pipeline) with identical signature.

**Plan ↔ spec mismatch flagged for the executor:**
- The spec self-review fix (derive_sea_polygons as pre-policy view) is realized in Task 5 (implementation) + Task 6 (policy ordering after sea derivation) + Task 12 (pipeline orchestrator composes in the right order). The named test `test_derive_sea_polygons_runs_against_raw_base_not_policied_themes` in Task 5 verifies this; the named test `test_pipeline_runs_derive_sea_polygons_before_apply_missing_value_policy` in Task 12 verifies the orchestrator honors the order.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-17-phase-1-sub-C-tile-extraction.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, controller reviews between tasks, fast iteration. Each phase (0–12) ships independently with its own commit(s). **Every implementer dispatch MUST include the Branch discipline section verbatim from the plan header** to prevent the sub-B2 Task 3 incident.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints between phases.

**Which approach?**
