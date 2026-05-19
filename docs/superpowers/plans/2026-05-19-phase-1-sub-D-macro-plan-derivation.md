# Phase 1 Sub-D Macro Plan Derivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 sub-D: a two-stage macro-plan derivation pipeline that first produces reviewable empirical analysis and then writes a digest-anchored sidecar dataset with macro-core targets and effective conditioning.

**Architecture:** Sub-D is a sidecar layer over immutable sub-C outputs. Phase A builds deterministic evidence metrics, frequency-analysis artifacts, a reviewed macro vocab/config at `configs/macro_plan/v1/macro_plan_vocab.yaml`, and golden analysis outputs. Phase B reads the locked vocab/config and writes `macro_core.parquet`, `derivation_evidence.parquet`, `effective_conditioning.yaml`, `provenance.yaml`, `manifest.yaml`, and `_SUCCESS` under `data/processed/sub_d/<release>/<region>/`.

**Tech Stack:** Python 3.11+, pyarrow, PyYAML, pytest, uv. Reuses sub-C public readers and neutral shared determinism helpers after Task 1 extracts them.

**Spec reference:** `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md` at commit `ce62d77`.

---

## Branch Discipline

All work stays on branch `phase-1-sub-D-macro-plan-derivation`.

Every implementer dispatch must include this text verbatim:

> Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

Each task has an atomic commit checkpoint. Review gates are explicit halt points; do not continue past a gate until the reviewer approves.

## Test Discipline

Use TDD for implementation tasks:

1. Write the failing test.
2. Run the focused test and confirm the expected failure.
3. Implement the smallest code change.
4. Run the focused test and confirm pass.
5. Run the task-level regression command.
6. Commit.

If real cached Singapore data violates an invariant, stop and escalate. Do not weaken tests to pass.

## Phase Boundaries And Review Gates

- **Gate 1:** Neutral helper extraction. Sub-C tests must pass with zero behavior change before any sub-D code begins.
- **Gate 2:** Vocab proposal review. Frequency-analysis outputs and proposed vocab/bucket cuts are reviewed before `configs/macro_plan/v1/macro_plan_vocab.yaml` is committed.
- **Phase A to Phase B contract:** Phase B begins only after `configs/macro_plan/v1/macro_plan_vocab.yaml` and golden frequency-analysis artifacts are committed.

Gate 1 fallback triggers are concrete:

- Hidden coupling discovered mid-extraction that would require breaking helper signatures used by sub-C.
- Sub-C tests fail after repointing imports and the fix path is non-trivial.
- Sub-C tests pass but a byte-output or golden-artifact comparison shows behavior changed.

If any trigger fires, halt at Gate 1 and ask whether to use the spec fallback: duplicate determinism helpers locally in `cfm.data.sub_d` rather than sharing neutral helpers.

## Dependency Map

- Task 1 blocks every sub-D task.
- Task 3 depends on Task 2.
- Task 4 depends on Task 3.
- Task 5 depends on Tasks 3 and 4.
- Task 6 depends on Tasks 3, 4, and 5.
- Task 7 depends on Tasks 4, 5, and 6.
- Task 8 depends on Task 7 and reviewer approval at Gate 2.
- Phase B tasks depend on Task 8.

## File Map

**Create shared helpers:**
- `src/cfm/data/io.py` - neutral YAML/parquet write helpers.
- `src/cfm/data/determinism.py` - neutral digest helpers and exclusion grammar.
- `tests/data/test_io_shared.py` - neutral I/O helper tests.
- `tests/data/test_determinism_shared.py` - neutral determinism helper tests.

**Modify sub-C to use shared helpers:**
- `src/cfm/data/sub_c/io.py`
- `src/cfm/data/sub_c/determinism.py`

**Create sub-D package:**
- `src/cfm/data/sub_d/__init__.py`
- `src/cfm/data/sub_d/enums.py`
- `src/cfm/data/sub_d/errors.py`
- `src/cfm/data/sub_d/versions.py`
- `src/cfm/data/sub_d/lattice.py`
- `src/cfm/data/sub_d/sub_c_reader.py`
- `src/cfm/data/sub_d/evidence.py`
- `src/cfm/data/sub_d/frequency_analysis.py`
- `src/cfm/data/sub_d/macro_vocab.py`
- `src/cfm/data/sub_d/io.py`
- `src/cfm/data/sub_d/conditioning.py`
- `src/cfm/data/sub_d/provenance.py`
- `src/cfm/data/sub_d/manifest.py`
- `src/cfm/data/sub_d/validator.py`
- `src/cfm/data/sub_d/pipeline.py`

**Create scripts:**
- `scripts/analyse_macro_plan_frequencies.py`
- `scripts/derive_macro_plan.py`
- `scripts/validate_macro_plan.py`

**Create config/artifacts:**
- `configs/macro_plan/v1/macro_plan_vocab.yaml`
- `reports/phase-1-sub-D/*.yaml`
- `tests/golden/sub_d/frequency_analysis/*.yaml`

**Create tests:**
- `tests/data/sub_d/__init__.py`
- `tests/data/sub_d/conftest.py`
- `tests/data/sub_d/test_versions.py`
- `tests/data/sub_d/test_lattice.py`
- `tests/data/sub_d/test_sub_c_reader.py`
- `tests/data/sub_d/test_evidence.py`
- `tests/data/sub_d/test_frequency_analysis.py`
- `tests/data/sub_d/test_macro_vocab.py`
- `tests/data/sub_d/test_io.py`
- `tests/data/sub_d/test_conditioning.py`
- `tests/data/sub_d/test_provenance.py`
- `tests/data/sub_d/test_manifest.py`
- `tests/data/sub_d/test_validator.py`
- `tests/data/sub_d/test_pipeline.py`
- `tests/data/sub_d/test_cli.py`
- `tests/data/sub_d/test_singapore_integration.py`

---

## Task 1: Neutral Determinism Helper Extraction

**Depends on:** none

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/io.py`
- Create: `src/cfm/data/determinism.py`
- Create: `tests/data/test_io_shared.py`
- Create: `tests/data/test_determinism_shared.py`
- Modify: `src/cfm/data/sub_c/io.py`
- Modify: `src/cfm/data/sub_c/determinism.py`

- [ ] **Step 1: Write failing shared I/O tests**

Create `tests/data/test_io_shared.py` with tests for:

```python
from cfm.data.io import PARQUET_WRITE_KWARGS, canonicalize_yaml


def test_shared_parquet_write_kwargs_match_sub_c_contract():
    assert PARQUET_WRITE_KWARGS["compression"] == "snappy"
    assert PARQUET_WRITE_KWARGS["row_group_size"] == 50_000
    assert PARQUET_WRITE_KWARGS["data_page_size"] == 1_048_576
    assert PARQUET_WRITE_KWARGS["write_batch_size"] == 10_000
    assert PARQUET_WRITE_KWARGS["use_dictionary"] is True
    assert PARQUET_WRITE_KWARGS["write_statistics"] is True
    assert PARQUET_WRITE_KWARGS["use_compliant_nested_type"] is True
    assert PARQUET_WRITE_KWARGS["version"] == "2.6"


def test_shared_canonicalize_yaml_is_byte_stable_and_sorted():
    data = {"z": 1, "a": {"b": 2, "a": 1}}
    first = canonicalize_yaml(data)
    second = canonicalize_yaml(data)
    assert first == second
    assert first.splitlines()[0].startswith("a:")
```

- [ ] **Step 2: Write failing shared determinism tests**

Create `tests/data/test_determinism_shared.py` with tests for:

```python
from cfm.data.determinism import (
    compute_sha256,
    compute_sha256_excluding,
    path_in_excluded,
)
from cfm.data.io import canonicalize_yaml


EXCLUSIONS = {
    "*": ["*_sha256"],
    "manifest.yaml": ["initial_extraction.started_utc"],
}


def test_path_in_excluded_uses_final_segment_sha_suffix():
    assert path_in_excluded("tiles[0].provenance_sha256", "*", EXCLUSIONS)
    assert not path_in_excluded("sha256_input", "*", EXCLUSIONS)


def test_compute_sha256_excluding_uses_supplied_exclusion_table():
    with_digest = {"a": 1, "nested": {"file_sha256": "abc"}}
    clean = {"a": 1, "nested": {}}
    assert compute_sha256_excluding(with_digest, "*", EXCLUSIONS) == compute_sha256(
        canonicalize_yaml(clean).encode("utf-8")
    )


def test_file_specific_timestamp_exclusion():
    first = {"initial_extraction": {"started_utc": "2026-01-01T00:00:00Z", "tile_count": 2}}
    second = {"initial_extraction": {"started_utc": "2026-01-02T00:00:00Z", "tile_count": 2}}
    assert compute_sha256_excluding(first, "manifest.yaml", EXCLUSIONS) == (
        compute_sha256_excluding(second, "manifest.yaml", EXCLUSIONS)
    )
```

- [ ] **Step 3: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/test_io_shared.py tests/data/test_determinism_shared.py -q
```

Expected: import failures for `cfm.data.io` and `cfm.data.determinism`.

- [ ] **Step 4: Implement neutral helpers**

Create:

```python
# src/cfm/data/io.py
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PARQUET_WRITE_KWARGS: dict = {
    "compression": "snappy",
    "row_group_size": 50_000,
    "data_page_size": 1_048_576,
    "write_batch_size": 10_000,
    "use_dictionary": True,
    "write_statistics": True,
    "use_compliant_nested_type": True,
    "version": "2.6",
}


def write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, **PARQUET_WRITE_KWARGS)


def canonicalize_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=yaml.SafeDumper,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        width=4096,
    )
```

Create:

```python
# src/cfm/data/determinism.py
from __future__ import annotations

import hashlib
import re

from cfm.data.io import canonicalize_yaml

ExclusionTable = dict[str, list[str]]


def compute_sha256(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


def path_in_excluded(dotted_path: str, file_key: str, exclusions: ExclusionTable) -> bool:
    final_seg = _final_segment(dotted_path)
    for pattern in exclusions.get("*", []):
        if _matches(pattern, dotted_path, final_seg):
            return True
    for pattern in exclusions.get(file_key, []):
        if _matches(pattern, dotted_path, final_seg):
            return True
    return False


def compute_sha256_excluding(data: dict, file_key: str, exclusions: ExclusionTable) -> str:
    stripped = _strip_excluded(data, file_key, exclusions)
    return compute_sha256(canonicalize_yaml(stripped).encode("utf-8"))


def _final_segment(dotted_path: str) -> str:
    last = dotted_path.rsplit(".", 1)[-1]
    return re.sub(r"\[\d+\]$", "", last)


def _matches(pattern: str, dotted_path: str, final_seg: str) -> bool:
    if pattern.startswith("*"):
        return final_seg.endswith(pattern[1:])
    return pattern == dotted_path


def _strip_excluded(
    data: object,
    file_key: str,
    exclusions: ExclusionTable,
    prefix: str = "",
) -> object:
    if isinstance(data, dict):
        result: dict = {}
        for key, value in data.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if path_in_excluded(child_path, file_key, exclusions):
                continue
            result[key] = _strip_excluded(value, file_key, exclusions, child_path)
        return result
    if isinstance(data, list):
        return [
            _strip_excluded(item, file_key, exclusions, f"{prefix}[{index}]")
            for index, item in enumerate(data)
        ]
    return data
```

- [ ] **Step 5: Repoint sub-C wrappers without changing public sub-C imports**

Modify `src/cfm/data/sub_c/io.py` so `_PARQUET_WRITE_KWARGS`, `write_parquet`, and `canonicalize_yaml` delegate to `cfm.data.io` while preserving existing sub-C names:

```python
from cfm.data.io import (
    PARQUET_WRITE_KWARGS as _PARQUET_WRITE_KWARGS,
    canonicalize_yaml,
    write_parquet,
)
```

Remove the old local definitions after imports are in place.

Modify `src/cfm/data/sub_c/determinism.py` so wrapper functions preserve the old API:

```python
from cfm.data.determinism import (
    compute_sha256,
    compute_sha256_excluding as _compute_sha256_excluding,
    path_in_excluded as _path_in_excluded,
)


def path_in_excluded(dotted_path: str, file_key: str) -> bool:
    return _path_in_excluded(dotted_path, file_key, EXCLUDED_FROM_SHA)


def compute_sha256_excluding(data: dict, file_key: str) -> str:
    return _compute_sha256_excluding(data, file_key, EXCLUDED_FROM_SHA)
```

Keep `EXCLUDED_FROM_SHA` and `EXCLUDED_FROM_TEST_COMPARE` in `sub_c.determinism`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/data/test_io_shared.py tests/data/test_determinism_shared.py tests/data/sub_c/test_io_determinism.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Run full fast test suite for Gate 1**

Run:

```bash
uv run pytest -q
```

Expected: all fast tests pass with the existing xfail count.

- [ ] **Step 8: Commit Gate 1 extraction**

Run:

```bash
git add src/cfm/data/io.py src/cfm/data/determinism.py src/cfm/data/sub_c/io.py src/cfm/data/sub_c/determinism.py tests/data/test_io_shared.py tests/data/test_determinism_shared.py
git commit -m "refactor(data): extract shared determinism helpers"
```

Expected: one commit containing only neutral helper extraction and sub-C repointing.

## Gate 1: Reviewer Approval

**Depends on:** Task 1

Stop here. Report:

- Commit SHA.
- Full fast-suite output.
- Whether any fallback trigger fired.

Reviewer must approve before Task 2 begins.

---

## Task 2: Sub-D Package Skeleton And Version Namespace Helper

**Depends on:** Gate 1 approval

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/__init__.py`
- Create: `src/cfm/data/sub_d/errors.py`
- Create: `src/cfm/data/sub_d/versions.py`
- Create: `tests/data/sub_d/__init__.py`
- Create: `tests/data/sub_d/test_versions.py`

- [ ] **Step 1: Write failing version tests**

Create `tests/data/sub_d/test_versions.py`:

```python
import pytest

from cfm.data.sub_d.versions import (
    VersionMismatchError,
    VersionNamespace,
    VersionNamespaceError,
    VersionRef,
    compare_version,
)


def test_compare_version_accepts_same_namespace_and_value():
    expected = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    actual = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    compare_version(VersionNamespace.DATA_SHAPE, expected, actual)


def test_compare_version_rejects_value_mismatch():
    expected = VersionRef(VersionNamespace.VOCAB, "1.0")
    actual = VersionRef(VersionNamespace.VOCAB, "1.1")
    with pytest.raises(VersionMismatchError):
        compare_version(VersionNamespace.VOCAB, expected, actual)


def test_compare_version_rejects_cross_namespace_expected():
    expected = VersionRef(VersionNamespace.ARTIFACT_FORMAT, "1.0")
    actual = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    with pytest.raises(VersionNamespaceError):
        compare_version(VersionNamespace.DATA_SHAPE, expected, actual)


def test_compare_version_rejects_cross_namespace_actual():
    expected = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    actual = VersionRef(VersionNamespace.ARTIFACT_FORMAT, "1.0")
    with pytest.raises(VersionNamespaceError):
        compare_version(VersionNamespace.DATA_SHAPE, expected, actual)
```

- [ ] **Step 2: Run focused test and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_versions.py -q
```

Expected: import failure for `cfm.data.sub_d.versions`.

- [ ] **Step 3: Implement version helper**

Create `src/cfm/data/sub_d/errors.py`:

```python
from __future__ import annotations


class SubDValidationError(Exception):
    pass


class VersionNamespaceError(SubDValidationError):
    pass


class VersionMismatchError(SubDValidationError):
    pass
```

Create `src/cfm/data/sub_d/versions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cfm.data.sub_d.errors import VersionMismatchError, VersionNamespaceError


class VersionNamespace(str, Enum):
    ARTIFACT_FORMAT = "artifact_format"
    DATA_SHAPE = "data_shape"
    VOCAB = "vocab"
    DERIVATION = "derivation"
    VALIDATOR = "validator"


@dataclass(frozen=True)
class VersionRef:
    namespace: VersionNamespace
    value: str


def compare_version(namespace: VersionNamespace, expected: VersionRef, actual: VersionRef) -> None:
    if expected.namespace != namespace or actual.namespace != namespace:
        raise VersionNamespaceError(
            f"version namespace mismatch: comparison={namespace.value}, "
            f"expected={expected.namespace.value}, actual={actual.namespace.value}"
        )
    if expected.value != actual.value:
        raise VersionMismatchError(
            f"version mismatch in {namespace.value}: expected {expected.value}, got {actual.value}"
        )
```

Create `src/cfm/data/sub_d/__init__.py` exporting the version helper.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_versions.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d tests/data/sub_d
git commit -m "feat(sub_d): add version namespace helper"
```

## Task 3: Fixed Lattice Utilities

**Depends on:** Task 2

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/enums.py`
- Create: `src/cfm/data/sub_d/lattice.py`
- Create: `tests/data/sub_d/test_lattice.py`

- [ ] **Step 1: Write failing lattice tests**

Create tests asserting:

- `iter_cell_slots()` yields 64 slots indexed 0-63.
- `iter_internal_edge_slots()` yields 112 slots.
- `iter_external_edge_slots()` yields 32 slots.
- Internal edge scope is `active`, `fully_masked`, or `scope_boundary` based on endpoint cell scopes.
- External edge scope is `external_deferred` when adjacent cell is active and `fully_masked` when adjacent cell is not active.

Use these exact test names:

- `test_cell_lattice_has_64_slots_in_row_major_order`
- `test_internal_edge_lattice_has_112_slots`
- `test_external_edge_lattice_has_32_slots`
- `test_internal_edge_scope_distinguishes_active_masked_and_boundary`
- `test_external_edge_scope_uses_deferred_only_when_interior_cell_active`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_lattice.py -q
```

Expected: import failure for `cfm.data.sub_d.lattice`.

- [ ] **Step 3: Implement lattice API**

Implement:

```python
CELL_GRID_SIZE = 8
CELL_SLOT_COUNT = 64
INTERNAL_EDGE_SLOT_COUNT = 112
EXTERNAL_EDGE_SLOT_COUNT = 32

@dataclass(frozen=True)
class CellSlot:
    slot_index: int
    cell_i: int
    cell_j: int

@dataclass(frozen=True)
class EdgeSlot:
    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int
```

`axis` uses the sub-C convention: `0=x`, `1=y`.

Implement these exact functions:

- `iter_cell_slots() -> list[CellSlot]`
- `iter_internal_edge_slots() -> list[EdgeSlot]`
- `iter_external_edge_slots() -> list[EdgeSlot]`
- `derive_internal_edge_scope(lower_active: bool, upper_active: bool) -> Scope`
- `derive_external_edge_scope(interior_active: bool) -> Scope`

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_lattice.py -q
```

Expected: all lattice tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/enums.py src/cfm/data/sub_d/lattice.py tests/data/sub_d/test_lattice.py
git commit -m "feat(sub_d): add fixed macro lattice utilities"
```

## Task 4: Sub-C Sidecar Reader And Digest Anchors

**Depends on:** Task 3

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/sub_c_reader.py`
- Create: `tests/data/sub_d/test_sub_c_reader.py`

- [ ] **Step 1: Write failing reader tests**

Use a tiny synthetic sub-C-like directory with `manifest.yaml`, `_SUCCESS`, one tile directory, and parquet files. Tests:

- `test_reader_refuses_region_without_success_marker`
- `test_reader_uses_manifest_tile_inventory_not_filesystem_glob`
- `test_reader_loads_tile_parquets_with_parquetfile`
- `test_reader_computes_digest_anchors_for_sub_c_inputs`

The `ParquetFile` test monkeypatches `pyarrow.parquet.ParquetFile` to record calls and fails if `pyarrow.parquet.read_table` is used.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_sub_c_reader.py -q
```

Expected: import failure for `cfm.data.sub_d.sub_c_reader`.

- [ ] **Step 3: Implement reader API**

Implement dataclasses:

```python
@dataclass(frozen=True)
class SubCTilePaths:
    tile_i: int
    tile_j: int
    tile_dir: Path
    cells: Path
    features: Path
    crossings: Path
    meta: Path
    provenance: Path

@dataclass(frozen=True)
class SubCTileInputs:
    paths: SubCTilePaths
    cells: pa.Table
    features: pa.Table
    crossings: pa.Table
    meta: dict
    provenance: dict
    digests: dict[str, str]
```

Implement these exact functions:

- `read_sub_c_manifest(region_dir: Path) -> dict`
- `iter_sub_c_tile_paths(region_dir: Path) -> list[SubCTilePaths]`
- `read_sub_c_tile_inputs(paths: SubCTilePaths) -> SubCTileInputs`

Use `pyarrow.parquet.ParquetFile(path).read()`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_sub_c_reader.py -q
```

Expected: all reader tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/sub_c_reader.py tests/data/sub_d/test_sub_c_reader.py
git commit -m "feat(sub_d): add sub-C sidecar reader"
```

## Task 5: Derivation Evidence Metric Primitives

**Depends on:** Tasks 3 and 4

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/evidence.py`
- Create: `tests/data/sub_d/test_evidence.py`

- [ ] **Step 1: Write Layer 1 failing evidence tests**

Tests use synthetic pyarrow tables only:

- `test_cell_scope_from_sub_c_cells_marks_complement_masked`
- `test_zoning_evidence_counts_class_composition_without_density_thresholds`
- `test_density_evidence_uses_building_footprint_ratio_only`
- `test_road_evidence_joins_crossings_to_features_by_source_feature_id`
- `test_road_evidence_ignores_non_road_crossings`

The zoning test must assert footprint-ratio intensity is not used to assign a zoning class in this primitive layer.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_evidence.py -q
```

Expected: import failure for `cfm.data.sub_d.evidence`.

- [ ] **Step 3: Implement evidence API**

Implement:

```python
@dataclass(frozen=True)
class EvidenceMetric:
    slot_kind: SlotKind
    slot_index: int
    metric_namespace: MetricNamespace
    metric_name: str
    value: float | int | str | bool
    derivation_version: str
```

Implement these exact functions:

- `derive_cell_scope_metrics(cells: pa.Table) -> dict[tuple[int, int], bool]`
- `derive_zoning_evidence(features: pa.Table, cells: pa.Table) -> list[EvidenceMetric]`
- `derive_density_evidence(features: pa.Table, cells: pa.Table) -> list[EvidenceMetric]`
- `derive_road_skeleton_evidence(crossings: pa.Table, features: pa.Table) -> list[EvidenceMetric]`

Keep assignment of final zoning/density/road classes out of this task; this task computes deterministic evidence only.

- [ ] **Step 4: Run Layer 1 evidence tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_evidence.py -q
```

Expected: all evidence primitive tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/evidence.py tests/data/sub_d/test_evidence.py
git commit -m "feat(sub_d): add derivation evidence primitives"
```

## Task 6: Frequency Analysis Artifacts

**Depends on:** Task 5

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/frequency_analysis.py`
- Create: `scripts/analyse_macro_plan_frequencies.py`
- Create: `tests/data/sub_d/test_frequency_analysis.py`

- [ ] **Step 1: Write failing frequency-analysis tests**

Layer 3 discipline starts here, but tests use synthetic fixtures first:

- `test_frequency_analysis_output_is_byte_identical_for_same_inputs`
- `test_frequency_analysis_enforces_non_empty_locked_buckets`
- `test_frequency_analysis_records_marginal_cost_monotonicity`
- `test_frequency_analysis_writes_reviewable_proposal_sections`
- `test_frequency_analysis_records_zoning_orthogonality_comparison`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_frequency_analysis.py -q
```

Expected: import failure for `cfm.data.sub_d.frequency_analysis`.

- [ ] **Step 3: Implement analysis artifact writer**

Implement these exact functions:

- `build_frequency_analysis(inputs: list[SubCTileInputs]) -> dict`
- `write_frequency_analysis(analysis: dict, path: Path) -> None`
- `validate_frequency_analysis(analysis: dict) -> None`

Write machine-readable YAML to `reports/phase-1-sub-D/<analysis_name>.yaml` using neutral `canonicalize_yaml`.

The CLI accepts:

```bash
uv run python scripts/analyse_macro_plan_frequencies.py \
  --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \
  --output-dir reports/phase-1-sub-D
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_frequency_analysis.py -q
```

Expected: all frequency-analysis unit tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/frequency_analysis.py scripts/analyse_macro_plan_frequencies.py tests/data/sub_d/test_frequency_analysis.py
git commit -m "feat(sub_d): add macro frequency analysis artifacts"
```

## Task 7: Deterministic Singapore Subset Selection And Review Proposal

**Depends on:** Task 6

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Modify: `src/cfm/data/sub_d/frequency_analysis.py`
- Modify: `scripts/analyse_macro_plan_frequencies.py`
- Create: `reports/phase-1-sub-D/README.md`
- Create after running CLI: `reports/phase-1-sub-D/*.yaml`

- [ ] **Step 1: Add tests for subset rationale**

Add tests:

- `test_subset_selection_is_deterministic_for_same_analysis`
- `test_subset_selection_rejects_random_or_fastest_only_policy`
- `test_subset_selection_records_rationale_per_tile`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_frequency_analysis.py -q
```

Expected: new subset tests fail.

- [ ] **Step 3: Implement deterministic subset selector**

Implement `select_layer3_subset(analysis: dict, max_tiles: int = 12) -> list[dict]`.

Each selected tile record includes `tile_i`, `tile_j`, and `rationale`. Rationale must mention one or more covered dimensions: zoning evidence spread, density spread, road-skeleton spread, coastal/inland/riverside coverage, active/masked cell or edge cases.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_frequency_analysis.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Write the proposal README**

Create `reports/phase-1-sub-D/README.md` with:

- a list of expected proposal artifacts in `reports/phase-1-sub-D/`
- the Gate 2 review process
- the exact file reviewers inspect first: `macro_vocab_proposal.yaml`
- what reviewers check: enum tokens, bucket cuts, subset rationale, marginal-cost-of-cut justifications, and frequency-analysis digests
- the rule that Gate 2 cannot close without real Singapore sub-C output

- [ ] **Step 6: Run the proposal CLI if sub-C output exists**

Run:

```bash
test -d data/processed/sub_c/2026-04-15.0/singapore && \
uv run python scripts/analyse_macro_plan_frequencies.py \
  --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \
  --output-dir reports/phase-1-sub-D \
  --proposal-only
```

Expected if sub-C output exists: proposal YAML files are written under `reports/phase-1-sub-D/`.
Expected if sub-C output does not exist: command exits before running; note the absence in the review handoff and do not invent proposal artifacts. Gate 2 cannot close without real Singapore sub-C output; there is no synthetic-data shortcut for vocab approval.

- [ ] **Step 7: Commit proposal machinery and any generated proposal artifacts**

Run:

```bash
git add src/cfm/data/sub_d/frequency_analysis.py scripts/analyse_macro_plan_frequencies.py tests/data/sub_d/test_frequency_analysis.py reports/phase-1-sub-D
git commit -m "expt(sub_d): generate macro vocab proposal artifacts"
```

## Gate 2: Reviewer Approval Of Vocab Proposal

**Depends on:** Task 7

Stop here. Report:

- Frequency-analysis artifact paths.
- Proposed enum tokens and bucket cuts.
- Deterministic Layer 3 tile subset rationale.
- Marginal-cost-of-cut justifications.
- Whether sub-C output was unavailable locally.

Reviewer must approve before Task 8 commits locked vocab/config. No sidecar derivation artifacts are written before this gate closes. If real Singapore sub-C output was unavailable, halt here until it is available and the proposal CLI has produced real-data artifacts.

## Task 8: Lock Macro Vocab And Golden Frequency Artifacts

**Depends on:** Gate 2 approval

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `configs/macro_plan/v1/macro_plan_vocab.yaml`
- Create: `tests/golden/sub_d/frequency_analysis/*.yaml`
- Create: `src/cfm/data/sub_d/macro_vocab.py`
- Create: `scripts/promote_macro_vocab.py`
- Create: `tests/data/sub_d/test_macro_vocab.py`

- [ ] **Step 1: Write failing macro vocab loader tests**

Tests:

- `test_macro_vocab_loads_locked_artifact`
- `test_macro_vocab_rejects_duplicate_token_ids`
- `test_macro_vocab_rejects_duplicate_token_names`
- `test_macro_vocab_records_frequency_analysis_digests`
- `test_macro_vocab_has_append_only_flags_for_every_enum`
- `test_promote_macro_vocab_derives_locked_artifact_from_proposal`
- `test_promote_macro_vocab_diff_is_status_marker_only`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_macro_vocab.py -q
```

Expected: import failure, missing `scripts/promote_macro_vocab.py`, or missing `configs/macro_plan/v1/macro_plan_vocab.yaml`.

- [ ] **Step 3: Implement promotion script**

Create `scripts/promote_macro_vocab.py`. It accepts:

```bash
uv run python scripts/promote_macro_vocab.py \
  --proposal reports/phase-1-sub-D/macro_vocab_proposal.yaml \
  --output configs/macro_plan/v1/macro_plan_vocab.yaml
```

The script canonicalizes YAML and derives the locked artifact from the approved proposal. The only allowed byte diff from proposal to locked artifact is the status marker, for example `status: proposal` to `status: locked`. `test_promote_macro_vocab_diff_is_status_marker_only` must read both files as bytes, normalize that marker back to `status: proposal`, and assert byte identity. No human hand-editing is allowed between the reviewed proposal and the locked artifact.

- [ ] **Step 4: Run promotion script after Gate 2 approval**

Run:

```bash
uv run python scripts/promote_macro_vocab.py \
  --proposal reports/phase-1-sub-D/macro_vocab_proposal.yaml \
  --output configs/macro_plan/v1/macro_plan_vocab.yaml
```

Expected: `configs/macro_plan/v1/macro_plan_vocab.yaml` is written. Do not hand-edit this file.

- [ ] **Step 5: Copy approved golden frequency-analysis artifacts**

Copy approved analysis artifacts to:

```text
tests/golden/sub_d/frequency_analysis/<analysis_name>.yaml
```

- [ ] **Step 6: Implement macro vocab loader**

Implement these exact functions:

- `load_macro_vocab(path: Path) -> dict`
- `validate_macro_vocab(data: dict) -> None`
- `token_name_to_id(section: str, token_name: str, vocab: dict) -> int`
- `token_id_to_name(section: str, token_id: int, vocab: dict) -> str`

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_macro_vocab.py tests/data/sub_d/test_frequency_analysis.py -q
```

Expected: all selected tests pass and golden comparisons pass.

- [ ] **Step 8: Commit locked vocab**

Run:

```bash
git add configs/macro_plan/v1/macro_plan_vocab.yaml tests/golden/sub_d/frequency_analysis src/cfm/data/sub_d/macro_vocab.py scripts/promote_macro_vocab.py tests/data/sub_d/test_macro_vocab.py tests/data/sub_d/test_frequency_analysis.py
git commit -m "data(sub_d): lock macro plan vocab v1"
```

---

## Task 9: Macro Core And Derivation Evidence Writers

**Depends on:** Task 8

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/io.py`
- Create: `tests/data/sub_d/test_io.py`

- [ ] **Step 1: Write failing I/O schema tests**

Use these exact test names:

- `test_macro_core_schema_matches_spec_11_2`
- `test_derivation_evidence_schema_matches_spec_11_3`
- `test_macro_core_writer_sorts_by_slot_kind_slot_index`
- `test_derivation_evidence_writer_sorts_by_canonical_key`
- `test_writers_are_byte_identical_for_same_rows`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_io.py -q
```

Expected: import failure for `cfm.data.sub_d.io`.

- [ ] **Step 3: Implement parquet schemas and writers**

Define `_MACRO_CORE_SCHEMA`, `_DERIVATION_EVIDENCE_SCHEMA`, row dataclasses, `write_macro_core_parquet`, `write_derivation_evidence_parquet`, and read helpers. Use neutral `write_parquet`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_io.py -q
```

Expected: all I/O tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/io.py tests/data/sub_d/test_io.py
git commit -m "feat(sub_d): add macro core artifact writers"
```

## Task 10: Effective Conditioning Overlay

**Depends on:** Task 9

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/conditioning.py`
- Create: `tests/data/sub_d/test_conditioning.py`

- [ ] **Step 1: Write failing conditioning tests**

Use these exact test names:

- `test_effective_conditioning_copies_schema_driven_sub_c_owned_fields`
- `test_effective_conditioning_fills_population_density_bucket`
- `test_effective_conditioning_does_not_copy_owner_marker_as_conditioning`
- `test_effective_conditioning_records_composite_versions_and_digests`
- `test_effective_conditioning_schema_uses_effective_conditioning_schema_version`
- `test_effective_conditioning_yaml_is_canonical`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_conditioning.py -q
```

Expected: import failure for `cfm.data.sub_d.conditioning`.

- [ ] **Step 3: Implement conditioning overlay**

Implement these exact functions:

- `build_effective_conditioning(meta: dict, manifest: dict, population_density_bucket: int, versions: dict, digests: dict) -> dict`
- `write_effective_conditioning(data: dict, path: Path) -> None`

Copy every field in `conditioning_per_tile` except fields ending with `_owner` and fields explicitly owned by sub-D before filling `population_density_bucket`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_conditioning.py -q
```

Expected: all conditioning tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/conditioning.py tests/data/sub_d/test_conditioning.py
git commit -m "feat(sub_d): add effective conditioning overlay"
```

## Task 11: Per-Tile Provenance

**Depends on:** Tasks 9 and 10

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/provenance.py`
- Create: `tests/data/sub_d/test_provenance.py`

- [ ] **Step 1: Write failing provenance tests**

Use these exact test names:

- `test_provenance_schema_uses_provenance_schema_version_not_bare_schema_version`
- `test_provenance_records_sub_c_input_digests`
- `test_provenance_records_locked_vocab_and_derivation_versions`
- `test_provenance_sha_excludes_extracted_utc_and_output_sha_fields`
- `test_provenance_sha_includes_rerun_reason`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_provenance.py -q
```

Expected: import failure for `cfm.data.sub_d.provenance`.

- [ ] **Step 3: Implement provenance builder and sub-D exclusions**

Implement `SUB_D_EXCLUDED_FROM_SHA` with file-keyed timestamp exclusions and final-segment `*_sha256` grammar. Implement these exact functions:

- `build_tile_provenance(tile_i: int, tile_j: int, extraction: dict, inputs: dict, versions: dict, outputs: dict) -> dict`
- `provenance_sha256(data: dict) -> str`
- `write_provenance(data: dict, path: Path) -> None`

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_provenance.py -q
```

Expected: all provenance tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/provenance.py tests/data/sub_d/test_provenance.py
git commit -m "feat(sub_d): add per-tile provenance"
```

## Task 12: Region Manifest And Inventory Digest Chain

**Depends on:** Task 11

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/manifest.py`
- Create: `tests/data/sub_d/test_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Use these exact test names:

- `test_manifest_schema_uses_manifest_schema_version`
- `test_manifest_tiles_sorted_by_tile_i_tile_j`
- `test_manifest_config_copied_from_sub_c_and_validated`
- `test_manifest_provenance_sha_matches_tile_provenance_sha`
- `test_success_marker_written_only_by_explicit_function`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_manifest.py -q
```

Expected: import failure for `cfm.data.sub_d.manifest`.

- [ ] **Step 3: Implement manifest API**

Implement these exact functions:

- `build_manifest(release: str, region: str, region_crs: str, sub_c_manifest: dict, inputs: dict, versions: dict, config: dict, initial_extraction: dict, tile_provenances: list[dict]) -> dict`
- `write_manifest(data: dict, path: Path) -> None`
- `read_manifest(path: Path) -> dict`
- `aggregate_tile_inventory(tile_provenances: list[dict]) -> list[dict]`
- `write_success_marker(region_dir: Path) -> None`

Validator-facing manifest config must include `config_source: "sub_c_manifest.config"` and assert copied values match sub-C.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_manifest.py -q
```

Expected: all manifest tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/manifest.py tests/data/sub_d/test_manifest.py
git commit -m "feat(sub_d): add region manifest"
```

## Task 13: Sidecar Validators

**Depends on:** Tasks 9 through 12

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/validator.py`
- Create: `tests/data/sub_d/test_validator.py`

- [ ] **Step 1: Write failing validator tests**

Use these exact test names:

- `test_validator_rejects_missing_macro_core_rows`
- `test_validator_rejects_target_class_on_masked_slot`
- `test_validator_rejects_effective_conditioning_digest_mismatch`
- `test_validator_rejects_sub_c_input_digest_mismatch`
- `test_validator_uses_compare_version_for_namespace_checks`
- `test_validator_files_do_not_compare_version_strings_directly`
- `test_validator_rejects_manifest_config_drift_from_sub_c`
- `test_validator_rejects_provenance_output_sha_mismatch`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_validator.py -q
```

Expected: import failure for `cfm.data.sub_d.validator`.

- [ ] **Step 3: Implement validators**

Implement these exact functions:

- `validate_tile(tile_dir: Path, sub_c_inputs: SubCTileInputs, macro_vocab: dict) -> None`
- `validate_region(region_dir: Path, sub_c_region_dir: Path) -> None`

All version comparisons use `compare_version`.

The meta-test `test_validator_files_do_not_compare_version_strings_directly` scans `src/cfm/data/sub_d/validator.py` and any future `src/cfm/data/sub_d/validator_*.py` files with `ast.parse`. It fails on any `ast.Compare` using `==` or `!=` where either side references a name, attribute, or subscript containing `_version` or `version`. The only sanctioned version equality path is `compare_version` from `src/cfm/data/sub_d/versions.py`; validator files must not compare version strings directly.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_validator.py -q
```

Expected: all validator tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/cfm/data/sub_d/validator.py tests/data/sub_d/test_validator.py
git commit -m "feat(sub_d): add sidecar validators"
```

## Task 14: Sidecar Pipeline And CLI

**Depends on:** Task 13

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `src/cfm/data/sub_d/pipeline.py`
- Create: `scripts/derive_macro_plan.py`
- Create: `scripts/validate_macro_plan.py`
- Create: `tests/data/sub_d/test_pipeline.py`
- Create: `tests/data/sub_d/test_cli.py`

- [ ] **Step 1: Write failing pipeline and CLI tests**

Use these exact test names:

- `test_pipeline_refuses_to_run_without_locked_macro_vocab`
- `test_pipeline_writes_all_per_tile_sidecar_artifacts`
- `test_pipeline_writes_manifest_then_success_after_validation`
- `test_pipeline_is_byte_identical_on_same_inputs`
- `test_derive_macro_plan_cli_resolves_default_paths`
- `test_validate_macro_plan_cli_returns_nonzero_on_validator_error`

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run pytest tests/data/sub_d/test_pipeline.py tests/data/sub_d/test_cli.py -q
```

Expected: import failures for `cfm.data.sub_d.pipeline` and missing scripts.

- [ ] **Step 3: Implement pipeline**

Implement `derive_region_macro_plan(sub_c_region_dir: Path, output_dir: Path, macro_vocab_path: Path, *, release: str, region: str, commit_sha: str, extracted_utc: str | None = None) -> dict`.

Pipeline order:

1. Read sub-C manifest and tile inputs.
2. Load locked macro vocab.
3. Build fixed lattice targets.
4. Write per-tile `macro_core.parquet`.
5. Write per-tile `derivation_evidence.parquet`.
6. Write per-tile `effective_conditioning.yaml`.
7. Write per-tile `provenance.yaml`.
8. Write region `manifest.yaml`.
9. Run region validator.
10. Write `_SUCCESS`.

- [ ] **Step 4: Implement CLIs**

`scripts/derive_macro_plan.py` supports:

```bash
uv run python scripts/derive_macro_plan.py \
  --region singapore \
  --release 2026-04-15.0 \
  --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \
  --output-dir data/processed/sub_d/2026-04-15.0/singapore \
  --macro-vocab configs/macro_plan/v1/macro_plan_vocab.yaml
```

`scripts/validate_macro_plan.py` supports `--output-dir` and `--sub-c-dir`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/data/sub_d/test_pipeline.py tests/data/sub_d/test_cli.py -q
```

Expected: all pipeline and CLI tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/cfm/data/sub_d/pipeline.py scripts/derive_macro_plan.py scripts/validate_macro_plan.py tests/data/sub_d/test_pipeline.py tests/data/sub_d/test_cli.py
git commit -m "feat(sub_d): add sidecar derivation pipeline"
```

## Task 15: Layer 3 Cached Singapore Integration

**Depends on:** Task 14

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `tests/data/sub_d/test_singapore_integration.py`
- Modify: `tests/data/sub_d/conftest.py`

- [ ] **Step 1: Write slow integration tests**

Tests marked `@pytest.mark.slow`:

- `test_cached_singapore_subset_tile_ids_have_rationales`
- `test_cached_singapore_subset_derivation_passes_validation`
- `test_cached_singapore_subset_derivation_is_byte_identical_on_rerun`
- `test_cross_environment_determinism_gap_is_documented_if_not_run`

- [ ] **Step 2: Run collection and fast suite to ensure slow tests are deselected**

Run:

```bash
uv run pytest tests/data/sub_d/test_singapore_integration.py -q
```

Expected: slow tests are deselected under default marker config.

- [ ] **Step 3: Run slow tests only if cached sub-C output exists**

Run:

```bash
test -d data/processed/sub_c/2026-04-15.0/singapore && \
uv run pytest tests/data/sub_d/test_singapore_integration.py -q -m slow
```

Expected if sub-C output exists: slow tests pass.
Expected if sub-C output does not exist: command exits before pytest; document this in the task handoff.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/data/sub_d/test_singapore_integration.py tests/data/sub_d/conftest.py
git commit -m "test(sub_d): add cached Singapore integration coverage"
```

## Task 16: Final Verification And Handoff

**Depends on:** Task 15

**Implementer dispatch text:** Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config.

**Files:**
- Create: `docs/handoffs/2026-05-19-end-of-sub-D.md`

- [ ] **Step 1: Run full fast suite**

Run:

```bash
uv run pytest -q
```

Expected: all fast tests pass with expected deselected slow tests and existing xfail count.

- [ ] **Step 2: Run sub-D focused suite**

Run:

```bash
uv run pytest tests/data/sub_d -q
```

Expected: all fast sub-D tests pass; slow tests deselected by default.

- [ ] **Step 3: Run slow sub-D integration if cached sub-C output exists**

Run:

```bash
test -d data/processed/sub_c/2026-04-15.0/singapore && \
uv run pytest tests/data/sub_d/test_singapore_integration.py -q -m slow
```

Expected if sub-C output exists: slow tests pass.
Expected if absent: record "not run; sub-C output absent" in handoff.

- [ ] **Step 4: Write handoff**

Create `docs/handoffs/2026-05-19-end-of-sub-D.md` with:

- branch and commit range
- tests run and exact outcomes
- Gate 1 and Gate 2 approval notes
- locked macro vocab path
- generated sidecar output path if produced locally
- known residual risks, including cross-environment determinism status
- next sub-project: sub-E boundary contracts

- [ ] **Step 5: Commit handoff**

Run:

```bash
git add docs/handoffs/2026-05-19-end-of-sub-D.md
git commit -m "docs(handoff): end of sub-D macro plan derivation"
```

---

## Plan Self-Review Checklist

- Spec coverage: every section of the sub-D spec maps to at least one task.
- Review gates: Gate 1 and Gate 2 are explicit halt points.
- Dependencies: every task has a dependency line.
- Branch discipline: every implementer dispatch repeats the no-branch/no-push/no-PR rule.
- No sidecar artifacts ship before vocab approval.
- `derivation_evidence.parquet` is implemented from spec Section 11.3.
- `configs/macro_plan/v1/macro_plan_vocab.yaml` is the Phase A to Phase B handoff.
- Provenance and manifest are split into separate tasks.
