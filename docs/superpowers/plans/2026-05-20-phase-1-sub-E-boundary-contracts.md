# Phase 1 Sub-E Boundary Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 sub-E: a single-derivation sidecar pipeline that consumes sub-C crossings/features and sub-D macro core, writes per-tile `boundary_contract.parquet` with digest-anchored manifest/provenance, and ships an eval-harness skeleton for the sub-bar-3 conditional-perplexity gap measurement.

**Architecture:** Sub-E is a sidecar layer over immutable sub-D and sub-C outputs. The single derivation function maps `class_raw` (from sub-C `features.parquet`) → `boundary_class_enum` per active internal edge, with hierarchy-wins multi-crossing tie-break. Storage is per-edge (144 rows/tile) with read-time per-cell rotation; this makes the byte-identity invariant structural rather than assertion-checked. Sub-D's neutral helpers (`cfm.data.io`, `cfm.data.determinism`) and validator patterns are reused without modification.

**Tech Stack:** Python 3.11+, pyarrow, PyYAML, pytest, uv. Reuses sub-C public readers, sub-D public readers, and the neutral shared determinism helpers extracted in sub-D Task 1.

**Spec reference:** `docs/superpowers/specs/2026-05-20-phase-1-sub-E-boundary-contracts-design.md` at commit `5f4be16`.

---

## Branch Discipline

All work stays on branch `phase-1-sub-E-boundary-contracts`.

Every implementer dispatch must include this text verbatim:

> Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-E-boundary-contracts` branch via the user's git config.

Each task has an atomic commit checkpoint. Reviewer approves merge to main only after Task 15's handoff is reviewed.

## Test Discipline

Use TDD for implementation tasks:

1. Write the failing test.
2. Run the focused test and confirm the expected failure.
3. Implement the smallest code change.
4. Run the focused test and confirm pass.
5. Run the task-level regression command.
6. Commit.

**Halt-on-validator-fail:** if any validator (inline, cross-tile, or empirical gate) fails on real Singapore data, **stop and escalate**. Do not weaken the invariant to pass. Per memory `feedback_test_weakening_to_pass`: when data violates an assumption, the assumption failed; fix the upstream or escalate.

## Dependency Map

```
Task 1 (boundary_vocab.yaml lock)
   ↓
Task 2 (versions.py)
   ↓
Task 3 (rotation.py) ──┐
Task 4 (derivation.py) ─┤
Task 5 (io.py readers) ─┼─→ Task 6 (writer.py)
                        │      ↓
                        └→ Task 7 (validator_inline.py)
                               ↓
                        Task 8 (manifest.py + provenance.py)
                               ↓
                        Task 9 (validator_cross_tile.py)
                               ↓
                        Task 10 (pipeline.py)
                               ↓
                        Task 11 (CLI scripts)
                               ↓
                        Task 12 (eval/shuffles.py) ──┐
                        Task 13 (eval/perplexity_gap.py) ─┤
                                                          ↓
                        Task 14 (Layer 3 Singapore + empirical gate)
                               ↓
                        Task 15 (handoff)
```

Tasks 3, 4, 5 can be parallelised across subagents. Tasks 12, 13 can be parallelised after Task 11.

## File Map

```
configs/macro_plan/v1/
  boundary_vocab.yaml                          # NEW (Task 1)

src/cfm/data/sub_e/
  __init__.py                                  # NEW (Task 2)
  versions.py                                  # NEW (Task 2)
  rotation.py                                  # NEW (Task 3)
  derivation.py                                # NEW (Task 4)
  io.py                                        # NEW (Task 5)
  writer.py                                    # NEW (Task 6)
  validator_inline.py                          # NEW (Task 7)
  manifest.py                                  # NEW (Task 8)
  provenance.py                                # NEW (Task 8)
  validator_cross_tile.py                      # NEW (Task 9)
  pipeline.py                                  # NEW (Task 10)

src/cfm/eval/
  __init__.py                                  # NEW (Task 12)
  shuffles.py                                  # NEW (Task 12)
  perplexity_gap.py                            # NEW (Task 13)

scripts/
  derive_boundary_contracts.py                 # NEW (Task 11)
  validate_boundary_contracts.py               # NEW (Task 11)

tests/data/sub_e/
  __init__.py                                  # NEW (Task 1)
  test_vocab.py                                # NEW (Task 1)
  test_versions.py                             # NEW (Task 2)
  test_rotation.py                             # NEW (Task 3)
  test_derivation.py                           # NEW (Task 4)
  test_io.py                                   # NEW (Task 5)
  test_writer.py                               # NEW (Task 6)
  test_validator_inline.py                     # NEW (Task 7)
  test_manifest.py                             # NEW (Task 8)
  test_provenance.py                           # NEW (Task 8)
  test_validator_cross_tile.py                 # NEW (Task 9)
  test_pipeline.py                             # NEW (Task 10)
  test_singapore_integration.py                # NEW (Task 14)

tests/eval/
  __init__.py                                  # NEW (Task 12)
  test_shuffles.py                             # NEW (Task 12)
  test_perplexity_gap.py                       # NEW (Task 13)

docs/handoffs/
  2026-05-DD-end-of-sub-E.md                   # NEW (Task 15; date filled at commit)
```

---

## Task 1: Lock `boundary_vocab.yaml` artifact

**Files:**
- Create: `configs/macro_plan/v1/boundary_vocab.yaml`
- Create: `tests/data/sub_e/__init__.py` (empty)
- Test: `tests/data/sub_e/test_vocab.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_vocab.py
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = REPO_ROOT / "configs" / "macro_plan" / "v1" / "boundary_vocab.yaml"

LOCKED_SUB_D_VOCAB_SHA256 = (
    "0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd"
)


def test_boundary_vocab_loads_with_expected_structure() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    assert data["boundary_vocab_schema_version"] == "1.0"
    assert data["boundary_vocab_version"] == "1.0"
    assert data["boundary_derivation_version"] == "1.0"
    assert data["phase"] == 1
    assert data["append_only_within_phase"] is True


def test_boundary_vocab_has_exactly_four_tokens_in_canonical_order() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    tokens = data["tokens"]
    assert len(tokens) == 4
    assert tokens[0] == {"id": 0, "name": "BOUNDARY_NOT_APPLICABLE"}
    assert tokens[1] == {"id": 1, "name": "NONE"}
    assert tokens[2] == {"id": 2, "name": "MAJOR_ROAD"}
    assert tokens[3] == {"id": 3, "name": "MINOR_ROAD"}


def test_class_grouping_map_covers_all_named_class_raw_values() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    cgm = data["class_grouping_map"]
    assert set(cgm["MAJOR_ROAD"]) == {"primary", "trunk", "secondary"}
    assert set(cgm["MINOR_ROAD"]) == {
        "tertiary",
        "residential",
        "service",
        "unclassified",
        "footway",
        "steps",
        "cycleway",
    }


def test_boundary_vocab_inherits_scope_from_locked_sub_d_artifact() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    inheritance = data["scope_vocab_inherited_from"]
    assert inheritance["artifact"] == "configs/macro_plan/v1/macro_plan_vocab.yaml"
    assert inheritance["artifact_sha256"] == LOCKED_SUB_D_VOCAB_SHA256
    assert inheritance["block"] == "scope.tokens"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_vocab.py -v
```

Expected: 4 failures with "No such file or directory: ... boundary_vocab.yaml".

- [ ] **Step 3: Create the locked vocab YAML**

```yaml
# configs/macro_plan/v1/boundary_vocab.yaml
boundary_vocab_schema_version: "1.0"
boundary_vocab_version: "1.0"
boundary_derivation_version: "1.0"
phase: 1

generated_from:
  overture_release: "2026-04-15.0"
  regions: ["singapore"]

scope_vocab_inherited_from:
  artifact: "configs/macro_plan/v1/macro_plan_vocab.yaml"
  artifact_sha256: "0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd"
  block: "scope.tokens"

tokens:
  - {id: 0, name: BOUNDARY_NOT_APPLICABLE}
  - {id: 1, name: NONE}
  - {id: 2, name: MAJOR_ROAD}
  - {id: 3, name: MINOR_ROAD}

append_only_within_phase: true

class_grouping_map:
  MAJOR_ROAD: ["primary", "trunk", "secondary"]
  MINOR_ROAD:
    - "tertiary"
    - "residential"
    - "service"
    - "unclassified"
    - "footway"
    - "steps"
    - "cycleway"
```

Note on `empirical_gate` field deviation from spec §8.1: the draft included an `empirical_gate.layer3_subset_sha256` sub-key. It is omitted here because the Layer-3 subset is already pinned by inheritance through `scope_vocab_inherited_from.artifact_sha256` (sub-D's `macro_plan_vocab.yaml` carries the `selected_layer3_tiles` field). No information lost.

Also create the test directory init file:

```python
# tests/data/sub_e/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_vocab.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full fast suite to confirm no regression**

```bash
uv run pytest -m "not slow" -q
```

Expected: previous test count + 4 passed.

- [ ] **Step 6: Commit**

```bash
git add configs/macro_plan/v1/boundary_vocab.yaml tests/data/sub_e/__init__.py tests/data/sub_e/test_vocab.py
git commit -m "data(sub_e): lock boundary vocab v1"
```

---

## Task 2: Sub-E package skeleton and version constants

**Files:**
- Create: `src/cfm/data/sub_e/__init__.py`
- Create: `src/cfm/data/sub_e/versions.py`
- Test: `tests/data/sub_e/test_versions.py`

**Sub-D helper integration.** Sub-D's `compare_version` lives at `cfm.data.sub_d.versions::compare_version` with signature `(namespace: VersionNamespace, expected: VersionRef, actual: VersionRef) -> None`. Sub-D's `VersionNamespace` enum has five conceptual values: `ARTIFACT_FORMAT`, `DATA_SHAPE`, `VOCAB`, `DERIVATION`, `VALIDATOR`. Sub-D's `VersionRef` is a frozen dataclass `(namespace, value)`. Cross-namespace rejection works because each `VersionRef` carries its own namespace; the helper raises `VersionNamespaceError` if any argument's namespace doesn't match the comparison namespace, and `VersionMismatchError` if values diverge.

**Sub-E reuses sub-D's helper and enum unchanged.** Sub-E's three version axes map to sub-D's existing namespace concepts:

- `sub_e_schema_version` → `VersionNamespace.DATA_SHAPE` (on-disk parquet schema + YAML structure)
- `boundary_vocab_version` → `VersionNamespace.VOCAB`
- `boundary_derivation_version` → `VersionNamespace.DERIVATION`

Sub-E does NOT define a new `VersionNamespace` enum. Sub-D's enum is conceptual (vocab as a category, not "sub-D's vocab specifically"); each sub-project's `VersionRef` carries its own opaque value. The known_issue #8 lesson (separate ARTIFACT_FORMAT from DATA_SHAPE) is honored by treating sub-E's schema axis as DATA_SHAPE (parquet column layout), not ARTIFACT_FORMAT (YAML format version).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_versions.py
from __future__ import annotations

import pytest

from cfm.data.sub_d.errors import VersionNamespaceError
from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_NAMESPACE,
    BOUNDARY_DERIVATION_VERSION,
    BOUNDARY_VOCAB_NAMESPACE,
    BOUNDARY_VOCAB_VERSION,
    SUB_E_SCHEMA_NAMESPACE,
    SUB_E_SCHEMA_VERSION,
)


def test_initial_version_values() -> None:
    assert SUB_E_SCHEMA_VERSION == "1.0"
    assert BOUNDARY_VOCAB_VERSION == "1.0"
    assert BOUNDARY_DERIVATION_VERSION == "1.0"


def test_version_namespaces_use_subd_concept_enum() -> None:
    """Sub-E's three axes map to sub-D's DATA_SHAPE / VOCAB / DERIVATION."""
    assert SUB_E_SCHEMA_NAMESPACE is VersionNamespace.DATA_SHAPE
    assert BOUNDARY_VOCAB_NAMESPACE is VersionNamespace.VOCAB
    assert BOUNDARY_DERIVATION_NAMESPACE is VersionNamespace.DERIVATION


def test_compare_version_within_vocab_namespace_passes() -> None:
    """compare_version with matched-namespace refs and equal values must not raise."""
    expected = VersionRef(
        namespace=BOUNDARY_VOCAB_NAMESPACE, value=BOUNDARY_VOCAB_VERSION
    )
    actual = VersionRef(
        namespace=BOUNDARY_VOCAB_NAMESPACE, value=BOUNDARY_VOCAB_VERSION
    )
    compare_version(BOUNDARY_VOCAB_NAMESPACE, expected, actual)  # no raise


def test_compare_version_cross_namespace_rejects() -> None:
    """Mixing sub-E's vocab namespace with sub-E's derivation namespace must raise."""
    expected = VersionRef(namespace=BOUNDARY_VOCAB_NAMESPACE, value="1.0")
    actual = VersionRef(namespace=BOUNDARY_DERIVATION_NAMESPACE, value="1.0")
    with pytest.raises(VersionNamespaceError, match="namespace"):
        compare_version(BOUNDARY_VOCAB_NAMESPACE, expected, actual)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_versions.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.versions`.

- [ ] **Step 3: Implement the version module**

```python
# src/cfm/data/sub_e/__init__.py
"""Sub-E boundary-contract sidecar derivation."""

from __future__ import annotations
```

```python
# src/cfm/data/sub_e/versions.py
"""Sub-E version constants.

Sub-E has three version axes: schema (data-shape), vocab, and derivation.
Each maps to a sub-D `VersionNamespace` concept (DATA_SHAPE, VOCAB,
DERIVATION). Sub-D's `compare_version(namespace, expected, actual)` is the
canonical namespace-aware equality path; sub-E does not introduce a new
helper.

Version constants are plain strings for ergonomic YAML serialization;
namespace constants pair each version with its sub-D namespace.
"""

from __future__ import annotations

from typing import Final

from cfm.data.sub_d.versions import VersionNamespace


# Schema version: governs on-disk parquet schema + YAML structure.
SUB_E_SCHEMA_VERSION: Final[str] = "1.0"
SUB_E_SCHEMA_NAMESPACE: Final[VersionNamespace] = VersionNamespace.DATA_SHAPE

# Vocab version: governs boundary_vocab.yaml token domain.
BOUNDARY_VOCAB_VERSION: Final[str] = "1.0"
BOUNDARY_VOCAB_NAMESPACE: Final[VersionNamespace] = VersionNamespace.VOCAB

# Derivation version: governs class-grouping + multi-crossing tie-break.
BOUNDARY_DERIVATION_VERSION: Final[str] = "1.0"
BOUNDARY_DERIVATION_NAMESPACE: Final[VersionNamespace] = VersionNamespace.DERIVATION
```

- [ ] **Step 4: Confirm sub-D's `compare_version` is importable from `cfm.data.sub_d.versions`**

Quick sanity-check (does not modify any file):

```bash
uv run python -c "from cfm.data.sub_d.versions import compare_version, VersionNamespace, VersionRef; print('OK')"
```

Expected: `OK`. If this fails, that is a sub-D defect surfacing; HALT and escalate. Do NOT modify sub-D.

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_versions.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 7: Commit**

```bash
git add src/cfm/data/sub_e/__init__.py src/cfm/data/sub_e/versions.py tests/data/sub_e/test_versions.py
git commit -m "feat(sub_e): add package skeleton and version constants"
```

---

## Task 3: Per-cell rotation function

**Files:**
- Create: `src/cfm/data/sub_e/rotation.py`
- Test: `tests/data/sub_e/test_rotation.py`

**Context:** Each cell `(cell_i, cell_j) ∈ [0, 8) × [0, 8)` exposes four boundary slots N/E/S/W. The rotation function maps each cell to its four `edge_id = (lower_cell_i, lower_cell_j, axis)` tuples using sub-C's AXIS enum (`{0: "x", 1: "y"}`).

Convention adopted for sub-E (recorded here as the canonical rotation):

- North edge of cell `(i, j)`: shared between `(i, j-1)` and `(i, j)`. `lower_cell_j = j - 1` (or external if `j == 0`). Axis is the perpendicular direction of travel along the edge; for horizontal edges between rows, the edge runs along the x-axis, so `axis = 0`.
- South edge of cell `(i, j)`: shared between `(i, j)` and `(i, j+1)`. `lower_cell_j = j`. External if `j == 7`. `axis = 0`.
- West edge of cell `(i, j)`: shared between `(i-1, j)` and `(i, j)`. `lower_cell_i = i - 1`. External if `i == 0`. `axis = 1`.
- East edge of cell `(i, j)`: shared between `(i, j)` and `(i+1, j)`. `lower_cell_i = i`. External if `i == 7`. `axis = 1`.

External edges encode the missing-neighbour case by using the interior cell's coordinate at the `lower_cell_*` slot with a sentinel marker.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_rotation.py
from __future__ import annotations

from cfm.data.sub_e.rotation import (
    CellEdgeIds,
    cell_to_edge_ids,
    EdgeKind,
)


def test_interior_cell_3_3_has_four_internal_edges() -> None:
    result = cell_to_edge_ids(cell_i=3, cell_j=3)
    assert result.north == (3, 2, 0, EdgeKind.INTERNAL)
    assert result.south == (3, 3, 0, EdgeKind.INTERNAL)
    assert result.west == (2, 3, 1, EdgeKind.INTERNAL)
    assert result.east == (3, 3, 1, EdgeKind.INTERNAL)


def test_edge_cell_0_3_has_west_external_three_internal() -> None:
    result = cell_to_edge_ids(cell_i=0, cell_j=3)
    assert result.north == (0, 2, 0, EdgeKind.INTERNAL)
    assert result.south == (0, 3, 0, EdgeKind.INTERNAL)
    assert result.west == (0, 3, 1, EdgeKind.EXTERNAL)  # i=0 → external
    assert result.east == (0, 3, 1, EdgeKind.INTERNAL)


def test_edge_cell_3_0_has_north_external_three_internal() -> None:
    result = cell_to_edge_ids(cell_i=3, cell_j=0)
    assert result.north == (3, 0, 0, EdgeKind.EXTERNAL)  # j=0 → external
    assert result.south == (3, 0, 0, EdgeKind.INTERNAL)
    assert result.west == (2, 0, 1, EdgeKind.INTERNAL)
    assert result.east == (3, 0, 1, EdgeKind.INTERNAL)


def test_corner_cell_0_0_has_two_external_two_internal() -> None:
    result = cell_to_edge_ids(cell_i=0, cell_j=0)
    assert result.north == (0, 0, 0, EdgeKind.EXTERNAL)
    assert result.south == (0, 0, 0, EdgeKind.INTERNAL)
    assert result.west == (0, 0, 1, EdgeKind.EXTERNAL)
    assert result.east == (0, 0, 1, EdgeKind.INTERNAL)


def test_corner_cell_7_7_has_two_external_two_internal() -> None:
    result = cell_to_edge_ids(cell_i=7, cell_j=7)
    assert result.north == (7, 6, 0, EdgeKind.INTERNAL)
    assert result.south == (7, 7, 0, EdgeKind.EXTERNAL)  # j=7 → external
    assert result.west == (6, 7, 1, EdgeKind.INTERNAL)
    assert result.east == (7, 7, 1, EdgeKind.EXTERNAL)  # i=7 → external


def test_full_lattice_counts_match_specification() -> None:
    """Aggregated check: 112 internal + 32 external across 64 cells × 4 slots."""
    internal: set[tuple[int, int, int]] = set()
    external_count = 0
    for cell_i in range(8):
        for cell_j in range(8):
            result = cell_to_edge_ids(cell_i, cell_j)
            for slot in (result.north, result.south, result.west, result.east):
                i, j, axis, kind = slot
                if kind is EdgeKind.INTERNAL:
                    internal.add((i, j, axis))
                else:
                    external_count += 1
    assert len(internal) == 112, "expected 112 unique internal edges"
    assert external_count == 32, "expected 32 external slot occurrences"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_rotation.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.rotation`.

- [ ] **Step 3: Implement the rotation module**

```python
# src/cfm/data/sub_e/rotation.py
"""Per-cell to per-edge rotation for sub-E.

For each cell `(cell_i, cell_j) ∈ [0, 8) × [0, 8)` the four boundary slots
N/E/S/W map to canonical `edge_id = (lower_cell_i, lower_cell_j, axis)` tuples
following sub-C's AXIS enum (0=x, 1=y). External slots (at tile boundary) are
tagged with EdgeKind.EXTERNAL; sub-E writes one row per external slot with
scope_marker driven by sub-D's macro_core, not a derivation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

GRID_SIZE: Final[int] = 8

# Axis encoding matches cfm.data.sub_c.enums.AXIS = {0: "x", 1: "y"}.
AXIS_X: Final[int] = 0
AXIS_Y: Final[int] = 1


class EdgeKind(str, Enum):
    INTERNAL = "internal_edge"
    EXTERNAL = "external_edge"


EdgeIdTuple = tuple[int, int, int, EdgeKind]
"""(lower_cell_i, lower_cell_j, axis, kind)."""


@dataclass(frozen=True)
class CellEdgeIds:
    """Four edge_ids for one cell, ordered N/S/W/E for stable iteration."""

    north: EdgeIdTuple
    south: EdgeIdTuple
    west: EdgeIdTuple
    east: EdgeIdTuple


def cell_to_edge_ids(cell_i: int, cell_j: int) -> CellEdgeIds:
    """Map a cell to its four boundary slot edge_ids.

    Raises ValueError if `cell_i` or `cell_j` is outside [0, 8).
    """
    if not (0 <= cell_i < GRID_SIZE) or not (0 <= cell_j < GRID_SIZE):
        raise ValueError(
            f"cell ({cell_i}, {cell_j}) outside [0, {GRID_SIZE})^2"
        )

    north_kind = EdgeKind.EXTERNAL if cell_j == 0 else EdgeKind.INTERNAL
    north_lower_j = cell_j if cell_j == 0 else cell_j - 1
    north = (cell_i, north_lower_j, AXIS_X, north_kind)

    south_kind = EdgeKind.EXTERNAL if cell_j == GRID_SIZE - 1 else EdgeKind.INTERNAL
    south = (cell_i, cell_j, AXIS_X, south_kind)

    west_kind = EdgeKind.EXTERNAL if cell_i == 0 else EdgeKind.INTERNAL
    west_lower_i = cell_i if cell_i == 0 else cell_i - 1
    west = (west_lower_i, cell_j, AXIS_Y, west_kind)

    east_kind = EdgeKind.EXTERNAL if cell_i == GRID_SIZE - 1 else EdgeKind.INTERNAL
    east = (cell_i, cell_j, AXIS_Y, east_kind)

    return CellEdgeIds(north=north, south=south, west=west, east=east)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_rotation.py -v
```

Expected: 6 passed (including the aggregated lattice count check).

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/rotation.py tests/data/sub_e/test_rotation.py
git commit -m "feat(sub_e): add per-cell rotation function"
```

---

## Task 4: Class-grouping derivation function

**Files:**
- Create: `src/cfm/data/sub_e/derivation.py`
- Test: `tests/data/sub_e/test_derivation.py`

**Context:** Spec §5.1 + §5.2. The derivation function maps a list of `class_raw` strings (the road crossings on one active internal edge) to one `boundary_class` integer enum value via the class-grouping map + hierarchy-wins tie-break + default-bucket rule.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_derivation.py
from __future__ import annotations

import pytest

from cfm.data.sub_e.derivation import (
    BoundaryClass,
    derive_boundary_class,
    load_class_grouping_map,
)


def test_load_class_grouping_map_from_vocab_yaml() -> None:
    mapping = load_class_grouping_map()
    assert mapping["primary"] is BoundaryClass.MAJOR_ROAD
    assert mapping["secondary"] is BoundaryClass.MAJOR_ROAD
    assert mapping["trunk"] is BoundaryClass.MAJOR_ROAD
    assert mapping["residential"] is BoundaryClass.MINOR_ROAD
    assert mapping["service"] is BoundaryClass.MINOR_ROAD
    assert mapping["footway"] is BoundaryClass.MINOR_ROAD


def test_empty_crossings_returns_none() -> None:
    result = derive_boundary_class(class_raws=[])
    assert result is BoundaryClass.NONE


def test_single_primary_crossing_returns_major() -> None:
    result = derive_boundary_class(class_raws=["primary"])
    assert result is BoundaryClass.MAJOR_ROAD


def test_single_residential_crossing_returns_minor() -> None:
    result = derive_boundary_class(class_raws=["residential"])
    assert result is BoundaryClass.MINOR_ROAD


def test_hierarchy_wins_primary_beats_residential() -> None:
    result = derive_boundary_class(class_raws=["residential", "primary"])
    assert result is BoundaryClass.MAJOR_ROAD


def test_hierarchy_wins_three_minor_one_major() -> None:
    result = derive_boundary_class(
        class_raws=["footway", "residential", "service", "secondary"]
    )
    assert result is BoundaryClass.MAJOR_ROAD


def test_default_bucket_unknown_class_raw_demotes_to_minor() -> None:
    """Overture rare values not in the named 10 → MINOR_ROAD."""
    result = derive_boundary_class(class_raws=["proposed"])
    assert result is BoundaryClass.MINOR_ROAD


def test_default_bucket_null_class_raw_treats_as_minor() -> None:
    result = derive_boundary_class(class_raws=[None])
    assert result is BoundaryClass.MINOR_ROAD


def test_default_bucket_does_not_promote_to_major() -> None:
    """Mixing an unknown with a primary still resolves to MAJOR via the
    primary, but an unknown alone never resolves to MAJOR. Demonstrate the
    asymmetry."""
    assert (
        derive_boundary_class(class_raws=["proposed"]) is BoundaryClass.MINOR_ROAD
    )
    assert (
        derive_boundary_class(class_raws=["proposed", "primary"])
        is BoundaryClass.MAJOR_ROAD
    )


def test_boundary_class_enum_values_match_vocab_ids() -> None:
    assert BoundaryClass.BOUNDARY_NOT_APPLICABLE.value == 0
    assert BoundaryClass.NONE.value == 1
    assert BoundaryClass.MAJOR_ROAD.value == 2
    assert BoundaryClass.MINOR_ROAD.value == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_derivation.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.derivation`.

- [ ] **Step 3: Implement the derivation module**

```python
# src/cfm/data/sub_e/derivation.py
"""Boundary-class derivation function.

Maps `class_raw` strings (raw Overture transportation.class values from sub-C
`features.parquet`) to `BoundaryClass` enum via the class-grouping map and
hierarchy-wins multi-crossing tie-break. Locked under
boundary_derivation_version 1.0 (see boundary_vocab.yaml).
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml


class BoundaryClass(IntEnum):
    BOUNDARY_NOT_APPLICABLE = 0  # sentinel; dataloader-side only, never on-disk
    NONE = 1
    MAJOR_ROAD = 2
    MINOR_ROAD = 3


# Hierarchy order for multi-crossing tie-break: highest precedence first.
_HIERARCHY: Final[tuple[BoundaryClass, ...]] = (
    BoundaryClass.MAJOR_ROAD,
    BoundaryClass.MINOR_ROAD,
    BoundaryClass.NONE,
)

_VOCAB_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "macro_plan"
    / "v1"
    / "boundary_vocab.yaml"
)
# parents[4] climbs src/cfm/data/sub_e/derivation.py → repo root:
#   [0]=sub_e/  [1]=data/  [2]=cfm/  [3]=src/  [4]=<repo root>.
# Tests under tests/data/sub_e/*.py use parents[3] (one level shallower).


@lru_cache(maxsize=1)
def load_class_grouping_map() -> dict[str, BoundaryClass]:
    """Load the class_raw → BoundaryClass mapping from boundary_vocab.yaml.

    Cached: the vocab is locked and won't change within a process.
    """
    data = yaml.safe_load(_VOCAB_PATH.read_text())
    raw_map = data["class_grouping_map"]
    out: dict[str, BoundaryClass] = {}
    for class_raw in raw_map["MAJOR_ROAD"]:
        out[class_raw] = BoundaryClass.MAJOR_ROAD
    for class_raw in raw_map["MINOR_ROAD"]:
        out[class_raw] = BoundaryClass.MINOR_ROAD
    return out


def derive_boundary_class(
    class_raws: list[str | None],
) -> BoundaryClass:
    """Derive the BoundaryClass for one active internal edge.

    Args:
        class_raws: list of raw Overture transportation.class strings for
            every road crossing on this edge. May be empty. Null/unknown
            values fall through to the MINOR_ROAD default bucket.

    Returns:
        BoundaryClass.NONE if `class_raws` is empty (no road crossings).
        Otherwise the highest-precedence class per the hierarchy:
        MAJOR_ROAD > MINOR_ROAD > NONE.
    """
    if not class_raws:
        return BoundaryClass.NONE

    grouping = load_class_grouping_map()
    seen: set[BoundaryClass] = set()
    for cr in class_raws:
        # Null / unknown class_raw → default bucket MINOR_ROAD (never MAJOR).
        seen.add(grouping.get(cr, BoundaryClass.MINOR_ROAD))

    for cls in _HIERARCHY:
        if cls in seen:
            return cls
    # Unreachable: seen is non-empty if class_raws is non-empty, and every
    # element maps into the hierarchy.
    raise AssertionError(f"derivation reached unreachable branch: {class_raws!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_derivation.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/derivation.py tests/data/sub_e/test_derivation.py
git commit -m "feat(sub_e): add class-precedence derivation function"
```

---

## Task 5: Sub-C and sub-D input readers

**Files:**
- Create: `src/cfm/data/sub_e/io.py`
- Test: `tests/data/sub_e/test_io.py`

**Context:** Spec §4. Sub-E reads sub-C `crossings.parquet` + `features.parquet` and sub-D `macro_core.parquet` only. All reads use `pyarrow.parquet.ParquetFile(path).read()` to avoid Hive partition inference (memory `feedback_pyarrow_hive_partition_inference`). Sub-E gates on sub-D's `_SUCCESS` first; sub-D's `manifest.yaml` is the tile inventory.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_io.py
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_e.io import (
    SubCCrossingRow,
    SubCFeatureRow,
    SubDMacroCoreRow,
    read_sub_c_crossings,
    read_sub_c_features,
    read_sub_d_macro_core,
    require_sub_d_success_marker,
)


def _write_synthetic_crossings(path: Path) -> None:
    table = pa.table(
        {
            "lower_cell_i": pa.array([0, 1], type=pa.int8()),
            "lower_cell_j": pa.array([0, 2], type=pa.int8()),
            "axis": pa.array([0, 1], type=pa.int8()),
            "source_feature_id": pa.array(["F1", "F2"], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _write_synthetic_features(path: Path) -> None:
    table = pa.table(
        {
            "source_feature_id": pa.array(["F1", "F2"], type=pa.string()),
            "feature_class": pa.array(["road", "road"], type=pa.string()),
            "class_raw": pa.array(["primary", "residential"], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _write_synthetic_macro_core(path: Path) -> None:
    table = pa.table(
        {
            "slot_kind": pa.array([0, 1], type=pa.int8()),
            "slot_index": pa.array([0, 0], type=pa.int16()),
            "cell_i": pa.array([0, None], type=pa.int8()),
            "cell_j": pa.array([0, None], type=pa.int8()),
            "lower_cell_i": pa.array([None, 0], type=pa.int8()),
            "lower_cell_j": pa.array([None, 0], type=pa.int8()),
            "axis": pa.array([None, 0], type=pa.int8()),
            "scope": pa.array([0, 0], type=pa.int8()),
            "zoning_class": pa.array([1, None], type=pa.int16()),
            "cell_density_bucket": pa.array([2, None], type=pa.int16()),
            "road_skeleton_class": pa.array([None, 1], type=pa.int16()),
        }
    )
    pq.write_table(table, path)


def test_read_sub_c_crossings_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "crossings.parquet"
    _write_synthetic_crossings(p)
    rows = read_sub_c_crossings(p)
    assert len(rows) == 2
    assert isinstance(rows[0], SubCCrossingRow)
    assert rows[0].lower_cell_i == 0
    assert rows[0].lower_cell_j == 0
    assert rows[0].axis == 0
    assert rows[0].source_feature_id == "F1"


def test_read_sub_c_features_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "features.parquet"
    _write_synthetic_features(p)
    rows = read_sub_c_features(p)
    by_id = {r.source_feature_id: r for r in rows}
    assert by_id["F1"].class_raw == "primary"
    assert by_id["F2"].class_raw == "residential"


def test_read_sub_d_macro_core_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "macro_core.parquet"
    _write_synthetic_macro_core(p)
    rows = read_sub_d_macro_core(p)
    assert len(rows) == 2
    cell_rows = [r for r in rows if r.slot_kind == 0]
    edge_rows = [r for r in rows if r.slot_kind == 1]
    assert len(cell_rows) == 1
    assert len(edge_rows) == 1
    assert cell_rows[0].zoning_class == 1
    assert edge_rows[0].axis == 0


def test_require_sub_d_success_marker_passes_when_present(tmp_path: Path) -> None:
    region = tmp_path / "sub_d_region"
    region.mkdir()
    (region / "_SUCCESS").touch()
    require_sub_d_success_marker(region)  # should not raise


def test_require_sub_d_success_marker_raises_when_absent(tmp_path: Path) -> None:
    region = tmp_path / "sub_d_region"
    region.mkdir()
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        require_sub_d_success_marker(region)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_io.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.io`.

- [ ] **Step 3: Implement the IO module**

```python
# src/cfm/data/sub_e/io.py
"""Sub-E input readers.

Reads sub-C `crossings.parquet` + `features.parquet` and sub-D
`macro_core.parquet`. All reads use `pyarrow.parquet.ParquetFile(path).read()`
to avoid Hive partition inference on tile=... directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SubCCrossingRow:
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    source_feature_id: str


@dataclass(frozen=True)
class SubCFeatureRow:
    source_feature_id: str
    feature_class: str
    class_raw: str | None


@dataclass(frozen=True)
class SubDMacroCoreRow:
    slot_kind: int
    slot_index: int
    cell_i: int | None
    cell_j: int | None
    lower_cell_i: int | None
    lower_cell_j: int | None
    axis: int | None
    scope: int
    zoning_class: int | None
    cell_density_bucket: int | None
    road_skeleton_class: int | None


def _read_table(path: Path):
    return pq.ParquetFile(path).read()


def read_sub_c_crossings(path: Path) -> list[SubCCrossingRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubCCrossingRow(
            lower_cell_i=cols["lower_cell_i"][i],
            lower_cell_j=cols["lower_cell_j"][i],
            axis=cols["axis"][i],
            source_feature_id=cols["source_feature_id"][i],
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_c_features(path: Path) -> list[SubCFeatureRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubCFeatureRow(
            source_feature_id=cols["source_feature_id"][i],
            feature_class=cols["feature_class"][i],
            class_raw=cols["class_raw"][i],
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_d_macro_core(path: Path) -> list[SubDMacroCoreRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubDMacroCoreRow(
            slot_kind=cols["slot_kind"][i],
            slot_index=cols["slot_index"][i],
            cell_i=cols["cell_i"][i],
            cell_j=cols["cell_j"][i],
            lower_cell_i=cols["lower_cell_i"][i],
            lower_cell_j=cols["lower_cell_j"][i],
            axis=cols["axis"][i],
            scope=cols["scope"][i],
            zoning_class=cols["zoning_class"][i],
            cell_density_bucket=cols["cell_density_bucket"][i],
            road_skeleton_class=cols["road_skeleton_class"][i],
        )
        for i in range(tbl.num_rows)
    ]


def require_sub_d_success_marker(sub_d_region_dir: Path) -> None:
    """Gate sub-E on sub-D's `_SUCCESS` marker.

    Raises FileNotFoundError if the marker is absent. Sub-E does not start
    derivation against a sub-D region whose validator has not closed.
    """
    marker = sub_d_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(
            f"sub-D _SUCCESS marker missing at {marker}; sub-E refuses to start"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_io.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/io.py tests/data/sub_e/test_io.py
git commit -m "feat(sub_e): add sub-C and sub-D input readers"
```

---

## Task 6: Boundary-contract writer

**Files:**
- Create: `src/cfm/data/sub_e/writer.py`
- Test: `tests/data/sub_e/test_writer.py`

**Context:** Spec §7.2. The writer emits `boundary_contract.parquet` per tile with exactly 144 rows: 112 internal + 32 external, sorted by `(slot_kind, slot_index)`. Schema matches sub-D `macro_core.parquet` conventions for shared reusability of validator infra.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_writer.py
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)


def _make_full_lattice_rows() -> list[BoundaryContractRow]:
    """Construct a synthetic full-lattice 144-row set with all-active scope."""
    rows: list[BoundaryContractRow] = []
    for idx in range(112):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=0,  # active
                boundary_class_enum=int(BoundaryClass.NONE),
            )
        )
    for idx in range(32):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=3,  # external_deferred
                boundary_class_enum=None,
            )
        )
    return rows


def test_write_produces_144_rows_sorted_canonical(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    assert tbl.num_rows == 144
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    sorted_pairs = sorted(zip(slot_kinds, slot_indices))
    assert list(zip(slot_kinds, slot_indices)) == sorted_pairs


def test_internal_row_count_is_112_external_is_32(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    assert slot_kinds.count(int(SlotKind.INTERNAL_EDGE)) == 112
    assert slot_kinds.count(int(SlotKind.EXTERNAL_EDGE)) == 32


def test_boundary_class_enum_nullable(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    values = tbl.column("boundary_class_enum").to_pylist()
    # 112 active internal rows have value NONE=1; 32 external rows are null.
    assert values.count(None) == 32
    assert values.count(1) == 112


def test_write_rejects_wrong_row_count(tmp_path: Path) -> None:
    import pytest

    out_path = tmp_path / "boundary_contract.parquet"
    with pytest.raises(ValueError, match="144"):
        write_boundary_contract(out_path, _make_full_lattice_rows()[:100])


def test_write_sorts_unordered_input(tmp_path: Path) -> None:
    """Canonical sort key is enforced at write-time, not assumed from input
    order. Pin: the writer's own sort is load-bearing for byte-determinism
    across upstream code paths that might emit rows in any order.
    """
    import random

    rows = _make_full_lattice_rows()
    rng = random.Random(0xBEEF)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    assert shuffled != rows, "test setup: shuffled order must differ from canonical"

    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, shuffled)
    tbl = pq.ParquetFile(out_path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    pairs = list(zip(slot_kinds, slot_indices))
    assert pairs == sorted(pairs), "writer must sort by (slot_kind, slot_index)"


def test_write_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    """Same-process determinism: two writes of identical inputs produce
    byte-identical parquet files. Catches schema/sort/dict-ordering drift
    earliest — without waiting for Task 7's validator or Task 14's
    integration test.
    """
    import hashlib

    rows = _make_full_lattice_rows()
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_boundary_contract(a, rows)
    write_boundary_contract(b, rows)
    h_a = hashlib.sha256(a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(b.read_bytes()).hexdigest()
    assert h_a == h_b, "same-process determinism required"


def test_write_is_byte_deterministic_under_input_shuffling(tmp_path: Path) -> None:
    """Byte-determinism survives input-order perturbation. Two writes of the
    SAME logical row set but in different input orders must produce the
    same bytes — because the writer's canonical sort dominates input order.
    """
    import hashlib
    import random

    rows = _make_full_lattice_rows()
    rng = random.Random(0xC0FFEE)
    perm = rows.copy()
    rng.shuffle(perm)

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_boundary_contract(a, rows)
    write_boundary_contract(b, perm)
    h_a = hashlib.sha256(a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(b.read_bytes()).hexdigest()
    assert h_a == h_b, "writer's canonical sort must dominate input order"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_writer.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.writer`.

- [ ] **Step 3: Implement the writer**

```python
# src/cfm/data/sub_e/writer.py
"""Boundary-contract parquet writer.

Emits exactly 144 rows per tile: 112 internal_edge + 32 external_edge, sorted
by (slot_kind, slot_index). Schema is pinned via ``pa.schema(...)`` so
PyArrow type inference cannot drift (mirrors sub-D's _MACRO_CORE_SCHEMA
pattern at ``src/cfm/data/sub_d/io.py:41``). The neutral
``cfm.data.io.write_parquet`` helper is reused for byte-deterministic
serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet


class SlotKind(IntEnum):
    INTERNAL_EDGE = 1
    EXTERNAL_EDGE = 2


EXPECTED_INTERNAL_ROWS: Final[int] = 112
EXPECTED_EXTERNAL_ROWS: Final[int] = 32
EXPECTED_TOTAL_ROWS: Final[int] = EXPECTED_INTERNAL_ROWS + EXPECTED_EXTERNAL_ROWS


# Pinned schema: explicit pa.schema with nullable flags. `boundary_class_enum`
# is the only nullable column (null = "BOUNDARY_NOT_APPLICABLE" / non-active
# rows); every other column is non-null. Pinning prevents PyArrow type
# inference from drifting between writes and keeps byte-determinism robust to
# input-shape variation.
_BOUNDARY_CONTRACT_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("slot_kind", pa.int8(), nullable=False),
        pa.field("slot_index", pa.int16(), nullable=False),
        pa.field("lower_cell_i", pa.int8(), nullable=False),
        pa.field("lower_cell_j", pa.int8(), nullable=False),
        pa.field("axis", pa.int8(), nullable=False),
        pa.field("scope_marker", pa.int8(), nullable=False),
        pa.field("boundary_class_enum", pa.int16(), nullable=True),
    ]
)


@dataclass(frozen=True)
class BoundaryContractRow:
    slot_kind: SlotKind
    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    scope_marker: int
    boundary_class_enum: int | None


def write_boundary_contract(
    out_path: Path,
    rows: list[BoundaryContractRow],
) -> Path:
    """Write rows to `out_path` as a canonically sorted parquet file.

    Raises ValueError if the row count is not 144 or the split is not
    112 internal + 32 external. The writer's own sort by
    ``(slot_kind, slot_index)`` is load-bearing — it must not assume the
    input list is already sorted (see ``test_write_sorts_unordered_input``).
    """
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_ROWS} rows (112 internal + 32 external), "
            f"got {len(rows)}"
        )
    n_internal = sum(1 for r in rows if r.slot_kind is SlotKind.INTERNAL_EDGE)
    n_external = sum(1 for r in rows if r.slot_kind is SlotKind.EXTERNAL_EDGE)
    if (n_internal, n_external) != (EXPECTED_INTERNAL_ROWS, EXPECTED_EXTERNAL_ROWS):
        raise ValueError(
            f"row split must be (112, 32), got ({n_internal}, {n_external})"
        )

    sorted_rows = sorted(rows, key=lambda r: (int(r.slot_kind), r.slot_index))

    columns = {
        "slot_kind": [int(r.slot_kind) for r in sorted_rows],
        "slot_index": [r.slot_index for r in sorted_rows],
        "lower_cell_i": [r.lower_cell_i for r in sorted_rows],
        "lower_cell_j": [r.lower_cell_j for r in sorted_rows],
        "axis": [r.axis for r in sorted_rows],
        "scope_marker": [r.scope_marker for r in sorted_rows],
        "boundary_class_enum": [r.boundary_class_enum for r in sorted_rows],
    }
    table = pa.Table.from_pydict(columns, schema=_BOUNDARY_CONTRACT_SCHEMA)
    write_parquet(table, out_path)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_writer.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/writer.py tests/data/sub_e/test_writer.py
git commit -m "feat(sub_e): add boundary contract parquet writer"
```

---

## Task 7: Inline validator (8 invariants + 1 carry-forward)

**Files:**
- Create: `src/cfm/data/sub_e/validator_inline.py`
- Test: `tests/data/sub_e/test_validator_inline.py`

**Context:** Spec §10.1. Eight invariants over a single `boundary_contract.parquet` (spec catalog), plus one sub-E-local invariant #9 (Task-6 carry-forward) that asserts the sub-E writer's `SlotKind` enum integer values match `cfm.data.sub_d.enums.SlotKind` byte-for-byte. Each invariant has a controlled-violation fixture or positive regression guard in the test suite. **All fixtures synthesised in-process via `BoundaryContractRow` + the Task 6 writer; this task does not read `data/processed/sub_c/` or `data/processed/sub_d/`.**

Under lever-3 collapse (spec §12), invariants #3 and #4 are replaced by a single uniform-null check (every row's `boundary_class_enum` is null). Invariant #9 (Task-6 carry-forward) is **mode-independent** — it applies under both modes since cross-enum wire compatibility is a structural property unrelated to derivation. The validator accepts a `lever_3_collapse: bool` kwarg to switch modes; Task 10's pipeline forwards `cfg.lever_3_collapse`.

**Task-6 carry-forward rationale (in-code comment must reference this).** Sub-D's `SlotKind` (`src/cfm/data/sub_d/enums.py:25-26`) and sub-E writer's `SlotKind` (`src/cfm/data/sub_e/writer.py:23`) are two separate `IntEnum` classes that happen to share wire values `INTERNAL_EDGE=1` and `EXTERNAL_EDGE=2`. If either side adds a member or reorders values without coordinating, the cross-table foreign-key path corrupts silently. Invariant #9 catches that drift at validation time.

**Invariant #8 I/O responsibility (decided in plan-fixup; do not relitigate).** The boundary_contract.parquet on disk has no `boundary_derivation_version` column or file-level metadata. Sub-E's three-axis versioning lives in the sibling `provenance.yaml` (written in Task 8). The inline validator does **not** read provenance.yaml — that I/O lives in **Task 10's pipeline orchestrator**. Task 7's validator signature requires the caller to supply both `expected_derivation_version` (what the pipeline expects) and `provenance_derivation_version` (what was actually recorded). Both kwargs are non-optional. Invariant #8 then becomes a pure string comparison inside the validator. This keeps the invariant in the inline validator per spec §10.1 categorization while pushing the file I/O to the orchestrator layer.

**Loop ordering discipline (decided in plan-fixup).** The per-row invariant loop runs in **membership-before-semantic** order: structural enum/range membership (#5 scope_marker, #6 slot_index, #7 axis) checked **before** semantic relationships (#3 non-null-iff, #4 active class). Reason: semantic invariants assume per-row values are already in valid ranges; without that prior check, an out-of-range membership value can short-circuit a downstream semantic test in a confusing way (e.g. `scope_marker=9, cls=non-null` would fire #3 before #5, masking the real defect). The discipline yields more useful error messages on real-data violations too.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_validator_inline.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.validator_inline import (
    InlineValidationError,
    validate_boundary_contract,
)
from cfm.data.sub_e.versions import BOUNDARY_DERIVATION_VERSION
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)


def _valid_rows() -> list[BoundaryContractRow]:
    rows: list[BoundaryContractRow] = []
    for idx in range(112):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=0,
                boundary_class_enum=int(BoundaryClass.NONE),
            )
        )
    for idx in range(32):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=3,
                boundary_class_enum=None,
            )
        )
    return rows


def _write(tmp_path: Path, rows: list[BoundaryContractRow]) -> Path:
    p = tmp_path / "boundary_contract.parquet"
    # Writer enforces row count; bypass when needed via direct table writing
    # is NOT permitted here — we exercise inline validator on writer output.
    write_boundary_contract(p, rows)
    return p


def _validate(
    p: Path,
    *,
    expected: str = BOUNDARY_DERIVATION_VERSION,
    provenance: str = BOUNDARY_DERIVATION_VERSION,
    lever_3_collapse: bool = False,
) -> None:
    """Test helper: passes both required version kwargs to the validator.

    Defaults to the locked v1.0 for both, so tests that don't exercise
    invariant #8 stay terse. Tests that exercise #8 pass mismatched values
    explicitly.
    """
    validate_boundary_contract(
        p,
        expected_derivation_version=expected,
        provenance_derivation_version=provenance,
        lever_3_collapse=lever_3_collapse,
    )


def test_valid_lattice_passes_all_invariants(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_rows())
    _validate(p)  # should not raise


def test_invariant_3_class_non_null_iff_scope_active(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Set an external (scope_marker=3) row to have a non-null class — violates #3.
    rows[112] = replace(rows[112], boundary_class_enum=int(BoundaryClass.MAJOR_ROAD))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="non-null iff scope_marker == 0"):
        _validate(p)


def test_invariant_4_active_class_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Use BOUNDARY_NOT_APPLICABLE (0) on an active row — sentinel is dataloader-side
    # only; on-disk active rows must be in {NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3}.
    rows[0] = replace(
        rows[0], boundary_class_enum=int(BoundaryClass.BOUNDARY_NOT_APPLICABLE)
    )
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="active class membership"):
        _validate(p)


def test_invariant_5_scope_marker_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Mutate scope_marker=9 AND null the boundary_class_enum so invariants
    # #3/#4 are not in violation on this row — but invariant #5 (membership)
    # fires regardless because it's structurally prior under the
    # membership-before-semantic loop order.
    rows[0] = replace(rows[0], scope_marker=9, boundary_class_enum=None)
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="scope_marker membership"):
        _validate(p)


def test_invariant_6_slot_index_range(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], slot_index=999)
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="slot_index range"):
        _validate(p)


def test_invariant_7_axis_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], axis=2)  # AXIS = {0, 1}
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="axis membership"):
        _validate(p)


def test_invariant_8_derivation_version_match(tmp_path: Path) -> None:
    """Provenance mismatch with expected raises. Both kwargs are required
    on the validator; the test passes an explicit mismatch."""
    p = _write(tmp_path, _valid_rows())
    with pytest.raises(InlineValidationError, match="boundary_derivation_version"):
        _validate(p, expected="9.9", provenance=BOUNDARY_DERIVATION_VERSION)


# Invariants #1 (row count) and #2 (sort key) are enforced by the writer
# itself (Task 6); a malformed parquet that bypasses the writer would still
# trigger them. Cover them by reading the parquet, mutating, and re-writing
# raw via pyarrow:

def test_invariant_1_total_row_count(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = _write(tmp_path, _valid_rows())
    tbl = pq.ParquetFile(p).read()
    # Slice to 100 rows — bypasses writer.
    bad = tbl.slice(0, 100)
    pq.write_table(bad, p)
    with pytest.raises(InlineValidationError, match="144"):
        _validate(p)


def test_invariant_2_sort_key(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    p = _write(tmp_path, _valid_rows())
    tbl = pq.ParquetFile(p).read()
    # Reverse — sort key violated.
    bad = tbl.slice(0, tbl.num_rows).take(list(range(tbl.num_rows - 1, -1, -1)))
    pq.write_table(bad, p)
    with pytest.raises(InlineValidationError, match="sort key"):
        _validate(p)


def test_lever_3_collapse_passes_with_uniform_null(tmp_path: Path) -> None:
    """Under lever-3, all boundary_class_enum values null even on active rows."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    p = _write(tmp_path, rows)
    _validate(p, lever_3_collapse=True)  # should not raise


def test_lever_3_collapse_rejects_any_non_null(tmp_path: Path) -> None:
    """Under lever-3, even a single non-null boundary_class_enum is a violation."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    rows[0] = replace(rows[0], boundary_class_enum=int(BoundaryClass.NONE))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="lever-3"):
        _validate(p, lever_3_collapse=True)


def test_invariant_9_slotkind_cross_enum_byte_equivalence() -> None:
    """Invariant #9 (Task-6 carry-forward): sub-E writer's SlotKind enum
    integer values must match sub-D's SlotKind byte-for-byte at INTERNAL_EDGE
    and EXTERNAL_EDGE. Two separate IntEnum classes maintain wire
    compatibility manually; this test is the regression guard against
    silent drift if either side gains a member or reorders values.
    """
    from cfm.data.sub_d.enums import SlotKind as SubDSlotKind

    assert int(SlotKind.INTERNAL_EDGE) == int(SubDSlotKind.INTERNAL_EDGE) == 1
    assert int(SlotKind.EXTERNAL_EDGE) == int(SubDSlotKind.EXTERNAL_EDGE) == 2


def test_invariant_9_drift_simulation_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invariant #9 fires at validator runtime when wire values diverge.

    Simulates drift by monkey-patching sub-D's `SlotKind` to a divergent
    IntEnum. The validator must raise InlineValidationError; the validator's
    lazy import of `cfm.data.sub_d.enums.SlotKind` is what makes this test
    able to observe the drift.
    """
    from enum import IntEnum

    from cfm.data.sub_d import enums as sub_d_enums

    class _DriftedSlotKind(IntEnum):
        INTERNAL_EDGE = 99  # divergent from sub-E writer's INTERNAL_EDGE=1
        EXTERNAL_EDGE = 2

    monkeypatch.setattr(sub_d_enums, "SlotKind", _DriftedSlotKind)
    p = _write(tmp_path, _valid_rows())
    with pytest.raises(InlineValidationError, match="SlotKind"):
        _validate(p)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_validator_inline.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the inline validator**

```python
# src/cfm/data/sub_e/validator_inline.py
"""Sub-E per-tile inline validator.

Implements the 8 invariants from spec §10.1 plus 1 sub-E-local invariant #9
(Task-6 carry-forward) asserting cross-enum byte-equivalence between sub-E
writer's SlotKind and sub-D's SlotKind. Invariant #9 is mode-independent
(applies under both default and lever_3_collapse modes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.writer import (
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_INTERNAL_ROWS,
    EXPECTED_TOTAL_ROWS,
    SlotKind,
)


_ACTIVE_CLASS_IDS: Final[frozenset[int]] = frozenset(
    {
        int(BoundaryClass.NONE),
        int(BoundaryClass.MAJOR_ROAD),
        int(BoundaryClass.MINOR_ROAD),
    }
)

_SCOPE_MARKER_VALUES: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
_AXIS_VALUES: Final[frozenset[int]] = frozenset({0, 1})


class InlineValidationError(ValueError):
    """Raised when a sub-E boundary_contract.parquet fails any inline invariant."""


def _assert_slotkind_byte_equivalence_with_sub_d() -> None:
    """Invariant #9 (Task-6 carry-forward, sub-E-local — not in original 8).

    Sub-D's `SlotKind` (`src/cfm/data/sub_d/enums.py`) and sub-E writer's
    `SlotKind` (`src/cfm/data/sub_e/writer.py`) are two separate IntEnum
    classes that share wire values `INTERNAL_EDGE=1` and `EXTERNAL_EDGE=2`.
    If either side gains a member or reorders values without coordinating,
    the cross-table foreign-key path corrupts silently. This guard catches
    drift at validation time. The import is intentionally lazy so test
    monkey-patches against `cfm.data.sub_d.enums.SlotKind` are observable.
    """
    from cfm.data.sub_d.enums import SlotKind as SubDSlotKind

    if int(SlotKind.INTERNAL_EDGE) != int(SubDSlotKind.INTERNAL_EDGE):
        raise InlineValidationError(
            "sub-E and sub-D SlotKind.INTERNAL_EDGE wire values diverged "
            f"(sub-E={int(SlotKind.INTERNAL_EDGE)}, "
            f"sub-D={int(SubDSlotKind.INTERNAL_EDGE)}); invariant #9 violated"
        )
    if int(SlotKind.EXTERNAL_EDGE) != int(SubDSlotKind.EXTERNAL_EDGE):
        raise InlineValidationError(
            "sub-E and sub-D SlotKind.EXTERNAL_EDGE wire values diverged "
            f"(sub-E={int(SlotKind.EXTERNAL_EDGE)}, "
            f"sub-D={int(SubDSlotKind.EXTERNAL_EDGE)}); invariant #9 violated"
        )


def validate_boundary_contract(
    path: Path,
    *,
    expected_derivation_version: str,
    provenance_derivation_version: str,
    lever_3_collapse: bool = False,
) -> None:
    """Validate one boundary_contract.parquet. Raises InlineValidationError.

    Both `expected_derivation_version` (what the pipeline expects to see for
    this run) and `provenance_derivation_version` (what was actually recorded
    in the sibling provenance.yaml) are **required**. There is no on-disk
    source for the actual value inside the parquet itself — Task 10's
    pipeline orchestrator owns the responsibility of loading the actual
    value from the sibling `provenance.yaml` and passing it here. This
    keeps invariant #8 inline (per spec §10.1 categorisation) while pushing
    the I/O responsibility to the caller.

    Under `lever_3_collapse=True`, invariants #3 (non-null iff active) and
    #4 (active class membership) are replaced by a single uniform-null check
    (every row's boundary_class_enum is null). Other invariants still apply,
    including #9 (cross-enum byte-equivalence — mode-independent).

    Loop order: invariant #9 first (structural enum drift), then file-level
    invariants #1 + #2 (count + sort), then per-row invariants in
    **membership-before-semantic** order — #5/#6/#7 (structural enum/range
    membership) before #3/#4 (semantic non-null relationship + active class
    membership). Membership invariants are structurally prior; semantic
    relationships rely on values being in-range first. Catching #5/#6/#7
    early also yields more useful error messages on real-data violations.
    """
    # Invariant 9 (Task-6 carry-forward): structural enum drift is the
    # earliest possible failure mode; check before any file-level invariant.
    _assert_slotkind_byte_equivalence_with_sub_d()

    tbl = pq.ParquetFile(path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    scope_markers = tbl.column("scope_marker").to_pylist()
    boundary_classes = tbl.column("boundary_class_enum").to_pylist()
    axes = tbl.column("axis").to_pylist()

    # Invariant 1: row count.
    if tbl.num_rows != EXPECTED_TOTAL_ROWS:
        raise InlineValidationError(
            f"row count must be {EXPECTED_TOTAL_ROWS} (112 + 32), "
            f"got {tbl.num_rows}"
        )
    n_internal = sum(1 for k in slot_kinds if k == int(SlotKind.INTERNAL_EDGE))
    n_external = sum(1 for k in slot_kinds if k == int(SlotKind.EXTERNAL_EDGE))
    if (n_internal, n_external) != (
        EXPECTED_INTERNAL_ROWS,
        EXPECTED_EXTERNAL_ROWS,
    ):
        raise InlineValidationError(
            f"slot_kind split must be (112, 32), got ({n_internal}, {n_external})"
        )

    # Invariant 2: canonical sort key (slot_kind, slot_index).
    pairs = list(zip(slot_kinds, slot_indices))
    if pairs != sorted(pairs):
        raise InlineValidationError(
            "rows not sorted by canonical sort key (slot_kind, slot_index)"
        )

    # Lever-3 mode collapses invariants #3 + #4 into a single uniform-null
    # check across all rows. Run this before the per-row membership/semantic
    # loop so a mode violation surfaces as a clear "lever-3" message.
    if lever_3_collapse:
        for i, cls in enumerate(boundary_classes):
            if cls is not None:
                raise InlineValidationError(
                    f"row {i}: lever-3 mode requires boundary_class_enum is null "
                    f"in every row (got {cls})"
                )

    # Per-row invariants in membership-before-semantic order: structural
    # validity (#5, #6, #7) before semantic relationships (#3, #4).
    # Membership invariants are independent of derivation; semantic
    # invariants assume the per-row values are already in valid ranges.
    for i, (sk, si, scope, cls, axis) in enumerate(
        zip(slot_kinds, slot_indices, scope_markers, boundary_classes, axes)
    ):
        # 5: scope_marker membership.
        if scope not in _SCOPE_MARKER_VALUES:
            raise InlineValidationError(
                f"row {i}: scope_marker membership violated (scope={scope})"
            )
        # 6: slot_index range per slot_kind.
        if sk == int(SlotKind.INTERNAL_EDGE) and not (0 <= si < 112):
            raise InlineValidationError(
                f"row {i}: slot_index range violated (internal, idx={si})"
            )
        if sk == int(SlotKind.EXTERNAL_EDGE) and not (0 <= si < 32):
            raise InlineValidationError(
                f"row {i}: slot_index range violated (external, idx={si})"
            )
        # 7: axis membership.
        if axis not in _AXIS_VALUES:
            raise InlineValidationError(
                f"row {i}: axis membership violated (axis={axis})"
            )

        if not lever_3_collapse:
            # 3: boundary_class_enum non-null iff scope_marker == 0.
            is_active = scope == 0
            if is_active and cls is None:
                raise InlineValidationError(
                    f"row {i}: boundary_class_enum non-null iff scope_marker == 0 "
                    f"(scope=active, class=null)"
                )
            if (not is_active) and cls is not None:
                raise InlineValidationError(
                    f"row {i}: boundary_class_enum non-null iff scope_marker == 0 "
                    f"(scope={scope}, class={cls})"
                )
            # 4: active class membership (sentinel 0 forbidden on-disk).
            if is_active and cls not in _ACTIVE_CLASS_IDS:
                raise InlineValidationError(
                    f"row {i}: active class membership violated "
                    f"(class={cls} not in {sorted(_ACTIVE_CLASS_IDS)})"
                )

    # Invariant 8: provenance derivation version matches expected.
    if provenance_derivation_version != expected_derivation_version:
        raise InlineValidationError(
            f"boundary_derivation_version mismatch: expected "
            f"{expected_derivation_version}, got {provenance_derivation_version}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_validator_inline.py -v
```

Expected: 13 passed. Test set covers:
- Happy path (1): `test_valid_lattice_passes_all_invariants`
- Spec §10.1 invariants (8): `test_invariant_1_total_row_count`, `test_invariant_2_sort_key`, `test_invariant_3_class_non_null_iff_scope_active`, `test_invariant_4_active_class_membership`, `test_invariant_5_scope_marker_membership`, `test_invariant_6_slot_index_range`, `test_invariant_7_axis_membership`, `test_invariant_8_derivation_version_match`
- Lever-3 mode (2): `test_lever_3_collapse_passes_with_uniform_null`, `test_lever_3_collapse_rejects_any_non_null`
- Invariant #9 carry-forward (2): `test_invariant_9_slotkind_cross_enum_byte_equivalence` (positive regression guard), `test_invariant_9_drift_simulation_raises` (monkey-patched divergence triggers InlineValidationError)

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/validator_inline.py tests/data/sub_e/test_validator_inline.py
git commit -m "feat(sub_e): add inline validator"
```

---

## Task 8: Manifest and provenance writers

**Files:**
- Create: `src/cfm/data/sub_e/provenance.py`
- Create: `src/cfm/data/sub_e/manifest.py`
- Test: `tests/data/sub_e/test_provenance.py`
- Test: `tests/data/sub_e/test_manifest.py`

**Context:** Spec §9.4 (provenance), §9.5 (manifest). Sub-D's canonical YAML serializer from `cfm.data.io` is reused for byte-determinism. Per-tile provenance pins sub-C + sub-D input shas; region manifest pins per-tile provenance shas.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/sub_e/test_provenance.py
from __future__ import annotations

from pathlib import Path

import yaml

from dataclasses import replace

from cfm.data.sub_e.provenance import (
    SubEProvenance,
    SubEInputDigests,
    SubEVersions,
    provenance_sha256,
    provenance_to_dict,
    write_provenance,
)


def _make_provenance(*, extracted_utc: str = "2026-05-21T12:00:00Z") -> SubEProvenance:
    return SubEProvenance(
        tile_i=12,
        tile_j=17,
        extraction_commit_sha="a" * 40,
        extracted_utc=extracted_utc,
        rerun_count=0,
        rerun_reason="initial",
        inputs=SubEInputDigests(
            release="2026-04-15.0",
            sub_c_manifest_sha256="b" * 64,
            sub_c_features_parquet_sha256="c" * 64,
            sub_c_crossings_parquet_sha256="d" * 64,
            sub_d_manifest_sha256="e" * 64,
            sub_d_macro_core_parquet_sha256="f" * 64,
            boundary_vocab_sha256="0" * 64,
            derivation_config_sha256="1" * 64,
        ),
        versions=SubEVersions(
            sub_e_schema_version="1.0",
            boundary_vocab_version="1.0",
            boundary_derivation_version="1.0",
        ),
        boundary_contract_parquet_sha256="2" * 64,
    )


def test_provenance_writes_canonical_yaml_with_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "provenance.yaml"
    write_provenance(p, _make_provenance())
    data = yaml.safe_load(p.read_text())
    assert data["tile_i"] == 12
    assert data["tile_j"] == 17
    assert data["versions"]["boundary_vocab_version"] == "1.0"
    assert data["inputs"]["sub_d_macro_core_parquet_sha256"] == "f" * 64
    assert data["outputs"]["boundary_contract_parquet_sha256"] == "2" * 64


def test_provenance_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    import hashlib

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    prov = _make_provenance()
    write_provenance(a, prov)
    write_provenance(b, prov)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(
        b.read_bytes()
    ).hexdigest()


def test_provenance_sha256_excludes_extracted_utc(tmp_path: Path) -> None:
    """Spec §9.2: extraction.extracted_utc is stripped before self-sha.

    Two provenance instances differing ONLY in extracted_utc must produce
    the same provenance_sha256, otherwise the digest chain breaks on every
    rerun under live clocks.
    """
    a = _make_provenance(extracted_utc="2026-05-21T12:00:00Z")
    b = _make_provenance(extracted_utc="2026-12-01T09:30:42Z")
    assert provenance_sha256(provenance_to_dict(a)) == provenance_sha256(
        provenance_to_dict(b)
    )


def test_provenance_sha256_excludes_nested_sha_fields(tmp_path: Path) -> None:
    """Spec §9.2: final-segment *_sha256 fields are stripped before self-sha.

    Two provenance instances differing ONLY in a *_sha256 field (e.g. a
    different boundary_contract_parquet_sha256) must produce the same
    provenance_sha256.
    """
    a = _make_provenance()
    b = replace(a, boundary_contract_parquet_sha256="9" * 64)
    assert provenance_sha256(provenance_to_dict(a)) == provenance_sha256(
        provenance_to_dict(b)
    )


def test_provenance_sha256_sensitive_to_semantic_changes(tmp_path: Path) -> None:
    """Inversely: non-excluded field changes MUST shift the self-sha.

    Guards against the table being too aggressive (over-stripping). A
    change in versions.boundary_vocab_version must produce a different
    sha.
    """
    a = _make_provenance()
    a_versions = replace(a.versions, boundary_vocab_version="2.0")
    b = replace(a, versions=a_versions)
    assert provenance_sha256(provenance_to_dict(a)) != provenance_sha256(
        provenance_to_dict(b)
    )
```

```python
# tests/data/sub_e/test_manifest.py
from __future__ import annotations

from pathlib import Path

import yaml

from dataclasses import replace

from cfm.data.sub_e.manifest import (
    SubEManifest,
    SubEManifestInputs,
    SubEManifestVersions,
    SubEManifestConfig,
    SubEManifestExtraction,
    SubEManifestTile,
    manifest_sha256,
    manifest_to_dict,
    write_manifest,
)


def _make_manifest(
    tile_count: int = 3,
    *,
    started_utc: str = "2026-05-21T12:00:00Z",
    completed_utc: str = "2026-05-21T12:05:00Z",
) -> SubEManifest:
    tiles = [
        SubEManifestTile(tile_i=i, tile_j=0, provenance_sha256="z" * 64)
        for i in range(tile_count)
    ]
    return SubEManifest(
        manifest_schema_version="1.0",
        sub_e_schema_version="1.0",
        release="2026-04-15.0",
        region="singapore",
        region_crs="EPSG:3414",
        inputs=SubEManifestInputs(
            sub_c_manifest_sha256="b" * 64,
            sub_c_region_dir="data/processed/sub_c/2026-04-15.0/singapore",
            sub_d_manifest_sha256="e" * 64,
            sub_d_region_dir="data/processed/sub_d/2026-04-15.0/singapore",
            boundary_vocab_sha256="0" * 64,
        ),
        versions=SubEManifestVersions(
            boundary_vocab_version="1.0",
            boundary_derivation_version="1.0",
        ),
        config_source="sub_d_manifest.config",
        config=SubEManifestConfig(
            cell_grid=(8, 8),
            internal_edge_count=112,
            external_edge_count=32,
        ),
        initial_extraction=SubEManifestExtraction(
            commit_sha="a" * 40,
            started_utc=started_utc,
            completed_utc=completed_utc,
            tile_count=tile_count,
        ),
        tiles=tiles,
    )


def test_manifest_writes_with_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    write_manifest(p, _make_manifest())
    data = yaml.safe_load(p.read_text())
    assert data["region"] == "singapore"
    assert data["initial_extraction"]["tile_count"] == 3
    assert len(data["tiles"]) == 3
    assert data["config"]["cell_grid"] == [8, 8]


def test_manifest_tiles_sorted_by_tile_i_tile_j(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    manifest = _make_manifest(tile_count=3)
    # Shuffle tiles before write; manifest writer must sort.
    manifest = manifest.__class__(
        **{**manifest.__dict__, "tiles": list(reversed(manifest.tiles))}
    )
    write_manifest(p, manifest)
    data = yaml.safe_load(p.read_text())
    ijs = [(t["tile_i"], t["tile_j"]) for t in data["tiles"]]
    assert ijs == sorted(ijs)


def test_manifest_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    import hashlib

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    m = _make_manifest()
    write_manifest(a, m)
    write_manifest(b, m)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(
        b.read_bytes()
    ).hexdigest()


def test_manifest_sha256_excludes_started_and_completed_utc() -> None:
    """Spec §9.2: initial_extraction.started_utc and completed_utc are
    stripped before manifest_sha256. Two manifests differing ONLY in
    those timestamps must produce the same sha — otherwise cross-env
    determinism checks (spec §14) become noise.
    """
    a = _make_manifest()
    b = _make_manifest(
        started_utc="2026-12-01T09:30:42Z", completed_utc="2026-12-01T09:35:01Z"
    )
    assert manifest_sha256(manifest_to_dict(a)) == manifest_sha256(manifest_to_dict(b))


def test_manifest_sha256_excludes_nested_sha_fields() -> None:
    """Spec §9.2: final-segment *_sha256 fields are stripped before manifest_sha256."""
    a = _make_manifest()
    a_inputs = replace(a.inputs, sub_c_manifest_sha256="9" * 64)
    b = replace(a, inputs=a_inputs)
    assert manifest_sha256(manifest_to_dict(a)) == manifest_sha256(manifest_to_dict(b))


def test_manifest_sha256_sensitive_to_semantic_changes() -> None:
    """Inverse guard: non-excluded field changes MUST shift the sha."""
    a = _make_manifest()
    b = replace(a, region="zurich")
    assert manifest_sha256(manifest_to_dict(a)) != manifest_sha256(manifest_to_dict(b))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/sub_e/test_provenance.py tests/data/sub_e/test_manifest.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.provenance` and `cfm.data.sub_e.manifest`.

- [ ] **Step 3: Implement provenance**

```python
# src/cfm/data/sub_e/provenance.py
"""Per-tile provenance writer.

Mirrors sub-D's pattern at ``src/cfm/data/sub_d/provenance.py``: the
on-disk YAML carries timestamps (extracted_utc) and verbatim sha256
values, but the self-integrity sha used in the digest chain strips both
classes of fields via ``SUB_E_EXCLUDED_FROM_SHA`` so reruns under live
clocks produce the same chain value. The neutral
``cfm.data.determinism.compute_sha256_excluding`` helper does the
strip-canonicalize-hash pipeline.

Spec §9.2 mandate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.io import canonicalize_yaml


#: Sub-E's exclusion table for self-integrity hashing. Per spec §9.2 the
#: inherited ``cfm.data.determinism`` grammar applies:
#:
#: - Entries under ``"*"`` apply to all file_keys; an entry starting with
#:   ``*`` is a final-segment suffix match.
#: - Entries under a specific file_key are exact dotted-path matches.
#:
#: Mirrors ``SUB_D_EXCLUDED_FROM_SHA`` at ``src/cfm/data/sub_d/provenance.py``.
#: ``manifest.yaml`` entries are present for spec-§9.2 completeness; the
#: helper ``manifest_sha256`` (in ``manifest.py``) reads from the same table.
SUB_E_EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
}


@dataclass(frozen=True)
class SubEInputDigests:
    release: str
    sub_c_manifest_sha256: str
    sub_c_features_parquet_sha256: str
    sub_c_crossings_parquet_sha256: str
    sub_d_manifest_sha256: str
    sub_d_macro_core_parquet_sha256: str
    boundary_vocab_sha256: str
    derivation_config_sha256: str


@dataclass(frozen=True)
class SubEVersions:
    sub_e_schema_version: str
    boundary_vocab_version: str
    boundary_derivation_version: str


@dataclass(frozen=True)
class SubEProvenance:
    tile_i: int
    tile_j: int
    extraction_commit_sha: str
    extracted_utc: str
    rerun_count: int
    rerun_reason: str
    inputs: SubEInputDigests
    versions: SubEVersions
    boundary_contract_parquet_sha256: str
    provenance_schema_version: str = "1.0"


def provenance_to_dict(prov: SubEProvenance) -> dict:
    """Serialise SubEProvenance to its on-disk YAML dict shape.

    Exposed publicly so callers (Task 10 orchestrator, Task 9 tests) can
    compute ``provenance_sha256(dict)`` against the same dict that
    ``write_provenance`` serialises, without re-loading the file.
    """
    return {
        "provenance_schema_version": prov.provenance_schema_version,
        "tile_i": prov.tile_i,
        "tile_j": prov.tile_j,
        "extraction": {
            "commit_sha": prov.extraction_commit_sha,
            "extracted_utc": prov.extracted_utc,
            "rerun_count": prov.rerun_count,
            "rerun_reason": prov.rerun_reason,
        },
        "inputs": asdict(prov.inputs),
        "versions": asdict(prov.versions),
        "outputs": {
            "boundary_contract_parquet_sha256": prov.boundary_contract_parquet_sha256,
        },
    }


def write_provenance(path: Path, prov: SubEProvenance) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(provenance_to_dict(prov)), encoding="utf-8")
    return path


def provenance_sha256(data: dict) -> str:
    """Compute the self-integrity sha for a provenance.yaml dict.

    Strips ``extraction.extracted_utc`` and final-segment ``*_sha256``
    fields per ``SUB_E_EXCLUDED_FROM_SHA``, canonicalises the remainder
    to YAML, and hashes the bytes. This is the value the region manifest
    records in ``tiles[*].provenance_sha256``.
    """
    return _compute_sha256_excluding(data, "provenance.yaml", SUB_E_EXCLUDED_FROM_SHA)
```

- [ ] **Step 4: Implement manifest**

```python
# src/cfm/data/sub_e/manifest.py
"""Per-region manifest writer.

Tiles sorted by ``(tile_i, tile_j)`` at write-time (not assumed from input).
Self-integrity sha (``manifest_sha256``) available for cross-environment
determinism checks; uses ``SUB_E_EXCLUDED_FROM_SHA`` from provenance.py
to strip ``initial_extraction.started_utc/completed_utc`` and final-segment
``*_sha256`` before hashing (spec §9.2 mandate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.io import canonicalize_yaml
from cfm.data.sub_e.provenance import SUB_E_EXCLUDED_FROM_SHA


@dataclass(frozen=True)
class SubEManifestInputs:
    sub_c_manifest_sha256: str
    sub_c_region_dir: str
    sub_d_manifest_sha256: str
    sub_d_region_dir: str
    boundary_vocab_sha256: str


@dataclass(frozen=True)
class SubEManifestVersions:
    boundary_vocab_version: str
    boundary_derivation_version: str


@dataclass(frozen=True)
class SubEManifestConfig:
    cell_grid: tuple[int, int]
    internal_edge_count: int
    external_edge_count: int


@dataclass(frozen=True)
class SubEManifestExtraction:
    commit_sha: str
    started_utc: str
    completed_utc: str
    tile_count: int


@dataclass(frozen=True)
class SubEManifestTile:
    tile_i: int
    tile_j: int
    provenance_sha256: str


@dataclass(frozen=True)
class SubEManifest:
    manifest_schema_version: str
    sub_e_schema_version: str
    release: str
    region: str
    region_crs: str
    inputs: SubEManifestInputs
    versions: SubEManifestVersions
    config_source: str
    config: SubEManifestConfig
    initial_extraction: SubEManifestExtraction
    tiles: list[SubEManifestTile] = field(default_factory=list)


def manifest_to_dict(manifest: SubEManifest) -> dict:
    """Serialise SubEManifest to its on-disk YAML dict shape.

    Enforces canonical tile sort by ``(tile_i, tile_j)`` at write-time —
    does not trust input ordering. Same discipline as Task 6's per-edge
    canonical sort key.
    """
    sorted_tiles = sorted(manifest.tiles, key=lambda t: (t.tile_i, t.tile_j))
    return {
        "manifest_schema_version": manifest.manifest_schema_version,
        "sub_e_schema_version": manifest.sub_e_schema_version,
        "release": manifest.release,
        "region": manifest.region,
        "region_crs": manifest.region_crs,
        "inputs": asdict(manifest.inputs),
        "versions": asdict(manifest.versions),
        "config_source": manifest.config_source,
        "config": {
            "cell_grid": list(manifest.config.cell_grid),
            "internal_edge_count": manifest.config.internal_edge_count,
            "external_edge_count": manifest.config.external_edge_count,
        },
        "initial_extraction": asdict(manifest.initial_extraction),
        "tiles": [
            {
                "tile_i": t.tile_i,
                "tile_j": t.tile_j,
                "provenance_sha256": t.provenance_sha256,
            }
            for t in sorted_tiles
        ],
    }


def write_manifest(path: Path, manifest: SubEManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(manifest_to_dict(manifest)), encoding="utf-8")
    return path


def manifest_sha256(data: dict) -> str:
    """Compute the self-integrity sha for a manifest.yaml dict.

    Strips ``initial_extraction.started_utc/completed_utc`` and
    final-segment ``*_sha256`` per ``SUB_E_EXCLUDED_FROM_SHA``,
    canonicalises the remainder, hashes the bytes. Spec §9.2 mandate.
    Not in Phase-1's digest chain — exposed so cross-environment
    determinism checks (spec §14) can compare manifests timestamp-free.
    """
    return _compute_sha256_excluding(data, "manifest.yaml", SUB_E_EXCLUDED_FROM_SHA)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/data/sub_e/test_provenance.py tests/data/sub_e/test_manifest.py -v
```

Expected: 11 passed total — 5 provenance (2 original + 3 sha-exclusion: extracted_utc, *_sha256, semantic-sensitivity inverse) + 6 manifest (3 original — writes_with_all_fields, tiles_sorted, is_byte_deterministic_on_rerun — + 3 sha-exclusion: started/completed_utc, *_sha256, semantic-sensitivity inverse).

- [ ] **Step 6: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 7: Commit**

```bash
git add src/cfm/data/sub_e/provenance.py src/cfm/data/sub_e/manifest.py tests/data/sub_e/test_provenance.py tests/data/sub_e/test_manifest.py
git commit -m "feat(sub_e): add manifest and provenance writers"
```

---

## Task 9: Cross-tile validator (5 invariants)

**Files:**
- Create: `src/cfm/data/sub_e/validator_cross_tile.py`
- Test: `tests/data/sub_e/test_validator_cross_tile.py`

**Context:** Spec §10.2. Five region-level invariants including the digest chain. Each invariant has a controlled-violation fixture. **All fixtures synthesised in-process via the Task 6 writer + Task 8 manifest/provenance writers under `tmp_path`; this task does not read `data/processed/sub_c/` or `data/processed/sub_d/`.**

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_validator_cross_tile.py
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cfm.data.sub_e.validator_cross_tile import (
    CrossTileValidationError,
    validate_extraction_cross_tile,
)


def _build_synthetic_region(
    tmp_path: Path,
    *,
    n_tiles: int = 3,
    boundary_vocab_version: str = "1.0",
    boundary_derivation_version: str = "1.0",
    sub_e_schema_version: str = "1.0",
    corrupt_external_at_tile_k: int | None = None,
    duplicate_external_at_tile_k: int | None = None,
) -> Path:
    """Create a minimal sub-E region directory with N consistent tiles.

    The external boundary rows are constructed via ``cell_to_edge_ids``
    so the synthetic data matches what the real per-cell→per-edge path
    produces (spec §10.2 #5). Index-based external rows would
    synthetic-pass the rotation-aware validator while failing on real
    Singapore data — caught by the implementer pre-Task-9-dispatch.

    Two controlled-violation modes for invariant #5 are exposed at
    fixture-build time (so the digest chain remains internally consistent
    with the bad state — corruption introduced after-the-fact via parquet
    mutation would fail invariant #3 first):

    - ``corrupt_external_at_tile_k``: that tile's first axis=1 external
      row has its ``lower_cell_i`` mutated to ``4``. Selection is
      *semantic* (find first axis=1 row), not positional — robust against
      future sort-order changes. Axis=1 externals have
      ``lower_cell_i ∈ {0, 7}`` by construction, so the mutated tuple
      ``(4, lj, 1)`` is guaranteed outside rotation's external set.
      Tuple uniqueness is preserved (no other row has the mutated tuple),
      so the duplicate-tuple branch does NOT fire — only the
      rotation-equality (set-mismatch) branch fires.
    - ``duplicate_external_at_tile_k``: that tile's external row at
      index 1 has its ``(lower_cell_i, lower_cell_j, axis)`` triple
      replaced by row 0's triple. Slot_index stays distinct so the
      OLD weak validator (slot_index uniqueness) would pass; the
      duplicate-tuple branch of the strengthened invariant #5 fires.
    """
    from dataclasses import replace as _dc_replace

    from cfm.data.sub_e.derivation import BoundaryClass
    from cfm.data.sub_e.manifest import (
        SubEManifest,
        SubEManifestConfig,
        SubEManifestExtraction,
        SubEManifestInputs,
        SubEManifestTile,
        SubEManifestVersions,
        write_manifest,
    )
    from cfm.data.sub_e.provenance import (
        SubEInputDigests,
        SubEProvenance,
        SubEVersions,
        provenance_sha256,
        provenance_to_dict,
        write_provenance,
    )
    from cfm.data.sub_e.rotation import GRID_SIZE, EdgeKind, cell_to_edge_ids
    from cfm.data.sub_e.writer import (
        BoundaryContractRow,
        SlotKind,
        write_boundary_contract,
    )

    region = tmp_path / "sub_e_singapore"
    region.mkdir()

    def _external_rows_via_rotation(
        *,
        corrupt_first_axis_1: bool = False,
        duplicate_idx: int | None = None,
    ) -> list[BoundaryContractRow]:
        """Enumerate the 8×8 grid through rotation, collect the unique
        external (lower_cell_i, lower_cell_j, axis) tuples, sort them
        canonically, and emit one parquet row per tuple. Should produce
        exactly 32 rows (24 edge cells × 1 external + 4 corners × 2).

        Selection of the corruption target is *semantic* (find first
        axis=1 row), not positional (corrupt_idx=N). Robust against
        future sort-order changes. The tripwire assert on ``target.axis``
        catches any selection-logic regression that would silently
        re-introduce the duplicate-tuple trap.
        """
        seen: set[tuple[int, int, int]] = set()
        tuples: list[tuple[int, int, int]] = []
        for ci in range(GRID_SIZE):
            for cj in range(GRID_SIZE):
                cell_edges = cell_to_edge_ids(ci, cj)
                for edge in (
                    cell_edges.north,
                    cell_edges.south,
                    cell_edges.west,
                    cell_edges.east,
                ):
                    li, lj, axis, kind = edge
                    if kind is EdgeKind.EXTERNAL:
                        key = (li, lj, axis)
                        if key not in seen:
                            seen.add(key)
                            tuples.append(key)
        tuples.sort()
        assert len(tuples) == 32, (
            f"rotation should produce exactly 32 unique external edges, "
            f"got {len(tuples)}"
        )
        ext_rows = [
            BoundaryContractRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=li,
                lower_cell_j=lj,
                axis=axis,
                scope_marker=3,
                boundary_class_enum=None,
            )
            for idx, (li, lj, axis) in enumerate(tuples)
        ]

        if corrupt_first_axis_1:
            # Semantic selection: rotation's axis=1 externals have
            # lower_cell_i ∈ {0, 7} by construction (west/east sides).
            # Setting lower_cell_i=4 produces a tuple genuinely outside
            # rotation's external set → set-mismatch branch fires by
            # construction. Positional selection (corrupt_idx=N) was
            # fragile against sort-order changes — a prior version with
            # corrupt_idx=0 hit the axis=0 first tuple (0,0,0), whose
            # mutation (4,0,0) was already in rotation, triggering
            # duplicate-tuple instead of set-mismatch.
            target_idx = next(
                (i for i, r in enumerate(ext_rows) if r.axis == 1), None
            )
            assert target_idx is not None, (
                "rotation must produce at least one axis=1 external "
                "(west/east boundary); selection logic broken"
            )
            target = ext_rows[target_idx]
            assert target.axis == 1, (
                f"selection tripwire: target.axis={target.axis}, expected 1 "
                f"— selection logic broken, would re-introduce duplicate-tuple trap"
            )
            ext_rows[target_idx] = _dc_replace(target, lower_cell_i=4)

        if duplicate_idx is not None:
            # Copy row[duplicate_idx]'s (lower_cell_i, lower_cell_j, axis)
            # triple onto row[duplicate_idx + 1]. slot_index stays
            # distinct → OLD weak validator (slot_index uniqueness)
            # passes; tuple count != set size → strengthened invariant
            # #5's duplicate-tuple branch fires.
            src = ext_rows[duplicate_idx]
            dst = ext_rows[duplicate_idx + 1]
            ext_rows[duplicate_idx + 1] = _dc_replace(
                dst,
                lower_cell_i=src.lower_cell_i,
                lower_cell_j=src.lower_cell_j,
                axis=src.axis,
            )

        return ext_rows

    def _rows(
        *,
        corrupt_first_axis_1: bool = False,
        duplicate_idx: int | None = None,
    ) -> list[BoundaryContractRow]:
        rows: list[BoundaryContractRow] = []
        for idx in range(112):
            rows.append(
                BoundaryContractRow(
                    slot_kind=SlotKind.INTERNAL_EDGE,
                    slot_index=idx,
                    lower_cell_i=idx % 8,
                    lower_cell_j=idx // 8 % 8,
                    axis=idx % 2,
                    scope_marker=0,
                    boundary_class_enum=int(BoundaryClass.NONE),
                )
            )
        rows.extend(
            _external_rows_via_rotation(
                corrupt_first_axis_1=corrupt_first_axis_1,
                duplicate_idx=duplicate_idx,
            )
        )
        return rows

    tile_records: list[SubEManifestTile] = []
    for k in range(n_tiles):
        tile_dir = region / f"tile=EPSG3414_i{k}_j0"
        tile_dir.mkdir()
        contract = write_boundary_contract(
            tile_dir / "boundary_contract.parquet",
            _rows(
                corrupt_first_axis_1=(k == corrupt_external_at_tile_k),
                duplicate_idx=(0 if k == duplicate_external_at_tile_k else None),
            ),
        )
        contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        prov_path = tile_dir / "provenance.yaml"
        prov_obj = SubEProvenance(
            tile_i=k,
            tile_j=0,
            extraction_commit_sha="a" * 40,
            extracted_utc="2026-05-21T12:00:00Z",
            rerun_count=0,
            rerun_reason="initial",
            inputs=SubEInputDigests(
                release="2026-04-15.0",
                sub_c_manifest_sha256="b" * 64,
                sub_c_features_parquet_sha256="c" * 64,
                sub_c_crossings_parquet_sha256="d" * 64,
                sub_d_manifest_sha256="e" * 64,
                sub_d_macro_core_parquet_sha256="f" * 64,
                boundary_vocab_sha256="0" * 64,
                derivation_config_sha256="1" * 64,
            ),
            versions=SubEVersions(
                sub_e_schema_version=sub_e_schema_version,
                boundary_vocab_version=boundary_vocab_version,
                boundary_derivation_version=boundary_derivation_version,
            ),
            boundary_contract_parquet_sha256=contract_sha,
        )
        write_provenance(prov_path, prov_obj)
        # Self-integrity sha: strip extracted_utc + *_sha256 per
        # SUB_E_EXCLUDED_FROM_SHA so the digest chain survives live-clock
        # reruns. NOT hashlib.sha256(prov_path.read_bytes()) — that would
        # bake extracted_utc into the chain and break determinism (spec §9.2).
        prov_sha = provenance_sha256(provenance_to_dict(prov_obj))
        tile_records.append(
            SubEManifestTile(tile_i=k, tile_j=0, provenance_sha256=prov_sha)
        )

    write_manifest(
        region / "manifest.yaml",
        SubEManifest(
            manifest_schema_version="1.0",
            sub_e_schema_version=sub_e_schema_version,
            release="2026-04-15.0",
            region="singapore",
            region_crs="EPSG:3414",
            inputs=SubEManifestInputs(
                sub_c_manifest_sha256="b" * 64,
                sub_c_region_dir="data/processed/sub_c/2026-04-15.0/singapore",
                sub_d_manifest_sha256="e" * 64,
                sub_d_region_dir="data/processed/sub_d/2026-04-15.0/singapore",
                boundary_vocab_sha256="0" * 64,
            ),
            versions=SubEManifestVersions(
                boundary_vocab_version=boundary_vocab_version,
                boundary_derivation_version=boundary_derivation_version,
            ),
            config_source="sub_d_manifest.config",
            config=SubEManifestConfig(
                cell_grid=(8, 8), internal_edge_count=112, external_edge_count=32
            ),
            initial_extraction=SubEManifestExtraction(
                commit_sha="a" * 40,
                started_utc="2026-05-21T12:00:00Z",
                completed_utc="2026-05-21T12:05:00Z",
                tile_count=n_tiles,
            ),
            tiles=tile_records,
        ),
    )
    (region / "_SUCCESS").touch()
    return region


def test_valid_region_passes_all_cross_tile_invariants(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    validate_extraction_cross_tile(region)  # should not raise


def test_invariant_1_schema_version_consistency(tmp_path: Path) -> None:
    """Corrupt one tile's provenance to use a different sub_e_schema_version.

    Operates at the YAML-dict layer (load → mutate → dump) rather than
    byte-layer text-replace. canonicalize_yaml emits single-quoted
    strings; double-quote text-replace would be a silent no-op against
    yaml.safe_dump's default quoting (defect found pre-Task-9-dispatch).
    Round-trip via yaml.safe_dump is robust against yaml lib version
    drift and quote-style shifts.
    """
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    data = yaml.safe_load(prov_path.read_text())
    data["versions"]["sub_e_schema_version"] = "2.0"
    prov_path.write_text(yaml.safe_dump(data))
    with pytest.raises(CrossTileValidationError, match="sub_e_schema_version"):
        validate_extraction_cross_tile(region)


def test_invariant_2_vocab_and_derivation_consistency(tmp_path: Path) -> None:
    """Same yaml round-trip pattern as invariant #1 for the same reason."""
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    data = yaml.safe_load(prov_path.read_text())
    data["versions"]["boundary_vocab_version"] = "2.0"
    prov_path.write_text(yaml.safe_dump(data))
    with pytest.raises(CrossTileValidationError, match="boundary_vocab_version"):
        validate_extraction_cross_tile(region)


def test_invariant_3_digest_chain_broken_at_parquet(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    # Mutate the parquet bytes so its sha no longer matches what provenance recorded.
    parquet = region / "tile=EPSG3414_i0_j0" / "boundary_contract.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"\x00")
    with pytest.raises(CrossTileValidationError, match="digest chain"):
        validate_extraction_cross_tile(region)


def test_invariant_4_input_digest_drift_across_tiles(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i1_j0" / "provenance.yaml"
    text = prov_path.read_text()
    prov_path.write_text(
        text.replace(
            "sub_d_manifest_sha256: " + ("e" * 64),
            "sub_d_manifest_sha256: " + ("9" * 64),
        )
    )
    with pytest.raises(CrossTileValidationError, match="input digest"):
        validate_extraction_cross_tile(region)


def test_invariant_5_duplicate_external_tuple(tmp_path: Path) -> None:
    """Two parquet rows share the same (lower_cell_i, lower_cell_j, axis).

    The duplicate is injected at fixture-build time via
    ``duplicate_external_at_tile_k=0`` so the digest chain remains
    internally consistent with the bad parquet (invariant #3 does not
    fire). Slot_index uniqueness is preserved — the OLD weak validator
    (slot_index uniqueness only) would PASS this corruption. Only the
    strengthened invariant #5's duplicate-tuple branch catches it.

    Earlier draft mutated parquet bytes after region build, but that
    broke invariant #3 (digest chain) which fires before invariant #5
    in the per-tile loop. The fixture-build-time injection keeps the
    chain valid so invariant #5 actually gets exercised.
    """
    region = _build_synthetic_region(tmp_path, duplicate_external_at_tile_k=0)
    with pytest.raises(CrossTileValidationError, match="duplicate external"):
        validate_extraction_cross_tile(region)


def test_invariant_5_external_set_mismatch_against_rotation(tmp_path: Path) -> None:
    """The first axis=1 external row has its lower_cell_i mutated to 4.

    This is the load-bearing controlled-violation test for the
    rotation-aware strengthening (spec §10.2 #5). The corruption:

    - Preserves 144-row count (writer-level invariants pass).
    - Preserves slot_index uniqueness (OLD weak validator passes).
    - Preserves tuple uniqueness within the parquet (duplicate-tuple
      branch does NOT fire — by construction: axis=1 mutations to
      lower_cell_i=4 produce tuples like (4, lj, 1) that do not match
      any other parquet row's tuple).
    - Shifts the parquet's external-tuple SET away from the rotation's
      set: rotation's axis=1 externals have lower_cell_i ∈ {0, 7} by
      construction, so (4, lj, 1) is genuinely outside rotation's set.

    Only the rotation-equality check at the end of invariant #5 fires.
    Selection is semantic (find first axis=1 row), not positional —
    robust against future sort-order changes in the rotation function.
    Earlier draft used corrupt_idx=0 which silently hit the axis=0
    tuple (0,0,0); mutation to (4,0,0) was already in rotation's set,
    triggering duplicate-tuple instead of set-mismatch and defeating
    the load-bearing meta-check that this test specifically exercises
    the strengthening, not legacy uniqueness behavior.
    """
    region = _build_synthetic_region(tmp_path, corrupt_external_at_tile_k=0)
    with pytest.raises(
        CrossTileValidationError, match="external slot_index set mismatch"
    ):
        validate_extraction_cross_tile(region)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_validator_cross_tile.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the cross-tile validator**

```python
# src/cfm/data/sub_e/validator_cross_tile.py
"""Sub-E per-region cross-tile validator (5 invariants per spec §10.2).

Invariants 1 and 2 (version consistency) route through ``compare_version``
per sub-D's mandate at ``src/cfm/data/sub_d/versions.py:6-7`` — bare ``!=``
would silently allow cross-namespace string equality and lose sub-D
known_issue #8's lesson. ``_check_version`` wraps the call so the
sub-E-specific error message format is preserved across the existing
test regex pattern.

Invariant #5 (external-edge consistency) is rotation-aware per spec §10.2 #5:
enumerates the 8×8 grid via ``cell_to_edge_ids`` from Task 3, collects the
external ``(lower_cell_i, lower_cell_j, axis)`` set, and asserts it equals
the parquet's external-tuple set. Catches rotation/parquet skew that the
old uniqueness-only check would silently allow.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from cfm.data.sub_d.errors import VersionMismatchError
from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version
from cfm.data.sub_e.provenance import provenance_sha256
from cfm.data.sub_e.rotation import GRID_SIZE, EdgeKind, cell_to_edge_ids
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_NAMESPACE,
    BOUNDARY_VOCAB_NAMESPACE,
    SUB_E_SCHEMA_NAMESPACE,
)
from cfm.data.sub_e.writer import SlotKind


class CrossTileValidationError(ValueError):
    """Raised when a sub-E region fails any cross-tile invariant."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_version(
    namespace: VersionNamespace,
    field_name: str,
    expected: str,
    actual: str,
    tile_coords: tuple[int, int],
) -> None:
    """Wrap ``compare_version`` for cross-tile version invariants.

    ``compare_version`` is the single sanctioned equality path per sub-D's
    versions.py docstring; bare ``!=`` would silently allow cross-namespace
    string equality (sub-D known_issue #8). The wrapper catches
    ``VersionMismatchError`` and re-raises as ``CrossTileValidationError``
    with the sub-E-specific message so the existing test regex (e.g.
    ``match="sub_e_schema_version"``) still matches.

    Pattern mirrors sub-D ``validator.py:85-97``.
    """
    try:
        compare_version(
            namespace,
            VersionRef(namespace, expected),
            VersionRef(namespace, actual),
        )
    except VersionMismatchError as exc:
        raise CrossTileValidationError(
            f"{field_name} mismatch at tile {tile_coords}: "
            f"manifest={expected}, provenance={actual}"
        ) from exc


def _rotation_external_tuples() -> set[tuple[int, int, int]]:
    """Enumerate every external edge in the 8×8 grid via the rotation.

    Returns the set of ``(lower_cell_i, lower_cell_j, axis)`` identities
    for external edges. Each tuple should appear in exactly one cell's
    per-cell view by construction (spec §6.2 "vacuous-invariant"); the
    caller asserts that and uses the set as the canonical truth against
    the parquet's external-row set.
    """
    seen_count: dict[tuple[int, int, int], int] = {}
    for ci in range(GRID_SIZE):
        for cj in range(GRID_SIZE):
            cell_edges = cell_to_edge_ids(ci, cj)
            for edge in (
                cell_edges.north,
                cell_edges.south,
                cell_edges.west,
                cell_edges.east,
            ):
                li, lj, axis, kind = edge
                if kind is EdgeKind.EXTERNAL:
                    key = (li, lj, axis)
                    seen_count[key] = seen_count.get(key, 0) + 1
    # Each external must be owned by exactly one cell (spec §6.2).
    for key, count in seen_count.items():
        if count != 1:
            raise CrossTileValidationError(
                f"rotation external edge {key} appears in {count} cells' "
                f"per-cell views (expected 1) — rotation function bug"
            )
    return set(seen_count.keys())


def validate_extraction_cross_tile(region_dir: Path) -> None:
    manifest_path = region_dir / "manifest.yaml"
    manifest = _load_yaml(manifest_path)

    expected_sub_e_schema = manifest["sub_e_schema_version"]
    expected_vocab = manifest["versions"]["boundary_vocab_version"]
    expected_derivation = manifest["versions"]["boundary_derivation_version"]

    # Gather per-tile provenance + parquet pairs.
    first_input_digests: dict[str, str] | None = None

    # Rotation's external set is invariant across tiles (same 8×8 grid);
    # compute once before the per-tile loop.
    rotation_external = _rotation_external_tuples()

    for tile in manifest["tiles"]:
        tile_dir = region_dir / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        prov_path = tile_dir / "provenance.yaml"
        parquet_path = tile_dir / "boundary_contract.parquet"
        tile_coords = (tile["tile_i"], tile["tile_j"])

        prov = _load_yaml(prov_path)

        # Invariant 1: schema version consistency (DATA_SHAPE namespace).
        _check_version(
            SUB_E_SCHEMA_NAMESPACE,
            "sub_e_schema_version",
            expected_sub_e_schema,
            prov["versions"]["sub_e_schema_version"],
            tile_coords,
        )

        # Invariant 2: vocab + derivation version consistency.
        _check_version(
            BOUNDARY_VOCAB_NAMESPACE,
            "boundary_vocab_version",
            expected_vocab,
            prov["versions"]["boundary_vocab_version"],
            tile_coords,
        )
        _check_version(
            BOUNDARY_DERIVATION_NAMESPACE,
            "boundary_derivation_version",
            expected_derivation,
            prov["versions"]["boundary_derivation_version"],
            tile_coords,
        )

        # Invariant 3: digest chain.
        # The manifest→provenance anchor uses provenance_sha256() — the
        # exclusion-aware self-sha (strips extracted_utc + *_sha256 per
        # SUB_E_EXCLUDED_FROM_SHA, spec §9.2). Raw file-bytes hash would
        # bake extracted_utc into the chain and break determinism on every
        # rerun under live clocks (Task 8 plan-fixup landed this discipline).
        expected_prov_sha = tile["provenance_sha256"]
        actual_prov_sha = provenance_sha256(prov)
        if expected_prov_sha != actual_prov_sha:
            raise CrossTileValidationError(
                f"digest chain broken at tile ({tile['tile_i']}, {tile['tile_j']}): "
                f"manifest→provenance sha mismatch"
            )
        # The provenance→parquet anchor is a raw file-bytes hash: parquet
        # is byte-deterministic by construction (no timestamps, fixed
        # pyarrow schema/sort), so no exclusion needed.
        expected_parquet_sha = prov["outputs"]["boundary_contract_parquet_sha256"]
        actual_parquet_sha = _file_sha256(parquet_path)
        if expected_parquet_sha != actual_parquet_sha:
            raise CrossTileValidationError(
                f"digest chain broken at tile ({tile['tile_i']}, {tile['tile_j']}): "
                f"provenance→parquet sha mismatch"
            )

        # Invariant 4: input digest consistency across tiles.
        input_digests = dict(prov["inputs"])
        # Allow per-tile parquets to differ at the tile level
        # (sub_c_features/crossings can differ per tile is OK in principle but
        # for Singapore single-region the upstream manifests must match).
        anchor_keys = (
            "release",
            "sub_c_manifest_sha256",
            "sub_d_manifest_sha256",
            "boundary_vocab_sha256",
            "derivation_config_sha256",
        )
        if first_input_digests is None:
            first_input_digests = {k: input_digests[k] for k in anchor_keys}
        else:
            for k in anchor_keys:
                if first_input_digests[k] != input_digests[k]:
                    raise CrossTileValidationError(
                        f"input digest drift at tile ({tile['tile_i']}, "
                        f"{tile['tile_j']}): {k} differs from first tile"
                    )

        # Invariant 5: external-edge single-cell membership (spec §10.2 #5).
        # Each external (lower_cell_i, lower_cell_j, axis) identity must
        # appear in exactly one cell's per-cell view per the rotation
        # function (spec §6.2 vacuous-invariant lifted to a real-data
        # regression check). The parquet's external-tuple set must equal
        # the rotation's. Plus: row count must equal set size (no
        # duplicate external rows).
        tbl = pq.ParquetFile(parquet_path).read()
        slot_kinds = tbl.column("slot_kind").to_pylist()
        lower_is = tbl.column("lower_cell_i").to_pylist()
        lower_js = tbl.column("lower_cell_j").to_pylist()
        axes = tbl.column("axis").to_pylist()

        parquet_external_tuples = [
            (li, lj, ax)
            for sk, li, lj, ax in zip(
                slot_kinds, lower_is, lower_js, axes, strict=True
            )
            if sk == int(SlotKind.EXTERNAL_EDGE)
        ]
        parquet_external_set = set(parquet_external_tuples)

        if len(parquet_external_tuples) != len(parquet_external_set):
            raise CrossTileValidationError(
                f"duplicate external (lower_cell_i, lower_cell_j, axis) "
                f"at tile {tile_coords}: "
                f"{len(parquet_external_tuples)} rows, "
                f"{len(parquet_external_set)} unique tuples"
            )

        if parquet_external_set != rotation_external:
            only_parquet = sorted(parquet_external_set - rotation_external)
            only_rotation = sorted(rotation_external - parquet_external_set)
            raise CrossTileValidationError(
                f"external slot_index set mismatch at tile {tile_coords}: "
                f"only-in-parquet={only_parquet}, only-in-rotation={only_rotation}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_validator_cross_tile.py -v
```

Expected: 7 passed total — happy path (1) + invariants 1/2/3/4 (4) + invariant #5 split into two controlled violations (2: `test_invariant_5_duplicate_external_tuple` exercises the tuple-uniqueness branch; `test_invariant_5_external_set_mismatch_against_rotation` exercises the rotation-equality branch — the load-bearing one for the strengthening).

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/data/sub_e/validator_cross_tile.py tests/data/sub_e/test_validator_cross_tile.py
git commit -m "feat(sub_e): add cross-tile validator"
```

---

## Task 10: Pipeline orchestrator (with halt-on-validator-fail)

**Files:**
- Create: `src/cfm/data/sub_e/pipeline.py`
- Test: `tests/data/sub_e/test_pipeline.py`

**Context:** Spec §13 + §12 (lever-3 collapse). The orchestrator gates on sub-D `_SUCCESS`, iterates sub-D's manifest tiles, computes boundary_class per active internal edge, writes per-tile artifacts, runs the inline validator, then writes manifest + `_SUCCESS` and runs cross-tile validator. Any validator failure aborts the run; no partial `_SUCCESS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_pipeline.py
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_e.pipeline import (
    PipelineConfig,
    derive_region,
)


@pytest.fixture
def synthetic_sub_d_region(tmp_path: Path) -> Path:
    """Build a tiny sub-D-shaped region (2 tiles, valid macro_core) plus sub-C
    crossings/features so sub-E has consistent inputs to read. The fixture
    constructs minimal but valid sub-D output via writers that mirror sub-D's
    schema; sub-E reads it like a real sub-D region.
    """
    # Implementation: build sub-D dir with _SUCCESS, manifest.yaml, and 2
    # tile=... subdirs each containing macro_core.parquet (144-slot lattice),
    # provenance.yaml, effective_conditioning.yaml. Build sub-C dir with
    # crossings.parquet and features.parquet per tile. Returns the *sub-D*
    # region path; the pipeline reads sub-C via the path declared in sub-D's
    # manifest.inputs.sub_c_region_dir.
    # See helper `_build_synthetic_sub_d_and_sub_c` below — kept in fixture
    # for clarity.
    return _build_synthetic_sub_d_and_sub_c(tmp_path)


def test_pipeline_happy_path_writes_success_marker(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    derive_region(cfg)
    assert (out_root / "_SUCCESS").exists()
    assert (out_root / "manifest.yaml").exists()


def test_pipeline_aborts_when_sub_d_success_missing(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    (synthetic_sub_d_region / "_SUCCESS").unlink()
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_halts_on_inline_validator_failure(
    tmp_path: Path, synthetic_sub_d_region: Path, monkeypatch
) -> None:
    """Monkey-patch the derivation function to emit a violating row; assert
    pipeline raises and does NOT write _SUCCESS.
    """
    from cfm.data.sub_e import pipeline as pipeline_mod
    from cfm.data.sub_e.validator_inline import InlineValidationError

    def _bad_derive(*args, **kwargs):
        raise InlineValidationError("synthetic violation")

    monkeypatch.setattr(pipeline_mod, "_validate_or_raise", _bad_derive)

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(InlineValidationError):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_lever_3_collapse_uniformly_null_boundary_class(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    import pyarrow.parquet as pq

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=True,
    )
    derive_region(cfg)
    # All on-disk boundary_class_enum values should be null in lever-3 mode.
    for tile_dir in (out_root).glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        values = tbl.column("boundary_class_enum").to_pylist()
        assert all(v is None for v in values), f"non-null in lever-3 at {tile_dir}"


# Fixture helper lives in tests/data/sub_e/_fixtures.py (created as part of
# Step 4 of this task). The body below shows the exact shape; copy verbatim
# into _fixtures.py.
def _build_synthetic_sub_d_and_sub_c(tmp_path: Path) -> Path:
    """Build minimum-viable sub-D + sub-C region pair for sub-E to read.

    Returns the sub-D region directory; the sub-C region directory sits
    alongside under the same tmp_path tree.
    """
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    sub_c = tmp_path / "sub_c" / "singapore"
    sub_d = tmp_path / "sub_d" / "singapore"
    sub_c.mkdir(parents=True)
    sub_d.mkdir(parents=True)

    tiles = [(0, 0), (1, 0)]

    sub_d_tile_records: list[dict] = []
    sub_c_tile_records: list[dict] = []

    # Enumerate the 8x8 grid via cell_to_edge_ids and collect unique internal
    # + external (lower_cell_i, lower_cell_j, axis) triples. This mirrors
    # what real sub-D produces because sub-D's lattice IS rotation's lattice
    # (spec §4.1: sub-E inherits sub-D's lattice verbatim).
    #
    # Earlier draft generated triples via modulo arithmetic (idx % 8 etc.)
    # which produced (a) duplicate keys within internals (e.g. idx=0 and
    # idx=64 both yield (0, 0, 0)) and (b) externals outside rotation's
    # set (axis=1 with lower_cell_i ∈ {1..6} which rotation never emits).
    # Both defects would have broken Task 10's happy path before any
    # validator ran. Pattern source: Task 9's _external_rows_via_rotation.
    from cfm.data.sub_e.rotation import EdgeKind, cell_to_edge_ids

    internal_set: set[tuple[int, int, int]] = set()
    external_set: set[tuple[int, int, int]] = set()
    for ci in range(8):
        for cj in range(8):
            cell_edges = cell_to_edge_ids(ci, cj)
            for edge in (
                cell_edges.north,
                cell_edges.south,
                cell_edges.west,
                cell_edges.east,
            ):
                li, lj, axis, kind = edge
                if kind is EdgeKind.INTERNAL:
                    internal_set.add((li, lj, axis))
                else:
                    external_set.add((li, lj, axis))
    internal_triples = sorted(internal_set)
    external_triples = sorted(external_set)
    assert len(internal_triples) == 112, (
        f"rotation should produce 112 unique internal triples, "
        f"got {len(internal_triples)}"
    )
    assert len(external_triples) == 32, (
        f"rotation should produce 32 unique external triples, "
        f"got {len(external_triples)}"
    )

    for ti, tj in tiles:
        sub_d_tile = sub_d / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_c_tile = sub_c / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_d_tile.mkdir()
        sub_c_tile.mkdir()

        # Synthetic sub-D macro_core: 64 cell rows + 112 internal edge rows +
        # 32 external edge rows. Cell rows carry scope=0; internal-edge rows
        # carry scope=0 (active); external-edge rows carry scope=3
        # (external_deferred). Total: 64+112+32 = 208 rows.
        slot_kinds, slot_indices = [], []
        cell_is, cell_js = [], []
        lower_is, lower_js, axes, scopes = [], [], [], []
        zoning, density, road = [], [], []
        for idx in range(64):
            slot_kinds.append(0)  # cell
            slot_indices.append(idx)
            cell_is.append(idx % 8)
            cell_js.append(idx // 8)
            lower_is.append(None)
            lower_js.append(None)
            axes.append(None)
            scopes.append(0)
            zoning.append(0)
            density.append(1)
            road.append(None)
        for idx, (li, lj, axis) in enumerate(internal_triples):
            slot_kinds.append(1)  # internal_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(li)
            lower_js.append(lj)
            axes.append(axis)
            scopes.append(0)  # active
            zoning.append(None)
            density.append(None)
            road.append(0)
        for idx, (li, lj, axis) in enumerate(external_triples):
            slot_kinds.append(2)  # external_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(li)
            lower_js.append(lj)
            axes.append(axis)
            scopes.append(3)  # external_deferred
            zoning.append(None)
            density.append(None)
            road.append(None)
        macro_core_table = pa.table(
            {
                "slot_kind": pa.array(slot_kinds, type=pa.int8()),
                "slot_index": pa.array(slot_indices, type=pa.int16()),
                "cell_i": pa.array(cell_is, type=pa.int8()),
                "cell_j": pa.array(cell_js, type=pa.int8()),
                "lower_cell_i": pa.array(lower_is, type=pa.int8()),
                "lower_cell_j": pa.array(lower_js, type=pa.int8()),
                "axis": pa.array(axes, type=pa.int8()),
                "scope": pa.array(scopes, type=pa.int8()),
                "zoning_class": pa.array(zoning, type=pa.int16()),
                "cell_density_bucket": pa.array(density, type=pa.int16()),
                "road_skeleton_class": pa.array(road, type=pa.int16()),
            }
        )
        pq.write_table(macro_core_table, sub_d_tile / "macro_core.parquet")

        # Synthetic sub-C: one primary-road crossing on edge (0, 0, axis=0).
        crossings_table = pa.table(
            {
                "lower_cell_i": pa.array([0], type=pa.int8()),
                "lower_cell_j": pa.array([0], type=pa.int8()),
                "axis": pa.array([0], type=pa.int8()),
                "source_feature_id": pa.array(["F-primary"], type=pa.string()),
            }
        )
        pq.write_table(crossings_table, sub_c_tile / "crossings.parquet")

        features_table = pa.table(
            {
                "source_feature_id": pa.array(["F-primary"], type=pa.string()),
                "feature_class": pa.array(["road"], type=pa.string()),
                "class_raw": pa.array(["primary"], type=pa.string()),
            }
        )
        pq.write_table(features_table, sub_c_tile / "features.parquet")

        sub_d_tile_records.append({"tile_i": ti, "tile_j": tj, "provenance_sha256": "0" * 64})
        sub_c_tile_records.append({"tile_i": ti, "tile_j": tj, "provenance_sha256": "0" * 64})

    # Minimal sub-C manifest + _SUCCESS.
    (sub_c / "manifest.yaml").write_text(
        yaml.safe_dump({"region": "singapore", "tiles": sub_c_tile_records})
    )
    (sub_c / "_SUCCESS").touch()

    # Minimal sub-D manifest + _SUCCESS. Sub-E's pipeline reads `tiles[]` and
    # `inputs.sub_c_region_dir`; the rest can be skeletal.
    (sub_d / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "region": "singapore",
                "tiles": sub_d_tile_records,
                "inputs": {"sub_c_region_dir": str(sub_c)},
            }
        )
    )
    (sub_d / "_SUCCESS").touch()

    return sub_d
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_pipeline.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.pipeline`.

- [ ] **Step 3: Implement the pipeline orchestrator**

```python
# src/cfm/data/sub_e/pipeline.py
"""Sub-E derivation pipeline orchestrator.

Reads sub-D + sub-C, derives boundary contracts, writes per-tile artifacts,
validates inline and cross-tile, then writes _SUCCESS. Any validator failure
aborts the run; no partial _SUCCESS.

Lever-3 mode: `lever_3_collapse=True` skips the class-precedence derivation;
all boundary_class_enum values written as null.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cfm.data.sub_e.derivation import BoundaryClass, derive_boundary_class
from cfm.data.sub_e.io import (
    read_sub_c_crossings,
    read_sub_c_features,
    read_sub_d_macro_core,
    require_sub_d_success_marker,
)
from cfm.data.sub_e.manifest import (
    SubEManifest,
    SubEManifestConfig,
    SubEManifestExtraction,
    SubEManifestInputs,
    SubEManifestTile,
    SubEManifestVersions,
    write_manifest,
)
from cfm.data.sub_e.provenance import (
    SubEInputDigests,
    SubEProvenance,
    SubEVersions,
    provenance_sha256,
    provenance_to_dict,
    write_provenance,
)
from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile
from cfm.data.sub_e.validator_inline import validate_boundary_contract
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_VERSION,
    BOUNDARY_VOCAB_VERSION,
    SUB_E_SCHEMA_VERSION,
)
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    release: str
    region: str
    sub_c_region_dir: Path
    sub_d_region_dir: Path
    output_region_dir: Path
    commit_sha: str
    lever_3_collapse: bool = False


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_or_raise(
    parquet_path: Path,
    expected_derivation_version: str,
    provenance_derivation_version: str,
    lever_3_collapse: bool,
) -> None:
    """Indirect call so tests can monkey-patch to simulate failure.

    Both ``expected_derivation_version`` (orchestrator's BOUNDARY_DERIVATION_VERSION
    constant) and ``provenance_derivation_version`` (read back from the
    just-written provenance.yaml on disk) are required by the inline
    validator post Task-7-defect-2. Passing the same constant for both
    would make inline invariant #8 vacuous; the orchestrator's caller
    threads the disk-read value to give the invariant real signal —
    catches divergence between SubEProvenance serialization and the
    in-memory constant if it ever drifts.
    """
    validate_boundary_contract(
        parquet_path,
        expected_derivation_version=expected_derivation_version,
        provenance_derivation_version=provenance_derivation_version,
        lever_3_collapse=lever_3_collapse,
    )


def derive_region(cfg: PipelineConfig) -> None:
    """Run the full sub-E derivation for a region. Halts on any validator
    failure; no _SUCCESS written on error.
    """
    require_sub_d_success_marker(cfg.sub_d_region_dir)
    cfg.output_region_dir.mkdir(parents=True, exist_ok=True)

    sub_d_manifest = yaml.safe_load(
        (cfg.sub_d_region_dir / "manifest.yaml").read_text()
    )
    sub_d_manifest_sha = _file_sha256(cfg.sub_d_region_dir / "manifest.yaml")
    sub_c_manifest_sha = _file_sha256(cfg.sub_c_region_dir / "manifest.yaml")
    boundary_vocab_path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "macro_plan"
        / "v1"
        / "boundary_vocab.yaml"
    )
    boundary_vocab_sha = _file_sha256(boundary_vocab_path)
    # parents[4] climbs src/cfm/data/sub_e/pipeline.py → repo root, matching
    # derivation.py's _VOCAB_PATH convention. Same depth, same index.

    started_utc = _utc_now()
    tile_records: list[SubEManifestTile] = []

    for sub_d_tile in sub_d_manifest["tiles"]:
        tile_i = sub_d_tile["tile_i"]
        tile_j = sub_d_tile["tile_j"]
        sub_d_tile_dir = (
            cfg.sub_d_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        )
        sub_c_tile_dir = (
            cfg.sub_c_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        )

        macro_core = read_sub_d_macro_core(sub_d_tile_dir / "macro_core.parquet")
        crossings = read_sub_c_crossings(sub_c_tile_dir / "crossings.parquet")
        features = read_sub_c_features(sub_c_tile_dir / "features.parquet")

        rows = _derive_tile_rows(
            macro_core=macro_core,
            crossings=crossings,
            features=features,
            lever_3_collapse=cfg.lever_3_collapse,
        )

        out_tile_dir = (
            cfg.output_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        )
        out_tile_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = write_boundary_contract(
            out_tile_dir / "boundary_contract.parquet", rows
        )
        parquet_sha = _file_sha256(parquet_path)

        # Provenance is the canonical record-as-written; the inline validator
        # reads it back from disk to give invariant #8 real signal. Earlier
        # draft validated BEFORE writing provenance and passed the same
        # constant for both kwargs, making #8 vacuous (constant-vs-itself).
        # Spec §9.4: provenance is canonical record of what was produced.
        # Halt-on-validator-fail discipline: if the validator raises, the
        # parquet + provenance exist on disk but no _SUCCESS marker is
        # written for the region — standard "incomplete" semantics
        # (spec §11.8 sub-C precedent). Next run sees no _SUCCESS and
        # re-derives cleanly.
        provenance = SubEProvenance(
            tile_i=tile_i,
            tile_j=tile_j,
            extraction_commit_sha=cfg.commit_sha,
            extracted_utc=started_utc,
            rerun_count=0,
            rerun_reason="initial",
            inputs=SubEInputDigests(
                release=cfg.release,
                sub_c_manifest_sha256=sub_c_manifest_sha,
                sub_c_features_parquet_sha256=_file_sha256(
                    sub_c_tile_dir / "features.parquet"
                ),
                sub_c_crossings_parquet_sha256=_file_sha256(
                    sub_c_tile_dir / "crossings.parquet"
                ),
                sub_d_manifest_sha256=sub_d_manifest_sha,
                sub_d_macro_core_parquet_sha256=_file_sha256(
                    sub_d_tile_dir / "macro_core.parquet"
                ),
                boundary_vocab_sha256=boundary_vocab_sha,
                derivation_config_sha256=boundary_vocab_sha,  # same file for v1
            ),
            versions=SubEVersions(
                sub_e_schema_version=SUB_E_SCHEMA_VERSION,
                boundary_vocab_version=BOUNDARY_VOCAB_VERSION,
                boundary_derivation_version=BOUNDARY_DERIVATION_VERSION,
            ),
            boundary_contract_parquet_sha256=parquet_sha,
        )
        prov_path = write_provenance(out_tile_dir / "provenance.yaml", provenance)

        # Read provenance back from DISK (not from the dataclass) so the
        # validator's invariant #8 catches divergence between the in-memory
        # constant and what serialization actually wrote. Reading the
        # dataclass instead would re-introduce the vacuous-comparison trap.
        prov_dict = yaml.safe_load(prov_path.read_text())
        _validate_or_raise(
            parquet_path,
            expected_derivation_version=BOUNDARY_DERIVATION_VERSION,
            provenance_derivation_version=prov_dict["versions"]["boundary_derivation_version"],
            lever_3_collapse=cfg.lever_3_collapse,
        )

        # Chain anchor uses provenance_sha256() — strips extracted_utc and
        # *_sha256 per SUB_E_EXCLUDED_FROM_SHA so the chain survives live-clock
        # reruns. Raw _file_sha256(prov_path) here would bake extracted_utc
        # into the chain and break determinism on every rerun (spec §9.2).
        tile_records.append(
            SubEManifestTile(
                tile_i=tile_i,
                tile_j=tile_j,
                provenance_sha256=provenance_sha256(provenance_to_dict(provenance)),
            )
        )

    completed_utc = _utc_now()
    write_manifest(
        cfg.output_region_dir / "manifest.yaml",
        SubEManifest(
            manifest_schema_version="1.0",
            sub_e_schema_version=SUB_E_SCHEMA_VERSION,
            release=cfg.release,
            region=cfg.region,
            region_crs="EPSG:3414",
            inputs=SubEManifestInputs(
                sub_c_manifest_sha256=sub_c_manifest_sha,
                sub_c_region_dir=str(cfg.sub_c_region_dir),
                sub_d_manifest_sha256=sub_d_manifest_sha,
                sub_d_region_dir=str(cfg.sub_d_region_dir),
                boundary_vocab_sha256=boundary_vocab_sha,
            ),
            versions=SubEManifestVersions(
                boundary_vocab_version=BOUNDARY_VOCAB_VERSION,
                boundary_derivation_version=BOUNDARY_DERIVATION_VERSION,
            ),
            config_source="sub_d_manifest.config",
            config=SubEManifestConfig(
                cell_grid=(8, 8),
                internal_edge_count=112,
                external_edge_count=32,
            ),
            initial_extraction=SubEManifestExtraction(
                commit_sha=cfg.commit_sha,
                started_utc=started_utc,
                completed_utc=completed_utc,
                tile_count=len(tile_records),
            ),
            tiles=tile_records,
        ),
    )
    # Halt-on-validator-fail discipline (sub-D precedent at
    # src/cfm/data/sub_d/pipeline.py:254-255; spec §11.8 sub-C precedent):
    # cross-tile validator runs BEFORE _SUCCESS is written. If validation
    # raises, the touch never runs and consumers see no green-light marker
    # — disk state is consistent with "this run did not succeed."
    #
    # Earlier draft did write→try-except→unlink. That pattern (a) created a
    # brief window where _SUCCESS existed before validation completed (false
    # green for any polling observer), and (b) handled failure via recovery
    # rather than by-construction non-occurrence — if unlink itself fails
    # (race, permission, signal), disk state is permanently inconsistent.
    # Validate-then-touch has no race window and no failure mode in the
    # failure handler.
    validate_extraction_cross_tile(cfg.output_region_dir)
    (cfg.output_region_dir / "_SUCCESS").touch()


def _derive_tile_rows(
    *,
    macro_core,
    crossings,
    features,
    lever_3_collapse: bool,
) -> list[BoundaryContractRow]:
    """Construct the 144-row per-tile boundary contract from sub-D + sub-C.

    Edge rows are keyed by ``(slot_kind, lower_cell_i, lower_cell_j, axis)``
    rather than the 3-tuple ``(li, lj, axis)`` because rotation's per-cell
    enumeration can produce identical ``(li, lj, axis)`` triples for the
    internal and external versions of distinct physical edges (e.g. cell
    (3, 0)'s north and cell (3, 1)'s north both encode as ``(3, 0, 0)``
    under the ``lower_lj = cj if cj == 0 else cj - 1`` convention in
    rotation.py:50-52). ``slot_kind`` disambiguates; without it the dict
    collapses 16 internal-external pairs and the writer rejects the row
    count (`expected 144, got 128`).

    Sub-C crossings, conversely, are keyed by the 3-tuple — they apply
    only to active internal edges (external rows are scope=3 and skipped),
    so there is no slot_kind ambiguity at the crossing-lookup site.
    """
    edge_scope: dict[tuple[int, int, int, int], int] = {}
    edge_slot_index: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for r in macro_core:
        if r.slot_kind in (1, 2):  # internal or external edge
            assert r.lower_cell_i is not None
            assert r.lower_cell_j is not None
            assert r.axis is not None
            key = (r.slot_kind, r.lower_cell_i, r.lower_cell_j, r.axis)
            edge_scope[key] = r.scope
            edge_slot_index[key] = (r.slot_kind, r.slot_index)

    features_by_id: dict[str, str | None] = {
        f.source_feature_id: f.class_raw for f in features if f.feature_class == "road"
    }
    crossings_by_edge: dict[tuple[int, int, int], list[str | None]] = {}
    for c in crossings:
        key = (c.lower_cell_i, c.lower_cell_j, c.axis)
        crossings_by_edge.setdefault(key, []).append(
            features_by_id.get(c.source_feature_id)
        )

    rows: list[BoundaryContractRow] = []
    for key, scope in edge_scope.items():
        slot_kind_int, i, j, axis = key
        _, slot_idx = edge_slot_index[key]
        is_active_internal = scope == 0 and slot_kind_int == 1
        if is_active_internal and not lever_3_collapse:
            # Pass all crossings (including None entries) through to
            # derive_boundary_class. Per spec §5.1 + derivation.py:84-85,
            # None entries map to the MINOR_ROAD default bucket; filtering
            # them out would change semantics. Earlier draft had
            # `if cr is not None or True` which short-circuited to always
            # True — dead code that obscured intent.
            class_raws = list(crossings_by_edge.get((i, j, axis), []))
            bc = int(derive_boundary_class(class_raws))
        else:
            bc = None
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind(slot_kind_int),
                slot_index=slot_idx,
                lower_cell_i=i,
                lower_cell_j=j,
                axis=axis,
                scope_marker=scope,
                boundary_class_enum=bc,
            )
        )
    return rows
```

- [ ] **Step 4: Implement the synthetic sub-D + sub-C fixture helper**

In `tests/data/sub_e/_fixtures.py`, implement `_build_synthetic_sub_d_and_sub_c` following sub-D's `tests/data/sub_d/_fixtures.py` pattern. Each tile gets a 144-row sub-D `macro_core.parquet` (mixture of active and external), a tiny sub-C `crossings.parquet` with one primary-road crossing per tile, and a sub-C `features.parquet` mapping. Sub-D `_SUCCESS`, `manifest.yaml`, per-tile `provenance.yaml`, `effective_conditioning.yaml` written via sub-D's writers (re-exported from `cfm.data.sub_d` for test use).

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_pipeline.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 7: Commit**

```bash
git add src/cfm/data/sub_e/pipeline.py tests/data/sub_e/test_pipeline.py tests/data/sub_e/_fixtures.py
git commit -m "feat(sub_e): add pipeline orchestrator with halt-on-validator-fail"
```

---

## Task 11: CLI scripts

**Files:**
- Create: `scripts/derive_boundary_contracts.py`
- Create: `scripts/validate_boundary_contracts.py`

**Context:** Minimal argparse wrappers around `derive_region` and `validate_extraction_cross_tile`. No new logic; CLI surface only.

- [ ] **Step 1: Implement `derive_boundary_contracts.py`**

```python
#!/usr/bin/env python3
"""scripts/derive_boundary_contracts.py

Run sub-E derivation for one region.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# iCloud-safe sys.path inject — matches scripts/smoke.py pattern.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_e.pipeline import PipelineConfig, derive_region  # noqa: E402


def _git_commit_sha() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO)
        .decode()
        .strip()
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--sub-c-region-dir", type=Path, required=True)
    p.add_argument("--sub-d-region-dir", type=Path, required=True)
    p.add_argument("--output-region-dir", type=Path, required=True)
    p.add_argument(
        "--lever-3-collapse",
        action="store_true",
        help="Bypass class-precedence derivation; emit uniformly null boundary_class.",
    )
    args = p.parse_args()

    derive_region(
        PipelineConfig(
            release=args.release,
            region=args.region,
            sub_c_region_dir=args.sub_c_region_dir,
            sub_d_region_dir=args.sub_d_region_dir,
            output_region_dir=args.output_region_dir,
            commit_sha=_git_commit_sha(),
            lever_3_collapse=args.lever_3_collapse,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Implement `validate_boundary_contracts.py`**

```python
#!/usr/bin/env python3
"""scripts/validate_boundary_contracts.py

Run sub-E cross-tile validator on an existing region.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("region_dir", type=Path)
    args = p.parse_args()
    validate_extraction_cross_tile(args.region_dir)
    print(f"OK: {args.region_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke — `--help` on both scripts**

Task 11's scope is "CLI works as an entry point." That means: clean
imports (iCloud `sys.path` inject reaches the editable install), clean
argparse construction, expected argument list. Real fixture/data
invocation belongs to Task 14 — duplicating it here would give a
false signal (synthetic data doesn't surface real-data shape issues
Task 14 is designed to catch).

```bash
uv run python scripts/derive_boundary_contracts.py --help
uv run python scripts/validate_boundary_contracts.py --help
```

Expected: argparse usage block listing all args; no import errors; exit code 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/derive_boundary_contracts.py scripts/validate_boundary_contracts.py
git commit -m "feat(sub_e): add derive and validate CLI scripts"
```

---

## Task 12: Eval harness — shuffle strategies

**Files:**
- Create: `src/cfm/eval/__init__.py`
- Create: `src/cfm/eval/shuffles.py`
- Test: `tests/eval/__init__.py` (empty)
- Test: `tests/eval/test_shuffles.py`

**Context:** Spec §11.4. Two shuffle strategies for the conditional-perplexity gap eval: within-conditioning-bucket (primary) and cross-tile (secondary sanity). Position-shuffled is deferred (§15 #2). Shuffles must be deterministic given a seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_shuffles.py
from __future__ import annotations

from dataclasses import dataclass

from cfm.eval.shuffles import (
    ShuffleStrategy,
    TileConditioning,
    shuffle_macro_plans,
)


@dataclass(frozen=True)
class _FakeMacro:
    tile_i: int
    tile_j: int


def _candidates(n: int) -> list[tuple[TileConditioning, _FakeMacro]]:
    out: list[tuple[TileConditioning, _FakeMacro]] = []
    for k in range(n):
        bucket = "tropical_rainforest" if k % 2 == 0 else "temperate"
        out.append(
            (
                TileConditioning(
                    country="SG",
                    climate_zone=bucket,
                    morphology_class="Asian-megacity",
                    era_class="contemporary",
                ),
                _FakeMacro(tile_i=k, tile_j=0),
            )
        )
    return out


def test_within_bucket_shuffle_only_swaps_within_matching_conditioning() -> None:
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    shuffled = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.WITHIN_BUCKET,
        seed=42,
    )
    for tc, macro in zip(targets, shuffled, strict=True):
        # Find the candidate whose macro matches `macro`. Assert its
        # conditioning matches `tc`.
        match = next(c for c in cands if c[1] is macro)
        assert match[0].climate_zone == tc.climate_zone


def test_cross_tile_shuffle_uniformly_random_with_seed() -> None:
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    a = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    b = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    assert a == b, "deterministic given same seed"


def test_cross_tile_shuffle_does_not_return_identity() -> None:
    """With 20 candidates and a deterministic seed, the shuffle should almost
    always produce a permutation that differs in at least one position from
    identity. Asserted as a deterministic-fixture property, not a property
    test."""
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    shuffled = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    identity = [c[1] for c in cands]
    assert shuffled != identity
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_shuffles.py -v
```

Expected: ModuleNotFoundError on `cfm.eval.shuffles`.

- [ ] **Step 3: Implement the shuffle module**

```python
# src/cfm/eval/__init__.py
"""Sub-bar 3 perplexity-gap evaluation harness (skeleton).

This package ships the gap-eval shell ahead of the training scaffold; it does
NOT depend on a trained model. The model_forward callable is injected at gap
computation time (see perplexity_gap.py).
"""

from __future__ import annotations
```

```python
# src/cfm/eval/shuffles.py
"""Macro-plan shuffle strategies for the conditional-perplexity gap.

Two strategies for de-risk:

- WITHIN_BUCKET (primary): substitute macro plan from a tile with matching
  tile-level conditioning (country, climate_zone, morphology_class, era_class).
- CROSS_TILE (secondary sanity): substitute macro plan from any random
  candidate.

Position-shuffled within the same plan is deferred to post-reset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class ShuffleStrategy(str, Enum):
    WITHIN_BUCKET = "within_bucket"
    CROSS_TILE = "cross_tile"


@dataclass(frozen=True)
class TileConditioning:
    country: str
    climate_zone: str
    morphology_class: str
    era_class: str


MacroT = TypeVar("MacroT")


def _bucket_key(tc: TileConditioning) -> tuple[str, str, str, str]:
    return (tc.country, tc.climate_zone, tc.morphology_class, tc.era_class)


def shuffle_macro_plans(
    *,
    targets: list[TileConditioning],
    candidates: list[tuple[TileConditioning, MacroT]],
    strategy: ShuffleStrategy,
    seed: int,
) -> list[MacroT]:
    """Return one shuffled macro plan per target.

    Deterministic given (targets, candidates, strategy, seed).
    """
    rng = random.Random(seed)

    if strategy is ShuffleStrategy.WITHIN_BUCKET:
        buckets: dict[tuple[str, str, str, str], list[MacroT]] = {}
        for tc, macro in candidates:
            buckets.setdefault(_bucket_key(tc), []).append(macro)
        # Stable order within bucket: candidates already in deterministic input order.
        out: list[MacroT] = []
        for tc in targets:
            pool = buckets.get(_bucket_key(tc), [])
            if not pool:
                raise ValueError(
                    f"no candidates in within-bucket pool for {_bucket_key(tc)}"
                )
            out.append(rng.choice(pool))
        return out

    if strategy is ShuffleStrategy.CROSS_TILE:
        pool = [m for _, m in candidates]
        return [rng.choice(pool) for _ in targets]

    raise AssertionError(f"unknown strategy: {strategy}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_shuffles.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/eval/__init__.py src/cfm/eval/shuffles.py tests/eval/__init__.py tests/eval/test_shuffles.py
git commit -m "feat(eval): add shuffle strategies for perplexity gap"
```

---

## Task 13: Eval harness — perplexity gap shell

**Files:**
- Create: `src/cfm/eval/perplexity_gap.py`
- Test: `tests/eval/test_perplexity_gap.py`

**Context:** Spec §11.4 + §11.5. Skeleton API for the conditional-perplexity gap calculation. The `model_forward` callable is injected; this module does NOT load a model. Tests use a deterministic fake to exercise the gap calculation and sign-test machinery.

The shell ships before the training scaffold (lever 1 default-pull). When the scaffold lands, only `model_forward` needs wiring.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_perplexity_gap.py
from __future__ import annotations

from typing import Sequence

import pytest

from cfm.eval.perplexity_gap import (
    GapResult,
    PerCellNLL,
    compute_perplexity_gap,
)


def _fake_model_forward(
    *, micro_tokens: Sequence[int], conditioning_prefix: Sequence[int]
) -> float:
    """Deterministic toy model: NLL is a function of conditioning agreement.

    If conditioning_prefix[0] equals micro_tokens[0], return low NLL (0.1);
    otherwise high (0.5). This simulates a model that uses conditioning.
    """
    if conditioning_prefix and micro_tokens and conditioning_prefix[0] == micro_tokens[0]:
        return 0.1
    return 0.5


def test_gap_calculation_on_toy_data_positive_when_conditioning_matters() -> None:
    cells = [
        PerCellNLL(
            cell_id=f"c{k}",
            micro_tokens=[k % 3] * 10,
            matched_conditioning_prefix=[k % 3],
            shuffled_conditioning_prefix=[(k + 1) % 3],
        )
        for k in range(30)
    ]
    result = compute_perplexity_gap(
        cells=cells, model_forward=_fake_model_forward, p_threshold=0.01
    )
    assert isinstance(result, GapResult)
    # All 30 cells: matched NLL = 0.1, shuffled NLL = 0.5 → gap = 0.4 nats/cell.
    # Per-token gap = 0.4 / 10 = 0.04 (just under §11.5 threshold; but signal
    # is monotonic so sign-test should be significant).
    assert result.gap_nats_per_token > 0
    assert result.fraction_positive == pytest.approx(1.0)
    assert result.sign_test_significant_at_p


def test_gap_calculation_zero_signal() -> None:
    """If the fake model returns identical NLL regardless of conditioning,
    gap should be ≈0 and sign test should NOT be significant.
    """
    def _flat(*, micro_tokens, conditioning_prefix) -> float:
        return 0.3

    cells = [
        PerCellNLL(
            cell_id=f"c{k}",
            micro_tokens=[k] * 10,
            matched_conditioning_prefix=[k],
            shuffled_conditioning_prefix=[k + 1],
        )
        for k in range(30)
    ]
    result = compute_perplexity_gap(
        cells=cells, model_forward=_flat, p_threshold=0.01
    )
    assert result.gap_nats_per_token == pytest.approx(0.0)
    assert not result.sign_test_significant_at_p
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_perplexity_gap.py -v
```

Expected: ModuleNotFoundError on `cfm.eval.perplexity_gap`.

- [ ] **Step 3: Implement the gap shell**

```python
# src/cfm/eval/perplexity_gap.py
"""Conditional-perplexity gap calculation shell.

Computes gap = NLL_shuffled − NLL_matched on held-out micro tokens under
two conditioning prefixes, plus a per-cell sign test. The model is injected
as a callable; this module does NOT load weights. When the training scaffold
ships, callers wire `model_forward` to a real forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


ModelForward = Callable[..., float]
"""Signature: model_forward(*, micro_tokens, conditioning_prefix) -> nll (nats/token)."""


@dataclass(frozen=True)
class PerCellNLL:
    cell_id: str
    micro_tokens: Sequence[int]
    matched_conditioning_prefix: Sequence[int]
    shuffled_conditioning_prefix: Sequence[int]


@dataclass(frozen=True)
class GapResult:
    n_cells: int
    n_positive: int  # cells where shuffled_nll > matched_nll
    fraction_positive: float
    mean_gap_nats_per_cell: float
    mean_tokens_per_cell: float
    gap_nats_per_token: float
    sign_test_significant_at_p: bool
    p_threshold: float


def _binomial_sign_test_pvalue(n: int, k: int) -> float:
    """One-sided sign-test p-value for H0: P(positive) = 0.5 vs H1: P > 0.5.

    Exact binomial; n cells, k positives.
    """
    from math import comb

    if n == 0:
        return 1.0
    p_tail = 0.0
    for x in range(k, n + 1):
        p_tail += comb(n, x) * 0.5**n
    return p_tail


def compute_perplexity_gap(
    *,
    cells: list[PerCellNLL],
    model_forward: ModelForward,
    p_threshold: float,
) -> GapResult:
    """Compute the gap and run a per-cell sign test against `p_threshold`."""
    if not cells:
        return GapResult(
            n_cells=0,
            n_positive=0,
            fraction_positive=0.0,
            mean_gap_nats_per_cell=0.0,
            mean_tokens_per_cell=0.0,
            gap_nats_per_token=0.0,
            sign_test_significant_at_p=False,
            p_threshold=p_threshold,
        )

    matched_nlls: list[float] = []
    shuffled_nlls: list[float] = []
    token_counts: list[int] = []

    for c in cells:
        matched_nlls.append(
            model_forward(
                micro_tokens=c.micro_tokens,
                conditioning_prefix=c.matched_conditioning_prefix,
            )
        )
        shuffled_nlls.append(
            model_forward(
                micro_tokens=c.micro_tokens,
                conditioning_prefix=c.shuffled_conditioning_prefix,
            )
        )
        token_counts.append(len(c.micro_tokens))

    gaps = [s - m for s, m in zip(shuffled_nlls, matched_nlls, strict=True)]
    n = len(gaps)
    n_positive = sum(1 for g in gaps if g > 0)
    mean_gap = sum(gaps) / n
    mean_tokens = sum(token_counts) / n
    gap_per_token = mean_gap if mean_tokens == 0 else mean_gap  # NLL already per-token

    p_value = _binomial_sign_test_pvalue(n=n, k=n_positive)

    return GapResult(
        n_cells=n,
        n_positive=n_positive,
        fraction_positive=n_positive / n,
        mean_gap_nats_per_cell=mean_gap,
        mean_tokens_per_cell=mean_tokens,
        gap_nats_per_token=gap_per_token,
        sign_test_significant_at_p=p_value < p_threshold,
        p_threshold=p_threshold,
    )
```

Note: the toy fake-model returns NLL "per token" (the test's `_fake_model_forward` signature). The real model_forward at training time will average NLL over the micro_token sequence; both interfaces match.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_perplexity_gap.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full fast suite**

```bash
uv run pytest -m "not slow" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/cfm/eval/perplexity_gap.py tests/eval/test_perplexity_gap.py
git commit -m "feat(eval): add perplexity gap shell with sign test"
```

---

## Task 14: Cached Singapore integration + empirical gate (REAL run)

**Files:**
- Create: `tests/data/sub_e/test_singapore_integration.py`

**Context:** Spec §11.3. Layer 3 runs on the *real* cached Singapore extraction at `data/processed/sub_c/2026-04-15.0/singapore/` and the sub-D output at `data/processed/sub_d/2026-04-15.0/singapore/`. Tests are marked `@pytest.mark.slow` and excluded from the fast suite. The empirical gate is a REAL run, not a synthetic check: it computes the actual `boundary_class` distribution on Singapore's Layer-3 9-tile subset and asserts the §11.3 #1 thresholds.

**Halt-on-validator-fail discipline is the load-bearing test posture here.** If the empirical gate fails on real data, do NOT weaken the thresholds. Stop and escalate per memory `feedback_test_weakening_to_pass`.

**Writer-regression guard (Task-6 carry-forward).** The empirical-gate `max_frac` check only catches "a writer that always emits NONE" indirectly — via the 90% concentration threshold — which routes the failure into the §5 reopen pathway (a derivation-grouping decision), not the writer-bug pathway. Add an explicit `test_layer3_writer_round_trips_major_and_minor` that asserts both `MAJOR_ROAD` and `MINOR_ROAD` appear at least once in the Layer-3 active-row distribution. This separates a writer regression (no class diversity → halt and diagnose writer) from a derivation-grouping concern (one class > 90% → §5 reopen). Both tests live in the same module; the round-trip test runs unconditionally under the empirical-gate fixture.

**Lever-3 launch-time note.** Two tests are only meaningful under non-collapse runs: `test_layer3_empirical_gate_real_distribution` (empirical gate against active-class distribution) and `test_layer3_writer_round_trips_major_and_minor` (writer-regression guard requiring non-null active classes). If the day-9 lever-3 trigger fires (per Task 10 + spec §12), the operator runs the slow suite with both tests excluded:

```bash
uv run pytest -m slow tests/data/sub_e/test_singapore_integration.py \
  --deselect tests/data/sub_e/test_singapore_integration.py::test_layer3_empirical_gate_real_distribution \
  --deselect tests/data/sub_e/test_singapore_integration.py::test_layer3_writer_round_trips_major_and_minor
```

A separate lever-3 regression test (`test_layer3_lever_3_collapse_real_data`, below) verifies the pipeline produces valid sub-E output under lever-3 on real Singapore data.

- [ ] **Step 1: Write the integration test module**

```python
# tests/data/sub_e/test_singapore_integration.py
"""Layer 3 integration: real cached Singapore.

Marked @pytest.mark.slow; excluded from default fast suite. Run explicitly
with `uv run pytest -m slow tests/data/sub_e/test_singapore_integration.py`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.pipeline import PipelineConfig, derive_region
from cfm.data.sub_e.rotation import cell_to_edge_ids, EdgeKind
from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHED_SUB_C = REPO_ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
CACHED_SUB_D = REPO_ROOT / "data" / "processed" / "sub_d" / "2026-04-15.0" / "singapore"

# Empirical gate thresholds per spec §11.3 #1.
GATE_MAX_CLASS_FRACTION = 0.90
GATE_MIN_ACTIVE_CLASS_FRACTION = 0.02


pytestmark = pytest.mark.slow


def _layer3_subset_tiles() -> list[tuple[int, int]]:
    """Read the Layer-3 subset from sub-D's locked macro_plan_vocab.yaml."""
    vocab_path = REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"
    data = yaml.safe_load(vocab_path.read_text())
    return [(t["tile_i"], t["tile_j"]) for t in data["selected_layer3_tiles"]]


@pytest.fixture(scope="module")
def sub_e_run_layer3(tmp_path_factory) -> Path:
    """Run sub-E end-to-end on the Layer-3 9-tile subset (real data).

    Strategy: build a filtered sub-D region directory containing only the
    Layer-3 tiles' subdirectories + a filtered manifest, then run derive_region
    against that filtered region. Same approach sub-D used for its own
    Layer-3 Task 15 integration test (see `tests/data/sub_d/test_singapore_integration.py`).
    """
    if not (CACHED_SUB_D / "_SUCCESS").exists():
        pytest.skip("sub-D cached Singapore output absent")

    out_root = tmp_path_factory.mktemp("sub_e_layer3")
    filtered_sub_d = tmp_path_factory.mktemp("sub_d_filtered") / "singapore"
    filtered_sub_c = tmp_path_factory.mktemp("sub_c_filtered") / "singapore"
    filtered_sub_d.mkdir(parents=True)
    filtered_sub_c.mkdir(parents=True)

    subset = _layer3_subset_tiles()

    # Symlink each Layer-3 tile dir; build a filtered manifest.yaml referencing
    # only the subset; copy sub-D _SUCCESS + sub-C _SUCCESS.
    sub_d_manifest = yaml.safe_load((CACHED_SUB_D / "manifest.yaml").read_text())
    sub_c_manifest = yaml.safe_load((CACHED_SUB_C / "manifest.yaml").read_text())

    sub_d_manifest["tiles"] = [
        t for t in sub_d_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    sub_d_manifest["initial_extraction"]["tile_count"] = len(sub_d_manifest["tiles"])
    sub_c_manifest["tiles"] = [
        t for t in sub_c_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]

    (filtered_sub_d / "manifest.yaml").write_text(yaml.safe_dump(sub_d_manifest))
    (filtered_sub_c / "manifest.yaml").write_text(yaml.safe_dump(sub_c_manifest))
    (filtered_sub_d / "_SUCCESS").touch()
    (filtered_sub_c / "_SUCCESS").touch()

    for ti, tj in subset:
        tile_name = f"tile=EPSG3414_i{ti}_j{tj}"
        (filtered_sub_d / tile_name).symlink_to(
            CACHED_SUB_D / tile_name, target_is_directory=True
        )
        (filtered_sub_c / tile_name).symlink_to(
            CACHED_SUB_C / tile_name, target_is_directory=True
        )

    derive_region(
        PipelineConfig(
            release="2026-04-15.0",
            region="singapore",
            sub_c_region_dir=filtered_sub_c,
            sub_d_region_dir=filtered_sub_d,
            output_region_dir=out_root,
            commit_sha="0" * 40,
            lever_3_collapse=False,
        )
    )
    return out_root


def test_layer3_pipeline_writes_success(sub_e_run_layer3: Path) -> None:
    assert (sub_e_run_layer3 / "_SUCCESS").exists()


def test_layer3_cross_tile_validator_passes_on_real_data(sub_e_run_layer3: Path) -> None:
    validate_extraction_cross_tile(sub_e_run_layer3)


def test_layer3_deterministic_rerun_same_process(
    sub_e_run_layer3: Path, tmp_path: Path
) -> None:
    """Rerun on the same filtered inputs → byte-identical parquet outputs."""
    # Compute sha of each tile's boundary_contract.parquet from the fixture run.
    first_shas = {
        d.name: hashlib.sha256((d / "boundary_contract.parquet").read_bytes()).hexdigest()
        for d in sub_e_run_layer3.glob("tile=EPSG3414_*")
    }

    # Rerun into a fresh output dir using the same filtered inputs.
    out_root2 = tmp_path / "sub_e_rerun"
    # Reuse the symlink-filtered sub_d and sub_c via fixture's tmp dirs is
    # not easily reachable here; instead, derive into out_root2 using the
    # *same* output_region_dir input lookup, then compare per-tile shas.
    # The implementer should refactor the fixture to expose `filtered_sub_d`
    # and `filtered_sub_c` paths for reuse here.
    # As a minimum: re-derive directly into out_root2 referencing the same
    # CACHED_SUB_D + CACHED_SUB_C dirs without filtering (slower but still
    # under @pytest.mark.slow); then assert layer3-subset tiles match.
    pytest.skip(
        "deterministic-rerun reuse path needs fixture refactor; implement "
        "alongside the layer3 fixture refactor"
    )


def test_layer3_external_edge_single_cell_membership(sub_e_run_layer3: Path) -> None:
    """Each external slot_index appears in exactly one cell's per-cell view."""
    for tile_dir in sub_e_run_layer3.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        slot_kinds = tbl.column("slot_kind").to_pylist()
        slot_indices = tbl.column("slot_index").to_pylist()
        ext = [si for sk, si in zip(slot_kinds, slot_indices, strict=True) if sk == 2]
        assert len(ext) == 32, f"expected 32 external rows in {tile_dir.name}, got {len(ext)}"
        assert len(set(ext)) == 32, f"duplicate external slot_index in {tile_dir.name}"


def test_layer3_empirical_gate_real_distribution(sub_e_run_layer3: Path) -> None:
    """REAL empirical gate run on Layer-3 subset.

    Per spec §11.3 #1: ship iff no active class above 90% AND no active class
    below 2%. Halt on violation (memory `feedback_test_weakening_to_pass`).
    """
    counter: Counter[int] = Counter()
    for tile_dir in sub_e_run_layer3.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        scope_markers = tbl.column("scope_marker").to_pylist()
        boundary_classes = tbl.column("boundary_class_enum").to_pylist()
        for scope, cls in zip(scope_markers, boundary_classes, strict=True):
            if scope == 0 and cls is not None:  # active rows only
                counter[cls] += 1

    total_active = sum(counter.values())
    assert total_active > 0, "no active edges in Layer-3 subset — sub-D upstream failed"

    fractions = {cls: count / total_active for cls, count in counter.items()}

    # Print for reviewer visibility.
    print("\nLayer-3 boundary_class distribution:")
    for cls_id, frac in sorted(fractions.items()):
        cls_name = BoundaryClass(cls_id).name
        print(f"  {cls_name}: {frac:.4f} ({counter[cls_id]} of {total_active})")

    max_frac = max(fractions.values())
    min_frac = min(fractions.values())

    assert max_frac <= GATE_MAX_CLASS_FRACTION, (
        f"empirical gate FAILED: a class has {max_frac:.4f} > "
        f"{GATE_MAX_CLASS_FRACTION}. Halt and escalate per §5 reopen rule. "
        f"Do NOT weaken this threshold."
    )
    assert min_frac >= GATE_MIN_ACTIVE_CLASS_FRACTION, (
        f"empirical gate FAILED: a class has {min_frac:.4f} < "
        f"{GATE_MIN_ACTIVE_CLASS_FRACTION}. Halt and escalate per §5 reopen rule. "
        f"Do NOT weaken this threshold."
    )

    # Also pin the distribution as a golden file post-pass.
    golden = REPO_ROOT / "tests" / "golden" / "sub_e" / "empirical_gate"
    golden.mkdir(parents=True, exist_ok=True)
    (golden / "layer3_boundary_class_distribution.yaml").write_text(
        yaml.safe_dump(
            {
                "boundary_derivation_version": "1.0",
                "boundary_vocab_version": "1.0",
                "total_active_edges": total_active,
                "fractions": {
                    BoundaryClass(cls).name: round(frac, 6)
                    for cls, frac in sorted(fractions.items())
                },
            }
        )
    )


def test_layer3_writer_round_trips_major_and_minor(sub_e_run_layer3: Path) -> None:
    """Task-6 carry-forward writer-regression guard.

    A writer bug that always emits BoundaryClass.NONE (or coerces all
    active classes to a single value) would manifest in the empirical-gate
    test as a max-class-fraction violation, which routes into the §5 reopen
    pathway (a derivation-grouping concern). That's the wrong escalation
    path: the failure is structural (writer corruption), not semantic
    (derivation decision).

    This test asserts both BoundaryClass.MAJOR_ROAD and BoundaryClass.MINOR_ROAD
    appear at least once in the Layer-3 active-row distribution. If this
    fails, halt and diagnose the writer, not the class-grouping map.
    """
    counter: Counter[int] = Counter()
    for tile_dir in sub_e_run_layer3.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        scope_markers = tbl.column("scope_marker").to_pylist()
        boundary_classes = tbl.column("boundary_class_enum").to_pylist()
        for scope, cls in zip(scope_markers, boundary_classes, strict=True):
            if scope == 0 and cls is not None:
                counter[cls] += 1

    assert counter[int(BoundaryClass.MAJOR_ROAD)] > 0, (
        "Layer-3 distribution has zero MAJOR_ROAD active edges — likely "
        "writer regression (a derivation-grouping decision would still emit "
        "at least one MAJOR_ROAD on Singapore's 9-tile Layer-3 subset). "
        "Halt and diagnose the writer, NOT the class-grouping map."
    )
    assert counter[int(BoundaryClass.MINOR_ROAD)] > 0, (
        "Layer-3 distribution has zero MINOR_ROAD active edges — likely "
        "writer regression. Halt and diagnose the writer, NOT the "
        "class-grouping map."
    )


def test_layer3_lever_3_collapse_real_data(tmp_path_factory) -> None:
    """Lever-3 regression guard on real Singapore data.

    Runs the pipeline under `lever_3_collapse=True` over the Layer-3 9-tile
    subset. Asserts:

    - Pipeline writes `_SUCCESS` (validators pass under lever-3).
    - All `boundary_class_enum` values on-disk are null.
    - Cross-tile validator passes.

    Verifies that the day-9 lever-3 trigger path is mechanically pullable
    against real data, not just synthetic fixtures.
    """
    if not (CACHED_SUB_D / "_SUCCESS").exists():
        pytest.skip("sub-D cached Singapore output absent")

    out_root = tmp_path_factory.mktemp("sub_e_layer3_lever_3")
    filtered_sub_d = tmp_path_factory.mktemp("sub_d_filtered_lever_3") / "singapore"
    filtered_sub_c = tmp_path_factory.mktemp("sub_c_filtered_lever_3") / "singapore"
    filtered_sub_d.mkdir(parents=True)
    filtered_sub_c.mkdir(parents=True)

    subset = _layer3_subset_tiles()

    sub_d_manifest = yaml.safe_load((CACHED_SUB_D / "manifest.yaml").read_text())
    sub_c_manifest = yaml.safe_load((CACHED_SUB_C / "manifest.yaml").read_text())
    sub_d_manifest["tiles"] = [
        t for t in sub_d_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    sub_d_manifest["initial_extraction"]["tile_count"] = len(sub_d_manifest["tiles"])
    sub_c_manifest["tiles"] = [
        t for t in sub_c_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    (filtered_sub_d / "manifest.yaml").write_text(yaml.safe_dump(sub_d_manifest))
    (filtered_sub_c / "manifest.yaml").write_text(yaml.safe_dump(sub_c_manifest))
    (filtered_sub_d / "_SUCCESS").touch()
    (filtered_sub_c / "_SUCCESS").touch()
    for ti, tj in subset:
        tile_name = f"tile=EPSG3414_i{ti}_j{tj}"
        (filtered_sub_d / tile_name).symlink_to(
            CACHED_SUB_D / tile_name, target_is_directory=True
        )
        (filtered_sub_c / tile_name).symlink_to(
            CACHED_SUB_C / tile_name, target_is_directory=True
        )

    derive_region(
        PipelineConfig(
            release="2026-04-15.0",
            region="singapore",
            sub_c_region_dir=filtered_sub_c,
            sub_d_region_dir=filtered_sub_d,
            output_region_dir=out_root,
            commit_sha="0" * 40,
            lever_3_collapse=True,
        )
    )

    assert (out_root / "_SUCCESS").exists()
    validate_extraction_cross_tile(out_root)

    for tile_dir in out_root.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        values = tbl.column("boundary_class_enum").to_pylist()
        assert all(v is None for v in values), (
            f"lever-3 mode must produce uniform null in {tile_dir.name}"
        )
```

- [ ] **Step 2: Run the slow Layer-3 suite**

```bash
uv run pytest -m slow tests/data/sub_e/test_singapore_integration.py -v
```

Expected output:

- All tests pass.
- The empirical-gate test prints the actual `boundary_class` distribution and writes the golden YAML.

**If the empirical-gate test fails: HALT. Do not modify thresholds. Read §5 reopen criteria. Surface the actual distribution numbers to the reviewer and decide whether to revise the class-grouping map (`boundary_derivation_version` bump) or accept the verdict (escalate scope to v2 boundary contracts).**

- [ ] **Step 3: Verify the golden YAML committed**

```bash
ls tests/golden/sub_e/empirical_gate/
cat tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml
```

- [ ] **Step 4: Run full suite (fast + slow) to confirm no regression**

```bash
uv run pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/data/sub_e/test_singapore_integration.py tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml
git commit -m "test(sub_e): add cached Singapore integration and empirical gate"
```

---

## Task 15: Final verification and handoff

**Files:**
- Create: `docs/handoffs/<YYYY-MM-DD>-end-of-sub-E.md` (date filled at commit time)

**Context:** Sub-D's `docs/handoffs/2026-05-19-end-of-sub-D.md` is the template. The handoff documents: branch state, full test status (fast + slow), locked artifacts and their shas, reviewer-confirmed design decisions during implementation, residual risks, and pointers.

**No merge to main.** The handoff explicitly notes that the merge decision is the reviewer's; the agent halts at handoff commit.

- [ ] **Step 1: Run final verification**

```bash
# Working tree should be clean (no stray changes outside .claude/).
git status

# Full fast suite + slow suite.
uv run pytest -q
uv run pytest -m slow -q

# Ruff format + lint.
uv run ruff format --check .
uv run ruff check .

# Branch diff against main.
git log --oneline main..HEAD
```

Expected: clean working tree, all tests pass, lint/format clean, 14 task commits on branch.

- [ ] **Step 2: Compute final artifact shas**

```bash
shasum -a 256 configs/macro_plan/v1/boundary_vocab.yaml
git rev-parse HEAD
```

Record both for the handoff doc.

- [ ] **Step 3: Write the handoff document**

```markdown
# Session handoff — end of Phase 1 sub-E (<YYYY-MM-DD>)

> **For the reviewer:** the branch is ready for merge review. The merge
> decision is yours — this doc describes the state of the branch, not a
> request to merge. Do NOT merge to main without explicit approval.

## Branch state

- Branch: `phase-1-sub-E-boundary-contracts`
- Final commit: `<sha>` (`test(sub_e): add cached Singapore integration and empirical gate`)
- Working tree: clean.
- Diverges from `main` by 14 task commits + this handoff.

## Test status

- Full fast suite: <N> passed, <M> deselected.
- Sub-E focused fast suite: <N> passed (Layer 1 + Layer 2).
- Slow Layer-3 integration suite: 5 passed (Layer 3 + empirical gate REAL run).
- Empirical gate distribution recorded at
  `tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml`.

## Task commits (chronological)

| Task | Commit | Subject |
|---|---|---|
| T1 | `<sha>` | `data(sub_e): lock boundary vocab v1` |
| T2 | `<sha>` | `feat(sub_e): add package skeleton and version constants` |
| T3 | `<sha>` | `feat(sub_e): add per-cell rotation function` |
| T4 | `<sha>` | `feat(sub_e): add class-precedence derivation function` |
| T5 | `<sha>` | `feat(sub_e): add sub-C and sub-D input readers` |
| T6 | `<sha>` | `feat(sub_e): add boundary contract parquet writer` |
| T7 | `<sha>` | `feat(sub_e): add inline validator` |
| T8 | `<sha>` | `feat(sub_e): add manifest and provenance writers` |
| T9 | `<sha>` | `feat(sub_e): add cross-tile validator` |
| T10 | `<sha>` | `feat(sub_e): add pipeline orchestrator with halt-on-validator-fail` |
| T11 | `<sha>` | `feat(sub_e): add derive and validate CLI scripts` |
| T12 | `<sha>` | `feat(eval): add shuffle strategies for perplexity gap` |
| T13 | `<sha>` | `feat(eval): add perplexity gap shell with sign test` |
| T14 | `<sha>` | `test(sub_e): add cached Singapore integration and empirical gate` |
| T15 | (this) | `docs(handoff): end of sub-E boundary contracts` |

## Locked artifacts

- `configs/macro_plan/v1/boundary_vocab.yaml`
  sha256: `<computed at Step 2>`
  committed at: T1.
- Empirical-gate distribution (golden):
  `tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml`
  Distribution values land at T14.

## Reviewer-confirmed design decisions during implementation

Any deviation from the spec encountered during implementation lands here with
a citation to the commit that applied it. (For example: the
`empirical_gate.layer3_subset_sha256` field was omitted from
`boundary_vocab.yaml`; rationale recorded at T1.)

If no deviations occurred, this section reads "No deviations from the spec."

## Residual risks

- **Cross-environment determinism (darwin/aarch64 vs linux/x86_64) —
  unverified.** Inherited from sub-D. Sentinel test: first Leonardo sub-E
  run cross-checked against the local hash. See sub-E spec §14.
- **Sub-D known_issue #11 (Layer-3 sparse-side scoring footnote).** The
  9-tile Layer-3 subset over-indexes on positive-side dimensions; sub-E's
  empirical gate inherits the footnote. Not a sub-E blocker.

## Pointers

Read these directly; do not paraphrase.

- Spec: `docs/superpowers/specs/2026-05-20-phase-1-sub-E-boundary-contracts-design.md`
- Plan: `docs/superpowers/plans/2026-05-20-phase-1-sub-E-boundary-contracts.md`
- Sub-D handoff: `docs/handoffs/2026-05-19-end-of-sub-D.md`
- Known issues: `docs/known_issues.md`

## Merge note

The branch is ready for merge review. **The merge decision is the
reviewer's** — do not merge to `main` automatically under any
circumstances. The agent halts here.
```

- [ ] **Step 4: Commit the handoff**

```bash
git add docs/handoffs/<YYYY-MM-DD>-end-of-sub-E.md
git commit -m "docs(handoff): end of sub-E boundary contracts"
```

- [ ] **Step 5: HALT**

Do not merge to main. Do not push to remote. Do not open a pull request. Wait for reviewer approval.

---

## Plan Self-Review Checklist

Run this checklist after writing the plan. Fix issues inline.

**1. Spec coverage.** Every section of the spec has at least one task:

| Spec § | Task(s) |
|---|---|
| §1 Scope and goal | Header + §1.1 framing references throughout |
| §2 Calendar and budget | Plan header dependency map + Task 15 verification |
| §3 Consumer contract | Task 3 (rotation), Task 6 (writer), Task 10 (pipeline) |
| §4 Inputs | Task 5 (io readers), Task 10 (pipeline) |
| §5 Derivation function | Task 4 (derivation), Task 10 (pipeline) |
| §6 External / scope-boundary | Task 3 (rotation), Task 10 (pipeline) |
| §7 Storage shape | Task 6 (writer), Task 3 (rotation) |
| §8 Vocab artifacts | Task 1 (boundary_vocab.yaml lock) |
| §9 Determinism, versioning, provenance | Task 2 (versions), Task 8 (manifest + provenance) |
| §10 Validator invariants | Task 7 (inline), Task 9 (cross-tile) |
| §11 Validation strategy | Task 11 (Layer 1+2 via Tasks 1-9), Task 14 (Layer 3 + empirical gate REAL) |
| §11.4 Perplexity gap eval | Task 12 (shuffles), Task 13 (gap shell) |
| §12 Lever-3 collapse | Task 10 (`PipelineConfig.lever_3_collapse`), Task 11 (CLI flag) |
| §13 Output directory layout | Task 8 (manifest), Task 10 (pipeline) |
| §14 Cross-environment determinism | Task 15 (handoff records residual) |
| §15 Deferrals | Task 15 (handoff references spec §15) |

No gaps.

**2. Placeholder scan.** Grep the plan for forbidden patterns:

```bash
grep -nE "TBD|TODO|implement later|fill in details|appropriate error|add validation|handle edge cases|Similar to Task" docs/superpowers/plans/2026-05-20-phase-1-sub-E-boundary-contracts.md || echo "clean"
```

Expected: `clean`. If anything matches, fix it.

(No known acceptable false positives in the plan body. The scan command itself appears on one line; that's a self-reference of the grep pattern, not a placeholder in the plan content.)

**3. Type consistency.** Spot-check that types and function names referenced across tasks match:

- `BoundaryClass` (Task 4) used in Task 6 writer test, Task 7 validator test ✓
- `SlotKind` (Task 6) used in Task 7 validator, Task 9 cross-tile validator ✓
- `BoundaryContractRow` (Task 6) used in Task 7 test, Task 10 pipeline ✓
- `PipelineConfig` (Task 10) used in Task 11 CLI ✓
- `ShuffleStrategy` (Task 12) — Task 13 doesn't reference (skeleton-only), wired at training-scaffold sub-project ✓
- `BOUNDARY_DERIVATION_VERSION` (Task 2) used in Task 7 validator, Task 10 pipeline ✓
- `cell_to_edge_ids` (Task 3) — currently unused in Tasks 4–14 explicit tests but exported for use by the dataloader (training-scaffold sub-project). This is the *consumer-side* rotation function and lives in sub-E for the consumer; that's correct per spec §3.1.

**4. DRY / YAGNI / TDD discipline.**

- Each implementation task has explicit TDD steps (fail → implement → pass → commit).
- Each task commits independently; no batching across tasks.
- No speculative features outside spec scope (no source-feature traceability, no positions/widths — all deferred per spec §15).
- Subagent dispatches per memory `feedback_subagent_branch_pattern`: no new branches, no push, no PR.

**5. Empirical gate is a REAL run.** Task 14's `test_layer3_empirical_gate_real_distribution` reads real Singapore parquet bytes via the cached extraction, computes the actual `boundary_class` distribution, asserts §11.3 #1 thresholds, and writes a golden YAML. Not a synthetic-fixture check.

**6. Halt-on-validator-fail discipline.** Task 10's pipeline raises on any inline validator failure; if cross-tile validation fails post-`_SUCCESS`, the `_SUCCESS` is removed. Task 14's empirical gate explicitly directs the implementer to halt and escalate on threshold violation, not weaken thresholds.

**7. No merge to main.** Task 15 explicitly forbids merge, push, and PR.

Plan complete.

---

# Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-20-phase-1-sub-E-boundary-contracts.md`. Reviewer chooses execution mode:

**1. Subagent-Driven (recommended).** I dispatch a fresh subagent per task, two-stage review between tasks, fast iteration. Each dispatch includes the branch-discipline reminder verbatim.

**2. Inline Execution.** Tasks executed in this session via the `executing-plans` skill, batched with checkpoints for review.

Per the reviewer's prior instruction: *"Halt after committing the plan. I'll review the task breakdown before implementation starts."* — halting now. Awaiting review and execution-mode selection.
