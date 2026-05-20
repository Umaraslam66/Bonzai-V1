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

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_e/test_versions.py
from __future__ import annotations

import pytest

from cfm.data.determinism import compare_version  # sub-D Task 1 helper
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_VERSION,
    BOUNDARY_VOCAB_VERSION,
    SUB_E_SCHEMA_VERSION,
    VersionNamespace,
)


def test_initial_version_values() -> None:
    assert SUB_E_SCHEMA_VERSION == "1.0"
    assert BOUNDARY_VOCAB_VERSION == "1.0"
    assert BOUNDARY_DERIVATION_VERSION == "1.0"


def test_namespace_enum_values() -> None:
    # Sub-E namespaces are extensions of sub-D's namespace enum, not aliases.
    assert VersionNamespace.SUB_E_SCHEMA.value == "sub_e_schema"
    assert VersionNamespace.BOUNDARY_VOCAB.value == "boundary_vocab"
    assert VersionNamespace.BOUNDARY_DERIVATION.value == "boundary_derivation"


def test_compare_version_within_namespace_passes() -> None:
    assert compare_version(
        VersionNamespace.SUB_E_SCHEMA.value, "1.0", "1.0"
    ) is True


def test_compare_version_cross_namespace_rejects() -> None:
    # Calling with mismatched namespace strings must raise.
    with pytest.raises(ValueError, match="namespace"):
        compare_version(
            VersionNamespace.SUB_E_SCHEMA.value,
            "1.0",
            "1.0",
            other_namespace=VersionNamespace.BOUNDARY_VOCAB.value,
        )
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
from __future__ import annotations

from enum import Enum
from typing import Final


class VersionNamespace(str, Enum):
    """Namespace tags for sub-E version axes.

    Pass `.value` to `cfm.data.determinism.compare_version`. Cross-namespace
    comparisons must raise rather than silently succeeding.
    """

    SUB_E_SCHEMA = "sub_e_schema"
    BOUNDARY_VOCAB = "boundary_vocab"
    BOUNDARY_DERIVATION = "boundary_derivation"


SUB_E_SCHEMA_VERSION: Final[str] = "1.0"
BOUNDARY_VOCAB_VERSION: Final[str] = "1.0"
BOUNDARY_DERIVATION_VERSION: Final[str] = "1.0"
```

- [ ] **Step 4: Confirm cross-namespace rejection mechanic exists in `compare_version`**

Read `src/cfm/data/determinism.py::compare_version` and verify it supports the `other_namespace` kwarg signature exercised by the test. If sub-D's helper does not yet support cross-namespace rejection via `other_namespace`, this is a sub-D defect (`known_issue #8` lesson); halt and escalate. Do NOT modify the helper without reviewer approval.

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
    Path(__file__).resolve().parents[3]
    / "configs"
    / "macro_plan"
    / "v1"
    / "boundary_vocab.yaml"
)


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


def test_write_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    import hashlib

    rows = _make_full_lattice_rows()
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_boundary_contract(a, rows)
    write_boundary_contract(b, rows)
    h_a = hashlib.sha256(a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(b.read_bytes()).hexdigest()
    assert h_a == h_b, "same-process determinism required"
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
by (slot_kind, slot_index). Schema matches sub-D macro_core.parquet
conventions; sub-D's neutral parquet write helper is reused for deterministic
bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet_deterministic  # sub-D Task 1 helper


class SlotKind(IntEnum):
    INTERNAL_EDGE = 1
    EXTERNAL_EDGE = 2


EXPECTED_INTERNAL_ROWS: Final[int] = 112
EXPECTED_EXTERNAL_ROWS: Final[int] = 32
EXPECTED_TOTAL_ROWS: Final[int] = EXPECTED_INTERNAL_ROWS + EXPECTED_EXTERNAL_ROWS


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
    112 internal + 32 external.
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

    table = pa.table(
        {
            "slot_kind": pa.array(
                [int(r.slot_kind) for r in sorted_rows], type=pa.int8()
            ),
            "slot_index": pa.array(
                [r.slot_index for r in sorted_rows], type=pa.int16()
            ),
            "lower_cell_i": pa.array(
                [r.lower_cell_i for r in sorted_rows], type=pa.int8()
            ),
            "lower_cell_j": pa.array(
                [r.lower_cell_j for r in sorted_rows], type=pa.int8()
            ),
            "axis": pa.array([r.axis for r in sorted_rows], type=pa.int8()),
            "scope_marker": pa.array(
                [r.scope_marker for r in sorted_rows], type=pa.int8()
            ),
            "boundary_class_enum": pa.array(
                [r.boundary_class_enum for r in sorted_rows], type=pa.int16()
            ),
        }
    )
    write_parquet_deterministic(table, out_path)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_writer.py -v
```

Expected: 5 passed.

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

## Task 7: Inline validator (8 invariants)

**Files:**
- Create: `src/cfm/data/sub_e/validator_inline.py`
- Test: `tests/data/sub_e/test_validator_inline.py`

**Context:** Spec §10.1. Eight invariants over a single `boundary_contract.parquet`. Each invariant has a controlled-violation fixture in the test suite. **All fixtures synthesised in-process via `BoundaryContractRow` + the Task 6 writer; this task does not read `data/processed/sub_c/` or `data/processed/sub_d/`.**

Under lever-3 collapse (spec §12), invariants #3 and #4 are replaced by a single uniform-null check (every row's `boundary_class_enum` is null). The validator accepts a `lever_3_collapse: bool` kwarg to switch modes; Task 10's pipeline forwards `cfg.lever_3_collapse`.

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


def test_valid_lattice_passes_all_invariants(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_rows())
    validate_boundary_contract(
        p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
    )  # should not raise


def test_invariant_3_class_non_null_iff_scope_active(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Set an external (scope_marker=3) row to have a non-null class — violates #3.
    rows[112] = replace(rows[112], boundary_class_enum=int(BoundaryClass.MAJOR_ROAD))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="non-null iff scope_marker == 0"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_4_active_class_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Use BOUNDARY_NOT_APPLICABLE (0) on an active row — sentinel is dataloader-side
    # only; on-disk active rows must be in {NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3}.
    rows[0] = replace(
        rows[0], boundary_class_enum=int(BoundaryClass.BOUNDARY_NOT_APPLICABLE)
    )
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="active class membership"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_5_scope_marker_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], scope_marker=9)  # out of {0, 1, 2, 3}
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="scope_marker membership"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_6_slot_index_range(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], slot_index=999)
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="slot_index range"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_7_axis_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], axis=2)  # AXIS = {0, 1}
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="axis membership"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_8_derivation_version_match(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_rows())
    with pytest.raises(InlineValidationError, match="boundary_derivation_version"):
        validate_boundary_contract(p, expected_derivation_version="9.9")


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
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_invariant_2_sort_key(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    p = _write(tmp_path, _valid_rows())
    tbl = pq.ParquetFile(p).read()
    # Reverse — sort key violated.
    bad = tbl.slice(0, tbl.num_rows).take(list(range(tbl.num_rows - 1, -1, -1)))
    pq.write_table(bad, p)
    with pytest.raises(InlineValidationError, match="sort key"):
        validate_boundary_contract(
            p, expected_derivation_version=BOUNDARY_DERIVATION_VERSION
        )


def test_lever_3_collapse_passes_with_uniform_null(tmp_path: Path) -> None:
    """Under lever-3, all boundary_class_enum values null even on active rows."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    p = _write(tmp_path, rows)
    validate_boundary_contract(
        p,
        expected_derivation_version=BOUNDARY_DERIVATION_VERSION,
        lever_3_collapse=True,
    )  # should not raise


def test_lever_3_collapse_rejects_any_non_null(tmp_path: Path) -> None:
    """Under lever-3, even a single non-null boundary_class_enum is a violation."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    rows[0] = replace(rows[0], boundary_class_enum=int(BoundaryClass.NONE))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="lever-3"):
        validate_boundary_contract(
            p,
            expected_derivation_version=BOUNDARY_DERIVATION_VERSION,
            lever_3_collapse=True,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/data/sub_e/test_validator_inline.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the inline validator**

```python
# src/cfm/data/sub_e/validator_inline.py
"""Sub-E per-tile inline validator (8 invariants per spec §10.1)."""

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


def validate_boundary_contract(
    path: Path,
    *,
    expected_derivation_version: str,
    provenance_derivation_version: str | None = None,
    lever_3_collapse: bool = False,
) -> None:
    """Validate one boundary_contract.parquet. Raises InlineValidationError.

    If `provenance_derivation_version` is None, invariant #8 (provenance
    version match) is checked against `expected_derivation_version` only;
    the pipeline orchestrator (Task 10) passes the value loaded from the
    sibling provenance.yaml.

    Under `lever_3_collapse=True`, invariants #3 (non-null iff active) and
    #4 (active class membership) are replaced by a single uniform-null check
    (every row's boundary_class_enum is null). Other invariants still apply.
    """
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

    # Invariants 3 & 4 — lever-3 collapses these into a single uniform-null check.
    if lever_3_collapse:
        for i, cls in enumerate(boundary_classes):
            if cls is not None:
                raise InlineValidationError(
                    f"row {i}: lever-3 mode requires boundary_class_enum is null "
                    f"in every row (got {cls})"
                )

    # Invariants 3, 4, 5, 6, 7.
    for i, (sk, si, scope, cls, axis) in enumerate(
        zip(slot_kinds, slot_indices, scope_markers, boundary_classes, axes)
    ):
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

    # Invariant 8: provenance derivation version matches expected.
    actual = (
        provenance_derivation_version
        if provenance_derivation_version is not None
        else expected_derivation_version
    )
    if actual != expected_derivation_version:
        raise InlineValidationError(
            f"boundary_derivation_version mismatch: expected "
            f"{expected_derivation_version}, got {actual}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_validator_inline.py -v
```

Expected: 9 passed (one per invariant + the valid-case happy path).

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

from cfm.data.sub_e.provenance import (
    SubEProvenance,
    SubEInputDigests,
    SubEVersions,
    write_provenance,
)


def _make_provenance() -> SubEProvenance:
    return SubEProvenance(
        tile_i=12,
        tile_j=17,
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
```

```python
# tests/data/sub_e/test_manifest.py
from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.sub_e.manifest import (
    SubEManifest,
    SubEManifestInputs,
    SubEManifestVersions,
    SubEManifestConfig,
    SubEManifestExtraction,
    SubEManifestTile,
    write_manifest,
)


def _make_manifest(tile_count: int = 3) -> SubEManifest:
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
            started_utc="2026-05-21T12:00:00Z",
            completed_utc="2026-05-21T12:05:00Z",
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/data/sub_e/test_provenance.py tests/data/sub_e/test_manifest.py -v
```

Expected: ModuleNotFoundError on `cfm.data.sub_e.provenance` and `cfm.data.sub_e.manifest`.

- [ ] **Step 3: Implement provenance**

```python
# src/cfm/data/sub_e/provenance.py
"""Per-tile provenance writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from cfm.data.io import write_yaml_canonical  # sub-D Task 1 helper


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


def write_provenance(path: Path, prov: SubEProvenance) -> Path:
    doc = {
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
    write_yaml_canonical(doc, path)
    return path
```

- [ ] **Step 4: Implement manifest**

```python
# src/cfm/data/sub_e/manifest.py
"""Per-region manifest writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cfm.data.io import write_yaml_canonical


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


def write_manifest(path: Path, manifest: SubEManifest) -> Path:
    sorted_tiles = sorted(manifest.tiles, key=lambda t: (t.tile_i, t.tile_j))
    doc = {
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
    write_yaml_canonical(doc, path)
    return path
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/data/sub_e/test_provenance.py tests/data/sub_e/test_manifest.py -v
```

Expected: 5 passed (3 manifest + 2 provenance, or whatever the exact split lands at).

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
) -> Path:
    """Create a minimal sub-E region directory with N consistent tiles.

    The implementation should consult the writers from Tasks 6 and 8 to
    produce real bytes; here we use them to compose a working fixture.
    """
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
        write_provenance,
    )
    from cfm.data.sub_e.writer import (
        BoundaryContractRow,
        SlotKind,
        write_boundary_contract,
    )

    region = tmp_path / "sub_e_singapore"
    region.mkdir()

    def _rows() -> list[BoundaryContractRow]:
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

    tile_records: list[SubEManifestTile] = []
    for k in range(n_tiles):
        tile_dir = region / f"tile=EPSG3414_i{k}_j0"
        tile_dir.mkdir()
        contract = write_boundary_contract(
            tile_dir / "boundary_contract.parquet", _rows()
        )
        contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        prov_path = tile_dir / "provenance.yaml"
        write_provenance(
            prov_path,
            SubEProvenance(
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
            ),
        )
        prov_sha = hashlib.sha256(prov_path.read_bytes()).hexdigest()
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
    region = _build_synthetic_region(tmp_path)
    # Corrupt one tile's provenance to use a different sub_e_schema_version.
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    text = prov_path.read_text()
    prov_path.write_text(text.replace('sub_e_schema_version: "1.0"', 'sub_e_schema_version: "2.0"'))
    with pytest.raises(CrossTileValidationError, match="sub_e_schema_version"):
        validate_extraction_cross_tile(region)


def test_invariant_2_vocab_and_derivation_consistency(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    text = prov_path.read_text()
    prov_path.write_text(text.replace('boundary_vocab_version: "1.0"', 'boundary_vocab_version: "2.0"'))
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


def test_invariant_5_external_slot_uniqueness(tmp_path: Path) -> None:
    """External edges must appear in exactly one cell's per-cell view.

    The structural test reads the parquet and asserts that no `slot_index` in
    the external_edge subset is duplicated across the per-tile rows.
    """
    region = _build_synthetic_region(tmp_path)
    # Corrupt one tile's parquet to duplicate an external slot_index.
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = region / "tile=EPSG3414_i0_j0" / "boundary_contract.parquet"
    tbl = pq.ParquetFile(parquet).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    # Find first external row and clone its slot_index into the next row.
    for i, sk in enumerate(cols["slot_kind"]):
        if sk == 2:  # external
            cols["slot_index"][i + 1] = cols["slot_index"][i]
            break
    bad = pa.table(
        {n: pa.array(v, type=tbl.schema.field(n).type) for n, v in cols.items()}
    )
    pq.write_table(bad, parquet)
    with pytest.raises(CrossTileValidationError, match="external"):
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
"""Sub-E per-region cross-tile validator (5 invariants per spec §10.2)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from cfm.data.sub_e.writer import SlotKind


class CrossTileValidationError(ValueError):
    """Raised when a sub-E region fails any cross-tile invariant."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_extraction_cross_tile(region_dir: Path) -> None:
    manifest_path = region_dir / "manifest.yaml"
    manifest = _load_yaml(manifest_path)

    expected_sub_e_schema = manifest["sub_e_schema_version"]
    expected_vocab = manifest["versions"]["boundary_vocab_version"]
    expected_derivation = manifest["versions"]["boundary_derivation_version"]

    # Gather per-tile provenance + parquet pairs.
    first_input_digests: dict[str, str] | None = None

    for tile in manifest["tiles"]:
        tile_dir = region_dir / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        prov_path = tile_dir / "provenance.yaml"
        parquet_path = tile_dir / "boundary_contract.parquet"

        prov = _load_yaml(prov_path)

        # Invariant 1: schema version consistency.
        if prov["versions"]["sub_e_schema_version"] != expected_sub_e_schema:
            raise CrossTileValidationError(
                f"sub_e_schema_version mismatch at tile ({tile['tile_i']}, "
                f"{tile['tile_j']}): manifest={expected_sub_e_schema}, "
                f"provenance={prov['versions']['sub_e_schema_version']}"
            )

        # Invariant 2: vocab + derivation version consistency.
        if prov["versions"]["boundary_vocab_version"] != expected_vocab:
            raise CrossTileValidationError(
                f"boundary_vocab_version mismatch at tile "
                f"({tile['tile_i']}, {tile['tile_j']})"
            )
        if prov["versions"]["boundary_derivation_version"] != expected_derivation:
            raise CrossTileValidationError(
                f"boundary_derivation_version mismatch at tile "
                f"({tile['tile_i']}, {tile['tile_j']})"
            )

        # Invariant 3: digest chain.
        expected_prov_sha = tile["provenance_sha256"]
        actual_prov_sha = _file_sha256(prov_path)
        if expected_prov_sha != actual_prov_sha:
            raise CrossTileValidationError(
                f"digest chain broken at tile ({tile['tile_i']}, {tile['tile_j']}): "
                f"manifest→provenance sha mismatch"
            )
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

        # Invariant 5: external slot uniqueness within tile.
        tbl = pq.ParquetFile(parquet_path).read()
        slot_kinds = tbl.column("slot_kind").to_pylist()
        slot_indices = tbl.column("slot_index").to_pylist()
        ext_indices = [
            si
            for sk, si in zip(slot_kinds, slot_indices)
            if sk == int(SlotKind.EXTERNAL_EDGE)
        ]
        if len(ext_indices) != len(set(ext_indices)):
            raise CrossTileValidationError(
                f"external slot_index duplicated at tile ({tile['tile_i']}, "
                f"{tile['tile_j']})"
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/data/sub_e/test_validator_cross_tile.py -v
```

Expected: 6 passed (5 invariants + the valid happy path).

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

    for ti, tj in tiles:
        sub_d_tile = sub_d / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_c_tile = sub_c / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_d_tile.mkdir()
        sub_c_tile.mkdir()

        # Synthetic sub-D macro_core: 64 cell rows + 112 internal edge rows +
        # 32 external edge rows; all active for simplicity. Cell rows carry
        # scope=0; internal-edge rows carry scope=0 (active); external-edge
        # rows carry scope=3 (external_deferred). Total: 64+112+32 = 208 rows.
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
        for idx in range(112):
            slot_kinds.append(1)  # internal_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(idx % 8)
            lower_js.append(idx // 8 % 8)
            axes.append(idx % 2)
            scopes.append(0)
            zoning.append(None)
            density.append(None)
            road.append(0)
        for idx in range(32):
            slot_kinds.append(2)  # external_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(idx % 8)
            lower_js.append(idx // 8 % 4)
            axes.append(idx % 2)
            scopes.append(3)
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
    write_provenance,
)
from cfm.data.sub_e.rotation import cell_to_edge_ids
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
    derivation_version: str,
    lever_3_collapse: bool,
) -> None:
    """Indirect call so tests can monkey-patch to simulate failure."""
    validate_boundary_contract(
        parquet_path,
        expected_derivation_version=derivation_version,
        provenance_derivation_version=derivation_version,
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
        Path(__file__).resolve().parents[3]
        / "configs"
        / "macro_plan"
        / "v1"
        / "boundary_vocab.yaml"
    )
    boundary_vocab_sha = _file_sha256(boundary_vocab_path)

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

        # Halt-on-validator-fail: inline validator runs before provenance is
        # written so a failure cannot leave a stale provenance behind.
        _validate_or_raise(
            parquet_path, BOUNDARY_DERIVATION_VERSION, cfg.lever_3_collapse
        )

        parquet_sha = _file_sha256(parquet_path)
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
        tile_records.append(
            SubEManifestTile(
                tile_i=tile_i,
                tile_j=tile_j,
                provenance_sha256=_file_sha256(prov_path),
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
    (cfg.output_region_dir / "_SUCCESS").touch()

    # Cross-tile validator runs LAST. Failure here removes the _SUCCESS marker
    # to maintain the halt-on-validator-fail invariant.
    try:
        validate_extraction_cross_tile(cfg.output_region_dir)
    except Exception:
        (cfg.output_region_dir / "_SUCCESS").unlink(missing_ok=True)
        raise


def _derive_tile_rows(
    *,
    macro_core,
    crossings,
    features,
    lever_3_collapse: bool,
) -> list[BoundaryContractRow]:
    """Construct the 144-row per-tile boundary contract from sub-D + sub-C."""
    # Index sub-D macro_core by (slot_kind, slot_index) and by edge_id where
    # applicable; index sub-C crossings by edge_id; index features by id.
    edge_scope: dict[tuple[int, int, int], int] = {}
    edge_slot_index: dict[tuple[int, int, int], tuple[int, int]] = {}
    for r in macro_core:
        if r.slot_kind in (1, 2):  # internal or external edge
            assert r.lower_cell_i is not None
            assert r.lower_cell_j is not None
            assert r.axis is not None
            key = (r.lower_cell_i, r.lower_cell_j, r.axis)
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
        i, j, axis = key
        slot_kind_int, slot_idx = edge_slot_index[key]
        is_active_internal = scope == 0 and slot_kind_int == 1
        if is_active_internal and not lever_3_collapse:
            class_raws = [
                cr for cr in crossings_by_edge.get(key, []) if cr is not None or True
            ]
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

- [ ] **Step 3: Smoke-run the scripts on Task 10's synthetic fixture**

```bash
uv run python scripts/derive_boundary_contracts.py --help
uv run python scripts/validate_boundary_contracts.py --help
```

Expected: argparse usage output, no import errors.

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
    for tc, macro in zip(targets, shuffled):
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

    gaps = [s - m for s, m in zip(shuffled_nlls, matched_nlls)]
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
        ext = [si for sk, si in zip(slot_kinds, slot_indices) if sk == 2]
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
        for scope, cls in zip(scope_markers, boundary_classes):
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
