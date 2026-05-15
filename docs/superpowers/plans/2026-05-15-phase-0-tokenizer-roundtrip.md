# Phase 0 Tokenizer Round-Trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md`

**Goal:** Land a TDD-anchored, single-cell GeoJSON tokenizer with a geometric-equivalence round-trip, the canonical Phase-0 vocabulary, three negative-fixture failure paths, repo scaffolding (uv + ruff + pytest), and CI — such that `uv sync && uv run pytest` is green on a clean Mac/Linux machine.

**Architecture:** A small Python package `cfm.tokenizer` with five single-responsibility modules: `errors`, `vocabulary`, `geometry` (the `geometric_equal` helper), `encode`, `decode`. The Phase-0 vocabulary lives in YAML under `configs/`; the human-readable reference under `docs/`. All tests TDD-style: write the failing test first, run it red, implement minimally, run it green, commit.

**Tech stack:** Python 3.11+, `uv` for env/lock, `ruff` for format+lint, `pytest`, `shapely` for geometric comparison, `pyyaml` for the vocabulary loader.

**Two implementation-level decisions locked here (not in the spec, local scope):**

1. **Axis-aligned only in Phase 0.** Encoder and decoder both handle only axis-aligned segments (Δx=0 or Δy=0). Diagonal segments raise `UnsupportedGeometry`. The 32 diagonal `MOVE_*` tokens stay reserved in the vocabulary so Phase 1 can switch them on without renumbering.
2. **Vertex boundaries inferred from direction change.** No `<VERTEX>` token. Consecutive moves in the same cardinal direction belong to one segment; a direction change starts a new vertex. The encoder pre-passes geometry to drop redundant collinear midpoints so the round-trip is clean. Phase 1 may revisit if collinear segments become a real concern.

**Branch:** all work happens on `phase-0-tokenizer`. Merge to `main` only after the Phase-0 done-check in Task 17.

---

## File map (responsibilities)

| File | Purpose |
|---|---|
| `pyproject.toml` | uv-managed project metadata + ruff + pytest config |
| `.python-version` | pin to 3.11 |
| `README.md` | project quick-start; Phase-0 smoke command |
| `.github/workflows/ci.yml` | run ruff + pytest on push/PR |
| `configs/tokenizer/vocab_phase0.yaml` | canonical Phase-0 token vocabulary |
| `docs/tokenizer/vocabulary.md` | human-readable vocab reference |
| `src/cfm/__init__.py` | package marker |
| `src/cfm/tokenizer/__init__.py` | public API re-exports |
| `src/cfm/tokenizer/errors.py` | `TokenizerError` + 4 subclasses |
| `src/cfm/tokenizer/vocabulary.py` | `Vocabulary` dataclass + YAML loader |
| `src/cfm/tokenizer/geometry.py` | `geometric_equal` helper |
| `src/cfm/tokenizer/encode.py` | `encode_cell` |
| `src/cfm/tokenizer/decode.py` | `decode_cell` |
| `tests/conftest.py` | fixture-path helpers |
| `tests/tokenizer/test_vocabulary.py` | YAML load + determinism |
| `tests/tokenizer/test_geometric_equal.py` | tolerance edge cases |
| `tests/tokenizer/test_encode.py` | encoder per-geometry-type |
| `tests/tokenizer/test_decode.py` | decoder per-geometry-type |
| `tests/tokenizer/test_round_trip.py` | end-to-end fixture round-trip |
| `tests/tokenizer/test_determinism.py` | encoding twice = identical tokens |
| `tests/tokenizer/test_errors.py` | three negative-fixture failure paths |
| `tests/fixtures/single_cell/{README.md, input.geojson, expected.yaml}` | positive fixture |
| `tests/fixtures/degenerate/{non_rectangular_building, unknown_class, out_of_bounds}.geojson` | negative fixtures |
| `scripts/smoke.py` | CLI: load fixture, round-trip, print deterministic summary |

---

## Task 1: Bootstrap branch and Python project

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_sanity.py`

- [ ] **Step 1.1: Create the working branch**

Run: `git checkout -b phase-0-tokenizer`
Expected: `Switched to a new branch 'phase-0-tokenizer'`.

- [ ] **Step 1.2: Pin Python version**

Create `.python-version`:

```
3.11
```

- [ ] **Step 1.3: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[project]
name = "cfm"
version = "0.0.1"
description = "Bonzai-OSM city foundation model"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "shapely>=2.0",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cfm"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "ANN", "RUF"]
ignore = ["ANN401"]  # Any allowed in geojson dict

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN"]
"scripts/**" = ["ANN"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
  "slow: tests slower than 5 seconds",
]
```

- [ ] **Step 1.4: Write minimal `README.md`**

Create `README.md`:

```markdown
# Bonzai-OSM

A generative foundation model for city geometry. See `PRD.md` for goals and `CLAUDE.md` for collaboration rules.

## Phase 0 quick start

```bash
uv sync --all-extras
uv run pytest
uv run python scripts/smoke.py
```

Phase 0 ends when all three commands succeed on a clean machine.
```

- [ ] **Step 1.5: Create test package marker and conftest**

Create `tests/__init__.py` (empty).

Create `tests/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def vocab_yaml_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "tokenizer" / "vocab_phase0.yaml"
```

- [ ] **Step 1.6: Write a sanity test to prove pytest is wired**

Create `tests/test_sanity.py`:

```python
from __future__ import annotations


def test_python_runs() -> None:
    assert 1 + 1 == 2
```

- [ ] **Step 1.7: Sync deps and run pytest**

Run: `uv sync --all-extras && uv run pytest -v`
Expected: 1 test passes, exit code 0. If `uv` is not on PATH, install per https://docs.astral.sh/uv/.

- [ ] **Step 1.8: Run ruff to verify formatter/linter is wired**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: both succeed; if format check fails, run `uv run ruff format .` then re-check.

- [ ] **Step 1.9: Commit**

```bash
git add .python-version pyproject.toml uv.lock README.md tests/__init__.py tests/conftest.py tests/test_sanity.py
git commit -m "chore: scaffold uv-managed Python 3.11 project with ruff + pytest"
```

---

## Task 2: Phase-0 vocabulary YAML + reference doc

**Files:**
- Create: `configs/tokenizer/vocab_phase0.yaml`
- Create: `docs/tokenizer/vocabulary.md`

- [ ] **Step 2.1: Write the canonical vocabulary YAML**

Create `configs/tokenizer/vocab_phase0.yaml`:

```yaml
# Phase 0 vocabulary for the city foundation model tokenizer.
# Frozen for Phase 0. Phase 1 may append; never reorder, never delete.
#
# The loader flattens this into a single ordered list of token names whose
# index is the canonical TokenId. Order: control, hierarchy, feature_class
# (road, building, poi, land_use), anchor_x, anchor_y, moves.
#
# Counts: control=8, hierarchy=4, feature_class=19, anchor=500, move=48 -> 579.

control:
  - PAD
  - BOS
  - EOS
  - CELL
  - END_CELL
  - FEATURE_START
  - FEATURE_END
  - EXIT

hierarchy:  # reserved; unused in Phase 0
  - MACRO
  - END_MACRO
  - MICRO
  - END_MICRO

feature_class:
  road:
    - R_motorway
    - R_primary
    - R_secondary
    - R_residential
    - R_service
  building:
    - B_residential
    - B_commercial
    - B_industrial
  poi:
    - POI_restaurant
    - POI_school
    - POI_retail
    - POI_park_amenity
    - POI_transit_stop
  land_use:
    - L_residential
    - L_commercial
    - L_industrial
    - L_park
    - L_water
    - L_agricultural

anchor:
  axis_count: 250  # 250 m cell on a 1 m grid -> ANCHOR_X_0..249, ANCHOR_Y_0..249

move:
  directions: [N, NE, E, SE, S, SW, W, NW]
  steps_m: [1, 2, 4, 8, 16, 32]
```

- [ ] **Step 2.2: Write the human-readable reference doc**

Create `docs/tokenizer/vocabulary.md`:

```markdown
# Phase 0 token vocabulary

Reference companion to `configs/tokenizer/vocab_phase0.yaml`. The YAML is the source of truth; this doc explains what the tokens *mean*.

## Why the vocabulary is the way it is

The Phase-0 vocabulary is a deliberately small starter set. It exists to lock the encode → decode round-trip contract before Phase 1's full Overture-driven frequency analysis. New tokens are append-only across phases so checkpoints stay readable. See spec `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md`.

Counts: 8 control + 4 hierarchy + 19 feature class + 500 anchor + 48 move = **579 tokens**.

## Control (8)

| Token | Role |
|---|---|
| `PAD` | sequence padding |
| `BOS` | beginning of sequence |
| `EOS` | end of sequence |
| `CELL` | start of a cell's contents |
| `END_CELL` | end of a cell's contents |
| `FEATURE_START` | start of a feature |
| `FEATURE_END` | end of a feature |
| `EXIT` | preceding line ends on a cell boundary |

## Hierarchy (4, reserved)

`MACRO`, `END_MACRO`, `MICRO`, `END_MICRO`. Reserved for Phase 1 macro/micro split. IDs frozen now so Phase-1 additions don't renumber.

## Feature classes (19)

- Roads (5): `R_motorway`, `R_primary`, `R_secondary`, `R_residential`, `R_service`
- Buildings (3): `B_residential`, `B_commercial`, `B_industrial`
- POIs (5): `POI_restaurant`, `POI_school`, `POI_retail`, `POI_park_amenity`, `POI_transit_stop`
- Land use (6): `L_residential`, `L_commercial`, `L_industrial`, `L_park`, `L_water`, `L_agricultural`

A feature's `properties.class` value must match exactly one of these.

## Anchor (500)

Absolute positions in cell-local metres, split as `ANCHOR_X_n` + `ANCHOR_Y_n` for `n` in `0..249`. Always emitted as a pair (X then Y). One pair marks the starting vertex of every feature.

## Move (48)

8 cardinal/diagonal directions × 6 dyadic step sizes {1, 2, 4, 8, 16, 32} m.

- Directions: `N, NE, E, SE, S, SW, W, NW`
- Sizes (m): `1, 2, 4, 8, 16, 32`
- Names: `MOVE_<dir>_<step>`, e.g. `MOVE_E_8`, `MOVE_NW_32`.

**Phase 0 emits cardinal moves only** (`N`, `E`, `S`, `W`). The 32 diagonal tokens are reserved; the encoder raises `UnsupportedGeometry` on non-axis-aligned segments. Phase 1 enables diagonals.

## Sequence shape (Phase 0)

```
<BOS> <CELL>
  <FEATURE_START> <class> <anchor_pair> [<move> ...] [<EXIT>] <FEATURE_END>
  ...
<END_CELL> <EOS>
```

For polygons, the move sequence closes (cumulative delta returns to the anchor). For lines that terminate on a cell edge, the move sequence is followed by `<EXIT>` before `<FEATURE_END>`. Vertex boundaries within a polygon or line are inferred by direction change (consecutive moves in the same cardinal direction collapse into one segment).
```

- [ ] **Step 2.3: Lint-check the new files exist and ruff is still clean**

Run: `uv run ruff check configs/ docs/`
Expected: no Python files in those dirs, ruff passes trivially.

- [ ] **Step 2.4: Commit**

```bash
git add configs/tokenizer/vocab_phase0.yaml docs/tokenizer/vocabulary.md
git commit -m "data: add Phase 0 tokenizer vocabulary YAML + reference doc"
```

---

## Task 3: Vocabulary class (TDD)

**Files:**
- Create: `src/cfm/__init__.py`
- Create: `src/cfm/tokenizer/__init__.py`
- Create: `src/cfm/tokenizer/vocabulary.py`
- Create: `tests/tokenizer/__init__.py`
- Create: `tests/tokenizer/test_vocabulary.py`

- [ ] **Step 3.1: Write the failing tests first**

Create `tests/tokenizer/__init__.py` (empty).

Create `tests/tokenizer/test_vocabulary.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer.vocabulary import Vocabulary


def test_load_phase0_total_count(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert len(vocab) == 579


def test_first_eight_ids_are_control_tokens(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    expected = ("PAD", "BOS", "EOS", "CELL", "END_CELL",
                "FEATURE_START", "FEATURE_END", "EXIT")
    assert vocab.id_to_token[:8] == expected


def test_anchor_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert "ANCHOR_X_0" in vocab.token_to_id
    assert "ANCHOR_X_249" in vocab.token_to_id
    assert "ANCHOR_Y_0" in vocab.token_to_id
    assert "ANCHOR_Y_249" in vocab.token_to_id
    assert "ANCHOR_X_250" not in vocab.token_to_id


def test_move_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert "MOVE_E_1" in vocab.token_to_id
    assert "MOVE_NW_32" in vocab.token_to_id
    assert "MOVE_E_3" not in vocab.token_to_id


def test_feature_class_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    for name in ("R_residential", "B_residential", "POI_restaurant", "L_park"):
        assert name in vocab.token_to_id


def test_load_is_deterministic(vocab_yaml_path: Path) -> None:
    a = Vocabulary.load(vocab_yaml_path)
    b = Vocabulary.load(vocab_yaml_path)
    assert a.id_to_token == b.id_to_token
    assert a.token_to_id == b.token_to_id


def test_token_to_id_is_inverse_of_id_to_token(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    for i, name in enumerate(vocab.id_to_token):
        assert vocab.token_to_id[name] == i


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Vocabulary.load(tmp_path / "does_not_exist.yaml")
```

- [ ] **Step 3.2: Run the tests; they must fail**

Run: `uv run pytest tests/tokenizer/test_vocabulary.py -v`
Expected: collection error (`cfm.tokenizer.vocabulary` doesn't exist).

- [ ] **Step 3.3: Create empty package markers**

Create `src/cfm/__init__.py`:

```python
"""Bonzai-OSM city foundation model package."""
```

Create `src/cfm/tokenizer/__init__.py`:

```python
"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.vocabulary import Vocabulary

__all__ = ["Vocabulary"]
```

- [ ] **Step 3.4: Implement `Vocabulary`**

Create `src/cfm/tokenizer/vocabulary.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

TokenId = int


@dataclass(frozen=True)
class Vocabulary:
    """An ordered, immutable token vocabulary.

    `id_to_token[i]` is the token name at id `i`.
    `token_to_id[name]` is the inverse lookup.
    """

    id_to_token: tuple[str, ...]
    token_to_id: Mapping[str, TokenId]

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = _flatten(data)
        token_to_id = MappingProxyType({name: i for i, name in enumerate(names)})
        return cls(id_to_token=tuple(names), token_to_id=token_to_id)


def _flatten(data: dict) -> list[str]:
    out: list[str] = []
    out.extend(data["control"])
    out.extend(data["hierarchy"])
    fc = data["feature_class"]
    for group in ("road", "building", "poi", "land_use"):
        out.extend(fc[group])
    axis_count = int(data["anchor"]["axis_count"])
    out.extend(f"ANCHOR_X_{i}" for i in range(axis_count))
    out.extend(f"ANCHOR_Y_{i}" for i in range(axis_count))
    move = data["move"]
    for direction in move["directions"]:
        for step in move["steps_m"]:
            out.append(f"MOVE_{direction}_{step}")
    return out
```

- [ ] **Step 3.5: Run the tests; they must pass**

Run: `uv run pytest tests/tokenizer/test_vocabulary.py -v`
Expected: 8 passed.

- [ ] **Step 3.6: Lint**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: both pass. If format fails, run `uv run ruff format src tests`.

- [ ] **Step 3.7: Commit**

```bash
git add src/cfm tests/tokenizer/__init__.py tests/tokenizer/test_vocabulary.py
git commit -m "feat(tokenizer): add Vocabulary class with YAML loader"
```

---

## Task 4: Error hierarchy

**Files:**
- Create: `src/cfm/tokenizer/errors.py`
- Modify: `src/cfm/tokenizer/__init__.py`
- Create: `tests/tokenizer/test_errors_basic.py`

- [ ] **Step 4.1: Write the failing tests first**

Create `tests/tokenizer/test_errors_basic.py`:

```python
from __future__ import annotations

import pytest

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)


def test_all_subclasses_inherit_from_tokenizer_error() -> None:
    for cls in (
        UnsupportedFeatureClass,
        UnsupportedGeometry,
        FeatureOutOfBounds,
        VocabularyMismatch,
    ):
        assert issubclass(cls, TokenizerError)


def test_tokenizer_error_is_value_error() -> None:
    assert issubclass(TokenizerError, ValueError)


def test_raising_and_catching_specific_subclass() -> None:
    with pytest.raises(UnsupportedFeatureClass):
        raise UnsupportedFeatureClass("unknown class 'X'")


def test_catching_as_tokenizer_error_catches_subclasses() -> None:
    with pytest.raises(TokenizerError):
        raise UnsupportedGeometry("triangle building")
```

- [ ] **Step 4.2: Run tests; expect failure**

Run: `uv run pytest tests/tokenizer/test_errors_basic.py -v`
Expected: import error (`cfm.tokenizer.errors` not found).

- [ ] **Step 4.3: Implement the error hierarchy**

Create `src/cfm/tokenizer/errors.py`:

```python
from __future__ import annotations


class TokenizerError(ValueError):
    """Base class for all tokenizer failures. Inherits from ValueError."""


class UnsupportedFeatureClass(TokenizerError):
    """A feature's class is not present in the active vocabulary."""


class UnsupportedGeometry(TokenizerError):
    """A geometry shape is not handled by the current tokenizer (e.g. non-rectangular building, diagonal road in Phase 0)."""


class FeatureOutOfBounds(TokenizerError):
    """A feature's geometry extends outside the cell bounds (except as a deliberate boundary exit)."""


class VocabularyMismatch(TokenizerError):
    """Decoded token IDs are not present in the active vocabulary."""
```

Update `src/cfm/tokenizer/__init__.py`:

```python
"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
from cfm.tokenizer.vocabulary import Vocabulary

__all__ = [
    "FeatureOutOfBounds",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
]
```

- [ ] **Step 4.4: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_errors_basic.py -v`
Expected: 4 passed.

- [ ] **Step 4.5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/__init__.py src/cfm/tokenizer/errors.py tests/tokenizer/test_errors_basic.py
git commit -m "feat(tokenizer): add TokenizerError hierarchy"
```

---

## Task 5: Positive fixture (single-cell GeoJSON)

**Files:**
- Create: `tests/fixtures/single_cell/README.md`
- Create: `tests/fixtures/single_cell/input.geojson`
- Create: `tests/fixtures/single_cell/expected.yaml`

- [ ] **Step 5.1: Write the fixture README**

Create `tests/fixtures/single_cell/README.md`:

```markdown
# Single-cell positive fixture

A hand-built 250 m × 250 m cell anchored at (0, 0) in cell-local metres. Contains exactly one feature of each kind the Phase-0 tokenizer must handle: one road exiting the east edge, one rectangular building, one POI, one axis-aligned land-use polygon. Small enough to read in your head.

Used by:
- `tests/tokenizer/test_round_trip.py`
- `tests/tokenizer/test_determinism.py`
- `scripts/smoke.py`

If you change `input.geojson`, update `expected.yaml` to match.
```

- [ ] **Step 5.2: Write the fixture GeoJSON**

Create `tests/fixtures/single_cell/input.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"class": "R_residential"},
      "geometry": {
        "type": "LineString",
        "coordinates": [[0, 125], [250, 125]]
      }
    },
    {
      "type": "Feature",
      "properties": {"class": "B_residential"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"class": "POI_restaurant"},
      "geometry": {
        "type": "Point",
        "coordinates": [50, 80]
      }
    },
    {
      "type": "Feature",
      "properties": {"class": "L_residential"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100, 0], [250, 0], [250, 150], [100, 150], [100, 0]]]
      }
    }
  ]
}
```

- [ ] **Step 5.3: Write the sanity-check expected file**

Create `tests/fixtures/single_cell/expected.yaml`:

```yaml
cell_origin: [0.0, 0.0]
cell_size_m: 250.0
feature_count: 4
classes:
  - R_residential
  - B_residential
  - POI_restaurant
  - L_residential
road_exits_east: true
```

- [ ] **Step 5.4: Commit**

```bash
git add tests/fixtures/single_cell
git commit -m "test(fixtures): add single-cell positive fixture"
```

---

## Task 6: Negative fixtures (three failure modes)

**Files:**
- Create: `tests/fixtures/degenerate/non_rectangular_building.geojson`
- Create: `tests/fixtures/degenerate/unknown_class.geojson`
- Create: `tests/fixtures/degenerate/out_of_bounds.geojson`
- Create: `tests/fixtures/degenerate/README.md`

- [ ] **Step 6.1: Document the degenerate fixtures**

Create `tests/fixtures/degenerate/README.md`:

```markdown
# Degenerate fixtures

One file per failure mode the Phase-0 tokenizer must surface as a specific `TokenizerError` subclass. Used exclusively by `tests/tokenizer/test_errors.py`.

| File | Trigger | Expected error |
|---|---|---|
| `non_rectangular_building.geojson` | L-shaped axis-aligned building polygon (6 vertices) | `UnsupportedGeometry` |
| `unknown_class.geojson` | feature with `class: "B_castle"` (not in Phase-0 vocab) | `UnsupportedFeatureClass` |
| `out_of_bounds.geojson` | POI at (300, 300) inside a 250 × 250 cell | `FeatureOutOfBounds` |
```

- [ ] **Step 6.2: Non-rectangular building fixture (L-shape, axis-aligned)**

Create `tests/fixtures/degenerate/non_rectangular_building.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"class": "B_residential"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[30, 30], [70, 30], [70, 50], [50, 50], [50, 70], [30, 70], [30, 30]]]
      }
    }
  ]
}
```

- [ ] **Step 6.3: Unknown class fixture**

Create `tests/fixtures/degenerate/unknown_class.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"class": "B_castle"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]]
      }
    }
  ]
}
```

- [ ] **Step 6.4: Out-of-bounds fixture**

Create `tests/fixtures/degenerate/out_of_bounds.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"class": "POI_restaurant"},
      "geometry": {
        "type": "Point",
        "coordinates": [300, 300]
      }
    }
  ]
}
```

- [ ] **Step 6.5: Commit**

```bash
git add tests/fixtures/degenerate
git commit -m "test(fixtures): add three degenerate negative fixtures"
```

---

## Task 7: `geometric_equal` helper (TDD)

**Files:**
- Create: `src/cfm/tokenizer/geometry.py`
- Modify: `src/cfm/tokenizer/__init__.py`
- Create: `tests/tokenizer/test_geometric_equal.py`

- [ ] **Step 7.1: Write the failing tests**

Create `tests/tokenizer/test_geometric_equal.py`:

```python
from __future__ import annotations

from copy import deepcopy

from cfm.tokenizer.geometry import geometric_equal


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _point(x: float, y: float, cls: str = "POI_restaurant") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def _rect(x0: float, y0: float, x1: float, y1: float, cls: str = "B_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
    }


def _line(coords: list[list[float]], cls: str = "R_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def test_identical_collections_equal() -> None:
    a = _fc(_point(50, 80), _rect(40, 40, 60, 60))
    b = deepcopy(a)
    assert geometric_equal(a, b) is True


def test_point_within_tolerance_equal() -> None:
    a = _fc(_point(50.0, 80.0))
    b = _fc(_point(50.4, 80.0))
    assert geometric_equal(a, b, tol_m=0.5) is True


def test_point_just_outside_tolerance_not_equal() -> None:
    a = _fc(_point(50.0, 80.0))
    b = _fc(_point(50.6, 80.0))
    assert geometric_equal(a, b, tol_m=0.5) is False


def test_different_class_not_equal() -> None:
    a = _fc(_point(50, 80, cls="POI_restaurant"))
    b = _fc(_point(50, 80, cls="POI_school"))
    assert geometric_equal(a, b) is False


def test_different_count_not_equal() -> None:
    a = _fc(_point(50, 80))
    b = _fc(_point(50, 80), _point(10, 10))
    assert geometric_equal(a, b) is False


def test_polygons_equal_under_vertex_rotation() -> None:
    # Same square, different starting vertex.
    a = _fc(_rect(0, 0, 20, 20))
    b = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"class": "B_residential"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20, 0], [20, 20], [0, 20], [0, 0], [20, 0]]],
            },
        }],
    }
    assert geometric_equal(a, b) is True


def test_lines_equal() -> None:
    a = _fc(_line([[0, 125], [250, 125]]))
    b = _fc(_line([[0, 125], [250, 125]]))
    assert geometric_equal(a, b) is True


def test_two_features_same_class_paired_greedy() -> None:
    a = _fc(_point(10, 10), _point(200, 200))
    b = _fc(_point(200, 200), _point(10, 10))
    assert geometric_equal(a, b) is True
```

- [ ] **Step 7.2: Run tests; expect failure**

Run: `uv run pytest tests/tokenizer/test_geometric_equal.py -v`
Expected: import error.

- [ ] **Step 7.3: Implement `geometric_equal`**

Create `src/cfm/tokenizer/geometry.py`:

```python
from __future__ import annotations

from collections import defaultdict

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

GeoJSON = dict


def geometric_equal(a: GeoJSON, b: GeoJSON, *, tol_m: float = 0.5) -> bool:
    """Return True iff two GeoJSON FeatureCollections are geometrically equivalent.

    Equivalence rule:
      - Features are grouped by `properties.class`. Class counts must match.
      - Within each class, features in `a` are greedily paired with the
        unmatched feature in `b` of minimum geometric distance.
      - Every pair must be within `tol_m`:
            * Points: Euclidean distance.
            * Lines/Polygons: symmetric Hausdorff distance.
    """
    grouped_a = _group_by_class(a)
    grouped_b = _group_by_class(b)
    if set(grouped_a) != set(grouped_b):
        return False
    for cls, geoms_a in grouped_a.items():
        geoms_b = list(grouped_b[cls])
        if len(geoms_a) != len(geoms_b):
            return False
        for ga in geoms_a:
            best_idx = None
            best_dist = float("inf")
            for i, gb in enumerate(geoms_b):
                d = _distance(ga, gb)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is None or best_dist > tol_m:
                return False
            geoms_b.pop(best_idx)
    return True


def _group_by_class(fc: GeoJSON) -> dict[str, list[BaseGeometry]]:
    out: dict[str, list[BaseGeometry]] = defaultdict(list)
    for feat in fc["features"]:
        cls = feat["properties"]["class"]
        out[cls].append(shape(feat["geometry"]))
    return out


def _distance(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.geom_type == "Point" and b.geom_type == "Point":
        return a.distance(b)
    if a.geom_type != b.geom_type:
        return float("inf")
    return max(a.hausdorff_distance(b), b.hausdorff_distance(a))
```

- [ ] **Step 7.4: Re-export from package**

Update `src/cfm/tokenizer/__init__.py`:

```python
"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
from cfm.tokenizer.geometry import geometric_equal
from cfm.tokenizer.vocabulary import Vocabulary

__all__ = [
    "FeatureOutOfBounds",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
    "geometric_equal",
]
```

- [ ] **Step 7.5: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_geometric_equal.py -v`
Expected: 8 passed.

- [ ] **Step 7.6: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/geometry.py src/cfm/tokenizer/__init__.py tests/tokenizer/test_geometric_equal.py
git commit -m "feat(tokenizer): add geometric_equal helper with Hausdorff distance"
```

---

## Task 8: Encoder skeleton + `CellTokens` + Point handling (TDD)

**Files:**
- Create: `src/cfm/tokenizer/encode.py`
- Modify: `src/cfm/tokenizer/__init__.py`
- Create: `tests/tokenizer/test_encode.py`

- [ ] **Step 8.1: Write the failing tests for Point encoding**

Create `tests/tokenizer/test_encode.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    Vocabulary,
)
from cfm.tokenizer.encode import CellTokens, encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _poi(x: float, y: float, cls: str = "POI_restaurant") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def test_empty_collection_produces_bos_cell_endcell_eos(vocab: Vocabulary) -> None:
    out = encode_cell(_fc(), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    assert isinstance(out, CellTokens)
    expected = (
        vocab.token_to_id["BOS"],
        vocab.token_to_id["CELL"],
        vocab.token_to_id["END_CELL"],
        vocab.token_to_id["EOS"],
    )
    assert out.tokens == expected
    assert out.cell_origin == (0.0, 0.0)
    assert out.cell_size_m == 250.0


def test_single_poi_encodes_as_anchor_pair(vocab: Vocabulary) -> None:
    out = encode_cell(_fc(_poi(50, 80)), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    expected = (
        vocab.token_to_id["BOS"],
        vocab.token_to_id["CELL"],
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id["POI_restaurant"],
        vocab.token_to_id["ANCHOR_X_50"],
        vocab.token_to_id["ANCHOR_Y_80"],
        vocab.token_to_id["FEATURE_END"],
        vocab.token_to_id["END_CELL"],
        vocab.token_to_id["EOS"],
    )
    assert out.tokens == expected


def test_poi_outside_cell_raises_out_of_bounds(vocab: Vocabulary) -> None:
    with pytest.raises(FeatureOutOfBounds):
        encode_cell(_fc(_poi(300, 300)), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)


def test_unknown_class_raises(vocab: Vocabulary) -> None:
    with pytest.raises(UnsupportedFeatureClass):
        encode_cell(_fc(_poi(50, 80, cls="POI_castle")), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
```

- [ ] **Step 8.2: Run tests; expect import failure**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: import error.

- [ ] **Step 8.3: Implement skeleton + Point**

Create `src/cfm/tokenizer/encode.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
)
from cfm.tokenizer.vocabulary import TokenId, Vocabulary

GeoJSON = dict


@dataclass(frozen=True)
class CellTokens:
    """Token sequence for a single cell, plus its spatial frame."""

    tokens: tuple[TokenId, ...]
    cell_origin: tuple[float, float]
    cell_size_m: float


def encode_cell(
    geojson: GeoJSON,
    *,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> CellTokens:
    out: list[TokenId] = [vocab.token_to_id["BOS"], vocab.token_to_id["CELL"]]
    for feature in geojson["features"]:
        out.extend(_encode_feature(feature, cell_origin, cell_size_m, vocab))
    out.extend([vocab.token_to_id["END_CELL"], vocab.token_to_id["EOS"]])
    return CellTokens(tokens=tuple(out), cell_origin=cell_origin, cell_size_m=cell_size_m)


def _encode_feature(
    feature: dict,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    cls = feature["properties"]["class"]
    if cls not in vocab.token_to_id:
        raise UnsupportedFeatureClass(f"unknown class {cls!r}")
    geom = feature["geometry"]
    gtype = geom["type"]
    if gtype == "Point":
        body = _encode_point(geom["coordinates"], cell_origin, cell_size_m, vocab)
    else:
        # Line and Polygon handlers arrive in later tasks.
        raise UnsupportedGeometry(f"Phase 0 does not yet handle geometry type {gtype!r}")
    return [
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id[cls],
        *body,
        vocab.token_to_id["FEATURE_END"],
    ]


def _encode_point(
    coords: list[float],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    x_local, y_local = _to_cell_local(coords[0], coords[1], cell_origin)
    _require_in_bounds(x_local, y_local, cell_size_m)
    return [
        vocab.token_to_id[f"ANCHOR_X_{int(round(x_local))}"],
        vocab.token_to_id[f"ANCHOR_Y_{int(round(y_local))}"],
    ]


def _to_cell_local(x: float, y: float, cell_origin: tuple[float, float]) -> tuple[float, float]:
    return x - cell_origin[0], y - cell_origin[1]


def _require_in_bounds(x: float, y: float, cell_size_m: float) -> None:
    if not (0 <= x <= cell_size_m and 0 <= y <= cell_size_m):
        raise FeatureOutOfBounds(f"point ({x}, {y}) outside [0, {cell_size_m}]^2")
```

Update `src/cfm/tokenizer/__init__.py` `__all__` to also export `CellTokens` and `encode_cell`:

```python
"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.encode import CellTokens, encode_cell
from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
from cfm.tokenizer.geometry import geometric_equal
from cfm.tokenizer.vocabulary import Vocabulary

__all__ = [
    "CellTokens",
    "FeatureOutOfBounds",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
    "encode_cell",
    "geometric_equal",
]
```

- [ ] **Step 8.4: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: 4 passed.

- [ ] **Step 8.5: Commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/encode.py src/cfm/tokenizer/__init__.py tests/tokenizer/test_encode.py
git commit -m "feat(tokenizer): encode_cell handles empty FC and Point features"
```

---

## Task 9: Encoder — Polygon (rectangle + axis-aligned)

**Files:**
- Modify: `src/cfm/tokenizer/encode.py`
- Modify: `tests/tokenizer/test_encode.py`

- [ ] **Step 9.1: Add failing tests for polygons**

Append to `tests/tokenizer/test_encode.py`:

```python
from cfm.tokenizer.errors import UnsupportedGeometry


def _rect(x0: float, y0: float, x1: float, y1: float, cls: str = "B_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
    }


def test_rectangular_building_encodes_with_dyadic_moves(vocab: Vocabulary) -> None:
    # 20m x 20m building at (40,40)-(60,60). Sides 20m = 16+4.
    out = encode_cell(_fc(_rect(40, 40, 60, 60)),
                      cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    t = vocab.token_to_id
    expected_core = (
        t["FEATURE_START"], t["B_residential"],
        t["ANCHOR_X_40"], t["ANCHOR_Y_40"],
        t["MOVE_E_16"], t["MOVE_E_4"],
        t["MOVE_N_16"], t["MOVE_N_4"],
        t["MOVE_W_16"], t["MOVE_W_4"],
        t["MOVE_S_16"], t["MOVE_S_4"],
        t["FEATURE_END"],
    )
    assert out.tokens[2:-2] == expected_core


def test_non_rectangular_building_raises(vocab: Vocabulary) -> None:
    # L-shape (6 vertices) under B_* class is rejected.
    feature = {
        "type": "Feature",
        "properties": {"class": "B_residential"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[30, 30], [70, 30], [70, 50], [50, 50], [50, 70], [30, 70], [30, 30]]],
        },
    }
    with pytest.raises(UnsupportedGeometry):
        encode_cell(_fc(feature), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)


def test_land_use_polygon_multi_segment_encodes(vocab: Vocabulary) -> None:
    # 150m x 150m square anchored at (100, 0). Sides 150m = 128+16+4+2.
    out = encode_cell(_fc(_rect(100, 0, 250, 150, cls="L_residential")),
                      cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    t = vocab.token_to_id
    # Just spot-check the anchor + first east-side decomposition.
    assert t["L_residential"] in out.tokens
    assert t["ANCHOR_X_100"] in out.tokens
    assert t["ANCHOR_Y_0"] in out.tokens
```

- [ ] **Step 9.2: Run tests; expect failure**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: the three new tests fail with `UnsupportedGeometry: Phase 0 does not yet handle geometry type 'Polygon'`.

- [ ] **Step 9.3: Implement Polygon encoding**

Replace the `if gtype == "Point":` branch in `_encode_feature` (in `src/cfm/tokenizer/encode.py`) to also dispatch on `Polygon`, and add the helpers below.

Update `src/cfm/tokenizer/encode.py` (replace the file):

```python
from __future__ import annotations

from dataclasses import dataclass

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
)
from cfm.tokenizer.vocabulary import TokenId, Vocabulary

GeoJSON = dict

DYADIC_STEPS_M: tuple[int, ...] = (32, 16, 8, 4, 2, 1)
CARDINAL: tuple[tuple[int, int, str], ...] = (
    (0, 1, "N"),
    (1, 0, "E"),
    (0, -1, "S"),
    (-1, 0, "W"),
)


@dataclass(frozen=True)
class CellTokens:
    """Token sequence for a single cell, plus its spatial frame."""

    tokens: tuple[TokenId, ...]
    cell_origin: tuple[float, float]
    cell_size_m: float


def encode_cell(
    geojson: GeoJSON,
    *,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> CellTokens:
    out: list[TokenId] = [vocab.token_to_id["BOS"], vocab.token_to_id["CELL"]]
    for feature in geojson["features"]:
        out.extend(_encode_feature(feature, cell_origin, cell_size_m, vocab))
    out.extend([vocab.token_to_id["END_CELL"], vocab.token_to_id["EOS"]])
    return CellTokens(tokens=tuple(out), cell_origin=cell_origin, cell_size_m=cell_size_m)


def _encode_feature(
    feature: dict,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    cls = feature["properties"]["class"]
    if cls not in vocab.token_to_id:
        raise UnsupportedFeatureClass(f"unknown class {cls!r}")
    geom = feature["geometry"]
    gtype = geom["type"]
    if gtype == "Point":
        body = _encode_point(geom["coordinates"], cell_origin, cell_size_m, vocab)
    elif gtype == "Polygon":
        body = _encode_polygon(cls, geom["coordinates"], cell_origin, cell_size_m, vocab)
    else:
        raise UnsupportedGeometry(f"Phase 0 does not yet handle geometry type {gtype!r}")
    return [
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id[cls],
        *body,
        vocab.token_to_id["FEATURE_END"],
    ]


def _encode_point(
    coords: list[float],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    x_local, y_local = _to_cell_local(coords[0], coords[1], cell_origin)
    _require_in_bounds(x_local, y_local, cell_size_m)
    return [
        vocab.token_to_id[f"ANCHOR_X_{int(round(x_local))}"],
        vocab.token_to_id[f"ANCHOR_Y_{int(round(y_local))}"],
    ]


def _encode_polygon(
    cls: str,
    coordinates: list[list[list[float]]],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    # Phase 0: ignore interior rings, exterior only.
    ring = coordinates[0]
    if ring[0] != ring[-1]:
        raise UnsupportedGeometry("polygon ring not closed (first != last)")
    vertices = [tuple(p) for p in ring[:-1]]  # drop the closing duplicate
    vertices_local = [_to_cell_local(x, y, cell_origin) for (x, y) in vertices]
    for x, y in vertices_local:
        _require_in_bounds(x, y, cell_size_m)
    vertices_local = _drop_collinear(vertices_local)
    if cls.startswith("B_") and len(vertices_local) != 4:
        raise UnsupportedGeometry(
            f"Phase 0 buildings must be axis-aligned rectangles (4 vertices); got {len(vertices_local)}"
        )
    return _encode_closed_path(vertices_local, vocab)


def _encode_closed_path(
    vertices_local: list[tuple[float, float]],
    vocab: Vocabulary,
) -> list[TokenId]:
    # First vertex becomes the anchor.
    ax, ay = vertices_local[0]
    body: list[TokenId] = [
        vocab.token_to_id[f"ANCHOR_X_{int(round(ax))}"],
        vocab.token_to_id[f"ANCHOR_Y_{int(round(ay))}"],
    ]
    # For each subsequent vertex (including a virtual return to the anchor for closure),
    # emit cardinal moves that sum to the segment delta.
    closed = [*vertices_local, vertices_local[0]]
    for i in range(len(vertices_local)):
        x0, y0 = closed[i]
        x1, y1 = closed[i + 1]
        body.extend(_encode_axis_aligned_segment(x1 - x0, y1 - y0, vocab))
    return body


def _encode_axis_aligned_segment(
    dx: float,
    dy: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    if dx != 0 and dy != 0:
        raise UnsupportedGeometry(
            f"Phase 0 supports axis-aligned segments only; got dx={dx}, dy={dy}"
        )
    if dx == 0 and dy == 0:
        return []  # degenerate; drop_collinear should have removed it
    if dx == 0:
        direction = "N" if dy > 0 else "S"
        length = int(round(abs(dy)))
    else:
        direction = "E" if dx > 0 else "W"
        length = int(round(abs(dx)))
    return [
        vocab.token_to_id[f"MOVE_{direction}_{step}"]
        for step in _dyadic_decomposition(length)
    ]


def _dyadic_decomposition(n: int) -> list[int]:
    """Greedy decomposition of `n` into the dyadic step set {32, 16, 8, 4, 2, 1}."""
    if n < 1:
        raise UnsupportedGeometry(f"segment length must be a positive integer metre; got {n}")
    out: list[int] = []
    remaining = n
    for step in DYADIC_STEPS_M:
        while remaining >= step:
            out.append(step)
            remaining -= step
    if remaining != 0:
        raise UnsupportedGeometry(f"segment length {n} not expressible in dyadic steps")
    return out


def _drop_collinear(verts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove vertices that lie on the same axis-aligned segment as their neighbours."""
    if len(verts) < 3:
        return verts
    n = len(verts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        prev = verts[(i - 1) % n]
        cur = verts[i]
        nxt = verts[(i + 1) % n]
        dx_in, dy_in = cur[0] - prev[0], cur[1] - prev[1]
        dx_out, dy_out = nxt[0] - cur[0], nxt[1] - cur[1]
        same_axis = (dx_in == 0 and dx_out == 0) or (dy_in == 0 and dy_out == 0)
        same_sign = (dx_in * dx_out + dy_in * dy_out) > 0
        if not (same_axis and same_sign):
            out.append(cur)
    return out


def _to_cell_local(x: float, y: float, cell_origin: tuple[float, float]) -> tuple[float, float]:
    return x - cell_origin[0], y - cell_origin[1]


def _require_in_bounds(x: float, y: float, cell_size_m: float) -> None:
    if not (0 <= x <= cell_size_m and 0 <= y <= cell_size_m):
        raise FeatureOutOfBounds(f"point ({x}, {y}) outside [0, {cell_size_m}]^2")
```

- [ ] **Step 9.4: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: 7 passed.

- [ ] **Step 9.5: Commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/encode.py tests/tokenizer/test_encode.py
git commit -m "feat(tokenizer): encode rectangular buildings and axis-aligned land use polygons"
```

---

## Task 10: Encoder — LineString with `<EXIT>`

**Files:**
- Modify: `src/cfm/tokenizer/encode.py`
- Modify: `tests/tokenizer/test_encode.py`

- [ ] **Step 10.1: Add failing tests for LineString with exit**

Append to `tests/tokenizer/test_encode.py`:

```python
def _line(coords: list[list[float]], cls: str = "R_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def test_road_crossing_east_edge_emits_exit(vocab: Vocabulary) -> None:
    # Road from (0,125) to (250,125). 250m = 32*7 + 16 + 8 + 2.
    out = encode_cell(_fc(_line([[0, 125], [250, 125]])),
                      cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    t = vocab.token_to_id
    assert t["EXIT"] in out.tokens
    assert t["ANCHOR_X_0"] in out.tokens
    assert t["ANCHOR_Y_125"] in out.tokens
    # MOVE_E_32 must appear at least 7 times.
    move_e_32 = t["MOVE_E_32"]
    assert sum(1 for tok in out.tokens if tok == move_e_32) == 7


def test_internal_road_no_exit(vocab: Vocabulary) -> None:
    out = encode_cell(_fc(_line([[20, 20], [40, 20]])),
                      cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    assert vocab.token_to_id["EXIT"] not in out.tokens


def test_diagonal_segment_raises(vocab: Vocabulary) -> None:
    with pytest.raises(UnsupportedGeometry):
        encode_cell(_fc(_line([[0, 0], [100, 100]])),
                    cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
```

- [ ] **Step 10.2: Run tests; expect failure**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: the three new tests fail with `UnsupportedGeometry: Phase 0 does not yet handle geometry type 'LineString'`.

- [ ] **Step 10.3: Add LineString handler**

Edit `src/cfm/tokenizer/encode.py`. In `_encode_feature`, add a `LineString` branch and a new helper:

```python
def _encode_feature(
    feature: dict,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    cls = feature["properties"]["class"]
    if cls not in vocab.token_to_id:
        raise UnsupportedFeatureClass(f"unknown class {cls!r}")
    geom = feature["geometry"]
    gtype = geom["type"]
    if gtype == "Point":
        body = _encode_point(geom["coordinates"], cell_origin, cell_size_m, vocab)
    elif gtype == "Polygon":
        body = _encode_polygon(cls, geom["coordinates"], cell_origin, cell_size_m, vocab)
    elif gtype == "LineString":
        body = _encode_linestring(geom["coordinates"], cell_origin, cell_size_m, vocab)
    else:
        raise UnsupportedGeometry(f"Phase 0 does not yet handle geometry type {gtype!r}")
    return [
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id[cls],
        *body,
        vocab.token_to_id["FEATURE_END"],
    ]


def _encode_linestring(
    coordinates: list[list[float]],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    vertices_local = [_to_cell_local(p[0], p[1], cell_origin) for p in coordinates]
    if len(vertices_local) < 2:
        raise UnsupportedGeometry("line must have at least 2 vertices")
    for x, y in vertices_local:
        _require_in_bounds(x, y, cell_size_m)
    vertices_local = _drop_collinear_open(vertices_local)
    ax, ay = vertices_local[0]
    body: list[TokenId] = [
        vocab.token_to_id[f"ANCHOR_X_{int(round(ax))}"],
        vocab.token_to_id[f"ANCHOR_Y_{int(round(ay))}"],
    ]
    for i in range(len(vertices_local) - 1):
        x0, y0 = vertices_local[i]
        x1, y1 = vertices_local[i + 1]
        body.extend(_encode_axis_aligned_segment(x1 - x0, y1 - y0, vocab))
    end_x, end_y = vertices_local[-1]
    if _on_cell_boundary(end_x, end_y, cell_size_m):
        body.append(vocab.token_to_id["EXIT"])
    return body


def _drop_collinear_open(verts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove interior collinear vertices from an open polyline (keeps endpoints)."""
    if len(verts) < 3:
        return verts
    out: list[tuple[float, float]] = [verts[0]]
    for i in range(1, len(verts) - 1):
        prev = verts[i - 1]
        cur = verts[i]
        nxt = verts[i + 1]
        dx_in, dy_in = cur[0] - prev[0], cur[1] - prev[1]
        dx_out, dy_out = nxt[0] - cur[0], nxt[1] - cur[1]
        same_axis = (dx_in == 0 and dx_out == 0) or (dy_in == 0 and dy_out == 0)
        same_sign = (dx_in * dx_out + dy_in * dy_out) > 0
        if not (same_axis and same_sign):
            out.append(cur)
    out.append(verts[-1])
    return out


def _on_cell_boundary(x: float, y: float, cell_size_m: float) -> bool:
    return x == 0 or x == cell_size_m or y == 0 or y == cell_size_m
```

- [ ] **Step 10.4: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_encode.py -v`
Expected: 10 passed.

- [ ] **Step 10.5: Commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/encode.py tests/tokenizer/test_encode.py
git commit -m "feat(tokenizer): encode LineStrings with <EXIT> marker on boundary"
```

---

## Task 11: Decoder — Point + skeleton (TDD)

**Files:**
- Create: `src/cfm/tokenizer/decode.py`
- Modify: `src/cfm/tokenizer/__init__.py`
- Create: `tests/tokenizer/test_decode.py`

- [ ] **Step 11.1: Write failing tests**

Create `tests/tokenizer/test_decode.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    CellTokens,
    Vocabulary,
    VocabularyMismatch,
)
from cfm.tokenizer.decode import decode_cell
from cfm.tokenizer.encode import encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def test_decode_empty_returns_empty_collection(vocab: Vocabulary) -> None:
    encoded = encode_cell(_fc(), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert decoded == {"type": "FeatureCollection", "features": []}


def test_decode_point(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "POI_restaurant"},
        "geometry": {"type": "Point", "coordinates": [50, 80]},
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert decoded["features"][0]["properties"]["class"] == "POI_restaurant"
    assert decoded["features"][0]["geometry"]["type"] == "Point"
    assert decoded["features"][0]["geometry"]["coordinates"] == [50.0, 80.0]


def test_unknown_token_id_raises(vocab: Vocabulary) -> None:
    bad = CellTokens(tokens=(999999,), cell_origin=(0.0, 0.0), cell_size_m=250.0)
    with pytest.raises(VocabularyMismatch):
        decode_cell(bad, vocab=vocab)
```

- [ ] **Step 11.2: Run tests; expect failure**

Run: `uv run pytest tests/tokenizer/test_decode.py -v`
Expected: import error.

- [ ] **Step 11.3: Implement decoder skeleton + Point**

Create `src/cfm/tokenizer/decode.py`:

```python
from __future__ import annotations

from cfm.tokenizer.encode import CellTokens
from cfm.tokenizer.errors import UnsupportedGeometry, VocabularyMismatch
from cfm.tokenizer.vocabulary import Vocabulary

GeoJSON = dict


def decode_cell(tokens: CellTokens, *, vocab: Vocabulary) -> GeoJSON:
    names = [_lookup(tid, vocab) for tid in tokens.tokens]
    cursor = _Cursor(names)
    cursor.expect("BOS")
    cursor.expect("CELL")
    features: list[dict] = []
    while cursor.peek() != "END_CELL":
        features.append(_decode_feature(cursor, tokens.cell_origin, tokens.cell_size_m))
    cursor.expect("END_CELL")
    cursor.expect("EOS")
    return {"type": "FeatureCollection", "features": features}


def _lookup(tid: int, vocab: Vocabulary) -> str:
    if not 0 <= tid < len(vocab):
        raise VocabularyMismatch(f"token id {tid} out of vocabulary range")
    return vocab.id_to_token[tid]


class _Cursor:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self._i = 0

    def peek(self) -> str:
        if self._i >= len(self._names):
            raise VocabularyMismatch("unexpected end of token sequence")
        return self._names[self._i]

    def take(self) -> str:
        name = self.peek()
        self._i += 1
        return name

    def expect(self, expected: str) -> None:
        got = self.take()
        if got != expected:
            raise VocabularyMismatch(f"expected {expected!r}, got {got!r}")


def _decode_feature(
    cursor: _Cursor,
    cell_origin: tuple[float, float],
    cell_size_m: float,
) -> dict:
    cursor.expect("FEATURE_START")
    cls = cursor.take()
    body: list[str] = []
    while cursor.peek() != "FEATURE_END":
        body.append(cursor.take())
    cursor.expect("FEATURE_END")
    return _materialise_feature(cls, body, cell_origin, cell_size_m)


def _materialise_feature(
    cls: str,
    body: list[str],
    cell_origin: tuple[float, float],
    cell_size_m: float,
) -> dict:
    # Phase 0 dispatch by class prefix.
    has_exit = body and body[-1] == "EXIT"
    if has_exit:
        body = body[:-1]
    anchor_x, anchor_y, rest = _read_anchor(body, cell_origin)
    if not rest and not has_exit and cls.startswith(("POI_",)):
        return _point_feature(cls, anchor_x, anchor_y)
    if cls.startswith(("R_",)):
        return _line_feature(cls, anchor_x, anchor_y, rest, has_exit, cell_size_m, cell_origin)
    if cls.startswith(("B_", "L_")):
        if has_exit:
            raise UnsupportedGeometry(f"<EXIT> not valid for class {cls}")
        return _polygon_feature(cls, anchor_x, anchor_y, rest, cell_origin)
    raise UnsupportedGeometry(f"unknown class prefix for {cls!r}")


def _read_anchor(
    body: list[str],
    cell_origin: tuple[float, float],
) -> tuple[float, float, list[str]]:
    if len(body) < 2 or not body[0].startswith("ANCHOR_X_") or not body[1].startswith("ANCHOR_Y_"):
        raise UnsupportedGeometry("feature body must start with anchor X/Y pair")
    x = float(body[0].removeprefix("ANCHOR_X_")) + cell_origin[0]
    y = float(body[1].removeprefix("ANCHOR_Y_")) + cell_origin[1]
    return x, y, body[2:]


def _point_feature(cls: str, x: float, y: float) -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def _line_feature(
    cls: str,
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
    has_exit: bool,
    cell_size_m: float,
    cell_origin: tuple[float, float],
) -> dict:
    coords = _apply_moves_as_polyline(anchor_x, anchor_y, moves)
    if has_exit:
        end_x, end_y = coords[-1]
        local_x = end_x - cell_origin[0]
        local_y = end_y - cell_origin[1]
        if not _on_cell_boundary(local_x, local_y, cell_size_m):
            raise UnsupportedGeometry("<EXIT> token but final vertex not on cell boundary")
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": [list(p) for p in coords]},
    }


def _polygon_feature(
    cls: str,
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
    cell_origin: tuple[float, float],
) -> dict:
    coords = _apply_moves_as_polyline(anchor_x, anchor_y, moves)
    # Closure check: final cursor must equal anchor (within 1m grid).
    if (round(coords[-1][0]) != round(anchor_x)) or (round(coords[-1][1]) != round(anchor_y)):
        raise UnsupportedGeometry("polygon move sequence does not return to anchor")
    # GeoJSON polygon ring repeats the first vertex; drop the moves' trailing duplicate.
    ring = [list(p) for p in coords]
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


_DIR_TO_DELTA: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}


def _apply_moves_as_polyline(
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
) -> list[tuple[float, float]]:
    """Apply move tokens, collapsing consecutive same-direction moves into one segment.

    Returns the vertex list starting with the anchor.
    """
    coords: list[tuple[float, float]] = [(anchor_x, anchor_y)]
    if not moves:
        return coords
    cur_x, cur_y = anchor_x, anchor_y
    seg_dir: str | None = None
    seg_len = 0
    for tok in moves:
        if not tok.startswith("MOVE_"):
            raise UnsupportedGeometry(f"expected MOVE_ token, got {tok!r}")
        _, direction, step_s = tok.split("_")
        if direction not in _DIR_TO_DELTA:
            raise UnsupportedGeometry(f"Phase 0 supports cardinal moves only; got {direction!r}")
        step = int(step_s)
        if seg_dir is None or direction == seg_dir:
            seg_dir = direction
            seg_len += step
        else:
            # Close the previous segment.
            dx, dy = _DIR_TO_DELTA[seg_dir]
            cur_x += dx * seg_len
            cur_y += dy * seg_len
            coords.append((cur_x, cur_y))
            seg_dir = direction
            seg_len = step
    # Close the trailing segment.
    if seg_dir is not None:
        dx, dy = _DIR_TO_DELTA[seg_dir]
        cur_x += dx * seg_len
        cur_y += dy * seg_len
        coords.append((cur_x, cur_y))
    return coords


def _on_cell_boundary(x: float, y: float, cell_size_m: float) -> bool:
    return x == 0 or x == cell_size_m or y == 0 or y == cell_size_m
```

Update `src/cfm/tokenizer/__init__.py`:

```python
"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.decode import decode_cell
from cfm.tokenizer.encode import CellTokens, encode_cell
from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
from cfm.tokenizer.geometry import geometric_equal
from cfm.tokenizer.vocabulary import Vocabulary

__all__ = [
    "CellTokens",
    "FeatureOutOfBounds",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
    "decode_cell",
    "encode_cell",
    "geometric_equal",
]
```

- [ ] **Step 11.4: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_decode.py -v`
Expected: 3 passed.

- [ ] **Step 11.5: Commit**

```bash
uv run ruff format src tests && uv run ruff check src tests
git add src/cfm/tokenizer/decode.py src/cfm/tokenizer/__init__.py tests/tokenizer/test_decode.py
git commit -m "feat(tokenizer): decode_cell handles Point features and skeleton dispatch"
```

---

## Task 12: Decoder — Polygon and LineString tests through round-trip pairing

**Files:**
- Modify: `tests/tokenizer/test_decode.py`

(Decoder already handles Polygon + LineString from Task 11. This task locks the per-shape behaviour with focused tests so regressions are easy to localise.)

- [ ] **Step 12.1: Add Polygon and LineString tests**

Append to `tests/tokenizer/test_decode.py`:

```python
def test_decode_rectangle_building(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "B_residential"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]],
        },
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    geom = decoded["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    assert ring[0] == [40.0, 40.0]
    # GeoJSON-valid: ring is closed
    assert ring[0] == ring[-1]


def test_decode_road_with_exit(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "R_residential"},
        "geometry": {"type": "LineString", "coordinates": [[0, 125], [250, 125]]},
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    coords = decoded["features"][0]["geometry"]["coordinates"]
    assert coords[0] == [0.0, 125.0]
    assert coords[-1] == [250.0, 125.0]


def test_decode_unclosed_polygon_raises(vocab: Vocabulary) -> None:
    # Hand-craft a token sequence whose polygon moves don't close.
    t = vocab.token_to_id
    tokens = (
        t["BOS"], t["CELL"],
        t["FEATURE_START"], t["B_residential"],
        t["ANCHOR_X_40"], t["ANCHOR_Y_40"],
        t["MOVE_E_16"],  # cursor at (56, 40) — no closure
        t["FEATURE_END"],
        t["END_CELL"], t["EOS"],
    )
    bad = CellTokens(tokens=tokens, cell_origin=(0.0, 0.0), cell_size_m=250.0)
    with pytest.raises(Exception) as excinfo:
        decode_cell(bad, vocab=vocab)
    # Expect UnsupportedGeometry (not VocabularyMismatch) — the IDs are valid.
    from cfm.tokenizer.errors import UnsupportedGeometry
    assert isinstance(excinfo.value, UnsupportedGeometry)
```

- [ ] **Step 12.2: Run tests; expect pass**

Run: `uv run pytest tests/tokenizer/test_decode.py -v`
Expected: 6 passed.

- [ ] **Step 12.3: Commit**

```bash
git add tests/tokenizer/test_decode.py
git commit -m "test(tokenizer): lock decoder behaviour for polygons and exit-lines"
```

---

## Task 13: End-to-end round-trip + determinism against the fixture

**Files:**
- Create: `tests/tokenizer/test_round_trip.py`
- Create: `tests/tokenizer/test_determinism.py`

- [ ] **Step 13.1: Write the round-trip test**

Create `tests/tokenizer/test_round_trip.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import (
    Vocabulary,
    decode_cell,
    encode_cell,
    geometric_equal,
)


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _load_fixture(fixtures_dir: Path) -> dict:
    with (fixtures_dir / "single_cell" / "input.geojson").open() as f:
        return json.load(f)


def test_single_cell_fixture_round_trips(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert geometric_equal(original, decoded, tol_m=0.5)


def test_round_trip_preserves_feature_count(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert len(decoded["features"]) == len(original["features"])


def test_round_trip_preserves_classes(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    orig_classes = sorted(f["properties"]["class"] for f in original["features"])
    decoded_classes = sorted(f["properties"]["class"] for f in decoded["features"])
    assert orig_classes == decoded_classes
```

- [ ] **Step 13.2: Write the determinism test**

Create `tests/tokenizer/test_determinism.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import Vocabulary, encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def test_encoding_fixture_is_deterministic(vocab: Vocabulary, fixtures_dir: Path) -> None:
    with (fixtures_dir / "single_cell" / "input.geojson").open() as f:
        original = json.load(f)
    a = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    b = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    assert a.tokens == b.tokens
    assert a.cell_origin == b.cell_origin
    assert a.cell_size_m == b.cell_size_m
```

- [ ] **Step 13.3: Run both new test files; expect pass**

Run: `uv run pytest tests/tokenizer/test_round_trip.py tests/tokenizer/test_determinism.py -v`
Expected: 4 passed.

- [ ] **Step 13.4: Commit**

```bash
git add tests/tokenizer/test_round_trip.py tests/tokenizer/test_determinism.py
git commit -m "test(tokenizer): end-to-end round-trip and determinism on single-cell fixture"
```

---

## Task 14: Negative-fixture error tests

**Files:**
- Create: `tests/tokenizer/test_errors.py`

- [ ] **Step 14.1: Write the negative test suite**

Create `tests/tokenizer/test_errors.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    Vocabulary,
    encode_cell,
)


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _load(fixtures_dir: Path, name: str) -> dict:
    with (fixtures_dir / "degenerate" / name).open() as f:
        return json.load(f)


@pytest.mark.parametrize(
    ("fixture_name", "expected_error"),
    [
        ("non_rectangular_building.geojson", UnsupportedGeometry),
        ("unknown_class.geojson", UnsupportedFeatureClass),
        ("out_of_bounds.geojson", FeatureOutOfBounds),
    ],
)
def test_degenerate_fixtures_raise_specific_error(
    fixture_name: str,
    expected_error: type[Exception],
    vocab: Vocabulary,
    fixtures_dir: Path,
) -> None:
    geo = _load(fixtures_dir, fixture_name)
    with pytest.raises(expected_error):
        encode_cell(geo, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
```

- [ ] **Step 14.2: Run; expect pass**

Run: `uv run pytest tests/tokenizer/test_errors.py -v`
Expected: 3 passed (parametrised).

- [ ] **Step 14.3: Commit**

```bash
git add tests/tokenizer/test_errors.py
git commit -m "test(tokenizer): degenerate fixtures raise specific TokenizerError subclasses"
```

---

## Task 15: Phase-0 smoke script

**Files:**
- Create: `scripts/smoke.py`

- [ ] **Step 15.1: Write the smoke script**

Create `scripts/smoke.py`:

```python
"""Phase 0 smoke: load the single-cell fixture, round-trip, print a deterministic summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from cfm.tokenizer import (
    Vocabulary,
    decode_cell,
    encode_cell,
    geometric_equal,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    vocab = Vocabulary.load(repo_root / "configs" / "tokenizer" / "vocab_phase0.yaml")
    fixture_path = repo_root / "tests" / "fixtures" / "single_cell" / "input.geojson"
    with fixture_path.open() as f:
        original = json.load(f)

    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    equal = geometric_equal(original, decoded, tol_m=0.5)

    token_bytes = ",".join(str(t) for t in encoded.tokens).encode("utf-8")
    digest = hashlib.sha256(token_bytes).hexdigest()[:16]

    print("Phase 0 smoke")
    print(f"  vocabulary size:     {len(vocab)}")
    print(f"  fixture features:    {len(original['features'])}")
    print(f"  encoded token count: {len(encoded.tokens)}")
    print(f"  token id sha256/16:  {digest}")
    print(f"  geometric_equal:     {equal}")

    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 15.2: Run the smoke**

Run: `uv run python scripts/smoke.py`
Expected output (token count and hash may differ slightly if the dyadic decomposition changes; `geometric_equal: True` is the contract):

```
Phase 0 smoke
  vocabulary size:     579
  fixture features:    4
  encoded token count: <some number>
  token id sha256/16:  <16-hex digits>
  geometric_equal:     True
```

Exit code 0.

- [ ] **Step 15.3: Commit**

```bash
git add scripts/smoke.py
git commit -m "feat(scripts): add Phase 0 smoke script"
```

---

## Task 16: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 16.1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.11
      - name: Install project
        run: uv sync --all-extras
      - name: Lint
        run: |
          uv run ruff check .
          uv run ruff format --check .
      - name: Test
        run: uv run pytest -v
      - name: Smoke
        run: uv run python scripts/smoke.py
```

- [ ] **Step 16.2: Verify all checks pass locally one more time**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv run python scripts/smoke.py
```

Expected: all four succeed; `pytest` reports the full Phase-0 test count.

- [ ] **Step 16.3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run ruff + pytest + smoke on push/PR"
```

---

## Task 17: Phase-0 done check, merge to main

**Files:**
- Modify: `README.md`

- [ ] **Step 17.1: Run the Phase-0 done check verbatim**

Run:

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv run python scripts/smoke.py
```

All four must succeed. If anything fails, fix and re-test before continuing.

- [ ] **Step 17.2: Expand `README.md` with the success criteria**

Replace `README.md` with:

```markdown
# Bonzai-OSM

A generative foundation model for city geometry. See `PRD.md` for goals and `CLAUDE.md` for collaboration rules.

## Phase 0 quick start

Phase 0 ships a single-cell GeoJSON tokenizer with a geometric-equivalence round-trip, the canonical Phase-0 vocabulary, and the three negative-fixture failure paths.

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -v
uv run python scripts/smoke.py
```

Phase 0 is "done" when all four commands succeed on a clean Mac or Linux machine and CI is green.

## Repository layout

See `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md` for the locked Phase-0 design and `docs/superpowers/plans/2026-05-15-phase-0-tokenizer-roundtrip.md` for the implementation plan.

## Next

Phase 1 takes over once Phase 0 is signed off: Overture loading, multi-cell tiles, boundary contracts, deterministic stitching, and the full ~100-class vocabulary derived from frequency analysis.
```

- [ ] **Step 17.3: Commit the README update**

```bash
git add README.md
git commit -m "docs: document Phase 0 done criteria in README"
```

- [ ] **Step 17.4: Merge to main**

```bash
git checkout main
git merge --no-ff phase-0-tokenizer -m "merge: Phase 0 tokenizer round-trip complete"
git log --oneline
```

Expected: clean fast-forward-free merge; `git log --oneline` shows the merge commit plus the feature commits.

- [ ] **Step 17.5: Final Phase-0 done check on main**

Run, one last time, from a fresh shell:

```bash
uv sync --all-extras
uv run pytest -v
uv run python scripts/smoke.py
```

If green, Phase 0 is done. The decision point at the end of PRD §11 is satisfied.

---

## Self-review notes

**Spec coverage check:**

| Spec section | Implemented in |
|---|---|
| §2 in-scope: fixture, package, vocab spec + YAML, geometric_equal, three Phase-0 tests, scaffolding | Tasks 1–7, 13, 14 |
| §3 load-bearing: single-cell scope | All tasks; no stitching |
| §3 load-bearing: cell-local metres | encode_cell `cell_origin` param (Task 8+) |
| §3 load-bearing: 1 m anchor grid, X+Y split, 500 anchors | Task 2 YAML, Task 3 loader |
| §3 load-bearing: 8 × 6 = 48 moves | Task 2 YAML |
| §3 load-bearing: geometric equivalence, 0.5 m default | Task 7 |
| §3 load-bearing: loud failures | Task 4 errors, Task 14 tests |
| §4 vocab counts | Task 2 YAML, Task 3 test asserts 579 |
| §5 interface signatures | Tasks 3, 7, 8, 11 |
| §6 fixture (single_cell + degenerate) | Tasks 5, 6 |
| §7 three Phase-0 tests | Tasks 13 (round-trip + determinism), 14 (errors) |
| §8 scaffolding | Tasks 1, 2, 15, 16 |
| §9 done criteria | Task 17 |

**Placeholder scan:** no "TBD", no "implement later", every code step shows the actual code, every command shows expected output. ✓

**Type consistency:** `CellTokens`, `Vocabulary`, error classes, `encode_cell`, `decode_cell`, `geometric_equal` are named identically across all tasks. ✓

**Note on Task 12:** decoder Polygon and LineString are implemented in Task 11; Task 12 only adds locking tests because the round-trip in Task 13 covers them implicitly. Splitting tests out keeps regressions easy to localise.
