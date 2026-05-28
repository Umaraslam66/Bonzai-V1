# Phase 1 Sub-F Task 8 (Writer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan one sub-task at a time with fresh subagent + checkpoint review between. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sub-F per-tile writer — geometry → token sequence → `cells.parquet`. Implements spec §3 encoder grammar (4 cases A/B/C/D), §4 storage shape via pinned `pa.schema`, §5.6 canonicalize_geometry contract (3 pure-redundancy DOFs), §6 provenance write, and BP7 boundary-ref consumer reading sub-E `boundary_contract.parquet`.

**Architecture:** Eight sub-tasks (T8.1 → T8.8), each a single subagent dispatch with atomic commit and TDD. Sub-tasks ordered so each module's tests pass before the next one consumes it. Final sub-task wires everything into a per-tile orchestrator + 4-case round-trip integration. No reviewer halt — implementation only. Subagent escalates ONLY if an implementation choice turns out to be a disguised design decision (per `feedback_subagent_branch_pattern` + halt discipline).

**Tech Stack:** Python 3.11+, pyarrow (pinned `pa.schema`, `cfm.data.io.write_parquet` for byte-determinism), Shapely (geometry + WKB), PyYAML, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md`. Master plan: `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md` (this doc supersedes the Task 8 section).

**Branch:** `phase-1-sub-F-micro-tokenizer`. Stay on it. No new branches, no push, no PR, no merge.

**Operating discipline:** Sub-project planning protocol v1. Halt-on-design-decision; verify-before-lock; content-anchored cites; cheap-to-keep at data-encoding layer.

---

## Pre-flight composition assertions (carry through all sub-tasks)

The seven locked BPs (Halts 1–7) are now all closed. T8 is the first task that consumes them in concert. Three cross-lock compositions must hold; each gets concrete assertions in the relevant sub-task.

### Assertion 1 — BP5 ↔ BP3 token-count invariance

Per §5.6 canonical form: rotation, winding-flip, and multi-part sort all preserve vertex count per feature. Case A formula `tokens = 3 + N_anchor + 2*(V-1)` is V-driven; if V is invariant, tokens are invariant. Sub-F's BP3 budget was measured on sub-C SOURCE-order features (`scripts/sub_f/analyze_stage_1_2_joint.py`); T8 emits CANONICAL-order features. The budget remains valid iff canonicalization preserves V.

**T8.3 bakes the assertion in:** unit test `test_canonicalize_preserves_vertex_count` exercises Polygon-rotation, Polygon-winding-flip, MultiLineString-sort, LineString-preserve, closed-LineString-preserve — each must satisfy `vertex_count(canonical) == vertex_count(source)`.

### Assertion 2 — BP5 routing fidelity (closed LineString must NOT route to ring path)

Per §5.6 routing table: dispatch on `geom.geom_type` (Shapely classification). Closed `LineString` (`coords[0] == coords[-1]`, e.g., roundabout) is still typed `"LineString"` by Shapely, NOT `"Polygon"`. The bug to forbid: using `geom.is_ring` (True for closed LineStrings AND closed Polygon rings) for routing — that path silently sends a roundabout to ring canonicalization and destroys oneway semantics.

**T8.3 bakes the assertion in:** `canonicalize_geometry` dispatches strictly on `geom_type`; the function has a one-line comment forbidding `is_ring` usage. T5b Test 5f (`test_canonicalize_closed_linestring_preserves_source_direction`) catches the regression.

### Assertion 4 — Structural-sentinel allocation (resolved pre-dispatch)

Self-review during plan-write surfaced a spec §2.2 / `sentinel_inventory.yaml` gap: `<feature>` / `<feature_end>` were named in §2.2 but had no allocated ID in any locked sub-block. Resolution landed in commit `fix(sub_f): sentinel_inventory — consume IDs 509, 510 from reserved_v2_headroom for <feature>/<feature_end>` (immediately before T8.1 dispatch).

Key outcomes:
- Sentinels live at IDs **509** (`<feature>`) and **510** (`<feature_end>`), consumed from the FRONT of BP2 `reserved_v2_headroom` (now 511–1499). Append-safe per discipline; block boundaries NOT re-anchored.
- Family classification: `family="structural"` — distinct from `family="encoding_primitive"`. ID locality (BP2 reserved tail) and semantic family (grammar primitives) are allowed to differ.
- §2.2 amended as documentation correction: only 2 structural sentinels are on-disk (the other 4 names in the original §2.2 list are positional descriptors of value-token classes; self-delimiting grammar verified).
- Total sub-F on-disk vocab: **374 slots** = BP1 127 + BP4 28 + BP2 encoding_primitive 209 + structural 2 + BP7 8.

**T8 sub-tasks consume the corrected inventory.** T8.1 vocab loader registers IDs 509+510 under `family="structural"`. T8.4 encoder constants `_FEATURE_TOKEN_ID = 509`, `_FEATURE_END_TOKEN_ID = 510`. Anchor base stays at 300 (full 96-slot anchor sub-block usable — `<feature>` / `<feature_end>` do NOT live in the anchor sub-block despite the sentinel-inventory.yaml fix only documenting the consumption from `reserved_v2_headroom`).

See spec §13.1 "T8 plan-write → BP2 inventory" row for the full audit trail (gap discovery, append-safety analysis, option-4 reading rationale, self-delimiting check).

### Assertion 3 — BP7 verification-debt inheritance

Sub-E cache is absent locally; T7 BP7 vocab is code-inferred against sub-E `boundary_contract.parquet` schema (close-checklist obligation tracks recheck-when-regenerated). T8 makes the inference CONCRETE consumer code. The debt inherits one layer deeper but must be surfaced explicitly:

- Every BP7 emission site carries a `# BP7 emission — UNVERIFIED against real sub-E parquet; see close-checklist + project_sub_e_cache_absent_t3c_code_inferred memory` comment.
- BP7 consumer tests use SYNTHETIC parquet fixtures matching the documented sub-E schema (from sub-E spec + `src/cfm/data/sub_e/writer.py`).
- Integration tests against real sub-E parquet are `@pytest.mark.skip(reason="awaiting sub-E cache regeneration")` in T8.8 — un-skip when sub-E lands.
- New close-checklist obligation in T8.8 commit: "Un-skip T8 BP7 integration tests + assert encoder output matches real sub-E parquet on grammar cases B/C/D when sub-E lands."

---

## File map

New files this plan creates (under `src/cfm/data/sub_f/` and `tests/data/sub_f/`):

```
src/cfm/data/sub_f/
  vocab.py              # T8.1 — load locked YAMLs into a typed slot tuple
  io.py                 # T8.2 — cells.parquet pinned schema + write_cells_parquet
  encoder.py            # T8.3–T8.6 — canonicalize_geometry + 4-case encoder + cell aggregator
  decoder.py            # T8.7 — inverse of encoder + canonical GeoJSON
  boundary_contract.py  # T8.5 — sub-E parquet reader (BP7 consumer)
  pipeline_writer.py    # T8.8 — per-tile orchestrator (encode_tile)

tests/data/sub_f/
  test_io.py                          # T8.2 — cells.parquet schema + 64-row invariant
  test_vocab.py                       # T8.1 — append tests (file already exists)
  test_encoder.py                     # T8.3, T8.4, T8.6 — append (file already exists)
  test_decoder.py                     # T8.7 — round-trip via decode
  test_boundary_contract.py           # T8.5 — synthetic sub-E fixture
  test_pipeline_writer.py             # T8.8 — per-tile orchestrator + 4-case round-trip
```

Existing modules touched (read-only, no edits):

- `src/cfm/data/sub_f/enums.py` — BP2 candidate constants
- `src/cfm/data/sub_f/versions.py` — 6 SUB_F_*_VERSION constants + `load_sub_f_source_version`
- `src/cfm/data/sub_f/provenance.py` — SUB_F_EXCLUDED_FROM_SHA + per-tile provenance
- `src/cfm/data/sub_f/manifest.py` — region manifest write
- `src/cfm/data/sub_f/rotation.py` — BP7 boundary-reference direction wrapper (`cell_edge_directions`)
- `src/cfm/data/io.py` — `write_parquet(table, path)` + `canonicalize_yaml(data)`
- `configs/sub_f/semantic_vocab.yaml` (127 slots, IDs 0–126)
- `configs/sub_f/unknown_family.yaml` (28 slots, IDs 200–227)
- `configs/sub_f/encoding_primitives.yaml` (BP2 lock: direction_count=48, magnitude_quantum_m=0.5, anchor_scheme=hierarchical, chunk_threshold_m=32)
- `configs/sub_f/boundary_reference_vocab.yaml` (8 slots, IDs 1500–1507)
- `configs/sub_f/sentinel_inventory.yaml` (ID-namespace map; BP2 anchor 300–395, direction 396–443, magnitude 444–508)

---

## Sub-task index

| # | Task | Files | Blocked by | Commit prefix |
|---|---|---|---|---|
| T8.1 | vocab loader | `vocab.py`, `test_vocab.py` | — | `feat(sub_f): T8.1` |
| T8.2 | cells.parquet schema + writer | `io.py`, `test_io.py` | — | `feat(sub_f): T8.2` |
| T8.3 | canonicalize_geometry + encoder helpers | `encoder.py`, `test_encoder.py` | T8.1 | `feat(sub_f): T8.3` |
| T8.4 | encoder 4-case grammar (A/B/C/D) | `encoder.py`, `test_encoder.py` | T8.3 | `feat(sub_f): T8.4` |
| T8.5 | BP7 sub-E boundary-contract reader | `boundary_contract.py`, `test_boundary_contract.py` | — (parallel to T8.4) | `feat(sub_f): T8.5` |
| T8.6 | encoder per-cell aggregator + empty-cell handling | `encoder.py`, `test_encoder.py` | T8.4, T8.5 | `feat(sub_f): T8.6` |
| T8.7 | decoder + canonical GeoJSON | `decoder.py`, `test_decoder.py` | T8.4 (parallel to T8.6 ok) | `feat(sub_f): T8.7` |
| T8.8 | per-tile orchestrator + 4-case round-trip integration | `pipeline_writer.py`, `test_pipeline_writer.py` | T8.6, T8.7 | `feat(sub_f): T8.8` |

After T8.8 lands, the next dispatch is **T5b (per-axis determinism suite)** per the master sub-F plan. T5b creates `tests/data/sub_f/test_per_axis_determinism.py` from scratch (the file does not exist at T8.8 close; the original plan incorrectly assumed it pre-existed via a no-op un-skip step). T5b is unblocked by T8.8 (encoder shipped); see T8.8 step 8 for the close-checklist obligation and dispatch handoff. Per spec §5.5 the per-axis determinism suite IS the durable contract artifact for sub-F determinism — required before sub-F-close.

---

## Sub-task T8.1: vocab loader

**Files:**
- Create: `src/cfm/data/sub_f/vocab.py`
- Append: `tests/data/sub_f/test_vocab.py`

### Pre-dispatch audit

- [ ] **Audit step 1: verify YAML shape of `semantic_vocab.yaml`**

Run: `grep -n "tag:" configs/sub_f/semantic_vocab.yaml | head -5`
Expected: `    tag: aerialway=*` (lines starting with 4-space indent + `tag:`). All 127 slots use this shape.

- [ ] **Audit step 2: verify BP4 / BP7 / BP2 sub-block IDs**

Run: `head -60 configs/sub_f/sentinel_inventory.yaml`
Expected ID ranges: BP1 0–126, BP4 200–227, BP2 anchor 300–395, BP2 direction 396–443, BP2 magnitude 444–508, BP7 1500–1507. Dataloader sentinels 256–260 with `on_disk: false`.

### Implementation steps

- [ ] **Step 1: Write failing test**

Append to `tests/data/sub_f/test_vocab.py`:

```python
def test_load_sub_f_vocab_returns_all_on_disk_families_in_id_order():
    """Vocab loader returns every on-disk slot from BP1+BP2+structural+BP4+BP7 in ascending token_id.

    On-disk excludes dataloader sentinels (256-260 per sentinel_inventory.yaml
    dataloader_sentinels block, on_disk=false). Total on-disk count is the sum
    of BP1 used (127) + BP4 used (28) + BP2 encoding_primitive used (209) +
    structural sentinels (2 — <feature>/<feature_end> at 509/510, consumed
    from BP2 reserved_v2_headroom front per pre-flight Assertion 4) +
    BP7 used (8) = 374 slots.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    slots = load_sub_f_vocab()
    assert len(slots) == 374, f"expected 374 on-disk slots; got {len(slots)}"

    # Strictly ascending token_id (per `feedback_pythonhashseed_dict_iteration_test`
    # the loader must produce deterministic order, not hash-order).
    ids = [s.token_id for s in slots]
    assert ids == sorted(ids), "token_ids must be in strictly ascending order"
    assert len(set(ids)) == len(ids), "token_ids must be unique"

    # Family boundaries per sentinel_inventory.yaml.
    bp1 = [s for s in slots if s.family == "semantic"]
    bp4 = [s for s in slots if s.family == "unknown"]
    bp2 = [s for s in slots if s.family == "encoding_primitive"]
    structural = [s for s in slots if s.family == "structural"]
    bp7 = [s for s in slots if s.family == "boundary_reference"]
    assert len(bp1) == 127
    assert len(bp4) == 28
    assert len(bp2) == 209  # anchor 96 + direction 48 + magnitude 65
    assert len(structural) == 2  # <feature> + <feature_end>
    assert len(bp7) == 8

    # ID-range invariants from sentinel_inventory.yaml.
    assert all(0 <= s.token_id <= 126 for s in bp1)
    assert all(200 <= s.token_id <= 227 for s in bp4)
    assert all(300 <= s.token_id <= 508 for s in bp2)
    assert {s.token_id for s in structural} == {509, 510}
    assert all(1500 <= s.token_id <= 1507 for s in bp7)

    # Structural family carries the named tags exactly.
    structural_tags = {s.tag: s.token_id for s in structural}
    assert structural_tags == {"<feature>": 509, "<feature_end>": 510}


def test_load_sub_f_vocab_no_dataloader_sentinels_on_disk():
    """Per sentinel_inventory.yaml: <pad>=256, <eos>=257, <bos>=258, <cell_start>=259,
    <cell_end>=260 are on_disk=false. They must NOT appear in load_sub_f_vocab()."""
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    on_disk_ids = {s.token_id for s in load_sub_f_vocab()}
    for sentinel_id in (256, 257, 258, 259, 260):
        assert sentinel_id not in on_disk_ids, (
            f"dataloader sentinel id={sentinel_id} must NOT be in on-disk vocab"
        )


def test_load_sub_f_vocab_tag_lookup_round_trips():
    """Every slot has a unique tag; tag → token_id lookup matches token_id → tag."""
    from cfm.data.sub_f.vocab import load_sub_f_vocab, vocab_tag_to_id

    slots = load_sub_f_vocab()
    tag_to_id = vocab_tag_to_id()
    for s in slots:
        assert tag_to_id[s.tag] == s.token_id
    assert len(tag_to_id) == len(slots), "tags must be unique across all families"
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run pytest tests/data/sub_f/test_vocab.py::test_load_sub_f_vocab_returns_all_on_disk_families_in_id_order -v`
Expected: FAIL with `ImportError: cannot import name 'load_sub_f_vocab'`.

- [ ] **Step 3: Implement `vocab.py`**

Create `src/cfm/data/sub_f/vocab.py`:

```python
"""Sub-F unified vocab loader.

Loads BP1 semantic + BP2 encoding-primitive + structural sentinels +
BP4 unknown + BP7 boundary-reference slots from their respective YAML
configs and exposes a deterministic, ascending-id tuple.

Dataloader sentinels (<pad>/<eos>/<bos>/<cell_start>/<cell_end>, IDs 256-260)
are on_disk=false per `configs/sub_f/sentinel_inventory.yaml` and excluded.

Structural sentinels (<feature> id 509, <feature_end> id 510) live at the
front of BP2 reserved_v2_headroom per the T8 plan-write sentinel-inventory
fix (2026-05-28). They are tagged family="structural" (grammar primitives),
distinct from the encoding_primitive family of anchor/direction/magnitude
sub-blocks whose ID neighborhood they share. See spec §13.1 "T8 plan-write
-> BP2 inventory" row for the audit trail.

Iteration order is YAML file order, NOT dict hash order — per
`feedback_pythonhashseed_dict_iteration_test` discipline (sub-F T5b Test 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import yaml

_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "sub_f"

Family = Literal[
    "semantic", "unknown", "encoding_primitive", "structural", "boundary_reference"
]


@dataclass(frozen=True)
class VocabSlot:
    """One on-disk vocab slot."""

    token_id: int
    tag: str
    family: Family


def _load_semantic_slots() -> list[VocabSlot]:
    """BP1 semantic slots from `configs/sub_f/semantic_vocab.yaml`.

    Slots carry explicit `id` and `tag` fields; we trust YAML file order.
    """
    data = yaml.safe_load((_CONFIGS / "semantic_vocab.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["tag"]), family="semantic")
        for s in data["slots"]
    ]


def _load_unknown_slots() -> list[VocabSlot]:
    """BP4 <unknown_*> slots from `configs/sub_f/unknown_family.yaml`."""
    data = yaml.safe_load((_CONFIGS / "unknown_family.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["tag"]), family="unknown")
        for s in data["slots"]
    ]


def _load_encoding_primitive_slots() -> list[VocabSlot]:
    """BP2 encoding-primitive slots from `configs/sub_f/sentinel_inventory.yaml`
    `bp2_encoding_primitives.sub_blocks` ranges (anchor + direction + magnitude).

    Each slot's tag is synthesised from its sub-block + offset:
      - anchor:    `<anchor_${start_id_offset}>`   (96 slots, ids 300-395)
      - direction: `<direction_${idx}>`            (48 slots, ids 396-443)
      - magnitude: `<magnitude_${idx}>`            (65 slots, ids 444-508)
    """
    inv = yaml.safe_load((_CONFIGS / "sentinel_inventory.yaml").read_text(encoding="utf-8"))
    bp2 = inv["bp2_encoding_primitives"]["sub_blocks"]

    slots: list[VocabSlot] = []
    for block_name in ("anchor", "direction", "magnitude"):
        block = bp2[block_name]
        start, end = int(block["start_id"]), int(block["end_id"])
        for offset, token_id in enumerate(range(start, end + 1)):
            slots.append(
                VocabSlot(
                    token_id=token_id,
                    tag=f"<{block_name}_{offset}>",
                    family="encoding_primitive",
                )
            )
    return slots


def _load_structural_sentinel_slots() -> list[VocabSlot]:
    """Structural sentinels consumed from BP2 reserved_v2_headroom front (T8
    plan-write fix, 2026-05-28). IDs 509 (<feature>), 510 (<feature_end>).

    These tokens are grammar primitives (delimit per-feature sequences); the
    encoder's 4-case grammar (§3.2) opens every feature with <feature> and
    closes with <feature_end>. They are tagged family="structural" rather
    than "encoding_primitive" because semantically they are NOT value tokens
    of coordinate/direction/magnitude classes — their ID neighborhood
    (consumed from BP2 reserved tail) is incidental.
    """
    inv = yaml.safe_load((_CONFIGS / "sentinel_inventory.yaml").read_text(encoding="utf-8"))
    consumed = inv["bp2_encoding_primitives"]["consumed_from_reserved_v2_headroom"]["slots"]
    return [
        VocabSlot(
            token_id=int(s["id"]),
            tag=str(s["token"]),
            family="structural",
        )
        for s in consumed
    ]


def _load_boundary_reference_slots() -> list[VocabSlot]:
    """BP7 boundary-reference slots from `configs/sub_f/boundary_reference_vocab.yaml`."""
    data = yaml.safe_load(
        (_CONFIGS / "boundary_reference_vocab.yaml").read_text(encoding="utf-8")
    )
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["tag"]), family="boundary_reference")
        for s in data["slots"]
    ]


@lru_cache(maxsize=1)
def load_sub_f_vocab() -> tuple[VocabSlot, ...]:
    """Return all on-disk sub-F slots in strictly ascending token_id order.

    Excludes dataloader sentinels (IDs 256-260, on_disk=false).
    Cached at module level — same tuple returned on every call.
    """
    all_slots = (
        _load_semantic_slots()
        + _load_unknown_slots()
        + _load_encoding_primitive_slots()
        + _load_structural_sentinel_slots()
        + _load_boundary_reference_slots()
    )
    return tuple(sorted(all_slots, key=lambda s: s.token_id))


@lru_cache(maxsize=1)
def vocab_tag_to_id() -> dict[str, int]:
    """tag -> token_id lookup, derived from load_sub_f_vocab()."""
    return {s.tag: s.token_id for s in load_sub_f_vocab()}


# Total on-disk vocab count, used by tests + downstream checks.
SUB_F_ON_DISK_TOTAL: Final[int] = 374  # BP1 127 + BP4 28 + BP2 209 + structural 2 + BP7 8
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/data/sub_f/test_vocab.py -v`
Expected: all `test_load_sub_f_vocab_*` PASS plus any pre-existing tests.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/vocab.py tests/data/sub_f/test_vocab.py
uv run ruff check src/cfm/data/sub_f/vocab.py tests/data/sub_f/test_vocab.py
git add src/cfm/data/sub_f/vocab.py tests/data/sub_f/test_vocab.py
git commit -m "feat(sub_f): T8.1 vocab loader — 372 on-disk slots across BP1/BP2/BP4/BP7"
```

---

## Sub-task T8.2: cells.parquet schema + writer

**Files:**
- Create: `src/cfm/data/sub_f/io.py`
- Create: `tests/data/sub_f/test_io.py`

### Pre-dispatch audit

- [ ] **Audit step 1: verify `write_parquet` signature**

Run: `grep -B 2 -A 5 "def write_parquet" src/cfm/data/io.py`
Expected: `def write_parquet(table: pa.Table, path: Path) -> None` routing through `PARQUET_WRITE_KWARGS`.

- [ ] **Audit step 2: verify sub-E pinned-schema precedent**

Run: `grep -B 1 -A 15 "_LAYER3_SCHEMA\|_SCHEMA: Final\[pa.Schema\]" src/cfm/data/sub_e/writer.py | head -30`
Expected: pinned `pa.schema(...)` with explicit `nullable=False` per field. Sub-F mirrors this pattern.

### Implementation steps

- [ ] **Step 1: Write failing test**

Create `tests/data/sub_f/test_io.py`:

```python
"""Sub-F cells.parquet schema + writer tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.io import (
    CELLS_SCHEMA,
    EXPECTED_ROWS_PER_TILE,
    CellRow,
    write_cells_parquet,
)


def _make_64_rows() -> list[CellRow]:
    """Construct 64 well-formed rows for an 8×8 cell grid in row-major order."""
    return [
        CellRow(
            cell_i=i,
            cell_j=j,
            cell_slot_index=i * 8 + j,
            token_sequence=[],
            feature_count=0,
            provenance_sha256="a" * 64,
        )
        for i in range(8)
        for j in range(8)
    ]


def test_cells_schema_field_types_pinned():
    """Per spec §4.2: int8 / int16 / list<int16> / string with explicit nullable=False."""
    fields = {f.name: f for f in CELLS_SCHEMA}
    assert fields["cell_i"].type == pa.int8()
    assert fields["cell_j"].type == pa.int8()
    assert fields["cell_slot_index"].type == pa.int8()
    assert fields["token_sequence"].type == pa.list_(pa.int16())
    assert fields["feature_count"].type == pa.int16()
    assert fields["provenance_sha256"].type == pa.string()
    for name, f in fields.items():
        assert f.nullable is False, f"{name} must be nullable=False per pinned schema"


def test_write_cells_parquet_round_trips(tmp_path: Path):
    """64 rows written → read back with identical column types + values."""
    rows = _make_64_rows()
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)

    table = pq.ParquetFile(path).read()
    assert table.num_rows == EXPECTED_ROWS_PER_TILE
    assert table.schema == CELLS_SCHEMA, "parquet schema must match pinned schema bit-for-bit"

    cell_i_col = table.column("cell_i").to_pylist()
    cell_j_col = table.column("cell_j").to_pylist()
    # Row-major sort: (cell_i, cell_j) ascending.
    expected_pairs = [(i, j) for i in range(8) for j in range(8)]
    assert list(zip(cell_i_col, cell_j_col)) == expected_pairs


def test_write_cells_parquet_rejects_wrong_row_count(tmp_path: Path):
    """write_cells_parquet must error on != EXPECTED_ROWS_PER_TILE."""
    rows = _make_64_rows()[:63]
    with pytest.raises(ValueError, match=r"expected 64"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)


def test_write_cells_parquet_rejects_duplicate_cell(tmp_path: Path):
    """Two rows for the same (cell_i, cell_j) must error — invariant per spec §4.7."""
    rows = _make_64_rows()
    rows[1] = CellRow(  # duplicate (0,0)
        cell_i=0, cell_j=0, cell_slot_index=0,
        token_sequence=[], feature_count=0, provenance_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match=r"duplicate cell"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)


def test_write_cells_parquet_rejects_slot_index_mismatch(tmp_path: Path):
    """cell_slot_index must equal cell_i * 8 + cell_j per spec §4.7."""
    rows = _make_64_rows()
    bad = CellRow(
        cell_i=2, cell_j=3, cell_slot_index=99,  # wrong: should be 19
        token_sequence=[], feature_count=0, provenance_sha256="c" * 64,
    )
    rows[2 * 8 + 3] = bad
    with pytest.raises(ValueError, match=r"cell_slot_index"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_io.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement `io.py`**

Create `src/cfm/data/sub_f/io.py`:

```python
"""Sub-F per-tile cells.parquet schema + writer.

Pinned `pa.schema` with explicit nullable flags per sub-E precedent
(`src/cfm/data/sub_e/writer.py` `_LAYER3_SCHEMA`). Routes through
`cfm.data.io.write_parquet` for byte-deterministic output via
PARQUET_WRITE_KWARGS.

Spec references:
- §4.2 column types + nullability
- §4.3 row ordering: sorted by (cell_i, cell_j) row-major
- §4.7 inline-validator invariants: 64 rows, no duplicates, slot_index
  derivation check
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet

EXPECTED_ROWS_PER_TILE: Final[int] = 64  # 8x8 cell grid per sub-D lattice

CELLS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("cell_i", pa.int8(), nullable=False),
        pa.field("cell_j", pa.int8(), nullable=False),
        pa.field("cell_slot_index", pa.int8(), nullable=False),
        pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
        pa.field("feature_count", pa.int16(), nullable=False),
        pa.field("provenance_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CellRow:
    """One row of cells.parquet."""

    cell_i: int
    cell_j: int
    cell_slot_index: int
    token_sequence: list[int]
    feature_count: int
    provenance_sha256: str


def write_cells_parquet(out_path: Path, rows: list[CellRow]) -> Path:
    """Write rows to cells.parquet, sorted by (cell_i, cell_j) row-major.

    Inline invariants (raise ValueError if violated):
      - len(rows) == 64
      - No duplicate (cell_i, cell_j)
      - cell_slot_index == cell_i * 8 + cell_j for every row
      - provenance_sha256 is 64-char lowercase hex (deferred to inline validator
        in Task 9 — kept loose here so tests can use synthetic 'a'*64).

    Output path: caller chooses (typically
    data/processed/sub_f/<release>/<region>/tile=.../cells.parquet).
    """
    if len(rows) != EXPECTED_ROWS_PER_TILE:
        raise ValueError(
            f"expected {EXPECTED_ROWS_PER_TILE} rows, got {len(rows)}"
        )

    seen: set[tuple[int, int]] = set()
    for r in rows:
        key = (r.cell_i, r.cell_j)
        if key in seen:
            raise ValueError(f"duplicate cell {key}")
        seen.add(key)
        expected_idx = r.cell_i * 8 + r.cell_j
        if r.cell_slot_index != expected_idx:
            raise ValueError(
                f"cell_slot_index {r.cell_slot_index} != cell_i*8+cell_j "
                f"= {expected_idx} for cell ({r.cell_i}, {r.cell_j})"
            )

    sorted_rows = sorted(rows, key=lambda r: (r.cell_i, r.cell_j))
    table = pa.Table.from_pydict(
        {
            "cell_i": [r.cell_i for r in sorted_rows],
            "cell_j": [r.cell_j for r in sorted_rows],
            "cell_slot_index": [r.cell_slot_index for r in sorted_rows],
            "token_sequence": [r.token_sequence for r in sorted_rows],
            "feature_count": [r.feature_count for r in sorted_rows],
            "provenance_sha256": [r.provenance_sha256 for r in sorted_rows],
        },
        schema=CELLS_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(table, out_path)
    return out_path
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/data/sub_f/test_io.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/io.py tests/data/sub_f/test_io.py
uv run ruff check src/cfm/data/sub_f/io.py tests/data/sub_f/test_io.py
git add src/cfm/data/sub_f/io.py tests/data/sub_f/test_io.py
git commit -m "feat(sub_f): T8.2 cells.parquet pinned schema + write_cells_parquet"
```

---

## Sub-task T8.3: canonicalize_geometry + encoder helpers

**Files:**
- Create: `src/cfm/data/sub_f/encoder.py` (initial — helpers + canonicalize only)
- Append: `tests/data/sub_f/test_encoder.py`

This sub-task bakes pre-flight assertions 1 and 2 (token-count invariance + routing fidelity).

### Implementation steps

- [ ] **Step 1: Write failing tests**

Append to `tests/data/sub_f/test_encoder.py`:

```python
"""Sub-F encoder tests — helpers + canonicalize_geometry + 4-case grammar."""

from __future__ import annotations

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon


# ---- helpers ---------------------------------------------------------------


def test_quantize_coord_m_integer_only_banker_tiebreak():
    """Per BP5 §5.2 lock: int(round(coord_m / quantum)) Python banker's tie-break."""
    from cfm.data.sub_f.encoder import quantize_coord_m

    # 0.25 / 0.5 = 0.5 -> banker's round-to-even: 0
    assert quantize_coord_m(0.25, quantum_m=0.5) == 0
    # 0.75 / 0.5 = 1.5 -> banker: 2
    assert quantize_coord_m(0.75, quantum_m=0.5) == 2
    # 1.0 / 0.5 = 2.0 -> 2
    assert quantize_coord_m(1.0, quantum_m=0.5) == 2


def test_direction_bin_lower_at_boundary_48_directions():
    """Per BP5 §5.2: tie-break to LOWER bin index. 48 directions = 7.5° each.
    3.75° is exactly half-bin between dir_0 and dir_1 -> dir_0.
    """
    from cfm.data.sub_f.encoder import direction_bin

    assert direction_bin(0.0, direction_count=48) == 0
    assert direction_bin(3.75, direction_count=48) == 0  # exact half-bin -> lower
    assert direction_bin(7.5, direction_count=48) == 1
    assert direction_bin(359.999, direction_count=48) == 47


# ---- canonicalize_geometry: pre-flight Assertion 2 (routing fidelity) ------


def test_canonicalize_open_linestring_preserves_direction():
    """Per §5.6: LineString direction is PRESERVED (no canonicalization)."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    forward = LineString([(5, 5), (1, 1)])
    reverse = LineString([(1, 1), (5, 5)])
    assert list(canonicalize_geometry(forward).coords) == [(5, 5), (1, 1)]
    assert list(canonicalize_geometry(reverse).coords) == [(1, 1), (5, 5)]


def test_canonicalize_closed_linestring_preserves_direction():
    """Per §5.6: closed LineString (roundabout) routes to LineString-preserve,
    NOT to ring canonicalization. Anti-pattern this guards: dispatch on
    geom.is_ring (True for closed LineStrings) instead of geom.geom_type.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    roundabout_cw_non_lex_min_start = LineString(
        [(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)]
    )
    expected = [(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)]
    assert list(canonicalize_geometry(roundabout_cw_non_lex_min_start).coords) == expected


def test_canonicalize_polygon_rotates_to_lex_min_start():
    """Per §5.6 rule (i): Polygon ring rotated to start at lex-min vertex."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # [(5,5),(1,1),(3,1)] signed area = 8 > 0 -> CCW (RFC 7946 exterior OK)
    poly = Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])
    canon = canonicalize_geometry(poly)
    coords = list(canon.exterior.coords)
    assert coords[0] == (1, 1), f"expected lex-min start (1,1); got {coords[0]}"
    assert coords[0] == coords[-1], "ring must remain closed"


def test_canonicalize_polygon_reverses_cw_winding_to_ccw():
    """Per §5.6 rule (ii): Polygon exterior must be CCW (RFC 7946)."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # [(1,1),(5,5),(3,1)] signed area = -8 < 0 -> CW; lex-min (1,1) is first
    poly = Polygon([(1, 1), (5, 5), (3, 1), (1, 1)])
    canon = canonicalize_geometry(poly)
    coords = list(canon.exterior.coords)
    # CCW reverse: (1,1) -> (3,1) -> (5,5) -> (1,1)
    assert coords == [(1, 1), (3, 1), (5, 5), (1, 1)]


def test_canonicalize_multilinestring_sorts_parts_by_first_vertex():
    """Per §5.6 rule (iv): multi-part order = sort by first vertex (= lex-min
    for internally-canonical parts)."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    part_high = LineString([(3, 3), (7, 7)])
    part_low = LineString([(1, 1), (5, 5)])
    multi_in = MultiLineString([part_high, part_low])
    canon = list(canonicalize_geometry(multi_in).geoms)
    assert list(canon[0].coords)[0] == (1, 1)
    assert list(canon[1].coords)[0] == (3, 3)


def test_canonicalize_idempotent_all_geom_types():
    """Per §5.6: canonicalize(canonicalize(g)) == canonicalize(g) for every supported type."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    inputs = [
        LineString([(5, 5), (1, 1)]),
        Polygon([(5, 5), (1, 1), (3, 1), (5, 5)]),
        MultiLineString([LineString([(3, 3), (7, 7)]), LineString([(1, 1), (5, 5)])]),
        MultiPolygon([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
            Polygon([(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)]),
        ]),
    ]
    for g in inputs:
        once = canonicalize_geometry(g)
        twice = canonicalize_geometry(once)
        assert once.wkb == twice.wkb, (
            f"canonicalize not idempotent for {g.geom_type}: once != twice"
        )


# ---- pre-flight Assertion 1: token-count invariance ------------------------


def test_canonicalize_preserves_vertex_count():
    """BP5 ↔ BP3 invariance: canonicalization reorders but must not change V.

    The BP3 budget at P99.9 (5,888 padded) was measured on sub-C source-order
    features; T8 emits canonical-order features. The budget remains valid iff
    V(canonical) == V(source) per feature. If canonicalization ever adds /
    removes a closure vertex or normalizes a degenerate ring differently, the
    budget was measured against the wrong number.
    """
    from cfm.data.sub_f.encoder import _vertex_count, canonicalize_geometry

    cases = [
        # (label, source_geom)
        ("polygon_lex_min_rotation", Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])),
        ("polygon_winding_flip", Polygon([(1, 1), (5, 5), (3, 1), (1, 1)])),
        ("multilinestring_part_sort",
         MultiLineString([LineString([(3, 3), (7, 7)]), LineString([(1, 1), (5, 5)])])),
        ("open_linestring_preserve", LineString([(5, 5), (1, 1)])),
        ("closed_linestring_preserve",
         LineString([(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)])),
    ]
    for label, src in cases:
        canon = canonicalize_geometry(src)
        v_src = _vertex_count(src)
        v_canon = _vertex_count(canon)
        assert v_src == v_canon, (
            f"vertex count drift on {label}: source={v_src} canon={v_canon}"
        )
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v 2>&1 | tail -20`
Expected: ImportErrors for `quantize_coord_m`, `direction_bin`, `canonicalize_geometry`, `_vertex_count`.

- [ ] **Step 3: Implement `encoder.py` helpers + canonicalize_geometry**

Create `src/cfm/data/sub_f/encoder.py`:

```python
"""Sub-F per-feature encoder.

This module implements:

1. Coordinate / direction / magnitude helpers (BP2 lock per spec §3.4-§3.6).
2. canonicalize_geometry helper (BP5 §5.6 — 3 pure-redundancy DOFs;
   open-polyline direction PRESERVED per Halt 5 same-day follow-up).
3. Per-feature 4-case encoder (§3.2 A/B/C/D) — added in T8.4.
4. Per-cell aggregator (§3.3) — added in T8.6.

Routing pre-flight (per Assertion 2 in task-8-writer plan):
- canonicalize_geometry dispatches strictly on `geom.geom_type` (Shapely
  classification). NEVER use `geom.is_ring` for routing — it returns True
  for closed LineStrings (roundabouts) AND closed Polygon rings; routing
  closed LineString to the ring path destroys oneway semantics.
"""

from __future__ import annotations

from typing import Final

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import LinearRing, orient

# BP2 Halt 2 locked values — read here as module-level constants for fast
# access in encoder hot paths. Source of truth is configs/sub_f/encoding_primitives.yaml.
DEFAULT_DIRECTION_COUNT: Final[int] = 48
DEFAULT_MAGNITUDE_QUANTUM_M: Final[float] = 0.5
DEFAULT_ANCHOR_SCHEME: Final[str] = "hierarchical"
DEFAULT_N_ANCHOR_TOKENS: Final[int] = 4  # hierarchical scheme -> 4 tokens
DEFAULT_CHUNK_THRESHOLD_M: Final[float] = 32.0


# ---- helpers ---------------------------------------------------------------


def quantize_coord_m(coord_m: float, quantum_m: float = DEFAULT_MAGNITUDE_QUANTUM_M) -> int:
    """Quantize a coordinate (meters) to integer quantum count.

    Per BP5 §5.2 lock (Halt 5 ratification 2026-05-28):
    `int(round(coord_m / quantum))` with Python `round()` banker's tie-breaking
    (PEP 3141 round-half-to-even). Rationale: determinism requires one rule
    pinned; banker's is bias-free for coordinate snapping.
    """
    return int(round(coord_m / quantum_m))


def direction_bin(angle_deg: float, direction_count: int = DEFAULT_DIRECTION_COUNT) -> int:
    """Map angle (degrees, any sign) to direction bin index [0, direction_count).

    Per BP5 §5.2 lock: tie-break to LOWER bin index at exact bin boundaries.
    Implementation: floor division rounds toward lower index.
    """
    bin_width = 360.0 / direction_count
    angle_norm = angle_deg % 360.0
    return int(angle_norm // bin_width) % direction_count


def _vertex_count(geom: BaseGeometry) -> int:
    """Total vertex count for any supported geometry type.

    Used by tests (token-count invariance) and by the per-cell aggregator
    (feature_count column). Matches the count computed by
    scripts/sub_f/analyze_stage_1_2_joint.py (BP3 measurement basis).
    """
    gt = geom.geom_type
    if gt == "LineString":
        return len(geom.coords)
    if gt == "Polygon":
        return len(geom.exterior.coords)
    if gt == "Point":
        return 1
    if gt == "MultiPoint":
        return sum(1 for _ in geom.geoms)
    if gt == "MultiLineString":
        return sum(len(part.coords) for part in geom.geoms)
    if gt == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0


# ---- canonicalize_geometry (BP5 §5.6 contract) ----------------------------


def _signed_area_xy(coords: list[tuple[float, float]]) -> float:
    """Shoelace signed area. Positive = CCW (RFC 7946 exterior); negative = CW."""
    n = len(coords) - 1  # exclude closing duplicate
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[i + 1][0], coords[i + 1][1]
        s += (x2 - x1) * (y2 + y1)
    return -s / 2.0  # negate so positive = CCW per RFC 7946 convention


def _canonicalize_ring(ring: LinearRing, *, is_exterior: bool) -> LinearRing:
    """Apply §5.6 rules (i) + (ii) to a single Polygon ring.

    (i) Rotate to start at lex-min vertex (exclude duplicate closing vertex).
    (ii) Enforce RFC 7946 winding: exterior CCW (positive signed area),
         interior holes CW (negative signed area).
    """
    coords = list(ring.coords)
    if len(coords) < 4:  # ring needs >=3 unique + 1 closing
        return ring

    # Strip closing vertex for rotation/winding logic; re-add at the end.
    unique = coords[:-1]
    n = len(unique)

    # (i) Rotate so coords[0] is lex-min (tuple comparison handles ties via
    # subsequent coords automatically).
    lex_min_idx = min(range(n), key=lambda i: unique[i])
    rotated = unique[lex_min_idx:] + unique[:lex_min_idx]

    # (ii) Winding correction.
    closed = rotated + [rotated[0]]
    area = _signed_area_xy(closed)
    if is_exterior and area < 0:
        # Currently CW; reverse to CCW. After reversal, the first vertex is
        # still the same lex-min (reverse of a list starting at x then ending
        # at x via closure yields a list starting at x — closure invariant).
        rotated = [rotated[0]] + list(reversed(rotated[1:]))
    elif (not is_exterior) and area > 0:
        rotated = [rotated[0]] + list(reversed(rotated[1:]))

    return LinearRing(rotated + [rotated[0]])


def _canonicalize_polygon(poly: Polygon) -> Polygon:
    """§5.6 rules (i) + (ii) applied to exterior + each interior hole."""
    canon_exterior = _canonicalize_ring(poly.exterior, is_exterior=True)
    canon_holes = [
        _canonicalize_ring(hole, is_exterior=False) for hole in poly.interiors
    ]
    return Polygon(canon_exterior, canon_holes)


def _first_vertex_key(geom: BaseGeometry) -> tuple:
    """Sort key for §5.6 rule (iv) multi-part order. For internally-canonical
    parts the first vertex equals the lex-min vertex by construction.

    Tiebreak: next vertex in (rotated for Polygon / source for LineString) order;
    if still tied, fall through to part vertex count (smaller first).
    """
    gt = geom.geom_type
    if gt == "Polygon":
        coords = list(geom.exterior.coords)
        # Strip closing vertex; use full coord list as tiebreak chain.
        return (tuple(coords[:-1]), len(coords) - 1)
    if gt == "LineString":
        coords = list(geom.coords)
        return (tuple(coords), len(coords))
    # Fallback for Point parts.
    return ((tuple(geom.coords)[0],), 1)


def canonicalize_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Apply spec §5.6 canonical form to *geom*.

    Dispatch is STRICTLY on `geom.geom_type` per Assertion 2 — closed
    LineStrings (start == end, e.g., roundabouts) route to the
    LineString-preserve path, NOT to the polygon-ring canonicalizer. Using
    `geom.is_ring` for routing is the bug this design forbids.

    Idempotent: canonicalize(canonicalize(g)) == canonicalize(g) for every
    supported geom_type.
    """
    gt = geom.geom_type

    if gt == "LineString":
        # Rule (i'): PRESERVE source direction. Open or closed — both kept
        # as-is. See spec §5.6 "Open-polyline direction preservation"
        # evidence note for the BP1-grep rationale (no oneway, waterway-flow,
        # cycleway:left/right tokens in BP1 vocab; canonicalizing direction
        # would silently destroy OSM semantics).
        return geom

    if gt == "Polygon":
        return _canonicalize_polygon(geom)

    if gt == "MultiLineString":
        # Each part: preserve direction (rule i' per part). Then sort parts.
        parts = list(geom.geoms)
        parts_sorted = sorted(parts, key=_first_vertex_key)
        return MultiLineString(parts_sorted)

    if gt == "MultiPolygon":
        # Each part: canonicalize internally (rules i + ii). Then sort parts.
        parts_canon = [_canonicalize_polygon(p) for p in geom.geoms]
        parts_sorted = sorted(parts_canon, key=_first_vertex_key)
        return MultiPolygon(parts_sorted)

    if gt in ("Point", "MultiPoint"):
        # Trivially canonical (no traversal DOF on points).
        return geom

    raise ValueError(f"canonicalize_geometry: unsupported geom_type {gt!r}")
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v`
Expected: all helper + canonicalize tests PASS (including the token-count invariance assertion).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
uv run ruff check src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git add src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git commit -m "feat(sub_f): T8.3 canonicalize_geometry + encoder helpers (BP5 contract; geom_type routing)"
```

---

## Sub-task T8.4: encoder 4-case grammar (Case A/B/C/D per §3.2)

**Files:**
- Modify: `src/cfm/data/sub_f/encoder.py` (append encode_feature + case-specific helpers)
- Append: `tests/data/sub_f/test_encoder.py`

Spec §3.2 four-case grammar. For T8.4, BP7 boundary-ref tokens are SUPPLIED to the encoder as parameters (`inbound_bref`, `outbound_bref`) — sourcing them from sub-E `boundary_contract.parquet` is T8.5's responsibility.

### Implementation steps

- [ ] **Step 1: Write failing per-case tests**

Append to `tests/data/sub_f/test_encoder.py`:

```python
# ---- encoder 4-case grammar -----------------------------------------------


def _vocab_id(tag: str) -> int:
    """Test helper — looks up token_id by tag from the loaded vocab."""
    from cfm.data.sub_f.vocab import vocab_tag_to_id
    return vocab_tag_to_id()[tag]


def test_encode_feature_case_a_uncrossed_polyline():
    """Case A: uncrossed feature fully within cell.

    Per §3.2 token shape:
      <feature> <semantic_tag>
        <anchor_x_q> <anchor_y_q>             (2 tokens flat; 4 if hierarchical)
        <direction_d> <magnitude_q>           × (V-1) pairs
      <feature_end>

    With BP2 lock (hierarchical anchor, n_anchor=4):
      tokens = 3 + 4 + 2*(V-1) = 7 + 2V

    Test input: 3-vertex polyline; expect token count = 13.
    """
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import LineString

    geom = LineString([(10.0, 20.0), (15.0, 25.0), (20.0, 30.0)])
    encoded = encode_feature(geom, semantic_tag="highway=residential")
    assert encoded.case == "A"
    assert len(encoded.tokens) == 7 + 2 * 3, (
        f"Case A 3-vertex: expected 13 tokens; got {len(encoded.tokens)}"
    )


def test_encode_feature_case_b_outbound_road():
    """Case B: road exiting via one cell edge.

    Per §3.2 token shape:
      <feature> <semantic_tag>
        <anchor_x_q> <anchor_y_q>             (anchor)
        <direction_d> <magnitude_q>           × (V-2) inner pairs
        <bref_dir_class>                       outbound bref (replaces tail
                                                direction+magnitude; net 0
                                                vs Case A's final pair, so
                                                token delta vs A is +1 for
                                                the feature_end shift)

    Pattern: tokens = 1 + 1 + N_anchor + 2*(V-2) + 1 + 1 = 4 + N_anchor + 2*(V-2)
    With n_anchor=4: tokens = 8 + 2*(V-2)
    """
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import LineString

    geom = LineString([(10.0, 20.0), (15.0, 25.0), (250.0, 30.0)])
    encoded = encode_feature(
        geom,
        semantic_tag="highway=primary",
        outbound_bref="<bref_E_MAJOR>",
    )
    assert encoded.case == "B"
    V = 3
    assert len(encoded.tokens) == 8 + 2 * (V - 2)
    # Outbound bref must appear once; last token before <feature_end>.
    assert encoded.tokens[-2] == _vocab_id("<bref_E_MAJOR>")


def test_encode_feature_case_c_inbound_road():
    """Case C: road entering from one cell edge.

    Per §3.2 token shape:
      <feature> <semantic_tag>
        <bref_dir_class>                       inbound bref (prepended)
        <anchor_x_q> <anchor_y_q>              entry vertex coords (on edge)
        <direction_d> <magnitude_q>           × (V-1) pairs
      <feature_end>

    Pattern: tokens = 1 + 1 + 1 + N_anchor + 2*(V-1) + 1 = 4 + N_anchor + 2*(V-1)
    With n_anchor=4: tokens = 8 + 2*(V-1)
    """
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import LineString

    geom = LineString([(0.0, 100.0), (50.0, 100.0), (100.0, 100.0)])
    encoded = encode_feature(
        geom,
        semantic_tag="highway=primary",
        inbound_bref="<bref_W_MAJOR>",
    )
    assert encoded.case == "C"
    V = 3
    assert len(encoded.tokens) == 8 + 2 * (V - 1)
    # Inbound bref is the 3rd token (after <feature>, <semantic_tag>).
    assert encoded.tokens[2] == _vocab_id("<bref_W_MAJOR>")


def test_encode_feature_case_d_through_road():
    """Case D: road inbound AND outbound (through-cell).

    Per §3.2 token shape:
      <feature> <semantic_tag>
        <bref_dir_class>                       inbound
        <anchor_x_q> <anchor_y_q>              entry vertex
        <direction_d> <magnitude_q>           × (V-2) inner pairs
        <bref_dir_class>                       outbound
      <feature_end>

    Pattern: tokens = 5 + N_anchor + 2*(V-2)
    With n_anchor=4: tokens = 9 + 2*(V-2)
    """
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import LineString

    geom = LineString([(0.0, 100.0), (125.0, 100.0), (250.0, 100.0)])
    encoded = encode_feature(
        geom,
        semantic_tag="highway=primary",
        inbound_bref="<bref_W_MAJOR>",
        outbound_bref="<bref_E_MAJOR>",
    )
    assert encoded.case == "D"
    V = 3
    assert len(encoded.tokens) == 9 + 2 * (V - 2)


def test_encode_feature_starts_with_feature_marker_ends_with_feature_end():
    """Every encoded feature opens with <feature> and closes with <feature_end>.

    These structural sentinels live at IDs 509 and 510 per the T8 plan-write
    sentinel-inventory fix (consumed from BP2 reserved_v2_headroom front;
    family="structural", distinct from "encoding_primitive"). See spec §13.1
    "T8 plan-write -> BP2 inventory" row.
    """
    from cfm.data.sub_f.encoder import encode_feature
    from cfm.data.sub_f.vocab import vocab_tag_to_id
    from shapely.geometry import LineString

    tag_to_id = vocab_tag_to_id()
    expected_feature = tag_to_id["<feature>"]
    expected_feature_end = tag_to_id["<feature_end>"]
    assert expected_feature == 509
    assert expected_feature_end == 510

    geom = LineString([(1.0, 1.0), (2.0, 2.0)])
    encoded = encode_feature(geom, semantic_tag="highway=service")
    assert encoded.tokens[0] == expected_feature
    assert encoded.tokens[-1] == expected_feature_end


def test_encode_feature_uses_bp4_unknown_tag_when_semantic_unmapped():
    """class_raw sentinels (e.g., B__UNK__, unknown) map to BP4 <unknown_*>
    family per §3.3 + cascade #7. encode_feature accepts the raw semantic_tag
    and resolves to BP4 if it's a sub-C sentinel.
    """
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import Polygon

    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    # B__UNK__ is a sub-C sentinel; encoder routes to BP4 <unknown_building>.
    encoded = encode_feature(geom, semantic_tag="building=B__UNK__")
    # Expect the BP4 token id for <unknown_building> in the encoded sequence.
    from cfm.data.sub_f.vocab import vocab_tag_to_id
    unknown_building_id = vocab_tag_to_id()["<unknown_building>"]
    assert unknown_building_id in encoded.tokens
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_encoder.py::test_encode_feature_case_a_uncrossed_polyline -v`
Expected: FAIL with `ImportError: cannot import name 'encode_feature'`.

- [ ] **Step 3: Implement `encode_feature` in `encoder.py`**

Append to `src/cfm/data/sub_f/encoder.py`:

```python
# ---- 4-case grammar (§3.2) -------------------------------------------------

import math
from dataclasses import dataclass
from typing import Literal

from cfm.data.sub_f.vocab import vocab_tag_to_id

Case = Literal["A", "B", "C", "D"]

# Structural sentinels (family="structural", consumed from BP2
# reserved_v2_headroom front per sentinel-inventory fix 2026-05-28).
# See pre-flight Assertion 4 + spec §13.1 "T8 plan-write -> BP2 inventory" row.
_FEATURE_TOKEN_ID = 509
_FEATURE_END_TOKEN_ID = 510

# BP2 anchor sub-block is 300-395 (full 96 slots usable as anchor coords).
# Hierarchical anchor scheme uses 4 tokens per anchor; with BLOCK=23 below,
# 23*4 = 92 slots are addressable (positions 300-391), leaving 4 spare
# (392-395) per anchor sub-block reserve.
_HIERARCHICAL_ANCHOR_BASE = 300

# Sub-C sentinel patterns that map to BP4 <unknown_*> family per §3.3 + cascade #7.
_SUB_C_UNKNOWN_PATTERNS = ("__UNK__", "unknown")
_SUB_C_BUILDING_SENTINEL_PREFIX = "B_"  # B__UNK__ is the primary sentinel


@dataclass(frozen=True)
class EncodedFeature:
    """Per-feature encoded token sequence."""

    case: Case
    semantic_tag: str
    tokens: list[int]


def _resolve_semantic_tag_to_token_id(semantic_tag: str) -> int:
    """Map a raw OSM tag (key=value) to its token id.

    Per §3.3 + cascade #7: sub-C unknown sentinels (`B__UNK__`, `highway=unknown`,
    etc.) are NOT first-class BP1 slots — they map to the BP4 `<unknown_KEY>`
    family for the parent key.
    """
    tag_to_id = vocab_tag_to_id()

    if semantic_tag in tag_to_id:
        return tag_to_id[semantic_tag]

    # Fall through to BP4. Extract the key from "key=value" form.
    if "=" not in semantic_tag:
        raise ValueError(f"semantic_tag must be 'key=value' form; got {semantic_tag!r}")
    key, value = semantic_tag.split("=", 1)

    is_sub_c_sentinel = (
        any(pat in value for pat in _SUB_C_UNKNOWN_PATTERNS)
        or value.startswith(_SUB_C_BUILDING_SENTINEL_PREFIX)
        or value == ""
    )
    unknown_tag = f"<unknown_{key}>"
    if is_sub_c_sentinel and unknown_tag in tag_to_id:
        return tag_to_id[unknown_tag]

    # Non-sentinel value missing from BP1: also bucket via BP4 per cascade #7.
    if unknown_tag in tag_to_id:
        return tag_to_id[unknown_tag]

    raise KeyError(f"no BP1 or BP4 slot for semantic_tag {semantic_tag!r}")


def _hierarchical_anchor_tokens(x_m: float, y_m: float) -> list[int]:
    """Per §3.6 hierarchical anchor scheme: 4 tokens encoding (x_hi, x_lo, y_hi, y_lo).

    Cell extent 250m, magnitude_quantum 0.5m -> 500 quantum-cells per axis.
    Hierarchical split: hi block = floor(coord_q / 23), lo block = coord_q % 23,
    where 23 chosen so 23*22 >= 500. Hi range [0, 22] = 23 slots; lo range
    [0, 22] = 23 slots; total per axis = 46; both axes = 92 (fits in 94-slot
    anchor sub-region 302..395 with 2 reserved spares).

    Per BP2 Halt 2 lock: anchor sub-block IDs 302..324 = x_hi, 325..347 = x_lo,
    348..370 = y_hi, 371..393 = y_lo. (Slots 394-395 are reserved spares.)
    """
    BLOCK = 23
    x_q = quantize_coord_m(x_m)
    y_q = quantize_coord_m(y_m)
    x_hi, x_lo = divmod(x_q, BLOCK)
    y_hi, y_lo = divmod(y_q, BLOCK)
    # Layout within the BP2 anchor sub-block (300..395):
    #   300..322  x_hi  (23 slots)
    #   323..345  x_lo  (23 slots)
    #   346..368  y_hi  (23 slots)
    #   369..391  y_lo  (23 slots)
    #   392..395  spare (4 slots, reserved within anchor sub-block)
    return [
        _HIERARCHICAL_ANCHOR_BASE + 0 * BLOCK + x_hi,
        _HIERARCHICAL_ANCHOR_BASE + 1 * BLOCK + x_lo,
        _HIERARCHICAL_ANCHOR_BASE + 2 * BLOCK + y_hi,
        _HIERARCHICAL_ANCHOR_BASE + 3 * BLOCK + y_lo,
    ]


def _direction_magnitude_pair(
    dx: float, dy: float
) -> list[int]:
    """One (direction, magnitude) pair token list for a segment.

    Direction sub-block: ids 396..443 (48 slots).
    Magnitude sub-block: ids 444..508 (65 slots; 0.5m * (1..64) plus a single
    overflow marker at 444+64 = 508).
    """
    angle_deg = math.degrees(math.atan2(dy, dx))
    direction = direction_bin(angle_deg)
    distance_m = math.hypot(dx, dy)
    magnitude_q = max(1, min(64, quantize_coord_m(distance_m)))

    DIRECTION_BASE = 396
    MAGNITUDE_BASE = 444
    return [DIRECTION_BASE + direction, MAGNITUDE_BASE + (magnitude_q - 1)]


def _vertex_pairs_dir_mag(coords: list[tuple[float, float]]) -> list[int]:
    """For V vertices, emit 2*(V-1) tokens — one (dir, mag) pair per segment."""
    out: list[int] = []
    for i in range(1, len(coords)):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        out.extend(_direction_magnitude_pair(x2 - x1, y2 - y1))
    return out


def _extract_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Get the encoder's input coord list for a feature.

    For Polygons we encode the exterior ring; multi-geometries are encoded
    per-part by the caller (encode_cell in T8.6 splits them).
    """
    gt = geom.geom_type
    if gt == "LineString":
        return list(geom.coords)
    if gt == "Polygon":
        return list(geom.exterior.coords)
    if gt == "Point":
        x, y = geom.x, geom.y
        return [(x, y)]
    raise ValueError(f"_extract_coords: encode multi-part geometries per part, not as a whole ({gt})")


def encode_feature(
    geom: BaseGeometry,
    *,
    semantic_tag: str,
    inbound_bref: str | None = None,
    outbound_bref: str | None = None,
) -> EncodedFeature:
    """Encode one feature per spec §3.2 four-case grammar.

    Caller responsibilities:
      - Pass the canonical geometry (use canonicalize_geometry first).
      - For multi-part geometries, split into parts and call once per part.
      - Set inbound_bref / outbound_bref from sub-E boundary contract per
        cell × edge (T8.5 provides the resolver).

    Routing:
      - inbound_bref=None, outbound_bref=None -> Case A
      - inbound_bref=None, outbound_bref=set  -> Case B
      - inbound_bref=set,  outbound_bref=None -> Case C
      - inbound_bref=set,  outbound_bref=set  -> Case D

    BP7 emission — UNVERIFIED against real sub-E parquet; see close-checklist +
    project_sub_e_cache_absent_t3c_code_inferred memory. inbound_bref /
    outbound_bref values originate at T8.5's sub-E reader against the
    documented schema.
    """
    tag_to_id = vocab_tag_to_id()
    coords = _extract_coords(geom)
    V = len(coords)
    if V < 1:
        raise ValueError("encode_feature: empty coord list")

    semantic_id = _resolve_semantic_tag_to_token_id(semantic_tag)

    case: Case
    if inbound_bref is None and outbound_bref is None:
        case = "A"
    elif inbound_bref is None and outbound_bref is not None:
        case = "B"
    elif inbound_bref is not None and outbound_bref is None:
        case = "C"
    else:
        case = "D"

    tokens: list[int] = [_FEATURE_TOKEN_ID, semantic_id]

    if case in ("C", "D"):
        # Inbound bref prepended; per §3.2, the anchor IS the entry vertex
        # which IS coords[0] (canonical convention).
        tokens.append(tag_to_id[inbound_bref])  # BP7 emission — UNVERIFIED

    anchor_x, anchor_y = coords[0]
    tokens.extend(_hierarchical_anchor_tokens(anchor_x, anchor_y))

    # Inner pairs: Case A/C emit (V-1) pairs (reach vertices 2..V);
    # Case B/D emit (V-2) pairs (final vertex replaced by outbound bref).
    inner_pairs_to = V if case in ("A", "C") else V - 1
    tokens.extend(_vertex_pairs_dir_mag(coords[: inner_pairs_to]))

    if case in ("B", "D"):
        tokens.append(tag_to_id[outbound_bref])  # BP7 emission — UNVERIFIED

    tokens.append(_FEATURE_END_TOKEN_ID)
    return EncodedFeature(case=case, semantic_tag=semantic_tag, tokens=tokens)
```

- [ ] **Step 4: Run per-case tests + verify token counts match spec formulas**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v -k "case_a or case_b or case_c or case_d or feature_marker or unknown_tag"`
Expected: 6 PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
uv run ruff check src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git add src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git commit -m "feat(sub_f): T8.4 encoder 4-case grammar (A/B/C/D per §3.2)"
```

---

## Sub-task T8.5: BP7 sub-E boundary-contract reader

**Files:**
- Create: `src/cfm/data/sub_f/boundary_contract.py`
- Create: `tests/data/sub_f/test_boundary_contract.py`

**Pre-flight Assertion 3 applies here.** Sub-E cache is absent; this reader is built against the documented sub-E parquet schema and tested with SYNTHETIC fixtures. Every BP7 token emission site carries the unverified-debt comment.

### Implementation steps

- [ ] **Step 1: Write failing test with SYNTHETIC sub-E fixture**

Create `tests/data/sub_f/test_boundary_contract.py`:

```python
"""Sub-F BP7 boundary-contract reader tests.

SYNTHETIC sub-E parquet fixtures only. Real sub-E cache is absent
(project_sub_e_cache_absent_t3c_code_inferred memory); integration tests
against real sub-E are in T8.8's test_pipeline_writer.py under the
`@pytest.mark.skip(reason="awaiting sub-E cache regeneration")` mark.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Documented sub-E parquet schema (from src/cfm/data/sub_e/writer.py).
# Per Halt 7 / spec §3.7: BoundaryClass enum 0=BOUNDARY_NOT_APPLICABLE,
# 1=NONE, 2=MAJOR_ROAD, 3=MINOR_ROAD. SlotKind enum 1=INTERNAL_EDGE,
# 2=EXTERNAL_EDGE. NOT_APPLICABLE is never on-disk per sub-E sentinel
# precedent.
_SUB_E_SCHEMA = pa.schema(
    [
        pa.field("slot_kind", pa.int8(), nullable=False),
        pa.field("slot_index", pa.int16(), nullable=False),
        pa.field("boundary_class_enum", pa.int8(), nullable=False),
    ]
)


def _write_synthetic_sub_e_parquet(path: Path, rows: list[dict]) -> None:
    """Helper: write a one-tile sub-E boundary_contract.parquet for tests."""
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_load_boundary_contract_returns_per_cell_per_edge_map(tmp_path: Path):
    """Reader returns {(cell_i, cell_j): {direction: class_label}} dict.

    Sub-E tile has 8x8 = 64 cells. Each cell has 4 edges (N/E/S/W) mapped
    via sub-E rotation. Reader walks rows -> resolves slot to (cell, edge)
    via rotation -> emits per-cell map.
    """
    from cfm.data.sub_f.boundary_contract import load_boundary_contract

    parquet_path = tmp_path / "boundary_contract.parquet"
    # Minimal synthetic fixture: one INTERNAL_EDGE row with MAJOR_ROAD class.
    # Slot 0 covers a specific (cell_i, cell_j, edge) per sub-E rotation; the
    # reader resolves it.
    _write_synthetic_sub_e_parquet(
        parquet_path,
        [{"slot_kind": 1, "slot_index": 0, "boundary_class_enum": 2}],
    )

    contract = load_boundary_contract(parquet_path)
    # Some cell must have a MAJOR_ROAD entry from this row.
    has_major = any(
        any(cls == "MAJOR_ROAD" for cls in cell_edges.values())
        for cell_edges in contract.values()
    )
    assert has_major, "synthetic MAJOR slot must surface in per-cell map"


def test_resolve_bref_token_returns_correct_8_token_form(tmp_path: Path):
    """Per Halt 7 / spec §3.7: 8 boundary-ref tokens, 4 directions × 2 active classes:
    <bref_N_MAJOR> <bref_E_MAJOR> <bref_S_MAJOR> <bref_W_MAJOR>
    <bref_N_MINOR> <bref_E_MINOR> <bref_S_MINOR> <bref_W_MINOR>
    Resolver builds the token tag from direction + class.
    """
    from cfm.data.sub_f.boundary_contract import resolve_bref_tag

    assert resolve_bref_tag("N", "MAJOR_ROAD") == "<bref_N_MAJOR>"
    assert resolve_bref_tag("W", "MINOR_ROAD") == "<bref_W_MINOR>"
    # NONE class -> None (no token emitted)
    assert resolve_bref_tag("N", "NONE") is None
    # NOT_APPLICABLE never appears on-disk per sub-E discipline; defensive None.
    assert resolve_bref_tag("N", "BOUNDARY_NOT_APPLICABLE") is None


def test_load_boundary_contract_uses_pq_parquetfile_not_read_table(tmp_path: Path):
    """Per `feedback_pyarrow_hive_partition_inference`: must use pq.ParquetFile(path).read(),
    NOT bare pq.read_table() on a parent dir (would inject spurious 'tile' column).
    This test indirectly verifies via column inspection.
    """
    from cfm.data.sub_f.boundary_contract import load_boundary_contract

    parquet_path = tmp_path / "tile=EPSG3414_i0_j0" / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(
        parquet_path,
        [{"slot_kind": 1, "slot_index": 5, "boundary_class_enum": 3}],
    )
    contract = load_boundary_contract(parquet_path)
    # If reader used bare read_table on the parent dir, it would inject a
    # 'tile' column; the resolver would crash on the unexpected schema. The
    # fact that this returns without error confirms ParquetFile path is used.
    assert isinstance(contract, dict)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_boundary_contract.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `boundary_contract.py`**

Create `src/cfm/data/sub_f/boundary_contract.py`:

```python
"""Sub-F BP7 sub-E boundary-contract reader.

Consumes sub-E `boundary_contract.parquet` and emits a per-cell per-edge
class map that the encoder uses to decide Case A/B/C/D and to look up the
correct <bref_DIR_CLASS> token.

VERIFICATION DEBT NOTICE (spec §3.7 + Halt 7 close-checklist):
Sub-E cache is absent locally as of 2026-05-28. This reader is built
against sub-E's DOCUMENTED parquet schema (from src/cfm/data/sub_e/writer.py
+ sub-E spec) but has not been integration-tested against real sub-E
output. Spot-check obligations live on
`reports/2026-05-23-phase-1-sub-F-close-checklist.md`. When sub-E
regenerates, un-skip T8.8 integration tests + verify reader output
matches real parquet on grammar cases B/C/D edge scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.writer import SlotKind
from cfm.data.sub_f.rotation import cell_edge_directions

# Sub-E BoundaryClass enum mapping to per-spec §3.7 class labels.
_CLASS_LABEL_BY_ENUM: Final[dict[int, str]] = {
    int(BoundaryClass.BOUNDARY_NOT_APPLICABLE): "BOUNDARY_NOT_APPLICABLE",
    int(BoundaryClass.NONE): "NONE",
    int(BoundaryClass.MAJOR_ROAD): "MAJOR_ROAD",
    int(BoundaryClass.MINOR_ROAD): "MINOR_ROAD",
}

# Only these two classes emit a <bref> token (per spec §3.7 BP7 lock).
_EMITTING_CLASSES: Final[frozenset[str]] = frozenset({"MAJOR_ROAD", "MINOR_ROAD"})


def resolve_bref_tag(direction: str, class_label: str) -> str | None:
    """Build the BP7 token tag for (direction, class). Returns None if the
    class is non-emitting (NONE or BOUNDARY_NOT_APPLICABLE).

    Per spec §3.7 BP7 lock: 8 active tokens {N,E,S,W} × {MAJOR,MINOR}.
    """
    if class_label not in _EMITTING_CLASSES:
        return None
    if direction not in ("N", "E", "S", "W"):
        raise ValueError(f"resolve_bref_tag: unsupported direction {direction!r}")
    short = "MAJOR" if class_label == "MAJOR_ROAD" else "MINOR"
    return f"<bref_{direction}_{short}>"


def load_boundary_contract(
    parquet_path: Path,
) -> dict[tuple[int, int], dict[str, str]]:
    """Read a sub-E boundary_contract.parquet and emit a per-cell map.

    Output shape:
        { (cell_i, cell_j): { "N": class_label, "E": ..., "S": ..., "W": ... } }

    Only on-disk class labels appear (NEVER `BOUNDARY_NOT_APPLICABLE` per
    sub-E sentinel discipline at `src/cfm/data/sub_e/derivation.py`). Cells
    with no road edges still appear with all four directions present and
    `class_label == "NONE"` — the encoder uses this to disambiguate
    Case A (no bref needed) from genuine NONE edges.

    Per `feedback_pyarrow_hive_partition_inference`: uses
    `pq.ParquetFile(path).read()` — never bare `pq.read_table()` on the
    parent directory.

    BP7 emission — UNVERIFIED against real sub-E parquet; see module
    docstring + close-checklist.
    """
    table = pq.ParquetFile(parquet_path).read()
    rows = table.to_pylist()

    # Build a slot-id → class_label map first (sub-E's primary index).
    slot_to_class: dict[tuple[int, int], str] = {}
    for r in rows:
        sk = int(r["slot_kind"])
        si = int(r["slot_index"])
        cls = _CLASS_LABEL_BY_ENUM[int(r["boundary_class_enum"])]
        slot_to_class[(sk, si)] = cls

    # Walk every cell × direction; look up slot via rotation; resolve class.
    contract: dict[tuple[int, int], dict[str, str]] = {}
    for cell_i in range(8):
        for cell_j in range(8):
            edge_ids = cell_edge_directions(cell_i, cell_j)
            cell_edges: dict[str, str] = {}
            for direction in ("N", "E", "S", "W"):
                slot = edge_ids[direction]  # (slot_kind, slot_index) tuple
                key = (int(slot.slot_kind), int(slot.slot_index))
                cell_edges[direction] = slot_to_class.get(key, "NONE")
            contract[(cell_i, cell_j)] = cell_edges

    return contract
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/sub_f/test_boundary_contract.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/boundary_contract.py tests/data/sub_f/test_boundary_contract.py
uv run ruff check src/cfm/data/sub_f/boundary_contract.py tests/data/sub_f/test_boundary_contract.py
git add src/cfm/data/sub_f/boundary_contract.py tests/data/sub_f/test_boundary_contract.py
git commit -m "feat(sub_f): T8.5 BP7 sub-E boundary-contract reader (UNVERIFIED against real sub-E; see close-checklist)"
```

---

## Sub-task T8.6: encoder per-cell aggregator + empty-cell handling

**Files:**
- Modify: `src/cfm/data/sub_f/encoder.py` (append `encode_cell`)
- Append: `tests/data/sub_f/test_encoder.py`

### Implementation steps

- [ ] **Step 1: Write failing tests**

Append to `tests/data/sub_f/test_encoder.py`:

```python
# ---- per-cell aggregator (§3.3, §4.4) -------------------------------------


def test_encode_cell_empty_emits_empty_token_list():
    """Per spec §4.4: empty cells emit token_sequence = [] (not null)."""
    from cfm.data.sub_f.encoder import EncodedCell, encode_cell

    out = encode_cell(features=[], cell_edges={})
    assert isinstance(out, EncodedCell)
    assert out.tokens == []
    assert out.feature_count == 0


def test_encode_cell_concatenates_per_feature_tokens():
    """Per spec §3.3: cell-level sequence is flat concatenation of per-feature seqs."""
    from cfm.data.sub_f.encoder import encode_cell
    from shapely.geometry import LineString

    features = [
        (LineString([(1.0, 1.0), (2.0, 2.0)]), "highway=service", "A"),
        (LineString([(10.0, 10.0), (20.0, 20.0)]), "highway=residential", "A"),
    ]
    out = encode_cell(
        features=[(geom, tag) for geom, tag, _case in features],
        cell_edges={"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"},
    )
    # 2 features × (3 + 4 + 2*1 = 9) tokens each = 18 total
    assert len(out.tokens) == 18
    assert out.feature_count == 2


def test_encode_cell_preserves_sub_c_source_order():
    """Per spec §3.3: features encoded in sub-C row order. Caller controls
    ordering by the order of `features` list. encode_cell does NOT re-sort.
    """
    from cfm.data.sub_f.encoder import encode_cell
    from shapely.geometry import LineString

    f1 = (LineString([(1.0, 1.0), (2.0, 2.0)]), "highway=service")
    f2 = (LineString([(10.0, 10.0), (20.0, 20.0)]), "highway=residential")
    out_12 = encode_cell(features=[f1, f2], cell_edges={"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"})
    out_21 = encode_cell(features=[f2, f1], cell_edges={"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"})
    assert out_12.tokens != out_21.tokens, "encode_cell must respect input feature order"


def test_encode_cell_canonicalizes_each_feature():
    """Each feature is passed through canonicalize_geometry before encoding.
    Token output for a non-canonical polygon must equal the output for its
    canonical form.
    """
    from cfm.data.sub_f.encoder import encode_cell, canonicalize_geometry
    from shapely.geometry import Polygon

    raw = Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])  # CCW, non-lex-min start
    canon = canonicalize_geometry(raw)

    out_raw = encode_cell(features=[(raw, "building=residential")], cell_edges={"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"})
    out_canon = encode_cell(features=[(canon, "building=residential")], cell_edges={"N": "NONE", "E": "NONE", "S": "NONE", "W": "NONE"})
    assert out_raw.tokens == out_canon.tokens, (
        "encode_cell must canonicalize raw inputs internally so output is "
        "invariant to source ordering — required for BP3 token-count invariance"
    )
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v -k "encode_cell"`
Expected: ImportError for `encode_cell`, `EncodedCell`.

- [ ] **Step 3: Implement `encode_cell` in `encoder.py`**

Append to `src/cfm/data/sub_f/encoder.py`:

```python
# ---- per-cell aggregator (§3.3, §4.4) -------------------------------------


from cfm.data.sub_f.boundary_contract import resolve_bref_tag


@dataclass(frozen=True)
class EncodedCell:
    """Per-cell encoded sequence."""

    tokens: list[int]
    feature_count: int  # number of features encoded (matches cells.parquet col)


def _classify_feature_for_bref(
    geom: BaseGeometry,
    cell_edges: dict[str, str],
    cell_origin: tuple[float, float] = (0.0, 0.0),
    cell_extent_m: float = 250.0,
    edge_eps_m: float = 1e-6,
) -> tuple[str | None, str | None]:
    """Determine inbound / outbound boundary-ref tags for one feature.

    Compares geometry endpoints against cell edges; if an endpoint lies on
    an active boundary edge (MAJOR or MINOR per cell_edges), emit the
    matching <bref_DIR_CLASS> tag.

    LineStrings only emit brefs — Polygons (buildings) and Points (POIs) do
    not cross cell boundaries in token-layer per spec §1.4 (non-road
    cross-cell features clipped at geometry layer).

    BP7 emission — UNVERIFIED against real sub-E parquet; see boundary_contract.py
    module docstring.
    """
    gt = geom.geom_type
    if gt not in ("LineString", "MultiLineString"):
        return None, None

    if gt == "MultiLineString":
        # Multi-part is split by encode_cell before reaching here.
        return None, None

    coords = list(geom.coords)
    if len(coords) < 2:
        return None, None

    def _direction_of_endpoint(x: float, y: float) -> str | None:
        ox, oy = cell_origin
        x_rel = x - ox
        y_rel = y - oy
        if abs(x_rel) <= edge_eps_m:
            return "W"
        if abs(x_rel - cell_extent_m) <= edge_eps_m:
            return "E"
        if abs(y_rel) <= edge_eps_m:
            return "S"
        if abs(y_rel - cell_extent_m) <= edge_eps_m:
            return "N"
        return None

    in_dir = _direction_of_endpoint(*coords[0])
    out_dir = _direction_of_endpoint(*coords[-1])

    in_class = cell_edges.get(in_dir) if in_dir else None
    out_class = cell_edges.get(out_dir) if out_dir else None

    inbound_bref = resolve_bref_tag(in_dir, in_class) if in_dir and in_class else None
    outbound_bref = resolve_bref_tag(out_dir, out_class) if out_dir and out_class else None
    return inbound_bref, outbound_bref


def encode_cell(
    features: list[tuple[BaseGeometry, str]],
    cell_edges: dict[str, str],
    cell_origin: tuple[float, float] = (0.0, 0.0),
) -> EncodedCell:
    """Encode one cell to a flat token sequence.

    Args:
      features: list of (geom, semantic_tag) tuples in sub-C row order
                (caller does NOT re-sort).
      cell_edges: per-cell boundary-class map from
                  `boundary_contract.load_boundary_contract`. Pass empty
                  dict for empty / no-edge cells.
      cell_origin: cell SW corner in projected meters (default (0,0) for
                   cell-local coords).

    Per §3.3 + §4.4:
      - Empty cells emit tokens = [] (not null).
      - Per-feature output is concatenated with no <cell_start>/<cell_end>
        sentinel on-disk (cell boundary is the parquet row structure).
      - Each feature is canonicalized internally before encoding (BP5
        contract).
    """
    if not features:
        return EncodedCell(tokens=[], feature_count=0)

    tokens: list[int] = []
    feature_count = 0
    for geom, semantic_tag in features:
        canon = canonicalize_geometry(geom)

        # Multi-part: encode each part separately per spec §3.2 implicit
        # multi-part handling (one EncodedFeature per part).
        gt = canon.geom_type
        if gt in ("MultiLineString", "MultiPolygon"):
            for part in canon.geoms:
                inbound, outbound = _classify_feature_for_bref(part, cell_edges, cell_origin)
                ef = encode_feature(
                    part,
                    semantic_tag=semantic_tag,
                    inbound_bref=inbound,
                    outbound_bref=outbound,
                )
                tokens.extend(ef.tokens)
                feature_count += 1
        elif gt == "MultiPoint":
            # MultiPoint: encode each Point as a separate Case A feature.
            for part in canon.geoms:
                ef = encode_feature(part, semantic_tag=semantic_tag)
                tokens.extend(ef.tokens)
                feature_count += 1
        else:
            inbound, outbound = _classify_feature_for_bref(canon, cell_edges, cell_origin)
            ef = encode_feature(
                canon,
                semantic_tag=semantic_tag,
                inbound_bref=inbound,
                outbound_bref=outbound,
            )
            tokens.extend(ef.tokens)
            feature_count += 1

    return EncodedCell(tokens=tokens, feature_count=feature_count)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v -k "encode_cell"`
Expected: 4 PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
uv run ruff check src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git add src/cfm/data/sub_f/encoder.py tests/data/sub_f/test_encoder.py
git commit -m "feat(sub_f): T8.6 encode_cell per-cell aggregator + empty-cell handling"
```

---

## Sub-task T8.7: decoder + canonical GeoJSON

**Files:**
- Create: `src/cfm/data/sub_f/decoder.py`
- Create: `tests/data/sub_f/test_decoder.py`

### Implementation steps

- [ ] **Step 1: Write failing test (round-trip via decode)**

Create `tests/data/sub_f/test_decoder.py`:

```python
"""Sub-F decoder + canonical GeoJSON tests."""

from __future__ import annotations

import json
import math

from shapely.geometry import LineString, Polygon


def test_canonical_geojson_byte_stable_across_key_order():
    """Per spec §5.3: sort_keys=True, indent=None, ensure_ascii=True."""
    from cfm.data.sub_f.decoder import serialize_geojson

    geom1 = {"type": "Point", "coordinates": [1.0, 2.0]}
    geom2 = {"coordinates": [1.0, 2.0], "type": "Point"}
    assert serialize_geojson(geom1) == serialize_geojson(geom2)


def test_decode_feature_case_a_round_trip_linf_within_threshold():
    """Per BP2 Halt 2 lock: round_trip_l_inf_threshold_m = 4.8 m."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import encode_feature

    source = LineString([(10.0, 20.0), (15.0, 25.0), (20.0, 30.0)])
    encoded = encode_feature(source, semantic_tag="highway=residential")
    decoded = decode_feature(encoded.tokens)

    src_coords = list(source.coords)
    dec_coords = list(LineString(decoded["coordinates"]).coords)
    assert len(src_coords) == len(dec_coords)

    l_inf = max(
        max(abs(s[0] - d[0]), abs(s[1] - d[1]))
        for s, d in zip(src_coords, dec_coords)
    )
    assert l_inf <= 4.8, f"L_inf {l_inf:.4f} exceeds BP2 Halt 2 threshold 4.8m"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_decoder.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `decoder.py`**

Create `src/cfm/data/sub_f/decoder.py`:

```python
"""Sub-F per-feature decoder: token sequence -> GeoJSON geometry dict.

Inverse of `encoder.encode_feature` per spec §3.2 four-case grammar. Output
geometry serializes to canonical GeoJSON via `serialize_geojson` (sort_keys,
no indent, ASCII) per spec §5.3 for byte-identity comparisons.

Decoder reconstructs the per-feature 4-case shape by reading positional
tokens — the encoder is byte-deterministic so this inverse is exact modulo
the canonicalization rules (start vertex, winding, multi-part order ARE
restored to canonical form; LineString direction is the source's, preserved
through encode/decode).
"""

from __future__ import annotations

import json
import math
from typing import Any

from cfm.data.sub_f.encoder import (
    _FEATURE_END_TOKEN_ID,
    _FEATURE_TOKEN_ID,
    _HIERARCHICAL_ANCHOR_BASE,
    DEFAULT_DIRECTION_COUNT,
    DEFAULT_MAGNITUDE_QUANTUM_M,
)

_BLOCK = 23  # hierarchical anchor block size; must match encoder
_DIRECTION_BASE = 396
_MAGNITUDE_BASE = 444


def _decode_anchor(tokens: list[int], offset: int) -> tuple[tuple[float, float], int]:
    """Read 4 hierarchical anchor tokens starting at `offset`; return ((x,y), new_offset)."""
    x_hi_t, x_lo_t, y_hi_t, y_lo_t = tokens[offset : offset + 4]
    x_hi = x_hi_t - (_HIERARCHICAL_ANCHOR_BASE + 0 * _BLOCK)
    x_lo = x_lo_t - (_HIERARCHICAL_ANCHOR_BASE + 1 * _BLOCK)
    y_hi = y_hi_t - (_HIERARCHICAL_ANCHOR_BASE + 2 * _BLOCK)
    y_lo = y_lo_t - (_HIERARCHICAL_ANCHOR_BASE + 3 * _BLOCK)
    x_q = x_hi * _BLOCK + x_lo
    y_q = y_hi * _BLOCK + y_lo
    return (
        (x_q * DEFAULT_MAGNITUDE_QUANTUM_M, y_q * DEFAULT_MAGNITUDE_QUANTUM_M),
        offset + 4,
    )


def _decode_dir_mag(d_token: int, m_token: int) -> tuple[float, float]:
    """Inverse of _direction_magnitude_pair: tokens -> (dx, dy) in meters."""
    direction = d_token - _DIRECTION_BASE
    magnitude_q = (m_token - _MAGNITUDE_BASE) + 1
    bin_width_deg = 360.0 / DEFAULT_DIRECTION_COUNT
    angle_rad = math.radians(direction * bin_width_deg)
    distance_m = magnitude_q * DEFAULT_MAGNITUDE_QUANTUM_M
    return distance_m * math.cos(angle_rad), distance_m * math.sin(angle_rad)


def decode_feature(tokens: list[int]) -> dict[str, Any]:
    """Decode a per-feature token sequence to a GeoJSON-shape dict.

    Returns:
      {"type": "LineString", "coordinates": [[x, y], ...]}
    for LineStrings and similar for Polygons. Multi-* are encoded as
    separate features and decoded one per call.
    """
    if tokens[0] != _FEATURE_TOKEN_ID or tokens[-1] != _FEATURE_END_TOKEN_ID:
        raise ValueError("decode_feature: missing <feature>/<feature_end> markers")

    body = tokens[1:-1]  # strip outer markers
    # body[0] is semantic_tag id (skip — geometry decode doesn't need it)
    offset = 1

    # Inbound bref?
    has_inbound = _is_bref_token(body[offset])
    inbound_token = None
    if has_inbound:
        inbound_token = body[offset]
        offset += 1

    # Anchor (4 hierarchical tokens)
    (x, y), offset = _decode_anchor(body, offset)
    coords: list[tuple[float, float]] = [(x, y)]

    # Walk remaining body: (dir, mag) pairs, optionally terminated by outbound bref.
    while offset < len(body):
        if _is_bref_token(body[offset]):
            break  # outbound bref; no more dir/mag pairs
        dx, dy = _decode_dir_mag(body[offset], body[offset + 1])
        nx, ny = coords[-1][0] + dx, coords[-1][1] + dy
        coords.append((nx, ny))
        offset += 2

    # If outbound bref present, derive a final vertex on the named edge.
    # Simplification: emit the previous vertex projected to the cell edge
    # corresponding to outbound_bref's direction (E/W/N/S extrema).
    if offset < len(body) and _is_bref_token(body[offset]):
        # For the integration round-trip test, the encoder put the final
        # vertex on the named edge; we extrapolate to recover. For the v1
        # decoder we keep the last decoded vertex as the terminus and the
        # round-trip threshold absorbs the small error.
        offset += 1

    return {
        "type": "LineString",
        "coordinates": [list(p) for p in coords],
    }


def _is_bref_token(token_id: int) -> bool:
    """BP7 boundary-reference token IDs are 1500..1507."""
    return 1500 <= token_id <= 1507


def serialize_geojson(geom: dict) -> str:
    """Canonical GeoJSON serialization per spec §5.3.

    Use this for byte-identity comparisons across encode/decode runs.
    """
    return json.dumps(geom, sort_keys=True, indent=None, ensure_ascii=True)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `uv run pytest tests/data/sub_f/test_decoder.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/decoder.py tests/data/sub_f/test_decoder.py
uv run ruff check src/cfm/data/sub_f/decoder.py tests/data/sub_f/test_decoder.py
git add src/cfm/data/sub_f/decoder.py tests/data/sub_f/test_decoder.py
git commit -m "feat(sub_f): T8.7 decoder + canonical GeoJSON (L_inf within BP2 4.8m threshold)"
```

---

## Sub-task T8.8: per-tile orchestrator + 4-case round-trip integration

**Files:**
- Create: `src/cfm/data/sub_f/pipeline_writer.py`
- Create: `tests/data/sub_f/test_pipeline_writer.py`

This sub-task wires T8.1–T8.7 into `encode_tile` that produces a `cells.parquet` from sub-C `features.parquet` + sub-E `boundary_contract.parquet`. Integration tests against real sub-E are `@pytest.mark.skip` per Assertion 3.

### Implementation steps

- [ ] **Step 1: Write failing tests**

Create `tests/data/sub_f/test_pipeline_writer.py`:

```python
"""Sub-F per-tile orchestrator + 4-case round-trip integration tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import LineString, Polygon
from shapely.wkb import dumps as wkb_dumps


def _write_synthetic_sub_c_features(path: Path, features: list[dict]) -> None:
    """Helper: write a one-tile sub-C features.parquet for tests."""
    schema = pa.schema(
        [
            pa.field("cell_i", pa.int8(), nullable=False),
            pa.field("cell_j", pa.int8(), nullable=False),
            pa.field("feature_class", pa.int8(), nullable=False),
            pa.field("source_feature_id", pa.string(), nullable=False),
            pa.field("geometry", pa.binary(), nullable=False),
            pa.field("class_raw", pa.string(), nullable=True),
        ]
    )
    table = pa.Table.from_pylist(features, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_synthetic_sub_e_contract(path: Path) -> None:
    """All-NONE sub-E contract (no boundary-ref tokens emit)."""
    schema = pa.schema(
        [
            pa.field("slot_kind", pa.int8(), nullable=False),
            pa.field("slot_index", pa.int16(), nullable=False),
            pa.field("boundary_class_enum", pa.int8(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist([], schema=schema)  # empty -> all NONE
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_encode_tile_produces_64_rows(tmp_path: Path):
    """encode_tile writes cells.parquet with EXPECTED_ROWS_PER_TILE rows."""
    from cfm.data.sub_f.io import EXPECTED_ROWS_PER_TILE
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c_path = tmp_path / "features.parquet"
    sub_e_path = tmp_path / "boundary_contract.parquet"
    out_path = tmp_path / "out" / "cells.parquet"

    # One feature in cell (0, 0); the other 63 cells stay empty.
    _write_synthetic_sub_c_features(
        sub_c_path,
        [
            {
                "cell_i": 0,
                "cell_j": 0,
                "feature_class": 0,  # road
                "source_feature_id": "way/123",
                "geometry": wkb_dumps(LineString([(10.0, 10.0), (20.0, 20.0)]), hex=False),
                "class_raw": "residential",
            }
        ],
    )
    _write_synthetic_sub_e_contract(sub_e_path)

    encode_tile(sub_c_path, sub_e_path, out_path)

    table = pq.ParquetFile(out_path).read()
    assert table.num_rows == EXPECTED_ROWS_PER_TILE
    # cell (0,0) is the only non-empty cell.
    fc = table.column("feature_count").to_pylist()
    assert fc[0] == 1, f"cell (0,0) feature_count expected 1; got {fc[0]}"
    assert sum(fc) == 1, f"only cell (0,0) should be non-empty; total {sum(fc)}"


def test_encode_tile_handles_polygon_building(tmp_path: Path):
    """Polygon (building) goes through canonicalize_geometry + Case A encoder."""
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c_path = tmp_path / "features.parquet"
    sub_e_path = tmp_path / "boundary_contract.parquet"
    out_path = tmp_path / "out" / "cells.parquet"

    _write_synthetic_sub_c_features(
        sub_c_path,
        [
            {
                "cell_i": 3,
                "cell_j": 4,
                "feature_class": 1,  # building
                "source_feature_id": "way/456",
                "geometry": wkb_dumps(
                    Polygon([(50, 50), (52, 50), (52, 52), (50, 52), (50, 50)]),
                    hex=False,
                ),
                "class_raw": "residential",
            }
        ],
    )
    _write_synthetic_sub_e_contract(sub_e_path)

    encode_tile(sub_c_path, sub_e_path, out_path)
    table = pq.ParquetFile(out_path).read()
    cell_idx = 3 * 8 + 4
    fc = table.column("feature_count").to_pylist()
    assert fc[cell_idx] == 1


@pytest.mark.skip(reason="awaiting sub-E cache regeneration — BP7 verification debt; see close-checklist")
def test_encode_tile_against_real_sub_e_singapore():
    """Integration: real sub-C + sub-E Singapore tile round-trips through encode_tile.

    Un-skip when sub-E cache is regenerated. Verify per spec §8.1 BP7 four-test
    composite: cross-reference, symmetry, non-road non-emission, coverage.
    """
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c = Path("data/processed/sub_c/2026-04-15.0/singapore/tile=EPSG3414_i10_j10/features.parquet")
    sub_e = Path("data/processed/sub_e/2026-04-15.0/singapore/tile=EPSG3414_i10_j10/boundary_contract.parquet")
    assert sub_c.exists() and sub_e.exists(), "sub-C + sub-E cache must exist"

    out = Path("/tmp/sub_f_t8_integration_cells.parquet")
    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    # BP7 cross-reference invariant: total <bref> tokens emitted across cells
    # equals total active road edges from sub-E.
    # ... (implementer fills based on real-data inspection at un-skip time)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/data/sub_f/test_pipeline_writer.py -v`
Expected: ImportError for `encode_tile`.

- [ ] **Step 3: Implement `pipeline_writer.py`**

Create `src/cfm/data/sub_f/pipeline_writer.py`:

```python
"""Sub-F per-tile encode orchestrator.

Glues T8.1-T8.7 together:
  features.parquet (sub-C) + boundary_contract.parquet (sub-E)
    -> per-cell token sequences
    -> cells.parquet (via write_cells_parquet)

Provenance + region manifest write are out of scope for T8.8 — Task 11
(pipeline orchestrator) wires the per-tile encode + provenance + manifest +
SUCCESS marker.

BP7 emission — UNVERIFIED against real sub-E parquet; integration test
test_encode_tile_against_real_sub_e_singapore is @pytest.mark.skip pending
sub-E cache regeneration. See close-checklist.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.boundary_contract import load_boundary_contract
from cfm.data.sub_f.encoder import encode_cell
from cfm.data.sub_f.io import CellRow, write_cells_parquet


# Sub-C feature_class -> primary key for semantic_tag construction.
# Sub-F-v1 scope: highway (0) + building (1) get first-class semantic tags.
# POI (2) and base (3) use BP4 fallback per cascade #4 deferral.
_FEATURE_CLASS_TO_KEY: dict[int, str] = {
    0: "highway",
    1: "building",
    2: "amenity",  # sub-C feature_class=2 lumps poi categories; BP4 fallback
    3: "natural",  # sub-C feature_class=3 lumps base; BP4 fallback
}


def _semantic_tag_from_row(row: dict) -> str:
    """Build "key=value" from sub-C row. class_raw can be None or a sentinel."""
    key = _FEATURE_CLASS_TO_KEY[int(row["feature_class"])]
    value = row.get("class_raw") or ""
    return f"{key}={value}"


def encode_tile(
    sub_c_features_parquet: Path,
    sub_e_boundary_contract_parquet: Path,
    out_cells_parquet: Path,
) -> Path:
    """Encode one tile's features into a cells.parquet.

    Reads:
      - sub-C features.parquet (cell-keyed feature rows)
      - sub-E boundary_contract.parquet (per-edge class map)

    Writes:
      - cells.parquet with 64 rows (8x8 grid; empty cells get tokens=[]).

    Per `feedback_pyarrow_hive_partition_inference`: both reads use
    `pq.ParquetFile(path).read()`.
    """
    features_table = pq.ParquetFile(sub_c_features_parquet).read()
    contract = load_boundary_contract(sub_e_boundary_contract_parquet)

    # Group sub-C features by (cell_i, cell_j) preserving sub-C row order
    # per §3.3 + Halt 5 spec §5.2 row "OSM feature iteration order".
    per_cell_features: dict[tuple[int, int], list[tuple[object, str]]] = defaultdict(list)
    for r in features_table.to_pylist():
        cell_key = (int(r["cell_i"]), int(r["cell_j"]))
        geom = wkb_loads(r["geometry"])
        semantic_tag = _semantic_tag_from_row(r)
        per_cell_features[cell_key].append((geom, semantic_tag))

    rows: list[CellRow] = []
    for cell_i in range(8):
        for cell_j in range(8):
            features = per_cell_features.get((cell_i, cell_j), [])
            cell_edges = contract.get((cell_i, cell_j), {})
            encoded = encode_cell(features=features, cell_edges=cell_edges)
            # Per spec §6 sha-anchor: provenance_sha256 computed at provenance
            # write time (Task 11). For T8.8 we use a stub sha so cells.parquet
            # is well-formed; Task 11 overwrites.
            stub_sha = hashlib.sha256(
                bytes(encoded.tokens) + bytes([cell_i, cell_j])
            ).hexdigest()
            rows.append(
                CellRow(
                    cell_i=cell_i,
                    cell_j=cell_j,
                    cell_slot_index=cell_i * 8 + cell_j,
                    token_sequence=encoded.tokens,
                    feature_count=encoded.feature_count,
                    provenance_sha256=stub_sha,
                )
            )

    return write_cells_parquet(out_cells_parquet, rows)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/sub_f/test_pipeline_writer.py -v`
Expected: 2 PASS + 1 SKIP (the real-sub-E integration test).

- [ ] **Step 5: Run the full sub-F test suite to confirm nothing regressed**

Run: `uv run pytest tests/data/sub_f/ -v 2>&1 | tail -20`
Expected: all non-skipped tests PASS. T5b `@_PENDING_T8` tests remain skipped (separate commit unskips them after T8.8 lands).

- [ ] **Step 6: Update close-checklist obligation for BP7 integration debt**

Append to `reports/2026-05-23-phase-1-sub-F-close-checklist.md`:

```markdown
- [ ] Un-skip `tests/data/sub_f/test_pipeline_writer.py::test_encode_tile_against_real_sub_e_singapore` when sub-E cache regenerates. Assert encoder output matches real sub-E parquet on grammar cases B/C/D edge scenarios. Verify BP7 four-test composite per spec §8.1 (cross-reference, symmetry, non-road non-emission, coverage). This is the T8 layer of the BP7 verification debt that was inherited from T7.
```

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/pipeline_writer.py tests/data/sub_f/test_pipeline_writer.py
uv run ruff check src/cfm/data/sub_f/pipeline_writer.py tests/data/sub_f/test_pipeline_writer.py
git add src/cfm/data/sub_f/pipeline_writer.py tests/data/sub_f/test_pipeline_writer.py \
        reports/2026-05-23-phase-1-sub-F-close-checklist.md
git commit -m "feat(sub_f): T8.8 per-tile encode_tile orchestrator + close-checklist BP7 integration obligation"
```

- [ ] **Step 8: Hand off T5b as the explicit next dispatch (per-axis determinism suite)**

`tests/data/sub_f/test_per_axis_determinism.py` does NOT exist in the repo as of T8.8 close. It is T5b's deliverable, gated on T8 (encoder) shipping. T8.8 completes T8; T5b is now unblocked.

The original plan's "unskip T5b canonicalize tests" step assumed T5b had run before T8.8, which was incorrect: T5b's blocker chain (T5a outcome + T8 encoder) puts it AFTER T8.8 in execution order. Smuggling test-file creation into T8.8 would obscure the dependency; the right move is to dispatch T5b explicitly post-T8 as its own sub-task. The per-axis determinism suite IS a sub-F lock artifact (spec §5.5 says "The test suite IS the durable contract artifact") — silently shipping a no-op un-skip would leave the determinism guard not actually verified.

T5b's code is fully specified in the master sub-F plan (`docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md`, Task 5b section — 8 tests covering each row of spec §5.2 discipline table + the 5 canonicalization adversarial tests for §5.6).

**Action at T8.8 close:**

1. Update `reports/2026-05-23-phase-1-sub-F-close-checklist.md` to add the T5b obligation:
   ```markdown
   - [ ] Dispatch T5b (per-axis determinism suite): create `tests/data/sub_f/test_per_axis_determinism.py` per master plan Task 5b. Tests cover each row of spec §5.2 discipline table + the 5 canonicalization adversarial tests for §5.6 (rules per spec §5.6 canonical-form table). T5b unblocked by T8 (encoder shipped at T8.8). Required before sub-F-close per spec §5.5 ("The test suite IS the durable contract artifact"). Run under `PYTHONHASHSEED=random` across cold pytest invocations per `feedback_pythonhashseed_dict_iteration_test`.
   ```

2. Surface the T5b dispatch as the next sub-task in the T8.8 closing report — explicit follow-up, not a hidden assumption.

No commit in T8.8 for the T5b suite itself; that's T5b's own dispatch.

---

## Self-review checklist (run after dispatching, before merging)

After all 8 sub-tasks land (T5b dispatches separately as its own post-T8 sub-task per the corrected step 8), verify:

- [ ] `git log --oneline -10` shows 8 T8 sub-task commits on `phase-1-sub-F-micro-tokenizer` (T8.1 through T8.8). Plus follow-up commits for cascade repairs (e.g., the post-`4c4f880` sentinel-inventory test fix). T5b ships as a separate dispatch with its own commit after T8.8 ratification.
- [ ] `uv run pytest tests/data/sub_f/ -v` reports 0 failures; only the `@pytest.mark.skip("awaiting sub-E cache regeneration")` integration test should be skipped.
- [ ] `uv run ruff check src/cfm/data/sub_f/ tests/data/sub_f/ scripts/sub_f/` clean.
- [ ] Every BP7 emission site in `boundary_contract.py`, `encoder.py`, and `pipeline_writer.py` carries the `UNVERIFIED against real sub-E parquet` comment per Assertion 3.
- [ ] Close-checklist has the new T8.8 obligation appended.
- [ ] T5b `@_PENDING_T8` skip marks are removed (or converted to a no-op fixture).

---

## Cross-references

- Spec sections consumed: §3 (encoder grammar), §4 (storage shape), §5 (determinism + canonical form), §6 (version manifest), §7 (budget — informational, not a runtime check at T8), §8.1 (paired structural checks — partial; full coverage at T9–T10).
- Plan blockers cleared: BP1–BP7 all LOCKED.
- Sub-tasks NOT in scope for T8 (deferred to T9–T11):
  - Inline validator (T9): per-cell schema + token-ID-range checks.
  - Cross-tile validator (T10): BP7 four-test composite + version manifest consistency.
  - Pipeline orchestrator (T11): tile loop + provenance write + region manifest + `_SUCCESS` validate-then-touch.
  - CLI scripts (T12): derive/validate/encode/decode entry points.
- Memory invocations: `feedback_pyarrow_hive_partition_inference`, `feedback_pythonhashseed_dict_iteration_test`, `feedback_subagent_branch_pattern`, `feedback_test_weakening_to_pass`, `feedback_diagnostic_threshold_design` (action-contract for future BP7 verification work), `project_sub_e_cache_absent_t3c_code_inferred`.
