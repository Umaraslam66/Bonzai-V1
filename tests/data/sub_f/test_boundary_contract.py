"""Sub-F BP7 boundary-contract reader tests.

Reader's contract is SOURCE-DERIVED from sub-E's locked writer/rotation/
derivation modules (see boundary_contract.py module docstring + the
_SUB_E_CONTRACT constant for file:line cites). Synthetic fixtures test
reader self-consistency against the source-derived contract. Real sub-E
parquet cache is absent (project_sub_e_cache_absent_t3c_code_inferred
memory); the residual absent-data debt is narrow: empirical numeric
ratios (T3c stage-4 avg crossings/cell) and first real-data integration
end-to-end. Integration tests against real sub-E live in T8.8's
test_pipeline_writer.py under `@pytest.mark.skip(reason="awaiting
sub-E cache regeneration")`.

FIXTURE_SOURCE_DERIVED_INDEX (below) documents which sub-E source claim
each fixture exercises, with file:line cites. Discipline tests guard
against future fixtures being added without an index entry.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Sub-E parquet schema - SOURCE-DERIVED, must match src/cfm/data/sub_e/writer.py:38-48
# _BOUNDARY_CONTRACT_SCHEMA bit-for-bit. 7 columns; boundary_class_enum is the
# only nullable column. See boundary_contract.py module docstring for sentinel
# semantics (NULL iff scope_marker != 0).
_SUB_E_SCHEMA = pa.schema(
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


# Source-derived fixture index. Each entry documents which sub-E source claim
# the fixture exercises (with file:line) - discipline tests guard against
# future fixtures being added without an index entry.
FIXTURE_SOURCE_DERIVED_INDEX: dict[str, dict] = {
    "single_active_major_row": {
        "encodes": (
            "Single INTERNAL_EDGE row (slot_kind=1, slot_index=0, scope_marker=0, "
            "boundary_class_enum=2 MAJOR_ROAD) at edge (lower_cell_i=0, "
            "lower_cell_j=0, axis=1) - south face of cell (0,0)."
        ),
        "exercises": (
            "Source-derived contract: writer.py:23-25 SlotKind enum; "
            "derivation.py:22 MAJOR_ROAD enum value=2; validator_inline.py:169 "
            "non-null iff scope_marker==0 invariant; reader join via "
            "(slot_kind, lower_cell_i, lower_cell_j, axis) from EdgeIdTuple "
            "in rotation.py:57-58."
        ),
        "revalidate_when_sub_e_lands": (
            "NOT REQUIRED. Pure source-derived schema/join test; reader "
            "behavior is determined by source. First-real-sub-E read is "
            "covered by the T8.8 integration test, not by re-running this."
        ),
    },
    "non_active_null_row": {
        "encodes": (
            "Single EXTERNAL_EDGE row (slot_kind=2, slot_index=0, "
            "scope_marker=1, boundary_class_enum=None)."
        ),
        "exercises": (
            "Source-derived sentinel encoding: writer.py:33-37 docstring + "
            "validator_inline.py:169-178 invariant (boundary_class_enum NULL "
            "iff scope_marker != 0). Reader must accept NULL and emit no "
            "BP7 token for this edge."
        ),
        "revalidate_when_sub_e_lands": ("NOT REQUIRED. NULL handling is source-determined."),
    },
    "tile_dir_path_layout": {
        "encodes": (
            "Single active MINOR_ROAD row written to tile=EPSG3414_i0_j0/boundary_contract.parquet."
        ),
        "exercises": (
            "feedback_pyarrow_hive_partition_inference: reader uses "
            "pq.ParquetFile(path).read(), never bare pq.read_table() on "
            "parent dir (would inject spurious 'tile' column and break "
            "schema match)."
        ),
        "revalidate_when_sub_e_lands": (
            "NOT REQUIRED. Hive-partition handling is reader code structure, source-determined."
        ),
    },
    "resolve_bref_tag_pure_function": {
        "encodes": (
            "No parquet - pure function test of resolve_bref_tag(direction, "
            "class_label). 8 active token tags + 2 non-emitting class returns."
        ),
        "exercises": (
            "Spec §3.7 BP7 vocab lock: 8 active tokens {N,E,S,W} x {MAJOR,MINOR}. "
            "Source: configs/sub_f/boundary_reference_vocab.yaml (slots 1500-1507)."
        ),
        "revalidate_when_sub_e_lands": (
            "NOT REQUIRED. Pure function; no sub-E parquet dependency."
        ),
    },
}


def _make_full_tile_rows(
    overrides: dict[tuple[int, int, int, int], dict] | None = None,
) -> list[dict]:
    """Construct 144 well-formed rows for one tile (112 INTERNAL + 32 EXTERNAL)
    per writer.py EXPECTED_TOTAL_ROWS. All rows default to non-active
    (scope_marker=1, boundary_class_enum=None) - caller overrides specific
    edges via a (slot_kind, lower_cell_i, lower_cell_j, axis) -> field-update dict.

    The 144-row enumeration walks every cell x edge per rotation.py and
    deduplicates internal edges shared between adjacent cells (each internal
    edge appears once, not twice).
    """
    from cfm.data.sub_e.rotation import EdgeKind, cell_to_edge_ids

    seen_internal: set[tuple[int, int, int]] = set()
    rows: list[dict] = []
    internal_slot_idx = 0
    external_slot_idx = 0
    overrides = overrides or {}

    for cell_i in range(8):
        for cell_j in range(8):
            edges = cell_to_edge_ids(cell_i, cell_j)
            for edge in (edges.north, edges.south, edges.west, edges.east):
                lower_i, lower_j, axis, kind = edge
                if kind is EdgeKind.INTERNAL:
                    key = (lower_i, lower_j, axis)
                    if key in seen_internal:
                        continue
                    seen_internal.add(key)
                    slot_kind = 1
                    slot_index = internal_slot_idx
                    internal_slot_idx += 1
                else:
                    slot_kind = 2
                    slot_index = external_slot_idx
                    external_slot_idx += 1
                row = {
                    "slot_kind": slot_kind,
                    "slot_index": slot_index,
                    "lower_cell_i": lower_i,
                    "lower_cell_j": lower_j,
                    "axis": axis,
                    "scope_marker": 1,  # non-active default
                    "boundary_class_enum": None,
                }
                ov = overrides.get((slot_kind, lower_i, lower_j, axis))
                if ov:
                    row.update(ov)
                rows.append(row)
    return rows


def _write_synthetic_sub_e_parquet(path: Path, rows: list[dict]) -> None:
    """Helper: write a one-tile sub-E boundary_contract.parquet for tests.

    Caller is responsible for row count + invariants - _make_full_tile_rows
    above produces a writer.py-compliant 144-row tile by default.
    """
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---- Reader behavior tests (source-derived contract) ---------------------


def test_load_boundary_contract_surfaces_active_major_on_correct_edge(tmp_path: Path):
    """Reader returns {(cell_i, cell_j): {direction: class_label}}; an active
    MAJOR row on south face of cell (0,0) - edge (lower_i=0, lower_j=0, axis=1)
    per rotation.py - surfaces as "MAJOR_ROAD" at cell (0,0) direction "S"
    AND at cell (0,1) direction "N" (shared internal edge).

    Fixture: single_active_major_row.
    """
    from cfm.data.sub_f.boundary_contract import load_boundary_contract

    rows = _make_full_tile_rows(
        overrides={
            (1, 0, 0, 1): {"scope_marker": 0, "boundary_class_enum": 2},  # active MAJOR
        }
    )
    parquet_path = tmp_path / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(parquet_path, rows)

    contract = load_boundary_contract(parquet_path)
    assert contract[(0, 0)]["S"] == "MAJOR_ROAD"
    assert contract[(0, 1)]["N"] == "MAJOR_ROAD"  # same edge from neighbor view
    # No other cell should report MAJOR_ROAD from this single override.
    other_majors = sum(
        1
        for (ci, cj), edges in contract.items()
        for d, cls in edges.items()
        if cls == "MAJOR_ROAD" and (ci, cj, d) not in {(0, 0, "S"), (0, 1, "N")}
    )
    assert other_majors == 0


def test_load_boundary_contract_null_row_yields_no_class(tmp_path: Path):
    """Per writer.py docstring + validator_inline.py:169 invariant:
    boundary_class_enum NULL iff scope_marker != 0. NULL rows are sub-E's
    on-disk encoding of "non-active" (the BOUNDARY_NOT_APPLICABLE semantic
    that's dataloader-only per derivation.py:20). Reader surfaces these as
    class_label "NONE" so the encoder does NOT emit a BP7 token.

    Fixture: non_active_null_row.
    """
    from cfm.data.sub_f.boundary_contract import load_boundary_contract

    rows = _make_full_tile_rows()  # all non-active by default
    parquet_path = tmp_path / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(parquet_path, rows)

    contract = load_boundary_contract(parquet_path)
    # Every cell x direction must have a class_label, defaulting to NONE
    # (no on-disk row -> not in our join, treated as NONE; null row also NONE).
    for (ci, cj), edges in contract.items():
        for d in ("N", "E", "S", "W"):
            assert edges[d] == "NONE", (
                f"cell ({ci},{cj}) direction {d}: expected NONE for non-active "
                f"row; got {edges[d]!r}"
            )


def test_resolve_bref_token_returns_correct_8_token_form():
    """Per spec §3.7 BP7 vocab: 8 active tokens {N,E,S,W} x {MAJOR,MINOR}.

    Fixture: resolve_bref_tag_pure_function.
    """
    from cfm.data.sub_f.boundary_contract import resolve_bref_tag

    assert resolve_bref_tag("N", "MAJOR_ROAD") == "<bref_N_MAJOR>"
    assert resolve_bref_tag("E", "MAJOR_ROAD") == "<bref_E_MAJOR>"
    assert resolve_bref_tag("S", "MINOR_ROAD") == "<bref_S_MINOR>"
    assert resolve_bref_tag("W", "MINOR_ROAD") == "<bref_W_MINOR>"
    # NONE -> None (no token emitted)
    assert resolve_bref_tag("N", "NONE") is None


def test_load_boundary_contract_uses_pq_parquetfile_not_read_table(tmp_path: Path):
    """Per `feedback_pyarrow_hive_partition_inference`: must use
    pq.ParquetFile(path).read(), NOT bare pq.read_table() on a parent dir
    (would inject spurious 'tile' column and break schema match).

    Fixture: tile_dir_path_layout.
    """
    from cfm.data.sub_f.boundary_contract import load_boundary_contract

    rows = _make_full_tile_rows(
        overrides={(1, 3, 3, 0): {"scope_marker": 0, "boundary_class_enum": 3}}
    )
    parquet_path = tmp_path / "tile=EPSG3414_i0_j0" / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(parquet_path, rows)

    contract = load_boundary_contract(parquet_path)
    assert isinstance(contract, dict)
    assert len(contract) == 64  # full 8x8 grid


# ---- Source-derived fixture index discipline ----------------------------


def test_every_fixture_appears_in_source_derived_index():
    """Every synthetic fixture must have a corresponding entry in
    FIXTURE_SOURCE_DERIVED_INDEX with `encodes`, `exercises`, and
    `revalidate_when_sub_e_lands` keys. Discipline guard against silent
    fixture additions that bypass the source-derived audit trail.
    """
    expected = {
        "single_active_major_row",
        "non_active_null_row",
        "tile_dir_path_layout",
        "resolve_bref_tag_pure_function",
    }
    actual = set(FIXTURE_SOURCE_DERIVED_INDEX.keys())
    missing = expected - actual
    assert not missing, (
        f"fixtures used in tests but not indexed in FIXTURE_SOURCE_DERIVED_INDEX: {missing}."
    )
    for key, entry in FIXTURE_SOURCE_DERIVED_INDEX.items():
        for required_field in ("encodes", "exercises", "revalidate_when_sub_e_lands"):
            assert required_field in entry, (
                f"fixture {key!r} missing required field {required_field!r}"
            )


# ---- Fail-loud parse-boundary defenses ----------------------------------


def test_load_raises_on_schema_mismatch_extra_column(tmp_path: Path):
    """If sub-E ever changes the parquet schema (adds column, changes dtype),
    reader must surface SubEContractViolation rather than silently consuming
    the wrong shape. Source: writer.py:38-48 _BOUNDARY_CONTRACT_SCHEMA is
    the contract; T8.5's _EXPECTED_SCHEMA must match bit-for-bit.
    """
    from cfm.data.sub_f.boundary_contract import (
        SubEContractViolation,
        load_boundary_contract,
    )

    wrong_schema = pa.schema(
        [
            pa.field("slot_kind", pa.int8(), nullable=False),
            pa.field("slot_index", pa.int16(), nullable=False),
            pa.field("lower_cell_i", pa.int8(), nullable=False),
            pa.field("lower_cell_j", pa.int8(), nullable=False),
            pa.field("axis", pa.int8(), nullable=False),
            pa.field("scope_marker", pa.int8(), nullable=False),
            pa.field("boundary_class_enum", pa.int16(), nullable=True),
            pa.field("extra_drift_column", pa.string(), nullable=False),
        ]
    )
    rows = _make_full_tile_rows()
    pylist = [{**r, "extra_drift_column": "x"} for r in rows]
    table = pa.Table.from_pylist(pylist, schema=wrong_schema)
    path = tmp_path / "boundary_contract.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)

    with pytest.raises(SubEContractViolation, match=r"schema"):
        load_boundary_contract(path)


def test_load_raises_on_wrong_row_count(tmp_path: Path):
    """Sub-E writer.py:73-76 enforces 144 rows per tile (112 INTERNAL + 32
    EXTERNAL); reader's source-derived contract requires the same. A real
    sub-E violating this would indicate a writer regression.
    """
    from cfm.data.sub_f.boundary_contract import (
        SubEContractViolation,
        load_boundary_contract,
    )

    rows = _make_full_tile_rows()[:140]  # short by 4
    path = tmp_path / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(path, rows)

    with pytest.raises(SubEContractViolation, match=r"row count|144"):
        load_boundary_contract(path)


def test_load_raises_on_null_with_active_scope_marker(tmp_path: Path):
    """validator_inline.py:169 invariant: boundary_class_enum non-null iff
    scope_marker == 0. A row with scope_marker=0 AND null enum violates the
    invariant - reader must surface, not silently treat as NONE.
    """
    from cfm.data.sub_f.boundary_contract import (
        SubEContractViolation,
        load_boundary_contract,
    )

    rows = _make_full_tile_rows(
        overrides={(1, 0, 0, 1): {"scope_marker": 0, "boundary_class_enum": None}}
    )
    path = tmp_path / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(path, rows)

    with pytest.raises(SubEContractViolation, match=r"scope_marker|invariant"):
        load_boundary_contract(path)


def test_load_raises_on_not_applicable_enum_on_disk(tmp_path: Path):
    """BoundaryClass.BOUNDARY_NOT_APPLICABLE (enum=0) is forbidden on-disk
    per derivation.py:20 docstring ("dataloader-side only, never on-disk").
    Sub-E encodes BOUNDARY_NOT_APPLICABLE as parquet NULL; if enum=0 ever
    appears as a non-null value, sub-E has violated its own discipline.
    """
    from cfm.data.sub_f.boundary_contract import (
        SubEContractViolation,
        load_boundary_contract,
    )

    rows = _make_full_tile_rows(
        overrides={(1, 0, 0, 1): {"scope_marker": 0, "boundary_class_enum": 0}}
    )
    path = tmp_path / "boundary_contract.parquet"
    _write_synthetic_sub_e_parquet(path, rows)

    with pytest.raises(SubEContractViolation, match=r"BOUNDARY_NOT_APPLICABLE|boundary_class_enum"):
        load_boundary_contract(path)
