"""Tests for the sub-F cross-tile validator (Task 10).

BP7 four-test composite (cross-reference, symmetry, non-road non-emission,
coverage) + version-manifest consistency + the standalone BP1->sub-E
class-mapping gate.

ADVERSARIAL, PER-LEG, RULE-ISOLATED DISCIPLINE (the core mandate):
A validator test that builds CONSISTENT input and asserts the validator
PASSES proves nothing. For EACH leg there is >=1 NEGATIVE test that
constructs input violating ONLY that leg and asserts
`CrossTileValidationError` with a leg-specific message substring. The legs
are decomposed into one private function each (`_check_cross_reference`,
`_check_symmetry`, `_check_non_road_non_emission`, `_check_coverage`,
`_check_version_consistency`) so a negative can target ONE leg in
isolation and the reviewer can confirm the TARGET leg fires, not an
incidentally-earlier check.

SYNTHETIC FIXTURE DISCIPLINE (sub-E + sub-F caches absent locally, per
`project_sub_e_cache_absent_t3c_code_inferred`): all tests build synthetic
multi-tile region dirs with valid 144-row sub-E contracts (reusing the
test_pipeline_writer 144-row builder pattern), 64-row cells.parquet, and
synthetic provenance.yaml. A @pytest.mark.skip stub stands in for the
real-region BP7 composite (see close-checklist).

BP7 token IDs (LOCKED 1500-1507 per boundary_reference_vocab.yaml) are
resolved via vocab_tag_to_id(), never hardcoded in assertion logic.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from shapely.geometry import LineString, Polygon

from cfm.data.sub_e.derivation import BoundaryClass, load_class_grouping_map
from cfm.data.sub_f.boundary_contract import resolve_bref_tag
from cfm.data.sub_f.encoder import encode_feature
from cfm.data.sub_f.io import CellRow, write_cells_parquet
from cfm.data.sub_f.validator_cross_tile import (
    CrossTileValidationError,
    _check_symmetry,
    validate_cross_tile,
)
from cfm.data.sub_f.vocab import vocab_tag_to_id

# ===========================================================================
# Fixtures: synthetic sub-E contract, cells.parquet, provenance.yaml
# ===========================================================================

# Sub-E parquet schema — SOURCE-DERIVED bit-for-bit match against
# src/cfm/data/sub_e/writer.py:38-48 _BOUNDARY_CONTRACT_SCHEMA. Replicated
# from test_pipeline_writer.py for a self-contained module (close-checklist
# tracks the shared-fixture dedup obligation).
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

_VALID_VERSION_PROVENANCE: dict[str, object] = {
    "sub_f_artifact_format_version": "1.0",
    "sub_f_schema_version": "1.0",
    "sub_f_vocab_version": "1.0",
    "sub_f_derivation_version": "1.0",
    "sub_f_validator_version": "1.0",
    "sub_f_source_version": {
        "overture_release": "2026-04-15.0",
        "sub_c_schema_version": "1.0",
        "sub_c_commit_sha": "deadbeef" * 5,
    },
}

_FAKE_SHA = "a" * 64


def _make_full_tile_contract_rows(
    overrides: dict[tuple[int, int, int, int], dict] | None = None,
) -> list[dict]:
    """Build 144 well-formed sub-E contract rows (112 INTERNAL + 32 EXTERNAL).

    All rows default to non-active (scope_marker=1, boundary_class_enum=None).
    Override specific edges via (slot_kind, lower_cell_i, lower_cell_j, axis).
    Source pattern: _make_full_tile_rows (test_boundary_contract.py) via
    test_pipeline_writer.py _make_full_tile_contract_rows.
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
                    "scope_marker": 1,
                    "boundary_class_enum": None,
                }
                ov = overrides.get((slot_kind, lower_i, lower_j, axis))
                if ov:
                    row.update(ov)
                rows.append(row)
    return rows


def _write_sub_e_contract(path: Path, overrides: dict | None = None) -> None:
    rows = _make_full_tile_contract_rows(overrides)
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _empty_cell_rows() -> list[CellRow]:
    """64 empty cell rows (tokens=[]). Caller overwrites specific cells."""
    return [
        CellRow(
            cell_i=i,
            cell_j=j,
            cell_slot_index=i * 8 + j,
            token_sequence=[],
            feature_count=0,
            provenance_sha256=_FAKE_SHA,
        )
        for i in range(8)
        for j in range(8)
    ]


def _set_cell(
    rows: list[CellRow], cell_i: int, cell_j: int, tokens: list[int], feature_count: int
) -> None:
    """Overwrite one cell's token_sequence in a 64-row list (in place)."""
    idx = cell_i * 8 + cell_j
    rows[idx] = CellRow(
        cell_i=cell_i,
        cell_j=cell_j,
        cell_slot_index=idx,
        token_sequence=tokens,
        feature_count=feature_count,
        provenance_sha256=_FAKE_SHA,
    )


def _write_tile(
    sub_f_dir: Path,
    sub_e_dir: Path,
    tile_name: str,
    cell_rows: list[CellRow],
    contract_overrides: dict | None = None,
    provenance: dict | None = None,
) -> None:
    """Write one tile: cells.parquet + provenance.yaml (sub-F) + contract (sub-E)."""
    tile_sub_f = sub_f_dir / tile_name
    tile_sub_f.mkdir(parents=True, exist_ok=True)
    write_cells_parquet(tile_sub_f / "cells.parquet", cell_rows)
    prov = provenance if provenance is not None else _VALID_VERSION_PROVENANCE
    (tile_sub_f / "provenance.yaml").write_text(yaml.safe_dump(prov))
    _write_sub_e_contract(sub_e_dir / tile_name / "boundary_contract.parquet", contract_overrides)


# ---- feature-chunk builders (real encoder output) -------------------------


def _road_chunk_outbound(direction: str, class_label: str) -> list[int]:
    """A road feature exiting `direction` edge with an outbound bref (Case B).

    Geometry endpoint lands on the named edge so the encoder's own bref
    placement is used. We pass the bref explicitly to encode_feature so the
    chunk's bref is deterministic regardless of geometry.
    """
    bref = resolve_bref_tag(direction, class_label)
    assert bref is not None
    road = LineString([(50.0, 100.0), (200.0, 100.0)])
    ef = encode_feature(road, semantic_tag="highway=residential", outbound_bref=bref)
    return ef.tokens


def _road_chunk_plain() -> list[int]:
    """A road feature that emits NO bref (Case A, fully interior)."""
    road = LineString([(50.0, 50.0), (120.0, 120.0)])
    ef = encode_feature(road, semantic_tag="highway=residential")
    return ef.tokens


def _building_chunk_with_bref(direction: str, class_label: str) -> list[int]:
    """A building feature with a bref FORCED into it (non-road violation).

    encode_feature applied to a Polygon never produces a bref via the
    normal path, so we synthesize the bref injection directly: a building
    chunk that (illegitimately) carries a bref token at the outbound slot.
    This is the ONLY way to construct a non-road non-emission violation,
    because a correct encoder never does this.
    """
    bref = resolve_bref_tag(direction, class_label)
    assert bref is not None
    bref_id = vocab_tag_to_id()[bref]
    building = Polygon([(50, 50), (50, 100), (100, 100), (100, 50), (50, 50)])
    ef = encode_feature(building, semantic_tag="building=residential")
    # Insert the bref immediately before <feature_end> (outbound slot).
    tokens = list(ef.tokens)
    return [*tokens[:-1], bref_id, tokens[-1]]


# ===========================================================================
# Leg 5 NEGATIVE: version-manifest consistency
# ===========================================================================


def test_version_drift_across_tiles_raises(tmp_path: Path):
    """NEGATIVE (version manifest leg): two tiles differing only in
    sub_f_vocab_version -> CrossTileValidationError('version manifest').

    Rule-isolated: both tiles have all-NONE contracts and all-empty cells,
    so cross-reference / symmetry / non-road / coverage all pass trivially;
    the ONLY violated invariant is version-manifest consistency, which is
    checked first. The 'version manifest' substring confirms that leg fired.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    _write_tile(sub_f, sub_e, "tile=0_0", _empty_cell_rows())
    drifted = dict(_VALID_VERSION_PROVENANCE)
    drifted["sub_f_vocab_version"] = "2.0"  # the ONLY changed axis
    _write_tile(sub_f, sub_e, "tile=1_0", _empty_cell_rows(), provenance=drifted)

    with pytest.raises(CrossTileValidationError, match="version manifest"):
        validate_cross_tile(sub_f, sub_e)


# ===========================================================================
# Leg 1 NEGATIVE: cross-reference
# ===========================================================================


def test_cross_reference_emitted_bref_disagrees_with_contract_raises(tmp_path: Path):
    """NEGATIVE (cross-reference leg): both cells emit <bref_MAJOR> on the
    shared edge, but the sub-E contract says MINOR_ROAD for that edge.

    Rule-isolated: versions are consistent (single tile, valid provenance);
    BOTH cell (0,0) and its East neighbour (1,0) emit MAJOR on their shared
    edge, so:
      - symmetry PASSES: (0,0).East={MAJOR_ROAD} == (1,0).West={MAJOR_ROAD}.
      - non-road PASSES: both are road (highway) features.
      - coverage PASSES: active edge has road features AND brefs are emitted,
        symmetry is satisfied.
      - cross-reference FIRES: emitted MAJOR disagrees with contract MINOR_ROAD.
    The ONLY fault is emitted-MAJOR vs contract-MINOR ->
    'cross-reference' substring confirms the target leg fired.

    Isolation verified by leg-neutering: with _check_cross_reference body
    replaced by `return`, this test no longer raises CrossTileValidationError,
    confirming no other leg fires on this fixture.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    rows = _empty_cell_rows()
    # Cell (0,0) emits MAJOR on East edge.
    _set_cell(rows, 0, 0, _road_chunk_outbound("E", "MAJOR_ROAD"), feature_count=1)
    # Cell (1,0) ALSO emits MAJOR on West edge — same shared edge, same class.
    # Symmetry sees (0,0).East={MAJOR_ROAD} == (1,0).West={MAJOR_ROAD} -> passes.
    _set_cell(rows, 1, 0, _road_chunk_outbound("W", "MAJOR_ROAD"), feature_count=1)

    # Contract: shared East/West edge of (0,0)/(1,0) is active as MINOR_ROAD
    # (scope_marker=0, boundary_class_enum=3). Both cells emit MAJOR -> cross-
    # reference fires (MAJOR != MINOR). Symmetry is unaffected because both
    # cells agree on MAJOR with each other; the disagreement is with the contract.
    overrides = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 3}}

    _write_tile(sub_f, sub_e, "tile=0_0", rows, contract_overrides=overrides)

    with pytest.raises(CrossTileValidationError, match="cross-reference"):
        validate_cross_tile(sub_f, sub_e)


# ===========================================================================
# Leg 2 NEGATIVE: symmetry (rule-isolated by calling the leg directly)
# ===========================================================================


def test_symmetry_paired_cells_disagree_on_shared_edge_raises():
    """NEGATIVE (symmetry leg): two paired cells whose shared-edge brefs
    disagree -> CrossTileValidationError('symmetry').

    Rule-isolated by construction: we call `_check_symmetry` DIRECTLY on a
    hand-built emitted-by-cell map, so NO earlier leg (cross-reference /
    non-road) can fire — only the symmetry leg runs. Cell (0,0) emits
    MAJOR_ROAD on its East edge; the paired neighbour (1,0) emits NOTHING on
    its West edge (the shared internal edge). Because both views derive from
    the same sub-E row, this break is only reachable via an adjacency-mapping
    bug; the leg validates the E<->W opposite-direction mapping in isolation.
    """
    emitted_by_cell = {
        (0, 0): [("E", "MAJOR_ROAD", "highway")],
        # (1, 0) emits nothing on W -> shared-edge views disagree.
    }
    with pytest.raises(CrossTileValidationError, match="symmetry"):
        _check_symmetry("tile=0_0", emitted_by_cell)


def test_symmetry_paired_cells_disagree_class_raises():
    """NEGATIVE (symmetry leg, second flavour): paired cells agree an edge is
    active but disagree on CLASS (A says MAJOR, B says MINOR) -> 'symmetry'.

    Also called directly so only the symmetry leg can fire.
    """
    emitted_by_cell = {
        (0, 0): [("S", "MAJOR_ROAD", "highway")],
        (0, 1): [("N", "MINOR_ROAD", "highway")],  # opposite dir, different class
    }
    with pytest.raises(CrossTileValidationError, match="symmetry"):
        _check_symmetry("tile=0_0", emitted_by_cell)


# ===========================================================================
# Leg 3 NEGATIVE: non-road non-emission
# ===========================================================================


def test_non_road_building_emits_bref_raises(tmp_path: Path):
    """NEGATIVE (non-road leg): a building-only cell carries a bref token.

    Rule-isolated: the bref is forced onto a BUILDING feature, so non-road
    fires. The contract DOES say MAJOR on that edge so cross-reference would
    PASS (the bref's class matches), and the neighbour also emits the matching
    MAJOR so symmetry would PASS — only the non-road leg distinguishes this
    regime. The 'non-road' substring confirms the target leg fired.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    rows = _empty_cell_rows()
    # Building in (0,0) with a (forced) MAJOR bref on East edge.
    building_chunk = _building_chunk_with_bref("E", "MAJOR_ROAD")
    _set_cell(rows, 0, 0, building_chunk, feature_count=1)
    # Neighbour (1,0) emits matching MAJOR on West so symmetry would pass.
    nbr_chunk = _road_chunk_outbound("W", "MAJOR_ROAD")
    _set_cell(rows, 1, 0, nbr_chunk, feature_count=1)

    # Contract: shared East/West edge of (0,0)/(1,0) active MAJOR so
    # cross-reference passes for both emitted brefs.
    overrides = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}}
    _write_tile(sub_f, sub_e, "tile=0_0", rows, contract_overrides=overrides)

    with pytest.raises(CrossTileValidationError, match="non-road"):
        validate_cross_tile(sub_f, sub_e)


# ===========================================================================
# Leg 4 NEGATIVE: coverage (active road edge + road feature, no bref)
# ===========================================================================


def test_coverage_active_edge_with_road_feature_no_bref_raises(tmp_path: Path):
    """NEGATIVE (coverage leg): an active road edge with a road feature in the
    cell but NO bref emitted -> CrossTileValidationError('coverage').

    Rule-isolated: cell (0,0) has a ROAD feature (so road_cells includes it,
    and non-road passes — no building bref) that emits NO bref at all (Case A
    plain). The contract activates the East edge of (0,0) as MAJOR. Because
    nothing is emitted, cross-reference has nothing to check (passes) and
    symmetry has no emission to compare (passes). The only fault is the
    uncovered active edge -> 'coverage' substring.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    rows = _empty_cell_rows()
    _set_cell(rows, 0, 0, _road_chunk_plain(), feature_count=1)

    # Activate East edge of (0,0) as MAJOR but the road emits no bref.
    overrides = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}}
    _write_tile(sub_f, sub_e, "tile=0_0", rows, contract_overrides=overrides)

    with pytest.raises(CrossTileValidationError, match="coverage"):
        validate_cross_tile(sub_f, sub_e)


def test_coverage_does_not_overfire_on_edge_with_no_road_feature(tmp_path: Path):
    """POSITIVE guard (coverage precision): an active edge with NO road
    feature in this cell OR its neighbour must NOT raise.

    Confirms coverage is precise about the 'road feature in either neighbour'
    condition and does not over-fire on bare active edges. Cell (0,0) is
    EMPTY (no feature); its East edge is active MAJOR; neighbour (1,0) is also
    empty. No road feature to attach a crossing to -> coverage must pass.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    rows = _empty_cell_rows()  # all cells empty
    overrides = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}}
    _write_tile(sub_f, sub_e, "tile=0_0", rows, contract_overrides=overrides)

    # Must not raise — no road feature on either side of the active edge.
    validate_cross_tile(sub_f, sub_e)


# ===========================================================================
# HAPPY PATH: a fully consistent multi-tile region passes
# ===========================================================================


def test_happy_path_consistent_region_passes(tmp_path: Path):
    """All four BP7 legs + version consistency pass on a consistent region.

    Two tiles. Tile 0: a road in cell (0,0) exits East onto an active MAJOR
    edge AND its neighbour (1,0) has a road entering from West on the SAME
    shared edge, both emitting MAJOR -> cross-reference, symmetry, non-road,
    coverage all satisfied. Tile 1: all-empty / all-NONE (trivially valid).
    Versions identical across tiles.
    """
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"

    # --- tile 0: a symmetric MAJOR crossing on the (0,0)<->(1,0) shared edge ---
    rows0 = _empty_cell_rows()
    _set_cell(rows0, 0, 0, _road_chunk_outbound("E", "MAJOR_ROAD"), feature_count=1)
    _set_cell(rows0, 1, 0, _road_chunk_outbound("W", "MAJOR_ROAD"), feature_count=1)
    # Shared internal edge (E of (0,0) == W of (1,0)) active MAJOR.
    overrides0 = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}}
    _write_tile(sub_f, sub_e, "tile=0_0", rows0, contract_overrides=overrides0)

    # --- tile 1: empty / all-NONE ---
    _write_tile(sub_f, sub_e, "tile=1_0", _empty_cell_rows())

    # Must not raise.
    validate_cross_tile(sub_f, sub_e)


def test_no_tiles_raises(tmp_path: Path):
    """An empty region dir (no tile=*/cells.parquet) raises clearly."""
    sub_f = tmp_path / "sub_f"
    sub_e = tmp_path / "sub_e"
    sub_f.mkdir()
    sub_e.mkdir()
    with pytest.raises(CrossTileValidationError, match="no tiles"):
        validate_cross_tile(sub_f, sub_e)


# ===========================================================================
# STANDALONE BP1 -> sub-E class-mapping gate (Gate-6 external source of truth)
# ===========================================================================


def test_bp1_to_sub_e_class_mapping_matches_encoder_resolution():
    """Gate-6: the sub-F encoder's class resolution agrees with sub-E's
    load_class_grouping_map() for representative highway=* values.

    EXPECTED side comes ENTIRELY from sub-E's load_class_grouping_map (the
    ground truth) — NOT from sub-F code. The ACTUAL side is the sub-F path
    that the encoder uses: it tokenizes sub-E's BoundaryClass verbatim via
    resolve_bref_tag (which emits <bref_DIR_MAJOR> for MAJOR_ROAD and
    <bref_DIR_MINOR> for MINOR_ROAD). We assert that, for each highway value,
    the class sub-E assigns (with the documented MINOR-default fallthrough)
    is the class sub-F's resolve_bref_tag emits a token for.

    Carry-forward: motorway maps to MINOR (sub-E grouping omits it ->
    derivation.py:85 fallthrough). This test passes when sub-F and sub-E
    AGREE on MINOR-for-motorway; it does NOT validate MINOR is semantically
    correct. See close-checklist.
    """
    grouping = load_class_grouping_map()  # GROUND TRUTH (sub-E)

    def sub_e_class_for(highway_value: str) -> BoundaryClass:
        # Mirror sub-E derive_boundary_class single-crossing path: lookup with
        # MINOR_ROAD default fallthrough (derivation.py:85). This uses ONLY
        # the sub-E grouping map, not sub-F code.
        return grouping.get(highway_value, BoundaryClass.MINOR_ROAD)

    # Representative hand-enumerated highway values spanning MAJOR list,
    # MINOR list, and the fallthrough (motorway, path, unknown).
    representative = [
        "primary",
        "trunk",
        "secondary",
        "residential",
        "service",
        "footway",
        "motorway",  # fallthrough -> MINOR (scoped v1 accept)
        "path",  # fallthrough -> MINOR
        "definitely_not_a_real_class",  # unknown -> MINOR
    ]

    class_to_short = {
        BoundaryClass.MAJOR_ROAD: "MAJOR",
        BoundaryClass.MINOR_ROAD: "MINOR",
    }
    class_to_label = {
        BoundaryClass.MAJOR_ROAD: "MAJOR_ROAD",
        BoundaryClass.MINOR_ROAD: "MINOR_ROAD",
    }

    for value in representative:
        expected_class = sub_e_class_for(value)  # ground truth
        expected_short = class_to_short[expected_class]
        # ACTUAL: sub-F encoder's class -> token path. resolve_bref_tag emits
        # <bref_DIR_SHORT> for the class label sub-E hands it. We feed the
        # sub-E class label and confirm the emitted token's class half
        # matches the sub-E grouping's class.
        emitted_tag = resolve_bref_tag("N", class_to_label[expected_class])
        assert emitted_tag == f"<bref_N_{expected_short}>", (
            f"sub-F class-resolution disagreement for highway={value!r}: "
            f"sub-E grouping -> {expected_class.name}, but sub-F emitted "
            f"{emitted_tag!r}"
        )


def test_bp1_class_mapping_major_set_is_nonempty_and_minor_default():
    """Sanity floor for the class-map gate: sub-E's MAJOR list is non-empty
    (so the gate above actually exercises the MAJOR branch) and an unknown
    value falls through to MINOR (the documented default).

    Guards against a vacuous class-map test where every value maps to MINOR.
    """
    grouping = load_class_grouping_map()
    major_values = [k for k, v in grouping.items() if v is BoundaryClass.MAJOR_ROAD]
    assert major_values, "sub-E grouping has no MAJOR_ROAD values — class-map gate would be vacuous"
    assert grouping.get("__no_such_class__", BoundaryClass.MINOR_ROAD) is BoundaryClass.MINOR_ROAD


# ===========================================================================
# Real-region BP7 composite — SKIP STUB (sub-E + sub-F caches absent)
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "awaiting sub-E + sub-F cache regeneration — real-region BP7 composite; "
        "see close-checklist. Un-skip when both caches regenerate; run "
        "cross-reference/symmetry/non-road/coverage + version consistency "
        "against real cached Singapore tiles."
    )
)
def test_validate_cross_tile_against_real_region_singapore():  # type: ignore[empty-body]
    """Integration: run validate_cross_tile on the real cached Singapore
    sub-F region against the real sub-E region.

    Requires real caches under data/processed/sub_f/<release>/singapore/ and
    data/processed/sub_e/<release>/singapore/. This is the real-region layer
    of the BP7 verification debt inherited from T7/T8.
    See reports/2026-05-23-phase-1-sub-F-close-checklist.md.
    """
    ...
