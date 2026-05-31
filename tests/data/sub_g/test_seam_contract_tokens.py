from __future__ import annotations

from shapely.geometry import LineString
from shapely.wkb import dumps as wkb_dumps

from cfm.data.sub_e.rotation import EdgeKind
from cfm.data.sub_f.rotation import cell_edge_directions
from cfm.data.sub_g.readers import SubEContractRow
from cfm.data.sub_g.seam_contract_tokens import (
    bref_id_to_dir_class,
    build_cell_contracts,
    check_cell_bijection,
    parse_actual_brefs_per_cell,
    predict_expected_brefs_per_cell,
)


def test_bref_id_mapping_matches_locked_vocab():
    # boundary_reference_vocab.yaml:29-68
    assert bref_id_to_dir_class(1500) == ("N", "MAJOR_ROAD")
    assert bref_id_to_dir_class(1503) == ("W", "MAJOR_ROAD")
    assert bref_id_to_dir_class(1504) == ("N", "MINOR_ROAD")
    assert bref_id_to_dir_class(1507) == ("W", "MINOR_ROAD")


def test_parse_actual_brefs_splits_features_and_collects_brefs():
    # Case C (inbound N_MAJOR) feature, then a Case A feature with no bref.
    seq = [
        509,
        7,
        1500,
        300,
        301,
        302,
        303,
        510,  # inbound N_MAJOR
        509,
        7,
        300,
        301,
        302,
        303,
        510,  # no bref
    ]
    assert parse_actual_brefs_per_cell(seq) == [("N", "MAJOR_ROAD")]


def test_predict_expects_bref_for_road_endpoint_on_major_edge():
    # A road LineString whose end vertex sits on the E edge (x=250) of a cell whose
    # contract marks E = MAJOR_ROAD -> expect one (E, MAJOR_ROAD) bref.
    road = wkb_dumps(LineString([(100.0, 100.0), (250.0, 100.0)]), byte_order=1)
    features = [{"feature_class": 0, "geometry": road, "source_feature_id": "r"}]
    cell_contract = {"N": "NONE", "E": "MAJOR_ROAD", "S": "NONE", "W": "NONE"}
    assert predict_expected_brefs_per_cell(features, cell_contract) == [("E", "MAJOR_ROAD")]


def test_predict_ignores_non_road_and_interior_endpoints():
    building = wkb_dumps(LineString([(10.0, 10.0), (20.0, 20.0)]), byte_order=1)
    features = [{"feature_class": 1, "geometry": building, "source_feature_id": "b"}]
    cell_contract = {"N": "MAJOR_ROAD", "E": "MAJOR_ROAD", "S": "MAJOR_ROAD", "W": "MAJOR_ROAD"}
    assert predict_expected_brefs_per_cell(features, cell_contract) == []


def test_check_cell_bijection_detects_missing_and_extra():
    # expected has (E, MAJOR); actual emitted nothing -> "missing" diagnostic.
    diags = check_cell_bijection("tile=i0_j0", (0, 0), [("E", "MAJOR_ROAD")], [])
    assert len(diags) == 1
    assert diags[0].invariant_name == "bref_bijection_contract_vs_tokens"
    assert "missing" in diags[0].signature

    # actual emitted (N, MINOR) that wasn't expected -> "extra".
    diags2 = check_cell_bijection("tile=i0_j0", (0, 0), [], [("N", "MINOR_ROAD")])
    assert "extra" in diags2[0].signature

    # match -> no diagnostic.
    assert (
        check_cell_bijection("tile=i0_j0", (0, 0), [("E", "MAJOR_ROAD")], [("E", "MAJOR_ROAD")])
        == []
    )


def test_build_cell_contracts_joins_sub_e_rows_via_lattice():
    # Use the real lattice helper to get cell (0,0)'s four edge ids, then build
    # sub-E rows marking the N edge MAJOR (enum 2) and the others NONE (enum 1).
    edges = cell_edge_directions(0, 0)
    rows: list[SubEContractRow] = []
    for direction, (li, lj, axis, kind) in edges.items():
        sk = 1 if kind is EdgeKind.INTERNAL else 2
        enum = 2 if direction == "N" else 1  # N=MAJOR_ROAD, others NONE
        rows.append(
            SubEContractRow(
                slot_kind=sk,
                slot_index=0,
                lower_cell_i=int(li),
                lower_cell_j=int(lj),
                axis=int(axis),
                scope_marker=0,  # active
                boundary_class_enum=enum,
            )
        )
    contracts = build_cell_contracts(rows)
    assert contracts[(0, 0)]["N"] == "MAJOR_ROAD"
    assert contracts[(0, 0)]["E"] == "NONE"


def test_endpoint_edge_maps_coordinates_to_directions():
    from cfm.data.sub_g.seam_contract_tokens import _endpoint_edge

    assert _endpoint_edge(250.0, 100.0) == "E"
    assert _endpoint_edge(0.0, 100.0) == "W"
    assert _endpoint_edge(100.0, 0.0) == "S"
    assert _endpoint_edge(100.0, 250.0) == "N"
    assert _endpoint_edge(100.0, 100.0) is None
