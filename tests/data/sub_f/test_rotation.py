"""Task 7 tests for BP7 boundary references and sub-E rotation wrapping."""

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import pytest
import yaml
from shapely.geometry import LineString, MultiLineString

from cfm.data.sub_e.derivation import (
    _HIERARCHY,
    BoundaryClass,
    derive_boundary_class,
    load_class_grouping_map,
)
from cfm.data.sub_e.rotation import cell_to_edge_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs" / "sub_f"
SUB_C_SINGAPORE_ROOT = (
    REPO_ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
)
HALT_REPORT_PATH = REPO_ROOT / "reports" / "2026-05-23-phase-1-sub-F-task-7-halt.md"
FEATURE_SPLITTING_REPORT_PATH = REPO_ROOT / "reports" / "sub_f_task_7_feature_splitting.yaml"
FEATURE_SPLITTING_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "sub_f" / "verify_sub_c_feature_splitting.py"
)

EXPECTED_BP7_TAGS: Final[tuple[str, ...]] = (
    "<bref_N_MAJOR>",
    "<bref_E_MAJOR>",
    "<bref_S_MAJOR>",
    "<bref_W_MAJOR>",
    "<bref_N_MINOR>",
    "<bref_E_MINOR>",
    "<bref_S_MINOR>",
    "<bref_W_MINOR>",
)
EXPECTED_HIGHWAY_VALUES: Final[tuple[str, ...]] = (
    "*",
    "bridleway",
    "busway",
    "cycleway",
    "footway",
    "living_street",
    "motorway",
    "motorway_link",
    "path",
    "pedestrian",
    "primary",
    "primary_link",
    "residential",
    "road",
    "secondary",
    "secondary_link",
    "service",
    "steps",
    "subway",
    "tertiary",
    "tertiary_link",
    "track",
    "trunk",
    "trunk_link",
    "unclassified",
)
EXPECTED_MAJOR_VALUES: Final[set[str]] = {"primary", "trunk", "secondary"}
EXPECTED_MINOR_VALUES: Final[set[str]] = {
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "footway",
    "steps",
    "cycleway",
}
EXPECTED_MISSING_FROM_SUB_E_GROUPING: Final[tuple[str, ...]] = (
    "*",
    "bridleway",
    "busway",
    "living_street",
    "motorway",
    "motorway_link",
    "path",
    "pedestrian",
    "primary_link",
    "road",
    "secondary_link",
    "subway",
    "tertiary_link",
    "track",
    "trunk_link",
)

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_feature_splitting_module():
    spec = importlib.util.spec_from_file_location(
        "verify_sub_c_feature_splitting",
        FEATURE_SPLITTING_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def boundary_reference_vocab() -> dict:
    return _load_yaml(CONFIG_ROOT / "boundary_reference_vocab.yaml")


@pytest.fixture(scope="module")
def singapore_missing_highway_evidence() -> dict[str, dict[str, int | None]]:
    row_counts: Counter[str] = Counter()
    source_tiles: dict[str, defaultdict[str, set[str]]] = {
        value: defaultdict(set) for value in EXPECTED_MISSING_FROM_SUB_E_GROUPING
    }

    for path in sorted(SUB_C_SINGAPORE_ROOT.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read(
            columns=["feature_class", "class_raw", "source_feature_id"]
        )
        tile = path.parent.name
        for row in table.to_pylist():
            value = row["class_raw"]
            if row["feature_class"] != 0 or value not in source_tiles:
                continue
            row_counts[value] += 1
            source_id = row.get("source_feature_id")
            if source_id:
                source_tiles[value][source_id].add(tile)

    return {
        value: {
            "singapore_row_count": row_counts[value],
            "multi_tile_source_feature_count": sum(
                1 for tiles in source_tiles[value].values() if len(tiles) > 1
            ),
        }
        for value in EXPECTED_MISSING_FROM_SUB_E_GROUPING
    }


def test_direction_order_is_boundary_vocab_order():
    from cfm.data.sub_f.rotation import DIRECTION_ORDER

    assert DIRECTION_ORDER == ("N", "E", "S", "W")


def test_cell_edge_directions_returns_exact_direction_keys():
    from cfm.data.sub_f.rotation import cell_edge_directions

    assert tuple(cell_edge_directions(3, 5)) == ("N", "E", "S", "W")


def test_cell_edge_directions_maps_sub_e_fields_explicitly():
    from cfm.data.sub_f.rotation import cell_edge_directions

    edge_ids = cell_to_edge_ids(3, 5)
    actual = cell_edge_directions(3, 5)

    assert actual == {
        "N": edge_ids.north,
        "E": edge_ids.east,
        "S": edge_ids.south,
        "W": edge_ids.west,
    }


def test_boundary_reference_vocab_locks_expected_block_and_slots(
    boundary_reference_vocab: dict,
):
    assert boundary_reference_vocab["_status"] == "LOCKED"
    assert boundary_reference_vocab["id_block"] == {
        "start_id": 1500,
        "end_id": 1599,
        "used_count": 8,
        "reserved_count": 92,
        "status": "LOCKED at Halt 7 approval",
    }

    slots = boundary_reference_vocab["slots"]
    assert len(slots) == 8
    assert [slot["id"] for slot in slots] == list(range(1500, 1508))
    assert [slot["local_id"] for slot in slots] == list(range(8))
    assert [slot["tag"] for slot in slots] == list(EXPECTED_BP7_TAGS)


def test_boundary_reference_vocab_has_exact_direction_class_cross_product(
    boundary_reference_vocab: dict,
):
    actual = {
        (slot["direction"], slot["boundary_class"])
        for slot in boundary_reference_vocab["slots"]
    }
    expected = {
        (direction, boundary_class)
        for boundary_class in ("MAJOR_ROAD", "MINOR_ROAD")
        for direction in ("N", "E", "S", "W")
    }

    assert boundary_reference_vocab["direction_order"] == ["N", "E", "S", "W"]
    assert boundary_reference_vocab["class_set"] == ["MAJOR_ROAD", "MINOR_ROAD"]
    assert actual == expected


def test_boundary_class_enum_values_match_sub_e_contract():
    assert [(c.name, int(c), c.value) for c in BoundaryClass] == [
        ("BOUNDARY_NOT_APPLICABLE", 0, 0),
        ("NONE", 1, 1),
        ("MAJOR_ROAD", 2, 2),
        ("MINOR_ROAD", 3, 3),
    ]


def test_boundary_class_hierarchy_matches_sub_e_contract():
    assert [c.name for c in _HIERARCHY] == ["MAJOR_ROAD", "MINOR_ROAD", "NONE"]


def test_bp1_to_sub_e_class_mapping_matches_hand_expected_sets():
    grouping = load_class_grouping_map()

    assert {
        key for key, value in grouping.items() if value is BoundaryClass.MAJOR_ROAD
    } == EXPECTED_MAJOR_VALUES
    assert {
        key for key, value in grouping.items() if value is BoundaryClass.MINOR_ROAD
    } == EXPECTED_MINOR_VALUES


def test_sub_f_consumes_sub_e_boundary_class_without_local_override():
    semantic_vocab = _load_yaml(CONFIG_ROOT / "semantic_vocab.yaml")
    actual_highways = tuple(
        sorted(
            slot["tag"].split("=", 1)[1]
            for slot in semantic_vocab["slots"]
            if slot["tag"].startswith("highway=")
        )
    )
    grouping = load_class_grouping_map()
    missing_from_grouping = tuple(
        value for value in actual_highways if value not in grouping
    )

    assert actual_highways == EXPECTED_HIGHWAY_VALUES
    assert missing_from_grouping == EXPECTED_MISSING_FROM_SUB_E_GROUPING

    # Architecture (b): sub-F tokenizes sub-E's boundary_contract parquet.
    # Present-but-unmapped class_raw values fall to sub-E's MINOR_ROAD default;
    # NONE is reserved for edges with no road crossings.
    for value in EXPECTED_MISSING_FROM_SUB_E_GROUPING:
        assert derive_boundary_class([value]) is BoundaryClass.MINOR_ROAD
    assert derive_boundary_class([]) is BoundaryClass.NONE


def test_locked_highway_coverage_diagnostic_is_surfaced_in_halt_report(
    singapore_missing_highway_evidence: dict[str, dict[str, int | None]],
):
    semantic_vocab = _load_yaml(CONFIG_ROOT / "semantic_vocab.yaml")
    actual_highways = tuple(
        sorted(
            slot["tag"].split("=", 1)[1]
            for slot in semantic_vocab["slots"]
            if slot["tag"].startswith("highway=")
        )
    )
    grouping = load_class_grouping_map()
    actual_missing = tuple(value for value in EXPECTED_HIGHWAY_VALUES if value not in grouping)
    report = HALT_REPORT_PATH.read_text(encoding="utf-8")

    assert actual_highways == EXPECTED_HIGHWAY_VALUES
    assert actual_missing == EXPECTED_MISSING_FROM_SUB_E_GROUPING
    assert "REAL §9.6.1 cascade #9 against upstream composition" in report
    assert "motorway" in report
    assert "motorway_link" in report

    for value, evidence in singapore_missing_highway_evidence.items():
        assert f"`{value}`" in report
        assert f"singapore_row_count={evidence['singapore_row_count']}" in report
        assert (
            "multi_tile_source_feature_count="
            f"{evidence['multi_tile_source_feature_count']}"
        ) in report


def test_multiline_part_edge_bucket_detects_same_cell_edge_multi_part():
    module = _load_feature_splitting_module()
    geom = MultiLineString(
        [
            LineString([(0.0, 20.0), (40.0, 30.0)]),
            LineString([(0.0, 80.0), (40.0, 90.0)]),
        ]
    )

    result = module.classify_multiline_part_edge_relationship(geom)

    assert result["bucket"] == "same_cell_edge_multi_part"
    assert result["repeated_edges"] == ["W"]


def test_multiline_part_edge_bucket_detects_different_cell_edges():
    module = _load_feature_splitting_module()
    geom = MultiLineString(
        [
            LineString([(0.0, 20.0), (40.0, 30.0)]),
            LineString([(200.0, 250.0), (210.0, 220.0)]),
        ]
    )

    result = module.classify_multiline_part_edge_relationship(geom)

    assert result["bucket"] == "different_cell_edges"
    assert result["repeated_edges"] == []


def test_multiline_part_edge_bucket_detects_no_boundary_interaction():
    module = _load_feature_splitting_module()
    geom = MultiLineString(
        [
            LineString([(20.0, 20.0), (40.0, 30.0)]),
            LineString([(100.0, 80.0), (140.0, 90.0)]),
        ]
    )

    result = module.classify_multiline_part_edge_relationship(geom)

    assert result["bucket"] == "no_multi_part_boundary_interaction"
    assert result["repeated_edges"] == []


def test_multiline_part_edge_bucket_detects_mergeable_artifact():
    module = _load_feature_splitting_module()
    geom = MultiLineString(
        [
            LineString([(20.0, 20.0), (40.0, 40.0)]),
            LineString([(40.0, 40.0), (80.0, 80.0)]),
        ]
    )

    result = module.classify_multiline_part_edge_relationship(geom)

    assert result["bucket"] == "mergeable_artifact"
    assert result["repeated_edges"] == []


def test_sentinel_inventory_locks_bp7_block():
    data = _load_yaml(CONFIG_ROOT / "sentinel_inventory.yaml")
    bp7 = data["bp7_boundary_ref"]

    assert data["_status"] == "LOCKED"
    assert "bp7_boundary_ref_placeholder" not in data
    assert bp7["start_id"] == 1500
    assert bp7["end_id"] == 1599
    assert bp7["placeholder"] is False
    assert bp7["used_count"] == 8
    assert bp7["reserved_count"] == 92
    assert bp7["status"] == "LOCKED at Halt 7 approval"


def test_feature_splitting_report_has_allowed_halt_outcome():
    data = _load_yaml(FEATURE_SPLITTING_REPORT_PATH)

    assert data["_status"] == "LOCKED - Halt 7 approved"
    assert data["tile_count"] > 1
    assert data["row_count"] > 0
    assert data["outcome"] in {"single_row_per_branch", "branched_multi_row_present"}
    if data["road_multiline_count"] == 0:
        assert data["outcome"] == "single_row_per_branch"
    else:
        assert data["outcome"] == "branched_multi_row_present"
    assert set(data["road_multiline_part_edge_buckets"]) <= {
        "different_cell_edges",
        "mergeable_artifact",
        "no_multi_part_boundary_interaction",
        "same_cell_edge_multi_part",
    }
    assert (
        sum(data["road_multiline_part_edge_buckets"].values())
        == data["road_multiline_count"]
    )
