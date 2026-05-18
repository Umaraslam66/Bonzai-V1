from __future__ import annotations

import pytest

from cfm.data.sub_c.enums import (
    AXIS,
    COASTAL_RIVER,
    EVENT_TYPE,
    FEATURE_CLASS,
    GEOMETRY_TYPE,
    decode_enum,
    encode_enum,
)
from cfm.data.sub_c.epsilon import (
    EPS_AREA_M2,
    EPS_COORD_M,
    EPS_LENGTH_M,
    EPS_RATIO,
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
