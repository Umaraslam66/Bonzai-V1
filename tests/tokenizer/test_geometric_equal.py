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
        "features": [
            {
                "type": "Feature",
                "properties": {"class": "B_residential"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 0], [20, 20], [0, 20], [0, 0], [20, 0]]],
                },
            }
        ],
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
