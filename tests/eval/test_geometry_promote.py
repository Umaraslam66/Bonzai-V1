"""Building-ring promotion + §9 regime-distinguishing guard (bake-off Task 1.5)."""

from __future__ import annotations

from cfm.eval.emergence import building_token_ids
from cfm.eval.geometry import promote_building_rings

_FEATURE = 509  # <feature> marker
_BUILDING = min(building_token_ids())  # a real building-class token id
_ROAD = 511  # a non-building token id (direction range; NOT in building_token_ids)

_CLOSED_RING = {"type": "LineString", "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]}
_OPEN_LINE = {"type": "LineString", "coordinates": [[0, 0], [3, 0], [3, 4]]}


def test_building_closed_ring_is_promoted_to_polygon() -> None:
    # MUST-PROMOTE: a building-class block whose ring closes becomes a Polygon.
    out = promote_building_rings([[_FEATURE, _BUILDING, 1, 2]], [dict(_CLOSED_RING)])
    assert out[0]["type"] == "Polygon"
    assert out[0]["coordinates"] == [_CLOSED_RING["coordinates"]]


def test_closed_road_ring_is_NOT_promoted() -> None:
    # MUST-NOT-PROMOTE (§9 regime-distinguishing twin): an identical closed ring on a
    # ROAD block (a roundabout) stays LineString -- proving the keying is feature-class,
    # not bare ring-closure. This is the case bare-closure promotion would get wrong.
    assert _ROAD not in building_token_ids()
    out = promote_building_rings([[_FEATURE, _ROAD, 1, 2]], [dict(_CLOSED_RING)])
    assert out[0]["type"] == "LineString"


def test_building_open_line_is_not_promoted() -> None:
    # a building-class block whose ring is NOT closed stays LineString (no false polygon)
    out = promote_building_rings([[_FEATURE, _BUILDING, 1, 2]], [dict(_OPEN_LINE)])
    assert out[0]["type"] == "LineString"


def test_non_linestring_passes_through_unchanged() -> None:
    pt = {"type": "Point", "coordinates": [1, 2]}
    poly = {"type": "Polygon", "coordinates": [_CLOSED_RING["coordinates"]]}
    out = promote_building_rings([[_FEATURE, _BUILDING], [_FEATURE, _BUILDING]], [pt, poly])
    assert out[0]["type"] == "Point" and out[1]["type"] == "Polygon"


def test_uses_the_task1_building_authority_by_identity() -> None:
    # one definition of "is a building": the promotion reuses Task 1's building_token_ids
    import cfm.eval.geometry as g

    assert g.building_token_ids is building_token_ids


def test_slice_metrics_uses_the_one_promotion_authority_by_identity() -> None:
    # one-source: slice_metrics promotes via the SAME helper (no second implementation to drift)
    import cfm.eval.geometry as g
    import cfm.eval.slice_metrics as sm

    assert sm.promote_building_rings is g.promote_building_rings
