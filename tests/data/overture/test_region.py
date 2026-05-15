from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pytest
from shapely.geometry import Polygon

from cfm.data.overture.region import (
    BboxScope,
    Region,
    RegionGeometry,
    SizeEstimate,
)


def _square_polygon() -> Polygon:
    return Polygon([(103.6, 1.16), (104.05, 1.16), (104.05, 1.48), (103.6, 1.48), (103.6, 1.16)])


def test_bbox_scope_is_frozen() -> None:
    bbox = BboxScope(min_lon=103.6, min_lat=1.16, max_lon=104.05, max_lat=1.48)
    with pytest.raises(FrozenInstanceError):
        bbox.min_lon = 0.0  # type: ignore[misc]


def test_bbox_scope_as_tuple_roundtrips() -> None:
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    assert bbox.as_tuple() == (103.6, 1.16, 104.05, 1.48)


def test_region_geometry_holds_polygon_and_source() -> None:
    poly = _square_polygon()
    geom = RegionGeometry(admin_polygon=poly, source="overture://divisions:country:SG")
    assert geom.admin_polygon is poly
    assert geom.source == "overture://divisions:country:SG"


def test_region_geometry_is_frozen() -> None:
    geom = RegionGeometry(admin_polygon=_square_polygon(), source="x")
    with pytest.raises(FrozenInstanceError):
        geom.source = "y"  # type: ignore[misc]


def test_size_estimate_fields() -> None:
    est = SizeEstimate(rows=12345, bytes=67890)
    assert est.rows == 12345
    assert est.bytes == 67890


def test_size_estimate_is_frozen() -> None:
    est = SizeEstimate(rows=1, bytes=1)
    with pytest.raises(FrozenInstanceError):
        est.rows = 2  # type: ignore[misc]


def test_region_construction(tmp_path: Path) -> None:
    poly = _square_polygon()
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    geometry = RegionGeometry(admin_polygon=poly, source="overture://divisions:country:SG")
    themes = {
        "buildings": pa.table({"id": [1, 2]}),
        "places": pa.table({"id": [3, 4]}),
    }
    region = Region(
        name="singapore",
        release="2026-04-15.0",
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=tmp_path / "manifest.yaml",
    )
    assert region.name == "singapore"
    assert region.release == "2026-04-15.0"
    assert region.themes["buildings"].num_rows == 2
    assert region.admin_polygon is poly
    assert region.bbox == (103.6, 1.16, 104.05, 1.48)


def test_region_is_frozen(tmp_path: Path) -> None:
    poly = _square_polygon()
    bbox = BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
    geometry = RegionGeometry(admin_polygon=poly, source="x")
    region = Region(
        name="singapore",
        release="2026-04-15.0",
        fetch_bbox=bbox,
        geometry=geometry,
        themes={},
        manifest_path=tmp_path / "manifest.yaml",
    )
    with pytest.raises(FrozenInstanceError):
        region.name = "elsewhere"  # type: ignore[misc]


def test_region_docstring_states_handoff_contract() -> None:
    """The Region docstring must tell downstream consumers that themes are
    bbox-filtered and admin_polygon is for their use, not the backend's.
    """
    assert "bbox" in Region.__doc__.lower()
    assert "admin_polygon" in Region.__doc__ or "admin polygon" in Region.__doc__.lower()
