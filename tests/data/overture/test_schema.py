from __future__ import annotations

import pyarrow as pa
import pytest

from cfm.data.overture.errors import OvertureSchemaMismatch
from cfm.data.overture.schema import (
    EXPECTED_THEMES,
    THEME_SCHEMAS,
    validate_schema,
)


def test_expected_themes_are_the_five_we_use() -> None:
    assert EXPECTED_THEMES == ("buildings", "places", "transportation", "base", "divisions")


def test_every_theme_has_a_schema() -> None:
    for theme in EXPECTED_THEMES:
        assert theme in THEME_SCHEMAS
        assert isinstance(THEME_SCHEMAS[theme], pa.Schema)


def test_buildings_schema_includes_geometry_and_class() -> None:
    s = THEME_SCHEMAS["buildings"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "class" in s.names


def test_places_schema_includes_categories() -> None:
    s = THEME_SCHEMAS["places"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "categories" in s.names


def test_transportation_schema_includes_class() -> None:
    s = THEME_SCHEMAS["transportation"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "class" in s.names


def test_base_schema_includes_subtype() -> None:
    s = THEME_SCHEMAS["base"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "subtype" in s.names


def test_divisions_schema_includes_country() -> None:
    s = THEME_SCHEMAS["divisions"]
    assert "id" in s.names
    assert "geometry" in s.names
    assert "country" in s.names


def test_validate_schema_accepts_exact_match() -> None:
    table = pa.table(
        {
            "id": pa.array(["a", "b"], type=pa.string()),
            "geometry": pa.array([b"g1", b"g2"], type=pa.binary()),
            "class": pa.array(["residential", "office"], type=pa.string()),
            "height": pa.array([10.0, 20.0], type=pa.float64()),
            "num_floors": pa.array([3, 5], type=pa.int32()),
        }
    )
    # Should not raise.
    validate_schema(table, theme="buildings")


def test_validate_schema_accepts_extra_columns() -> None:
    # Real Overture parquet may have many more columns than we use. We only
    # require the curated set; extras are tolerated.
    table = pa.table(
        {
            "id": pa.array(["a"], type=pa.string()),
            "geometry": pa.array([b"g1"], type=pa.binary()),
            "class": pa.array(["residential"], type=pa.string()),
            "height": pa.array([10.0], type=pa.float64()),
            "num_floors": pa.array([3], type=pa.int32()),
            "extra_column_we_dont_use": pa.array([42], type=pa.int64()),
        }
    )
    validate_schema(table, theme="buildings")


def test_validate_schema_rejects_missing_required_column() -> None:
    table = pa.table(
        {
            "id": pa.array(["a"], type=pa.string()),
            "geometry": pa.array([b"g1"], type=pa.binary()),
            # "class" missing
            "height": pa.array([10.0], type=pa.float64()),
            "num_floors": pa.array([3], type=pa.int32()),
        }
    )
    with pytest.raises(OvertureSchemaMismatch) as exc_info:
        validate_schema(table, theme="buildings")
    assert "class" in str(exc_info.value)


def test_validate_schema_rejects_unknown_theme() -> None:
    table = pa.table({"id": pa.array(["a"], type=pa.string())})
    with pytest.raises(OvertureSchemaMismatch) as exc_info:
        validate_schema(table, theme="not_a_theme")
    assert "not_a_theme" in str(exc_info.value)
