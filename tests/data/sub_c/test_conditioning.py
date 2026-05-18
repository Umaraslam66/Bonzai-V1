"""Per-tile conditioning vector derivation (spec §11.9).

Testing the coastal_inland_river enum derivation rule:
- coastal iff sea_water_fraction > 0 for any cell in tile
- riverside iff Σ length(features WHERE class IN {river, stream}) >= 500 m (strict β threshold)
- coastal_riverside iff both
- inland otherwise

Morphology and era classes default to Singapore-day-one values;
population_density_bucket left null with owner: sub-D.
"""

from __future__ import annotations

from cfm.data.sub_c.conditioning import compute_conditioning_per_tile
from cfm.data.sub_c.enums import COASTAL_RIVER, encode_enum


def test_compute_conditioning_inland_when_no_sea_no_rivers():
    """All inputs zero/empty → code 0 (inland)."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "inland")
    assert result["coastal_inland_river"] == 0


def test_compute_conditioning_coastal_when_any_cell_has_sea():
    """One cell with sea_water_fraction > 0 → code 1 (coastal)."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.5],
        river_stream_lengths_m=[0.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "coastal")
    assert result["coastal_inland_river"] == 1


def test_compute_conditioning_riverside_when_river_length_meets_500m_threshold():
    """River length = 500.0 (exactly at threshold) → code 2 (riverside).

    Strict >= β user-threshold per spec §4.3 + §11.9.
    """
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.0],
        river_stream_lengths_m=[500.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "riverside")
    assert result["coastal_inland_river"] == 2


def test_compute_conditioning_river_length_below_500m_is_inland():
    """River length = 499.99 (below threshold) → code 0 (inland).

    Not riverside; strict >= comparison means 499.99 < 500.0.
    """
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.0],
        river_stream_lengths_m=[499.99],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "inland")
    assert result["coastal_inland_river"] == 0


def test_compute_conditioning_coastal_riverside_when_both_conditions_hold():
    """sea_water_fraction > 0 AND river_stream_length >= 500 → code 3 (coastal_riverside)."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.3],
        river_stream_lengths_m=[600.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "coastal_riverside")
    assert result["coastal_inland_river"] == 3


def test_compute_conditioning_returns_dict_shape_per_spec():
    """Verify returned dict has all required fields per spec §11.9."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
    )
    # Required fields per spec §11.9 meta.yaml schema
    assert "admin_region" in result
    assert "morphology_class" in result
    assert "era_class" in result
    assert "coastal_inland_river" in result
    assert "population_density_bucket" in result
    assert "population_density_bucket_owner" in result


def test_compute_conditioning_admin_region_passthrough():
    """admin_region passed in should appear in output unchanged."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="East Region",
    )
    assert result["admin_region"] == "East Region"


def test_compute_conditioning_morphology_class_defaults_to_asian_megacity():
    """morphology_class defaults to 'Asian-megacity' per spec §11.9."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
    )
    assert result["morphology_class"] == "Asian-megacity"


def test_compute_conditioning_era_class_defaults_to_contemporary():
    """era_class defaults to 'contemporary' per spec §11.9."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
    )
    assert result["era_class"] == "contemporary"


def test_compute_conditioning_population_density_bucket_null_with_owner():
    """population_density_bucket left null per spec §11.9; owner: sub-D."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
    )
    assert result["population_density_bucket"] is None
    assert result["population_density_bucket_owner"] == "sub-D"


def test_compute_conditioning_multiple_river_lengths_sum():
    """Multiple river/stream features sum to >= 500m threshold."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.0],
        river_stream_lengths_m=[250.0, 250.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "riverside")
    assert result["coastal_inland_river"] == 2


def test_compute_conditioning_multiple_cells_any_with_sea():
    """Multiple cells; any single one with sea_water_fraction > 0 → coastal."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.0, 0.0, 0.15, 0.0],
        river_stream_lengths_m=[0.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "coastal")
    assert result["coastal_inland_river"] == 1


def test_compute_conditioning_morphology_class_override():
    """morphology_class parameter overrides default."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
        morphology_class="European-dense-town",
    )
    assert result["morphology_class"] == "European-dense-town"


def test_compute_conditioning_era_class_override():
    """era_class parameter overrides default."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[],
        river_stream_lengths_m=[],
        admin_region="Central Region",
        era_class="colonial",
    )
    assert result["era_class"] == "colonial"


def test_compute_conditioning_zero_sea_fraction_is_not_coastal():
    """sea_water_fraction = 0.0 (strictly) → NOT coastal."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[0.0],
        river_stream_lengths_m=[0.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "inland")
    assert result["coastal_inland_river"] == 0


def test_compute_conditioning_very_small_sea_fraction_is_coastal():
    """sea_water_fraction = 1e-10 > 0 → coastal (strict > 0 comparison)."""
    result = compute_conditioning_per_tile(
        cell_sea_water_fractions=[1e-10],
        river_stream_lengths_m=[0.0],
        admin_region="Central Region",
    )
    assert result["coastal_inland_river"] == encode_enum(COASTAL_RIVER, "coastal")
    assert result["coastal_inland_river"] == 1
