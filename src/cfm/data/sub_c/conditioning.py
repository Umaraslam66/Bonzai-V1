"""Per-tile conditioning vector derivation (spec §11.9).

Computes the 7-field conditioning vector for a tile:
- admin_region: second-level admin division name
- morphology_class: urban morphology type (per-tile from day one)
- era_class: historical era (per-tile from day one)
- coastal_inland_river: int8 enum (0=inland, 1=coastal, 2=riverside, 3=coastal_riverside)
- population_density_bucket: null at sub-C (owned by sub-D)
- population_density_bucket_owner: "sub-D" marker

Derivation rule for coastal_inland_river (spec §11.9):
- coastal iff sea_water_fraction > 0 for any cell in tile
- riverside iff Σ length(features WHERE class IN {river, stream}) >= 500 m
  (strict β threshold per §4.3, no EPSILON)
- coastal_riverside iff both
- inland otherwise
"""

from __future__ import annotations

from cfm.data.sub_c.enums import COASTAL_RIVER, encode_enum


def compute_conditioning_per_tile(
    *,
    cell_sea_water_fractions: list[float],
    river_stream_lengths_m: list[float],
    admin_region: str | None,
    morphology_class: str = "Asian-megacity",
    era_class: str = "contemporary",
) -> dict:
    """Compute the per-tile conditioning vector per spec §11.9.

    Args:
        cell_sea_water_fractions: List of sea_water_fraction values from kept cells in tile.
            Values in [0, 1]. Empty list if no kept cells.
        river_stream_lengths_m: List of river/stream feature lengths (in meters) in tile.
            Represents sum of length(features WHERE class IN {river, stream}) per cell,
            aggregated to tile level. Empty list if no such features.
        admin_region: Second-level admin division name (e.g., "Central Region"), or None
            if the tile centroid falls outside all known region polygons (e.g. maritime
            tiles) or if the divisions theme is absent.
        morphology_class: Urban morphology type; defaults to "Asian-megacity".
        era_class: Historical era; defaults to "contemporary".

    Returns:
        Dict with keys:
        - admin_region: passed through unchanged
        - morphology_class: passed through or defaulted
        - era_class: passed through or defaulted
        - coastal_inland_river: int8 enum code (0/1/2/3)
        - population_density_bucket: None (owned by sub-D)
        - population_density_bucket_owner: "sub-D"
    """

    # Determine coastal condition: any cell with sea_water_fraction > 0
    is_coastal = any(frac > 0.0 for frac in cell_sea_water_fractions)

    # Determine riverside condition: total river/stream length >= 500m
    # (strict β threshold per §4.3, no EPSILON)
    total_river_stream_length_m = sum(river_stream_lengths_m)
    is_riverside = total_river_stream_length_m >= 500.0

    # Map conditions to enum code per spec §11.9
    if is_coastal and is_riverside:
        coastal_inland_river_code = encode_enum(COASTAL_RIVER, "coastal_riverside")
    elif is_coastal:
        coastal_inland_river_code = encode_enum(COASTAL_RIVER, "coastal")
    elif is_riverside:
        coastal_inland_river_code = encode_enum(COASTAL_RIVER, "riverside")
    else:
        coastal_inland_river_code = encode_enum(COASTAL_RIVER, "inland")

    return {
        "admin_region": admin_region,
        "morphology_class": morphology_class,
        "era_class": era_class,
        "coastal_inland_river": coastal_inland_river_code,
        "population_density_bucket": None,
        "population_density_bucket_owner": "sub-D",
    }
