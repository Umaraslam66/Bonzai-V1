"""Sub-C tile extraction pipeline.

See docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md
"""

from cfm.data.sub_c.enums import (
    AXIS,
    COASTAL_RIVER,
    EVENT_TYPE,
    FEATURE_CLASS,
    GEOMETRY_TYPE,
    decode_enum,
    encode_enum,
)
from cfm.data.sub_c.epsilon import EPS_AREA_M2, EPS_COORD_M, EPS_LENGTH_M, EPS_RATIO
from cfm.data.sub_c.errors import PolicyError, TileValidationError
from cfm.data.sub_c.validator_cross_tile import validate_extraction_cross_tile

__all__ = [
    "AXIS",
    "COASTAL_RIVER",
    "EPS_AREA_M2",
    "EPS_COORD_M",
    "EPS_LENGTH_M",
    "EPS_RATIO",
    "EVENT_TYPE",
    "FEATURE_CLASS",
    "GEOMETRY_TYPE",
    "PolicyError",
    "TileValidationError",
    "decode_enum",
    "encode_enum",
    "validate_extraction_cross_tile",
]
