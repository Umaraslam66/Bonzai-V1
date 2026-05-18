"""int8 enum mappings for sub-C output columns.

Per spec §14.3: small enums are int8 (not strings) for byte-determinism
across PyArrow versions. Open string columns (class_raw, subtype_raw,
categories_primary, categories_alternate, admin_region, morphology_class,
era_class) stay as strings — Overture-driven unbounded domain.

Adding a value to a closed enum = append-only-within-phase; triggers
sub_c_schema_version bump per spec §14.9.
"""

from __future__ import annotations

GEOMETRY_TYPE: dict[int, str] = {0: "Point", 1: "LineString", 2: "Polygon"}
FEATURE_CLASS: dict[int, str] = {0: "road", 1: "building", 2: "poi", 3: "base"}
AXIS: dict[int, str] = {0: "x", 1: "y"}
EVENT_TYPE: dict[int, str] = {0: "enter", 1: "exit", 2: "interval"}
COASTAL_RIVER: dict[int, str] = {
    0: "inland",
    1: "coastal",
    2: "riverside",
    3: "coastal_riverside",
}


def encode_enum(mapping: dict[int, str], label: str) -> int:
    """Reverse lookup: label → int8 code. Raises KeyError if label unknown."""
    for code, value in mapping.items():
        if value == label:
            return code
    raise KeyError(f"label {label!r} not in enum mapping {mapping!r}")


def decode_enum(mapping: dict[int, str], code: int) -> str:
    """Forward lookup: int8 code → label. Raises KeyError if code unknown."""
    return mapping[code]
