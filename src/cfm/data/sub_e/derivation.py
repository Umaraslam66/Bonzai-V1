"""Boundary-class derivation function.

Maps `class_raw` strings (raw Overture transportation.class values from sub-C
`features.parquet`) to `BoundaryClass` enum via the class-grouping map and
hierarchy-wins multi-crossing tie-break. Locked under
boundary_derivation_version 1.0 (see boundary_vocab.yaml).
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml


class BoundaryClass(IntEnum):
    BOUNDARY_NOT_APPLICABLE = 0  # sentinel; dataloader-side only, never on-disk
    NONE = 1
    MAJOR_ROAD = 2
    MINOR_ROAD = 3


# Hierarchy order for multi-crossing tie-break: highest precedence first.
_HIERARCHY: Final[tuple[BoundaryClass, ...]] = (
    BoundaryClass.MAJOR_ROAD,
    BoundaryClass.MINOR_ROAD,
    BoundaryClass.NONE,
)

_VOCAB_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "configs" / "macro_plan" / "v1" / "boundary_vocab.yaml"
)


@lru_cache(maxsize=1)
def load_class_grouping_map() -> dict[str, BoundaryClass]:
    """Load the class_raw → BoundaryClass mapping from boundary_vocab.yaml.

    Cached: the vocab is locked and won't change within a process.
    """
    data = yaml.safe_load(_VOCAB_PATH.read_text())
    raw_map = data["class_grouping_map"]
    out: dict[str, BoundaryClass] = {}
    for class_raw in raw_map["MAJOR_ROAD"]:
        out[class_raw] = BoundaryClass.MAJOR_ROAD
    for class_raw in raw_map["MINOR_ROAD"]:
        out[class_raw] = BoundaryClass.MINOR_ROAD
    return out


def derive_boundary_class(
    class_raws: list[str | None],
) -> BoundaryClass:
    """Derive the BoundaryClass for one active internal edge.

    Args:
        class_raws: list of raw Overture transportation.class strings for
            every road crossing on this edge. May be empty. Null/unknown
            values fall through to the MINOR_ROAD default bucket.

    Returns:
        BoundaryClass.NONE if `class_raws` is empty (no road crossings).
        Otherwise the highest-precedence class per the hierarchy:
        MAJOR_ROAD > MINOR_ROAD > NONE.
    """
    if not class_raws:
        return BoundaryClass.NONE

    grouping = load_class_grouping_map()
    seen: set[BoundaryClass] = set()
    for cr in class_raws:
        # Null / unknown class_raw → default bucket MINOR_ROAD (never MAJOR).
        seen.add(grouping.get(cr, BoundaryClass.MINOR_ROAD))

    for cls in _HIERARCHY:
        if cls in seen:
            return cls
    # Unreachable: seen is non-empty if class_raws is non-empty, and every
    # element maps into the hierarchy.
    raise AssertionError(f"derivation reached unreachable branch: {class_raws!r}")
