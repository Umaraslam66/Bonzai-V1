from __future__ import annotations

import pytest

from cfm.eval.holdout.lineage_audit import (
    Artifact,
    HoldoutLeakError,
    _holdout_tile_refs,
    audit_no_holdout_leak,
)

# whole-city manifest enumerating ONLY (0,0) for krakow
MANIFEST = {
    "regions": {
        "krakow": {
            "holdout_kind": "whole_city",
            "tiles": [{"tile_i": 0, "tile_j": 0}],
        }
    }
}


def test_cityguard_trips_on_unenumerated_holdout_city_tile():
    art = Artifact(path="train/x", lineage=frozenset({("krakow", 9, 9)}))  # NOT enumerated
    # NON-VACUITY: the existing tile-key would MISS this (it's not in the enumerated holdout set)
    assert ("krakow", 9, 9) not in _holdout_tile_refs(MANIFEST)
    # so ONLY the city-guard can catch it -> it must trip, and the message must name the city-guard
    with pytest.raises(HoldoutLeakError, match="city-guard"):
        audit_no_holdout_leak(MANIFEST, [art])
