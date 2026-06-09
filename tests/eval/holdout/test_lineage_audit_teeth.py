"""Leak-guard 4-case gating teeth (A/B/C/D, spec §6).

HALT-gate for the city-identity holdout-leak guard. Each case is red-before/
green-after and proven NON-VACUOUS. A failing tooth means the guard (T7) is
wrong, which is a HALT condition the controller escalates - do NOT weaken a
tooth to make it pass.

Spec §6 HALT meanings:
- B fails -> the city-guard is vacuous (it doesn't catch un-enumerated
  held-out-city tiles). STOP and report.
- D fails -> the guard over-trips partial holdouts (wrongly scoped to all
  regions). STOP and report.
"""

from __future__ import annotations

import pytest

from cfm.eval.holdout.lineage_audit import (
    Artifact,
    HoldoutLeakError,
    _holdout_tile_refs,
    _whole_city_regions,
    audit_no_holdout_leak,
)

# whole-city manifest enumerating (0,0) and (0,1) for krakow
WC = {
    "regions": {
        "krakow": {
            "holdout_kind": "whole_city",
            "tiles": [{"tile_i": 0, "tile_j": 0}, {"tile_i": 0, "tile_j": 1}],
        }
    }
}


def test_A_enumerated_leak_trips():
    # tile-key: an ENUMERATED held-out tile in training lineage trips
    art = Artifact("train/a", frozenset({("krakow", 0, 0)}))
    assert ("krakow", 0, 0) in _holdout_tile_refs(WC)  # precondition: it IS enumerated
    with pytest.raises(HoldoutLeakError):
        audit_no_holdout_leak(WC, [art])


def test_B_unenumerated_holdout_city_tile_trips():
    # city-guard NON-REDUNDANT: an UN-enumerated held-out-city tile trips the city-guard
    art = Artifact("train/b", frozenset({("krakow", 7, 7)}))  # NOT in WC enumerated tiles
    # NON-VACUITY: the tile-key alone MISSES this (not in the enumerated holdout set) ->
    # if this raised WITHOUT the city-guard, the guard would be vacuous. It must be the city-guard.
    assert ("krakow", 7, 7) not in _holdout_tile_refs(WC)
    with pytest.raises(HoldoutLeakError, match="city-guard"):
        audit_no_holdout_leak(WC, [art])


def test_C_clean_passes():
    # a train-only city tile -> no leak -> returns None (no raise)
    art = Artifact("train/c", frozenset({("hamburg", 1, 1)}))
    assert audit_no_holdout_leak(WC, [art]) is None


def test_D_partial_holdout_does_NOT_overtrip():
    # FORWARD-PROTECTION: a synthetic tile_sample (partial) holdout must NOT over-trip on its
    # OWN train-side tiles. v1 has no partial holdout, so this guards a config that does not
    # exist yet.
    ts = {
        "regions": {
            "singapore": {
                "holdout_kind": "tile_sample",
                "tiles": [{"tile_i": 5, "tile_j": 5}],
            }
        }
    }
    train_sg = Artifact("train/sg", frozenset({("singapore", 1, 1)}))  # a singapore TRAIN-side tile
    # NON-VACUITY: singapore is NOT a whole_city region, so the city-guard must NOT be in scope ->
    # if the guard fired here it would be wrongly scoped to all regions.
    assert "singapore" not in _whole_city_regions(ts)
    assert ("singapore", 1, 1) not in _holdout_tile_refs(ts)  # also not the enumerated (5,5)
    assert audit_no_holdout_leak(ts, [train_sg]) is None  # MUST NOT raise
