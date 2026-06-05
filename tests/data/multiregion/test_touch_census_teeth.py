"""Teeth proof for the touch-census `anomaly` drop check (PI gate, 2026-06-05).

The sub-F validator v1.2 relax widened the symmetry + coverage legs so a §8.3
road termination is no longer a false positive. Its intended in-corpus twin
(`assert_lossless_clip`) is TESTED-BUT-UNWIRED (no production caller), and the
independent `--source-trace` gate only traces symmetry-DISAGREEMENT edges — so it
sees nothing in the 20 already-passing cities. For those cities the SOLE
corpus-wide drop check is the touch-census `anomaly` column: a road point-crossing
record whose clipped fragment is in NEITHER flanking cell (an orphaned crossing =
the drop signature).

A gate that reports `anomaly == 0` is only meaningful if it CAN report nonzero on
a real orphan. These tests construct synthetic orphans and confirm the REAL
classifier (`classify_point_crossings`, the exact function `touch_census` calls)
counts them — `anomaly == 0` must mean "looked and found none," not "cannot
detect." Mirrors the synthetic-fixture-blind-regime-at-validator discipline.

Red-before evidence (recorded in the close-out): mutating the classifier's
`else: anomaly` branch to `else: touch_as_cross` turns
test_orphan_no_feature_row_is_anomaly and test_three_regimes_distinguished RED;
deleting the `in_a or in_b` arm so neither/one both fall through likewise reds.
Reverting restores green. The gate is therefore non-vacuous.
"""

from __future__ import annotations

from scripts.multiregion.diagnostics.symmetry_probe import (
    NONROAD_CLASS_RAW,
    classify_point_crossings,
)


def _cross(fid: str, li: int, lj: int, axis: int, edge_extent: float = 0.0) -> dict:
    """A sub-C crossings.parquet row (point-crossing when edge_extent == 0)."""
    return {
        "source_feature_id": fid,
        "lower_cell_i": li,
        "lower_cell_j": lj,
        "axis": axis,
        "edge_extent_length_m": edge_extent,
    }


def _feat(fid: str, ci: int, cj: int, class_raw: str = "residential") -> dict:
    """A sub-C features.parquet row (the clipped fragment present in cell (ci,cj))."""
    return {"source_feature_id": fid, "cell_i": ci, "cell_j": cj, "class_raw": class_raw}


# --- THE TEETH: an orphaned road crossing must be COUNTED as anomaly ---------


def test_orphan_no_feature_row_is_anomaly() -> None:
    """Road crosses edge (per crossings) but sub-C wrote NO fragment anywhere.

    fid absent from features entirely -> fid_class.get(fid) is None (not nonroad,
    so NOT filtered) -> in_a == in_b == False -> anomaly. This is the literal drop
    signature the census exists to catch.
    """
    crossings = [_cross("ORPH", li=2, lj=3, axis=0)]
    features: list[dict] = []  # the dropped road left no fragment on either side
    counts = classify_point_crossings(crossings, features)
    assert counts["anomaly"] == 1, "orphaned crossing must be counted as anomaly"
    assert counts["touch_as_cross"] == 0
    assert counts["real_cross"] == 0
    assert counts["point_cross"] == 1


def test_orphan_road_fragment_in_far_cell_is_anomaly() -> None:
    """fid present in features but only in a NON-flanking cell still = anomaly.

    Guards against a classifier that checks "fid present in the city at all"
    instead of "present in THESE two flanking cells." Flanking cells of
    (li=2,lj=3,axis=0) are (2,3) and (3,3); the fragment sits in (7,7).
    """
    crossings = [_cross("ORPH", li=2, lj=3, axis=0)]
    features = [_feat("ORPH", 7, 7, class_raw="residential")]
    counts = classify_point_crossings(crossings, features)
    assert counts["anomaly"] == 1
    assert counts["touch_as_cross"] == 0
    assert counts["real_cross"] == 0


def test_orphan_on_axis1_uses_correct_flanking_cells() -> None:
    """axis==1 flanks (li,lj) and (li,lj+1); an orphan there is still anomaly.

    A fragment placed in (li+1,lj) — the axis-0 neighbour — must NOT count as
    present, or an axis confusion would mask a real drop as a touch.
    """
    crossings = [_cross("ORPH", li=2, lj=3, axis=1)]  # flanks (2,3) and (2,4)
    features = [_feat("ORPH", 3, 3, class_raw="service")]  # axis-0 neighbour, wrong
    counts = classify_point_crossings(crossings, features)
    assert counts["anomaly"] == 1
    assert counts["touch_as_cross"] == 0


# --- DISTINGUISHMENT: the three regimes must route to three distinct buckets --


def test_three_regimes_distinguished() -> None:
    """real (both) / touch (one) / orphan (neither) in one census, each counted
    once in its own bucket. A gate that cannot tell orphan from touch would let a
    drop hide in the (large, expected-nonzero) touch population."""
    crossings = [
        _cross("REAL", li=1, lj=1, axis=0),  # both flanks present
        _cross("TOUCH", li=4, lj=4, axis=0),  # one flank present (§8.3 touch)
        _cross("ORPH", li=6, lj=6, axis=0),  # neither flank present (drop)
    ]
    features = [
        _feat("REAL", 1, 1),
        _feat("REAL", 2, 1),  # flanks of (1,1,axis0) = (1,1),(2,1)
        _feat("TOUCH", 4, 4),  # only one flank of (4,4,axis0) = (4,4)
        # ORPH: no fragment anywhere
    ]
    counts = classify_point_crossings(crossings, features)
    assert counts == {
        "point_cross": 3,
        "real_cross": 1,
        "touch_as_cross": 1,
        "anomaly": 1,
    }


# --- FILTER DIRECTION: exclusions must not swallow a road orphan, and must not
#     manufacture an anomaly from a non-road / interval record ----------------


def test_nonroad_orphan_excluded_not_counted_as_anomaly() -> None:
    """A waterway/rail crossing in neither cell is EXCLUDED (it never emits a
    bref / sets the edge class per §5.1), so it must not inflate anomaly. Proves
    the nonroad filter fires in the correct direction."""
    for cls in sorted(NONROAD_CLASS_RAW):
        crossings = [_cross("WATER", li=2, lj=3, axis=0)]
        features = [_feat("WATER", 7, 7, class_raw=cls)]  # orphan-shaped, but nonroad
        counts = classify_point_crossings(crossings, features)
        assert counts["anomaly"] == 0, f"nonroad {cls} must not count as anomaly"
        assert counts["point_cross"] == 0, f"nonroad {cls} must be excluded entirely"


def test_interval_record_excluded() -> None:
    """edge_extent_length_m != 0 is a polygon edge-interval, not a road
    point-crossing — skipped even if it would otherwise look orphaned."""
    crossings = [_cross("POLY", li=2, lj=3, axis=0, edge_extent=12.5)]
    features: list[dict] = []
    counts = classify_point_crossings(crossings, features)
    assert counts == {"point_cross": 0, "real_cross": 0, "touch_as_cross": 0, "anomaly": 0}


def test_clean_corpus_reports_zero_anomaly() -> None:
    """Sanity: a tile with only genuine crossings + touches reports anomaly 0 —
    so a real corpus reporting 0 is a meaningful negative, not a constant."""
    crossings = [_cross("REAL", li=1, lj=1, axis=0), _cross("TOUCH", li=4, lj=4, axis=1)]
    features = [
        _feat("REAL", 1, 1),
        _feat("REAL", 2, 1),
        _feat("TOUCH", 4, 4),  # one flank of (4,4,axis1) = (4,4),(4,5)
    ]
    counts = classify_point_crossings(crossings, features)
    assert counts["anomaly"] == 0
    assert counts["real_cross"] == 1
    assert counts["touch_as_cross"] == 1
