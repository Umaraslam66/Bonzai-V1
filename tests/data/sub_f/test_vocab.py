"""Tests for sub-F vocab loading + Gate 6 structural check.

Per spec §8.1 BP1 row + cascade #5 + Safeguard 2: vocab passes iff:
(a) F frequency floor cuts at the chosen quantile, AND
(b) every hand-enumerated wiki Map_features must-appear is a first-class slot,
    enumeration verified by set-equality AND per-key count assertions.

Assertion logic does NOT use sub-F's own derivation in expected-value
computation per Gate 6 + spec §13.5 protocol-v2 candidate iii (reviewer-supplied
lists are untrusted; hand-counts derived independently from pair sets).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"

# Hand-derived from wikitext at configs/sub_f/wiki_map_features/2026-04-15.0.wikitext
# `==Primary features==` section transclusion count. Independently counted from
# the WIKI_L1_MUST_APPEARS tuple in floor_analysis.py per Safeguard 2.
N_L1_MUST_APPEARS_EXPECTED: Final[int] = 28

# Hand-derived from Template:Map_Features:highway value table (Roads + Link
# roads + Special road types + Paths + Lifecycle subsections). Excludes
# Other highway features (stops, signals, infrastructure point-features).
N_L2_HIGHWAY_EXPECTED: Final[int] = 23

# Hand-derived from Template:Building_typology value table + 1 "yes" catch-all.
N_L2_BUILDING_EXPECTED: Final[int] = 33


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_vocab_floor_analysis_has_28_l1_must_appears():
    """L1 enumeration covers all 28 wiki primary feature keys per cascade #5."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    # Hand-pinned (assertion does NOT read sub-F's own enumeration into expected).
    expected_l1 = {
        "aerialway", "aeroway", "amenity", "barrier", "boundary", "building",
        "craft", "emergency", "geological", "healthcare", "highway", "historic",
        "landuse", "leisure", "man_made", "military", "natural", "office",
        "place", "power", "public_transport", "railway", "route", "shop",
        "telecom", "tourism", "water", "waterway",
    }
    actual_l1 = set(data["wiki_l1_must_appears"])
    assert actual_l1 == expected_l1, (
        f"L1 set drift: missing={expected_l1 - actual_l1}, "
        f"extra={actual_l1 - expected_l1}"
    )


def test_vocab_floor_analysis_l1_count_matches_independent_hand_count():
    """Safeguard 2: per-key count derived independently from set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual_count = len(data["wiki_l1_must_appears"])
    assert actual_count == N_L1_MUST_APPEARS_EXPECTED, (
        f"L1 count mismatch: floor_analysis.py shipped {actual_count}, "
        f"hand-counted from wikitext = {N_L1_MUST_APPEARS_EXPECTED}. "
        f"If wikitext changed: re-count and update N_L1_MUST_APPEARS_EXPECTED."
    )


def test_vocab_floor_analysis_l2_highway_count_matches_independent_hand_count():
    """Safeguard 2: highway L2 count derived independently from pair set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual = data["wiki_l2_highway_count"]
    assert actual == N_L2_HIGHWAY_EXPECTED, (
        f"L2 highway count mismatch: floor_analysis.py shipped {actual}, "
        f"hand-counted from Template:Map_Features:highway = {N_L2_HIGHWAY_EXPECTED}. "
        f"If template changed: re-count and update N_L2_HIGHWAY_EXPECTED."
    )


def test_vocab_floor_analysis_l2_building_count_matches_independent_hand_count():
    """Safeguard 2: building L2 count derived independently from pair set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual = data["wiki_l2_building_count"]
    assert actual == N_L2_BUILDING_EXPECTED, (
        f"L2 building count mismatch: floor_analysis.py shipped {actual}, "
        f"hand-counted from Template:Building_typology + 'yes' = {N_L2_BUILDING_EXPECTED}. "
        f"If template changed: re-count and update N_L2_BUILDING_EXPECTED."
    )


def test_vocab_floor_analysis_l3_deferred_per_spec_12_10():
    """L3 explicitly deferred per cascade #5 + spec §12 #10."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    l3_row = next(r for r in data["curve"] if r["level"] == 3)
    assert l3_row["f_min"] is None
    assert l3_row["vocab_size"] is None
    assert l3_row["must_appears_count"] == 0
    assert "deferred" in l3_row["level_description"].lower()


def test_vocab_floor_analysis_curve_includes_l1_and_l2_rows():
    """Curve has L1 + L2 rows (L3 row exists but deferred per above test)."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    levels = {r["level"] for r in data["curve"]}
    assert levels == {1, 2, 3}, f"missing curve levels: {{1,2,3}} - {levels}"


def test_vocab_floor_analysis_singapore_x_threshold_scoped_to_highway_building():
    """Singapore X-threshold scope per cascade #4 (POI + base deferred per §12 #11)."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    x = data["proposed_x_threshold"]
    assert "highway + building" in x["scope_note"]
    assert "POI + base" in x["scope_note"]
    # Candidates surface concrete values (not just hand-waved).
    assert "candidate_a_singapore_elbow" in x
    assert "candidate_b_median_must_appear_freq" in x


def test_vocab_floor_analysis_filters_sub_c_unknown_sentinels():
    """Cascade #7: X-threshold excludes sub-C normalization sentinels."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    sentinel_filter = data["proposed_x_threshold"]["sentinel_filter"]
    excluded_pairs = {
        (p["key"], p["value"]): p["count"]
        for p in sentinel_filter["excluded_pairs"]
    }
    assert sentinel_filter["status"] == "applied before X derivation per cascade #7"
    assert excluded_pairs[("building", "B__UNK__")] > 0
    assert excluded_pairs[("highway", "unknown")] > 0


def test_taginfo_snapshot_paginates_building_values():
    """Cascade #6: building value rows require multi-page taginfo coverage."""
    csv_path = CONFIG_ROOT / "taginfo" / "2026-04-15.0.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    building_values = [
        r for r in rows
        if r["row_type"] == "value" and r["parent_key"] == "building"
    ]
    assert len(building_values) >= 8000, (
        "Expected building value rows to include paginated taginfo results. "
        f"Observed only {len(building_values)} rows, which indicates a likely "
        "regression to single-page rp=999 coverage."
    )
