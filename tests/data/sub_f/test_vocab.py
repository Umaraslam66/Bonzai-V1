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
from collections import Counter
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"
SUB_C_SG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "sub_c"
    / "2026-04-15.0"
    / "singapore"
)

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
UNKNOWN_FAMILY_START_ID: Final[int] = 200
UNKNOWN_FAMILY_END_ID: Final[int] = 255
UNKNOWN_FAMILY_USED_END_ID: Final[int] = 227
UNKNOWN_FAMILY_SLOT_STATUS: Final[str] = "LOCKED"
SENTINEL_INVENTORY_STATUS: Final[str] = "LOCKED"
SENTINEL_VALUES: Final[frozenset[str]] = frozenset({"unknown"})
SENTINEL_PREFIXES: Final[tuple[str, ...]] = ("B_",)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_sub_c_unknown_sentinel(value: str) -> bool:
    return value in SENTINEL_VALUES or "__UNK__" in value or value.startswith(SENTINEL_PREFIXES)


@pytest.fixture(scope="module")
def semantic_vocab() -> dict:
    return _load_yaml(CONFIG_ROOT / "semantic_vocab.yaml")


@pytest.fixture(scope="module")
def unknown_family() -> dict:
    return _load_yaml(CONFIG_ROOT / "unknown_family.yaml")


@pytest.fixture(scope="module")
def sentinel_inventory() -> dict:
    return _load_yaml(CONFIG_ROOT / "sentinel_inventory.yaml")


@pytest.fixture(scope="module")
def sub_c_singapore_counts() -> Counter[tuple[str, str], int]:
    feature_class_to_key = {0: "highway", 1: "building"}
    counts: Counter[tuple[str, str], int] = Counter()
    for path in sorted(SUB_C_SG_ROOT.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            key = feature_class_to_key.get(row["feature_class"])
            value = row.get("class_raw")
            if key and value:
                counts[(key, value)] += 1
    return counts


def test_vocab_floor_analysis_has_28_l1_must_appears():
    """L1 enumeration covers all 28 wiki primary feature keys per cascade #5."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    # Hand-pinned (assertion does NOT read sub-F's own enumeration into expected).
    expected_l1 = {
        "aerialway",
        "aeroway",
        "amenity",
        "barrier",
        "boundary",
        "building",
        "craft",
        "emergency",
        "geological",
        "healthcare",
        "highway",
        "historic",
        "landuse",
        "leisure",
        "man_made",
        "military",
        "natural",
        "office",
        "place",
        "power",
        "public_transport",
        "railway",
        "route",
        "shop",
        "telecom",
        "tourism",
        "water",
        "waterway",
    }
    actual_l1 = set(data["wiki_l1_must_appears"])
    assert actual_l1 == expected_l1, (
        f"L1 set drift: missing={expected_l1 - actual_l1}, extra={actual_l1 - expected_l1}"
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
    excluded_pairs = {(p["key"], p["value"]): p["count"] for p in sentinel_filter["excluded_pairs"]}
    assert sentinel_filter["status"] == "applied before X derivation per cascade #7"
    assert excluded_pairs[("building", "B__UNK__")] > 0
    assert excluded_pairs[("highway", "unknown")] > 0


def test_taginfo_snapshot_paginates_building_values():
    """Cascade #6: building value rows require multi-page taginfo coverage."""
    csv_path = CONFIG_ROOT / "taginfo" / "2026-04-15.0.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    building_values = [
        r for r in rows if r["row_type"] == "value" and r["parent_key"] == "building"
    ]
    assert len(building_values) >= 8000, (
        "Expected building value rows to include paginated taginfo results. "
        f"Observed only {len(building_values)} rows, which indicates a likely "
        "regression to single-page rp=999 coverage."
    )


def test_unknown_family_has_28_locked_slots(unknown_family):
    assert unknown_family["_status"] == UNKNOWN_FAMILY_SLOT_STATUS
    assert unknown_family["slot_list_status"] == "LOCKED"
    assert len(unknown_family["slots"]) == N_L1_MUST_APPEARS_EXPECTED


def test_unknown_family_zero_firing_slots_keep_policy_is_locked(unknown_family):
    policy = unknown_family["zero_firing_slots_policy"]
    assert policy["status"] == "LOCKED_KEEP_ALL_28_UNKNOWN_SLOTS"
    assert "scope-of-coverage-zero, not OSM-real-zero" in policy["note"]
    assert "multi-region unknown expansion" in policy["note"]


def test_unknown_family_slot_order_follows_semantic_vocab_l1_order(semantic_vocab, unknown_family):
    expected_keys = [
        slot["tag"].split("=", 1)[0]
        for slot in semantic_vocab["slots"]
        if slot["tag"].endswith("=*")
    ]
    assert expected_keys == [slot["key"] for slot in unknown_family["slots"]]


def test_unknown_family_uses_bp4_reserved_block(unknown_family):
    block = unknown_family["family_block"]
    slots = unknown_family["slots"]
    assert block["start_id"] == UNKNOWN_FAMILY_START_ID
    assert block["end_id"] == UNKNOWN_FAMILY_END_ID
    assert block["used_start_id"] == UNKNOWN_FAMILY_START_ID
    assert block["used_end_id"] == UNKNOWN_FAMILY_USED_END_ID
    assert [slot["id"] for slot in slots] == list(
        range(UNKNOWN_FAMILY_START_ID, UNKNOWN_FAMILY_USED_END_ID + 1)
    )


def test_unknown_family_real_osm_counts_exclude_all_locked_semantic_tags(
    semantic_vocab, unknown_family, sub_c_singapore_counts
):
    semantic_tags = {slot["tag"] for slot in semantic_vocab["slots"]}
    expected_real_unknown = Counter()
    expected_sentinel = Counter()
    for (key, value), count in sub_c_singapore_counts.items():
        tag = f"{key}={value}"
        if _is_sub_c_unknown_sentinel(value):
            expected_sentinel[key] += count
        elif tag not in semantic_tags:
            expected_real_unknown[key] += count

    slots_by_key = {slot["key"]: slot for slot in unknown_family["slots"]}
    assert (
        slots_by_key["highway"]["singapore_count_real_osm_below_F"]
        == expected_real_unknown["highway"]
    )
    assert (
        slots_by_key["building"]["singapore_count_real_osm_below_F"]
        == expected_real_unknown["building"]
    )
    assert slots_by_key["highway"]["singapore_count_subc_sentinels"] == expected_sentinel["highway"]
    assert (
        slots_by_key["building"]["singapore_count_subc_sentinels"] == expected_sentinel["building"]
    )


def test_unknown_family_subc_sentinel_counts_are_reported_separately(unknown_family):
    slots_by_key = {slot["key"]: slot for slot in unknown_family["slots"]}
    assert slots_by_key["highway"]["singapore_count_subc_sentinels"] > 0
    assert slots_by_key["building"]["singapore_count_subc_sentinels"] > 0
    assert (
        slots_by_key["highway"]["singapore_count_total"]
        == slots_by_key["highway"]["singapore_count_real_osm_below_F"]
        + slots_by_key["highway"]["singapore_count_subc_sentinels"]
    )
    assert (
        slots_by_key["building"]["singapore_count_total"]
        == slots_by_key["building"]["singapore_count_real_osm_below_F"]
        + slots_by_key["building"]["singapore_count_subc_sentinels"]
    )


def test_unknown_family_reports_building_b_unk_raw_cache_decomposition(unknown_family):
    building = unknown_family["sentinel_decomposition"]["building_b_unk"]
    assert building["total_count"] == 301418
    assert building["source_id_join_missing_count"] == 0
    assert building["raw_class_top20"][0] == {
        "value": "<NULL>",
        "count": 301418,
        "fraction": 1.0,
    }
    assert building["raw_subtype_top20"][0]["value"] == "<NULL>"
    assert building["raw_subtype_top20"][0]["count"] == 299237
    assert "root_cause_b" in building["root_cause_classification"]
    assert "no_cascade_8" in building["root_cause_classification"]


def test_unknown_family_reports_highway_unknown_raw_cache_decomposition(unknown_family):
    highway = unknown_family["sentinel_decomposition"]["highway_unknown"]
    assert highway["total_count"] == 9748
    assert highway["source_id_join_missing_count"] == 0
    assert highway["raw_class_top20"][0] == {
        "value": "unknown",
        "count": 9748,
        "fraction": 1.0,
    }
    subtype_counts = {r["value"]: r["count"] for r in highway["raw_subtype_top20"]}
    assert subtype_counts == {"road": 8226, "rail": 1522}
    assert "root_cause_b" in highway["root_cause_classification"]
    assert "no_cascade_8" in highway["root_cause_classification"]


def test_unknown_family_over_firing_classification_is_locked_no_cascade_8(
    unknown_family,
):
    classification = unknown_family["over_firing_classification"]
    assert classification["status"] == "LOCKED_NO_CASCADE_8"
    assert classification["no_sub_c_v2_candidate"] is True
    assert "Root cause (b)" in classification["building"]
    assert "Root cause (b)" in classification["highway"]


def test_sentinel_inventory_reserves_dataloader_only_ids(sentinel_inventory):
    assert sentinel_inventory["_status"] == SENTINEL_INVENTORY_STATUS
    dataloader = sentinel_inventory["dataloader_sentinels"]
    assert dataloader["block"]["start_id"] == 256
    assert dataloader["block"]["end_id"] == 299
    assert dataloader["block"]["status"] == "LOCKED at Halt 3 continuation"
    expected = {
        "<pad>": 256,
        "<eos>": 257,
        "<bos>": 258,
        "<cell_start>": 259,
        "<cell_end>": 260,
    }
    actual = {slot["token"]: slot["id"] for slot in dataloader["slots"]}
    assert actual == expected
    assert all(slot["on_disk"] is False for slot in dataloader["slots"])


def test_sentinel_inventory_has_locked_bp2_and_bp7_blocks(sentinel_inventory):
    assert sentinel_inventory["bp1_semantic"]["status"] == "LOCKED at Halt 3 continuation"
    assert sentinel_inventory["bp4_unknown_family"]["status"] == "LOCKED at Halt 3 continuation"
    bp2 = sentinel_inventory["bp2_encoding_primitives"]
    bp7 = sentinel_inventory["bp7_boundary_ref"]
    assert bp2["start_id"] == 300
    assert bp2["end_id"] == 1499
    assert bp2["placeholder"] is False
    # Post `fix(sub_f): sentinel_inventory consume 509, 510 for <feature>/<feature_end>`
    # (commit 4c4f880, 2026-05-28): BP2 status carries the consumption note;
    # used_count/reserved_count reflect the 2 consumed slots;
    # reserved_v2_headroom shrank from the front (509-1499 -> 511-1499).
    assert bp2["status"] == (
        "LOCKED at Halt 2 approval; "
        "structural_sentinels consumed at T8 plan-write 2026-05-28"
    )
    assert bp2["used_count"] == 211
    assert bp2["reserved_count"] == 989
    assert bp2["sub_blocks"]["anchor"] == {
        "start_id": 300,
        "end_id": 395,
        "slot_count": 96,
    }
    assert bp2["sub_blocks"]["direction"] == {
        "start_id": 396,
        "end_id": 443,
        "slot_count": 48,
    }
    assert bp2["sub_blocks"]["magnitude"] == {
        "start_id": 444,
        "end_id": 508,
        "slot_count": 65,
    }
    assert bp2["sub_blocks"]["reserved_v2_headroom"] == {
        "start_id": 511,
        "end_id": 1499,
        "slot_count": 989,
    }
    # Structural sentinels consumed from reserved_v2_headroom front. Locked
    # record lives under bp2["consumed_from_reserved_v2_headroom"]["slots"].
    consumed = bp2["consumed_from_reserved_v2_headroom"]["slots"]
    assert [s["id"] for s in consumed] == [509, 510]
    assert [s["token"] for s in consumed] == ["<feature>", "<feature_end>"]
    assert all(s["on_disk"] is True for s in consumed)
    assert all(s["family_tag"] == "structural" for s in consumed)
    assert bp7["start_id"] == 1500
    assert bp7["end_id"] == 1599
    assert bp7["placeholder"] is False
    assert bp7["status"] == "LOCKED at Halt 7 approval"
    assert bp7["used_count"] == 8
    assert bp7["reserved_count"] == 92


def test_load_sub_f_vocab_returns_all_on_disk_families_in_id_order():
    """Vocab loader returns every on-disk slot in ascending token_id order.

    Families: BP1 semantic + BP4 unknown + BP2 encoding_primitive + structural + BP7.
    On-disk excludes dataloader sentinels (256-260 per sentinel_inventory.yaml
    dataloader_sentinels block, on_disk=false). Total on-disk count is the sum
    of BP1 used (127) + BP4 used (28) + BP2 encoding_primitive used (209) +
    structural sentinels (2 - <feature>/<feature_end> at 509/510, consumed
    from BP2 reserved_v2_headroom front per pre-flight Assertion 4) +
    BP7 used (8) = 374 slots.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    slots = load_sub_f_vocab()
    assert len(slots) == 374, f"expected 374 on-disk slots; got {len(slots)}"

    # Strictly ascending token_id (per `feedback_pythonhashseed_dict_iteration_test`
    # the loader must produce deterministic order, not hash-order).
    ids = [s.token_id for s in slots]
    assert ids == sorted(ids), "token_ids must be in strictly ascending order"
    assert len(set(ids)) == len(ids), "token_ids must be unique"

    # Family boundaries per sentinel_inventory.yaml.
    bp1 = [s for s in slots if s.family == "semantic"]
    bp4 = [s for s in slots if s.family == "unknown"]
    bp2 = [s for s in slots if s.family == "encoding_primitive"]
    structural = [s for s in slots if s.family == "structural"]
    bp7 = [s for s in slots if s.family == "boundary_reference"]
    assert len(bp1) == 127
    assert len(bp4) == 28
    assert len(bp2) == 209  # anchor 96 + direction 48 + magnitude 65
    assert len(structural) == 2  # <feature> + <feature_end>
    assert len(bp7) == 8

    # ID-range invariants from sentinel_inventory.yaml.
    assert all(0 <= s.token_id <= 126 for s in bp1)
    assert all(200 <= s.token_id <= 227 for s in bp4)
    assert all(300 <= s.token_id <= 508 for s in bp2)
    assert {s.token_id for s in structural} == {509, 510}
    assert all(1500 <= s.token_id <= 1507 for s in bp7)

    # Structural family carries the named tags exactly.
    structural_tags = {s.tag: s.token_id for s in structural}
    assert structural_tags == {"<feature>": 509, "<feature_end>": 510}


def test_load_sub_f_vocab_no_dataloader_sentinels_on_disk():
    """Per sentinel_inventory.yaml: <pad>=256, <eos>=257, <bos>=258, <cell_start>=259,
    <cell_end>=260 are on_disk=false. They must NOT appear in load_sub_f_vocab()."""
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    on_disk_ids = {s.token_id for s in load_sub_f_vocab()}
    for sentinel_id in (256, 257, 258, 259, 260):
        assert sentinel_id not in on_disk_ids, (
            f"dataloader sentinel id={sentinel_id} must NOT be in on-disk vocab"
        )


def test_load_sub_f_vocab_tag_lookup_round_trips():
    """Every slot has a unique tag; tag -> token_id lookup matches token_id -> tag."""
    from cfm.data.sub_f.vocab import load_sub_f_vocab, vocab_tag_to_id

    slots = load_sub_f_vocab()
    tag_to_id = vocab_tag_to_id()
    for s in slots:
        assert tag_to_id[s.tag] == s.token_id
    assert len(tag_to_id) == len(slots), "tags must be unique across all families"
