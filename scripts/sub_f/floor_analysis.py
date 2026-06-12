"""Compute BP1 vocab floor marginal-cost curve + Gate 6 structural check.

Outputs configs/sub_f/vocab_floor_analysis.yaml at Halt 1.

Per spec §2.1 + plan cascade #4 + #5 resolutions:
- L1: full 28 keys from wiki Map_features ==Primary features== section.
- L2: highway + building only (Singapore-X-applicable per cascade #4).
- L3: deferred entirely per spec §12 #10.
- Singapore X-threshold: highway + building only per cascade #4.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# WIKI ENUMERATIONS — hand-enumerated from configs/sub_f/wiki_map_features/
# 2026-04-15.0.wikitext (Gate 6 canonical source, NOT reviewer-supplied or
# memory-inferred per cascade #5 lesson + spec §13.5 protocol-v2 candidate iii).
# Per-key hand-counts asserted independently in tests/data/sub_f/test_vocab.py
# per Safeguard 2 (catches per-section enumeration errors flat set comparison
# misses).
# ---------------------------------------------------------------------------

# L1: full 28 primary feature keys from Map_features ==Primary features==
# transclusion list. Each `{{Map_Features:X}}` or `{{Building typology}}` line
# in that section contributes one key.
WIKI_L1_MUST_APPEARS: Final[tuple[str, ...]] = (
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
)

# L2: highway + building only per cascade #4 (Singapore X scope).
# Other 26 keys' L2 deferred per spec §12 #11.
#
# WIKI_L2_HIGHWAY: hand-enumerated from Template:Map_Features:highway value
# table — the "way-class" subset (Roads + Link roads + Special road types +
# Paths). Excludes "Other highway features" (stops, signals, milestones,
# infrastructure) which are point-features not way-classifications.
WIKI_L2_HIGHWAY: Final[tuple[str, ...]] = (
    # Roads (7 main road network tags per template's road_network annotation)
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    # Link roads
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    # Special road types
    "living_street",
    "service",
    "pedestrian",
    "busway",
    # Paths
    "footway",
    "cycleway",
    "bridleway",
    "path",
    "steps",
    "track",
    # Lifecycle placeholder
    "road",
)

# WIKI_L2_BUILDING: hand-enumerated from Template:Building_typology value
# table + "yes" catch-all (OSM convention for unspecified buildings,
# extremely common but not in the typology template).
WIKI_L2_BUILDING: Final[tuple[str, ...]] = (
    "yes",  # catch-all (added manually per OSM convention, not in template)
    # Typology values from Building_typology template
    "annexe",
    "apartments",
    "barn",
    "barracks",
    "bungalow",
    "cabin",
    "commercial",
    "detached",
    "dormitory",
    "entrance",
    "farm",
    "farm_auxiliary",
    "gatehouse",
    "ger",
    "hangar",
    "hotel",
    "house",
    "houseboat",
    "library",
    "office",
    "public",
    "residential",
    "semidetached_house",
    "service",
    "shed",
    "static_caravan",
    "stilt_house",
    "supermarket",
    "terrace",
    "train_station",
    "tree_house",
    "trullo",
)

WIKI_L2_PRIMARY_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {("highway", v) for v in WIKI_L2_HIGHWAY} | {("building", v) for v in WIKI_L2_BUILDING}
)

# L3: deferred entirely per spec §12 #10. Placeholder for future expansion.
WIKI_L3_ALL_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset()

# ---------------------------------------------------------------------------
# SUB-C FEATURE_CLASS MAPPING — scoped to highway + building per cascade #4
# (sub-C feature_class=2 poi has NULL class_raw; feature_class=3 base lumps
# water+landuse+natural with ambiguous parent key). Per spec §12 #11.
# ---------------------------------------------------------------------------
FEATURE_CLASS_TO_KEY: Final[dict[int, str]] = {
    0: "highway",  # road class — exact 1:1 (sub-C extracts only highway)
    1: "building",  # exact 1:1
    # 2 (poi) + 3 (base) deferred per cascade #4.
}


def load_taginfo(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dominant_element_type(key_row: dict) -> str:
    """Return 'ways' / 'nodes' / 'relations' for key's dominant element type."""
    counts = {
        "ways": int(key_row["count_ways"]),
        "nodes": int(key_row["count_nodes"]),
        "relations": int(key_row["count_relations"]),
    }
    return max(counts, key=lambda k: counts[k])


def et_totals_from_taginfo(rows: list[dict]) -> dict[str, int]:
    """Approximate global ET totals as sum of key-row counts per ET.

    Documented approximation: sums over the top-N keys snapshotted; not the
    true global element population (which would require taginfo /site/info).
    Sufficient for fraction-comparison purposes at Halt 1.
    """
    totals = Counter()
    for r in rows:
        if r["row_type"] == "key":
            totals["ways"] += int(r["count_ways"])
            totals["nodes"] += int(r["count_nodes"])
            totals["relations"] += int(r["count_relations"])
    return dict(totals)


def fraction_within_et(
    row: dict, et_totals: dict[str, int], key_rows_by_name: dict[str, dict]
) -> float:
    """Fraction-of-feature-bearing-elements within row's dominant ET (BP1 fix 2)."""
    if row["row_type"] == "key":
        et = dominant_element_type(row)
        denom = et_totals.get(et, 0)
        numerator = int(row[f"count_{et}"])
    else:  # value row: inherit parent key's dominant ET (documented approximation)
        parent = key_rows_by_name.get(row["parent_key"])
        if not parent:
            return 0.0
        et = dominant_element_type(parent)
        denom = et_totals.get(et, 0)
        numerator = int(row["count_all"])  # parent ET distribution assumed for value
    return numerator / denom if denom else 0.0


def f_min_for_level(
    rows: list[dict],
    level: int,
    et_totals: dict[str, int],
    key_rows_by_name: dict[str, dict],
) -> float:
    """Smallest F such that all must-appears at level are admitted."""
    if level == 1:
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "key" and r["key"] in WIKI_L1_MUST_APPEARS
        ]
    elif level == 2:
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "value" and (r["parent_key"], r["value"]) in WIKI_L2_PRIMARY_PAIRS
        ]
    elif level == 3:
        if not WIKI_L3_ALL_PAIRS:
            return float("nan")  # L3 deferred per spec §12 #10
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "value" and (r["parent_key"], r["value"]) in WIKI_L3_ALL_PAIRS
        ]
    else:
        raise ValueError(f"unknown level {level}")
    return min(candidates) if candidates else 0.0


def vocab_size_at_F(
    rows: list[dict],
    F: float,
    level: int,
    et_totals: dict[str, int],
    key_rows_by_name: dict[str, dict],
) -> int:
    """Count slots at granularity level passing F (Bug 3 fix: level-aware row-type filter)."""
    if level == 1:
        return sum(
            1
            for r in rows
            if r["row_type"] == "key" and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    if level == 2:
        return sum(
            1
            for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L2_PRIMARY_PAIRS
            and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    if level == 3:
        if not WIKI_L3_ALL_PAIRS:
            return 0  # L3 deferred
        return sum(
            1
            for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L3_ALL_PAIRS
            and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    raise ValueError(f"unknown level {level}")


# ---------------------------------------------------------------------------
# SINGAPORE X-THRESHOLD computation (Bug 5 fix per cascade #4 scope).
# ---------------------------------------------------------------------------

SUB_C_UNKNOWN_SENTINEL_VALUES: Final[frozenset[str]] = frozenset({"unknown"})
SUB_C_UNKNOWN_SENTINEL_PREFIXES: Final[tuple[str, ...]] = ("B_",)


def is_sub_c_unknown_sentinel(value: str) -> bool:
    """Return True for sub-C normalization sentinels, not real OSM values."""
    return (
        value in SUB_C_UNKNOWN_SENTINEL_VALUES
        or "__UNK__" in value
        or value.startswith(SUB_C_UNKNOWN_SENTINEL_PREFIXES)
    )


def compute_singapore_frequencies(
    sub_c_region: Path,
) -> dict[tuple[str, str], int]:
    """Count (inferred_key, class_raw) tag pairs on cached Singapore sub-C extracts.

    Scoped to FEATURE_CLASS_TO_KEY = {0: highway, 1: building} per cascade #4.
    POI (2) + base (3) Singapore mapping deferred per spec §12 #11.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for path in sorted(sub_c_region.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            key = FEATURE_CLASS_TO_KEY.get(r["feature_class"])
            value = r.get("class_raw")
            if key and value:  # NULL class_raw skipped (POI rows have NULL)
                counts[(key, value)] += 1
    return dict(counts)


def filter_sub_c_unknown_sentinels(
    sg_freqs: dict[tuple[str, str], int],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Remove sub-C unknown sentinels before deriving Singapore X (cascade #7)."""
    filtered: dict[tuple[str, str], int] = {}
    excluded: dict[tuple[str, str], int] = {}
    for pair, count in sg_freqs.items():
        value = pair[1]
        if is_sub_c_unknown_sentinel(value):
            excluded[pair] = count
        else:
            filtered[pair] = count
    return filtered, excluded


def derive_x_threshold(
    sg_freqs: dict[tuple[str, str], int],
    wiki_must_appears: frozenset[tuple[str, str]],
) -> dict:
    """Compute X threshold candidates from Singapore distribution.

    Candidate A: Singapore's own elbow F-equivalent (min Singapore-fraction
                 of wiki must-appears actually present on Singapore).
    Candidate B: median Singapore frequency across present must-appears.
    """
    total_sg = sum(sg_freqs.values())
    if not total_sg:
        return {"error": "no Singapore data — sub-C cache missing or empty"}

    must_appear_fractions = sorted(
        (sg_freqs.get(p, 0) / total_sg for p in wiki_must_appears),
        reverse=True,
    )
    present_fractions = [f for f in must_appear_fractions if f > 0]
    return {
        "candidate_a_singapore_elbow": float(min(present_fractions)) if present_fractions else 0.0,
        "candidate_b_median_must_appear_freq": (
            float(present_fractions[len(present_fractions) // 2]) if present_fractions else 0.0
        ),
        "n_must_appears_present_in_singapore": len(present_fractions),
        "n_must_appears_total": len(wiki_must_appears),
        "scope_note": (
            "highway + building only per cascade #4; sub-C unknown sentinels "
            "filtered per cascade #7; POI + base deferred per spec §12 #11."
        ),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    parser.add_argument(
        "--sub-c-region-dir",
        type=Path,
        default=Path("data/processed/sub_c/2026-04-15.0/singapore"),
    )
    args = parser.parse_args()

    taginfo_csv = ROOT / "configs" / "sub_f" / "taginfo" / f"{args.release}.csv"
    rows = load_taginfo(taginfo_csv)
    et_totals = et_totals_from_taginfo(rows)
    key_rows_by_name = {r["key"]: r for r in rows if r["row_type"] == "key"}

    # L1 + L2 curve (L3 deferred per spec §12 #10).
    f_l1 = f_min_for_level(rows, level=1, et_totals=et_totals, key_rows_by_name=key_rows_by_name)
    vocab_l1 = vocab_size_at_F(
        rows, f_l1, level=1, et_totals=et_totals, key_rows_by_name=key_rows_by_name
    )
    f_l2 = f_min_for_level(rows, level=2, et_totals=et_totals, key_rows_by_name=key_rows_by_name)
    vocab_l2 = vocab_size_at_F(
        rows, f_l2, level=2, et_totals=et_totals, key_rows_by_name=key_rows_by_name
    )

    # Singapore X-threshold per BP1 fix C (Bug 5 fix per cascade #4 scope).
    # Cascade #7: filter sub-C normalization sentinels before X derivation.
    sg_freqs_raw = compute_singapore_frequencies(args.sub_c_region_dir)
    sg_freqs, sg_filtered = filter_sub_c_unknown_sentinels(sg_freqs_raw)
    x_threshold = derive_x_threshold(sg_freqs, WIKI_L2_PRIMARY_PAIRS)

    output = {
        "release": args.release,
        "f_denominator": (
            "fraction-of-feature-bearing-elements within dominant ET per key "
            "(BP1 fix 2; value rows inherit parent's dominant ET as documented "
            "approximation)"
        ),
        "wiki_l1_must_appears": list(WIKI_L1_MUST_APPEARS),
        "wiki_l2_primary_pairs_count": len(WIKI_L2_PRIMARY_PAIRS),
        "wiki_l2_highway_count": len(WIKI_L2_HIGHWAY),
        "wiki_l2_building_count": len(WIKI_L2_BUILDING),
        "wiki_l3_status": "deferred per spec §12 #10",
        "curve": [
            {
                "level": 1,
                "level_description": "top-level keys (28 must-appears)",
                "f_min": float(f_l1),
                "vocab_size": int(vocab_l1),
                "must_appears_count": len(WIKI_L1_MUST_APPEARS),
                "must_appears_admitted": len(WIKI_L1_MUST_APPEARS),
            },
            {
                "level": 2,
                "level_description": "(key, primary-value) pairs — highway + building per "
                "cascade #4",
                "f_min": float(f_l2),
                "vocab_size": int(vocab_l2),
                "must_appears_count": len(WIKI_L2_PRIMARY_PAIRS),
                "must_appears_admitted": len(WIKI_L2_PRIMARY_PAIRS),
            },
            {
                "level": 3,
                "level_description": "all wiki-documented pairs — deferred per spec §12 #10",
                "f_min": None,
                "vocab_size": None,
                "must_appears_count": 0,
                "must_appears_admitted": 0,
                "deferral_reason": "Cascade #5 + recursive marginal-cost-of-cut: enumerate "
                "where reviewable benefit exists.",
            },
        ],
        "proposed_elbow": {
            "status": "LOCKED_BY_REVIEWER_FOR_F_ELBOW; X-threshold pending",
            "granularity": "L1+L2-mixed",
            "f_value": float(f_l2),
            "slot_count_before_x_exceptions": (
                len(WIKI_L1_MUST_APPEARS) + len(WIKI_L2_PRIMARY_PAIRS)
            ),
            "exception_list": [],
            "rationale": (
                "Mixed-B lock: 28 L1 semantic categories + 56 L2 highway/building "
                "primary pairs. Discretionary L1 keys at F_l1 are metadata-heavy "
                "and are not admitted by default."
            ),
        },
        "proposed_x_threshold": {
            "candidate_a_singapore_elbow": x_threshold.get("candidate_a_singapore_elbow"),
            "candidate_b_median_must_appear_freq": x_threshold.get(
                "candidate_b_median_must_appear_freq"
            ),
            "scope_note": x_threshold.get("scope_note"),
            "n_must_appears_present_in_singapore": x_threshold.get(
                "n_must_appears_present_in_singapore"
            ),
            "n_must_appears_total": x_threshold.get("n_must_appears_total"),
            "sentinel_filter": {
                "status": "applied before X derivation per cascade #7",
                "patterns": [
                    "value == 'unknown'",
                    "'__UNK__' in value",
                    "value startswith 'B_'",
                ],
                "excluded_pair_count": len(sg_filtered),
                "excluded_feature_count": int(sum(sg_filtered.values())),
                "excluded_pairs": [
                    {"key": key, "value": value, "count": int(count)}
                    for (key, value), count in sorted(sg_filtered.items())
                ],
            },
            "paired_structural_check": "For each Singapore-frequency-≥X (highway, value) and "
            "(building, value) pair: must appear above F in semantic_vocab.yaml. POI + base "
            "scope deferred per spec §12 #11.",
        },
        "_status": "PROPOSED — pending Halt 1 reviewer approval per spec §10.3.",
    }

    out_path = ROOT / "configs" / "sub_f" / "vocab_floor_analysis.yaml"
    out_path.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[floor_analysis] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
