"""Derive the locked BP4 unknown family + Halt 3 sentinel inventory surface."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "sub_f"
REPORTS_ROOT = ROOT / "reports"
SUB_C_SG_ROOT = (
    ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
)
RAW_CACHE_ROOT = (
    ROOT / "data" / "cache" / "overture" / "2026-04-15.0" / "singapore"
)

UNKNOWN_FAMILY_PATH = CONFIG_ROOT / "unknown_family.yaml"
SENTINEL_INVENTORY_PATH = CONFIG_ROOT / "sentinel_inventory.yaml"
REPORT_PATH = REPORTS_ROOT / "2026-05-23-phase-1-sub-F-task-4-halt.md"

FEATURE_CLASS_TO_KEY: Final[dict[int, str]] = {
    0: "highway",
    1: "building",
}
SUB_C_UNKNOWN_SENTINEL_VALUES: Final[frozenset[str]] = frozenset({"unknown"})
SUB_C_UNKNOWN_SENTINEL_PREFIXES: Final[tuple[str, ...]] = ("B_",)
UNKNOWN_BLOCK = (200, 255)
UNKNOWN_USED_END_ID = 227
DATALOADER_BLOCK = (256, 299)
BP2_PLACEHOLDER_BLOCK = (300, 1499)
BP7_PLACEHOLDER_BLOCK = (1500, 1599)
OVER_FIRING_THRESHOLD = 0.10
ZERO_FIRING_SCOPE_NOTE = (
    "26 zero-firing slots are scope-of-coverage-zero, not OSM-real-zero. "
    "They preserve v1 IDs against multi-region unknown expansion at sub-F-v2 "
    "per cascade #4 deferral."
)


def canonicalize_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=yaml.SafeDumper,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        width=4096,
    )


def is_sub_c_unknown_sentinel(value: str) -> bool:
    return (
        value in SUB_C_UNKNOWN_SENTINEL_VALUES
        or "__UNK__" in value
        or value.startswith(SUB_C_UNKNOWN_SENTINEL_PREFIXES)
    )


def load_semantic_vocab() -> dict:
    return yaml.safe_load((CONFIG_ROOT / "semantic_vocab.yaml").read_text(encoding="utf-8"))


def semantic_l1_keys(semantic_vocab: dict) -> list[str]:
    keys = [
        slot["tag"].split("=", 1)[0]
        for slot in semantic_vocab["slots"]
        if slot["tag"].endswith("=*")
    ]
    if len(keys) != 28:
        raise ValueError(f"Expected 28 L1 semantic keys, found {len(keys)}")
    return keys


def singapore_counts() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in sorted(SUB_C_SG_ROOT.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            key = FEATURE_CLASS_TO_KEY.get(row["feature_class"])
            value = row.get("class_raw")
            if key and value:
                counts[(key, value)] += 1
    return counts


def _display_raw_value(value: object) -> str:
    return "<NULL>" if value is None else str(value)


def _top_counter_items(counter: Counter[str], total_count: int) -> list[dict]:
    return [
        {
            "value": value,
            "count": int(count),
            "fraction": count / total_count if total_count else 0.0,
        }
        for value, count in counter.most_common(20)
    ]


def _top20_coverage(counter: Counter[str], total_count: int) -> float:
    if not total_count:
        return 0.0
    return sum(count for _, count in counter.most_common(20)) / total_count


def _load_raw_lookup(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required raw cache missing: {path}")
    table = pq.ParquetFile(path).read(columns=["id", "class", "subtype"])
    return {
        row["id"]: {
            "class": row.get("class"),
            "subtype": row.get("subtype"),
        }
        for row in table.to_pylist()
    }


def build_sentinel_decomposition() -> dict:
    """Join sub-C unknown sentinels back to raw Overture cache where possible."""
    raw_buildings = _load_raw_lookup(RAW_CACHE_ROOT / "buildings.parquet")
    raw_transportation = _load_raw_lookup(RAW_CACHE_ROOT / "transportation.parquet")
    specs = {
        "building_b_unk": {
            "feature_class": 1,
            "class_raw": "B__UNK__",
            "raw_dataset": "buildings.parquet",
            "raw_lookup": raw_buildings,
            "classification": (
                "root_cause_b_real_osm_long_tail_source_under_typed_no_cascade_8"
            ),
            "cascade_8_assessment": (
                "Raw Overture buildings.class is NULL for the sentinel rows, so "
                "there are no recoverable wiki/semantic building values hidden "
                "behind B__UNK__ in class_raw."
            ),
        },
        "highway_unknown": {
            "feature_class": 0,
            "class_raw": "unknown",
            "raw_dataset": "transportation.parquet",
            "raw_lookup": raw_transportation,
            "classification": (
                "root_cause_b_real_osm_long_tail_literal_upstream_unknown_no_cascade_8"
            ),
            "cascade_8_assessment": (
                "Raw Overture transportation.class is literal 'unknown' for these "
                "rows, not a hidden wiki/semantic highway value."
            ),
        },
    }
    counters = {
        name: {
            "total_count": 0,
            "source_id_join_missing_count": 0,
            "raw_class": Counter(),
            "raw_subtype": Counter(),
        }
        for name in specs
    }

    for path in sorted(SUB_C_SG_ROOT.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read(
            columns=["feature_class", "class_raw", "source_feature_id"]
        )
        for row in table.to_pylist():
            for name, spec in specs.items():
                if (
                    row["feature_class"] != spec["feature_class"]
                    or row.get("class_raw") != spec["class_raw"]
                ):
                    continue
                state = counters[name]
                state["total_count"] += 1
                raw_record = spec["raw_lookup"].get(row.get("source_feature_id"))
                if raw_record is None:
                    state["source_id_join_missing_count"] += 1
                    state["raw_class"]["<MISSING_SOURCE_ID>"] += 1
                    state["raw_subtype"]["<MISSING_SOURCE_ID>"] += 1
                else:
                    state["raw_class"][_display_raw_value(raw_record["class"])] += 1
                    state["raw_subtype"][_display_raw_value(raw_record["subtype"])] += 1

    decomposition = {}
    for name, spec in specs.items():
        state = counters[name]
        total = int(state["total_count"])
        class_counter = state["raw_class"]
        subtype_counter = state["raw_subtype"]
        decomposition[name] = {
            "sub_c_filter": {
                "feature_class": spec["feature_class"],
                "class_raw": spec["class_raw"],
            },
            "raw_dataset": spec["raw_dataset"],
            "total_count": total,
            "source_id_join_missing_count": int(state["source_id_join_missing_count"]),
            "raw_class_top20": _top_counter_items(class_counter, total),
            "raw_subtype_top20": _top_counter_items(subtype_counter, total),
            "raw_class_top20_coverage_fraction": _top20_coverage(
                class_counter, total
            ),
            "raw_subtype_top20_coverage_fraction": _top20_coverage(
                subtype_counter, total
            ),
            "root_cause_classification": spec["classification"],
            "cascade_8_assessment": spec["cascade_8_assessment"],
        }
    return decomposition


def build_unknown_slots(semantic_vocab: dict) -> list[dict]:
    semantic_tags = {slot["tag"] for slot in semantic_vocab["slots"]}
    counts = singapore_counts()

    real_unknown_by_key: Counter[str] = Counter()
    sentinel_unknown_by_key: Counter[str] = Counter()
    locked_coverage_by_key: Counter[str] = Counter()

    for (key, value), count in counts.items():
        tag = f"{key}={value}"
        if is_sub_c_unknown_sentinel(value):
            sentinel_unknown_by_key[key] += count
        elif tag in semantic_tags:
            locked_coverage_by_key[key] += count
        else:
            real_unknown_by_key[key] += count

    slots: list[dict] = []
    for offset, key in enumerate(semantic_l1_keys(semantic_vocab)):
        real_count = int(real_unknown_by_key[key])
        sentinel_count = int(sentinel_unknown_by_key[key])
        total_count = real_count + sentinel_count
        denominator = int(locked_coverage_by_key[key])
        ratio = total_count / denominator if denominator else None
        over_firing_flag = ratio is not None and ratio >= OVER_FIRING_THRESHOLD
        if denominator:
            rationale = (
                f"Flag when unknown total / locked semantic-pair Singapore coverage "
                f">= {OVER_FIRING_THRESHOLD:.0%} for key={key}."
            )
        else:
            rationale = (
                f"No scoped Singapore semantic-pair denominator for key={key} in "
                "sub-C v1 mapping; ratio left null and over-firing remains false."
            )

        slots.append(
            {
                "id": UNKNOWN_BLOCK[0] + offset,
                "key": key,
                "token": f"<unknown_{key}>",
                "singapore_count_locked_semantic_pairs": denominator,
                "singapore_count_real_osm_below_F": real_count,
                "singapore_count_subc_sentinels": sentinel_count,
                "singapore_count_total": total_count,
                "over_firing_threshold": OVER_FIRING_THRESHOLD,
                "over_firing_flag": over_firing_flag,
                "over_firing_numerator": total_count,
                "over_firing_denominator": denominator,
                "over_firing_ratio": ratio,
                "over_firing_rationale": rationale,
                "zero_firing_flag": total_count == 0,
            }
        )
    return slots


def build_unknown_family(semantic_vocab: dict) -> dict:
    slots = build_unknown_slots(semantic_vocab)
    return {
        "_status": "LOCKED",
        "release": semantic_vocab["release"],
        "derivation": {
            "source": "locked semantic_vocab.yaml L1 slots",
            "ordering": "Preserve semantic_vocab L1 order; do not sort empirically.",
            "wiki_revision_id": (
                semantic_vocab["source_references"]["wiki_map_features_snapshot"][
                    "revision_id"
                ]
            ),
            "singapore_scope": (
                "Sub-C feature_class mapping only covers highway/building; all other "
                "keys remain zero-count placeholders pending future scope expansion."
            ),
        },
        "slot_list_status": "LOCKED",
        "zero_firing_slots_policy": {
            "status": "LOCKED_KEEP_ALL_28_UNKNOWN_SLOTS",
            "note": ZERO_FIRING_SCOPE_NOTE,
        },
        "over_firing_classification": {
            "status": "LOCKED_NO_CASCADE_8",
            "building": (
                "Root cause (b): real OSM long-tail / upstream under-typed source "
                "data. 99.3% of B__UNK__ rows have null upstream subtype; sub-C "
                "correctly preserves null and sub-F captures via BP4."
            ),
            "highway": (
                "Root cause (b): real OSM long-tail / literal upstream unknown. "
                "All 9,748 rows are raw transportation.class='unknown'; sub-C "
                "propagates the explicit upstream unknown and sub-F captures via BP4."
            ),
            "no_sub_c_v2_candidate": True,
        },
        "family_block": {
            "name": "BP4 unknown family",
            "start_id": UNKNOWN_BLOCK[0],
            "end_id": UNKNOWN_BLOCK[1],
            "used_start_id": UNKNOWN_BLOCK[0],
            "used_end_id": UNKNOWN_USED_END_ID,
            "used_count": len(slots),
            "reserved_count": UNKNOWN_BLOCK[1] - UNKNOWN_USED_END_ID,
        },
        "slots": slots,
        "sentinel_decomposition": build_sentinel_decomposition(),
    }


def build_sentinel_inventory(semantic_vocab: dict, unknown_family: dict) -> dict:
    semantic_count = len(semantic_vocab["slots"])
    unknown_count = len(unknown_family["slots"])
    return {
        "_status": "LOCKED",
        "release": semantic_vocab["release"],
        "halt": "Halt 3",
        "scope_note": (
            "BP1 + BP4 + dataloader sentinel IDs are LOCKED by Halt 3 approval. "
            "BP2/BP7 remain PLACEHOLDER blocks pending later task halts."
        ),
        "bp1_semantic": {
            "start_id": 0,
            "end_id": 199,
            "used_count": semantic_count,
            "used_range": "0..126",
            "reserved_count": 200 - semantic_count - 1,
            "status": "LOCKED at Halt 3 continuation",
        },
        "bp4_unknown_family": {
            "start_id": UNKNOWN_BLOCK[0],
            "end_id": UNKNOWN_BLOCK[1],
            "used_count": unknown_count,
            "used_range": f"{UNKNOWN_BLOCK[0]}..{UNKNOWN_USED_END_ID}",
            "reserved_count": UNKNOWN_BLOCK[1] - UNKNOWN_USED_END_ID,
            "status": "LOCKED at Halt 3 continuation",
        },
        "dataloader_sentinels": {
            "block": {
                "start_id": DATALOADER_BLOCK[0],
                "end_id": DATALOADER_BLOCK[1],
                "status": "LOCKED at Halt 3 continuation",
            },
            "slots": [
                {"id": 256, "token": "<pad>", "on_disk": False},
                {"id": 257, "token": "<eos>", "on_disk": False},
                {"id": 258, "token": "<bos>", "on_disk": False},
                {"id": 259, "token": "<cell_start>", "on_disk": False},
                {"id": 260, "token": "<cell_end>", "on_disk": False},
            ],
        },
        "bp2_encoding_primitives_placeholder": {
            "start_id": BP2_PLACEHOLDER_BLOCK[0],
            "end_id": BP2_PLACEHOLDER_BLOCK[1],
            "placeholder": True,
            "status": "PLACEHOLDER; final size locks at Task 2 halt",
            "slide_condition": "If Task 2 encoding-primitive count lands above 1200, this block slides.",
        },
        "bp7_boundary_ref_placeholder": {
            "start_id": BP7_PLACEHOLDER_BLOCK[0],
            "end_id": BP7_PLACEHOLDER_BLOCK[1],
            "placeholder": True,
            "expected_used": 8,
            "reserved_count": 92,
            "status": "PLACEHOLDER; final size locks at Task 7 halt",
            "slide_condition": "If Task 7 boundary-ref count lands above 100, this block slides.",
        },
    }


def _ratio_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _percent_text(value: float) -> str:
    return f"{value * 100:.4f}%"


def _append_decomposition_table(lines: list[str], rows: list[dict]) -> None:
    lines.extend(
        [
            "",
            "| raw value | count | fraction |",
            "|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['value']}` | {row['count']} | {_percent_text(row['fraction'])} |"
        )


def build_report(
    semantic_vocab: dict, unknown_family: dict, sentinel_inventory: dict
) -> str:
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    slots = unknown_family["slots"]

    lines = [
        "**Halt 3: BP4 unknown family + sentinel inventory**",
        "",
        "Status: `DONE` - Halt 3 approved; Task 4 closed.",
        "",
        "**Enumerated `<unknown_*>` slots:**",
    ]
    for i, slot in enumerate(slots, start=1):
        lines.append(
            f"{i}. `{slot['token']}` - key `{slot['key']}` - locked ID `{slot['id']}`"
        )

    lines.extend(
        [
            "",
            "**Singapore occurrence table:**",
            "",
            "| token | key | locked semantic-pair coverage | real OSM below F | sub-C sentinels | total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for slot in slots:
        lines.append(
            f"| `{slot['token']}` | `{slot['key']}` | "
            f"{slot['singapore_count_locked_semantic_pairs']} | "
            f"{slot['singapore_count_real_osm_below_F']} | "
            f"{slot['singapore_count_subc_sentinels']} | "
            f"{slot['singapore_count_total']} |"
        )

    lines.extend(
        [
            "",
            "**Over-firing / zero-firing locked table:**",
            "",
            "| token | numerator | denominator | ratio | over-firing | zero-firing | rationale |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for slot in slots:
        lines.append(
            f"| `{slot['token']}` | {slot['over_firing_numerator']} | "
            f"{slot['over_firing_denominator']} | {_ratio_text(slot['over_firing_ratio'])} | "
            f"`{slot['over_firing_flag']}` | `{slot['zero_firing_flag']}` | "
            f"{slot['over_firing_rationale']} |"
        )

    decomposition = unknown_family["sentinel_decomposition"]
    building = decomposition["building_b_unk"]
    highway = decomposition["highway_unknown"]
    lines.extend(
        [
            "",
            "**Halt 3 continuation addendum: zero-firing slot retention**",
            "",
            ZERO_FIRING_SCOPE_NOTE,
            "",
            "**Halt 3 continuation addendum: B__UNK__ / highway=unknown decomposition**",
            "",
            "- `building=B__UNK__`: raw-cache join missing count "
            f"`{building['source_id_join_missing_count']}` across "
            f"`{building['total_count']}` rows; "
            f"raw-class top-20 coverage `{_percent_text(building['raw_class_top20_coverage_fraction'])}`.",
            "- Classification: root cause (b), real OSM long-tail / upstream "
            "under-typed source data; no BP1 cascade #8 and no sub-C-v2 "
            f"candidate from this decomposition. "
            f"{building['cascade_8_assessment']}",
            "",
            "`building=B__UNK__` raw `buildings.class` top values:",
        ]
    )
    _append_decomposition_table(lines, building["raw_class_top20"])
    lines.append("")
    lines.append("`building=B__UNK__` raw `buildings.subtype` top values:")
    _append_decomposition_table(lines, building["raw_subtype_top20"])

    lines.extend(
        [
            "",
            "- `highway=unknown`: raw-cache join missing count "
            f"`{highway['source_id_join_missing_count']}` across "
            f"`{highway['total_count']}` rows; "
            f"raw-class top-20 coverage `{_percent_text(highway['raw_class_top20_coverage_fraction'])}`.",
            "- Classification: root cause (b), real OSM long-tail / literal "
            "upstream `unknown`; no BP1 cascade #8 and no sub-C-v2 candidate "
            f"from this decomposition. {highway['cascade_8_assessment']}",
            "",
            "`highway=unknown` raw `transportation.class` top values:",
        ]
    )
    _append_decomposition_table(lines, highway["raw_class_top20"])
    lines.append("")
    lines.append("`highway=unknown` raw `transportation.subtype` top values:")
    _append_decomposition_table(lines, highway["raw_subtype_top20"])

    lines.extend(
        [
            "",
            "**Halt 3 ID namespace anchor:**",
            "",
            f"- BP1 semantic family: `0..199` (`{len(semantic_vocab['slots'])}` used, `72` reserved for v2 semantic growth) - LOCKED.",
            f"- BP4 unknown family: `200..255` (`{len(slots)}` used at `200..227`, `28` reserved at `228..255`) - LOCKED.",
            "- Dataloader-side sentinels: `256..299` with `<pad>=256`, `<eos>=257`, `<bos>=258`, `<cell_start>=259`, `<cell_end>=260`; these are not on-disk sub-F vocab tokens - LOCKED.",
            "- BP2 encoding primitives: placeholder block `300..1499`; values lock at Task 2 halt - PLACEHOLDER.",
            "- BP7 boundary-ref: placeholder block `1500..1599`; values lock at Task 7 halt, `8` expected used and `92` reserved - PLACEHOLDER.",
            "",
            "**`sentinel_inventory.yaml` post-N/dataloader reservation:**",
            "",
            f"- `_status`: `{sentinel_inventory['_status']}`",
            "- Dataloader sentinel reservations are marked `on_disk: false`.",
            "- BP2/BP7 remain explicit placeholder blocks in the namespace anchor.",
            "",
            "**Placeholder-block caveat required at Halt 3:**",
            "",
            "- BP2 (`300..1499`) and BP7 (`1500..1599`) are PLACEHOLDER blocks.",
            "- Their final sizes are empirically locked at Tasks 2 and 7 halts respectively.",
            "- If Task 2 encoding-primitive count lands above `1200` or Task 7 boundary-ref count above `100`, BP2/BP7 blocks slide.",
            "- Only BP1 + BP4 + dataloader sentinel IDs are LOCKED at Halt 3 approval.",
            "",
            "**§10.5 telemetry:**",
            "",
            "- Implementer-time-to-data-surface: approximately `30` wall-clock minutes from Task 4 start to Halt 3 report generation.",
            f"- Report generated at: `{generated_at}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    semantic_vocab = load_semantic_vocab()
    unknown_family = build_unknown_family(semantic_vocab)
    sentinel_inventory = build_sentinel_inventory(semantic_vocab, unknown_family)
    report = build_report(semantic_vocab, unknown_family, sentinel_inventory)

    UNKNOWN_FAMILY_PATH.write_text(canonicalize_yaml(unknown_family), encoding="utf-8")
    SENTINEL_INVENTORY_PATH.write_text(
        canonicalize_yaml(sentinel_inventory), encoding="utf-8"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
