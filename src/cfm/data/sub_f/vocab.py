"""Sub-F unified vocab loader.

Loads BP1 semantic + BP2 encoding-primitive + structural sentinels +
BP4 unknown + BP7 boundary-reference slots from their respective YAML
configs and exposes a deterministic, ascending-id tuple.

Dataloader sentinels <pad>/<eos>/<bos>/<cell_start> (IDs 256-259) are on_disk=false
per `configs/sub_f/sentinel_inventory.yaml` and excluded. <cell_end> (260) is
on_disk=true (cell-EOS): the cell-level stop signal, family "terminator", loaded by
`_load_cell_end_slot`.

Structural sentinels (<feature> id 509, <feature_end> id 510) live at the
front of BP2 reserved_v2_headroom per the T8 plan-write sentinel-inventory
fix (2026-05-28). They are tagged family="structural" (grammar primitives),
distinct from the encoding_primitive family of anchor/direction/magnitude
sub-blocks whose ID neighborhood they share. See spec §13.1 "T8 plan-write
-> BP2 inventory" row for the audit trail.

Iteration order is YAML file order, NOT dict hash order - per
`feedback_pythonhashseed_dict_iteration_test` discipline (sub-F T5b Test 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import yaml

_CONFIGS = Path(__file__).resolve().parents[4] / "configs" / "sub_f"

Family = Literal[
    "semantic", "unknown", "encoding_primitive", "structural", "terminator", "boundary_reference"
]

#: The cell-level terminator <cell_end> token id (cell-EOS). ONE source — build_shards
#: appends it at non-empty cell-end and generate.py breaks on it, both via this name.
CELL_END_TOKEN_ID: Final[int] = 260


@dataclass(frozen=True)
class VocabSlot:
    """One on-disk vocab slot."""

    token_id: int
    tag: str
    family: Family


def _load_semantic_slots() -> list[VocabSlot]:
    """BP1 semantic slots from `configs/sub_f/semantic_vocab.yaml`.

    Slots carry explicit `id` and `tag` fields; we trust YAML file order.
    """
    data = yaml.safe_load((_CONFIGS / "semantic_vocab.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["tag"]), family="semantic")
        for s in data["slots"]
    ]


def _load_unknown_slots() -> list[VocabSlot]:
    """BP4 <unknown_*> slots from `configs/sub_f/unknown_family.yaml`.

    Slots carry a `key` field (the BP1 L1 key, e.g. "aerialway"); the
    on-disk tag is `<unknown_${key}>` per spec §2.3 + §2.4 naming.
    """
    data = yaml.safe_load((_CONFIGS / "unknown_family.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(
            token_id=int(s["id"]),
            tag=f"<unknown_{s['key']}>",
            family="unknown",
        )
        for s in data["slots"]
    ]


@lru_cache(maxsize=1)
def unknown_family_tag_to_key() -> dict[str, str]:
    """{<unknown_${key}>: key} for the BP4 unknown family.

    The authority for resolving an <unknown_*> token back to its BP1 L1 key
    (e.g. <unknown_highway> -> "highway", a road). The key lives in
    unknown_family.yaml, NOT in the tag string; this mirrors
    _load_unknown_slots' `<unknown_${key}>` tag construction so the resolver and
    the tag can never drift.
    """
    data = yaml.safe_load((_CONFIGS / "unknown_family.yaml").read_text(encoding="utf-8"))
    return {f"<unknown_{s['key']}>": str(s["key"]) for s in data["slots"]}


ROAD_L1_KEY = "highway"  # the BP1 L1 key denoting a road (LineString) feature


def semantic_tag_to_l1_key(semantic_tag: str) -> str:
    """The single authority for a feature's BP1 L1 key (road == ``ROAD_L1_KEY``).

    `<key=value>` tags carry the key inline (`<highway=residential>` -> "highway").
    BP4 `<unknown_${key}>` tags carry the key in the unknown-family vocab, NOT the
    tag string, so resolve those via `unknown_family_tag_to_key`
    (`<unknown_highway>` -> "highway"). Used by BOTH the encoder's bref-emission
    gate (encode_cell) and the validator's non-road / road-cell legs so neither
    re-determines road-ness with a local parse — the bug class behind sub-G T11
    cycles 3 (validator) and 4 (encoder).
    """
    if "=" in semantic_tag:
        return semantic_tag.split("=", 1)[0].lstrip("<")
    return unknown_family_tag_to_key().get(semantic_tag, semantic_tag)


def _load_encoding_primitive_slots() -> list[VocabSlot]:
    """BP2 encoding-primitive slots from `configs/sub_f/sentinel_inventory.yaml`
    `bp2_encoding_primitives.sub_blocks` ranges (anchor + direction + magnitude).

    Each slot's tag is synthesised from its sub-block + offset:
      - anchor:    `<anchor_${start_id_offset}>`   (96 slots, ids 300-395)
      - direction: `<direction_${idx}>`            (360 slots, ids 511-870; Halt-2
                   revisit 2026-05-29 widened 48->360 + relocated from 396-443)
      - magnitude: `<magnitude_${idx}>`            (65 slots, ids 444-508)
    Only ("anchor", "direction", "magnitude") are emitted; the retired
    "direction_v1_deprecated" block (396-443) is NOT iterated.
    """
    inv = yaml.safe_load((_CONFIGS / "sentinel_inventory.yaml").read_text(encoding="utf-8"))
    bp2 = inv["bp2_encoding_primitives"]["sub_blocks"]

    slots: list[VocabSlot] = []
    for block_name in ("anchor", "direction", "magnitude"):
        block = bp2[block_name]
        start, end = int(block["start_id"]), int(block["end_id"])
        for offset, token_id in enumerate(range(start, end + 1)):
            slots.append(
                VocabSlot(
                    token_id=token_id,
                    tag=f"<{block_name}_{offset}>",
                    family="encoding_primitive",
                )
            )
    return slots


def _load_structural_sentinel_slots() -> list[VocabSlot]:
    """Structural sentinels consumed from BP2 reserved_v2_headroom front (T8
    plan-write fix, 2026-05-28). IDs 509 (<feature>), 510 (<feature_end>).

    These tokens are grammar primitives (delimit per-feature sequences); the
    encoder's 4-case grammar (§3.2) opens every feature with <feature> and
    closes with <feature_end>. They are tagged family="structural" rather
    than "encoding_primitive" because semantically they are NOT value tokens
    of coordinate/direction/magnitude classes - their ID neighborhood
    (consumed from BP2 reserved tail) is incidental.
    """
    inv = yaml.safe_load((_CONFIGS / "sentinel_inventory.yaml").read_text(encoding="utf-8"))
    consumed = inv["bp2_encoding_primitives"]["consumed_from_reserved_v2_headroom"]["slots"]
    return [
        VocabSlot(
            token_id=int(s["id"]),
            tag=str(s["token"]),
            family="structural",
        )
        for s in consumed
    ]


def _load_boundary_reference_slots() -> list[VocabSlot]:
    """BP7 boundary-reference slots from `configs/sub_f/boundary_reference_vocab.yaml`."""
    data = yaml.safe_load((_CONFIGS / "boundary_reference_vocab.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["tag"]), family="boundary_reference")
        for s in data["slots"]
    ]


def _load_cell_end_slot() -> list[VocabSlot]:
    """The cell-level terminator <cell_end> (id 260) — the one dataloader sentinel
    flipped on_disk=true (cell-EOS). It is a STOP signal, NOT a feature-grammar
    primitive like <feature>/<feature_end> (509/510), so it gets its OWN family
    "terminator": this keeps the structural-range invariant ({509,510}) honest rather
    than widening it to admit 260 (which would silently admit 261-508).

    The dataloader_sentinels block is otherwise unread by the loader, so flipping
    on_disk in the YAML is inert without this loader. Emits a slot for every sentinel
    with on_disk=true — after the flip, exactly one: 260.
    """
    inv = yaml.safe_load((_CONFIGS / "sentinel_inventory.yaml").read_text(encoding="utf-8"))
    return [
        VocabSlot(token_id=int(s["id"]), tag=str(s["token"]), family="terminator")
        for s in inv["dataloader_sentinels"]["slots"]
        if s.get("on_disk", False)
    ]


@lru_cache(maxsize=1)
def load_sub_f_vocab() -> tuple[VocabSlot, ...]:
    """Return all on-disk sub-F slots in strictly ascending token_id order.

    Excludes dataloader sentinels 256-259 (on_disk=false); <cell_end> 260 is
    on_disk=true (family=terminator, cell-EOS) and IS included.
    Cached at module level - same tuple returned on every call.
    """
    all_slots = (
        _load_semantic_slots()
        + _load_unknown_slots()
        + _load_encoding_primitive_slots()
        + _load_structural_sentinel_slots()
        + _load_cell_end_slot()
        + _load_boundary_reference_slots()
    )
    return tuple(sorted(all_slots, key=lambda s: s.token_id))


@lru_cache(maxsize=1)
def vocab_tag_to_id() -> dict[str, int]:
    """tag -> token_id lookup, derived from load_sub_f_vocab()."""
    return {s.tag: s.token_id for s in load_sub_f_vocab()}


# Total on-disk vocab count, used by tests + downstream checks.
# Halt-2 revisit 2026-05-29: direction 48->360 grew BP2 encoding_primitive 209->521
# (96 anchor + 360 direction + 65 magnitude); retired direction_v1_deprecated (48) is
# NOT emitted. Total on-disk: BP1 127 + BP4 28 + BP2 521 + structural 2 + terminator 1
# (cell-EOS <cell_end> 260) + BP7 8 = 687.
SUB_F_ON_DISK_TOTAL: Final[int] = 687
