from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

TokenId = int


@dataclass(frozen=True)
class Vocabulary:
    """An ordered, immutable token vocabulary.

    `id_to_token[i]` is the token name at id `i`.
    `token_to_id[name]` is the inverse lookup.
    `anchor_axis_count` is the per-axis anchor token count (matches the largest supported
    `cell_size_m`).
    """

    id_to_token: tuple[str, ...]
    token_to_id: Mapping[str, TokenId]
    anchor_axis_count: int

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names, anchor_axis_count = _flatten(data)
        token_to_id = MappingProxyType({name: i for i, name in enumerate(names)})
        return cls(
            id_to_token=tuple(names),
            token_to_id=token_to_id,
            anchor_axis_count=anchor_axis_count,
        )


def _flatten(data: dict) -> tuple[list[str], int]:
    out: list[str] = []
    out.extend(data["control"])
    out.extend(data["hierarchy"])
    fc = data["feature_class"]
    for group in ("road", "building", "poi", "land_use"):
        out.extend(fc[group])
    axis_count = int(data["anchor"]["axis_count"])
    out.extend(f"ANCHOR_X_{i}" for i in range(axis_count))
    out.extend(f"ANCHOR_Y_{i}" for i in range(axis_count))
    move = data["move"]
    for direction in move["directions"]:
        for step in move["steps_m"]:
            out.append(f"MOVE_{direction}_{step}")
    return out, axis_count
