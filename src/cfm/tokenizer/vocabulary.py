from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from cfm.tokenizer.errors import LoaderError

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
    # Iterate feature_class groups in YAML order. Each group is either a flat
    # list (Phase 0 shape) or a dict with a `tokens` key plus metadata (Phase 1).
    for group_name, group_value in fc.items():
        if isinstance(group_value, list):
            tokens = group_value
        elif isinstance(group_value, dict) and "tokens" in group_value:
            tokens = group_value["tokens"]
        else:
            value_type = type(group_value).__name__
            raise LoaderError(
                f"feature_class.{group_name} has unexpected shape "
                f"(expected list-of-strings or dict-with-tokens-key); got {value_type}"
            )
        _validate_section_tokens(group_name, tokens)
        out.extend(tokens)
    axis_count = int(data["anchor"]["axis_count"])
    out.extend(f"ANCHOR_X_{i}" for i in range(axis_count))
    out.extend(f"ANCHOR_Y_{i}" for i in range(axis_count))
    move = data["move"]
    for direction in move["directions"]:
        for step in move["steps_m"]:
            out.append(f"MOVE_{direction}_{step}")
    # Global duplicate-name check across all sections.
    if len(set(out)) != len(out):
        from collections import Counter

        dupes = [name for name, count in Counter(out).items() if count > 1]
        raise LoaderError(f"duplicate token name(s) across sections: {dupes}")
    return out, axis_count


def _validate_section_tokens(group_name: str, tokens: list[str]) -> None:
    """Enforce convention: any token containing `__UNK__` must be at section index 0.

    The double-underscore marker is the reserved placeholder for missing-value
    handling. Data-derived tokens (e.g. `R_unknown` from Overture's
    transportation.class category "unknown") use the bare suffix and are
    allowed at any position.
    """
    for i, name in enumerate(tokens):
        if "__UNK__" in name and i != 0:
            raise LoaderError(
                f"feature_class.{group_name}: token {name!r} contains '__UNK__' "
                f"but is at position {i}; must be at position 0 within its section"
            )
