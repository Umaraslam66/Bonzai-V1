"""Macro-plan shuffle strategies for the conditional-perplexity gap.

Two strategies for de-risk:

- WITHIN_BUCKET (primary): substitute macro plan from a tile with matching
  tile-level conditioning (country, climate_zone, morphology_class, era_class).
- CROSS_TILE (secondary sanity): substitute macro plan from any random
  candidate.

Position-shuffled within the same plan is deferred to post-reset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar


class ShuffleStrategy(StrEnum):
    WITHIN_BUCKET = "within_bucket"
    CROSS_TILE = "cross_tile"


@dataclass(frozen=True)
class TileConditioning:
    country: str
    climate_zone: str
    morphology_class: str
    era_class: str


MacroT = TypeVar("MacroT")


def _bucket_key(tc: TileConditioning) -> tuple[str, str, str, str]:
    return (tc.country, tc.climate_zone, tc.morphology_class, tc.era_class)


def shuffle_macro_plans(
    *,
    targets: list[TileConditioning],
    candidates: list[tuple[TileConditioning, MacroT]],
    strategy: ShuffleStrategy,
    seed: int,
) -> list[MacroT]:
    """Return one shuffled macro plan per target.

    Deterministic given (targets, candidates, strategy, seed).
    """
    rng = random.Random(seed)

    if strategy is ShuffleStrategy.WITHIN_BUCKET:
        buckets: dict[tuple[str, str, str, str], list[MacroT]] = {}
        for tc, macro in candidates:
            buckets.setdefault(_bucket_key(tc), []).append(macro)
        # Stable order within bucket: candidates already in deterministic input order.
        out: list[MacroT] = []
        for tc in targets:
            pool = buckets.get(_bucket_key(tc), [])
            if not pool:
                raise ValueError(f"no candidates in within-bucket pool for {_bucket_key(tc)}")
            out.append(rng.choice(pool))
        return out

    if strategy is ShuffleStrategy.CROSS_TILE:
        pool = [m for _, m in candidates]
        return [rng.choice(pool) for _ in targets]

    raise AssertionError(f"unknown strategy: {strategy}")
