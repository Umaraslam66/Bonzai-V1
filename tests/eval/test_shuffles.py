from __future__ import annotations

from dataclasses import dataclass

from cfm.eval.shuffles import (
    ShuffleStrategy,
    TileConditioning,
    shuffle_macro_plans,
)


@dataclass(frozen=True)
class _FakeMacro:
    tile_i: int
    tile_j: int


def _candidates(n: int) -> list[tuple[TileConditioning, _FakeMacro]]:
    out: list[tuple[TileConditioning, _FakeMacro]] = []
    for k in range(n):
        bucket = "tropical_rainforest" if k % 2 == 0 else "temperate"
        out.append(
            (
                TileConditioning(
                    country="SG",
                    climate_zone=bucket,
                    morphology_class="Asian-megacity",
                    era_class="contemporary",
                ),
                _FakeMacro(tile_i=k, tile_j=0),
            )
        )
    return out


def test_within_bucket_shuffle_only_swaps_within_matching_conditioning() -> None:
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    shuffled = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.WITHIN_BUCKET,
        seed=42,
    )
    for tc, macro in zip(targets, shuffled, strict=True):
        # Find the candidate whose macro matches `macro`. Assert its
        # conditioning matches `tc`.
        match = next(c for c in cands if c[1] is macro)
        assert match[0].climate_zone == tc.climate_zone


def test_cross_tile_shuffle_uniformly_random_with_seed() -> None:
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    a = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    b = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    assert a == b, "deterministic given same seed"


def test_cross_tile_shuffle_does_not_return_identity() -> None:
    """With 20 candidates and a deterministic seed, the shuffle should almost
    always produce a permutation that differs in at least one position from
    identity. Asserted as a deterministic-fixture property, not a property
    test."""
    cands = _candidates(20)
    targets = [c[0] for c in cands]
    shuffled = shuffle_macro_plans(
        targets=targets,
        candidates=cands,
        strategy=ShuffleStrategy.CROSS_TILE,
        seed=42,
    )
    identity = [c[1] for c in cands]
    assert shuffled != identity
