"""Local (torch-free) unit tests for the 4-GPU eval-sharding core (``cfm.eval.shard``).

These run on a laptop during the $WORK outage — no CUDA, no mamba. They pin the STRUCTURAL
property the GPU equivalence golden rests on: every held-out cell is partitioned to exactly one
rank and reassembled exactly once (no drop, no double-count), ragged-safe; and the gather is
shard-order-independent (byte-deterministic). The held-out cities 523/579/156/601 are exercised
directly — 3 of the 4 are NOT divisible by 4, so the ragged boundary is covered.
"""

from __future__ import annotations

import pytest

from cfm.eval.shard import assert_conservation, gather_in_order, partition_indices

# The real scored-eval per-city workloads (glasgow/eisenhuttenstadt/munich/krakow) + edges.
CITY_COUNTS = [523, 579, 156, 601]
WORLD = 4


@pytest.mark.parametrize("n", [*CITY_COUNTS, 1859, 0, 1, 3, 4, 8, 17])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_partition_is_balanced_and_conservative(n: int, world_size: int) -> None:
    shards = partition_indices(n, world_size)
    assert len(shards) == world_size
    flat = [i for shard in shards for i in shard]
    # conservation: union == range(n), no duplicates
    assert sorted(flat) == list(range(n))
    assert len(flat) == len(set(flat)) == n
    # balance: sizes differ by at most 1 (ragged-safe)
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    # the structural guard agrees
    assert_conservation(shards, n)


def test_ragged_city_no_boundary_cell_lost() -> None:
    # 523 is NOT divisible by 4: the explicit ragged case the golden must include.
    shards = partition_indices(523, WORLD)
    assert [len(s) for s in shards] == [131, 131, 131, 130]
    # the very last cell (522) is owned by exactly one rank, and 0 is too
    owners_of_522 = [r for r, s in enumerate(shards) if 522 in s]
    owners_of_0 = [r for r, s in enumerate(shards) if 0 in s]
    assert owners_of_522 == [522 % WORLD]
    assert owners_of_0 == [0]
    assert_conservation(shards, 523)


def test_assert_conservation_raises_on_drop() -> None:
    shards = partition_indices(523, WORLD)
    shards[3].pop()  # drop one boundary cell
    with pytest.raises(ValueError, match="dropped"):
        assert_conservation(shards, 523)


def test_assert_conservation_raises_on_double_count() -> None:
    shards = partition_indices(100, WORLD)
    shards[1].append(shards[0][0])  # same cell now owned by two ranks
    with pytest.raises(ValueError, match="double-counted"):
        assert_conservation(shards, 100)


def test_assert_conservation_raises_on_out_of_range() -> None:
    shards = partition_indices(10, WORLD)
    shards[0].append(10)  # index == n_items is out of range
    with pytest.raises(ValueError, match="out of range"):
        assert_conservation(shards, 10)


def _scored(per_rank_indices: list[list[int]]) -> list[list[tuple[int, str]]]:
    """Simulate each rank returning (global_index, a per-cell result) for its shard."""
    return [[(i, f"score-{i}") for i in shard] for shard in per_rank_indices]


def test_gather_reassembles_in_canonical_order() -> None:
    shards = partition_indices(523, WORLD)
    merged = gather_in_order(_scored(shards), 523)
    assert merged == [f"score-{i}" for i in range(523)]


def test_gather_is_shard_order_independent() -> None:
    # Byte-determinism: the gathered sequence must not depend on the order ranks report in.
    shards = partition_indices(579, WORLD)
    scored = _scored(shards)
    canonical = gather_in_order(scored, 579)
    for perm in ([3, 1, 0, 2], list(reversed(range(WORLD))), [2, 0, 3, 1]):
        reordered = [scored[r] for r in perm]
        assert gather_in_order(reordered, 579) == canonical


def test_gather_raises_on_missing_cell() -> None:
    shards = partition_indices(601, WORLD)
    scored = _scored(shards)
    scored[2] = scored[2][:-1]  # one rank fails to report its last cell
    with pytest.raises(ValueError, match="never reported"):
        gather_in_order(scored, 601)


def test_gather_raises_on_duplicate_cell() -> None:
    shards = partition_indices(156, WORLD)
    scored = _scored(shards)
    scored[0].append(scored[1][0])  # two ranks report the same global index
    with pytest.raises(ValueError, match="double-counted"):
        gather_in_order(scored, 156)


def test_gather_preserves_none_results() -> None:
    # A legitimate ``None`` per-cell result must survive the merge (not be read as "missing").
    scored = [[(0, None)], [(1, None)]]
    assert gather_in_order(scored, 2) == [None, None]


def test_partition_rejects_bad_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        partition_indices(10, 0)
