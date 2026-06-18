"""4-GPU eval-sharding: pure partition + count-conservation + canonical gather.

The scored bake-off eval generates ~1,859 held-out cells PER run (523/579/156/601 over the 4
held-out EU cities). Post-train eval ran on rank 0 only (1 GPU works, the node's other 3 GPUs
are allocated-and-billed but idle → 4x waste). Sharding the cells across the node's `world_size`
GPUs recovers that ~4x.

This module is the **torch-free core** (so it builds and unit-tests on a laptop with no CUDA /
no mamba, during the $WORK outage): the cell-partition, the count-conservation guard, and the
gather/merge that returns per-cell results in canonical global-cell order (so the assembled
sequence — and therefore every downstream score and the worst-case-city verdict — is
independent of the order ranks happen to report in, i.e. byte-deterministic across re-runs).

The thin distributed wrapper that performs the actual ``all_gather_object`` lives at the eval
call site (it imports ``torch.distributed`` lazily); it delegates the merge + the conservation
guard to :func:`gather_in_order` here. The two TEETH of the GPU equivalence golden (per-cell
scores bit-identical to the rank-0 baseline; count-conservation on the REAL distributed run incl.
a ragged-partition city) are deferred to $WORK recovery; the structural property they rest on
(no cell dropped or double-counted, ragged-safe) is what the local tests here pin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

#: Sentinel for an unfilled gather slot (distinct from any legitimate ``None`` result).
_MISSING = object()


def partition_indices(n_items: int, world_size: int) -> list[list[int]]:
    """Partition ``range(n_items)`` into ``world_size`` disjoint, balanced shards (round-robin).

    ``shards[r]`` is the ascending list of global cell indices owned by rank ``r``. Round-robin
    (``range(r, n_items, world_size)``) is ragged-SAFE by construction: when ``n_items`` is not
    divisible by ``world_size`` the first ``n_items % world_size`` ranks simply get one extra
    item, so shard sizes differ by at most 1 and **every index 0..n_items-1 lands in exactly one
    shard** — no boundary cell dropped or double-counted (the property the eval golden re-checks
    on the real distributed run). Returns ``world_size`` lists; some may be empty when
    ``n_items < world_size``.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    return [list(range(r, n_items, world_size)) for r in range(world_size)]


def assert_conservation(shards: list[list[int]], n_items: int) -> None:
    """Raise ``ValueError`` unless ``shards`` is an exact partition of ``range(n_items)``.

    Every index in ``[0, n_items)`` must appear in exactly one shard: no out-of-range index, no
    index assigned to two ranks (double-count), and none missing (drop). This is the structural
    half of the eval golden's tooth #2 — aggregate-score equality alone would NOT catch a dropped
    or double-counted boundary cell, so the count is checked explicitly.
    """
    owner: dict[int, int] = {}
    for r, shard in enumerate(shards):
        for i in shard:
            if not (0 <= i < n_items):
                raise ValueError(
                    f"shard conservation: index {i} (rank {r}) out of range [0, {n_items})"
                )
            if i in owner:
                raise ValueError(
                    f"shard conservation: index {i} double-counted (ranks {owner[i]} and {r})"
                )
            owner[i] = r
    missing = [i for i in range(n_items) if i not in owner]
    if missing:
        raise ValueError(
            f"shard conservation: {len(missing)} index(es) dropped (none assigned), "
            f"e.g. {missing[:8]}"
        )


def gather_in_order(per_rank_results: list[list[tuple[int, T]]], n_items: int) -> list[T]:
    """Merge per-rank ``[(global_index, result), ...]`` into one list ordered by global index.

    Each rank contributes the results for the cells it owned, each tagged with the cell's GLOBAL
    index. The merge places every result in its global slot, so the returned list is in canonical
    cell order **regardless of the order ranks are gathered in** — making the assembled sequence
    (and every downstream per-cell score + the worst-case-city verdict) byte-deterministic across
    re-runs. Enforces conservation: every index ``0..n_items-1`` must be filled exactly once
    (raises on out-of-range, double-count, or drop).
    """
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    slots: list[object] = [_MISSING] * n_items
    for shard_results in per_rank_results:
        for idx, result in shard_results:
            if not (0 <= idx < n_items):
                raise ValueError(f"gather: index {idx} out of range [0, {n_items})")
            if slots[idx] is not _MISSING:
                raise ValueError(f"gather: index {idx} double-counted across shards")
            slots[idx] = result
    missing = [i for i, v in enumerate(slots) if v is _MISSING]
    if missing:
        raise ValueError(f"gather: {len(missing)} cell(s) never reported, e.g. {missing[:8]}")
    return [s for s in slots]  # type: ignore[misc]  # all slots filled (checked above)


def indices_for_rank(n_items: int, rank: int, world_size: int) -> list[int]:
    """The global cell indices owned by ``rank`` (the round-robin shard ``rank``). Equivalent to
    ``partition_indices(n_items, world_size)[rank]`` but without materializing the other shards."""
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    return list(range(rank, n_items, world_size))


def sharded_eval(
    n_items: int,
    score_one: Callable[[int], T],
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[T]:
    """Run ``score_one(global_index)`` over all ``n_items`` cells, SHARDED across the active
    ``torch.distributed`` group, and return the per-cell results in canonical global order on
    EVERY rank.

    Each rank scores only ``indices_for_rank(...)`` (so the node's 4 GPUs share the work, no
    rank-0-only idle), then ``all_gather_object`` exchanges per-cell ``(global_index, result)``
    pairs and :func:`gather_in_order` merges + enforces count-conservation (every cell exactly
    once — the ragged-safe property the golden re-checks on the real run). Because the merge is
    keyed on the GLOBAL index, the assembled sequence is identical regardless of how ranks
    interleave — so downstream scores and the worst-case-city verdict are byte-deterministic.

    ``torch.distributed`` is imported LAZILY here so this module stays torch-free for the local
    unit tests; ``rank``/``world_size`` default to the live process group. ``score_one`` must be
    keyed on the GLOBAL index (e.g. ``seed = base + i``) so a cell's result does not depend on
    which rank computed it — exactly what golden tooth #1 (bit-identity vs rank-0) verifies."""
    import torch.distributed as dist

    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()
    local: list[tuple[int, T]] = [
        (i, score_one(i)) for i in indices_for_rank(n_items, rank, world_size)
    ]
    gathered: list[list[tuple[int, T]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    # every rank contributed (no None left); flatten None-typed slots away for the merge.
    return gather_in_order([g for g in gathered if g is not None], n_items)
