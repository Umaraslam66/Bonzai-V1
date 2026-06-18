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
