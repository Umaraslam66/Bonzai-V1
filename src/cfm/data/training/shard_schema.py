"""Tier-1 locked per-tile training-shard schema (spec §5).

The format provisions the FULL tile structure even though the cell-unit slice
reads only `cells[*].tokens`, `cells[*].cell_density_bucket`,
`cells[*].character_stats`, and `tile_conditioning`. `macro_tokens` and
`cells[*].boundary_contracts` are carried so the bake-off's tile-AR (candidate 1)
and hierarchical (candidate 2) runs read the SAME shards — under-provisioning the
write-once format to the slice's subset is the fatal §10.1 direction.

`lineage` is `frozenset[TileRef] | None`: None must be representable so a shard
with absent lineage reaches the holdout audit as a genuine None (G-F4 fail-closed),
never a path-synthesized value.
"""

from __future__ import annotations

from dataclasses import dataclass

#: (region, tile_i, tile_j) — identical to cfm.eval.holdout.lineage_audit.TileRef
TileRef = tuple[str, int, int]


@dataclass(frozen=True)
class CellPayload:
    """One cell's payload within a tile shard."""

    cell_i: int
    cell_j: int
    cell_slot_index: int  # == cell_i*8 + cell_j (sub-F io.py invariant)
    tokens: tuple[int, ...]  # the cell's sub-F token sequence (slice reads this)
    cell_density_bucket: int | None  # per-cell scalar (trigger-2 granularity; slice reads this)
    boundary_contracts: tuple[int, ...]  # provisioned for candidate 2; slice does NOT read
    #: Task 24b (spec §8 + mini-spec §1): the 7 fixed-log10 character channels derived
    #: from sub-C features.parquet at shard-build time (slice reads this — it feeds the
    #: continuous conditioning prefix position). REQUIRED, no default: a shard built
    #: before 24b cannot construct this schema — version-skew fails LOUD at read,
    #: never a silent zero-vector (fail-closed by construction).
    character_stats: tuple[float, ...]


@dataclass(frozen=True)
class TrainingShard:
    """Tier-1 locked per-tile training shard — FULL tile structure."""

    region: str
    tile_i: int
    tile_j: int
    tile_conditioning: dict  # the locked conditioning schema (Task 3); slice reads this
    macro_tokens: tuple[int, ...]  # provisioned for candidates 1/2; slice does NOT read
    cells: tuple[CellPayload, ...]
    lineage: frozenset[TileRef] | None  # None = untracked -> G-F4 fail-closed
