from __future__ import annotations

import typing

from cfm.data.training.shard_schema import CellPayload, TrainingShard


def test_shard_carries_full_tile_structure_not_slice_subset():
    """§10.1: the tier-1 format provisions macro tokens + per-cell boundary
    contracts even though the cell-unit slice does not READ them — so the
    bake-off's tile-AR / hierarchical candidates can."""
    fields = TrainingShard.__dataclass_fields__
    assert "tile_conditioning" in fields  # tile-level labels (slice reads)
    assert "macro_tokens" in fields  # candidate 1/2 read these (slice does not)
    assert "cells" in fields
    assert "lineage" in fields  # frozenset[TileRef] | None (G-F4 fail-closed)

    cell_fields = CellPayload.__dataclass_fields__
    assert "tokens" in cell_fields
    assert "cell_density_bucket" in cell_fields  # per-cell scalar (trigger-2 granularity)
    assert "boundary_contracts" in cell_fields  # candidate 2 reads these (slice does not)


def test_lineage_is_optional_so_absence_is_representable():
    """G-F4 requires that 'absent lineage' is a real None, not a synthesized value."""
    hint = typing.get_type_hints(TrainingShard)["lineage"]
    assert type(None) in typing.get_args(hint)  # frozenset[TileRef] | None
