"""Held-out cache round-trip (spec §5 / cache opt): cached inputs must be byte-identical
to a fresh load, so the optimization can NEVER silently change the gap inputs.

The held-out cells are checkpoint-independent, so a byte-identical write->read round-trip is
the cached≡uncached guarantee (the gap is a deterministic function of cells + model).
"""

from __future__ import annotations

from cfm.eval.standing.heldout_cells import HeldoutCell, read_heldout_cache, write_heldout_cache


def _cells():
    return [
        HeldoutCell(
            region="glasgow",
            body_tokens=[1, 2, 3],
            own_prefix=[300, 301, 0],
            own_char=[1.5, 0.0, 2.25, 3.1, 4.0, 1.0, 1.0],
            donor_prefix=[302, 303, 0],
            donor_char=[2.5, 1.0, 0.0, 1.1, 2.0, 1.0, 0.0],
        ),
        HeldoutCell(
            region="munich",
            body_tokens=[4, 5],
            own_prefix=[304, 0],
            own_char=[0.1] * 7,
            donor_prefix=[305, 0],
            donor_char=[0.2] * 7,
        ),
    ]


def test_cache_round_trip_byte_identical(tmp_path):
    cells = _cells()
    p = tmp_path / "cache.json"
    write_heldout_cache(cells, p)
    assert read_heldout_cache(p) == cells  # cached inputs == fresh inputs, field-for-field


def test_cache_preserves_float_precision(tmp_path):
    cells = [HeldoutCell("g", [1], [0], [1.7817553746524688] + [0.0] * 6, [0], [0.0] * 7)]
    p = tmp_path / "c.json"
    write_heldout_cache(cells, p)
    assert read_heldout_cache(p)[0].own_char[0] == 1.7817553746524688


def test_cache_round_trip_types_are_exact(tmp_path):
    cells = _cells()
    p = tmp_path / "cache.json"
    write_heldout_cache(cells, p)
    back = read_heldout_cache(p)
    assert all(isinstance(c, HeldoutCell) for c in back)
    assert all(isinstance(t, int) for t in back[0].body_tokens)
    assert all(isinstance(x, float) for x in back[0].own_char)
