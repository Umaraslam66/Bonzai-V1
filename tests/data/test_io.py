"""Crash-safety of the shared parquet writer (known_issues #18: pq.write_table is
not atomic — an in-place write truncates the prior-good file during its write
window, so a kill there leaves an unreadable parquet). write_parquet must write to
a temp on the same dir and os.replace it into place, so a failed/killed write
leaves the destination untouched and never corrupts a prior-good file.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data import io


def _table(d: dict) -> pa.Table:
    return pa.table(d)


def test_write_parquet_roundtrips(tmp_path: Path) -> None:
    dest = tmp_path / "cells.parquet"
    io.write_parquet(_table({"a": [1, 2, 3]}), dest)
    assert pq.read_table(dest).column("a").to_pylist() == [1, 2, 3]


def test_write_parquet_success_leaves_no_temp_files(tmp_path: Path) -> None:
    dest = tmp_path / "cells.parquet"
    io.write_parquet(_table({"a": [1, 2, 3]}), dest)
    # only the final file remains — no stray .tmp scratch in the dir
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cells.parquet"]


def test_write_parquet_failed_write_leaves_prior_file_intact(tmp_path, monkeypatch) -> None:
    """The discriminating crash-safety test: a write that fails mid-way must NOT
    corrupt the prior-good destination. Old in-place code writes straight to dest,
    so a partial write clobbers it; atomic temp+replace leaves dest untouched."""
    dest = tmp_path / "cells.parquet"
    io.write_parquet(_table({"a": [1]}), dest)  # prior-good artifact
    good_bytes = dest.read_bytes()

    def partial_then_raise(table, where, **kwargs):
        # Simulate a half-written file wherever the writer points, then fail.
        Path(where).write_bytes(b"PARTIAL-CORRUPT")
        raise OSError("disk full mid-write")

    monkeypatch.setattr(io.pq, "write_table", partial_then_raise)

    with pytest.raises(OSError):
        io.write_parquet(_table({"a": [2]}), dest)

    assert dest.read_bytes() == good_bytes  # prior-good intact, NOT corrupted
    # the corrupt partial must not survive as a stray temp either
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cells.parquet"]


def test_write_parquet_failed_write_no_dest_when_no_prior(tmp_path, monkeypatch) -> None:
    """With no prior file, a failed write leaves NO destination at all (not a
    truncated/half file)."""
    dest = tmp_path / "cells.parquet"

    def partial_then_raise(table, where, **kwargs):
        Path(where).write_bytes(b"PARTIAL-CORRUPT")
        raise OSError("disk full mid-write")

    monkeypatch.setattr(io.pq, "write_table", partial_then_raise)

    with pytest.raises(OSError):
        io.write_parquet(_table({"a": [2]}), dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []  # no stray temp left behind
