"""Tests for the guarded re-derive safety guards (lock / compare / atomic-swap /
halt-on-content-change). These pin the corpus-integrity guarantees added after
the 2026-06-05 double-nohup near-miss. They exercise the real guard functions
with synthetic dirs (compare uses raw sha256, so byte files suffice — no real
derive needed), plus one monkeypatched-derive integration test for the
halt-before-swap + live-untouched guarantee.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.multiregion import guarded_rederive as gr


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# --- Guard 1: lockfile -------------------------------------------------------


def test_lock_blocks_second_concurrent_invocation(tmp_path: Path) -> None:
    lock = tmp_path / ".rederive.lock"
    fh = gr.acquire_lock(lock)
    try:
        with pytest.raises(gr.ConcurrentRederiveError):
            gr.acquire_lock(lock)  # second holder while first is alive -> refused
    finally:
        fh.close()


def test_lock_reacquirable_after_release(tmp_path: Path) -> None:
    lock = tmp_path / ".rederive.lock"
    gr.acquire_lock(lock).close()  # acquire + release
    gr.acquire_lock(lock).close()  # now free again


# --- Guard 1b: PER-CITY lock (enables corpus-wide parallel fan-out) -----------
# The global lock prevented ALL concurrency; the safety property only needs
# "no two concurrent re-derives of the SAME city" (each city writes its own
# per-city temp/live/.bak dirs — no shared mutable state). Per-city locking lets
# different cities re-derive in parallel while preserving same-city protection.


def test_city_rederive_lock_path_distinct_per_city(tmp_path: Path) -> None:
    a = gr.city_rederive_lock_path(tmp_path, "amsterdam")
    b = gr.city_rederive_lock_path(tmp_path, "rotterdam")
    assert a != b
    assert "amsterdam" in a.name
    assert "rotterdam" in b.name


def test_two_different_city_locks_held_simultaneously(tmp_path: Path) -> None:
    # The whole point of the per-city change: different cities re-derive
    # concurrently. Under the OLD global lock this raised ConcurrentRederiveError.
    la = gr.acquire_lock(gr.city_rederive_lock_path(tmp_path, "amsterdam"))
    try:
        lb = gr.acquire_lock(gr.city_rederive_lock_path(tmp_path, "rotterdam"))
        lb.close()  # both acquired -> no cross-city block
    finally:
        la.close()


def test_same_city_second_acquire_refuses(tmp_path: Path) -> None:
    p = gr.city_rederive_lock_path(tmp_path, "amsterdam")
    fh = gr.acquire_lock(p)
    try:
        with pytest.raises(gr.ConcurrentRederiveError):
            gr.acquire_lock(p)  # same city, still held -> refused
    finally:
        fh.close()


# --- Guard 3: byte-identity comparison (gates the swap) ----------------------


def test_compare_detects_byte_difference(tmp_path: Path) -> None:
    tmp_r, live_r = tmp_path / "tmp", tmp_path / "live"
    _write(tmp_r / "tile=A" / "cells.parquet", b"NEW")
    _write(live_r / "tile=A" / "cells.parquet", b"OLD")
    assert gr.compare_cells(tmp_r, live_r) == ["tile=A"]


def test_compare_identical_is_empty(tmp_path: Path) -> None:
    tmp_r, live_r = tmp_path / "tmp", tmp_path / "live"
    for r in (tmp_r, live_r):
        _write(r / "tile=A" / "cells.parquet", b"SAME")
    assert gr.compare_cells(tmp_r, live_r) == []


def test_compare_new_tile_absent_in_live_is_not_a_difference(tmp_path: Path) -> None:
    tmp_r, live_r = tmp_path / "tmp", tmp_path / "live"
    _write(tmp_r / "tile=NEW" / "cells.parquet", b"X")  # no live counterpart
    assert gr.compare_cells(tmp_r, live_r) == []


# --- Guard 2: atomic swap ----------------------------------------------------


def test_atomic_swap_replaces_live_and_cleans_up(tmp_path: Path) -> None:
    tmp_r, live_r = tmp_path / "tmp", tmp_path / "live"
    _write(tmp_r / "tile=A" / "cells.parquet", b"NEW")
    _write(live_r / "tile=A" / "cells.parquet", b"OLD")
    gr.atomic_swap(tmp_r, live_r)
    assert (live_r / "tile=A" / "cells.parquet").read_bytes() == b"NEW"
    assert not tmp_r.exists()
    assert not live_r.with_name(live_r.name + ".bak_rederive").exists()


def test_atomic_swap_when_no_prior_live(tmp_path: Path) -> None:
    tmp_r, live_r = tmp_path / "tmp", tmp_path / "live"
    _write(tmp_r / "tile=A" / "cells.parquet", b"NEW")
    gr.atomic_swap(tmp_r, live_r)
    assert (live_r / "tile=A" / "cells.parquet").read_bytes() == b"NEW"
    assert not tmp_r.exists()


# --- Integration: halt-on-content-change leaves live UNTOUCHED ----------------


def _patch_derive_to_write(monkeypatch, payload: bytes):
    """Replace derive_region with a stub that populates the temp region with one
    tile carrying `payload` (simulates an encoder whose output changed)."""

    def fake_derive(cfg):
        _write(cfg.output_region_dir / "tile=A" / "cells.parquet", payload)
        (cfg.output_region_dir / "_SUCCESS").touch()

    monkeypatch.setattr(gr, "derive_region", fake_derive)


def _make_base(tmp_path: Path, city: str, live_bytes: bytes) -> Path:
    base = tmp_path / "processed"
    for stage in ("sub_c", "sub_d", "sub_e"):
        (base / stage / "2026-04-15.0" / city).mkdir(parents=True, exist_ok=True)
    _write(base / "sub_f" / "2026-04-15.0" / city / "tile=A" / "cells.parquet", live_bytes)
    return base


def test_halt_on_content_change_leaves_live_untouched(tmp_path, monkeypatch) -> None:
    city = "testville"
    base = _make_base(tmp_path, city, live_bytes=b"OLD")
    _patch_derive_to_write(monkeypatch, payload=b"CHANGED")  # differs from live
    live_cells = base / "sub_f" / "2026-04-15.0" / city / "tile=A" / "cells.parquet"

    with pytest.raises(gr.ContentChangedError):
        gr.guarded_rederive_city("2026-04-15.0", city, base, allow_content_change=False)

    # live MUST be untouched (overwrite-then-discover is the bug we guard against)
    assert live_cells.read_bytes() == b"OLD"
    # temp must be cleaned up, not left as a half-state
    assert not (base / "sub_f" / "2026-04-15.0" / f".tmp_rederive_{city}").exists()


def test_allow_content_change_authorizes_overwrite(tmp_path, monkeypatch) -> None:
    city = "testville"
    base = _make_base(tmp_path, city, live_bytes=b"OLD")
    _patch_derive_to_write(monkeypatch, payload=b"CHANGED")
    live_cells = base / "sub_f" / "2026-04-15.0" / city / "tile=A" / "cells.parquet"

    res = gr.guarded_rederive_city("2026-04-15.0", city, base, allow_content_change=True)
    assert live_cells.read_bytes() == b"CHANGED"
    assert res["tiles_changed"] == 1


def test_identical_rederive_swaps_without_change(tmp_path, monkeypatch) -> None:
    city = "testville"
    base = _make_base(tmp_path, city, live_bytes=b"SAME")
    _patch_derive_to_write(monkeypatch, payload=b"SAME")  # byte-identical refactor
    res = gr.guarded_rederive_city("2026-04-15.0", city, base, allow_content_change=False)
    assert res["tiles_changed"] == 0
    assert res["success_marker"] is True
