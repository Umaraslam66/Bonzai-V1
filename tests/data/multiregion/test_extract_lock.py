"""Per-city extraction lock (the headline guard for the sub-C run path). After the
2026-06-05 double-nohup near-miss (two concurrent in-place re-derives of the SAME
city dirs, prevented only by wait-loop timing), a second concurrent extract of a
city must refuse IMMEDIATELY. Different cities may extract concurrently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.multiregion import extract_lock as el


def test_second_extract_of_same_city_is_refused(tmp_path: Path) -> None:
    p = el.city_lock_path(tmp_path, "paris")
    fh = el.acquire_city_lock(p)
    try:
        with pytest.raises(el.ConcurrentExtractError):
            el.acquire_city_lock(p)  # second holder while first is alive -> refused
    finally:
        fh.close()


def test_different_cities_extract_concurrently(tmp_path: Path) -> None:
    fh1 = el.acquire_city_lock(el.city_lock_path(tmp_path, "paris"))
    fh2 = el.acquire_city_lock(el.city_lock_path(tmp_path, "madrid"))  # different city: OK
    try:
        assert fh1 is not None and fh2 is not None
    finally:
        fh1.close()
        fh2.close()


def test_lock_reacquirable_after_release(tmp_path: Path) -> None:
    p = el.city_lock_path(tmp_path, "paris")
    el.acquire_city_lock(p).close()  # acquire + release
    el.acquire_city_lock(p).close()  # now free again


def test_lock_path_is_per_city_under_processed(tmp_path: Path) -> None:
    a = el.city_lock_path(tmp_path, "paris")
    b = el.city_lock_path(tmp_path, "madrid")
    assert a != b
    assert a.name == "paris.extract.lock"
    assert ".locks" in a.parts
    # co-located with the processed corpus, not scattered at the repo root
    assert "processed" in a.parts
