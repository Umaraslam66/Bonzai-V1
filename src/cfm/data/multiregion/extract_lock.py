"""Per-city extraction lock — the headline guard for the (destructive) sub-C run
path, baked into the standard driver so it is NEVER an opt-in step that can be
forgotten.

Lesson from the 2026-06-05 double-nohup near-miss (known_issues #18): a hand-rolled
``nohup`` loop was accidentally launched twice, and two concurrent in-place
re-derives of the same city dirs would have corrupted the single-copy corpus —
prevented only by wait-loop timing, NOT a safeguard. An exclusive, non-blocking
``fcntl.flock`` makes a second concurrent extract of the same city exit
immediately, so corpus integrity never rests on timing again. Locks are per-city
(distinct lock files), so different cities still extract concurrently.

This mirrors ``scripts/multiregion/guarded_rederive.acquire_lock`` (sub-F side);
the difference is the per-city scope and that it is wired into ``driver.run_city``.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO


class ConcurrentExtractError(RuntimeError):
    """Raised when another extraction already holds this city's lock."""


def city_lock_path(repo_root: Path, city: str) -> Path:
    """The per-city lock file, co-located with the processed corpus it guards.

    Both concurrent ``extract_region_batch.py`` processes compute the SAME path
    from the repo root, so the flock coordinates across processes."""
    return (
        Path(repo_root) / "data" / "processed" / "multiregion" / ".locks" / f"{city}.extract.lock"
    )


def acquire_city_lock(lock_path: Path) -> IO[str]:
    """Acquire an exclusive, non-blocking flock for one city.

    Returns the held file handle — keep it alive for the lock's lifetime; closing
    it releases the lock. Raises ConcurrentExtractError if another process (or a
    second open in this one) already holds it, so a second extract cannot double-run.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise ConcurrentExtractError(
            f"another extraction holds {lock_path}; refusing to double-run"
        ) from exc
    fh.write(f"pid={os.getpid()}\n")
    fh.flush()
    return fh
