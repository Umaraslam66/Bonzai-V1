"""Guarded, crash-safe, idempotent sub-F re-derive — the STANDARD tool for any
DESTRUCTIVE in-place re-derive of the (currently single-copy) processed corpus.

Three guards, each a lesson from the 2026-06-05 double-nohup near-miss (an
unguarded hand-rolled loop nearly ran two concurrent in-place re-derives of the
same city dirs; only wait-loop timing prevented corpus corruption):

  1. LOCKFILE (fcntl.flock, non-blocking). A second concurrent invocation exits
     non-zero IMMEDIATELY. Corpus integrity never rests on timing again.

  2. ATOMIC SWAP. Each city is derived into a TEMP dir on the SAME filesystem;
     the live region is replaced only on full success (validator passed +
     _SUCCESS in temp), via a rename-based swap (live -> .bak -> remove). The
     long, kill-prone work (the derive) writes ONLY to temp — a watchdog kill or
     crash mid-derive leaves the temp incomplete and the LIVE artifact intact.
     `pq.write_table` is not atomic (writes in place), so deriving straight into
     live would truncate a tile on a kill; deriving into temp removes that window.

  3. HALT-ON-NON-IDENTICAL. Before swapping, every temp cells.parquet is compared
     (raw sha256) against the live one. If a prior live exists and ANY tile is
     NOT byte-identical, the city HALTS and surfaces the differing tile — it does
     NOT overwrite-then-discover. A non-identical result means the encoder change
     is no longer a pure refactor, and the operator must SEE that before the v1.1
     artifact is destroyed. Intentional content-changing regen (e.g. the #17
     touch-as-cross root fix, which DOES change bytes) must pass
     --allow-content-change to authorize the overwrite.

Usage:
  python -m scripts.multiregion.guarded_rederive --release 2026-04-15.0 \
      --city bruges [--city krakow ...] [--allow-content-change]
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import logging
import os
import shutil
from pathlib import Path

from cfm.data.sub_f.pipeline import PipelineConfig, derive_region

log = logging.getLogger("guarded_rederive")

LOCK_PATH = Path("data/processed/sub_f/.rederive.lock")


class ConcurrentRederiveError(RuntimeError):
    """Raised when another guarded re-derive already holds the lock."""


class ContentChangedError(RuntimeError):
    """Raised when a re-derive would change live cells.parquet bytes without
    --allow-content-change (the encoder change is not a pure refactor, or this
    is an intentional regen that must be authorized explicitly)."""


def acquire_lock(lock_path: Path = LOCK_PATH):
    """Acquire an exclusive, non-blocking flock. Returns the held file handle
    (keep it alive for the lock's lifetime). Raises ConcurrentRederiveError if
    another process holds it — so a second invocation cannot double-run."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise ConcurrentRederiveError(
            f"another guarded re-derive holds {lock_path}; refusing to double-run"
        ) from exc
    fh.write(f"pid={os.getpid()}\n")
    fh.flush()
    return fh


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare_cells(tmp_region: Path, live_region: Path) -> list[str]:
    """Return the list of tile names whose cells.parquet differs (raw bytes)
    between the freshly-derived temp region and the live region. Tiles absent
    from live are NOT differences (a fill). Used to gate the swap."""
    differing: list[str] = []
    for tmp_cells in sorted(tmp_region.glob("tile=*/cells.parquet")):
        tile = tmp_cells.parent.name
        live_cells = live_region / tile / "cells.parquet"
        if not live_cells.exists():
            continue  # new tile (fill), not a content change
        if _sha(tmp_cells) != _sha(live_cells):
            differing.append(tile)
    return differing


def atomic_swap(tmp_region: Path, live_region: Path) -> None:
    """Atomically replace live_region with tmp_region (same filesystem).

    rename(live -> .bak), rename(tmp -> live), remove(.bak). The only window is
    two fast renames; the .bak is retained until both succeed, so a crash between
    them leaves a recoverable .bak. The long derive already happened in temp."""
    bak = live_region.with_name(live_region.name + ".bak_rederive")
    if bak.exists():
        shutil.rmtree(bak)
    if live_region.exists():
        live_region.rename(bak)
    try:
        tmp_region.rename(live_region)
    except OSError:
        if bak.exists() and not live_region.exists():
            bak.rename(live_region)  # roll back
        raise
    if bak.exists():
        shutil.rmtree(bak)


def guarded_rederive_city(release: str, city: str, base: Path, allow_content_change: bool) -> dict:
    """Derive one city to temp, gate on byte-identity, then atomically swap.

    Returns a result dict. Raises ContentChangedError (live untouched) if the
    derive changes bytes and allow_content_change is False."""
    live = base / "sub_f" / release / city
    tmp = base / "sub_f" / release / f".tmp_rederive_{city}"  # same fs -> atomic swap
    if tmp.exists():
        shutil.rmtree(tmp)
    cfg = PipelineConfig(
        release=release,
        region=city,
        sub_c_region_dir=base / "sub_c" / release / city,
        sub_d_region_dir=base / "sub_d" / release / city,
        sub_e_region_dir=base / "sub_e" / release / city,
        output_region_dir=tmp,
        run_alpha_drop_report=False,
    )
    log.info("deriving %s into temp %s (live untouched during derive)", city, tmp)
    derive_region(cfg)  # writes ONLY to tmp; runs v1.2 validator; _SUCCESS last

    differing = compare_cells(tmp, live)
    had_live = live.exists()
    if differing and not allow_content_change:
        shutil.rmtree(tmp)  # discard; live untouched
        raise ContentChangedError(
            f"{city}: {len(differing)} tile(s) changed bytes vs live "
            f"(first: {differing[0]}). Live LEFT UNTOUCHED. If intentional regen, "
            f"re-run with --allow-content-change; else the encoder change is not a "
            f"pure refactor and must be investigated before overwriting v1.1."
        )
    atomic_swap(tmp, live)
    return {
        "city": city,
        "had_prior_live": had_live,
        "tiles_changed": len(differing),
        "success_marker": (live / "_SUCCESS").exists(),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--city", action="append", required=True, dest="cities")
    ap.add_argument("--base", default="data/processed")
    ap.add_argument(
        "--allow-content-change",
        action="store_true",
        help="authorize overwriting live when re-derived bytes differ (intentional regen)",
    )
    args = ap.parse_args(argv)
    base = Path(args.base)

    try:
        lock = acquire_lock()
    except ConcurrentRederiveError as exc:
        log.error("%s", exc)
        return 3
    try:
        for city in args.cities:
            try:
                res = guarded_rederive_city(args.release, city, base, args.allow_content_change)
            except ContentChangedError as exc:
                log.error("HALT %s", exc)
                return 4
            log.info(
                "%s: swapped (prior_live=%s, tiles_changed=%d, _SUCCESS=%s)",
                res["city"],
                res["had_prior_live"],
                res["tiles_changed"],
                res["success_marker"],
            )
    finally:
        lock.close()  # releases the flock
    log.info("guarded re-derive complete for %d cities", len(args.cities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
