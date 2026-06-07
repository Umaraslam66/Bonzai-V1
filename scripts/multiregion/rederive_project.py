#!/usr/bin/env python3
"""Running-total projection + HALT gate for the #19 corpus-wide re-derive.

Reads GROUND TRUTH from disk (not a result-file side-channel that could drift):
each city's sub_f manifest `sub_f_derivation_version` is "1.2" once re-derived
under the fix, "1.1" if still pending. For re-derived cities it measures the
post-fix token count and reads the sub_g `_PHASE1_VALIDATED` marker.

Projection is on the CORPUS-NORMAL set ONLY (33 validated-keep + 7 recover).
rotterdam/warsaw (degraded source, ~12-13x) are EXCLUDED — they are a separate
conditional bonus that only counts if they earn re-admission via their own clean
sub_g verdict; blending their ~79M into the projection would let a halt/proceed
call rest on tokens that may evaporate (PI, 2026-06-07).

projected_final = sum(done corpus-normal, validated, post-fix tokens)
               + sum(pending corpus-normal, prefix_tokens * (1 - max_observed_shrink))
HALT (exit 2) if projected_final < FLOOR. Conservative: max-observed-shrink applied
to all pending, and the densest/most-densified cities are re-derived FIRST so the
worst shrink is measured early.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pyarrow.parquet as pq
import yaml

RELEASE = "2026-04-15.0"
FLOOR = 550_000_000
PROC = Path("data/processed")
G4_REPORT = Path("reports/2026-06-05-phase-2-g4-corpus-dod.yaml")

# Degraded source (known_issue #20) — EXCLUDED from the projection; separate verdict.
DEGRADED = {"rotterdam", "warsaw"}
# Never part of this EU corpus (Phase-1 / test cities) — not re-derived here.
NOT_IN_CORPUS = {"singapore", "berlin"}
# Never extracted (no sub_f) — not re-derivable.
NEVER_EXTRACTED = {"lyon", "welwyn", "paris", "madrid", "rome"}


def _tokens(city: str) -> int:
    n = 0
    for f in sorted(glob.glob(str(PROC / "sub_f" / RELEASE / city / "tile=*" / "cells.parquet"))):
        t = pq.ParquetFile(f).read(columns=["token_sequence"])
        n += len(t.column("token_sequence").combine_chunks().flatten())
    return n


def _derivation_version(city: str) -> str | None:
    m = PROC / "sub_f" / RELEASE / city / "manifest.yaml"
    if not m.exists():
        return None
    return str(yaml.safe_load(m.read_text()).get("sub_f_derivation_version"))


def _validated(city: str) -> bool:
    return (PROC / "sub_g" / RELEASE / city / "_PHASE1_VALIDATED").exists()


def main() -> int:
    rows = yaml.safe_load(G4_REPORT.read_text())["per_city"]
    prefix = {r["name"]: r["tokens"] for r in rows}
    corpus_normal = [
        r["name"]
        for r in rows
        if r["name"] not in DEGRADED
        and r["name"] not in NOT_IN_CORPUS
        and r["name"] not in NEVER_EXTRACTED
        and prefix[r["name"]] > 0
    ]

    done_validated = 0
    done_tokens = 0
    pending: list[str] = []
    failed: list[str] = []
    shrinks: list[float] = []
    print("=== per-city (corpus-normal set) ===")
    for c in sorted(corpus_normal, key=lambda z: -prefix[z]):
        dv = _derivation_version(c)
        if dv == "1.2":
            tok = _tokens(c)
            val = _validated(c)
            shrink = (1 - tok / prefix[c]) if prefix[c] else 0.0
            shrinks.append(shrink)
            tag = "VALID" if val else "FAILED-VAL"
            if val:
                done_validated += 1
                done_tokens += tok
            else:
                failed.append(c)
            print(f"  {c:<16} re-derived  {tag:<10} tok={tok:>10,} shrink={shrink * 100:+.2f}%")
        else:
            pending.append(c)

    max_shrink = max(shrinks) if shrinks else 0.0
    projected_pending = sum(int(prefix[c] * (1 - max_shrink)) for c in pending)
    projected_final = done_tokens + projected_pending

    print("\n=== projection (corpus-normal ONLY; rotterdam/warsaw excluded) ===")
    pct = f"{max_shrink * 100:.2f}%"
    print(f"  done validated: {done_validated} cities, {done_tokens:,} tokens")
    print(f"  pending: {len(pending)} cities, projected {projected_pending:,} @ max_shrink={pct}")
    print(f"  PROJECTED FINAL: {projected_final:,}  (floor {FLOOR:,})")
    if failed:
        print(f"  ⚠ post-fix VALIDATION FAILURES (corpus-normal): {failed}")

    # Degraded pair — reported separately, NOT in the projection.
    print("\n=== degraded pair (separate verdict; NOT in projection) ===")
    for c in sorted(DEGRADED):
        dv = _derivation_version(c)
        if dv == "1.2":
            print(f"  {c}: re-derived validated={_validated(c)} tok={_tokens(c):,}")
        else:
            print(f"  {c}: pending")

    halt = projected_final < FLOOR
    verdict = (
        "HALT — projected < floor; add cities before finishing"
        if halt
        else "PROCEED — projection clears floor"
    )
    print(f"\n  VERDICT: {verdict}")
    return 2 if halt else 0


if __name__ == "__main__":
    raise SystemExit(main())
