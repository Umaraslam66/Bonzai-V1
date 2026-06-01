"""Fail-loud, fail-closed holdout-leak audit (spec §F). The guard the training
scaffold calls (one source) to prove no training-reachable artifact's lineage
includes any held-out tile - tiles AND every derivative.

Belt-and-suspenders is justified because the failure is silent AND unrecoverable:
a contaminated holdout invalidates every eval number undetectably.

Guards (each must FAIL in the leak regime):
- G-F1: a held-out TILE in training's path -> trips.
- G-F2: a held-out-DERIVED artifact (lineage includes a held-out tile) -> trips.
- G-F3: the tokenizer-on-real R2 baseline referenced from training -> trips
  (same mechanism as G-F2; the R2 baseline's lineage is its source tiles).
- G-F4: a training-reachable artifact with ABSENT lineage -> trips ON THE ABSENCE
  (fail-closed). Without it the guarantee is only "no artifact with RECORDED
  held-out lineage leaks" - strictly weaker, and the gap is where untracked
  derivatives hide.

Region-keyed (spec §B): the audit iterates regions with one code path; a 2-region
manifest exercises identical logic (no per-region special-casing).
"""

from __future__ import annotations

from dataclasses import dataclass

#: (region, tile_i, tile_j)
TileRef = tuple[str, int, int]


@dataclass(frozen=True)
class Artifact:
    path: str
    lineage: frozenset[TileRef] | None  # None = untracked (G-F4 fail-closed)


@dataclass(frozen=True)
class LineageFailure:
    path: str
    reason: str


class HoldoutLeakError(Exception):
    def __init__(self, failures: list[LineageFailure]) -> None:
        self.failures = failures
        super().__init__(
            "held-out lineage leak detected:\n"
            + "\n".join(f"  {f.path}: {f.reason}" for f in failures)
        )


def _holdout_tile_refs(holdout_manifest: dict) -> set[TileRef]:
    refs: set[TileRef] = set()
    for region, payload in holdout_manifest["regions"].items():
        for t in payload["tiles"]:
            refs.add((region, int(t["tile_i"]), int(t["tile_j"])))
    return refs


def audit_no_holdout_leak(holdout_manifest: dict, training_reachable: list[Artifact]) -> None:
    """Raise HoldoutLeakError listing EVERY failure; return None iff clean."""
    holdout = _holdout_tile_refs(holdout_manifest)
    failures: list[LineageFailure] = []
    for art in training_reachable:
        if art.lineage is None:  # G-F4
            failures.append(LineageFailure(art.path, "absent lineage (fail-closed)"))
            continue
        leaked = art.lineage & holdout  # G-F1/F2/F3
        if leaked:
            failures.append(
                LineageFailure(art.path, f"lineage includes held-out tiles {sorted(leaked)}")
            )
    if failures:
        raise HoldoutLeakError(failures)
