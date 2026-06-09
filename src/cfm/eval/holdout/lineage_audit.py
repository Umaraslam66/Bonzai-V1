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
- G-F5 (city-guard, spec §6): scoped to regions declared `holdout_kind ==
  "whole_city"`. Trips if any training artifact's lineage touches a wholly-held-out
  REGION via a tile NOT already covered by the enumerated tile-key (G-F1/F2/F3),
  catching un-enumerated held-out tiles (manifest drift / partial enumeration) within
  a tiles-bearing whole-city region and decoupling the guarantee from per-tile
  enumeration completeness for such regions. A payload missing the `tiles` key
  entirely is NOT silently caught here — it raises KeyError at the enumerated-refs
  step before G-F5 runs (fail-loud crash, unreachable under frozen schema-2.0 §2.2b).
  It is reported ALWAYS - even when the tile-key already fired - so a complete message
  names both leak classes.

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


def _whole_city_regions(holdout_manifest: dict) -> set[str]:
    """Regions declared `holdout_kind == "whole_city"` (spec §6 city-guard scope)."""
    return {
        r for r, p in holdout_manifest["regions"].items() if p.get("holdout_kind") == "whole_city"
    }


def audit_no_holdout_leak(holdout_manifest: dict, training_reachable: list[Artifact]) -> None:
    """Raise HoldoutLeakError listing EVERY failure; return None iff clean."""
    holdout = _holdout_tile_refs(holdout_manifest)
    whole_city = _whole_city_regions(holdout_manifest)  # G-F5 city-guard scope
    failures: list[LineageFailure] = []
    for art in training_reachable:
        if art.lineage is None:  # G-F4 fail-closed
            failures.append(LineageFailure(art.path, "absent lineage (fail-closed)"))
            continue
        leaked = art.lineage & holdout  # G-F1/F2/F3 enumerated tile-key
        if leaked:
            failures.append(
                LineageFailure(art.path, f"lineage includes held-out tiles {sorted(leaked)}")
            )
        # G-F5 city-guard: any lineage tile whose region is a wholly-held-out city,
        # enumerated or not. Reported ALWAYS (even when `leaked` is non-empty) so a
        # complete message names BOTH leak classes - no `and not leaked` short-circuit.
        # Subtract `leaked` so the city-guard reports only the city-only (un-enumerated)
        # tiles the enumerated tile-key already missed.
        city_only = sorted(
            {(r, i, j) for (r, i, j) in art.lineage if r in whole_city} - set(leaked)
        )
        if city_only:
            cities = sorted({r for r, _, _ in city_only})
            failures.append(
                LineageFailure(
                    art.path,
                    f"lineage touches wholly-held-out city/cities {cities} "
                    f"(city-guard; tiles not enumerated: {city_only})",
                )
            )
    if failures:
        raise HoldoutLeakError(failures)
