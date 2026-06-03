"""Roll-up manifest (spec §6): the reproducibility record + the diversity-by-
construction proof (axis-coverage matrix). A failed city is present-and-LOUD,
never dropped.

The proceed-to-next-batch gate is STRUCTURAL, not a total count (threshold-pairing,
protocol §2): an aggregate "N validated" can hide that failures clustered in one
morphology (or a city was excluded), leaving an axis label with zero validated
representatives. The gate pairs the failure check with an axis-coverage check that
fires on exactly that hidden hole.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

VALIDATED = "validated"
FAILED = "failed"
PENDING = "pending"

_AXES = ("morphology", "density", "geography")


@dataclass
class CityRecord:
    name: str
    morphology: str
    density: str
    geography: str
    region_crs: str
    tile_count: int
    fetch_seconds: float
    stage_shas: dict[str, str]
    release: str
    validation_status: str  # validated | failed | pending
    token_count: int
    regime: str = ""  # set when failed: the surfaced regime (loud, never dropped)


@dataclass
class RollUp:
    cities: list[CityRecord] = field(default_factory=list)


def _axis_label(city: CityRecord, axis: str) -> str:
    return getattr(city, axis)


def unaddressed_failures(r: RollUp) -> list[str]:
    return [c.name for c in r.cities if c.validation_status == FAILED]


def axis_coverage(r: RollUp) -> dict[str, dict[str, int]]:
    """Per-axis label -> count of VALIDATED cities. Proves diversity by construction;
    failed/pending cities do NOT count toward coverage."""
    cov: dict[str, dict[str, int]] = {axis: defaultdict(int) for axis in _AXES}
    for c in r.cities:
        if c.validation_status != VALIDATED:
            continue
        for axis in _AXES:
            cov[axis][_axis_label(c, axis)] += 1
    return {axis: dict(labels) for axis, labels in cov.items()}


def _intended_labels(
    r: RollUp, required_axis_labels: dict[str, set[str]] | None
) -> dict[str, set[str]]:
    """The labels the validated set MUST cover. Prefer an explicit intended span
    (fixed at selection time, so it does not shrink when a city is excluded);
    otherwise derive from ALL selected cities (any status)."""
    if required_axis_labels is not None:
        return required_axis_labels
    intended: dict[str, set[str]] = {axis: set() for axis in _AXES}
    for c in r.cities:
        for axis in _AXES:
            intended[axis].add(_axis_label(c, axis))
    return intended


def uncovered_axis_labels(
    r: RollUp, required_axis_labels: dict[str, set[str]] | None = None
) -> list[str]:
    """Intended (axis, label) pairs with ZERO validated cities — the diversity
    holes an aggregate count would hide."""
    cov = axis_coverage(r)
    intended = _intended_labels(r, required_axis_labels)
    uncovered: list[str] = []
    for axis, labels in intended.items():
        for label in labels:
            if cov.get(axis, {}).get(label, 0) == 0:
                uncovered.append(f"{axis}={label}")
    return sorted(uncovered)


def assert_ready_for_next_batch(
    r: RollUp, required_axis_labels: dict[str, set[str]] | None = None
) -> None:
    """Gate proceeding to the next batch. STRUCTURAL, not a count: requires (1)
    zero unaddressed failures AND (2) complete axis coverage. (2) is independent
    of (1) — a city excluded after an unfixable regime can leave an axis uncovered
    while (1) passes; the count alone would read 'clean'."""
    failed = unaddressed_failures(r)
    if failed:
        raise RuntimeError(
            f"{len(failed)} cities failed-needs-attention; address (fix+rerun or "
            f"explicit exclusion) before the next batch — do not silently drop: {failed}"
        )
    uncovered = uncovered_axis_labels(r, required_axis_labels)
    if uncovered:
        raise RuntimeError(
            f"axis coverage incomplete — these intended labels have ZERO validated "
            f"cities (a clustered failure or selection gap the aggregate count hides): "
            f"{uncovered}"
        )


def total_validated_tokens(r: RollUp) -> int:
    return sum(c.token_count for c in r.cities if c.validation_status == VALIDATED)


def total_validated_tiles(r: RollUp) -> int:
    return sum(c.tile_count for c in r.cities if c.validation_status == VALIDATED)


def write_rollup(r: RollUp, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "cities": [asdict(c) for c in r.cities],
        "totals": {
            "validated_cities": sum(1 for c in r.cities if c.validation_status == VALIDATED),
            "validated_tiles": total_validated_tiles(r),
            "validated_tokens": total_validated_tokens(r),
            "axis_coverage": axis_coverage(r),
        },
    }
    if extra:
        payload["totals"].update(extra)  # tile-budget vs achieved, proxy verdict, ...
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
