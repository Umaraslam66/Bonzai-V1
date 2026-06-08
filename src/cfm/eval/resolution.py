"""Trigger-3 eval-harness resolution seam (spec §6 trigger 3; known_issues #12.2).

`assert_resolution_sufficient(needed_gap)` is the §10.1 deferred check landing in
its consumer: the frozen eval set handed the model-facing resolution check forward
"with a fail-loud assertion and a named escalation". It only FULLY activates at
the bake-off (a "needed gap" requires >=2 architectures to compare), so in the
thin slice it is a real, tested pure function awaiting real input — not a stub.

Two pins:
- Marker-sourced, fail-CLOSED on absence (same shape as the holdout audit's G-F4):
  reads ks_resolved_gap_binding / ks_single_region_floor from the frozen eval-set
  marker; a missing / unreadable marker, or missing fields, RAISES — never defaults
  permissive or silently no-ops (a resolution check that no-ops when it can't find
  its threshold is the trigger-3 version of path-synthesized lineage).
- Two failure KINDS with distinct messages + escalations: in [floor, resolved) the
  frozen set can't resolve it but more/larger held-out data could; below the floor
  the gap is finer than any single held-out region can resolve (resolvable-gap
  ceiling — needs more/larger held-out data, not an N-tuning knob).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.eval.holdout.paths import eval_set_locked_marker


class InsufficientResolutionError(Exception):
    """The bake-off's needed architecture-distinguishing gap is finer than the
    frozen eval set can resolve. Carries the named escalation."""


def assert_resolution_sufficient(
    needed_gap: float,
    *,
    marker_path: Path | None = None,
    release: str = "2026-04-15.0",
) -> None:
    """Raise iff the frozen eval set cannot resolve ``needed_gap``.

    Fail-closed: a missing/unreadable marker or missing required fields RAISES.
    """
    path = Path(marker_path) if marker_path is not None else eval_set_locked_marker(release)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))  # FileNotFoundError if absent -> loud
    if not isinstance(data, dict):
        raise InsufficientResolutionError(
            f"resolution marker {path} is unreadable/empty; cannot verify resolution (fail-closed)"
        )
    resolved = data["ks_resolved_gap_binding"]  # KeyError if missing -> loud
    floor = data["ks_single_region_floor"]

    if needed_gap >= resolved:
        return
    if needed_gap >= floor:
        raise InsufficientResolutionError(
            f"needed gap {needed_gap} < this held-out set's resolved gap {resolved}: this "
            f"held-out set CANNOT resolve it; more/larger held-out data could (region-"
            f"extraction is moot at 42 cities). NOTE: this is the KS-resolution concern only "
            f"— it PRODUCES the resolved-gap NUMBER; the architecture-discrimination verdict "
            f"and its escalation are owned by assert_coherence_power_sufficient (T12), not "
            f"this check."
        )
    raise InsufficientResolutionError(
        f"needed gap {needed_gap} < single-region floor {floor}: the resolvable-gap CEILING "
        f"— finer than any single held-out region can resolve; needs more/larger held-out data."
    )
