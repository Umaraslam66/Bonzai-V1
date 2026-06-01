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
  frozen set can't resolve it but a larger / second-region set could (extract a
  second region); below the floor no single-region set can EVER resolve it
  (categorically a multi-region need, not an N-tuning knob).
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
            f"needed architecture-distinguishing gap {needed_gap} < this frozen set's "
            f"resolved gap {resolved}: this frozen set CANNOT resolve it, but a LARGER / "
            f"SECOND-REGION set could in principle. Escalate: extract a second region "
            f"(the deferred B-decision)."
        )
    raise InsufficientResolutionError(
        f"needed architecture-distinguishing gap {needed_gap} < single-region floor {floor}: "
        f"no single-region set can EVER resolve this — single-region is FUNDAMENTALLY "
        f"insufficient for this gap. Escalate: this requires multi-region data, not more "
        f"Singapore tiles (categorically, not an N-tuning knob)."
    )
