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

from cfm.eval.holdout.paths import (
    _EU_HELD_OUT_CITIES,
    DEFAULT_REGION,
    eval_set_locked_marker,
    multiregion_eval_set_locked_marker,
)


class InsufficientResolutionError(Exception):
    """The needed resolving gap is finer than the frozen eval set can resolve
    (KS-resolution only — the architecture-discrimination verdict is T12's,
    assert_coherence_power_sufficient). Carries the named escalation."""


def resolution_marker_for_region(release: str, region: str) -> Path:
    """REGION-AWARE resolution marker (F9, Task 20) — mirrors
    ``holdout_manifest_for_region``'s fail-closed routing (cfm.eval.holdout.paths):

      - ``"singapore"``                 -> the SG ``_EVAL_SET_LOCKED`` (carries KS fields)
      - one of the 4 EU held-out cities -> the multiregion ``_EVAL_SET_LOCKED``
      - anything else                   -> raise (fail-closed; never silently mis-route)

    NOTE: the REAL multiregion marker carries NO ks fields today; routing an EU
    region here means the read below fails LOUDLY (KeyError) until the EU KS numbers
    are derived — never a silent fallback to the SG numbers."""
    if region == DEFAULT_REGION:
        return eval_set_locked_marker(release)
    if region in _EU_HELD_OUT_CITIES:
        return multiregion_eval_set_locked_marker(release)
    raise ValueError(
        f"resolution_marker_for_region: unknown region {region!r}; expected "
        f"{DEFAULT_REGION!r} (SG) or one of the EU held-out cities "
        f"{sorted(_EU_HELD_OUT_CITIES)}"
    )


def assert_resolution_sufficient(
    needed_gap: float,
    *,
    marker_path: Path | None = None,
    region: str | None = None,
    release: str = "2026-04-15.0",
) -> None:
    """Raise iff the frozen eval set cannot resolve ``needed_gap``.

    Fail-closed: a missing/unreadable marker or missing required fields RAISES.
    Marker precedence: explicit ``marker_path`` > ``region`` routing
    (``resolution_marker_for_region``) > today's default (the SG marker).
    """
    if marker_path is not None:
        path = Path(marker_path)
    elif region is not None:
        path = resolution_marker_for_region(release, region)
    else:
        path = eval_set_locked_marker(release)
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


class CoherencePowerInsufficientError(Exception):
    """Held-out usable-n cannot resolve the model-vs-real coherence effect for architecture
    discrimination on this stratum (spec §7). SOLE verdict for that question."""


def assert_coherence_power_sufficient(
    *,
    stratum: str,
    usable_n: int,
    resolved_gap: float,
    model_vs_real_effect: float,
) -> None:
    """The ONE architecture-discrimination POWER verdict (spec §7). Fires at the first trained
    model checkpoint; dormant until then.

    Inputs (all supplied by callers — this gate does NOT compute them):
    - ``resolved_gap``: the NUMBER produced by ``assert_resolution_sufficient`` (train-split KS).
    - ``usable_n``: the held-out power side (munich's 156 is the floor).
    - ``model_vs_real_effect``: the model-vs-real coherence effect size, arriving at the first
      trained model. **Its DEFINITION is an OPEN first-model decision, deliberately NOT pinned
      here** — whatever it is later defined as, it MUST be anti-leak-proven (a model that merely
      ECHOES the handed tile-mode conditioning, scoring high ABSOLUTE coherence without generating
      real structure, MUST fail it; the absolute band alone is conditioning-contaminated). This
      gate only CONSUMES the effect; it never computes it.

    NOTE: this POWER gate is distinct from the T11 metric-VALIDATION finding. munich's tooth-3
    shuffle-gap saturates (dense-core #21) — that did NOT trigger a swap (structural exclusion,
    munich stays). The munich->manchester swap below is the POWER escalation that fires only if,
    at first model, usable-n cannot resolve the effect.
    """
    if model_vs_real_effect < resolved_gap:  # finer than the train split can resolve
        raise CoherencePowerInsufficientError(
            f"stratum {stratum!r}: model-vs-real coherence effect {model_vs_real_effect} is finer "
            f"than the train-resolved gap {resolved_gap}; held-out usable_n={usable_n} cannot "
            f"discriminate architectures here. Escalate (owned by THIS gate): munich->manchester "
            f"(swap the floor stratum to a larger held-out city) or add-a-train-city, then re-lock "
            f"the multi-region eval set (write-once-per-version)."
        )
