"""External-source-of-truth tests for the Lane-S floor adapter (Task 4).

These tests load the REAL locked conditioning floor artifact (~164k lines, ~1-2s) and assert
the adapter's output against the ground-truth numbers derived during R3 analysis.  They are
``@pytest.mark.slow`` and excluded from the default fast suite.

LOCK-AND-GUARDS-TRAVEL-TOGETHER: the sha assertion in
``test_real_floor_target_shape_and_min_building_n`` ensures that a floor re-derivation which
changes ``n_a``/``n_b`` turns this test RED, forcing ``EXPECTED_FLOOR_SHA256`` + the build
CLI guard to update in the SAME commit (spec invariant, PI 2026-06-21).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.eval import lane_s_sampler as ls

FLOOR = Path("reports/conditioning_floor/2026-04-15.0/conditioning-floor.yaml")


@pytest.mark.slow
def test_real_floor_target_shape_and_min_building_n():
    payload = yaml.safe_load(FLOOR.read_text())
    # LOCK-AND-GUARDS-TRAVEL-TOGETHER: a floor re-derivation that changes n_a/n_b changes this
    # sha -> this assertion goes RED, forcing EXPECTED_FLOOR_SHA256 + the guard to update with it.
    assert payload["floor_sha256"] == ls.EXPECTED_FLOOR_SHA256
    targets = ls.floored_targets(payload)
    assert len(targets) == 146  # distinct (city, 4-tuple) floored 4-tuples (spec Gate 1)
    both = [t for t in targets.values() if t.binding_metric == ls.BUILDING_METRIC]
    assert len(both) == 119  # building_area owed (binds) in 119 of 146
    counts = ls.heldout_feature_counts(payload)
    building_ns = [counts[(t.city, ls.BUILDING_METRIC, t.stratum)] for t in both]
    assert min(building_ns) == 59  # R3: min real building feature count


@pytest.mark.slow
def test_real_floor_reproduces_R3_ceiling_bound_counts():
    payload = yaml.safe_load(FLOOR.read_text())
    counts = ls.heldout_feature_counts(payload)
    both = [
        t for t in ls.floored_targets(payload).values() if t.binding_metric == ls.BUILDING_METRIC
    ]
    ns = [counts[(t.city, ls.BUILDING_METRIC, t.stratum)] for t in both]
    # ceiling_bound <=> floor_n < target*headroom (available cancels). R3 census numbers:
    assert sum(1 for n in ns if n < 50 * 1.0) == 0
    assert sum(1 for n in ns if n < 50 * 2.0) == 6
