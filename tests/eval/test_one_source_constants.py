"""W5: the KS two-sample critical coefficient c(0.05)=1.358 is ONE-SOURCED.

Three modules used to carry value-equal-but-independent literals
(feature_resolution, holdout/sizing, holdout/pipeline). The one source is
``cfm.eval.feature_resolution.KS_C_ALPHA_05``; the consumers import it. The
source-scan assertions make a re-introduced literal go red, and the behavior
anchors prove the formulas still produce the known values through the import.
"""

from __future__ import annotations

import inspect

import pytest


def test_consumers_carry_no_independent_literal():
    from cfm.eval.holdout import pipeline, sizing

    for mod in (pipeline, sizing):
        assert "1.358" not in inspect.getsource(mod), (
            f"{mod.__name__} carries an independent 1.358 literal — the KS critical "
            f"coefficient is one-sourced at feature_resolution.KS_C_ALPHA_05"
        )


def test_the_one_source_is_public_and_exact():
    from cfm.eval.feature_resolution import KS_C_ALPHA_05

    assert KS_C_ALPHA_05 == 1.358


def test_behavior_anchors_through_the_import():
    from cfm.eval.holdout.pipeline import _gap_from_cells
    from cfm.eval.holdout.sizing import ks_two_sample_floor

    # sqrt(2/2) == 1: the gap at 2 cells IS the coefficient
    assert _gap_from_cells(2) == pytest.approx(1.358)
    # the locked eval-set sizing fact: KS floor at effect 0.08 (the frozen
    # held-out set's resolvable gap) stays exactly what it was
    assert ks_two_sample_floor(effect=0.08) == 577
