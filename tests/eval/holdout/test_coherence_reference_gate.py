"""Unit tests for the tooth-3 validation-separation gate (spec §3.3/§3.5a).

Pure unit tests of ``assert_validation_separation`` / ``is_dense_core_saturated`` — no
yaml, no corpus. They prove (a) the gate is NON-VACUOUS on the kept (moderate) set — it
fires on a synthetic moderate-density defect — and (b) the dense-core exemption keys on the
STRUCTURAL ``mean_road_edges`` threshold, never on a city name.
"""

from __future__ import annotations

import pytest

from cfm.eval.holdout.coherence_reference import (
    DENSE_CORE_EDGE_THRESHOLD,
    ValidationSeparationError,
    assert_validation_separation,
    is_dense_core_saturated,
)


def test_gate_fires_on_moderate_defect():
    # A MODERATE stratum (below the dense-core threshold) whose real-vs-permuted
    # separation falls under the 0.70 floor MUST trip the gate — proving the gate is
    # non-vacuous on the kept set.
    per_stratum = {
        "synthetic_moderate": {
            "mean_road_edges": 25.0,
            "real_vs_permuted_positive_fraction": 0.50,
        }
    }
    with pytest.raises(ValidationSeparationError):
        assert_validation_separation(per_stratum)


def test_gate_passes_when_moderate_strata_separate():
    # Moderate strata all at/above the 0.70 floor → no raise.
    per_stratum = {
        "a": {"mean_road_edges": 24.7, "real_vs_permuted_positive_fraction": 0.95},
        "b": {"mean_road_edges": 29.2, "real_vs_permuted_positive_fraction": 0.81},
        "c": {"mean_road_edges": 36.3, "real_vs_permuted_positive_fraction": 0.88},
    }
    assert_validation_separation(per_stratum)  # must NOT raise


def test_dense_core_stratum_exempt_by_structure():
    # A dense-core stratum (mean_road_edges above the threshold) at a LOW separation
    # (0.43) is EXEMPT — but the exemption keys on mean_road_edges, not on a name: a
    # moderate stratum at the SAME 0.43 in the SAME dict must still raise.
    exempt_only = {
        "dense": {"mean_road_edges": 48.0, "real_vs_permuted_positive_fraction": 0.43},
    }
    assert_validation_separation(exempt_only)  # dense-core exempt → no raise

    mixed = {
        "dense": {"mean_road_edges": 48.0, "real_vs_permuted_positive_fraction": 0.43},
        "moderate": {"mean_road_edges": 25.0, "real_vs_permuted_positive_fraction": 0.43},
    }
    with pytest.raises(ValidationSeparationError) as exc:
        assert_validation_separation(mixed)
    # Only the MODERATE stratum is named in the failure (structural, not by-name).
    assert "moderate" in str(exc.value)
    assert "dense" not in str(exc.value)


def test_threshold_is_capacity_fraction():
    # Threshold is set from the 60-edge capacity (2/3 of 60), not fit to munich.
    assert DENSE_CORE_EDGE_THRESHOLD == 40.0
    # munich (47.8) above; krakow (36.3) below.
    assert is_dense_core_saturated(47.8) is True
    assert is_dense_core_saturated(36.3) is False
