from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from cfm.eval.holdout.coherence_reference import assert_validation_separation

_REF = Path("reports/2026-06-08-coherence-reference.yaml")
_STRATA = {"glasgow", "eisenhuttenstadt", "munich", "krakow"}


def _load():
    if not _REF.exists():
        pytest.skip("coherence reference not yet measured on Leonardo")
    return yaml.safe_load(_REF.read_text(encoding="utf-8"))


def test_reference_is_per_stratum_measured_not_constant():
    ref = _load()
    assert set(ref["per_stratum"]) == _STRATA
    for s in ref["per_stratum"].values():
        # measured, finite, not a literal threshold (and not a locked nan:
        # `float("nan") is not None` is True, so assert FINITE, not just not-None)
        assert math.isfinite(s["continuity_gap"]) and math.isfinite(s["fragmentation_gap"])


def test_reference_carries_both_measures():
    # PI ruling: the reference records BOTH the absolute coherence BAND (the real
    # arrangement scores) AND the shuffle-GAP, plus the structural fields that drive
    # the tooth-3 dense-core exemption. All finite (a locked nan must not slip through).
    ref = _load()
    for s in ref["per_stratum"].values():
        # absolute band
        assert math.isfinite(s["continuity_real"])
        assert math.isfinite(s["giant_real"])
        assert math.isfinite(s["zoning_real"])
        # shuffle-gap
        assert math.isfinite(s["continuity_gap"])
        assert math.isfinite(s["fragmentation_gap"])
        # structural fields
        assert math.isfinite(s["mean_road_edges"])
        assert isinstance(s["dense_core_saturated"], bool)


def test_reference_records_measurement_regime():
    ref = _load()
    assert ref["n_shuffle"] >= 100  # PI-locked precision
    for s in ref["per_stratum"].values():
        # noise floor recorded
        assert "continuity_gap_sd" in s and "fragmentation_gap_sd" in s


def test_tooth3_gate_passes_on_moderate_strata():
    # tooth-3 HALT-gate, scoped by STRUCTURE not identity: moderate strata must
    # separate (>= 0.70); dense-core strata (#21, e.g. munich) saturate the
    # shuffle-null and are exempt by their recorded mean_road_edges. The committed
    # reference must pass this gate.
    ref = _load()
    assert_validation_separation(ref["per_stratum"])  # must NOT raise
