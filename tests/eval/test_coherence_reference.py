from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
        # measured, not a literal threshold
        assert s["continuity_gap"] is not None and s["fragmentation_gap"] is not None


def test_reference_records_measurement_regime():
    ref = _load()
    assert ref["n_shuffle"] >= 100  # PI-locked precision
    for s in ref["per_stratum"].values():
        # noise floor recorded
        assert "continuity_gap_sd" in s and "fragmentation_gap_sd" in s


def test_tooth3_real_vs_permuted_separation_gate():
    # tooth-3 HALT-gate: the metric reads arrangement on REAL held-out tiles
    ref = _load()
    for city, s in ref["per_stratum"].items():
        assert s["real_vs_permuted_positive_fraction"] >= 0.7, (
            f"{city} separation {s['real_vs_permuted_positive_fraction']} < 0.7"
        )
