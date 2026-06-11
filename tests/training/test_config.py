"""Readiness-closure F16: every config (and through save_hyperparameters, every
checkpoint; and through _write_report's model_dump, every report) carries a
``conditioning_scheme`` tag, so a checkpoint trained under one prefix scheme can
never be silently loaded under the other."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cfm.training.config import ScaffoldConfig


def test_config_carries_conditioning_scheme_default_value():
    assert ScaffoldConfig().conditioning_scheme == "value"


def test_config_rejects_unknown_conditioning_scheme():
    with pytest.raises(ValidationError):
        ScaffoldConfig(conditioning_scheme="bogus")


def test_conditioning_scheme_lands_in_model_dump():
    # scripts/train_scaffold.py::_write_report renders cfg.model_dump() wholesale
    # into the report's Config block, so presence in the dump == presence in every
    # reports/ summary (no report-side code needed).
    assert "conditioning_scheme" in ScaffoldConfig().model_dump()


# --- Task 24a: the conditioning_ablation switch (spec §8 Lane S instrument) ---


def test_config_carries_conditioning_ablation_default_full():
    assert ScaffoldConfig().conditioning_ablation == "full"
    assert "conditioning_ablation" in ScaffoldConfig().model_dump()  # reproducibility record


def test_config_accepts_the_three_ablation_modes():
    for mode in ("full", "no_city", "no_character"):
        assert ScaffoldConfig(conditioning_ablation=mode).conditioning_ablation == mode


def test_config_rejects_unknown_conditioning_ablation():
    with pytest.raises(ValidationError):
        ScaffoldConfig(conditioning_ablation="no_everything")
