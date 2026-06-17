from __future__ import annotations

import pytest

from cfm.data.sub_f.vocab import vocab_tag_to_id  # noqa: F401 (ensures src import path works)
from cfm.training.env_lock import (
    _MAMBA_EXPECTED,
    LOCKED_CAUSAL_CONV1D,
    LOCKED_LIGHTNING,
    LOCKED_MAMBA_SSM,
    LOCKED_PYDANTIC,
    LOCKED_TORCH,
    LOCKED_TRITON,
    TrainingEnvMismatch,
    check_versions,
)

#: The SHARED lock now includes triton (torch.compile uses it on every backbone).
_LOCKED = {
    "torch": LOCKED_TORCH,
    "lightning": LOCKED_LIGHTNING,
    "pydantic": LOCKED_PYDANTIC,
    "triton": LOCKED_TRITON,
}
_MAMBA_LOCKED = {"mamba-ssm": LOCKED_MAMBA_SSM, "causal-conv1d": LOCKED_CAUSAL_CONV1D}


def test_check_passes_on_the_locked_set():
    check_versions(dict(_LOCKED))  # no raise


def test_torch_carries_the_cu121_build_tag():
    """The CUDA build is part of the lock — a cpu/cu124 build is non-comparable."""
    assert LOCKED_TORCH.endswith("+cu121")


@pytest.mark.parametrize("pkg", ["torch", "lightning", "pydantic", "triton"])
def test_check_raises_on_any_single_version_drift(pkg):
    """Regime-distinguishing: a drift in ANY one shared package fails loud, naming it."""
    bad = dict(_LOCKED)
    bad[pkg] = "9.9.9"
    with pytest.raises(TrainingEnvMismatch) as e:
        check_versions(bad)
    assert pkg in str(e.value)


def test_check_raises_on_missing_package():
    """A package absent from the actual env (None) is a drift, not a pass."""
    bad = dict(_LOCKED)
    del bad["torch"]
    with pytest.raises(TrainingEnvMismatch):
        check_versions(bad)


# --- the conditional mamba pins (GPU verify-before-lock PASS 2026-06-17) --------------
def test_mamba_pins_are_NOT_in_the_shared_lock():
    """The mamba pins must stay OUT of the shared set, else transformer-ar / diffusion
    runs (repo .venv, no mamba installed) would fail the lock."""
    assert "mamba-ssm" not in _LOCKED and "causal-conv1d" not in _LOCKED
    assert set(_MAMBA_EXPECTED) == {"mamba-ssm", "causal-conv1d"}


def test_mamba_check_passes_on_the_locked_mamba_set():
    check_versions(dict(_MAMBA_LOCKED), expected=_MAMBA_EXPECTED)  # no raise


@pytest.mark.parametrize("pkg", ["mamba-ssm", "causal-conv1d"])
def test_mamba_check_raises_on_drift_or_absent(pkg):
    """A mamba pin drift OR an absent mamba package (the repo-.venv reality) fails loud."""
    for bad_value in ("9.9.9", "ABSENT"):
        bad = dict(_MAMBA_LOCKED)
        bad[pkg] = bad_value
        with pytest.raises(TrainingEnvMismatch) as e:
            check_versions(bad, expected=_MAMBA_EXPECTED)
        assert pkg in str(e.value)
