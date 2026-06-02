from __future__ import annotations

import pytest

from cfm.data.sub_f.vocab import vocab_tag_to_id  # noqa: F401 (ensures src import path works)
from cfm.training.env_lock import (
    LOCKED_LIGHTNING,
    LOCKED_PYDANTIC,
    LOCKED_TORCH,
    TrainingEnvMismatch,
    check_versions,
)

_LOCKED = {"torch": LOCKED_TORCH, "lightning": LOCKED_LIGHTNING, "pydantic": LOCKED_PYDANTIC}


def test_check_passes_on_the_locked_set():
    check_versions(dict(_LOCKED))  # no raise


def test_torch_carries_the_cu121_build_tag():
    """The CUDA build is part of the lock — a cpu/cu124 build is non-comparable."""
    assert LOCKED_TORCH.endswith("+cu121")


@pytest.mark.parametrize("pkg", ["torch", "lightning", "pydantic"])
def test_check_raises_on_any_single_version_drift(pkg):
    """Regime-distinguishing: a drift in ANY one package fails loud, naming it."""
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
