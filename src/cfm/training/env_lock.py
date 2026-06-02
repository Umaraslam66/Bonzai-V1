"""Training-environment comparability lock (Task 0).

The Phase-2 bake-off compares 12 runs (4 architectures x 3 scales) by fitting
scaling curves; the curves are only comparable if every run shares the IDENTICAL
training stack. So torch/lightning/pydantic are pinned (pyproject `training` extra
+ configs/training/env-lock-leonardo-cu121.txt) AND enforced here at the training
entrypoint — turning the lock from a comment that can silently drift into a guard
that fails loud (same shape as the resolution seam reading the frozen marker).

Resolved + A100-verified 2026-06-02 (driver 535.274.02, CUDA 12.2; torch.cuda
matmul OK on NVIDIA A100-SXM-64GB).
"""

from __future__ import annotations

#: The locked versions. torch carries its +cu121 local build tag (the CUDA build
#: is part of the lock — a cpu or cu124 build is a different, non-comparable env).
LOCKED_TORCH: str = "2.5.1+cu121"
LOCKED_LIGHTNING: str = "2.6.5"
LOCKED_PYDANTIC: str = "2.13.4"

_EXPECTED: dict[str, str] = {
    "torch": LOCKED_TORCH,
    "lightning": LOCKED_LIGHTNING,
    "pydantic": LOCKED_PYDANTIC,
}


class TrainingEnvMismatch(Exception):
    """A training-stack version differs from the bake-off comparability lock."""


def check_versions(actual: dict[str, str]) -> None:
    """Pure check: raise iff any actual version differs from the lock.

    Separated from the import-the-packages wrapper so it is testable without the
    training stack installed (the regime-distinguishing test runs anywhere)."""
    drift = {
        pkg: (actual.get(pkg), expected)
        for pkg, expected in _EXPECTED.items()
        if actual.get(pkg) != expected
    }
    if drift:
        lines = [
            f"  {pkg}: found {found!r}, locked {locked!r}" for pkg, (found, locked) in drift.items()
        ]
        raise TrainingEnvMismatch(
            "training-stack version drift breaks bake-off comparability "
            "(all 12 runs must share the identical stack):\n" + "\n".join(lines)
        )


def assert_training_env_locked() -> None:
    """Gather the actual installed versions and enforce the lock. Call at the
    training entrypoint BEFORE any run. Imports the stack lazily so importing this
    module never requires torch to be installed."""
    import lightning
    import pydantic
    import torch

    check_versions(
        {
            "torch": torch.__version__,
            "lightning": lightning.__version__,
            "pydantic": pydantic.__version__,
        }
    )
