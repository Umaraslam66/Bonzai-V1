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
#: triton is torch 2.5.1's companion and torch.compile uses it on EVERY backbone, so
#: it is part of the SHARED lock checked for all 12 runs.
LOCKED_TRITON: str = "3.1.0"
#: The mamba-ssm verify-before-lock GPU verdict (A100, 2026-06-17; fused kernel numerics
#: vs the pure-PyTorch reference, fwd+bwd; torch held 2.5.1+cu121). These pins bind ONLY
#: the mamba backbone — the transformer-ar / discrete-diffusion runs use the repo .venv
#: which has no mamba installed, so they are enforced conditionally (assert_mamba_env_locked,
#: called at mamba-backbone construction), NOT in the shared _EXPECTED.
LOCKED_MAMBA_SSM: str = "2.3.1"
LOCKED_CAUSAL_CONV1D: str = "1.6.2.post1"

#: The stack EVERY bake-off run shares (curves are only comparable if identical).
_EXPECTED: dict[str, str] = {
    "torch": LOCKED_TORCH,
    "lightning": LOCKED_LIGHTNING,
    "pydantic": LOCKED_PYDANTIC,
    "triton": LOCKED_TRITON,
}
#: mamba-only pins — checked when (and only when) a mamba backbone is built.
_MAMBA_EXPECTED: dict[str, str] = {
    "mamba-ssm": LOCKED_MAMBA_SSM,
    "causal-conv1d": LOCKED_CAUSAL_CONV1D,
}


class TrainingEnvMismatch(Exception):
    """A training-stack version differs from the bake-off comparability lock."""


def check_versions(actual: dict[str, str], expected: dict[str, str] | None = None) -> None:
    """Pure check: raise iff any actual version differs from the lock.

    Separated from the import-the-packages wrapper so it is testable without the
    training stack installed (the regime-distinguishing test runs anywhere). ``expected``
    defaults to the shared ``_EXPECTED``; pass ``_MAMBA_EXPECTED`` to reuse the same
    grammar for the conditional mamba pins."""
    expected_lock = _EXPECTED if expected is None else expected
    drift = {
        pkg: (actual.get(pkg), want)
        for pkg, want in expected_lock.items()
        if actual.get(pkg) != want
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
    """Gather the actual installed versions and enforce the SHARED lock. Call at the
    training entrypoint BEFORE any run. Imports the stack lazily so importing this
    module never requires torch to be installed."""
    import lightning
    import pydantic
    import torch
    import triton

    check_versions(
        {
            "torch": torch.__version__,
            "lightning": lightning.__version__,
            "pydantic": pydantic.__version__,
            "triton": triton.__version__,
        }
    )


def assert_mamba_env_locked() -> None:
    """Enforce the mamba-only pins (mamba-ssm, causal-conv1d) at mamba-backbone
    construction. The GPU verify-before-lock verdict (A100, 2026-06-17) locked these
    against the comparability stack; this is the guard that keeps a mamba run on them.

    Kept SEPARATE from ``assert_training_env_locked`` because the transformer-ar /
    discrete-diffusion runs use the repo ``.venv`` which does not install mamba — adding
    these to the shared lock would fail those runs. Reads dist METADATA (not an import),
    so it needs neither the compiled CUDA extension nor the gcc-12 libstdc++ preload — a
    missing package surfaces as a loud lock failure, not an ImportError."""
    import importlib.metadata as md

    actual: dict[str, str] = {}
    for dist in _MAMBA_EXPECTED:
        try:
            actual[dist] = md.version(dist)
        except md.PackageNotFoundError:
            actual[dist] = "ABSENT"
    check_versions(actual, expected=_MAMBA_EXPECTED)
