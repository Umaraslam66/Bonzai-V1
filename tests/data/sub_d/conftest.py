"""Session-scoped fixtures for sub-D tests.

The cache-availability skip pattern mirrors sub-C's ``conftest.py``: any test
that requires the locally cached sub-C Singapore extraction requests the
``cached_sub_c_singapore_dir`` fixture, which skips when the cache is
absent. This keeps Layer-3 integration tests usable in CI environments
without the cache, and on contributor machines that haven't run sub-C yet.

The fixture returns the on-disk path; tests treat it as read-only and copy
to ``tmp_path`` if they need to mutate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHED_SUB_C_RELEASE = "2026-04-15.0"
_CACHED_SUB_C_SINGAPORE_DIR = (
    _REPO_ROOT / "data" / "processed" / "sub_c" / _CACHED_SUB_C_RELEASE / "singapore"
)


@pytest.fixture(scope="session")
def cached_sub_c_singapore_dir() -> Path:
    """Return the cached sub-C Singapore region directory, or skip the test.

    Skip predicate is the presence of ``_SUCCESS`` (sub-C's green-light
    marker). A region whose extraction crashed mid-write has artifacts but
    no marker — sub-D refuses to consume it, so the test correctly skips
    rather than reporting a false failure.
    """
    success_marker = _CACHED_SUB_C_SINGAPORE_DIR / "_SUCCESS"
    if not success_marker.is_file():
        pytest.skip(
            f"cached sub-C Singapore output not found at "
            f"{_CACHED_SUB_C_SINGAPORE_DIR} (no _SUCCESS marker). "
            "Run sub-C extraction first or pass -m 'not slow' to skip "
            "Layer-3 integration tests."
        )
    return _CACHED_SUB_C_SINGAPORE_DIR
