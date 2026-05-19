"""Session-scoped fixtures for sub-C tests (spec §13.2 P5).

The torture-tile extraction runs ONCE per pytest session and is shared across
test modules in ``tests/data/sub_c/``.  Tests that need to corrupt the output
MUST copy this directory to their own ``tmp_path`` before modifying anything
(pattern used in Task 16 diagnostic tests).

History: this fixture was originally defined inline in
``test_fixture_builders.py`` (Task 15).  Task 16 promoted it to conftest.py so
that the new test modules (``test_pipeline_torture_tile.py``, the unskipped
``test_inline_validator_passes_on_clean_torture_tile_output`` in
``test_validator_inline.py``) can request it without per-module re-extraction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_c.pipeline import extract_region
from tests.fixtures.sub_c.build_torture_tile import build_torture_region

# ---------------------------------------------------------------------------
# Config paths (repo-root relative)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"

# Fixed timestamps for byte-deterministic extraction in tests.
_EXTRACTED_UTC = "2026-05-18T00:00:00Z"
_COMMIT_SHA = "b86c509" + "0" * 33  # canonical 40-char sha


@pytest.fixture(scope="session")
def torture_tile_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the torture-tile synthetic region once per pytest session.

    Returns the output directory. All torture-tile tests share this session-
    level extraction to avoid the (small but non-zero) overhead of re-running
    the pipeline for each test.

    Tests that need to corrupt the output MUST copy this directory to their
    own ``tmp_path`` before modifying anything (pattern used in
    ``test_pipeline_torture_tile.py``).
    """
    out = tmp_path_factory.mktemp("torture_tile_session", numbered=False)
    region = build_torture_region()
    extract_region(
        region,
        out,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-05-18.torture",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=1,
    )
    return out
