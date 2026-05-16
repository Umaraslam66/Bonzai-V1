from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.overture.region import BboxScope


@pytest.fixture(scope="session")
def overture_mini_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "overture_mini"


@pytest.fixture(scope="session")
def singapore_bbox() -> BboxScope:
    return BboxScope.from_tuple((103.6, 1.16, 104.05, 1.48))
