from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cfm.data.overture.backend import S3DuckDBBackend
from cfm.data.overture.region import BboxScope
from cfm.data.overture.schema import EXPECTED_THEMES, validate_schema


@pytest.fixture(scope="module")
def tiny_singapore_bbox() -> BboxScope:
    # 0.01 deg x 0.01 deg around Bukit Timah / Bishan area.
    return BboxScope.from_tuple((103.85, 1.29, 103.86, 1.30))


@pytest.mark.slow
@pytest.mark.parametrize("theme", EXPECTED_THEMES)
def test_real_s3_theme_returns_schema_matching_table(
    theme: str, tiny_singapore_bbox: BboxScope, repo_root: Path
) -> None:
    """Fetch a tiny real bbox and assert our curated schema is satisfied."""

    # Read the currently pinned release so this test moves with the pin.
    import yaml

    pin = yaml.safe_load((repo_root / "configs" / "data" / "overture_release.yaml").read_text())
    backend = S3DuckDBBackend()
    table = backend.read_theme(theme=theme, bbox=tiny_singapore_bbox, release=pin["release"])
    assert isinstance(table, pa.Table)
    validate_schema(table, theme=theme)
