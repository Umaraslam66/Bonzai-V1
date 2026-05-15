from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cfm.data.overture.backend import LocalFixtureBackend, OvertureBackend
from cfm.data.overture.region import BboxScope


def test_local_backend_implements_protocol(overture_mini_dir: Path) -> None:
    backend: OvertureBackend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    # Static-typing check via assignment to protocol-typed variable plus a runtime call.
    assert hasattr(backend, "read_theme")
    assert hasattr(backend, "estimate_size")


def test_local_backend_reads_each_theme(overture_mini_dir: Path, singapore_bbox: BboxScope) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    for theme in ("buildings", "places", "transportation", "base", "divisions"):
        table = backend.read_theme(theme=theme, bbox=singapore_bbox, release="ignored")
        assert isinstance(table, pa.Table)
        assert table.num_rows > 0


def test_local_backend_estimate_size_is_cheap(
    overture_mini_dir: Path, singapore_bbox: BboxScope
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    est = backend.estimate_size(theme="buildings", bbox=singapore_bbox, release="ignored")
    assert est.rows > 0
    assert est.bytes > 0


def test_local_backend_unknown_theme_raises(
    overture_mini_dir: Path, singapore_bbox: BboxScope
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(FileNotFoundError):
        backend.read_theme(theme="not_a_theme", bbox=singapore_bbox, release="ignored")
