from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from cfm.data.overture.region import BboxScope, SizeEstimate


class OvertureBackend(Protocol):
    """Reads Overture theme parquet for a bounding box at a given release.

    Phase 1 backends apply only the bounding-box filter. The region's
    admin polygon is NOT used here; downstream consumers apply it for
    precise clipping. See docs/data/handoffs.md.
    """

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table: ...

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate: ...


class LocalFixtureBackend:
    """Reads from tests/fixtures/overture_mini/. Ignores bbox and release.

    Used by the fast test suite. The fixtures are committed parquets generated
    by scripts/snapshot_overture_fixtures.py.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = Path(fixtures_dir)

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        return pq.read_table(path)

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        meta = pq.read_metadata(path)
        return SizeEstimate(rows=meta.num_rows, bytes=path.stat().st_size)
