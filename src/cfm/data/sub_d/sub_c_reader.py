"""Pure consumer of sub-C's on-disk artifacts (sub-D Phase B input layer).

This module never imports from ``cfm.data.sub_c.*``: sub-D treats sub-C as an
external producer whose contract is the on-disk layout plus the manifest's
``tiles[]`` inventory. Digest anchors are sha256 over the raw file bytes so
they pin sub-C's output bytes themselves, not its writer implementation: if
sub-C refactors its writers without changing output bytes, the anchors still
match.

On-disk layout (sub-C convention):

    <region_dir>/
        _SUCCESS
        manifest.yaml          # contains tiles[] inventory + region_crs
        tile=<EPSG_LABEL>_i<i>_j<j>/
            cells.parquet
            features.parquet
            crossings.parquet
            meta.yaml
            provenance.yaml

``<EPSG_LABEL>`` is the manifest's ``region_crs`` with the colon stripped
(``EPSG:3414`` -> ``EPSG3414``). Parquet reads use ``pq.ParquetFile(path).read()``
rather than ``pq.read_table`` to side-step the hive-style ``tile=...``
partition inference pitfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.sub_d.errors import SubDValidationError


class SubCReaderError(SubDValidationError):
    """Raised when a sub-C region cannot be safely consumed by sub-D."""


@dataclass(frozen=True)
class SubCTilePaths:
    """Resolved on-disk paths for a single sub-C tile."""

    tile_i: int
    tile_j: int
    tile_dir: Path
    cells: Path
    features: Path
    crossings: Path
    meta: Path
    provenance: Path


@dataclass(frozen=True)
class SubCTileInputs:
    """Loaded sub-C tile artefacts plus digest anchors over the raw bytes."""

    paths: SubCTilePaths
    cells: pa.Table
    features: pa.Table
    crossings: pa.Table
    meta: dict
    provenance: dict
    digests: dict[str, str]


def read_sub_c_manifest(region_dir: Path) -> dict:
    """Load sub-C's ``manifest.yaml`` as a raw dict.

    The ``_SUCCESS`` marker check runs before the manifest read so a region
    that crashed mid-write (manifest present but cross-tile validator never
    ran) is refused without parsing.
    """
    if not (region_dir / "_SUCCESS").exists():
        raise SubCReaderError(
            f"sub-C region {region_dir} has no _SUCCESS marker; refusing to read"
        )
    manifest_path = region_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise SubCReaderError(
            f"sub-C region {region_dir} has no manifest.yaml"
        )
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


def iter_sub_c_tile_paths(region_dir: Path) -> list[SubCTilePaths]:
    """Resolve per-tile paths from the manifest's ``tiles[]`` inventory.

    The manifest is canonical; stray ``tile=...`` directories that are not
    in ``manifest["tiles"]`` are ignored. Tile-directory names are
    reconstructed from ``manifest["region_crs"]`` so a future region with a
    different CRS works without code changes (provided sub-C keeps the
    ``tile=<EPSG_LABEL>_i<i>_j<j>`` naming convention).
    """
    manifest = read_sub_c_manifest(region_dir)
    epsg_label = str(manifest["region_crs"]).replace(":", "")

    paths: list[SubCTilePaths] = []
    for entry in manifest["tiles"]:
        tile_i = int(entry["tile_i"])
        tile_j = int(entry["tile_j"])
        tile_dir = region_dir / f"tile={epsg_label}_i{tile_i}_j{tile_j}"
        paths.append(
            SubCTilePaths(
                tile_i=tile_i,
                tile_j=tile_j,
                tile_dir=tile_dir,
                cells=tile_dir / "cells.parquet",
                features=tile_dir / "features.parquet",
                crossings=tile_dir / "crossings.parquet",
                meta=tile_dir / "meta.yaml",
                provenance=tile_dir / "provenance.yaml",
            )
        )
    return paths


def read_sub_c_tile_inputs(paths: SubCTilePaths) -> SubCTileInputs:
    """Read one sub-C tile's artefacts and snapshot digest anchors.

    Parquet reads go through ``pq.ParquetFile(path).read()`` so the
    ``tile=...`` directory name is not interpreted as a hive partition key.
    Digest anchors are sha256 over the raw file bytes; they answer "what did
    I literally read" and serve as the integrity-chain input to sub-D's
    per-tile provenance.
    """
    cells = pq.ParquetFile(paths.cells).read()
    features = pq.ParquetFile(paths.features).read()
    crossings = pq.ParquetFile(paths.crossings).read()

    meta = yaml.safe_load(paths.meta.read_text(encoding="utf-8"))
    provenance = yaml.safe_load(paths.provenance.read_text(encoding="utf-8"))

    digests: dict[str, str] = {
        "cells_parquet_sha256": compute_sha256(paths.cells.read_bytes()),
        "features_parquet_sha256": compute_sha256(paths.features.read_bytes()),
        "crossings_parquet_sha256": compute_sha256(paths.crossings.read_bytes()),
        "meta_yaml_sha256": compute_sha256(paths.meta.read_bytes()),
        "provenance_yaml_sha256": compute_sha256(paths.provenance.read_bytes()),
    }

    return SubCTileInputs(
        paths=paths,
        cells=cells,
        features=features,
        crossings=crossings,
        meta=meta,
        provenance=provenance,
        digests=digests,
    )
