"""Tests for the sub-D sub-C sidecar reader.

The reader is a *pure consumer* of sub-C's on-disk artifacts: it never reaches
back into sub-C internals. Digest anchors are computed over raw file bytes so
they pin sub-C's output bytes, not its writer implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml, write_parquet
from cfm.data.sub_d.sub_c_reader import (
    SubCReaderError,
    SubCTileInputs,
    SubCTilePaths,
    iter_sub_c_tile_paths,
    read_sub_c_manifest,
    read_sub_c_tile_inputs,
)


def _build_fake_sub_c_region(
    root: Path,
    *,
    region: str = "singapore",
    region_crs: str = "EPSG:3414",
    tile_keys: Iterable[tuple[int, int]] = ((0, 0),),
    write_success: bool = True,
) -> Path:
    """Build a minimal valid sub-C region directory tree for tests.

    Layout (matches sub-C's pipeline):
        <region>/_SUCCESS
        <region>/manifest.yaml          (with tiles[] inventory)
        <region>/tile=<EPSG_LABEL>_i{i}_j{j}/{cells,features,crossings}.parquet
        <region>/tile=<EPSG_LABEL>_i{i}_j{j}/{meta,provenance}.yaml
    """
    region_dir = root / region
    region_dir.mkdir(parents=True, exist_ok=True)
    epsg_label = region_crs.replace(":", "")

    tiles_inventory: list[dict] = []
    for tile_i, tile_j in tile_keys:
        tile_dir = region_dir / f"tile={epsg_label}_i{tile_i}_j{tile_j}"
        tile_dir.mkdir()
        write_parquet(pa.table({"cell_i": [tile_i], "cell_j": [tile_j]}), tile_dir / "cells.parquet")
        write_parquet(pa.table({"feature_class": [0]}), tile_dir / "features.parquet")
        write_parquet(pa.table({"axis": [0]}), tile_dir / "crossings.parquet")
        meta_content = {
            "schema_version": "1.0",
            "tile_i": tile_i,
            "tile_j": tile_j,
            "aggregates": {"kept_cell_count": 1},
        }
        (tile_dir / "meta.yaml").write_text(canonicalize_yaml(meta_content), encoding="utf-8")
        provenance_content = {
            "schema_version": "1.0",
            "tile_i": tile_i,
            "tile_j": tile_j,
            "extraction": {"extracted_utc": "2026-04-15T00:00:00Z"},
            "outputs": {"cells_parquet_sha256": "deadbeef"},
        }
        (tile_dir / "provenance.yaml").write_text(
            canonicalize_yaml(provenance_content), encoding="utf-8"
        )
        tiles_inventory.append(
            {"tile_i": tile_i, "tile_j": tile_j, "provenance_sha256": "deadbeef"}
        )

    manifest = {
        "schema_version": "1.0",
        "release": "2026-04-15.0",
        "region": region,
        "region_crs": region_crs,
        "tiles": tiles_inventory,
    }
    (region_dir / "manifest.yaml").write_text(canonicalize_yaml(manifest), encoding="utf-8")

    if write_success:
        (region_dir / "_SUCCESS").write_bytes(b"")

    return region_dir


def test_reader_refuses_region_without_success_marker(tmp_path: Path):
    region_dir = _build_fake_sub_c_region(tmp_path, write_success=False)
    with pytest.raises(SubCReaderError):
        read_sub_c_manifest(region_dir)
    # iter_sub_c_tile_paths goes through read_sub_c_manifest -> also refuses.
    with pytest.raises(SubCReaderError):
        iter_sub_c_tile_paths(region_dir)


def test_reader_uses_manifest_tile_inventory_not_filesystem_glob(tmp_path: Path):
    # Manifest claims two tiles: (0, 0) and (3, 5).
    region_dir = _build_fake_sub_c_region(tmp_path, tile_keys=[(0, 0), (3, 5)])
    # Add a stray tile directory NOT in the manifest. A glob-based reader
    # would pick this up; a manifest-driven reader must not.
    (region_dir / "tile=EPSG3414_i9_j9").mkdir()

    paths = iter_sub_c_tile_paths(region_dir)
    assert {(p.tile_i, p.tile_j) for p in paths} == {(0, 0), (3, 5)}
    # Order follows manifest's sorted tiles[] (sub-C sorts by (i, j)).
    assert [(p.tile_i, p.tile_j) for p in paths] == [(0, 0), (3, 5)]
    # Returned paths point inside the tile=... subdir; not the stray.
    assert all("tile=EPSG3414_i9_j9" not in str(p.tile_dir) for p in paths)


def test_reader_loads_tile_parquets_with_parquetfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    region_dir = _build_fake_sub_c_region(tmp_path)
    [paths] = iter_sub_c_tile_paths(region_dir)

    # Monkeypatch ParquetFile to record calls; forbid pq.read_table entirely.
    parquet_file_calls: list[Path] = []
    real_parquet_file = pq.ParquetFile

    def recording_parquet_file(path, *args, **kwargs):
        parquet_file_calls.append(Path(path))
        return real_parquet_file(path, *args, **kwargs)

    def forbidden_read_table(*args, **kwargs):
        raise AssertionError(
            "read_sub_c_tile_inputs must use pq.ParquetFile(path).read(), "
            "never pq.read_table (per the hive-partition tile=... pitfall)."
        )

    monkeypatch.setattr(pq, "ParquetFile", recording_parquet_file)
    monkeypatch.setattr(pq, "read_table", forbidden_read_table)

    inputs = read_sub_c_tile_inputs(paths)

    assert isinstance(inputs, SubCTileInputs)
    assert isinstance(inputs.cells, pa.Table)
    assert isinstance(inputs.features, pa.Table)
    assert isinstance(inputs.crossings, pa.Table)
    # All three parquets went through ParquetFile.
    seen_names = {p.name for p in parquet_file_calls}
    assert seen_names == {"cells.parquet", "features.parquet", "crossings.parquet"}


def test_reader_computes_digest_anchors_for_sub_c_inputs(tmp_path: Path):
    region_dir = _build_fake_sub_c_region(tmp_path)
    [paths] = iter_sub_c_tile_paths(region_dir)
    inputs = read_sub_c_tile_inputs(paths)

    # Anchors are sha256 over the raw file bytes — implementation-independent,
    # pins sub-C's output bytes themselves.
    expected = {
        "cells_parquet_sha256": compute_sha256(paths.cells.read_bytes()),
        "features_parquet_sha256": compute_sha256(paths.features.read_bytes()),
        "crossings_parquet_sha256": compute_sha256(paths.crossings.read_bytes()),
        "meta_yaml_sha256": compute_sha256(paths.meta.read_bytes()),
        "provenance_yaml_sha256": compute_sha256(paths.provenance.read_bytes()),
    }
    assert inputs.digests == expected
    # Path dataclass surfaces the tile directory and the five artefact paths.
    assert isinstance(paths, SubCTilePaths)
    assert paths.tile_dir == paths.cells.parent
