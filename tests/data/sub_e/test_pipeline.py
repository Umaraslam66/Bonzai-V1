from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_e.pipeline import (
    PipelineConfig,
    derive_region,
)
from tests.data.sub_e._fixtures import _build_synthetic_sub_d_and_sub_c


@pytest.fixture
def synthetic_sub_d_region(tmp_path: Path) -> Path:
    """Build a tiny sub-D-shaped region (2 tiles, valid macro_core) plus sub-C
    crossings/features so sub-E has consistent inputs to read. The fixture
    constructs minimal but valid sub-D output via writers that mirror sub-D's
    schema; sub-E reads it like a real sub-D region.
    """
    return _build_synthetic_sub_d_and_sub_c(tmp_path)


def test_pipeline_happy_path_writes_success_marker(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    derive_region(cfg)
    assert (out_root / "_SUCCESS").exists()
    assert (out_root / "manifest.yaml").exists()


def test_pipeline_aborts_when_sub_d_success_missing(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    (synthetic_sub_d_region / "_SUCCESS").unlink()
    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_halts_on_inline_validator_failure(
    tmp_path: Path, synthetic_sub_d_region: Path, monkeypatch
) -> None:
    """Monkey-patch the derivation function to emit a violating row; assert
    pipeline raises and does NOT write _SUCCESS.
    """
    from cfm.data.sub_e import pipeline as pipeline_mod
    from cfm.data.sub_e.validator_inline import InlineValidationError

    def _bad_derive(*args, **kwargs):
        raise InlineValidationError("synthetic violation")

    monkeypatch.setattr(pipeline_mod, "_validate_or_raise", _bad_derive)

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=False,
    )
    with pytest.raises(InlineValidationError):
        derive_region(cfg)
    assert not (out_root / "_SUCCESS").exists()


def test_pipeline_lever_3_collapse_uniformly_null_boundary_class(
    tmp_path: Path, synthetic_sub_d_region: Path
) -> None:
    import pyarrow.parquet as pq

    out_root = tmp_path / "sub_e_out"
    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=synthetic_sub_d_region.parent.parent / "sub_c" / "singapore",
        sub_d_region_dir=synthetic_sub_d_region,
        output_region_dir=out_root,
        commit_sha="a" * 40,
        lever_3_collapse=True,
    )
    derive_region(cfg)
    # All on-disk boundary_class_enum values should be null in lever-3 mode.
    for tile_dir in (out_root).glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        values = tbl.column("boundary_class_enum").to_pylist()
        assert all(v is None for v in values), f"non-null in lever-3 at {tile_dir}"
