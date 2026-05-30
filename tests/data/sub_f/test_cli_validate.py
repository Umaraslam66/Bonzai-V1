"""T12 CLI: scripts/sub_f/validate.py — inline-per-tile + cross-tile (two-arg).

Contract under test (§13.1 T12 lock + the T10 two-arg fix):
  - --region-dir (sub-F region) AND --sub-e-region-dir BOTH required. The
    cross-tile validator gained a second argument this session
    (validate_cross_tile(sub_f_region_dir, sub_e_region_dir)); the master-plan
    one-arg snippet would TypeError. This test pins both args as required.
  - Runs validate_inline on every tile=*/cells.parquet, then
    validate_cross_tile(region_dir, sub_e_region_dir). Success → exit 0.
    Any validation error → clear stderr + exit 1.

Real composition (not mocks): the happy path builds a real valid region via
derive_region, then the negatives CORRUPT that region — an inline failure
(63-row cells.parquet) and a cross-tile failure (duplicated provenance sha).
Leg isolation is by exception-type name in stderr: an inline failure reports
InlineValidationError (NOT CrossTileValidationError) and vice-versa, so one
cannot satisfy the other's assertion.

The region builder is reused from test_pipeline.py rather than copied — the
close-checklist already flags extracting the triplicated builder to a shared
conftest; adding a 4th copy here would worsen that debt. (T12 scope does not
include the conftest-dedup refactor of the 3 existing consumers.)
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest
import yaml

from cfm.data.sub_f.pipeline import derive_region
from scripts.sub_f.validate import main
from tests.data.sub_f.test_pipeline import _build_region_inputs, _make_cfg


@pytest.fixture
def valid_region(tmp_path):
    """A derived, valid 2-tile sub-F region. Returns (sub_f_out_dir, sub_e_dir)."""
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    return out, sub_e


def test_validate_happy_path_returns_zero(valid_region, capsys):
    out, sub_e = valid_region
    rc = main(["--region-dir", str(out), "--sub-e-region-dir", str(sub_e)])
    assert rc == 0
    assert "passed" in capsys.readouterr().out.lower()


def test_validate_missing_sub_e_arg_exits_2(valid_region):
    out, _sub_e = valid_region
    # The T10 two-arg fix: --sub-e-region-dir is REQUIRED (the one-arg snippet
    # would have TypeError'd inside validate_cross_tile).
    with pytest.raises(SystemExit) as exc:
        main(["--region-dir", str(out)])
    assert exc.value.code == 2


def test_validate_inline_failure_returns_nonzero(valid_region, capsys):
    out, sub_e = valid_region
    # Corrupt one tile to 63 rows → validate_inline row-count check fires.
    cells = sorted(out.glob("tile=*/cells.parquet"))[0]
    table = pq.ParquetFile(cells).read()
    pq.write_table(table.slice(0, 63), cells)

    rc = main(["--region-dir", str(out), "--sub-e-region-dir", str(sub_e)])

    assert rc != 0
    err = capsys.readouterr().err
    # Leg isolation: an INLINE failure, not a cross-tile one.
    assert "InlineValidationError" in err
    assert "CrossTileValidationError" not in err


def test_validate_cross_tile_failure_returns_nonzero(valid_region, capsys):
    out, sub_e = valid_region
    # Corrupt provenance so two tiles share a sha → cross-tile sha-uniqueness
    # fires. Inline does NOT read provenance shas, so inline still passes and
    # we genuinely reach the cross-tile leg.
    prov_paths = sorted(out.glob("tile=*/provenance.yaml"))
    assert len(prov_paths) >= 2
    first = yaml.safe_load(prov_paths[0].read_text())
    other = yaml.safe_load(prov_paths[1].read_text())
    other["provenance_sha256"] = first["provenance_sha256"]
    prov_paths[1].write_text(yaml.safe_dump(other))

    rc = main(["--region-dir", str(out), "--sub-e-region-dir", str(sub_e)])

    assert rc != 0
    err = capsys.readouterr().err
    # Leg isolation: a CROSS-TILE failure, not an inline one.
    assert "CrossTileValidationError" in err
    assert "InlineValidationError" not in err


def test_validate_empty_region_returns_nonzero(tmp_path, capsys):
    empty = tmp_path / "empty_region"
    empty.mkdir()
    sub_e = tmp_path / "sub_e"
    sub_e.mkdir()
    rc = main(["--region-dir", str(empty), "--sub-e-region-dir", str(sub_e)])
    assert rc != 0
    assert "no tile" in capsys.readouterr().err.lower()
