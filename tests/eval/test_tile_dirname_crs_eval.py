"""Task-9 step-0 — eval-side tile_dirname CRS-label regression (twin of the Task-8 fix).

Both eval read sites (`geometry.py` holdout-density + `holdout/pipeline.py`
`generate_eval_set`) built per-tile dir names with `tile_dirname(ti, tj)`, which DEFAULTS
`epsg_label` to the Singapore `EPSG3414`. For an EU region that resolves to a dir that
does not exist: `pipeline.py` would `FileNotFoundError`, and `geometry.py` would SILENTLY
return a vacuous 0.0 (every tile skipped by the `if not cells_path.exists(): continue`).

Each test asserts the downstream reader is handed a path carrying the REGION's CRS label
(EPSG25832), and is RED-ON-DIVERGENCE: reverting the call to the defaulted
`tile_dirname(ti, tj)` makes the path carry EPSG3414 and fails the assertion.
"""

from __future__ import annotations

import pytest
import yaml

_RELEASE = "2026-04-15.0"


# =========================================================================== #
# geometry.py — holdout_polygons_per_active_cell (the gate-input-(i) read path)
# =========================================================================== #
def test_geometry_holdout_density_uses_region_crs_label(tmp_path, monkeypatch):
    from cfm.eval import geometry as G

    # 1-tile munich manifest the function reads via holdout_manifest_for_region(...).read_text()
    manifest_p = tmp_path / "holdout.yaml"
    manifest_p.write_text(
        yaml.safe_dump({"regions": {"munich": {"tiles": [{"tile_i": 5, "tile_j": 9}]}}})
    )

    # geometry.py imports these LOCALLY from cfm.eval.holdout.paths -> patch the source.
    import cfm.eval.holdout.paths as P

    monkeypatch.setattr(P, "holdout_manifest_for_region", lambda release, region: manifest_p)
    monkeypatch.setattr(P, "sub_f_region_dir", lambda release, region: tmp_path)
    monkeypatch.setattr(P, "epsg_label_for_region", lambda region: "EPSG25832")

    # Pre-create ONLY the correctly-labelled tile dir + cells.parquet so cells_path.exists()
    # is True iff the region label is used. (Defaulted EPSG3414 dir is absent -> skipped.)
    (tmp_path / "tile=EPSG25832_i5_j9").mkdir()
    (tmp_path / "tile=EPSG25832_i5_j9" / "cells.parquet").write_bytes(b"")

    captured: dict[str, str] = {}

    def fake_read_sub_f_cells(path):
        captured["path"] = str(path)
        return {}  # empty -> inner decode loop is a no-op

    # read_sub_f_cells is imported locally from cfm.data.sub_g.readers -> patch the source.
    import cfm.data.sub_g.readers as R

    monkeypatch.setattr(R, "read_sub_f_cells", fake_read_sub_f_cells)

    G.holdout_polygons_per_active_cell(release=_RELEASE, region="munich")

    assert "path" in captured, "read_sub_f_cells was never called -> SG-default dir was used (RED)"
    assert "tile=EPSG25832_i5_j9" in captured["path"], captured["path"]
    assert "EPSG3414" not in captured["path"]


# =========================================================================== #
# holdout/pipeline.py — generate_eval_set tile read
# =========================================================================== #
def test_pipeline_generate_eval_set_uses_region_crs_label(tmp_path, monkeypatch):
    from cfm.eval.holdout import pipeline as PL

    monkeypatch.setattr(
        PL,
        "_load_inventory",
        lambda release, region: [{"tile_i": 5, "tile_j": 9, "provenance_sha256": "x"}],
    )
    import cfm.eval.holdout.paths as P

    monkeypatch.setattr(P, "sub_d_region_dir", lambda release, region: tmp_path / "sd")
    monkeypatch.setattr(P, "sub_f_region_dir", lambda release, region: tmp_path / "sf")
    monkeypatch.setattr(P, "epsg_label_for_region", lambda region: "EPSG25832")

    captured: dict[str, str] = {}

    class _Stop(Exception):
        pass

    def fake_read_tile_labels(tile_dir, *, tile_i, tile_j):
        captured["dir"] = str(tile_dir)
        raise _Stop  # short-circuit before the heavy co-optimization

    monkeypatch.setattr(PL, "read_tile_labels", fake_read_tile_labels)

    with pytest.raises(_Stop):
        PL.generate_eval_set(release=_RELEASE, region="munich")

    assert "tile=EPSG25832_i5_j9" in captured["dir"], captured["dir"]
    assert "EPSG3414" not in captured["dir"]
