"""T12 CLI: scripts/sub_f/derive.py — explicit PipelineConfig field mapping.

Contract under test (§13.1 T12 lock + the explicit-enumeration fix):
  - 6 required args (--release, --region, --sub-{c,d,e}-region-dir,
    --output-region-dir) map to PipelineConfig by EXPLICIT field enumeration
    (NOT PipelineConfig(**vars(args)), which silently breaks the moment a
    non-field arg like --verbose is added).
  - --extracted-utc (optional, default None → live clock) lets a run be made
    byte-reproducible. --no-alpha-drop-report flips run_alpha_drop_report off.
  - Defaults: extracted_utc None, run_alpha_drop_report True (matching
    PipelineConfig defaults).

The field-mapping tests capture the PipelineConfig handed to a stubbed
derive_region (the CLI's own job is the mapping; derive_region itself is
covered by test_pipeline.py). One end-to-end test runs the REAL derive_region
to prove the glue actually composes and writes _SUCCESS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.sub_f.derive as derive_cli
from scripts.sub_f.derive import main
from tests.data.sub_f.test_pipeline import _build_region_inputs


@pytest.fixture
def capture_cfg(monkeypatch):
    box = {}

    def _stub(cfg):
        box["cfg"] = cfg

    monkeypatch.setattr(derive_cli, "derive_region", _stub)
    return box


def _required_args(tmp_path) -> list[str]:
    return [
        "--release",
        "2026-04-15.0",
        "--region",
        "singapore",
        "--sub-c-region-dir",
        str(tmp_path / "c"),
        "--sub-d-region-dir",
        str(tmp_path / "d"),
        "--sub-e-region-dir",
        str(tmp_path / "e"),
        "--output-region-dir",
        str(tmp_path / "out"),
    ]


def test_derive_maps_all_config_fields(capture_cfg, tmp_path):
    rc = main(
        [
            *_required_args(tmp_path),
            "--extracted-utc",
            "2026-05-30T00:00:00Z",
            "--no-alpha-drop-report",
        ]
    )
    assert rc == 0
    cfg = capture_cfg["cfg"]
    assert cfg.release == "2026-04-15.0"
    assert cfg.region == "singapore"
    assert cfg.sub_c_region_dir == tmp_path / "c"
    assert cfg.sub_d_region_dir == tmp_path / "d"
    assert cfg.sub_e_region_dir == tmp_path / "e"
    assert cfg.output_region_dir == tmp_path / "out"
    assert cfg.extracted_utc == "2026-05-30T00:00:00Z"
    assert cfg.run_alpha_drop_report is False
    # Path-typed args really are Paths (argparse type=Path).
    assert isinstance(cfg.sub_c_region_dir, Path)


def test_derive_defaults(capture_cfg, tmp_path):
    rc = main(_required_args(tmp_path))
    assert rc == 0
    cfg = capture_cfg["cfg"]
    assert cfg.extracted_utc is None
    assert cfg.run_alpha_drop_report is True


def test_derive_missing_required_arg_exits_2(capture_cfg, tmp_path):
    # Drop --region.
    args = _required_args(tmp_path)
    idx = args.index("--region")
    del args[idx : idx + 2]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_derive_end_to_end_creates_success(tmp_path):
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    rc = main(
        [
            "--release",
            "2026-04-15.0",
            "--region",
            "singapore",
            "--sub-c-region-dir",
            str(sub_c),
            "--sub-d-region-dir",
            str(sub_d),
            "--sub-e-region-dir",
            str(sub_e),
            "--output-region-dir",
            str(out),
            "--extracted-utc",
            "2026-05-30T00:00:00Z",
            "--no-alpha-drop-report",
        ]
    )
    assert rc == 0
    assert (out / "_SUCCESS").exists()
    assert (out / "manifest.yaml").exists()
