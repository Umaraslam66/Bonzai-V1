"""Sub-G chain runner: subprocess sub-E -> sub-F derive, then validate.

Contract (spec Decision 4):
- precondition: sub-C + sub-D `_SUCCESS` present, else fail loud (sub-G never
  regenerates them);
- per stage in [sub-E, sub-F]: `_SUCCESS` present and not `force` -> skip with
  log; else `subprocess.run([sys.executable, <script>, ...], check=True)`;
- halt on any stage crash (do not proceed past a failed stage);
- then `validate_region` -> finalize (writes `_PHASE1_VALIDATED` iff clean).

Resume-from-`_SUCCESS` makes halt-and-revisit cheap. Invocation is by SUBPROCESS
to the existing CLI scripts (spec §9 #5; clean process boundary, low coupling).
Logging is to stdout (file logging is operator responsibility via shell redirect).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cfm.data.sub_g.validator import ValidationResult, validate_region

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUB_E_SCRIPT = _REPO_ROOT / "scripts" / "derive_boundary_contracts.py"
_SUB_F_SCRIPT = _REPO_ROOT / "scripts" / "sub_f" / "derive.py"


@dataclass(frozen=True)
class ChainConfig:
    region: str
    release: str
    sub_c_region_dir: Path
    sub_d_region_dir: Path
    sub_e_region_dir: Path
    sub_f_region_dir: Path
    output_dir: Path
    volatile: dict[str, str]
    force: bool = False


def _has_success(region_dir: Path) -> bool:
    return (region_dir / "_SUCCESS").exists()


def _require_success(region_dir: Path, stage: str) -> None:
    if not _has_success(region_dir):
        raise FileNotFoundError(
            f"{stage} _SUCCESS missing at {region_dir / '_SUCCESS'}; "
            f"sub-G chain refuses to start (it does not regenerate {stage})"
        )


def _run_derive(
    stage: str, script: Path, args: list[str], output_region_dir: Path, force: bool
) -> None:
    if _has_success(output_region_dir) and not force:
        _log.info("%s: _SUCCESS present, skipping derive (use force=True to re-run)", stage)
        return
    _log.info("%s: running derive via %s", stage, script.name)
    subprocess.run([sys.executable, str(script), *args], check=True)  # halts on CalledProcessError
    if not _has_success(output_region_dir):
        raise RuntimeError(
            f"{stage} derive returned 0 but wrote no _SUCCESS at {output_region_dir}"
        )


def run_chain(cfg: ChainConfig) -> ValidationResult:
    """Run sub-E then sub-F on the region (resume-from-_SUCCESS), then validate."""
    _require_success(cfg.sub_c_region_dir, "sub-C")
    _require_success(cfg.sub_d_region_dir, "sub-D")

    _run_derive(
        "sub-E",
        _SUB_E_SCRIPT,
        [
            "--release",
            cfg.release,
            "--region",
            cfg.region,
            "--sub-c-region-dir",
            str(cfg.sub_c_region_dir),
            "--sub-d-region-dir",
            str(cfg.sub_d_region_dir),
            "--output-region-dir",
            str(cfg.sub_e_region_dir),
        ],
        cfg.sub_e_region_dir,
        cfg.force,
    )

    _run_derive(
        "sub-F",
        _SUB_F_SCRIPT,
        [
            "--release",
            cfg.release,
            "--region",
            cfg.region,
            "--sub-c-region-dir",
            str(cfg.sub_c_region_dir),
            "--sub-d-region-dir",
            str(cfg.sub_d_region_dir),
            "--sub-e-region-dir",
            str(cfg.sub_e_region_dir),
            "--output-region-dir",
            str(cfg.sub_f_region_dir),
        ],
        cfg.sub_f_region_dir,
        cfg.force,
    )

    return validate_region(
        cfg.sub_c_region_dir,
        cfg.sub_d_region_dir,
        cfg.sub_e_region_dir,
        cfg.sub_f_region_dir,
        cfg.region,
        cfg.release,
        cfg.output_dir,
        cfg.volatile,
    )
