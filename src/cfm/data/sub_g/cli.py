"""Sub-G CLI logic (importable + testable). Thin scripts wrap derive_main /
validate_main in scripts/sub_g/.

Exit codes (reviewer-set 2026-05-31):
  0 = clean (validator passed, _PHASE1_VALIDATED written)
  1 = quarantine non-empty / sanity-floor breach (validator ran, found defects)
  2 = precondition failure (missing _SUCCESS, etc.)

Logging is to stdout (matches the repo's sub-* CLIs; file logging is operator
responsibility via shell redirect).
"""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from cfm.data.sub_g.pipeline import ChainConfig, run_chain
from cfm.data.sub_g.validator import ValidationResult, validate_region

EXIT_CLEAN = 0
EXIT_QUARANTINE = 1
EXIT_PRECONDITION = 2

_log = logging.getLogger("cfm.data.sub_g.cli")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # provenance only; never block a run on git
        return "unknown"


def build_volatile() -> dict[str, str]:
    """The EXCLUDED-from-digest run metadata (stamped live per run)."""
    return {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "run_uuid": uuid.uuid4().hex,
        "sub_g_commit_sha": _git_sha(),
    }


def exit_code_for(result: ValidationResult) -> int:
    return EXIT_CLEAN if result.passed else EXIT_QUARANTINE


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--region", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--sub-c-region-dir", required=True, type=Path)
    p.add_argument("--sub-d-region-dir", required=True, type=Path)
    p.add_argument("--sub-e-region-dir", required=True, type=Path)
    p.add_argument("--sub-f-region-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)


def derive_main(argv: list[str]) -> int:
    """Run sub-E -> sub-F chain then validate a region."""
    _configure_logging()
    p = argparse.ArgumentParser(description="sub-G: run sub-E->sub-F chain then validate a region")
    _add_common_args(p)
    p.add_argument(
        "--force", action="store_true", help="re-derive sub-E/sub-F even if _SUCCESS present"
    )
    args = p.parse_args(argv)
    cfg = ChainConfig(
        region=args.region,
        release=args.release,
        sub_c_region_dir=args.sub_c_region_dir,
        sub_d_region_dir=args.sub_d_region_dir,
        sub_e_region_dir=args.sub_e_region_dir,
        sub_f_region_dir=args.sub_f_region_dir,
        output_dir=args.output_dir,
        volatile=build_volatile(),
        force=args.force,
    )
    try:
        result = run_chain(cfg)
    except FileNotFoundError as exc:
        _log.error("precondition failure: %s", exc)
        return EXIT_PRECONDITION
    _log.info(
        "sub-G derive: passed=%s groups=%d sanity_floor_violated=%s",
        result.passed,
        result.n_quarantine_groups,
        result.sanity_floor_violated,
    )
    return exit_code_for(result)


def validate_main(argv: list[str]) -> int:
    """Validate an already-materialized region (no chain run)."""
    _configure_logging()
    p = argparse.ArgumentParser(
        description="sub-G: validate an already-materialized region (no chain run)"
    )
    _add_common_args(p)
    args = p.parse_args(argv)
    try:
        result = validate_region(
            args.sub_c_region_dir,
            args.sub_d_region_dir,
            args.sub_e_region_dir,
            args.sub_f_region_dir,
            args.region,
            args.release,
            args.output_dir,
            build_volatile(),
        )
    except FileNotFoundError as exc:
        _log.error("precondition failure: %s", exc)
        return EXIT_PRECONDITION
    _log.info(
        "sub-G validate: passed=%s groups=%d sanity_floor_violated=%s",
        result.passed,
        result.n_quarantine_groups,
        result.sanity_floor_violated,
    )
    return exit_code_for(result)
