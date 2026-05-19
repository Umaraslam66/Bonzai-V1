"""CLI: validate an existing sub-D macro plan region (Task 14).

Usage:
    uv run python scripts/validate_macro_plan.py \\
        --output-dir data/processed/sub_d/2026-04-15.0/singapore \\
        --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore

Per the Task 14 plan: thin wrapper over
``cfm.data.sub_d.validator.validate_region``. Exits 0 if the sub-D region
passes every contract check; exits 1 with the error message on stderr
otherwise.

The validator's checks are documented in
``src/cfm/data/sub_d/validator.py`` (digest chain, B6 config copy, version
namespaces via compare_version, lattice cardinality, masked-slot target
rule, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("validate_macro_plan")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate an existing sub-D macro plan region (Task 14).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Path to the sub-D region directory to validate.",
    )
    p.add_argument(
        "--sub-c-dir",
        dest="sub_c_dir",
        required=True,
        help="Path to the upstream sub-C region directory.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    sub_c_dir = Path(args.sub_c_dir)

    log.info("Validating sub-D region: %s", output_dir)
    log.info("Against sub-C region:    %s", sub_c_dir)

    if not output_dir.exists():
        log.error("Sub-D output dir does not exist: %s", output_dir)
        return 1
    if not sub_c_dir.exists():
        log.error("Sub-C dir does not exist: %s", sub_c_dir)
        return 1

    try:
        from cfm.data.sub_d.errors import SubDValidationError
        from cfm.data.sub_d.validator import validate_region
    except ImportError as exc:
        log.error("Import error: %s", exc)
        return 1

    try:
        validate_region(output_dir, sub_c_dir)
    except SubDValidationError as exc:
        log.error("Validation FAILED: %s", exc)
        return 1
    except Exception as exc:
        log.error("Validation raised unexpected error: %s", exc)
        return 1

    log.info("Validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
