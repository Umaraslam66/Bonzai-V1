"""CLI: validate an existing sub-C tile extraction.

Usage:
    uv run python scripts/validate_extraction.py \
        --region singapore \
        [--release 2026-04-15.0] \
        [--output-dir <path>] \
        [--pool-size 1]

Per spec §15.2:
    - Resolves output_dir = data/processed/sub_c/<release>/<region>/ unless
      --output-dir is given explicitly.
    - Calls validate_extraction_cross_tile(output_dir).
    - Exits 0 on success.
    - Prints TileValidationError JSON-serialized payload to stderr; exits 1 on
      failure.

--pool-size: accepted for API completeness (spec §12.2 mentions parallelism for
Sweden-scale); currently a no-op because sequential validation completes in
< 60 seconds for Singapore's ~150-250 tiles. Implement when Sweden enrollment
reaches tile-count scale that makes parallelism necessary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("validate_extraction")

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_DEFAULT_RELEASE_PIN_YAML = _REPO_ROOT / "configs" / "data" / "overture_release.yaml"


def _load_pinned_release() -> str:
    """Read the sub-A pinned release string from overture_release.yaml."""
    import yaml

    if not _DEFAULT_RELEASE_PIN_YAML.exists():
        raise FileNotFoundError(f"overture_release.yaml not found at {_DEFAULT_RELEASE_PIN_YAML}")
    data = yaml.safe_load(_DEFAULT_RELEASE_PIN_YAML.read_text(encoding="utf-8"))
    return str(data["release"])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate an existing sub-C tile extraction (spec §15.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--region",
        default=None,
        help="Region name, e.g. 'singapore'. Used to resolve default output-dir.",
    )
    p.add_argument(
        "--release",
        default=None,
        help=(
            "Overture release string, e.g. '2026-04-15.0'. "
            "Defaults to sub-A pinned release. Used to resolve default output-dir."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help=(
            "Path to the already-extracted region directory. "
            "Defaults to data/processed/sub_c/<release>/<region>/ relative to repo root. "
            "If provided, --region and --release are not required for dir resolution."
        ),
    )
    p.add_argument(
        "--pool-size",
        type=int,
        default=1,
        dest="pool_size",
        help=(
            "Digest-check parallelism (default: 1). "
            "Currently a no-op; sequential validation is < 60s for Singapore. "
            "Implement when Sweden tile-count makes parallelism necessary."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success, 1 = failure)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Resolve output_dir --------------------------------------------
    output_dir: Path
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        if args.region is None:
            log.error("--region is required unless --output-dir is given explicitly.")
            return 1

        release: str
        if args.release is not None:
            release = args.release
        else:
            try:
                release = _load_pinned_release()
                log.info("Using pinned release: %s", release)
            except Exception as exc:
                log.error("Could not load pinned release: %s", exc)
                return 1

        output_dir = _REPO_ROOT / "data" / "processed" / "sub_c" / release / args.region

    log.info("Validating extraction at: %s", output_dir)

    if not output_dir.exists():
        log.error("Output dir does not exist: %s", output_dir)
        return 1

    # ---- Import cfm modules --------------------------------------------
    try:
        from cfm.data.sub_c.errors import TileValidationError
        from cfm.data.sub_c.validator_cross_tile import validate_extraction_cross_tile
    except ImportError as exc:
        log.error("Import error: %s", exc)
        return 1

    # ---- Run validator -------------------------------------------------
    try:
        validate_extraction_cross_tile(output_dir)
    except TileValidationError as exc:
        payload = {
            "tile": exc.tile,
            "invariant": exc.invariant,
            "failed_row": exc.failed_row,
            "detail": exc.detail,
        }
        log.error("Validation FAILED: %s", exc)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except Exception as exc:
        log.error("Validation raised unexpected error: %s", exc)
        return 1

    log.info("Validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
