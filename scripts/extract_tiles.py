"""CLI: extract sub-C tiles for a region.

Usage:
    uv run python scripts/extract_tiles.py \
        --region singapore \
        [--release 2026-04-15.0] \
        [--output-dir data/processed/sub_c/<release>/<region>/] \
        [--pool-size 1] \
        [--rerun <tile_i,tile_j>] \
        [--rerun-reason <str>]

Per spec §15.2 and §11.8 write order:
    1. Parse args; resolve defaults.
    2. Capture git commit sha (abort if git unavailable).
    3. load_region(region) from sub-A (cache-hit ~1s expected).
    4. Full extraction via extract_region() OR single-tile --rerun.
    5. validate_extraction_cross_tile(output_dir).
    6. Write _SUCCESS iff validator passes.
    7. Exit non-zero on any failure; do NOT write _SUCCESS.

_SUCCESS is written LAST per spec §11.8 and ONLY iff the cross-tile validator
passes. extract_region() does NOT write _SUCCESS — that is exclusively this
script's responsibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configure before any cfm imports so cfm modules inherit the level.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("extract_tiles")

# ---------------------------------------------------------------------------
# Repo root resolution (scripts/ lives one level below repo root)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

# Default config paths (resolved from repo root, not cwd).
_DEFAULT_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_DEFAULT_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"
_DEFAULT_RELEASE_PIN_YAML = _REPO_ROOT / "configs" / "data" / "overture_release.yaml"


def _load_pinned_release() -> str:
    """Read the sub-A pinned release string from overture_release.yaml."""
    import yaml

    if not _DEFAULT_RELEASE_PIN_YAML.exists():
        raise FileNotFoundError(f"overture_release.yaml not found at {_DEFAULT_RELEASE_PIN_YAML}")
    data = yaml.safe_load(_DEFAULT_RELEASE_PIN_YAML.read_text(encoding="utf-8"))
    return str(data["release"])


def _capture_commit_sha() -> str:
    """Run git rev-parse HEAD; abort with RuntimeError if git fails.

    Per spec §17: RuntimeError from CLI if git rev-parse HEAD fails.
    """
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract sub-C tiles for a region (spec §15.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--region",
        required=True,
        help="Region name, e.g. 'singapore'.",
    )
    p.add_argument(
        "--release",
        default=None,
        help="Overture release string, e.g. '2026-04-15.0'. Defaults to sub-A pinned release.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help=(
            "Output directory. Defaults to "
            "data/processed/sub_c/<release>/<region>/ relative to repo root."
        ),
    )
    p.add_argument(
        "--pool-size",
        type=int,
        default=1,
        dest="pool_size",
        help="Process-pool size for per-tile extraction (default: 1 = sequential).",
    )
    p.add_argument(
        "--rerun",
        default=None,
        help=(
            "Re-extract a single tile, e.g. '--rerun 12,17'. "
            "STUB: raises NotImplementedError. Full implementation deferred — "
            "not on the critical path for Singapore Phase 1."
        ),
    )
    p.add_argument(
        "--rerun-reason",
        default="rerun",
        dest="rerun_reason",
        help="Free-form audit string for --rerun; included in provenance sha.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success, 1 = failure)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Resolve release -----------------------------------------------
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

    # ---- Resolve output_dir --------------------------------------------
    output_dir: Path
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _REPO_ROOT / "data" / "processed" / "sub_c" / release / args.region
    log.info("Output dir: %s", output_dir)

    # ---- --rerun stub --------------------------------------------------
    if args.rerun is not None:
        # DECISION: --rerun path is a placeholder stub per Task 14 spec.
        # Implementing single-tile re-extraction requires updating only that
        # tile's provenance.yaml + manifest.tiles[] entry in place, then
        # re-running the cross-tile validator. This is a non-trivial
        # orchestration concern; spec allows deferral for Singapore Phase 1.
        # Revisit when single-tile re-extraction is required operationally.
        raise NotImplementedError(
            "--rerun is not yet implemented. "
            "Single-tile re-extraction deferred per Task 14 spec note. "
            "For now, re-run the full extraction to update a tile."
        )

    # ---- Capture commit SHA --------------------------------------------
    try:
        commit_sha = _capture_commit_sha()
        log.info("Commit SHA: %s", commit_sha)
    except RuntimeError as exc:
        log.error("Cannot determine commit SHA: %s", exc)
        return 1

    # ---- Import cfm modules (after sys.path is set up by uv) -----------
    try:
        from cfm.data.overture import load_region
        from cfm.data.sub_c.errors import TileValidationError
        from cfm.data.sub_c.manifest import write_success_marker
        from cfm.data.sub_c.pipeline import extract_region
        from cfm.data.sub_c.validator_cross_tile import validate_extraction_cross_tile
    except ImportError as exc:
        log.error("Import error: %s", exc)
        return 1

    # ---- Load region (sub-A cache; ~1s on cache-hit) -------------------
    log.info("Loading region: %s (release=%s)", args.region, release)
    try:
        region = load_region(args.region)
    except Exception as exc:
        log.error("load_region(%r) failed: %s", args.region, exc)
        return 1

    # ---- Full extraction -----------------------------------------------
    log.info(
        "Starting extraction: region=%s release=%s pool_size=%d output_dir=%s",
        args.region,
        release,
        args.pool_size,
        output_dir,
    )

    # Delete stale _SUCCESS before extracting (full re-extraction protocol).
    success_path = output_dir / "_SUCCESS"
    if success_path.exists():
        log.info("Removing stale _SUCCESS before full extraction.")
        success_path.unlink()

    try:
        extract_region(
            region,
            output_dir,
            policy_yaml_path=_DEFAULT_POLICY_YAML,
            vocab_yaml_path=_DEFAULT_VOCAB_YAML,
            release=release,
            commit_sha=commit_sha,
            pool_size=args.pool_size,
        )
    except Exception as exc:
        log.error("extract_region failed: %s", exc)
        return 1

    log.info("Extraction complete. Running cross-tile validator...")

    # ---- Cross-tile validator (spec §12.2 gate) -------------------------
    try:
        validate_extraction_cross_tile(output_dir)
    except TileValidationError as exc:
        payload = {
            "tile": exc.tile,
            "invariant": exc.invariant,
            "failed_row": exc.failed_row,
            "detail": exc.detail,
        }
        log.error(
            "Cross-tile validator FAILED: %s",
            json.dumps(payload, indent=2, sort_keys=True),
        )
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except Exception as exc:
        log.error("Cross-tile validator raised unexpected error: %s", exc)
        return 1

    # ---- Write _SUCCESS (LAST per spec §11.8) ---------------------------
    write_success_marker(output_dir)
    log.info("_SUCCESS written. Extraction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
