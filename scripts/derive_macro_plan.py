"""CLI: derive a sub-D macro plan from a finalised sub-C region (Task 14).

Usage:
    uv run python scripts/derive_macro_plan.py \\
        --region singapore \\
        --release 2026-04-15.0 \\
        --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \\
        --output-dir data/processed/sub_d/2026-04-15.0/singapore \\
        --macro-vocab configs/macro_plan/v1/macro_plan_vocab.yaml

Per the Task 14 plan: thin wrapper over
``cfm.data.sub_d.pipeline.derive_region_macro_plan``. The CLI's only
responsibilities are argument parsing, default-path resolution, and exit-code
translation. All business logic lives in the library.

Default-path resolution:
    --output-dir defaults to ``<repo>/data/processed/sub_d/<release>/<region>/``.
    --output-root overrides the ``<repo>/data/processed/sub_d`` prefix
    (useful in tests; production callers pass --output-dir explicitly).

``--rerun-reason`` is preserved verbatim in the per-tile provenance.yaml's
``extraction.rerun_reason`` field (per sub-C F2 precedent — audit-trail
purpose). There is no ``--rerun <tile>`` flag: per known_issue #7, sub-D
supports full re-derivation only.

``--pool-size`` is accepted for API completeness; currently a no-op because
sequential derivation on Singapore-scale tile counts is fast enough.
Implement when Sweden enrollment makes parallelism necessary.

``--dry-run`` resolves paths and prints them, then exits 0 without writing.
Useful for verifying default-path resolution before a real run.
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
log = logging.getLogger("derive_macro_plan")

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "data" / "processed" / "sub_d"
_DEFAULT_SUB_C_ROOT = _REPO_ROOT / "data" / "processed" / "sub_c"
_DEFAULT_MACRO_VOCAB = _REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Derive sub-D macro plan from a sub-C region (Task 14).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--region", required=True, help="Region name, e.g. 'singapore'.")
    p.add_argument(
        "--release",
        required=True,
        help="Overture release string, e.g. '2026-04-15.0'.",
    )
    p.add_argument(
        "--sub-c-dir",
        dest="sub_c_dir",
        default=None,
        help=(
            "Path to the sub-C region directory. Defaults to "
            "<repo>/data/processed/sub_c/<release>/<region>/."
        ),
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=("Path to write sub-D artifacts. Defaults to <output-root>/<release>/<region>/."),
    )
    p.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help=(
            "Override the default output-dir prefix "
            "(<repo>/data/processed/sub_d). Used to compose --output-dir "
            "from --release + --region. Production callers pass "
            "--output-dir explicitly."
        ),
    )
    p.add_argument(
        "--macro-vocab",
        dest="macro_vocab",
        default=str(_DEFAULT_MACRO_VOCAB),
        help=(
            "Path to the locked macro vocab YAML. Defaults to "
            "<repo>/configs/macro_plan/v1/macro_plan_vocab.yaml."
        ),
    )
    p.add_argument(
        "--commit-sha",
        dest="commit_sha",
        required=True,
        help="40-char commit sha to record in extraction provenance.",
    )
    p.add_argument(
        "--extracted-utc",
        dest="extracted_utc",
        default=None,
        help=(
            "Pinned UTC timestamp (ISO 8601) to record. If omitted, wall-clock "
            "UTC is used. Pin for byte-deterministic re-runs."
        ),
    )
    p.add_argument(
        "--rerun-reason",
        dest="rerun_reason",
        default="initial",
        help=(
            "Reason string for this run (F2 audit-trail). "
            "Common values: 'initial', 'rerun-after-sub-c-bump', "
            "'rerun-after-vocab-bump'."
        ),
    )
    p.add_argument(
        "--rerun-count",
        dest="rerun_count",
        type=int,
        default=0,
        help="Re-run counter (0 for first run).",
    )
    p.add_argument(
        "--pool-size",
        dest="pool_size",
        type=int,
        default=1,
        help=(
            "Per-tile parallelism (default 1). Currently a no-op; sequential "
            "derivation is fast enough for Singapore. Implement when Sweden "
            "enrollment makes parallelism necessary."
        ),
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Resolve paths and print them; do not run the pipeline.",
    )
    return p


def _resolve_sub_c_dir(args: argparse.Namespace) -> Path:
    if args.sub_c_dir is not None:
        return Path(args.sub_c_dir)
    return _DEFAULT_SUB_C_ROOT / args.release / args.region


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    output_root = Path(args.output_root) if args.output_root is not None else _DEFAULT_OUTPUT_ROOT
    return output_root / args.release / args.region


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sub_c_dir = _resolve_sub_c_dir(args)
    output_dir = _resolve_output_dir(args)
    macro_vocab_path = Path(args.macro_vocab)

    log.info("Sub-C input dir: %s", sub_c_dir)
    log.info("Output dir:      %s", output_dir)
    log.info("Macro vocab:     %s", macro_vocab_path)
    log.info("Region/release:  %s / %s", args.region, args.release)
    log.info("rerun_reason:    %s (count=%d)", args.rerun_reason, args.rerun_count)
    if args.pool_size != 1:
        log.warning(
            "--pool-size=%d ignored (currently a no-op; sequential derivation).",
            args.pool_size,
        )

    if args.dry_run:
        log.info("Dry run: paths resolved, exiting without writing.")
        return 0

    try:
        from cfm.data.sub_d.errors import SubDValidationError
        from cfm.data.sub_d.pipeline import derive_region_macro_plan
    except ImportError as exc:
        log.error("Import error: %s", exc)
        return 1

    try:
        derive_region_macro_plan(
            sub_c_region_dir=sub_c_dir,
            output_dir=output_dir,
            macro_vocab_path=macro_vocab_path,
            release=args.release,
            region=args.region,
            commit_sha=args.commit_sha,
            extracted_utc=args.extracted_utc,
            rerun_count=args.rerun_count,
            rerun_reason=args.rerun_reason,
        )
    except SubDValidationError as exc:
        log.error("Derivation FAILED: %s", exc)
        return 1
    except Exception as exc:
        log.error("Derivation raised unexpected error: %s", exc)
        return 1

    log.info("Derivation PASSED. _SUCCESS written at %s", output_dir / "_SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
