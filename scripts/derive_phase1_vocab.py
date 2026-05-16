"""Derive Phase 1 vocabulary + missing-value policy YAMLs from B1's cached data.

Reads the cached Singapore Overture data via sub-A's loader (cache-hit ~1 s),
applies the locked B2 decisions, and writes both artifacts byte-deterministically.

Usage:
    uv run python scripts/derive_phase1_vocab.py
    uv run python scripts/derive_phase1_vocab.py --output-dir /tmp/test --rerun-reason "rerun-test"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure src/ is on the path when the script is invoked from the repo root
# without a previously-built editable install.

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfm.data.frequency import compute_field_frequencies  # noqa: E402
from cfm.data.overture import load_region  # noqa: E402
from cfm.data.overture.backend import LocalFixtureBackend, S3DuckDBBackend  # noqa: E402
from cfm.data.vocab_derivation import (  # noqa: E402
    canonicalize_yaml,
    compute_yaml_sha256,
    derive_phase1_policy,
    derive_phase1_vocab,
    policy_to_dict,
    vocab_to_dict,
)

# The 5 vocab-relevant fields per the B2 spec §2.
# Each entry: (theme_name, column_path, label_for_FieldFrequencyResult.field, is_list_field).
_FIELD_SPEC = [
    ("buildings", "class", "buildings.class", False),
    ("transportation", "class", "transportation.class", False),
    ("base", "class", "base.class", False),
    ("places", "categories.primary", "places.categories.primary", False),
    ("places", "categories.alternate", "places.categories.alternate", True),
]

_SOURCE_REPORT_PATH = "reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md"
_OVERTURE_RELEASE = "2026-04-15.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-reason", default="initial")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--backend", choices=["real", "fixture"], default="real")
    args = parser.parse_args(argv)

    t0 = time.monotonic()

    # 1. Resolve git commit sha.
    commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # 2. Capture timestamp once.
    run_ts = datetime.now(UTC)

    # 3. Select backend (matches sub-B1's pattern).
    if args.backend == "fixture":
        backend = LocalFixtureBackend(ROOT / "tests" / "fixtures" / "overture_mini")
    else:
        backend = S3DuckDBBackend()

    # 4. Load the cached Singapore region.
    region = load_region("singapore", backend=backend)

    # 5. Compute frequencies for each of the 5 fields.
    field_results: dict = {}
    for theme, column, label, is_list_field in _FIELD_SPEC:
        table = region.themes[theme]
        field_results[label] = compute_field_frequencies(
            table,
            column,
            label=label,
            is_list_field=is_list_field,
        )

    # 6. Derive both artifacts.
    vocab = derive_phase1_vocab(
        field_results=field_results,
        overture_release=_OVERTURE_RELEASE,
        source_report_path=_SOURCE_REPORT_PATH,
        commit_sha=commit_sha,
        run_timestamp_utc=run_ts,
    )
    policy = derive_phase1_policy(
        field_results=field_results,
        overture_release=_OVERTURE_RELEASE,
        source_report_path=_SOURCE_REPORT_PATH,
        commit_sha=commit_sha,
        run_timestamp_utc=run_ts,
    )

    # 7. Serialise to dict, compute sha256, embed, serialise to YAML.
    vocab_dict = vocab_to_dict(vocab)
    vocab_dict["vocab_sha256"] = compute_yaml_sha256(vocab_dict)
    vocab_yaml = canonicalize_yaml(vocab_dict)

    policy_dict = policy_to_dict(policy)
    policy_dict["policy_sha256"] = compute_yaml_sha256(policy_dict)
    policy_yaml = canonicalize_yaml(policy_dict)

    # 8. Write outputs.
    vocab_path = args.output_dir / "configs" / "tokenizer" / "vocab_phase1.yaml"
    policy_path = args.output_dir / "configs" / "data" / "missing_value_policy.yaml"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(vocab_yaml)
    policy_path.write_text(policy_yaml)

    elapsed = time.monotonic() - t0
    print(f"Wrote {vocab_path} ({len(vocab_yaml):,} bytes)")
    print(f"  vocab_sha256: {vocab_dict['vocab_sha256']}")
    print(f"Wrote {policy_path} ({len(policy_yaml):,} bytes)")
    print(f"  policy_sha256: {policy_dict['policy_sha256']}")
    print(f"Total wall-clock: {elapsed:.2f}s")
    print(f"Rerun reason: {args.rerun_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
