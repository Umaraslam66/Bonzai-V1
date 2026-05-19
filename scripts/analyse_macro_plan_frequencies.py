"""CLI for sub-D macro frequency analysis (Task 6).

Reads a sub-C region directory, derives Layer-1 evidence per tile, and
writes a reviewer-facing frequency-analysis YAML to ``--output-dir``.

Example:

    uv run python scripts/analyse_macro_plan_frequencies.py \\
      --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \\
      --output-dir reports/phase-1-sub-D

Task 7 will extend the CLI with deterministic Singapore subset selection and
a ``--proposal-only`` flag for generating the reviewer-facing artifact prior
to Gate 2. Until then, this script just produces ``frequency_analysis.yaml``
from whichever tiles the sub-C manifest claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfm.data.sub_d.frequency_analysis import (
    build_frequency_analysis,
    validate_frequency_analysis,
    write_frequency_analysis,
)
from cfm.data.sub_d.sub_c_reader import (
    iter_sub_c_tile_paths,
    read_sub_c_tile_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive sub-D macro frequency analysis artifacts from a sub-C region.",
    )
    parser.add_argument(
        "--sub-c-dir",
        type=Path,
        required=True,
        help="Path to a sub-C region directory (must contain _SUCCESS and manifest.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the frequency_analysis.yaml artifact into.",
    )
    parser.add_argument(
        "--output-name",
        default="frequency_analysis.yaml",
        help="Filename for the analysis artifact (default: frequency_analysis.yaml).",
    )
    args = parser.parse_args()

    tile_paths = iter_sub_c_tile_paths(args.sub_c_dir)
    inputs = [read_sub_c_tile_inputs(p) for p in tile_paths]
    analysis = build_frequency_analysis(inputs)
    validate_frequency_analysis(analysis)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    write_frequency_analysis(analysis, output_path)
    print(f"wrote {output_path} (tile_count={analysis['tile_count']})")


if __name__ == "__main__":
    main()
