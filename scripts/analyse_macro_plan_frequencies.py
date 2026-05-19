"""CLI for sub-D macro frequency analysis + Gate 2 proposal generation.

Reads a sub-C region directory, derives Layer-1 evidence per tile, and
writes reviewer-facing YAML artifact(s) to ``--output-dir``.

Default mode writes ``frequency_analysis.yaml`` (the raw analysis dict —
single file, for ad-hoc inspection only).

``--proposal-only`` mode writes the **4+1 Gate 2 artifact layout**:

  zoning_analysis.yaml                            (namespace file)
  cell_density_analysis.yaml                      (namespace file)
  tile_population_density_analysis.yaml           (namespace file)
  road_skeleton_analysis.yaml                     (namespace file)
  macro_vocab_proposal.yaml                       (index file)

Namespace files are content-pinned via sha256 references in the index. The
reviewer at Gate 2 reads the index first, drills into namespace files for
``candidate_strategies`` detail, then hand-edits ``locked_buckets`` /
``locked_proxy`` in the **index** to lock different cuts. Task 8's
``scripts/promote_macro_vocab.py`` flips ``status: proposal`` ->
``status: locked`` on the index file with byte-identity to the rest.

Examples
--------

    uv run python scripts/analyse_macro_plan_frequencies.py \\
      --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \\
      --output-dir reports/phase-1-sub-D

    uv run python scripts/analyse_macro_plan_frequencies.py \\
      --sub-c-dir data/processed/sub_c/2026-04-15.0/singapore \\
      --output-dir reports/phase-1-sub-D \\
      --proposal-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfm.data.sub_d.frequency_analysis import (
    INDEX_ARTIFACT_FILENAME,
    NAMESPACE_ARTIFACT_FILENAMES,
    build_frequency_analysis,
    select_layer3_subset,
    validate_frequency_analysis,
    validate_proposal_index,
    write_frequency_analysis,
    write_proposal_artifacts,
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
        help="Directory to write the analysis or proposal artifact(s) into.",
    )
    parser.add_argument(
        "--proposal-only",
        action="store_true",
        help=(
            "Write the 4+1 Gate 2 artifact layout (4 namespace files + 1 "
            "index file) instead of the single-file frequency_analysis.yaml."
        ),
    )
    parser.add_argument(
        "--max-subset-tiles",
        type=int,
        default=12,
        help=(
            "Max tiles in the Layer-3 subset (only used with --proposal-only). "
            "Default 12 — matches the plan's Task 7 spec and is large enough "
            "for non-trivial per-tile rationale traces while staying review-able."
        ),
    )
    args = parser.parse_args()

    tile_paths = iter_sub_c_tile_paths(args.sub_c_dir)
    inputs = [read_sub_c_tile_inputs(p) for p in tile_paths]
    analysis = build_frequency_analysis(inputs)
    validate_frequency_analysis(analysis)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.proposal_only:
        layer3_subset = select_layer3_subset(analysis, max_tiles=args.max_subset_tiles)
        index = write_proposal_artifacts(
            analysis,
            args.output_dir,
            layer3_subset=layer3_subset,
            status="proposal",
        )
        validate_proposal_index(index)
        namespace_filenames = sorted(NAMESPACE_ARTIFACT_FILENAMES.values())
        print(
            f"wrote {INDEX_ARTIFACT_FILENAME} + {len(namespace_filenames)} namespace "
            f"files to {args.output_dir} "
            f"(tile_count={analysis['tile_count']}, "
            f"subset={len(layer3_subset)}, status=proposal)"
        )
    else:
        output_path = args.output_dir / "frequency_analysis.yaml"
        write_frequency_analysis(analysis, output_path)
        print(f"wrote {output_path} (tile_count={analysis['tile_count']})")


if __name__ == "__main__":
    main()
