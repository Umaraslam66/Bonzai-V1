"""Promote a reviewed proposal to the locked macro vocab artifact (Task 8).

Reads the proposal index file as bytes, replaces the single
``status: proposal`` line with ``status: locked``, and writes the result to
the output path. NO other transformation is performed. The
byte-identity-modulo-status-marker test in
``tests/data/sub_d/test_macro_vocab.py`` verifies exactly this contract:
the locked artifact normalized back to ``status: proposal`` must match the
proposal bytes byte-for-byte. Any hand-edit beyond the status marker fails
the test.

Usage::

    uv run python scripts/promote_macro_vocab.py \\
      --proposal reports/phase-1-sub-D/macro_vocab_proposal.yaml \\
      --output configs/macro_plan/v1/macro_plan_vocab.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfm.data.sub_d.macro_vocab import load_macro_vocab

PROPOSAL_STATUS_LINE: bytes = b"status: proposal\n"
LOCKED_STATUS_LINE: bytes = b"status: locked\n"


def promote(proposal_path: Path, output_path: Path) -> None:
    """Flip ``status: proposal`` to ``status: locked`` and write to *output_path*.

    Raises ``ValueError`` if the proposal does not contain exactly one
    ``status: proposal`` line — that is a malformed or already-locked
    artifact and must not be silently re-promoted.
    """
    proposal_bytes = proposal_path.read_bytes()
    count = proposal_bytes.count(PROPOSAL_STATUS_LINE)
    if count != 1:
        raise ValueError(
            f"proposal at {proposal_path} contains {count} 'status: proposal' lines; "
            "expected exactly 1. Promotion refused."
        )
    locked_bytes = proposal_bytes.replace(PROPOSAL_STATUS_LINE, LOCKED_STATUS_LINE, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(locked_bytes)

    # Round-trip-validate the locked artifact so any structural issue surfaces
    # at promotion time, not later in the pipeline.
    load_macro_vocab(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a reviewed sub-D macro vocab proposal to a locked artifact.",
    )
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    promote(args.proposal, args.output)
    print(f"locked artifact written: {args.output}")


if __name__ == "__main__":
    main()
