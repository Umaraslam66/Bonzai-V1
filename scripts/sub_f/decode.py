#!/usr/bin/env python3
"""scripts/sub_f/decode.py — DEVELOPER INSPECTION TOOL (sub-F T12).

Decode ONE sub-F per-feature token sequence into a bare GeoJSON geometry.

This is NOT a v1 pipeline consumer. The pipeline persona derives + validates a
whole region with `derive.py` + `validate.py`; `encode.py` / `decode.py` exist
to hand-inspect encoder/decoder behaviour on a single feature during
development and debugging (§13.1 T12 lock).

Contract:
  - Input: a JSON array of integer token IDs, via a positional file path OR
    stdin (so it composes in a pipe).
  - Output: a BARE GeoJSON geometry on stdout (one compact line via
    `serialize_geojson`) — a geometry object, NOT a Feature / FeatureCollection
    wrapper. This is exactly what `encode.py` consumes, so
    `decode.py tokens.json | encode.py --semantic-tag …` round-trips.
  - Errors: malformed JSON or a token list that is not a well-formed
    `<feature>…<feature_end>` sequence → clear stderr message + nonzero exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/derive_boundary_contracts.py
# (parents[2] because this script sits one level deeper, under scripts/sub_f/).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_f.decoder import decode_feature, serialize_geojson  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DEVELOPER INSPECTION TOOL: decode a sub-F per-feature token "
            "sequence into a bare GeoJSON geometry. Not a v1 pipeline consumer "
            "(the pipeline persona uses derive.py + validate.py)."
        )
    )
    parser.add_argument(
        "token_file",
        nargs="?",
        type=Path,
        help="path to a JSON array of integer token IDs; omit to read from stdin",
    )
    args = parser.parse_args(argv)

    raw = args.token_file.read_text() if args.token_file is not None else sys.stdin.read()

    try:
        tokens = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[decode] invalid JSON input: {e}", file=sys.stderr)
        return 1

    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        print(
            "[decode] input must be a JSON array of integer token IDs "
            f"(e.g. [509, 642, ...]); got {type(tokens).__name__}",
            file=sys.stderr,
        )
        return 1

    try:
        geom = decode_feature(tokens)
    except (ValueError, IndexError) as e:
        # decode_feature raises ValueError on missing <feature>/<feature_end>
        # markers and IndexError on a truncated sequence (too short to index).
        print(
            f"[decode] could not decode feature token sequence: {e} "
            "(expected one <feature>…<feature_end> per-feature sequence)",
            file=sys.stderr,
        )
        return 1

    print(serialize_geojson(geom))
    return 0


if __name__ == "__main__":
    sys.exit(main())
