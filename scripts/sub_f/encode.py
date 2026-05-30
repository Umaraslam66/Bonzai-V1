#!/usr/bin/env python3
"""scripts/sub_f/encode.py — DEVELOPER INSPECTION TOOL (sub-F T12).

Encode ONE GeoJSON geometry into a sub-F per-feature token sequence.

This is NOT a v1 pipeline consumer. Production encoding runs through
`encode_tile` inside `derive.py`; `encode.py` / `decode.py` exist to hand-inspect
encoder/decoder behaviour on a single feature during development and debugging
(§13.1 T12 lock).

Contract:
  - Input: ONE GeoJSON geometry (a geometry object, not a Feature) via
    --geom-file PATH or stdin.
  - --semantic-tag REQUIRED ("key=value", e.g. highway=residential); passed
    straight through to encode_feature.
  - --inbound-bref / --outbound-bref take DIR:CLASS (e.g. E:MAJOR_ROAD) and are
    resolved via boundary_contract.resolve_bref_tag — the exact production path
    (encode_cell → _classify_feature_for_bref → resolve_bref_tag). Setting a
    bref forces Case B/C/D so the lossy round-trip can be inspected. The encoder
    does NOT check the bref against the geometry; the inbound/outbound brefs you
    pass are emitted as-is.
  - --cell-origin x,y (default 0,0): a CLI-side pre-translation (translate by
    -x,-y) applied BEFORE canonicalize + encode. encode_feature has no
    cell_origin parameter — it quantizes the anchor straight from coords[0] over
    a 0..250m cell-local range — so positioning a feature within its cell is the
    caller's job here, exactly as encode_cell does it before calling
    encode_feature.
  - --round-trip: encode → decode → report L_inf (max per-vertex |Δx|,|Δy| in
    metres) against the CANONICALIZED input (what was actually encoded;
    Polygons are reordered by canonicalization, so the canonical form is the
    honest baseline) to STDERR. The token array still goes to stdout. For Cases
    B/C/D the L_inf is inflated by the v2-scoped bref vertex by design (§1.4).
  - Output: a JSON array of integer token IDs on stdout, one line. This is what
    `decode.py` reads, so encode/decode round-trip in a pipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely.affinity import translate
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

# iCloud-safe sys.path inject — mirrors scripts/derive_boundary_contracts.py
# (parents[2] because this script sits one level deeper, under scripts/sub_f/).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_f.boundary_contract import resolve_bref_tag  # noqa: E402
from cfm.data.sub_f.decoder import decode_feature  # noqa: E402
from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature  # noqa: E402


class _CliError(Exception):
    """A user-facing CLI error: caught in main(), printed to stderr, exit 1."""


def _parse_bref_flag(value: str, flag: str) -> str:
    """Resolve a DIR:CLASS bref flag to its vocab tag via resolve_bref_tag.

    resolve_bref_tag raises ValueError on an unsupported DIRECTION but returns
    None for a non-emitting CLASS (the `if class_label not in _EMITTING_CLASSES:
    return None` branch). So the flag-provided⟹non-None guard lives here, NOT in
    resolve_bref_tag — otherwise `--inbound-bref E:NOT_A_CLASS` would silently
    encode as Case A instead of erroring.
    """
    if ":" not in value:
        raise _CliError(
            f"{flag} must be DIR:CLASS (e.g. E:MAJOR_ROAD); got {value!r} with no ':' separator"
        )
    direction, class_label = value.split(":", 1)
    try:
        tag = resolve_bref_tag(direction, class_label)
    except ValueError as e:
        # resolve_bref_tag raises on an unsupported direction.
        raise _CliError(f"{flag}: {e}") from e
    if tag is None:
        raise _CliError(
            f"{flag}: class {class_label!r} is not an emitting bref class "
            "(expected MAJOR_ROAD or MINOR_ROAD)"
        )
    return tag


def _parse_cell_origin(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise _CliError(f"--cell-origin must be 'x,y'; got {value!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise _CliError(f"--cell-origin must be 'x,y' floats; got {value!r}") from e


def _read_geometry(geom_file: Path | None) -> BaseGeometry:
    raw = geom_file.read_text() if geom_file is not None else sys.stdin.read()
    try:
        geom_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _CliError(f"invalid JSON geometry input: {e}") from e
    try:
        return shape(geom_dict)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise _CliError(f"input is not a valid GeoJSON geometry: {e}") from e


def _coords_of(geom: BaseGeometry) -> list[tuple[float, float]]:
    """The encoder's input coord list for a single-part feature."""
    gt = geom.geom_type
    if gt == "LineString":
        return [(x, y) for x, y in geom.coords]
    if gt == "Polygon":
        return [(x, y) for x, y in geom.exterior.coords]
    if gt == "Point":
        return [(geom.x, geom.y)]
    raise _CliError(f"--round-trip unsupported for geom_type {gt!r} (split multi-part first)")


def _decoded_coords(geom_dict: dict) -> list[tuple[float, float]]:
    coords = geom_dict["coordinates"]
    if geom_dict["type"] == "Point":
        return [(coords[0], coords[1])]
    return [(c[0], c[1]) for c in coords]


def _round_trip_linf(canon: BaseGeometry, tokens: list[int]) -> tuple[float, int, int]:
    """L_inf (metres) between the canonicalized input and the decoded output."""
    src = _coords_of(canon)
    decoded = _decoded_coords(decode_feature(tokens))
    linf = 0.0
    for (sx, sy), (dx, dy) in zip(src, decoded, strict=False):
        linf = max(linf, abs(sx - dx), abs(sy - dy))
    return linf, len(src), len(decoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DEVELOPER INSPECTION TOOL: encode one GeoJSON geometry into a "
            "sub-F per-feature token sequence. Not a v1 pipeline consumer "
            "(the pipeline persona uses derive.py + validate.py)."
        )
    )
    parser.add_argument(
        "--semantic-tag", required=True, help="'key=value', e.g. highway=residential"
    )
    parser.add_argument("--geom-file", type=Path, help="GeoJSON geometry file; omit to read stdin")
    parser.add_argument("--inbound-bref", help="DIR:CLASS, e.g. E:MAJOR_ROAD (forces Case C/D)")
    parser.add_argument("--outbound-bref", help="DIR:CLASS, e.g. W:MINOR_ROAD (forces Case B/D)")
    parser.add_argument("--cell-origin", default="0,0", help="cell SW corner 'x,y' (default 0,0)")
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="also decode and report L_inf vs the canonicalized input to stderr",
    )
    args = parser.parse_args(argv)

    try:
        origin_x, origin_y = _parse_cell_origin(args.cell_origin)
        inbound = (
            _parse_bref_flag(args.inbound_bref, "--inbound-bref") if args.inbound_bref else None
        )
        outbound = (
            _parse_bref_flag(args.outbound_bref, "--outbound-bref") if args.outbound_bref else None
        )
        geom = _read_geometry(args.geom_file)
        # CLI-side cell-origin pre-translation (faithful to encode_cell, which
        # positions each feature relative to the cell SW corner before encode).
        local = translate(geom, xoff=-origin_x, yoff=-origin_y)
        canon = canonicalize_geometry(local)
        ef = encode_feature(
            canon,
            semantic_tag=args.semantic_tag,
            inbound_bref=inbound,
            outbound_bref=outbound,
        )
    except _CliError as e:
        print(f"[encode] {e}", file=sys.stderr)
        return 1

    print(json.dumps(ef.tokens))

    if args.round_trip:
        linf, n_in, n_out = _round_trip_linf(canon, ef.tokens)
        note = (
            " (incl. v2-scoped bref vertex; inflated by design per §1.4)"
            if ef.case in ("B", "C", "D")
            else ""
        )
        print(
            f"[encode] round-trip case={ef.case} L_inf={linf:.3f}m "
            f"vertices in={n_in} out={n_out}{note}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
