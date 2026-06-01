"""Seam 3: token sequences decode to valid GeoJSON (gate) + accuracy baseline.

GATE (quarantinable, spec Decision 3c): every cell's token_sequence splits on
509/510 into features; each decodes via sub-F's decoder; the decoded geometry
must be OGC-valid (provenance OUTSIDE sub-F) and within a loose structural vertex
bound (250m lattice + margin).

MEASUREMENT (reported region-level, NOT a per-tile gate): per-feature position
error (max vertex distance to ORIGINAL sub-C geometry — Decision 3c baseline =
original, not the canonical intermediate) and angle error (max abs segment-
bearing difference). The validator (T8) aggregates these into p99.9/p95 for the
_PHASE1_ACCURACY_BASELINE and applies the sanity floor.

DEFERRED (diagnostic refinement, not gating): the per-stage DECOMPOSITION of the
error (canonicalization vs direction-quantization vs encode/decode, protocol §5)
is NOT implemented here — it has multiple valid attribution schemes and only the
end-to-end position+angle is needed for the baseline + sanity floor. Re-add in
the baseline-analysis when the accuracy gate is locked (sub-G design §8 trigger).
"""

from __future__ import annotations

import math
from itertools import pairwise

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_g.diagnostics import Diagnostic

_FEATURE, _FEATURE_END = 509, 510
_VERTEX_BOUND_M = 250.0 + 50.0  # cell extent + margin; loose structural bound


def split_cell_into_features(token_sequence: list[int]) -> list[list[int]]:
    """Split a flat cell token list into [509, ..., 510] feature subsequences.

    A trailing run that never reaches 510 is incomplete and dropped (it cannot be
    a well-formed feature; the inline validator owns that failure mode).
    """
    feats: list[list[int]] = []
    cur: list[int] = []
    for tok in token_sequence:
        if tok == _FEATURE:
            cur = [tok]
        elif tok == _FEATURE_END:
            if cur:
                cur.append(tok)
                feats.append(cur)
                cur = []
        elif cur:
            cur.append(tok)
    return feats


def _max_abs_coord(geom: dict) -> float:
    coords = geom["coordinates"]
    if geom["type"] == "Point":
        return max(abs(coords[0]), abs(coords[1]))
    if geom["type"] == "Polygon":
        return max(max(abs(x), abs(y)) for ring in coords for x, y in ring)
    return max(max(abs(x), abs(y)) for x, y in coords)


def _decoded_coords(geom: dict) -> list[tuple[float, float]]:
    if geom["type"] == "Point":
        return [(geom["coordinates"][0], geom["coordinates"][1])]
    if geom["type"] == "Polygon":
        return [(x, y) for x, y in geom["coordinates"][0]]
    return [(x, y) for x, y in geom["coordinates"]]


def _original_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    if geom.geom_type == "Polygon":
        return list(geom.exterior.coords)
    if geom.geom_type in ("LineString", "LinearRing"):
        return list(geom.coords)
    # Multi*: take the first part (positional match mirrors encode_cell's split;
    # full multi-part matching is a baseline-analysis refinement). Recurse so the
    # part goes through the per-type dispatch above — a MultiPolygon's first part
    # is a Polygon, whose flat `.coords` is undefined in shapely (NotImplementedError);
    # the recursion routes it to `.exterior.coords`.
    return _original_coords(geom.geoms[0])


def _segment_bearings(coords: list[tuple[float, float]]) -> list[float]:
    bearings: list[float] = []
    for (x0, y0), (x1, y1) in pairwise(coords):
        bearings.append(math.degrees(math.atan2(y1 - y0, x1 - x0)))
    return bearings


def _accuracy_record(decoded: dict, sub_c_feature: dict) -> dict:
    """Max vertex position error (m) + max segment-bearing error (deg) vs ORIGINAL
    sub-C geometry. Positional vertex match over the common prefix (Decision 3c).
    """
    orig = wkb_loads(bytes(sub_c_feature["geometry"]))
    dec = _decoded_coords(decoded)
    src = _original_coords(orig)
    n = min(len(dec), len(src))
    pos_err = max((math.dist(dec[i], src[i]) for i in range(n)), default=0.0)

    db, sb = _segment_bearings(dec), _segment_bearings(src)
    m = min(len(db), len(sb))
    angle_err = 0.0
    for i in range(m):
        diff = abs(db[i] - sb[i]) % 360.0
        angle_err = max(angle_err, min(diff, 360.0 - diff))
    return {"position_err_m": pos_err, "angle_err_deg": angle_err}


def check_decodability(
    tile_id: str,
    cell: tuple[int, int],
    token_sequence: list[int],
    sub_c_features: list[dict],
) -> tuple[list[Diagnostic], list[dict]]:
    """Returns (gate_diagnostics, per_feature_accuracy_records).

    Accuracy records are matched positionally: decoded[k] <-> sub_c_features[k]
    (encode_tile preserves sub-C row order).
    """
    diags: list[Diagnostic] = []
    errors: list[dict] = []
    for k, ftokens in enumerate(split_cell_into_features(token_sequence)):
        try:
            geom = decode_feature(ftokens)
        except Exception as exc:  # decode failure is a gate failure
            diags.append(
                Diagnostic(
                    tile_id=tile_id,
                    invariant_name="decodable_to_valid_geojson",
                    artifact_left="sub_f.token_sequence",
                    observed_left=f"cell={cell} feature[{k}]",
                    artifact_right="decoder",
                    observed_right=f"{type(exc).__name__}: {exc}",
                    expected_relationship="decode_feature returns valid GeoJSON",
                    spec_clause_citation="PRD §5 'decodable to valid GeoJSON'",
                    signature="decode raised exception",
                )
            )
            continue

        if geom["type"] in ("Polygon", "LineString") and not shape(geom).is_valid:
            diags.append(
                Diagnostic(
                    tile_id=tile_id,
                    invariant_name="decodable_to_valid_geojson",
                    artifact_left="sub_f.token_sequence",
                    observed_left=f"cell={cell} feature[{k}]",
                    artifact_right="shapely.is_valid",
                    observed_right=False,
                    expected_relationship="decoded geometry is OGC-valid",
                    spec_clause_citation="PRD §5 + OGC simple-features",
                    signature="decoded geometry not OGC-valid",
                )
            )

        if _max_abs_coord(geom) > _VERTEX_BOUND_M:
            diags.append(
                Diagnostic(
                    tile_id=tile_id,
                    invariant_name="decoded_vertex_within_cell_bound",
                    artifact_left="sub_f decoded vertex (max abs coord)",
                    observed_left=round(_max_abs_coord(geom), 2),
                    artifact_right="cell bound (m)",
                    observed_right=_VERTEX_BOUND_M,
                    expected_relationship="every decoded vertex within cell extent + margin",
                    spec_clause_citation="sub-D lattice (250m cells) + structural margin",
                    signature="decoded vertex implausibly far from cell",
                )
            )

        if k < len(sub_c_features):
            errors.append(_accuracy_record(geom, sub_c_features[k]))
    return diags, errors
