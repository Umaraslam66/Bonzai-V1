"""Seam 3: token sequences decode to valid GeoJSON (gate) + accuracy baseline.

GATE (quarantinable, spec Decision 3c): every cell's token_sequence splits on
509/510 into features; each decodes via sub-F's decoder; the decoded geometry
must be OGC-valid (provenance OUTSIDE sub-F) and within a loose structural vertex
bound (250m lattice + margin). EXCEPTION (sub-G T11 H3, reviewer 2026-06-01): the
v1-by-design outbound-bref placeholder collapse -- a V=2 crossing road with no
interior vertex decodes to a zero-length [anchor, anchor] LineString because the
exit vertex is v1-unencoded (decoder.py:13-22, v2-scoped) -- is excluded from the
gate by CONSTRUCTION IDENTITY (_is_bref_placeholder_collapse) and reported as a
COUNT in the accuracy baseline, NOT gated. This is the same v1-unencoded outbound
bref vertex H1 reports-not-gates, in its most degenerate form (same crossing roads
as the baseline's position_full residual). A degenerate geom WITHOUT an outbound
bref still quarantines (guard).

MEASUREMENT (reported region-level, NOT a per-tile gate): per-FEATURE round-trip
error vs the CANONICAL ORIGINAL sub-C geometry (Decision 3c baseline = original;
canonicalize first because the decoder emits canonical-order vertices). Two
position numbers per feature:

  position_full_m  — symmetric vertex Hausdorff over ALL decoded vertices vs the
                     original. INCLUDES the v1-by-design unencoded outbound
                     bref edge-crossing vertex; reported, NOT gated.
  position_core_m  — same, but EXCLUDING that bref-crossing vertex. The sanity
                     floor gates on CORE (reviewer 2026-06-01): the bref vertex
                     carries no encoded position (decoder.py:13-22, v2-scoped per
                     spec §1.4), so gating on it would conflate a designed v1
                     info-loss with "broken encode/decode" and fire on every real
                     region forever. The exclusion is by CONSTRUCTION IDENTITY
                     (the token body ends in a bref -> Case B/D outbound), NEVER
                     by error magnitude — see has_outbound_bref.

  angle_core_deg   — max segment-bearing error vs the canonical original, defined
                     ONLY where decoded and canonical vertex counts match (no
                     chunking subdivision -> 1:1 segment correspondence); None
                     otherwise (caller skips it from the angle distribution).

Pairing: encode_cell splits a Multi* sub-C row into one feature block PER PART
(encoder.py:578-601), so decoded blocks > sub-C rows in a cell. The accuracy loop
walks sub-C features and advances the decoded pointer by #parts of
canonicalize(orig) — NOT a positional decoded[k] <-> sub_c[k] match (that drift
was the 318m/180deg first-measurement artifact; sub-G T11 H1, 2026-06-01).

The full vs canonical-intermediate per-stage DECOMPOSITION (protocol §5) remains
DEFERRED to the baseline-analysis when the accuracy gate locks (design §8).
"""

from __future__ import annotations

import math
from itertools import pairwise

from shapely.geometry import LineString, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.decoder import _is_bref_token, decode_feature
from cfm.data.sub_f.encoder import canonicalize_geometry
from cfm.data.sub_g.diagnostics import Diagnostic

_FEATURE, _FEATURE_END = 509, 510
_VERTEX_BOUND_M = 250.0 + 50.0  # cell extent + margin; loose structural bound
_MIN_ANGLE_SEG_M = 2.0  # segments shorter than this have noisy bearings (structural)


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


def _part_coords(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Vertices of a single-part canonical geometry (Point / LineString / Polygon
    exterior). Callers pass already-split Multi* parts."""
    gt = geom.geom_type
    if gt == "Point":
        return [(geom.x, geom.y)]
    if gt == "Polygon":
        return list(geom.exterior.coords)
    return list(geom.coords)  # LineString / LinearRing


def _canon_parts(canon: BaseGeometry) -> list[BaseGeometry]:
    """Parts of a canonical geometry in the order encode_cell emits them
    (it iterates ``canon.geoms`` for Multi*)."""
    if canon.geom_type.startswith("Multi"):
        return list(canon.geoms)
    return [canon]


def _shape(coords: list[tuple[float, float]]) -> BaseGeometry:
    return Point(coords[0]) if len(coords) == 1 else LineString(coords)


def _sym_hausdorff(a: BaseGeometry, b: BaseGeometry) -> float:
    return max(a.hausdorff_distance(b), b.hausdorff_distance(a))


def has_outbound_bref(block: list[int]) -> bool:
    """True iff this feature's token BODY ends in a bref token (Case B/D outbound).

    Construction identity (encoder.py:438-456 + decoder.py:104-134): the ONLY
    v1-unencoded geometry vertex is the OUTBOUND crossing vertex; the encoder
    replaces the last real vertex with the outbound bref and the decoder appends a
    placeholder as decoded[-1]. An INBOUND bref (Case C/D) carries its position via
    the anchor (coords[0]) and is NOT an exclusion. This is a token-structure fact,
    NOT an error-magnitude test (see module docstring; reviewer guard 2026-06-01).
    """
    body = block[1:-1]  # strip <feature> / <feature_end>
    return len(body) >= 1 and _is_bref_token(body[-1])


def _is_bref_placeholder_collapse(block: list[int], geom: dict) -> bool:
    """True iff an OGC-invalid decoded geom is the v1-by-design outbound-bref
    placeholder collapse, by CONSTRUCTION IDENTITY (never magnitude, never a bare
    zero-length test): the feature BODY ends in an outbound bref (Case B/D) AND the
    decoded vertices collapse to <2 distinct points.

    This is the most degenerate form of the SAME v1-unencoded outbound bref vertex
    H1 reports-not-gates: a crossing road with NO interior vertex (V=2) drops its
    exit vertex (outbound bref, v2-scoped per decoder.py:13-22), so the decoder's
    placeholder duplicates the anchor and the LineString has zero length. These are
    the SAME roads as the accuracy baseline's reported full-distribution residual
    (position_full p99.9 229m); see render_accuracy_baseline (sub-G T11 H3).

    A degenerate geom WITHOUT an outbound bref is unreachable from a real encoder
    (interior pairs always move; magnitude_q >= 1) and so signals corruption / a
    genuine decode defect -> it is NOT excluded; the gate still quarantines it
    (guard: test_check_decodability_GATE_FIRES_on_degenerate_without_outbound_bref).
    """
    if not has_outbound_bref(block):
        return False
    return len(set(_decoded_coords(geom))) < 2


def _segment_bearing_err(
    dec: list[tuple[float, float]], orig: list[tuple[float, float]]
) -> float | None:
    """Max abs segment-bearing difference (deg, undirected), defined ONLY when the
    decoded and canonical-original vertex counts match (1:1 segment correspondence,
    i.e. no chunking subdivision). Returns None when undefined. Segments shorter
    than _MIN_ANGLE_SEG_M are skipped (noisy bearings)."""
    if len(dec) != len(orig) or len(dec) < 2:
        return None
    worst = 0.0
    for (da, db), (oa, ob) in zip(pairwise(dec), pairwise(orig), strict=True):
        if math.dist(da, db) < _MIN_ANGLE_SEG_M:
            continue
        d_bear = math.degrees(math.atan2(db[1] - da[1], db[0] - da[0]))
        o_bear = math.degrees(math.atan2(ob[1] - oa[1], ob[0] - oa[0]))
        diff = abs(d_bear - o_bear) % 360.0
        worst = max(worst, min(diff, 360.0 - diff))
    return worst


def _feature_accuracy(
    decoded_dicts: list[dict],
    has_outbound: list[bool],
    canon_parts: list[BaseGeometry],
) -> dict:
    """Geometry-aware round-trip error of one (possibly multi-part) feature vs its
    canonical original. ``decoded_dicts[i]`` / ``has_outbound[i]`` correspond to
    ``canon_parts[i]`` (canonical part order = encode_cell emission order).

    Returns {position_core_m, position_full_m, angle_core_deg|None}. Core excludes
    the v1-unencoded outbound bref vertex (last vertex of both decoded placeholder
    and the true exit crossing) by construction identity, never by magnitude.
    """
    dec_full, orig_full, dec_core, orig_core = [], [], [], []
    angles: list[float] = []
    for dd, out, cp in zip(decoded_dicts, has_outbound, canon_parts, strict=True):
        dco = _decoded_coords(dd)
        oco = _part_coords(cp)
        dec_full.append(_shape(dco))
        orig_full.append(_shape(oco))
        dco_c = dco[:-1] if (out and len(dco) > 1) else dco
        oco_c = oco[:-1] if (out and len(oco) > 1) else oco
        if dco_c and oco_c:
            dec_core.append(_shape(dco_c))
            orig_core.append(_shape(oco_c))
            a = _segment_bearing_err(dco_c, oco_c)
            if a is not None:
                angles.append(a)

    def _union(shapes: list[BaseGeometry]) -> BaseGeometry:
        return shapes[0] if len(shapes) == 1 else unary_union(shapes)

    return {
        "position_full_m": _sym_hausdorff(_union(dec_full), _union(orig_full)),
        "position_core_m": _sym_hausdorff(_union(dec_core), _union(orig_core)) if dec_core else 0.0,
        "angle_core_deg": max(angles) if angles else None,
    }


def check_decodability(
    tile_id: str,
    cell: tuple[int, int],
    token_sequence: list[int],
    sub_c_features: list[dict],
) -> tuple[list[Diagnostic], list[dict], int]:
    """Returns (gate_diagnostics, per_feature_accuracy_records, n_bref_collapse).

    The GATE runs per decoded block (independent). ACCURACY pairs each sub-C
    feature with the #parts decoded blocks it produced (encode_cell splits Multi*),
    then measures geometry-aware error vs the canonical original.

    ``n_bref_collapse`` counts decoded blocks excluded from the OGC-validity gate
    by construction identity (the v1-by-design outbound-bref placeholder collapse;
    sub-G T11 H3, consistent with H1). Reported in the accuracy baseline, NOT gated.
    """
    diags: list[Diagnostic] = []
    errors: list[dict] = []
    n_bref_collapse = 0
    blocks = split_cell_into_features(token_sequence)

    # --- GATE: decode each block once; validity + structural bound ---
    decoded: list[dict | None] = []
    for k, ftokens in enumerate(blocks):
        try:
            geom = decode_feature(ftokens)
        except Exception as exc:  # decode failure is a gate failure
            decoded.append(None)
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
        decoded.append(geom)

        if geom["type"] in ("Polygon", "LineString") and not shape(geom).is_valid:
            if _is_bref_placeholder_collapse(ftokens, geom):
                # v1-by-design outbound-bref placeholder collapse (V=2 crossing road,
                # no interior vertex). Reported in the accuracy baseline, NOT gated,
                # by construction identity -- consistent with H1's bref-vertex call
                # (sub-G T11 H3, reviewer 2026-06-01). The guard above proves the gate
                # still fires on a degenerate geom that is NOT this construction.
                n_bref_collapse += 1
            else:
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

    # --- ACCURACY: pair each sub-C feature with its #parts decoded blocks ---
    di = 0
    for f in sub_c_features:
        canon = canonicalize_geometry(wkb_loads(bytes(f["geometry"])))
        parts = _canon_parts(canon)
        grp_blocks = blocks[di : di + len(parts)]
        grp_decoded = decoded[di : di + len(parts)]
        di += len(parts)
        if len(grp_decoded) < len(parts) or any(g is None for g in grp_decoded):
            continue  # incomplete trailing feature or a part failed to decode
        errors.append(
            _feature_accuracy(
                grp_decoded,
                [has_outbound_bref(b) for b in grp_blocks],
                parts,
            )
        )
    return diags, errors, n_bref_collapse
