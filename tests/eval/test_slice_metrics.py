"""Task 10 discrimination tests for the per-cell slice eval (spec §8, §9).

The slice eval is REPORTED-NOT-GATED (no pass/fail threshold). It:
  * computes the bref-placeholder collapse rate via the SHARED D3 instrument
    (identity-locked: ``_bref_rate_fn is bref_placeholder_rate``), never a local
    reimplementation;
  * EXCLUDES bref-collapse instances from the OGC-validity denominator by the
    sub-G construction-identity predicate (structural exclusion, not magnitude),
    while still REPORTING the bref rate;
  * measures decodability as decoded / ATTEMPTED blocks (a real rate);
  * is scoped per-cell -- tile coherence is named UNSCORED, never implied.
"""

from __future__ import annotations

from cfm.eval import slice_metrics as S
from cfm.eval.emergence import building_token_ids
from cfm.eval.holdout import bref_rate

# A valid unit square (4 right-angle corners) and a real bref-collapse pair
# (body ends in bref token 1500 -> decodes to 2 identical points -> invalid).
_SQUARE_BLOCK = [509, 510]
_SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
_COLLAPSE_BLOCK = [509, 41, 300, 323, 363, 369, 1500, 510]
_COLLAPSE_GEOM = {"type": "LineString", "coordinates": [[0.0, 195.5], [0.0, 195.5]]}


def test_validity_excludes_bref_collapse_via_shared_instrument():
    # identity-lock: the bref rate is the ONE shared D3 instrument, not a fork.
    assert S._bref_rate_fn is bref_rate.bref_placeholder_rate


def test_metrics_are_per_cell_scoped_and_report_not_gate():
    r = S.slice_eval([_SQUARE_BLOCK], [_SQUARE], [1])
    assert {
        "decodability_rate",
        "ogc_valid_rate",
        "right_angle_rate",
        "bref_collapse_rate",
        "scope",
    } <= set(r)
    assert r["scope"] == "per-cell; tile-coherence UNSCORED"  # named, not implied
    assert "bref_collapse_rate" in r  # reported, never gates pass/fail


def test_square_is_valid_with_right_angles():
    r = S.slice_eval([_SQUARE_BLOCK], [_SQUARE], [1])
    assert r["decodability_rate"] == 1.0  # 1 decoded / 1 attempted
    assert r["ogc_valid_rate"] == 1.0
    assert r["right_angle_rate"] == 1.0  # all 4 corners ~90 degrees
    assert r["bref_collapse_rate"] == 0.0
    assert r["n_polygons"] == 1 and r["n_corners"] == 4  # disambiguates a 0.0 rate


def test_right_angle_rate_zero_disambiguated_by_polygon_count():
    """A LineString-only cell -> right_angle_rate 0.0 with n_polygons==0, i.e.
    'no polygons emitted', NOT 'polygons without right angles'."""
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    r = S.slice_eval([_SQUARE_BLOCK], [line], [1])
    assert r["right_angle_rate"] == 0.0
    assert r["n_polygons"] == 0 and r["n_corners"] == 0


def test_bref_collapse_excluded_from_ogc_valid_denominator():
    """The collapse geom is OGC-invalid but a KNOWN v1 limitation: it is removed
    from the validity denominator (so ogc_valid_rate stays 1.0, not 0.5) AND the
    bref rate reports it. Structural exclusion, not magnitude."""
    blocks = [_COLLAPSE_BLOCK, _SQUARE_BLOCK]
    geoms = [_COLLAPSE_GEOM, _SQUARE]
    strata = [1, 1]
    r = S.slice_eval(blocks, geoms, strata)
    assert r["ogc_valid_rate"] == 1.0  # square only; collapse excluded by construction id
    assert r["bref_collapse_rate"] == 0.5  # 1 collapse of 2 blocks, reported
    assert r["decodability_rate"] == 1.0  # both decoded


def test_decodability_uses_attempted_block_count():
    """decodability = decoded / attempted: an undecodable block lowers the rate
    even though only decoded (block, geom) pairs reach slice_eval."""
    r = S.slice_eval([_SQUARE_BLOCK], [_SQUARE], [1], n_attempted_blocks=3)
    assert r["decodability_rate"] == 1 / 3  # 1 decoded of 3 attempted


def test_real_invalid_non_bref_counts_against_validity():
    """Regime-distinguishing twin: a NON-bref invalid geom IS counted as invalid
    (the exclusion is bref-collapse-specific, not 'exclude all invalids')."""
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    r = S.slice_eval([_SQUARE_BLOCK, _SQUARE_BLOCK], [_SQUARE, bowtie], [1, 1])
    assert r["ogc_valid_rate"] == 0.5  # square valid, self-intersecting bowtie invalid
    assert r["bref_collapse_rate"] == 0.0  # neither is a bref collapse


def test_slice_eval_promotes_building_rings_on_its_live_path():
    """F4-C1b guard: deleting slice_eval's promote_building_rings call must turn this RED.

    Decoder contract: building rings arrive as closed LineString, never Polygon. Every
    other fixture in this file feeds pre-made Polygons, so only this test exercises the
    live promotion call inside ``slice_eval`` (the ``is``-identity test alone would still
    pass with the call deleted)."""
    building = min(building_token_ids())  # a real building-class token id (one authority)
    block = [509, building, 1, 2]  # <feature> + building token, per test_geometry_promote
    ring = {"type": "LineString", "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]}
    r = S.slice_eval([block], [ring], [1])
    assert r["n_polygons"] >= 1  # un-promoted, the ring stays LineString -> n_polygons == 0
    assert r["right_angle_rate"] == 1.0  # promoted square ring: all 4 corners ~90 degrees


def test_points_excluded_from_ogc_denominator_and_counted():
    """Task 26 (f) Point-semantics pin: a decoded Point is trivially OGC-valid,
    so leaving it in the denominator inflates ogc_valid_rate (here it would mask
    the bowtie: 1/2 instead of 0/1). Points are EXCLUDED from the validity
    denominator and reported as ``n_points`` instead — never silently scored."""
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    point = {"type": "Point", "coordinates": [1.0, 2.0]}
    r = S.slice_eval([_SQUARE_BLOCK, _SQUARE_BLOCK], [bowtie, point], [1, 1])
    assert r["n_points"] == 1  # reported, so the exclusion is visible
    assert r["ogc_valid_rate"] == 0.0  # denominator = the bowtie alone


def test_points_alone_yield_zero_rate_with_explicit_count():
    """All-Point cell: an empty validity denominator reads 0.0 — and n_points
    disambiguates 'nothing scoreable' from 'everything invalid'."""
    pts = [{"type": "Point", "coordinates": [float(i), 0.0]} for i in range(3)]
    r = S.slice_eval([_SQUARE_BLOCK] * 3, pts, [1, 1, 1])
    assert r["n_points"] == 3
    assert r["ogc_valid_rate"] == 0.0
