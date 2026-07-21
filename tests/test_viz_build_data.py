"""Guard the visualizer's data builder: faithful, construction-identity classification.

The whole value of the viz is that it recovers building footprints that the probe's
GeoJSON demotes to ``LineString`` ("near-closed buildings misread as roads"). If the
classification regresses to a geometry-shape heuristic, these counts move -- so we pin
them against the real probe artifact.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_TOKENS = REPO / "reports" / "_eyeball_probe" / "gen_tokens.json"

_spec = importlib.util.spec_from_file_location("viz_build_data", REPO / "viz" / "build_data.py")
assert _spec and _spec.loader
build_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_data)


@pytest.fixture(scope="module")
def bundle():
    if not GEN_TOKENS.exists():
        pytest.skip("eyeball-probe gen_tokens.json not present")
    return build_data.build_probe_data(GEN_TOKENS)


def test_class_totals_match_grammar(bundle):
    """Faithful counts across all 21 cells (see reports/_eyeball_probe/SUMMARY.md)."""
    totals: dict[str, int] = {}
    for c in bundle["cells"]:
        for k, v in c["counts"].items():
            totals[k] = totals.get(k, 0) + v
    # RE-PINNED 2026-07-19 with the defect-(a) closure fix (float-drift epsilon in
    # cfm.eval.geometry._is_closed_ring, commit b30d604): 37 building rings close to within
    # ~1e-14 m and were previously misread as unsealed by exact `==` (24/234 -> 61/197 sealed
    # /unsealed; sum conserved at 258, roads untouched). Matches the standing-fixture re-pin in
    # tests/eval/test_standing_geometry_validity.py (same gen_tokens.json, same classifier).
    assert totals["building_sealed"] == 61
    assert totals["building_unsealed"] == 197  # hidden among the probe's "road" lines
    assert totals["road"] == 439
    assert totals["road_node"] == 33
    # Raw probe geojson has 24 polygons + 673 linestrings = 697 non-point features. The
    # closure fix only re-labels linestrings (24 native + 37 promoted = 61 sealed; the other
    # 636 linestrings split 197 unsealed + 439 road), so this total is conserved across it.
    assert totals["building_sealed"] + totals["building_unsealed"] + totals["road"] == 697


def test_directional_response_is_monotonic(bundle):
    """Buildings and roads per cell must decrease dense -> medium -> sparse."""
    by_ctx = {s["context"]: s for s in bundle["summary"]}
    order = ["dense_urban", "medium_mixed", "sparse_suburban"]
    builds = [by_ctx[c]["med_buildings"] for c in order]
    roads = [by_ctx[c]["med_roads"] for c in order]
    tokens = [by_ctx[c]["med_tokens"] for c in order]
    assert builds[0] > builds[1] > builds[2]
    assert roads[0] > roads[1] > roads[2]
    assert tokens[0] > tokens[1] > tokens[2]


def test_every_cell_has_conditioning_and_geometry(bundle):
    assert len(bundle["cells"]) == 21
    for c in bundle["cells"]:
        assert len(c["stratum"]) == 4
        assert len(c["char_decoded"]) == 7
        assert c["features"], f"{c['context']}#{c['cell_index']} decoded to no features"
        assert len(c["bbox"]) == 4


def test_unsealed_buildings_carry_closure_gap(bundle):
    """Unsealed footprints expose the gap metric so the UI can show 'near-closed'."""
    gaps = [
        f["gap"]
        for c in bundle["cells"]
        for f in c["features"]
        if f["cls"] == "building_unsealed" and "gap" in f
    ]
    assert gaps, "expected unsealed buildings to carry a closure gap"
    gaps.sort()
    median = gaps[len(gaps) // 2]
    assert 0.0 < median < 0.10  # SUMMARY: median ~3%
