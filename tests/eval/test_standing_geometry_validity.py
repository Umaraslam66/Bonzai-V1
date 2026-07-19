"""Geometry-validity metric (spec §4): echo-immune structural metrics on probe tokens.

Pinned against the existing transformer probe (reports/_eyeball_probe/) whose construction
-identity class counts and dense road-fragmentation were established by
scripts/_road_connectivity_diag.py and tests/test_viz_build_data.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.eval.standing.geometry_validity import geometry_validity_report

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "reports" / "_eyeball_probe" / "gen_tokens.json"

pytestmark = pytest.mark.skipif(not GEN.exists(), reason="transformer probe tokens absent")


@pytest.fixture(scope="module")
def report():
    return geometry_validity_report(GEN, tau=1.0)


def test_construction_identity_class_totals(report):
    tot = {"building_sealed": 0, "building_unsealed": 0, "road": 0, "road_node": 0}
    for cg in report.per_context.values():
        for k in tot:
            tot[k] += cg.counts[k]
    assert tot["building_sealed"] == 24
    assert tot["building_unsealed"] == 234  # NOT counted as roads
    assert tot["road"] == 439
    assert tot["road_node"] == 33


def test_decode_and_self_term(report):
    # ~100% decodable; 19/21 self-terminate (2 dense cells hit the 1536 probe cap)
    n_self = sum(cg.n_cells * cg.self_term_frac for cg in report.per_context.values())
    assert round(n_self) == 19
    for cg in report.per_context.values():
        assert cg.decode_frac > 0.99


def test_dense_road_fragmentation_matches_diag(report):
    dense = report.per_context["dense_urban"]
    # _road_connectivity_diag.py: dense median components/segment = 0.87 at tau=1.0
    assert abs(dense.median_components_per_segment - 0.87) < 0.03
    assert dense.dangling_endpoint_frac > 0.7  # diag: ~0.85


def test_closure_gap_distribution(report):
    dense = report.per_context["dense_urban"]
    assert 0.0 < dense.closure_gap_median < 0.10
    assert dense.closure_within_5pct > 0.5  # majority of footprints close within 5%
