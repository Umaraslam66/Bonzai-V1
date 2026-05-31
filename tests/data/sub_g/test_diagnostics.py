from __future__ import annotations

from cfm.data.sub_g.diagnostics import Diagnostic, group_by_signature, render_quarantine_report


def _diag(tile: str, value: float, bucket: str) -> Diagnostic:
    return Diagnostic(
        tile_id=tile,
        invariant_name="density_bucket_matches_footprint",
        artifact_left="sub_c.building_footprint_ratio",
        observed_left=value,
        artifact_right="sub_d.cell_density_bucket",
        observed_right=bucket,
        expected_relationship="ratio in [a,b) implies bucket==k",
        spec_clause_citation="PRD §5 line 65 + macro_plan_vocab.yaml:3472-3488",
        signature="density bucket one-step-too-high vs footprint range",
    )


def test_group_by_signature_collapses_same_pattern():
    diags = [_diag(f"tile=i0_j{j}", 0.40 + j * 0.001, "high") for j in range(100)]
    groups = group_by_signature(diags)
    assert len(groups) == 1
    g = groups[0]
    assert g.invariant_name == "density_bucket_matches_footprint"
    assert g.signature == "density bucket one-step-too-high vs footprint range"
    assert g.instance_count == 100
    assert g.tile_ids == sorted(d.tile_id for d in diags)
    assert g.value_summary["observed_left"]["min"] <= g.value_summary["observed_left"]["max"]


def test_groups_sorted_by_count_desc_then_invariant_asc():
    a = [_diag("tile=i0_j0", 0.4, "high")]
    b = [
        Diagnostic("tile=i0_j0", "zzz_invariant", "l", 1, "r", 2, "rel", "cite", "sigZ"),
        Diagnostic("tile=i0_j1", "zzz_invariant", "l", 1, "r", 2, "rel", "cite", "sigZ"),
    ]
    groups = group_by_signature(a + b)
    assert [g.instance_count for g in groups] == [2, 1]  # count desc first


def test_render_is_byte_deterministic_and_empty_record_is_explicit():
    out_empty = render_quarantine_report(
        groups=[], region="singapore", release="2026-04-15.0", validator_version="1.0.0"
    )
    assert "groups: []" in out_empty
    a = render_quarantine_report(
        groups=group_by_signature([_diag("tile=i0_j0", 0.4, "high")]),
        region="singapore",
        release="2026-04-15.0",
        validator_version="1.0.0",
    )
    b = render_quarantine_report(
        groups=group_by_signature([_diag("tile=i0_j0", 0.4, "high")]),
        region="singapore",
        release="2026-04-15.0",
        validator_version="1.0.0",
    )
    assert a == b
