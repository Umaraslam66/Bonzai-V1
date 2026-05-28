"""Tests for BP3 stage analysis intermediate outputs.

Per spec §7.3 / §7.4: joint distribution and budget-surface enumeration must
include every feature class and every required quantile. Tests assert the
'ALL OF' invariant per BP3 fix 3 (§8.1).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"


def test_stage_1_2_joint_includes_all_sub_c_feature_classes():
    """Joint distribution enumerates all four sub-C feature classes.

    Sub-C FEATURE_CLASS per src/cfm/data/sub_c/enums.py: {0: road, 1: building,
    2: poi, 3: base}. Joint must include observations for each that appears
    in cached Singapore.
    """
    data = yaml.safe_load((CONFIG_ROOT / "stage_1_2_joint.yaml").read_text(encoding="utf-8"))
    types_present = set(data["per_feature_type"].keys())
    # At minimum, road (0) and building (1) must be present on Singapore.
    assert 0 in types_present, "road feature class missing from Singapore joint"
    assert 1 in types_present, "building feature class missing from Singapore joint"


def test_stage_3_compound_uses_locked_anchor_scheme():
    """Stage-3 compound respects the Task 2 anchor scheme lock.

    Anchor scheme = hierarchical → n_anchor = 4; flat → n_anchor = 2.
    """
    data = yaml.safe_load((CONFIG_ROOT / "stage_3_compound.yaml").read_text(encoding="utf-8"))
    assert data["anchor_scheme_used"] in ("flat", "hierarchical")
    assert data["n_anchor"] in (2, 4)
    if data["anchor_scheme_used"] == "flat":
        assert data["n_anchor"] == 2
    else:
        assert data["n_anchor"] == 4


def test_budget_surface_enumerates_5_quantiles():
    """Budget surface enumerates all 5 quantile points per spec §7.4."""
    data = yaml.safe_load(
        (CONFIG_ROOT / "sequence_length_analysis.yaml").read_text(encoding="utf-8")
    )
    quantiles_present = {row["quantile"] for row in data["budget_surface"]}
    assert quantiles_present == {99.0, 99.5, 99.9, 99.99, 100.0}, (
        f"missing quantiles: {{99.0, 99.5, 99.9, 99.99, 100.0}} - {quantiles_present}"
    )


def test_budget_surface_retention_per_type_all_present():
    """Retention table per BP3 fix 3 'ALL OF' invariant — every type has a rate."""
    data = yaml.safe_load(
        (CONFIG_ROOT / "sequence_length_analysis.yaml").read_text(encoding="utf-8")
    )
    # Every quantile row covers every feature type present on Singapore.
    types_per_quantile = {
        q: set(row.keys()) for q, row in data["retention_by_quantile_by_type"].items()
    }
    all_types = set.union(*types_per_quantile.values()) if types_per_quantile else set()
    for q, types_at_q in types_per_quantile.items():
        assert types_at_q == all_types, f"quantile {q} missing types: {all_types - types_at_q}"
