from __future__ import annotations

import pytest
import yaml

from cfm.eval.resolution import InsufficientResolutionError, assert_resolution_sufficient


def _marker(tmp, **fields):
    p = tmp / "_EVAL_SET_LOCKED"
    p.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return p


def test_gap_at_or_above_resolved_passes(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    assert_resolution_sufficient(0.10, marker_path=m)  # no raise


def test_gap_between_floor_and_resolved_fails_with_second_region_message(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e:
        assert_resolution_sufficient(0.06, marker_path=m)
    assert "second-region" in str(e.value).lower()
    assert "fundamentally" not in str(e.value).lower()


def test_gap_below_floor_fails_with_categorical_message_distinct_from_second_region(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e_below:
        assert_resolution_sufficient(0.03, marker_path=m)
    assert "fundamentally" in str(e_below.value).lower()
    # the two failure KINDS must produce DIFFERENT messages
    with pytest.raises(InsufficientResolutionError) as e_between:
        assert_resolution_sufficient(0.06, marker_path=m)
    assert str(e_below.value) != str(e_between.value)


def test_marker_absent_raises_not_no_ops(tmp_path):
    with pytest.raises((FileNotFoundError, InsufficientResolutionError)):
        assert_resolution_sufficient(0.10, marker_path=tmp_path / "missing")


def test_marker_missing_required_field_raises(tmp_path):
    bad = _marker(tmp_path, ks_target_gap=0.08)  # lacks the two required fields
    with pytest.raises((KeyError, InsufficientResolutionError)):
        assert_resolution_sufficient(0.10, marker_path=bad)


def test_reads_the_real_frozen_marker_fields():
    """The real frozen marker carries the fields the seam reads (artifact
    dependency confirmed); 0.10 >= the real resolved gap (~0.076) passes."""
    assert_resolution_sufficient(0.10)  # uses the real _EVAL_SET_LOCKED on disk
