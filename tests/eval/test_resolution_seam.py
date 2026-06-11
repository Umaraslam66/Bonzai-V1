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


def test_gap_between_floor_and_resolved_escalates_generically(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e:
        assert_resolution_sufficient(0.06, marker_path=m)
    msg = str(e.value).lower()
    assert "more/larger held-out" in msg
    assert "second region" not in msg
    assert "munich" not in msg


def test_gap_below_floor_fails_with_categorical_message_distinct_from_second_region(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e_below:
        assert_resolution_sufficient(0.03, marker_path=m)
    msg_below = str(e_below.value).lower()
    # below message: conveys it is the resolvable-gap ceiling
    assert "ceiling" in msg_below
    assert "second region" not in msg_below
    # the two failure KINDS must produce DIFFERENT messages
    with pytest.raises(InsufficientResolutionError) as e_between:
        assert_resolution_sufficient(0.06, marker_path=m)
    assert str(e_below.value) != str(e_between.value)


def test_escalation_is_multiregion_not_second_region(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.10, ks_single_region_floor=0.05)
    with pytest.raises(InsufficientResolutionError) as e:
        assert_resolution_sufficient(0.07, marker_path=m)
    msg = str(e.value).lower()
    assert "second region" not in msg
    assert "munich" not in msg  # the swap is the COHERENCE gate's (T12), NOT KS-resolution's
    assert "more/larger held-out" in msg


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


# ----- Task 20 (F9): region-aware marker selection -----

_RELEASE = "2026-04-15.0"


def test_region_routing_singapore_selects_sg_marker():
    from cfm.eval.holdout.paths import eval_set_locked_marker
    from cfm.eval.resolution import resolution_marker_for_region

    assert resolution_marker_for_region(_RELEASE, "singapore") == eval_set_locked_marker(_RELEASE)


def test_region_routing_eu_cities_select_multiregion_marker():
    from cfm.eval.holdout.paths import multiregion_eval_set_locked_marker
    from cfm.eval.resolution import resolution_marker_for_region

    expected = multiregion_eval_set_locked_marker(_RELEASE)
    for city in ("eisenhuttenstadt", "glasgow", "krakow", "munich"):
        assert resolution_marker_for_region(_RELEASE, city) == expected


def test_region_unknown_raises_fail_closed():
    """Mirrors holdout_manifest_for_region's fail-closed routing: never silently
    mis-route an unknown region to either marker."""
    from cfm.eval.resolution import resolution_marker_for_region

    with pytest.raises(ValueError, match="unknown region"):
        resolution_marker_for_region(_RELEASE, "berlin")


def test_region_singapore_passes_against_real_marker():
    assert_resolution_sufficient(0.10, region="singapore")  # real SG marker on disk


def test_region_eu_real_marker_missing_ks_fields_is_loud_keyerror():
    """The REAL multiregion _EVAL_SET_LOCKED carries NO ks fields; reading it must
    be a loud KeyError (fail-closed dict-index), never a permissive default."""
    with pytest.raises(KeyError, match="ks_resolved_gap_binding"):
        assert_resolution_sufficient(0.10, region="munich")


def test_explicit_marker_path_wins_over_region(tmp_path):
    """Precedence: explicit marker_path > region routing (the munich route would
    KeyError on the real EU marker, so passing proves marker_path won)."""
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    assert_resolution_sufficient(0.10, marker_path=m, region="munich")  # no raise
