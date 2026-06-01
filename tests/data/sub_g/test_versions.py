from __future__ import annotations

from cfm.data.sub_g.versions import (
    VALIDATOR_VERSION,
    _percentile,
    render_accuracy_baseline,
    render_validated_marker,
)


def test_percentile_ignores_non_finite():
    # NaN/inf are not orderable; leaving them in sorted() yields a non-monotonic
    # result (p95 > p99.9). They must be dropped before the nearest-rank pick.
    vals = [1.0, 2.0, 3.0, 4.0, float("nan"), float("inf")]
    assert _percentile(vals, 95.0) <= _percentile(vals, 99.9)
    assert _percentile(vals, 99.9) == 4.0
    assert _percentile([float("nan")], 50.0) == 0.0  # all non-finite -> empty -> 0.0


def test_validator_version_is_semver():
    # 1.1.0: seam-3 geometry-aware core/full accuracy metric (sub-G T11 H1, 2026-06-01).
    assert VALIDATOR_VERSION == "1.1.0"
    parts = VALIDATOR_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_marker_carries_stable_digest_and_segregates_volatile():
    a = render_validated_marker(
        region="singapore",
        release="2026-04-15.0",
        content_digest="abc",
        volatile={
            "run_timestamp": "T1",
            "host": "h",
            "run_uuid": "u1",
            "sub_g_commit_sha": "deadbeef",
        },
    )
    b = render_validated_marker(
        region="singapore",
        release="2026-04-15.0",
        content_digest="abc",
        volatile={
            "run_timestamp": "T2",
            "host": "h",
            "run_uuid": "u2",
            "sub_g_commit_sha": "deadbeef",
        },
    )
    # stable content (digest + identity) identical across runs; volatile differs but is written.
    assert "content_digest: abc" in a and "content_digest: abc" in b
    assert "validator_version: 1.1.0" in a
    assert "T1" in a and "T2" in b
    # the only difference between the two renders is inside the volatile block.
    assert a.replace("T1", "T2").replace("u1", "u2") == b


def test_accuracy_baseline_records_core_and_full_percentiles():
    out = render_accuracy_baseline(
        position_core=[1.0, 2.0, 3.0, 4.0],
        position_full=[1.0, 2.0, 3.0, 300.0],  # crossing-road bref residual
        angle_core=[0.5, 1.0, 1.5],
        region="singapore",
        release="2026-04-15.0",
        structural_bound_breaches=0,
        bref_collapse_excluded=27958,
    )
    assert "position_core_p99_9" in out
    assert "position_core_p95" in out
    assert "position_full_p99_9" in out  # full reported + visible
    assert "angle_core_p95" in out
    assert "n_features" in out
    assert "n_angle_features" in out
    assert "core_excludes" in out  # self-documents the structural exclusion
    # sub-G T11 H3: the OGC-validity gate's construction-identity exclusion is
    # reported (not silently dropped) and cross-references the position_full residual.
    assert "ogc_bref_collapse_excluded_from_gate: 27958" in out


def test_accuracy_baseline_handles_empty():
    out = render_accuracy_baseline(
        position_core=[],
        position_full=[],
        angle_core=[],
        region="singapore",
        release="2026-04-15.0",
        structural_bound_breaches=0,
    )
    assert "position_core_p99_9: 0.0" in out
