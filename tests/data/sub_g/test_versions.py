from __future__ import annotations

from cfm.data.sub_g.versions import (
    VALIDATOR_VERSION,
    render_accuracy_baseline,
    render_validated_marker,
)


def test_validator_version_is_1_0_0_semver():
    assert VALIDATOR_VERSION == "1.0.0"
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
    assert "validator_version: 1.0.0" in a
    assert "T1" in a and "T2" in b
    # the only difference between the two renders is inside the volatile block.
    assert a.replace("T1", "T2").replace("u1", "u2") == b


def test_accuracy_baseline_records_percentiles():
    out = render_accuracy_baseline(
        position_errors=[1.0, 2.0, 3.0, 4.0],
        angle_errors=[0.5, 1.0, 1.5],
        region="singapore",
        release="2026-04-15.0",
        structural_bound_breaches=0,
    )
    assert "position_p99_9" in out
    assert "position_p95" in out
    assert "angle_p99_9" in out
    assert "angle_p95" in out
    assert "n_features" in out


def test_accuracy_baseline_handles_empty():
    out = render_accuracy_baseline(
        position_errors=[],
        angle_errors=[],
        region="singapore",
        release="2026-04-15.0",
        structural_bound_breaches=0,
    )
    assert "position_p99_9: 0.0" in out
