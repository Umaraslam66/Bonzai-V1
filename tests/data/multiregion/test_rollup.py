"""Roll-up gate is STRUCTURAL, not a count (threshold-pairing): a full validated
count must NOT pass the gate if failures cluster in one axis (or a city is
excluded) leaving an intended label uncovered. The coverage check is proven
independent of the failure check."""

from __future__ import annotations

from cfm.data.multiregion import rollup


def _city(name, morph, dens, status, tiles=400, tokens=11_000_000, geo="DE"):
    return rollup.CityRecord(
        name=name,
        morphology=morph,
        density=dens,
        geography=geo,
        region_crs="EPSG:25833",
        tile_count=tiles,
        fetch_seconds=800.0,
        stage_shas={"sub_c": "abc"},
        release="2026-04-15.0",
        validation_status=status,
        token_count=tokens,
    )


def test_gate_passes_when_all_validated_and_axes_covered():
    r = rollup.RollUp(
        cities=[
            _city("a", "planned-grid", "dense-core", "validated"),
            _city("b", "medieval-organic", "moderate", "validated"),
            _city("c", "modernist-sprawl", "sparse", "validated"),
        ]
    )
    rollup.assert_ready_for_next_batch(r)  # no raise


def test_gate_raises_on_unaddressed_failure():
    r = rollup.RollUp(
        cities=[
            _city("a", "planned-grid", "dense-core", "validated"),
            _city("b", "medieval-organic", "moderate", "failed"),
        ]
    )
    try:
        rollup.assert_ready_for_next_batch(r)
        raise AssertionError("expected gate to raise on unaddressed failure")
    except RuntimeError as exc:
        assert "failed-needs-attention" in str(exc)


def test_full_count_but_clustered_failure_does_not_pass_gate():
    # 3 validated (high count) but BOTH medieval cities failed → the aggregate
    # count looks fine yet morphology=medieval-organic is uncovered. Gate must NOT pass.
    r = rollup.RollUp(
        cities=[
            _city("g1", "planned-grid", "dense-core", "validated"),
            _city("g2", "planned-grid", "dense-core", "validated"),
            _city("s1", "modernist-sprawl", "sparse", "validated"),
            _city("m1", "medieval-organic", "moderate", "failed"),
            _city("m2", "medieval-organic", "moderate", "failed"),
        ]
    )
    try:
        rollup.assert_ready_for_next_batch(r)
        raise AssertionError("expected gate to NOT pass with a clustered failure")
    except RuntimeError:
        pass  # raises (failures present and/or medieval uncovered) — either way, not clean


def test_coverage_check_is_independent_of_failure_check():
    # ZERO failures, but the intended span includes a morphology no validated city
    # has (e.g. a city was excluded). The count is full; the COVERAGE check alone
    # must fire — proving it is not redundant with the failure check.
    r = rollup.RollUp(
        cities=[
            _city("g1", "planned-grid", "dense-core", "validated"),
            _city("s1", "modernist-sprawl", "sparse", "validated"),
        ]
    )
    assert rollup.unaddressed_failures(r) == []  # failure check would PASS
    required = {
        "morphology": {"planned-grid", "modernist-sprawl", "medieval-organic"},
        "density": {"dense-core", "sparse"},
        "geography": {"DE"},
    }
    try:
        rollup.assert_ready_for_next_batch(r, required_axis_labels=required)
        raise AssertionError("expected coverage check to fire on uncovered medieval-organic")
    except RuntimeError as exc:
        assert "morphology=medieval-organic" in str(exc)


def test_axis_coverage_counts_validated_only():
    r = rollup.RollUp(
        cities=[
            _city("a", "planned-grid", "dense-core", "validated"),
            _city("b", "planned-grid", "dense-core", "validated"),
            _city("c", "medieval-organic", "sparse", "failed"),
        ]
    )
    cov = rollup.axis_coverage(r)
    assert cov["morphology"]["planned-grid"] == 2
    assert cov["morphology"].get("medieval-organic", 0) == 0  # failed excluded


def test_total_tokens_and_tiles_sum_validated_only():
    r = rollup.RollUp(
        cities=[
            _city("a", "planned-grid", "dense-core", "validated", tiles=10, tokens=100),
            _city("b", "mixed", "moderate", "failed", tiles=999, tokens=999),
        ]
    )
    assert rollup.total_validated_tokens(r) == 100
    assert rollup.total_validated_tiles(r) == 10
