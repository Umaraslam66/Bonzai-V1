"""The proxy is ADVISORY: it records a diagnostic verdict label + relative deltas,
but the budget ALWAYS sizes up and the r-unresolved flag is ALWAYS set — the
verdict never changes the budget (PI decision 2026-06-03; bake-off is sole r
authority). Constants pinned; comparison is relative (label), not absolute on EU."""

from __future__ import annotations

import pytest

from cfm.data.multiregion import proxy

BASE = 20_600
SIZED_UP = round(BASE * 1.5)  # 30900


def test_constants_are_pinned_not_fit_to_data():
    assert proxy.X_AMBIGUOUS_BAND == 0.10
    assert proxy.Y_SIZE_UP == 0.5


@pytest.mark.parametrize(
    ("geo", "expected_label"),
    [
        (0.80, "more_redundant_than_language"),  # rel_lang +0.33
        (0.62, "ambiguous"),  # rel_lang +0.03 (within band)
        (0.40, "less_redundant_than_language"),  # rel_lang -0.33
    ],
)
def test_budget_always_sizes_up_and_flags_regardless_of_verdict(geo, expected_label):
    # The decisive advisory property: across ALL three verdict regimes the budget
    # is base+Y and the flag is set. The verdict does NOT gate the budget.
    v = proxy.proxy_decision(
        geometry_redundancy=geo,
        language_baseline=0.60,
        singapore_redundancy=0.61,
        base_tile_budget=BASE,
    )
    assert v.verdict == expected_label  # label is recorded (diagnostic)
    assert v.recommended_tile_budget == SIZED_UP  # ALWAYS size up
    assert v.r_unresolved_flag is True  # ALWAYS flagged — bake-off resolves r


def test_singapore_disagreement_does_not_change_budget_only_records_rel():
    # Even a strong Singapore contradiction (EU far less redundant than Singapore)
    # leaves the budget at base+Y + flag — Singapore is recorded (rel_singapore),
    # not a gate.
    v = proxy.proxy_decision(
        geometry_redundancy=0.70,
        language_baseline=0.60,
        singapore_redundancy=0.90,
        base_tile_budget=BASE,
    )
    assert v.recommended_tile_budget == SIZED_UP
    assert v.r_unresolved_flag is True
    assert v.rel_singapore < 0  # the disagreement is recorded for anomaly-spotting


def test_verdict_label_is_relative_not_absolute_on_eu():
    # Scaling geometry+language by a common factor preserves the relative position
    # → same label. An absolute EU threshold would flip 0.80 vs 0.40.
    v1 = proxy.proxy_decision(
        geometry_redundancy=0.80,
        language_baseline=0.60,
        singapore_redundancy=0.78,
        base_tile_budget=BASE,
    )
    v2 = proxy.proxy_decision(
        geometry_redundancy=0.40,
        language_baseline=0.30,
        singapore_redundancy=0.39,
        base_tile_budget=BASE,
    )
    assert v1.verdict == v2.verdict == "more_redundant_than_language"


def test_rel_deltas_recorded_against_both_references():
    v = proxy.proxy_decision(
        geometry_redundancy=0.66,
        language_baseline=0.60,
        singapore_redundancy=0.60,
        base_tile_budget=BASE,
    )
    assert v.rel_language == pytest.approx((0.66 - 0.60) / 0.60)
    assert v.rel_singapore == pytest.approx((0.66 - 0.60) / 0.60)


def test_compression_redundancy_orders_repetitive_above_varied():
    repetitive = b"AB" * 5000
    varied = bytes((i * 97 + 13) % 256 for i in range(10000))
    assert proxy.compression_redundancy(repetitive) > proxy.compression_redundancy(varied)
    assert proxy.compression_redundancy(b"") == 0.0
