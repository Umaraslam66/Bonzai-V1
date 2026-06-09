"""Multi-region realism aggregation: per-city KS + worst-case decision_ks (Task 6).

The teeth here are the GAP-not-DRIFT coverage (delta-spec §4): missing aggregation never
reddened the rebase suite, so these net-new tests ARE the aggregation's only proof. They lock:
- no pooling: KS is computed per city, never against a concatenated reference;
- worst-case, not mean: the decision binds on the worst city;
- the Rec-2 dict->list bridge in ``decision_ks`` (a dict passed to ``worst_case_city`` would
  not fail loudly -- it would iterate string keys and ``AttributeError`` at the wrong layer);
- the empty-feature edge surfaces as KS 1.0 and becomes the worst city.
"""

from __future__ import annotations

from cfm.eval.multiregion_realism import decision_ks, per_city_ks
from cfm.eval.realism import FeatureMetric


def test_per_city_ks_is_computed_against_each_city_separately_not_pooled() -> None:
    generated = {"glasgow": [10.0, 11.0, 12.0], "munich": [50.0, 51.0]}
    real = {"glasgow": [10.0, 11.0, 12.0], "munich": [10.0, 11.0]}
    pc = per_city_ks(generated, real, metric=FeatureMetric.BUILDING_AREA)
    assert pc["glasgow"].ks == 0.0  # identical -> 0
    assert pc["munich"].ks > 0.5  # far -> large, NOT diluted by glasgow's match


def test_decision_ks_is_worst_case_not_mean() -> None:
    generated = {"a": [1.0], "b": [1.0]}
    real = {"a": [1.0], "b": [100.0]}
    assert decision_ks(generated, real, metric=FeatureMetric.BUILDING_AREA) == 1.0


def test_module_exposes_no_pooled_or_concatenated_reference_path() -> None:
    # Structural guard: there is no pool/concat seam to accidentally call. The ONLY aggregation
    # is worst-case-over-cities; a pooled reference would dilute the worst city.
    import cfm.eval.multiregion_realism as mr

    assert not any("pool" in n.lower() or "concat" in n.lower() for n in dir(mr))


def test_a_matching_city_does_not_dilute_a_far_off_city() -> None:
    # Behavioral no-pooling guard: glasgow matching perfectly (KS 0.0) must NOT pull munich's
    # KS down. If cities were pooled into one reference, munich's far samples would mix with
    # glasgow's and its KS would shrink.
    generated = {"glasgow": [10.0, 11.0, 12.0], "munich": [50.0, 51.0, 52.0]}
    real = {"glasgow": [10.0, 11.0, 12.0], "munich": [10.0, 11.0, 12.0]}
    pc = per_city_ks(generated, real, metric=FeatureMetric.BUILDING_AREA)
    assert pc["glasgow"].ks == 0.0
    assert pc["munich"].ks == 1.0  # disjoint -> maximally far, undiluted
    # and the decision surfaces the undiluted worst city
    assert decision_ks(generated, real, metric=FeatureMetric.BUILDING_AREA) == 1.0


def test_decision_ks_returns_worst_city_proving_dict_to_list_bridge() -> None:
    # Rec-2 regression-lock: decision_ks must do worst_case_city(list(per_city.values())).ks.
    # Passing the dict straight to worst_case_city would iterate string keys and AttributeError
    # (max(dict, key=lambda c: c.ks) on str keys) -- this test would crash, not return 1.0.
    generated = {"x": [1.0, 2.0, 3.0], "y": [1.0]}
    real = {"x": [1.0, 2.0, 3.0], "y": [100.0]}
    # x matches at 0.0; y is far at 1.0 -> worst-case picks y.
    assert decision_ks(generated, real, metric=FeatureMetric.BUILDING_AREA) == 1.0


def test_empty_feature_city_is_ks_one_and_becomes_the_worst() -> None:
    # Empty-feature edge: ks_distance returns 1.0 when either sample is empty (a backbone
    # emitting no features of a kind is maximally unrealistic). decision_ks surfaces it.
    generated = {"good": [1.0, 2.0, 3.0], "silent": []}
    real = {"good": [1.0, 2.0, 3.0], "silent": [5.0, 6.0]}
    pc = per_city_ks(generated, real, metric=FeatureMetric.BUILDING_AREA)
    assert pc["good"].ks == 0.0
    assert pc["silent"].ks == 1.0
    assert decision_ks(generated, real, metric=FeatureMetric.BUILDING_AREA) == 1.0
