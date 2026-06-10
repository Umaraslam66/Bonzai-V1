"""Multi-region realism: per-city KS + the worst-case ``decision_ks`` y-value (bake-off Task 6).

The single-region eval (``realism.ks_distance``) has no city dimension. Generalization across
the held-out cities is a WORST-CASE property (delta-spec §4): the decision binds on the worst
city, NEVER a mean and NEVER a pooled (concatenated) cross-city reference. Pooling would let a
city that matches perfectly dilute a city that is far off, hiding a backbone that fails to
generalize. So KS is computed ONCE PER CITY here, then aggregated worst-case.

``decision_ks`` is the per-scale decision-axis y-value: at each ladder scale it produces the
backbone's single multi-region KS. Task 12 (Phase-D point-building) is what feeds these per-scale
values into ``curve.fit_scaling_curve`` as the ``(measured_node_h, KS)`` point tuples -- this
module does NOT build those points and does NOT touch ``fit_scaling_curve``. As of this commit no
call site builds per-scale points yet (the only ``fit_scaling_curve`` caller is ``test_curve.py``,
which hand-builds points); Task 12 must repoint that y-value to ``decision_ks``.
"""

from __future__ import annotations

from cfm.eval.city_aggregate import PerCityKS, worst_case_city
from cfm.eval.realism import FeatureMetric, ks_distance


def per_city_ks(
    generated_by_city: dict[str, list[float]],
    real_by_city: dict[str, list[float]],
    *,
    metric: FeatureMetric,
) -> dict[str, PerCityKS]:
    """Per-city KS over already-extracted feature samples -- one KS per city, never pooled.

    For each city, KS is computed against THAT city's own reference only. Cities are never
    concatenated into a single reference (pooling would dilute the worst city and defeat the
    worst-case bar). The geom -> float ``feature_samples`` step is the caller's job, upstream;
    ``metric`` is the per-metric context these samples were extracted under (e.g. building areas
    vs road lengths) -- it documents which metric this call scores and is carried for that
    context, not used to re-extract here.

    ``n_features`` is the reference feature count for the city's binding metric, fed forward to
    the #21 power gate in ``city_aggregate.binding_city_verdict``.
    """
    return {
        city: PerCityKS(
            city=city,
            ks=ks_distance(generated_by_city[city], real_by_city[city]),
            n_features=len(real_by_city[city]),
        )
        for city in real_by_city
    }


def decision_ks(
    generated_by_city: dict[str, list[float]],
    real_by_city: dict[str, list[float]],
    *,
    metric: FeatureMetric,
) -> float:
    """The decision-axis y-value: the WORST held-out city's KS (worst-case, never mean/pooled).

    Rec-2 (the dict->list seam, implemented HERE and nowhere else): ``worst_case_city`` takes a
    LIST of ``PerCityKS``. ``per_city_ks`` returns a dict keyed by city, so the explicit
    ``list(per_city.values())`` bridge is mandatory: passing the dict straight through would NOT
    fail loudly -- ``max(dict, key=lambda c: c.ks)`` would iterate the dict's STRING keys and
    raise ``AttributeError`` at the wrong layer (inside ``worst_case_city``), masking the seam.
    """
    per_city = per_city_ks(generated_by_city, real_by_city, metric=metric)
    return worst_case_city(list(per_city.values())).ks
