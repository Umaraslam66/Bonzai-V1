"""Value-bearing conditioning prefix + identity-lock (Phase-2 bake-off Task 6)."""

from __future__ import annotations

import os
import subprocess
import sys

from cfm.data.training import conditioning
from cfm.eval.holdout import labels


def _prefix(**overrides: object) -> list[int]:
    base: dict[str, object] = dict(
        population_density_bucket=0,
        zoning_class=1,
        road_skeleton_class=1,
        cell_density_bucket=0,
        region="singapore",
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=7,
    )
    base.update(overrides)
    return conditioning.build_value_bearing_prefix(**base)  # type: ignore[arg-type]


def test_conditioning_derivation_is_the_one_source_by_identity() -> None:
    # The model conditioning and the eval both resolve to the SAME derivation object.
    assert conditioning.derive_tile_conditioning is labels._derive_tile_conditioning


def test_value_bearing_prefix_differs_across_distinct_conditioning() -> None:
    assert _prefix(population_density_bucket=0, cell_density_bucket=0) != _prefix(
        population_density_bucket=5, cell_density_bucket=5
    )


def test_prefix_ids_stay_above_subf_vocab() -> None:
    assert all(i >= conditioning.CONDITIONING_ID_BASE for i in _prefix())


def test_prefix_length_is_the_field_count() -> None:
    assert len(_prefix()) == conditioning.CONDITIONING_PREFIX_LEN == 8


def test_each_id_lands_in_its_own_fields_block() -> None:
    # field i's id is in [BASE + i*STRIDE, BASE + (i+1)*STRIDE)
    prefix = _prefix()
    base = conditioning.CONDITIONING_VALUE_BASE
    stride = conditioning._VALUE_STRIDE
    for i, tok in enumerate(prefix):
        assert base + i * stride <= tok < base + (i + 1) * stride


def test_seed_is_not_value_embedded() -> None:
    # changing only the seed must NOT change the prefix (seed is a sampling control)
    assert _prefix(seed=7) == _prefix(seed=999)


def test_none_label_buckets_to_zero_distinct_from_real_values() -> None:
    # an absent label (None) must not collide with a real value's bucket
    assert _prefix(population_density_bucket=None)[0] != _prefix(population_density_bucket=0)[0]


def test_value_bucket_injective_on_live_string_sets() -> None:
    """F6/F16 hygiene: the CURRENT live string set per string field is
    {None, "Asian-megacity"} (region is None in the multi-region shards;
    sub_c_morphology_class is the constant). Injectivity over the live set =
    None and the one live string land in different buckets, and bucket 0
    (the reserved None/absent bucket) is NEVER produced by a non-None value."""
    assert conditioning._value_bucket(None) == 0  # None -> reserved bucket 0
    live = conditioning._value_bucket("Asian-megacity")
    assert 1 <= live < conditioning._VALUE_STRIDE  # string -> 1..63, never 0
    assert live != conditioning._value_bucket(None)
    # no string can reach the reserved bucket (the +1 shift guarantees 1..63)
    for s in ("Asian-megacity", "singapore", "", "x", "europe", "almere", "0"):
        assert conditioning._value_bucket(s) != 0


def test_city_name_value_bucket_aliasing_is_pinned() -> None:
    """pinned 2026-06-10: if city names ever become value-bearing through
    _value_bucket, this aliasing is the cost — see readiness spec §3.2/§4.4;
    Task 24 gives city_identity its own field instead. The 38 Task-8 train-city
    names hash (SHA-256 % 63 + 1) into 8 colliding pairs, e.g. almere/umea,
    edinburgh/eindhoven, manchester/tilburg/toledo (a 3-way -> 3 pairs)."""
    import itertools
    import pathlib

    import yaml

    report_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "reports"
        / "2026-06-10-task8-multiregion-train-shards-build.yaml"
    )
    report = yaml.safe_load(report_path.read_text())
    cities = report["train_cities"]
    assert len(cities) == report["n_train_cities"] == 38
    buckets = {c: conditioning._value_bucket(c) for c in cities}
    colliding_pairs = [
        (a, b) for a, b in itertools.combinations(cities, 2) if buckets[a] == buckets[b]
    ]
    assert len(colliding_pairs) == 8  # pinned measurement (see docstring)


def test_string_field_encoding_is_deterministic_across_pythonhashseed() -> None:
    # builtin hash() is PYTHONHASHSEED-salted; SHA-256 must give the SAME id across cold
    # processes (determinism across runs is mandatory for comparability).
    snippet = (
        "from cfm.data.training.conditioning import build_value_bearing_prefix as b;"
        "print(b(population_density_bucket=0,zoning_class=1,road_skeleton_class=1,"
        "cell_density_bucket=0,region='singapore',coastal_inland_river=0,"
        "sub_c_morphology_class='Asian-megacity',seed=7)[4])"  # region slot id
    )
    outs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.check_output([sys.executable, "-c", snippet], env=env).strip())
    assert outs[0] == outs[1]
