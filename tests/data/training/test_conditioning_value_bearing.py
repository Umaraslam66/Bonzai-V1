"""Value-bearing conditioning prefix + identity-lock (Phase-2 bake-off Task 6)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

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
        city_identity="singapore",  # Task 24a: 9th field, registry-encoded
    )
    base.update(overrides)
    return conditioning.build_value_bearing_prefix(**base)  # type: ignore[arg-type]


#: index of the city_identity slot (the appended 9th field) and its block base.
_CITY_SLOT = conditioning._CONDITIONING_FIELDS.index("city_identity")


def _city_block_base() -> int:
    return conditioning.CONDITIONING_VALUE_BASE + _CITY_SLOT * conditioning._VALUE_STRIDE


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
    # Task 24a: city_identity appended as the 9th field (append-only; never reorder)
    assert len(_prefix()) == conditioning.CONDITIONING_PREFIX_LEN == 9


def test_conditioning_id_span_is_576_after_the_city_identity_append() -> None:
    # 9 fields * 64 stride; CONDITIONING_ID_BASE itself must NOT move (append-only)
    assert conditioning.conditioning_id_span() == 9 * conditioning._VALUE_STRIDE == 576
    from cfm.data.sub_f.vocab import vocab_tag_to_id

    assert conditioning.CONDITIONING_ID_BASE == max(vocab_tag_to_id().values()) + 1


def test_city_identity_field_is_appended_last_never_reordered() -> None:
    assert conditioning._CONDITIONING_FIELDS[-1] == "city_identity"
    assert conditioning._CONDITIONING_FIELDS[:8] == (
        "population_density_bucket",
        "zoning_class",
        "road_skeleton_class",
        "cell_density_bucket",
        "region",
        "coastal_inland_river",
        "sub_c_morphology_class",
        "seed",
    )


def test_city_identity_encodes_via_registry_injectively_not_via_value_bucket() -> None:
    """madrid and rome COLLIDE under _value_bucket (sha256%63+1) — the builder must
    give them DISTINCT city-slot ids (registry index+1), and all 49 must be distinct."""
    assert conditioning._value_bucket("madrid") == conditioning._value_bucket("rome")
    assert _prefix(city_identity="madrid")[_CITY_SLOT] != _prefix(city_identity="rome")[_CITY_SLOT]
    ids = {_prefix(city_identity=c)[_CITY_SLOT] for c in conditioning.load_city_registry()}
    assert len(ids) == 49
    base = _city_block_base()
    assert all(base + 1 <= i < base + conditioning._VALUE_STRIDE for i in ids)  # never bucket 0


def test_none_city_identity_buckets_to_zero() -> None:
    assert _prefix(city_identity=None)[_CITY_SLOT] == _city_block_base()


def test_unknown_city_identity_refuses_loud() -> None:
    with pytest.raises(conditioning.CityRegistryError, match="zurich"):
        _prefix(city_identity="zurich")


def test_ablation_no_city_zeroes_only_the_city_slot() -> None:
    full = _prefix()
    ablated = _prefix(ablation="no_city")
    assert ablated[:_CITY_SLOT] == full[:_CITY_SLOT]  # the other 8 ids untouched
    assert ablated[_CITY_SLOT] == _city_block_base()  # city slot forced to bucket 0
    assert full[_CITY_SLOT] != ablated[_CITY_SLOT]  # the ablation actually ablates


def test_ablation_no_character_is_loud_until_task_24b() -> None:
    with pytest.raises(NotImplementedError, match="24b"):
        _prefix(ablation="no_character")


def test_unknown_ablation_value_raises() -> None:
    with pytest.raises(ValueError, match="ablation"):
        _prefix(ablation="no_everything")


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
        "sub_c_morphology_class='Asian-megacity',seed=7,"
        "city_identity='singapore')[4])"  # region slot id
    )
    outs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.check_output([sys.executable, "-c", snippet], env=env).strip())
    assert outs[0] == outs[1]
