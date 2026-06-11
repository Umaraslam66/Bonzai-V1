"""City-identity registry teeth (readiness-closure Task 24a; spec §8 Lane S/D).

The registry is the sha-locked, append-only city-name -> bucket authority for the
``city_identity`` conditioning field. It exists because the generic string hash
(``_value_bucket``: sha256 % 63 + 1) COLLIDES over the 49 known cities (11 colliding
groups / 13 pairs, incl. madrid=rome, berlin=warsaw, manchester=tilburg=toledo) —
identity must be injective, so it is registry-indexed, never hashed.

Lock grammar mirrors the Task-20 holdout-manifest discipline: ``registry_sha256``
over the canonical YAML EXCLUDING itself + a ``_CITY_REGISTRY_LOCKED`` marker beside
the file; the reader REFUSES on sha mismatch / missing sha / missing marker /
unknown city (fail-loud, never silent bucket-0).
"""

from __future__ import annotations

import itertools

import pytest
import yaml

from cfm.data.training import conditioning
from cfm.data.training.conditioning import (
    CityRegistryError,
    city_identity_bucket,
    city_registry_sha256,
    freeze_city_registry,
    load_city_registry,
)

#: The frozen 49-name registry, pinned EXACTLY (append-only teeth): ids are
#: index+1, so any reorder or removal moves ids and MUST turn this red. Future
#: cities APPEND at the end — extend this tuple at its end, never edit within.
_FROZEN_49 = (
    "a_coruna",
    "almere",
    "amsterdam",
    "barcelona",
    "berlin",
    "bologna",
    "bruges",
    "budapest",
    "cergy",
    "copenhagen",
    "debrecen",
    "edinburgh",
    "eindhoven",
    "eisenhuttenstadt",
    "espoo",
    "glasgow",
    "hamburg",
    "helsinki",
    "karlsruhe",
    "krakow",
    "linz",
    "lisbon",
    "ljubljana",
    "lodz",
    "lyon",
    "madrid",
    "malmo",
    "manchester",
    "mannheim",
    "milton_keynes",
    "munich",
    "paris",
    "prague",
    "rome",
    "rotterdam",
    "singapore",
    "szczecin",
    "tallinn",
    "telford",
    "tilburg",
    "toledo",
    "turin",
    "tychy",
    "umea",
    "valencia",
    "vienna",
    "warsaw",
    "welwyn",
    "wolfsburg",
)


# ----- the committed registry: frozen list + append-only teeth -----


def test_committed_registry_is_the_frozen_49_list_in_frozen_order():
    """Red-on-divergence: any reorder/removal of an entry moves ids -> FAIL."""
    assert load_city_registry() == _FROZEN_49
    assert len(load_city_registry()) == 49


def test_committed_registry_verifies_under_the_freeze_grammar():
    """The on-disk artifact's stored sha equals the recomputed sha (grammar pin)."""
    loaded = yaml.safe_load(conditioning.CITY_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert loaded["registry_sha256"] == city_registry_sha256(loaded)
    assert (conditioning.CITY_REGISTRY_PATH.parent / "_CITY_REGISTRY_LOCKED").exists()


def test_bucket_is_registry_index_plus_one_and_fits_the_stride_block():
    buckets = [city_identity_bucket(c) for c in _FROZEN_49]
    assert buckets == list(range(1, 50))  # index+1; bucket 0 reserved for None
    assert max(buckets) < conditioning._VALUE_STRIDE  # 49 < 64: fits with headroom


def test_none_city_buckets_to_zero():
    assert city_identity_bucket(None) == 0


def test_unknown_city_refuses_loud_never_silent_bucket_zero():
    with pytest.raises(CityRegistryError, match="zurich"):
        city_identity_bucket("zurich")


# ----- the regime witness: the old hash collides; the registry does not -----


def test_madrid_and_rome_collide_under_the_old_hash_but_get_distinct_registry_ids():
    """The collision that FORCED the registry: sha256%63+1 maps madrid and rome to
    the same bucket (so it must never carry city identity); registry ids differ."""
    assert conditioning._value_bucket("madrid") == conditioning._value_bucket("rome")
    assert city_identity_bucket("madrid") != city_identity_bucket("rome")


def test_old_hash_collision_evidence_over_the_49_is_pinned():
    """11 colliding groups / 13 pairs over the 49 known cities under _value_bucket."""
    buckets = {c: conditioning._value_bucket(c) for c in _FROZEN_49}
    pairs = [(a, b) for a, b in itertools.combinations(_FROZEN_49, 2) if buckets[a] == buckets[b]]
    assert len(pairs) == 13
    assert ("berlin", "warsaw") in pairs
    assert {("manchester", "tilburg"), ("manchester", "toledo"), ("tilburg", "toledo")} <= set(
        pairs
    )
    # registry encoding is injective over all 49 (the property the hash cannot give)
    assert len({city_identity_bucket(c) for c in _FROZEN_49}) == 49


# ----- tamper / lock-marker / write-once teeth (tmp copies; never the real file) -----


def test_content_tamper_with_stale_sha_is_refused(tmp_path):
    path = tmp_path / "city_identity_registry.yaml"
    freeze_city_registry(["aaa", "bbb"], path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["cities"] = ["bbb", "aaa"]  # reorder = id reassignment; sha left stale
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CityRegistryError, match="sha"):
        load_city_registry(path)


def test_missing_sha_field_is_refused_fail_closed(tmp_path):
    path = tmp_path / "city_identity_registry.yaml"
    freeze_city_registry(["aaa"], path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["registry_sha256"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(CityRegistryError, match="registry_sha256"):
        load_city_registry(path)


def test_missing_lock_marker_is_refused(tmp_path):
    path = tmp_path / "city_identity_registry.yaml"
    freeze_city_registry(["aaa"], path)
    (tmp_path / "_CITY_REGISTRY_LOCKED").unlink()
    with pytest.raises(CityRegistryError, match="_CITY_REGISTRY_LOCKED"):
        load_city_registry(path)


def test_freeze_is_write_once(tmp_path):
    path = tmp_path / "city_identity_registry.yaml"
    freeze_city_registry(["aaa"], path)
    with pytest.raises(FileExistsError):
        freeze_city_registry(["aaa"], path)


def test_clean_tmp_freeze_round_trips(tmp_path):
    path = tmp_path / "city_identity_registry.yaml"
    freeze_city_registry(["aaa", "bbb"], path)
    assert load_city_registry(path) == ("aaa", "bbb")
