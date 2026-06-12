"""W2: the shared locked-YAML freeze grammar (rule of three met).

Three sealed-artifact instances share one grammar — sha over the canonical YAML
EXCLUDING the sha field itself, write-once, lock marker beside the file:

  1. holdout manifest      (manifest_sha256 / _EVAL_SET_LOCKED; marker managed
                            separately from the freeze)
  2. conditioning floor    (floor_sha256 / _CONDITIONING_FLOOR_LOCKED)
  3. city-identity registry (registry_sha256 / _CITY_REGISTRY_LOCKED)

``cfm.data.locked_yaml`` is the ONE source of the shared primitives. The
instances' verified-READ taxonomies deliberately stay per-instance: their check
ORDERS and exception shapes differ observably (floor/registry check the marker
FIRST; the holdout guard checks the sha shape FIRST and raises a structured
HoldoutLeakError) — the precedence pins below freeze those orders so the
extraction provably preserves behavior instead of silently unifying it.

Cross-instance regression: each instance's sha function must equal the shared
primitive on identical data INCLUDING the real sealed artifacts in-repo (the
external source of truth — a refactor that changes the hash bytes goes red
against files frozen before the refactor existed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.training.conditioning import (
    CITY_REGISTRY_LOCK_NAME,
    CITY_REGISTRY_PATH,
    city_registry_sha256,
    freeze_city_registry,
)
from cfm.data.training.holdout_guard import HoldoutLeakError, _verify_manifest_integrity
from cfm.eval.conditioning_floor import (
    FLOOR_ARTIFACT_LOCK_NAME,
    FloorArtifactError,
    floor_artifact_sha256,
    load_verified_floor,
)
from cfm.eval.holdout.manifest import manifest_sha256

_REPO = Path(__file__).resolve().parents[2]
_REAL_FLOOR = _REPO / "reports" / "conditioning_floor" / "2026-04-15.0" / "conditioning-floor.yaml"
_REAL_HOLDOUT = (
    _REPO
    / "data"
    / "processed"
    / "eval_set"
    / "2026-04-15.0"
    / "multiregion"
    / "holdout_manifest.yaml"
)

_PAYLOAD = {"alpha": [1, 2, 3], "nested": {"b": "x", "a": 0.5}, "schema": "9.9"}


# --- the shared sha primitive == every instance's sha function ------------------------


def test_sha_primitive_matches_floor_instance():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = {**_PAYLOAD, "floor_sha256": "garbage-to-exclude"}
    assert sha256_excluding_field(data, "floor_sha256") == floor_artifact_sha256(data)


def test_sha_primitive_matches_registry_instance():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = {**_PAYLOAD, "registry_sha256": "garbage-to-exclude"}
    assert sha256_excluding_field(data, "registry_sha256") == city_registry_sha256(data)


def test_sha_primitive_matches_manifest_instance():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = {**_PAYLOAD, "manifest_sha256": "garbage-to-exclude"}
    assert sha256_excluding_field(data, "manifest_sha256") == manifest_sha256(data)


# --- real sealed artifacts: the external source of truth ------------------------------


def test_real_registry_sha_pins_to_the_primitive():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = yaml.safe_load(CITY_REGISTRY_PATH.read_text(encoding="utf-8"))
    stored = data["registry_sha256"]
    assert sha256_excluding_field(data, "registry_sha256") == stored
    assert city_registry_sha256(data) == stored


def test_real_holdout_manifest_sha_pins_to_the_primitive():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = yaml.safe_load(_REAL_HOLDOUT.read_text(encoding="utf-8"))
    stored = data["manifest_sha256"]
    assert sha256_excluding_field(data, "manifest_sha256") == stored
    assert manifest_sha256(data) == stored


@pytest.mark.slow
def test_real_floor_artifact_sha_pins_to_the_primitive():
    from cfm.data.locked_yaml import sha256_excluding_field

    data = yaml.safe_load(_REAL_FLOOR.read_text(encoding="utf-8"))
    stored = data["floor_sha256"]
    assert sha256_excluding_field(data, "floor_sha256") == stored
    assert floor_artifact_sha256(data) == stored


@pytest.mark.slow
def test_real_floor_artifact_loads_through_the_sealed_reader():
    """Behavior-preservation pin on the SEALED reader against the SEALED file:
    the frozen artifact must keep loading through load_verified_floor unchanged."""
    artifact = load_verified_floor(_REAL_FLOOR)
    assert artifact.payload["floor_sha256"].startswith("95abb88b")
    assert artifact.payload["floor_schema_version"] == "2.0"


# --- stamp_and_seal -------------------------------------------------------------------


def test_stamp_and_seal_writes_verifiable_canonical_yaml(tmp_path: Path):
    from cfm.data.locked_yaml import sha256_excluding_field, stamp_and_seal

    p = tmp_path / "a" / "art.yaml"
    stamp_and_seal(dict(_PAYLOAD), p, sha_field="x_sha256", lock_name="_X_LOCKED")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["x_sha256"] == sha256_excluding_field(data, "x_sha256")
    assert (p.parent / "_X_LOCKED").exists()


def test_stamp_and_seal_refuses_overwrite(tmp_path: Path):
    from cfm.data.locked_yaml import stamp_and_seal

    p = tmp_path / "art.yaml"
    stamp_and_seal(dict(_PAYLOAD), p, sha_field="x_sha256", lock_name="_X_LOCKED")
    with pytest.raises(FileExistsError, match="write-once"):
        stamp_and_seal(dict(_PAYLOAD), p, sha_field="x_sha256", lock_name="_X_LOCKED")


def test_stamp_and_seal_without_marker_mode(tmp_path: Path):
    # the holdout-manifest case: the freeze writes no marker (managed separately)
    from cfm.data.locked_yaml import stamp_and_seal

    p = tmp_path / "art.yaml"
    stamp_and_seal(dict(_PAYLOAD), p, sha_field="x_sha256", lock_name=None)
    assert p.exists()
    assert list(tmp_path.iterdir()) == [p]  # no marker file appeared


def test_stamp_and_seal_is_byte_deterministic(tmp_path: Path):
    from cfm.data.locked_yaml import stamp_and_seal

    p1, p2 = tmp_path / "d1" / "art.yaml", tmp_path / "d2" / "art.yaml"
    stamp_and_seal(dict(_PAYLOAD), p1, sha_field="x_sha256", lock_name=None)
    stamp_and_seal(dict(_PAYLOAD), p2, sha_field="x_sha256", lock_name=None)
    assert p1.read_bytes() == p2.read_bytes()


# --- verify_sealed_yaml: the generic reader for NEW instances (W3 shard cache) --------


class _NewLockError(RuntimeError):
    pass


def _seal(tmp_path: Path, payload: dict | None = None) -> Path:
    from cfm.data.locked_yaml import stamp_and_seal

    p = tmp_path / "art.yaml"
    stamp_and_seal(
        payload if payload is not None else {**_PAYLOAD, "kind": "thing", "v": "1.0"},
        p,
        sha_field="x_sha256",
        lock_name="_X_LOCKED",
    )
    return p


def _verify(p: Path):
    from cfm.data.locked_yaml import verify_sealed_yaml

    return verify_sealed_yaml(
        p,
        sha_field="x_sha256",
        lock_name="_X_LOCKED",
        schema_field="v",
        schema_version="1.0",
        required_key="kind",
        error=_NewLockError,
    )


def test_verify_sealed_yaml_happy_path(tmp_path: Path):
    data = _verify(_seal(tmp_path))
    assert data["kind"] == "thing"


def test_verify_sealed_yaml_refuses_missing_file(tmp_path: Path):
    with pytest.raises(_NewLockError, match="does not exist"):
        _verify(tmp_path / "absent.yaml")


def test_verify_sealed_yaml_refuses_missing_marker(tmp_path: Path):
    p = _seal(tmp_path)
    (p.parent / "_X_LOCKED").unlink()
    with pytest.raises(_NewLockError, match="_X_LOCKED"):
        _verify(p)


def test_verify_sealed_yaml_refuses_edited_content(tmp_path: Path):
    p = _seal(tmp_path)
    p.write_text(p.read_text(encoding="utf-8").replace("thing", "TAMPERED"), encoding="utf-8")
    with pytest.raises(_NewLockError, match="sha mismatch"):
        _verify(p)


def test_verify_sealed_yaml_refuses_missing_sha(tmp_path: Path):
    p = _seal(tmp_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    del data["x_sha256"]
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(_NewLockError, match="NO x_sha256"):
        _verify(p)


def test_verify_sealed_yaml_refuses_version_skew(tmp_path: Path):
    from cfm.data.locked_yaml import stamp_and_seal

    p = tmp_path / "art.yaml"
    stamp_and_seal(
        {**_PAYLOAD, "kind": "thing", "v": "0.9"}, p, sha_field="x_sha256", lock_name="_X_LOCKED"
    )
    with pytest.raises(_NewLockError, match=r"0\.9"):
        _verify(p)


def test_verify_sealed_yaml_refuses_missing_required_key(tmp_path: Path):
    p = _seal(tmp_path, payload={**_PAYLOAD, "v": "1.0"})  # no "kind"
    with pytest.raises(_NewLockError, match="kind"):
        _verify(p)


def test_verify_sealed_yaml_precedence_marker_before_sha(tmp_path: Path):
    # multiply-broken artifact: missing marker AND tampered content -> the MARKER
    # refusal must fire (the floor-taxonomy order new instances inherit)
    p = _seal(tmp_path)
    (p.parent / "_X_LOCKED").unlink()
    p.write_text(p.read_text(encoding="utf-8").replace("thing", "TAMPERED"), encoding="utf-8")
    with pytest.raises(_NewLockError, match="_X_LOCKED"):
        _verify(p)


# --- precedence pins on the EXISTING readers (behavior preservation) ------------------
# Written GREEN against the pre-extraction implementations; they must stay green
# afterward — the extraction must not silently unify the per-instance check orders.


def test_floor_reader_precedence_marker_before_sha(tmp_path: Path):
    from cfm.eval.conditioning_floor import freeze_floor_artifact

    p = tmp_path / "floor.yaml"
    freeze_floor_artifact({"floors": [], "floor_schema_version": "2.0", "held_out_cities": []}, p)
    (p.parent / FLOOR_ARTIFACT_LOCK_NAME).unlink()
    p.write_text(p.read_text(encoding="utf-8").replace("floors", "flxxrs"), encoding="utf-8")
    with pytest.raises(FloorArtifactError, match=FLOOR_ARTIFACT_LOCK_NAME):
        load_verified_floor(p)


def test_registry_reader_precedence_marker_before_sha(tmp_path: Path):
    from cfm.data.training.conditioning import CityRegistryError, load_city_registry

    p = tmp_path / "registry.yaml"
    freeze_city_registry(["aaa", "bbb"], p)
    (p.parent / CITY_REGISTRY_LOCK_NAME).unlink()
    p.write_text(p.read_text(encoding="utf-8").replace("aaa", "zzz"), encoding="utf-8")
    with pytest.raises(CityRegistryError, match=CITY_REGISTRY_LOCK_NAME):
        load_city_registry(p)


def test_holdout_guard_precedence_sha_shape_before_marker(tmp_path: Path):
    # the holdout taxonomy checks the sha SHAPE first (no marker exists here
    # either, but the missing-sha refusal must win — the historical order)
    with pytest.raises(HoldoutLeakError, match="NO manifest_sha256"):
        _verify_manifest_integrity({"held_out_cities": []}, tmp_path / "holdout_manifest.yaml")
