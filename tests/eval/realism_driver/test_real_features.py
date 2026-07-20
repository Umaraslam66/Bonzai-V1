"""Task 5 — real-feature extraction CLI (``scripts/realism_eval_real_features.py``)
and its serialization/decoder helpers (``cfm.eval.realism_driver.scoring``).

All torch-free: the tested surface is the YAML record schema (must load with the
SAME ``_features_from_records`` shape ``scripts/run_bakeoff_decision.py`` uses), the
train-city set read from a frozen floor artifact (the downstream memorization check
refuses a train-city-set mismatch), the write-once refusal, the ``--cities`` filter,
and the held-out ``ConditionedCell -> DecodedCell`` mapping. The heavy real extraction
(``build_shards_in_memory`` + ``flatten_shards_to_cells`` over 29 training cities, and
the manifest join) needs Leonardo data + torch and is an ops step (T8), never here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.realism_eval_real_features as rf
from cfm.eval.conditioning_floor import build_floor_artifact_payload, freeze_floor_artifact
from cfm.eval.realism_driver import scoring
from cfm.eval.realism_driver.conditioning import ConditionedCell

# The exact loader the decision runner uses to read a real-features YAML.
from scripts.run_bakeoff_decision import _features_from_records

# --------------------------------------------------------------------------- #
# Fixtures mirroring tests/eval/test_bakeoff_decision.py's floor-artifact shape
# --------------------------------------------------------------------------- #

_SA = ("R", "S1", 1, "inland")
_M = "building_area_m2"
_HELD = ["d_city", "h_city"]
_TRAIN_CITIES = ["t1_city", "t2_city"]
#: KS(d,h)=0.2 (family-1) sits between the collapse (0.049) and explosion (0.5) halts.
#: Held-out-ONLY so any train_cities set is legal (train cities need not appear in
#: features — the unknown-city halt only fires on a feature city in NEITHER set).
_SHIFTS = {"d_city": 0, "h_city": 20}


def _grid(shift: int, n: int = 100) -> list[float]:
    return [float(i + shift) for i in range(n)]


def _frozen_floor(tmp_path: Path, *, train_cities: list[str] = _TRAIN_CITIES) -> Path:
    features = {(c, _SA, _M): _grid(s) for c, s in _SHIFTS.items()}
    payload = build_floor_artifact_payload(
        features, release="test", held_out_cities=_HELD, train_cities=train_cities
    )
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def _gen_features() -> dict[tuple[str, tuple], list[float]]:
    """A synthetic per-city GenFeatures block (two (metric, stratum) keys, one
    with an int in the stratum tuple, so the YAML int/str roundtrip is exercised)."""
    return {
        (_M, _SA): [12.5, 30.0, 7.25],
        ("road_length_m", ("C", "S2", 3, "coastal")): [100.0, 250.5],
    }


# --------------------------------------------------------------------------- #
# Torch discipline
# --------------------------------------------------------------------------- #


def test_cli_module_is_torch_free():
    """Importing the CLI must not pull torch (arg-parse + serialization stay GPU-free)."""
    assert not hasattr(rf, "torch")


def test_scoring_module_is_torch_free():
    assert not hasattr(scoring, "torch")


# --------------------------------------------------------------------------- #
# Schema roundtrip against the DOWNSTREAM loader
# --------------------------------------------------------------------------- #


def test_features_records_have_exact_schema():
    records = scoring.features_to_records(_gen_features())
    for rec in records:
        assert set(rec) == {"metric", "stratum", "samples"}
        assert isinstance(rec["stratum"], list)
        assert all(isinstance(x, float) for x in rec["samples"])


def test_real_features_schema_roundtrips_through_decision_loader(tmp_path: Path):
    """GenFeatures -> YAML records -> reload via the runner's ``_features_from_records``
    must reproduce the original (metric, stratum)->samples mapping byte-for-byte."""
    real_by_city = {"d_city": _gen_features(), "h_city": _gen_features()}
    real_train_by_city = {"t1_city": _gen_features(), "t2_city": _gen_features()}
    payload = scoring.build_real_features_payload(
        meta={"spec": "test"},
        real_by_city=real_by_city,
        real_train_by_city=real_train_by_city,
    )
    out = tmp_path / "real-features.yaml"
    scoring.write_real_features(out, payload)

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(data["real_by_city"]) == {"d_city", "h_city"}
    assert set(data["real_train_by_city"]) == {"t1_city", "t2_city"}
    # The runner's own loader reproduces the exact keyed features.
    for city, feats in real_by_city.items():
        assert _features_from_records(data["real_by_city"][city]) == feats
    for city, feats in real_train_by_city.items():
        assert _features_from_records(data["real_train_by_city"][city]) == feats


def test_partial_payload_omits_the_uncomputed_half(tmp_path: Path):
    """A held-out-only partial omits ``real_train_by_city`` ENTIRELY so the runner's
    STRICT read refuses it (missing key), never silently accepts an empty train set."""
    payload = scoring.build_real_features_payload(
        meta={"spec": "test"}, real_by_city={"d_city": _gen_features()}
    )
    assert "real_by_city" in payload
    assert "real_train_by_city" not in payload


# --------------------------------------------------------------------------- #
# Train-city set is read FROM the floor artifact (never hardcoded)
# --------------------------------------------------------------------------- #


def test_train_cities_read_from_floor_artifact(tmp_path: Path):
    floor = _frozen_floor(tmp_path, train_cities=["alpha", "beta", "gamma"])
    assert rf.train_cities_from_floor(floor) == ["alpha", "beta", "gamma"]


def test_emitted_train_city_keys_equal_artifact(tmp_path: Path):
    """The canonical ``real_train_by_city`` keys must equal the artifact's
    ``train_cities`` exactly — else ``memorization_check`` refuses the artifact."""
    floor = _frozen_floor(tmp_path, train_cities=_TRAIN_CITIES)
    train_cities = rf.train_cities_from_floor(floor)
    real_train_by_city = {c: _gen_features() for c in train_cities}
    payload = scoring.build_real_features_payload(
        meta={"spec": "test"},
        real_by_city={c: _gen_features() for c in _HELD},
        real_train_by_city=real_train_by_city,
    )
    assert set(payload["real_train_by_city"]) == set(train_cities)


# --------------------------------------------------------------------------- #
# Write-once discipline
# --------------------------------------------------------------------------- #


def test_write_real_features_is_write_once(tmp_path: Path):
    out = tmp_path / "real-features.yaml"
    payload = scoring.build_real_features_payload(meta={}, real_by_city={"d_city": _gen_features()})
    scoring.write_real_features(out, payload)
    with pytest.raises(FileExistsError):
        scoring.write_real_features(out, payload)


# --------------------------------------------------------------------------- #
# --cities filter
# --------------------------------------------------------------------------- #


def test_select_cities_intersects_in_universe_order():
    got = rf.select_cities("t2_city,t1_city", ["t1_city", "t2_city", "t3_city"])
    assert got == ["t1_city", "t2_city"]  # universe order, not request order


def test_select_cities_none_returns_full_universe():
    assert rf.select_cities(None, ["a", "b"]) == ["a", "b"]


def test_select_cities_unknown_city_raises():
    with pytest.raises(SystemExit, match="not in"):
        rf.select_cities("nope", ["a", "b"])


# --------------------------------------------------------------------------- #
# Held-out ConditionedCell -> DecodedCell mapping
# --------------------------------------------------------------------------- #


def _conditioned(
    city: str, ti: int, tj: int, bucket: int, tokens: tuple[int, ...]
) -> ConditionedCell:
    return ConditionedCell(
        cell_key=(city, ti, tj, 0, 0),
        density_bucket=bucket,
        prefix_ids=tuple(range(10)),
        char_stats=(0.0,) * 7,
        real_body_tokens=tokens,
    )


def test_heldout_decoded_cells_maps_key_and_density():
    cells = [
        _conditioned("glasgow", 2, 5, 3, (1, 2, 3)),
        _conditioned("munich", 7, 1, 1, ()),
    ]
    decoded = rf.heldout_decoded_cells(cells)
    assert [(d.city, d.tile_i, d.tile_j, d.cell_density_bucket) for d in decoded] == [
        ("glasgow", 2, 5, 3),
        ("munich", 7, 1, 1),
    ]
    for d in decoded:
        assert isinstance(d.blocks, list)
        assert isinstance(d.geoms, list)
        assert len(d.blocks) == len(d.geoms)


def test_decode_tokens_to_cell_empty_is_empty():
    blocks, geoms = scoring.decode_tokens_to_cell(())
    assert blocks == []
    assert geoms == []


# --------------------------------------------------------------------------- #
# CLI arg surface
# --------------------------------------------------------------------------- #


def test_arg_parser_defaults():
    args = rf.build_arg_parser().parse_args(
        ["--floor-artifact", "f.yaml", "--manifest", "m.json", "--out", "o.yaml"]
    )
    assert args.floor_artifact == "f.yaml"
    assert args.manifest == "m.json"
    assert args.out == "o.yaml"
    assert args.release == rf.DEFAULT_RELEASE
    assert args.held_out_only is False
    assert args.train_only is False
    assert args.cities is None


def test_arg_parser_half_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        rf.build_arg_parser().parse_args(
            [
                "--floor-artifact",
                "f",
                "--manifest",
                "m",
                "--out",
                "o",
                "--held-out-only",
                "--train-only",
            ]
        )


# --------------------------------------------------------------------------- #
# Train extraction routing (Leonardo job 49904814 regression)
# --------------------------------------------------------------------------- #


def _noop_flattener(shards, *, seed, ablation):
    """Torch-free flattener stand-in: zero examples (routing is what's under test)."""
    return [], {}


def test_extract_train_routes_via_all_validated_tiles_never_holdout(monkeypatch):
    """REGIME SPY (job 49904814): a TRAIN city's default extraction path must go
    through ``build_train_city_shards`` — explicit ALL-validated ``tile_ids`` — and
    must NEVER touch ``holdout_manifest_for_region`` (train cities have no holdout
    manifest; the old ``build_shards_in_memory(release, city)`` call left
    ``tile_ids=None``, whose holdout-subtracting path raises ``ValueError`` for any
    non-held-out region). This test FAILS under the old regime: the fake builder
    below would record ``tile_ids=None`` instead of the explicit sorted inventory."""
    import cfm.data.training.build_shards as bs

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "holdout_manifest_for_region called for a TRAIN city — the job-49904814 "
            "regression path is live again"
        )

    monkeypatch.setattr(bs, "holdout_manifest_for_region", _forbidden)
    # Synthetic validated inventory for the fake train city (unsorted on purpose:
    # build_train_city_shards must sort).
    monkeypatch.setattr(
        bs,
        "_validated_inventory",
        lambda release, region: [{"tile_i": 2, "tile_j": 1}, {"tile_i": 0, "tile_j": 3}],
    )

    seen: dict = {}

    def _fake_build_shards_in_memory(release, region, *, tile_ids=None):
        seen["release"], seen["region"], seen["tile_ids"] = release, region, tile_ids
        return ["shard-sentinel"]

    monkeypatch.setattr(bs, "build_shards_in_memory", _fake_build_shards_in_memory)

    got = rf.extract_train(
        release="test-rel",
        ablation="full",
        seed=0,
        train_cities=["a_coruna"],  # the region job 49904814 actually died on
        flattener=_noop_flattener,  # torch-free; shard_builder left DEFAULT (real routing)
    )
    assert seen["region"] == "a_coruna"
    # Explicit sorted all-validated tile ids — NEVER None (None = the holdout path).
    assert seen["tile_ids"] == [(0, 3), (2, 1)]
    assert got == {}  # zero examples -> zero feature cities (routing-only fixture)


def test_extract_train_injected_shard_builder_is_honored():
    """The injectable seam (Task-1 discipline): an injected builder is called once
    per train city with (release, city) and the real modules are never imported."""
    calls: list[tuple[str, str]] = []

    def _builder(release: str, city: str) -> list:
        calls.append((release, city))
        return []

    got = rf.extract_train(
        release="r",
        ablation="full",
        seed=0,
        train_cities=["x_city", "y_city"],
        shard_builder=_builder,
        flattener=_noop_flattener,
    )
    assert calls == [("r", "x_city"), ("r", "y_city")]
    assert got == {}
