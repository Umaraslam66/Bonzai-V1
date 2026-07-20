"""Task 1 — manifest -> matched conditioning join (fast, no torch/Leonardo).

The fake flattener returns namedtuple ``FakeCell`` objects mirroring the fields
``build_conditioned_cells`` reads off a real ``CellExample``: ``.key``,
``.prefix_ids``, ``.character_stats``, ``.tokens``. This keeps the unit suite off
Leonardo parquet and off torch (datamodule's ``flatten_shards_to_cells`` imports
torch; the core module must not).
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from cfm.eval.realism_driver import conditioning as cond

# Mirror of the real CellExample surface the join consumes (A1/A2/A6).
FakeCell = namedtuple("FakeCell", "key prefix_ids character_stats tokens")

# A valid 10-id prefix (9 value-bearing + 1 char placeholder) and a 7-float char vec.
_PREFIX10 = tuple(range(10))
_CHAR7 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _fake_cell(key: tuple, *, prefix=_PREFIX10, char=_CHAR7, tokens=(11, 12, 13)) -> FakeCell:
    return FakeCell(key=key, prefix_ids=prefix, character_stats=char, tokens=tokens)


def _make_injected(examples_by_city: dict[str, list[FakeCell]], *, recorder: dict | None = None):
    """Build (shard_builder, flattener) fakes over a city -> [FakeCell] map.

    shard_builder returns an opaque marker carrying the city + tile_ids so the
    flattener knows which city's examples to return (and the test can inspect the
    tile_ids that were derived from the manifest).
    """

    def shard_builder(release, city, *, tile_ids):
        return [("shards", release, city, tuple(tile_ids))]

    def flattener(shards, *, seed=0, ablation="full"):
        city = shards[0][2]
        if recorder is not None:
            recorder["ablation"] = ablation
            recorder["seed"] = seed
            recorder.setdefault("tile_ids", {})[city] = shards[0][3]
        return list(examples_by_city.get(city, [])), {"empty": 0, "too_long": 0}

    return shard_builder, flattener


def _manifest(cells: list[dict]) -> dict:
    return {"release": "eu.test", "cells": cells}


# Four cells across two cities, deliberately NOT in canonical sort order, to prove
# the join preserves manifest ORDER (glasgow tile 5 emitted before glasgow tile 1).
_CELLS = [
    {"city": "glasgow", "tile_i": 5, "tile_j": 0, "cell_i": 1, "cell_j": 2, "density_bucket": 3},
    {"city": "munich", "tile_i": 2, "tile_j": 2, "cell_i": 0, "cell_j": 0, "density_bucket": 0},
    {"city": "glasgow", "tile_i": 1, "tile_j": 0, "cell_i": 4, "cell_j": 4, "density_bucket": 1},
    {"city": "munich", "tile_i": 2, "tile_j": 2, "cell_i": 7, "cell_j": 7, "density_bucket": 2},
]


def _key(c: dict) -> tuple:
    return (c["city"], c["tile_i"], c["tile_j"], c["cell_i"], c["cell_j"])


def _examples_for(cells: list[dict]) -> dict[str, list[FakeCell]]:
    by_city: dict[str, list[FakeCell]] = {}
    for i, c in enumerate(cells):
        # give each cell a distinguishable prefix/token so we can assert the right
        # example landed on the right manifest cell
        fc = _fake_cell(_key(c), prefix=tuple(range(i, i + 10)), tokens=(100 + i, 200 + i))
        by_city.setdefault(c["city"], []).append(fc)
    return by_city


def test_join_maps_every_manifest_cell():
    shard_builder, flattener = _make_injected(_examples_for(_CELLS))
    out = cond.build_conditioned_cells(
        _manifest(_CELLS),
        release="eu.test",
        ablation="no_city",
        shard_builder=shard_builder,
        flattener=flattener,
    )
    assert len(out) == 4
    # manifest ORDER preserved (not canonical-sorted)
    assert [cc.cell_key for cc in out] == [_key(c) for c in _CELLS]
    for i, (cc, c) in enumerate(zip(out, _CELLS, strict=True)):
        assert cc.density_bucket == c["density_bucket"]
        assert cc.prefix_ids == tuple(range(i, i + 10))
        assert cc.char_stats == _CHAR7
        assert cc.real_body_tokens == (100 + i, 200 + i)


def test_unmatched_manifest_cell_raises():
    exemplars = _examples_for(_CELLS)
    # drop the third manifest cell's example -> its key is unmatched
    missing = _CELLS[2]
    exemplars["glasgow"] = [fc for fc in exemplars["glasgow"] if fc.key != _key(missing)]
    shard_builder, flattener = _make_injected(exemplars)
    with pytest.raises(cond.ConditioningJoinError) as ei:
        cond.build_conditioned_cells(
            _manifest(_CELLS),
            release="eu.test",
            ablation="full",
            shard_builder=shard_builder,
            flattener=flattener,
        )
    # the error names the missing 5-tuple
    assert str(_key(missing)) in str(ei.value)


def test_prefix_len_is_10_when_char_present():
    shard_builder, flattener = _make_injected(_examples_for(_CELLS))
    out = cond.build_conditioned_cells(
        _manifest(_CELLS),
        release="eu.test",
        ablation="full",
        shard_builder=shard_builder,
        flattener=flattener,
    )
    for cc in out:
        assert len(cc.prefix_ids) == 10  # 9 value-bearing + 1 char placeholder (A6)
        assert len(cc.char_stats) == 7  # A1


def test_ablation_is_threaded_not_hardcoded():
    recorder: dict = {}
    shard_builder, flattener = _make_injected(_examples_for(_CELLS), recorder=recorder)
    cond.build_conditioned_cells(
        _manifest(_CELLS),
        release="eu.test",
        ablation="no_city",
        shard_builder=shard_builder,
        flattener=flattener,
        conditioning_seed=0,
    )
    assert recorder["ablation"] == "no_city"
    # tile_ids for glasgow were derived from the manifest cells (tiles (1,0) and (5,0))
    assert recorder["tile_ids"]["glasgow"] == ((1, 0), (5, 0))


# --------------------------------------------------------------------------
# Manifest verification wrapper (shas + counts)
# --------------------------------------------------------------------------

from cfm.eval import lane_s_sampler as ls  # noqa: E402

_FLOOR_SHA = "95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6"
_CENSUS_SHA = "236cea99dc370021113352c9c737da2404791ad200ca6d8d7e908e81ca6cb373"


def _sealable_payload(*, floor_sha: str, census_sha: str, n_cells: int, n_strata: int) -> dict:
    return {
        "sampler_schema_version": ls.SAMPLER_SCHEMA_VERSION,
        "release": "eu.test",
        "floor_sha256": floor_sha,
        "census_sha256": census_sha,
        "methodology": {"target_features": 50, "headroom": 2.0, "seed": 7},
        "held_out_cities": ["glasgow"],
        "strata": [{"city": "glasgow", "stratum": [i, 0, 0, 0]} for i in range(n_strata)],
        "cells": [
            {
                "city": "glasgow",
                "tile_i": 0,
                "tile_j": 0,
                "cell_i": i // 8,
                "cell_j": i % 8,
                "density_bucket": 0,
            }
            for i in range(n_cells)
        ],
    }


def test_verification_accepts_correct_manifest(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(
        _sealable_payload(floor_sha=_FLOOR_SHA, census_sha=_CENSUS_SHA, n_cells=5705, n_strata=146),
        path,
    )
    loaded = cond.load_verified_manifest_or_raise(path)
    assert len(loaded["cells"]) == 5705
    assert len(loaded["strata"]) == 146


def test_manifest_verification_rejects_bad_sha(tmp_path):
    # (a) tampered sealed bytes -> the lane_s verify path raises SamplerArtifactError
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(
        _sealable_payload(floor_sha=_FLOOR_SHA, census_sha=_CENSUS_SHA, n_cells=5705, n_strata=146),
        path,
    )
    path.write_text(path.read_text().replace("release: eu.test", "release: tampered.9"))
    with pytest.raises(ls.SamplerArtifactError):
        cond.load_verified_manifest_or_raise(path)


def test_manifest_verification_rejects_wrong_census_sha(tmp_path):
    # (b) validly-sealed but census_sha256 != pinned constant -> AssertionError
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(
        _sealable_payload(floor_sha=_FLOOR_SHA, census_sha="c0ffee", n_cells=5705, n_strata=146),
        path,
    )
    with pytest.raises(AssertionError):
        cond.load_verified_manifest_or_raise(path)


def test_manifest_verification_rejects_wrong_cell_count(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(
        _sealable_payload(floor_sha=_FLOOR_SHA, census_sha=_CENSUS_SHA, n_cells=10, n_strata=146),
        path,
    )
    with pytest.raises(AssertionError):
        cond.load_verified_manifest_or_raise(path)
