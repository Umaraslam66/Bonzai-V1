from __future__ import annotations

import os
import random
import subprocess
import sys

import pytest

from cfm.eval import lane_s_sampler as ls


def _minimal_payload() -> dict:
    return {
        "sampler_schema_version": ls.SAMPLER_SCHEMA_VERSION,
        "release": "test.0",
        "floor_sha256": "deadbeef",
        "methodology": {
            "target_features": 50,
            "headroom": 2.0,
            "seed": 7,
            "selection": "blake2b_hash_rank",
        },
        "held_out_cities": ["glasgow"],
        "strata": [],
        "cells": [],
    }


def test_seal_then_verified_load_round_trips(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    assert (tmp_path / ls.SAMPLER_LOCK_NAME).exists()
    loaded = ls.load_verified_manifest(path)
    assert loaded["release"] == "test.0"
    assert loaded["methodology"]["target_features"] == 50


def test_verified_load_refuses_tampered_content(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    text = path.read_text().replace("release: test.0", "release: tampered.9")
    path.write_text(text)
    with pytest.raises(ls.SamplerArtifactError, match="sha mismatch"):
        ls.load_verified_manifest(path)


def test_seal_is_write_once(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    with pytest.raises(FileExistsError):
        ls.seal_manifest(_minimal_payload(), path)


def test_binding_metric_is_building_when_owed():
    assert ls.binding_metric(frozenset({ls.BUILDING_METRIC, ls.ROAD_METRIC})) == ls.BUILDING_METRIC
    assert ls.binding_metric(frozenset({ls.ROAD_METRIC})) == ls.ROAD_METRIC


def test_size_stratum_ceiling_bound_depends_only_on_floor_n():
    # floor_n=50, target=50, headroom=1.0 -> raw=ceil(50*1*A/50)=A -> NOT ceiling-bound
    r = ls.size_stratum(target_features=50, headroom=1.0, floor_n_binding=50, available_cells=200)
    assert not r.ceiling_bound and r.n_cells_selected == 200 and r.n_cells_target == 200
    # floor_n=59 (the real min), headroom=2.0 -> target*headroom=100 > 59 -> ceiling-bound
    r2 = ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=59, available_cells=40)
    assert r2.ceiling_bound and r2.n_cells_selected == 40  # take-all
    assert r2.n_cells_target == 68  # ceil(50*2.0*40/59)
    # plentiful stratum: floor_n=3000, headroom=2.0, available big -> small draw, not ceiling
    r3 = ls.size_stratum(
        target_features=50, headroom=2.0, floor_n_binding=3000, available_cells=2000
    )
    assert not r3.ceiling_bound and r3.n_cells_selected < 2000
    # R3 invariant: available cancels in the ceiling test — ceiling_bound is True AND
    # n_cells_selected equals available_cells (take-all) regardless of pool size.
    for avail in (40, 200, 1000):
        ra = ls.size_stratum(
            target_features=50, headroom=2.0, floor_n_binding=59, available_cells=avail
        )
        assert ra.ceiling_bound, f"expected ceiling_bound=True for available={avail}"
        assert ra.n_cells_selected == avail, f"expected take-all for available={avail}"


def test_size_stratum_rejects_unfloored_n():
    with pytest.raises(ValueError, match="floor_n_binding"):
        ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=0, available_cells=10)


def test_size_stratum_rejects_empty_pool():
    with pytest.raises(ValueError, match="available_cells"):
        ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=10, available_cells=0)


# ---------------------------------------------------------------------------
# Task 3: Selection — blake2b hash-rank (PYTHONHASHSEED-proof)
# ---------------------------------------------------------------------------


def _cells(n: int) -> list[ls.SampledCell]:
    return [
        ls.SampledCell(
            city="glasgow", tile_i=i, tile_j=0, cell_i=i % 7, cell_j=i // 7, density_bucket=1
        )
        for i in range(n)
    ]


def test_select_cells_take_all_when_capped():
    cells = _cells(10)
    out = ls.select_cells(cells, 25, seed=7)
    assert len(out) == 10  # take-all, never over-draw


def test_select_cells_is_input_order_independent():
    cells = _cells(100)
    a = ls.select_cells(cells, 30, seed=7)
    shuffled = cells[:]
    random.Random(123).shuffle(shuffled)
    b = ls.select_cells(shuffled, 30, seed=7)
    assert a == b  # hash-rank keys on identity, not input order -> PYTHONHASHSEED-proof


def test_select_cells_seed_changes_subset():
    cells = _cells(100)
    assert ls.select_cells(cells, 30, seed=7) != ls.select_cells(cells, 30, seed=8)


def test_select_cells_output_canonically_sorted():
    out = ls.select_cells(_cells(100), 30, seed=7)
    assert out == sorted(out, key=ls._cell_sort_key)


def test_select_cells_stable_across_pythonhashseed(tmp_path):
    snippet = (
        "from cfm.eval import lane_s_sampler as ls\n"
        "cells=[ls.SampledCell('glasgow',i,0,i%7,i//7,1) for i in range(200)]\n"
        "out=ls.select_cells(cells,40,seed=7)\n"
        "print(';'.join(f'{c.tile_i},{c.cell_i},{c.cell_j}' for c in out))\n"
    )
    outs = []
    for hs in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2], "selection drifted across PYTHONHASHSEED"
