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


# ---------------------------------------------------------------------------
# Task 4: Floor adapter — floored targets + held-out feature counts
# ---------------------------------------------------------------------------


def _floor_payload() -> dict:
    # Two held-out cities so a family-1 D-D pair exists per stratum. stratum lists per the floor.
    S = ["R", "S1", 1, "inland"]
    return {
        "held_out_cities": ["glasgow", "krakow"],
        "floors": [
            {"city": "glasgow", "metric": ls.BUILDING_METRIC, "stratum": S},
            {"city": "glasgow", "metric": ls.ROAD_METRIC, "stratum": S},
            {"city": "krakow", "metric": ls.ROAD_METRIC, "stratum": S},
        ],
        "pairs": [
            {
                "city_a": "glasgow",
                "city_b": "krakow",
                "metric": ls.BUILDING_METRIC,
                "stratum": S,
                "n_a": 59,
                "n_b": 120,
            },
            {
                "city_a": "glasgow",
                "city_b": "krakow",
                "metric": ls.ROAD_METRIC,
                "stratum": S,
                "n_a": 800,
                "n_b": 950,
            },
        ],
        "cross_pairs": [],
    }


def test_floored_targets_groups_owed_metrics_and_binding():
    targets = ls.floored_targets(_floor_payload())
    assert ("glasgow", ("R", "S1", 1, "inland")) in targets
    g = targets[("glasgow", ("R", "S1", 1, "inland"))]
    assert g.owed_metrics == frozenset({ls.BUILDING_METRIC, ls.ROAD_METRIC})
    assert g.binding_metric == ls.BUILDING_METRIC
    k = targets[("krakow", ("R", "S1", 1, "inland"))]
    assert k.owed_metrics == frozenset({ls.ROAD_METRIC}) and k.binding_metric == ls.ROAD_METRIC


def test_heldout_feature_counts_reads_n_from_pairs():
    counts = ls.heldout_feature_counts(_floor_payload())
    assert counts[("glasgow", ls.BUILDING_METRIC, ("R", "S1", 1, "inland"))] == 59
    assert counts[("krakow", ls.ROAD_METRIC, ("R", "S1", 1, "inland"))] == 950


# ---------------------------------------------------------------------------
# Task 5: Per-stratum cell-key census — emit + read
# ---------------------------------------------------------------------------


def test_cell_census_round_trips_grouped_by_city_4tuple(tmp_path):
    rows = [
        ls.SampledCell("glasgow", 1, 2, 0, 0, 1),
        ls.SampledCell("glasgow", 1, 2, 0, 1, 1),
        ls.SampledCell("krakow", 3, 4, 5, 6, 2),
    ]
    strata = {  # (city, tile_i, tile_j) -> (zoning, skeleton, coastal)
        ("glasgow", 1, 2): ("R", "S1", "inland"),
        ("krakow", 3, 4): ("C", "S2", "coastal"),
    }
    path = tmp_path / "census.parquet"
    ls.write_cell_census(rows, strata, path)
    pool = ls.read_cell_census(path)
    # grouped by (city, 4-tuple); density from the cell, (zoning,skeleton,coastal) from the tile
    assert len(pool[("glasgow", ("R", "S1", 1, "inland"))]) == 2
    assert pool[("krakow", ("C", "S2", 2, "coastal"))][0].cell_i == 5


def test_cell_census_byte_deterministic(tmp_path):
    """Two writes of identical input must produce identical bytes."""
    rows = [
        ls.SampledCell("glasgow", 1, 2, 0, 0, 1),
        ls.SampledCell("krakow", 3, 4, 5, 6, 2),
    ]
    strata = {
        ("glasgow", 1, 2): ("R", "S1", "inland"),
        ("krakow", 3, 4): ("C", "S2", "coastal"),
    }
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    ls.write_cell_census(rows, strata, path_a)
    ls.write_cell_census(rows, strata, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_cell_census_byte_deterministic_reversed_input_order(tmp_path):
    """Write the SAME logical cells in REVERSED input order — output must still be byte-identical.

    This pins the manifest-reproducibility guarantee: rows.sort() inside write_cell_census
    makes the output input-order-independent, not merely same-order-idempotent.
    """
    rows = [
        ls.SampledCell("glasgow", 1, 2, 0, 0, 1),
        ls.SampledCell("glasgow", 1, 2, 0, 1, 1),
        ls.SampledCell("krakow", 3, 4, 5, 6, 2),
    ]
    strata = {
        ("glasgow", 1, 2): ("R", "S1", "inland"),
        ("krakow", 3, 4): ("C", "S2", "coastal"),
    }
    path_a = tmp_path / "fwd.parquet"
    path_b = tmp_path / "rev.parquet"
    ls.write_cell_census(rows, strata, path_a)
    ls.write_cell_census(list(reversed(rows)), strata, path_b)
    assert path_a.read_bytes() == path_b.read_bytes(), (
        "write_cell_census is NOT input-order-independent: the canonical sort is broken"
    )


# ---------------------------------------------------------------------------
# Task 6: Build orchestrator — manifest assembly
# ---------------------------------------------------------------------------


def test_build_manifest_sizes_and_selects_per_stratum(tmp_path):
    floor = _floor_payload()  # glasgow: building n=59 + road n=800; krakow: road n=950
    S = ("R", "S1", 1, "inland")
    pool = {
        ("glasgow", S): [ls.SampledCell("glasgow", 0, 0, i % 9, i // 9, 1) for i in range(40)],
        ("krakow", S): [ls.SampledCell("krakow", 0, 0, i % 9, i // 9, 1) for i in range(500)],
    }
    payload = ls.build_manifest(
        floor_payload=floor,
        floor_sha256="abc123",
        cell_pool=pool,
        release="test.0",
        seed=7,
        target_features=50,
        headroom=2.0,
    )
    by_key = {(s["city"], tuple(s["stratum"])): s for s in payload["strata"]}
    g = by_key[("glasgow", S)]
    # glasgow building: target*headroom=100 > 59 -> ceiling-bound -> take all 40
    assert g["binding_metric"] == ls.BUILDING_METRIC and g["ceiling_bound"] is True
    assert g["n_cells_selected"] == 40
    k = by_key[("krakow", S)]
    # krakow road n=950: raw=ceil(50*2*500/950)=ceil(52.6)=53 -> not ceiling-bound
    assert k["binding_metric"] == ls.ROAD_METRIC and k["ceiling_bound"] is False
    assert k["n_cells_selected"] == 53
    # cells[] holds exactly the selected union
    assert len(payload["cells"]) == 40 + 53
    assert payload["floor_sha256"] == "abc123"


def test_build_manifest_skips_strata_absent_from_pool(tmp_path, caplog):
    floor = _floor_payload()
    payload = ls.build_manifest(
        floor_payload=floor,
        floor_sha256="abc",
        cell_pool={},
        release="test.0",
        seed=7,
        target_features=50,
        headroom=2.0,
    )
    assert payload["strata"] == [] and payload["cells"] == []
    assert "no census cells" in caplog.text.lower()


def test_select_cells_stable_across_pythonhashseed():
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
