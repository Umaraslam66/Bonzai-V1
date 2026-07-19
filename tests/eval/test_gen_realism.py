"""Gen-side 4-tuple keying (Phase-2 bake-off Task 11/[B]).

Two teeth:
  A. gen_features_by_city emits keys in the floor's EXACT 4-tuple grammar
     ``(metric, (zoning, road_skeleton, density, coastal))``.
  B. GRAMMAR TRIPWIRE (red-before-green): Lane-S only populates when gen is keyed in that
     grammar. A wrong-grammar gen (e.g. density-only) makes Lane-S VACUOUS — every floor
     stratum skipped-thin, the "zero qualifying" refusal — which is exactly the silent failure
     that the 4-tuple keying exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.eval.conditioning_floor import (
    build_floor_artifact_payload,
    freeze_floor_artifact,
    lane_s_excess,
)
from cfm.eval.gen_realism import DecodedCell, gen_features_by_city

_M = "building_area_m2"
_STRAT = (1, 2, 3, 0)  # a floor 4-tuple: (zoning, road_skeleton, density, coastal)


def _grid(shift: int, n: int = 100) -> list[float]:
    """n distinct values: KS(grid(a), grid(b)) == |a-b|/n exactly (mirrors the floor tests)."""
    return [float(i + shift) for i in range(n)]


# ----------------------- Tooth A: the 4-tuple grammar is emitted ----------------------- #


def test_gen_features_emit_floor_4tuple_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    import cfm.eval.gen_realism as gr

    class _Lbl:
        class morphology_stratum:
            dominant_zoning_class = 1
            modal_road_skeleton_class = 2

        coastal_inland_river = 0

    # Mock the IO (read_tile_labels) + the shared classifier (_tile_features); the unit under
    # test is the KEYING, not decode/label IO. _tile_features echoes the density it is handed.
    monkeypatch.setattr(gr, "read_tile_labels", lambda d, *, tile_i, tile_j: _Lbl())
    monkeypatch.setattr(gr, "epsg_label_for_region", lambda c: "EPSG")
    monkeypatch.setattr(gr, "sub_d_region_dir", lambda rel, c: Path("/x"))
    monkeypatch.setattr(gr, "tile_dirname", lambda i, j, e: "t")
    # _tile_features returns (features, n_bref_excluded); mirror that 2-tuple shape.
    monkeypatch.setattr(
        gr,
        "_tile_features",
        lambda blocks, geoms, dens: ([(_M, 10.0, dens[0]), ("road_length_m", 5.0, dens[0])], 0),
    )

    cells = [
        DecodedCell(
            city="krakow", tile_i=0, tile_j=0, cell_density_bucket=3, blocks=[[1]], geoms=[{}]
        )
    ]
    out = gen_features_by_city(cells, release="2026-04-15.0")

    # exactly the floor grammar: (metric, (zoning=1, skeleton=2, density=3, coastal=0))
    assert set(out["krakow"]) == {(_M, (1, 2, 3, 0)), ("road_length_m", (1, 2, 3, 0))}
    for _metric, stratum in out["krakow"]:
        assert len(stratum) == 4  # never density-only (1-tuple) or a permuted shape


# --------------- Tooth B: grammar tripwire (red-before-green) via Lane-S --------------- #


def _frozen_floor(tmp_path: Path) -> Path:
    # 2 held-out (d_city, t1_city) + 1 train (t2_city), one stratum; KS(d,t1)=0.3 in
    # [0.049, 0.5] so neither collapse nor explosion fires -> d_city gets a floor.
    feats = {
        ("d_city", _STRAT, _M): _grid(0),
        ("t1_city", _STRAT, _M): _grid(30),
        ("t2_city", _STRAT, _M): _grid(50),
    }
    payload = build_floor_artifact_payload(
        feats,
        release="test",
        held_out_cities=["d_city", "t1_city"],
        train_cities=["t2_city"],
        min_n=50,
        alpha=0.05,
        delta=0.15,
    )
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def test_wrong_key_grammar_makes_lane_s_vacuous_right_grammar_populates(tmp_path: Path) -> None:
    path = _frozen_floor(tmp_path)
    real = {(_M, _STRAT): _grid(0)}

    # GREEN: gen keyed by (metric, 4-tuple) matching the floor -> the stratum is SCORED.
    gen_right = {(_M, _STRAT): _grid(10)}
    res = lane_s_excess(gen_right, real, path, city="d_city", min_n=50)
    assert res.n_qualifying == 1

    # RED: gen keyed by a density-only 1-tuple (the grammar the eval used to discard 3 dims
    # under) -> the floor's (metric, 4-tuple) key never matches -> every stratum skipped-thin
    # -> the loud "zero qualifying" refusal (a silently-vacuous Lane-S, caught).
    gen_wrong = {(_M, (3,)): _grid(10)}
    with pytest.raises(ValueError, match="zero qualifying"):
        lane_s_excess(gen_wrong, real, path, city="d_city", min_n=50)
