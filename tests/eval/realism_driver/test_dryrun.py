"""Task 7a — LOCAL scoring dry-run teeth (GPU-free, torch-free).

The dry-run scores a TINY slice from ONE checkpoint artifact so a human can exercise the
decode -> gen-features -> coverage -> Lane-S wiring locally, WITHOUT a checkpoint and WITHOUT
the locked 2x3 scored-run shape. Its contract, pinned here:

  * it STOPS before any seed aggregation / verdict and is structurally INCAPABLE of a crown
    (no aggregate_seed_verdict / binding_city_verdict / score_and_decide reachable) and writes
    NO decision.yaml / memorization.yaml / summary.md;
  * ``verify_tokens`` re-decodes each cell's tokens and FAILS LOUD on any drift from the stored
    aligned (blocks, geoms) — the determinism check the 7b PASS criteria require;
  * ``single_stratum_gen_features`` is a REAL adaptation (heldout-cache stand-in carries no tile
    identity, so the disk-label ``gen_features_by_city`` cannot key the floor 4-tuple) — it
    reuses the SAME ``_tile_features`` classification and stamps one synthetic stratum.

The real-token fixtures below are actual glasgow cells lifted from
``data/_diag/heldout_cache.json`` (schema: {region, body_tokens, ...}) so the decode +
verify_tokens roundtrip exercises the true decoder, not a hand-rolled shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.realism_eval_decide as decide_cli
from cfm.eval.conditioning_floor import (
    build_floor_artifact_payload,
    freeze_floor_artifact,
    load_verified_floor,
)
from cfm.eval.gen_realism import DecodedCell
from cfm.eval.realism_driver import scoring
from cfm.eval.realism_driver.driver import GenCellRecord, write_gen_artifact

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_SA = ("R", "S1", 1, "inland")
_M = "building_area_m2"
_HELD = ["d_city", "h_city"]
_TRAIN = ["t1_city", "t2_city"]

#: A real, short, decodable glasgow cell (a Point/POI) — tokens end in <cell_end>=260.
_REAL_POINT_TOKENS = [509, 202, 314, 329, 354, 378, 510, 260]
_REAL_POINT_BLOCKS = [[509, 202, 314, 329, 354, 378, 510]]
_REAL_POINT_GEOMS = [{"type": "Point", "coordinates": [164.0, 96.5]}]

#: A real glasgow cell that decodes to ONE road -> _tile_features yields road_length_m=308.0.
_REAL_ROAD_TOKENS = [
    509,
    216,
    316,
    327,
    367,
    386,
    720,
    507,
    720,
    497,
    736,
    507,
    736,
    476,
    749,
    502,
    725,
    507,
    725,
    482,
    733,
    507,
    733,
    460,
    768,
    483,
    791,
    495,
    766,
    494,
    721,
    458,
    510,
    260,
]
_ROAD_METRIC = "road_length_m"
_ROAD_STRATUM = ("R", "S1", 2, "inland")  # density 2 -> matches the cell's conditioned bucket


def _grid(shift: int, n: int = 100) -> list[float]:
    return [float(i + shift) for i in range(n)]


def _frozen_floor(tmp_path: Path) -> Path:
    # Held-out cities MUST differ (D-D pairwise KS >= collapse floor) or freeze refuses.
    features = {
        ("d_city", _SA, _M): _grid(0),
        ("h_city", _SA, _M): _grid(20),
        ("t1_city", _SA, _M): _grid(30),
        ("t2_city", _SA, _M): _grid(50),
    }
    payload = build_floor_artifact_payload(
        features, release="test", held_out_cities=_HELD, train_cities=_TRAIN
    )
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def _real() -> dict[str, dict[tuple[str, tuple], list[float]]]:
    return {"d_city": {(_M, _SA): _grid(0)}, "h_city": {(_M, _SA): _grid(20)}}


def _rec(city: str, tokens: list[int], blocks: list[list[int]], geoms: list[dict]) -> GenCellRecord:
    return GenCellRecord(
        cell_key=(city, 0, 0, 0, 0),
        density_bucket=1,
        tokens=tokens,
        blocks=blocks,
        geoms=geoms,
        self_terminated=bool(tokens) and tokens[-1] == 260,
    )


def _fake_gen_features(cells, *, release):
    """Injected gen-features fn: 20 far-from-real building_area features per held-out city so
    Lane-S has >= min_n in the floored stratum (excess > 0). Bypasses the disk-label read."""
    return {c: {(_M, _SA): _grid(60)} for c in _HELD}


# --------------------------------------------------------------------------- #
# Torch discipline
# --------------------------------------------------------------------------- #


def test_scoring_is_torch_free():
    assert not hasattr(scoring, "torch")


# --------------------------------------------------------------------------- #
# 1. Smoke: the decode -> features -> coverage -> Lane-S wiring runs end-to-end
# --------------------------------------------------------------------------- #


def test_dry_run_score_smoke(tmp_path: Path):
    """7a on a 5-record synthetic slice: coverage OK, a non-empty LaneSResult per city, the
    expected 4-tuple stratum keys surfaced — and NO verdict field (a dry run cannot crown)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    records = [
        _rec("d_city", [1, 260], [[1]], [{"type": "Point"}]),
        _rec("h_city", [2, 260], [[2]], [{"type": "Point"}]),
        _rec("d_city", [3, 260], [[3]], [{"type": "Point"}]),
        _rec("h_city", [4, 260], [[4]], [{"type": "Point"}]),
        _rec("d_city", [5, 260], [[5]], [{"type": "Point"}]),
    ]
    report = scoring.dry_run_score(
        meta={"release": "test"},
        records=records,
        real_by_city=_real(),
        verified=verified,
        release="test",
        gen_features_fn=_fake_gen_features,
        manifest=None,
        min_n=5,
        verify_tokens=False,
    )
    assert report.n_cells == 5
    assert report.n_self_terminated == 5
    assert report.coverage.ok  # non-empty coverage report
    assert set(report.scored_cities) == {"d_city", "h_city"}
    for city in _HELD:
        r = report.lane_s_by_city[city]
        assert r.n_qualifying >= 1
        assert r.median_excess >= 0.0
    # the expected 4-tuple stratum keys are surfaced per city
    assert report.gen_stratum_keys["d_city"] == [(_M, _SA)]
    # a dry run is structurally incapable of a crown
    assert not hasattr(report, "verdict")


# --------------------------------------------------------------------------- #
# 2. A dry run is INCAPABLE of a crown / of writing a decision
# --------------------------------------------------------------------------- #


def test_dry_run_never_reaches_crown_or_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The crown symbols are sabotaged to raise; a would-be-decisive dry run still completes
    (they are never called) and writes nothing anywhere."""

    def _boom(*a, **k):
        raise AssertionError("crown path must be UNREACHABLE in a dry run")

    monkeypatch.setattr(scoring, "aggregate_seed_verdict", _boom)
    monkeypatch.setattr(scoring, "binding_city_verdict", _boom)
    monkeypatch.setattr(decide_cli, "score_and_decide", _boom)

    verified = load_verified_floor(_frozen_floor(tmp_path))
    records = [_rec("d_city", [1, 260], [[1]], [{}]), _rec("h_city", [2, 260], [[2]], [{}])]
    report = scoring.dry_run_score(
        meta={"release": "test"},
        records=records,
        real_by_city=_real(),
        verified=verified,
        release="test",
        gen_features_fn=_fake_gen_features,
        min_n=5,
        verify_tokens=False,
    )
    assert report.scored_cities  # it DID run the scoring chain
    for name in ("decision.yaml", "memorization.yaml", "summary.md"):
        assert not (tmp_path / name).exists()


def test_run_dry_run_cli_writes_nothing_and_prints_sentinel(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    """End-to-end through the CLI on a REAL road cell + --synthetic-stratum: the out-dir stays
    empty (no decision.yaml), the dry-run sentinel is printed, and NOT the decision sentinel."""
    # floor + real for the single road stratum/metric; >=2 held-out cities with distinct
    # distributions so the D-D pairwise floor does not collapse (only glasgow is scored below).
    features = {
        ("glasgow", _ROAD_STRATUM, _ROAD_METRIC): _grid(0),
        ("g2_city", _ROAD_STRATUM, _ROAD_METRIC): _grid(20),
        ("t1_city", _ROAD_STRATUM, _ROAD_METRIC): _grid(30),
    }
    floor_payload = build_floor_artifact_payload(
        features,
        release="test",
        held_out_cities=["glasgow", "g2_city"],
        train_cities=["t1_city"],
    )
    floor_path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(floor_payload, floor_path)

    real_path = tmp_path / "real-features.yaml"
    scoring.write_real_features(
        real_path,
        scoring.build_real_features_payload(
            meta={"release": "test"},
            real_by_city={"glasgow": {(_ROAD_METRIC, _ROAD_STRATUM): _grid(0)}},
            real_train_by_city={"t1_city": {(_ROAD_METRIC, _ROAD_STRATUM): _grid(30)}},
        ),
    )

    # Store the REAL aligned decode so single_stratum's _tile_features yields the road feature;
    # verify_tokens (default ON in dry-run) re-decodes the tokens and confirms they reproduce it.
    from cfm.eval.realism_driver.scoring import decode_tokens_to_cell

    road_blocks, road_geoms = decode_tokens_to_cell(_REAL_ROAD_TOKENS)
    gen_path = tmp_path / "gen-mamba-seed0.json"
    write_gen_artifact(
        [
            GenCellRecord(
                cell_key=("glasgow", 0, 0, 0, 0),
                density_bucket=2,
                tokens=_REAL_ROAD_TOKENS,
                blocks=road_blocks,
                geoms=road_geoms,
                self_terminated=True,
            )
        ],
        gen_path,
        meta={"release": "test", "backbone": "mamba", "seed": 0},
    )

    out_dir = tmp_path / "out"
    args = decide_cli.build_arg_parser().parse_args(
        [
            "--dry-run",
            "--gen-artifact",
            str(gen_path),
            "--real-features",
            str(real_path),
            "--floor-artifact",
            str(floor_path),
            "--out-dir",
            str(out_dir),
            "--synthetic-stratum",
            "R,S1,2,inland",
            "--min-n",
            "1",
        ]
    )
    report = decide_cli.run_dry_run(args)
    assert report.n_cells == 1
    assert report.scored_cities == ["glasgow"]
    captured = capsys.readouterr()
    assert decide_cli.DRY_RUN_SENTINEL in captured.out
    assert decide_cli.SENTINEL not in captured.out  # never the decision sentinel
    # nothing written under the out-dir
    assert not out_dir.exists() or not any(out_dir.iterdir())


# --------------------------------------------------------------------------- #
# 3. verify_tokens: tampered tokens fail loud; honest tokens pass
# --------------------------------------------------------------------------- #


def test_verify_tokens_mismatch_raises(tmp_path: Path):
    """A record whose stored (blocks, geoms) disagree with re-decoding its own tokens FAILS
    LOUD under verify_tokens; the honest record (stored == re-decode) passes."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    tampered = GenCellRecord(
        cell_key=("d_city", 0, 0, 0, 0),
        density_bucket=1,
        tokens=_REAL_POINT_TOKENS,
        blocks=[],  # tamper: real tokens decode to one block, but stored says none
        geoms=[],
        self_terminated=True,
    )
    with pytest.raises(ValueError, match="determinism drift"):
        scoring.dry_run_score(
            meta={"release": "test"},
            records=[tampered],
            real_by_city=_real(),
            verified=verified,
            release="test",
            gen_features_fn=_fake_gen_features,
            min_n=5,
            verify_tokens=True,
        )

    honest = GenCellRecord(
        cell_key=("d_city", 0, 0, 0, 0),
        density_bucket=1,
        tokens=_REAL_POINT_TOKENS,
        blocks=_REAL_POINT_BLOCKS,
        geoms=_REAL_POINT_GEOMS,
        self_terminated=True,
    )
    report = scoring.dry_run_score(
        meta={"release": "test"},
        records=[honest],
        real_by_city=_real(),
        verified=verified,
        release="test",
        gen_features_fn=_fake_gen_features,
        min_n=5,
        verify_tokens=True,
    )
    assert report.verify_tokens is True


# --------------------------------------------------------------------------- #
# 4. single_stratum_gen_features: the ops adaptation is real (no disk read)
# --------------------------------------------------------------------------- #


def test_single_stratum_gen_features_on_real_cell():
    """A real road cell -> single_stratum -> the road_length_m feature keyed to the ONE
    synthetic 4-tuple stratum (no disk tile-label read)."""
    from cfm.eval.realism_driver.scoring import decode_tokens_to_cell

    blocks, geoms = decode_tokens_to_cell(_REAL_ROAD_TOKENS)
    cell = DecodedCell(
        city="glasgow",
        tile_i=0,
        tile_j=0,
        cell_density_bucket=2,
        blocks=blocks,
        geoms=geoms,
    )
    gen = scoring.single_stratum_gen_features([cell], stratum=_ROAD_STRATUM)
    assert (_ROAD_METRIC, _ROAD_STRATUM) in gen["glasgow"]
    assert gen["glasgow"][(_ROAD_METRIC, _ROAD_STRATUM)] == [pytest.approx(308.0)]
