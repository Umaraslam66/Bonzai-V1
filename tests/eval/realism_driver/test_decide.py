"""Task 6 — scored realism-eval decision runner teeth (local, GPU-free).

The most safety-critical surface of the plan: it computes the scientific verdict of the
PI-approved scored eval. These teeth pin (in order of importance):

  * MEMORIZATION FIRST — a regurgitator halts BEFORE any ``lane_s_excess`` runs (a spy
    proves excess is never invoked past the halt).
  * THE SEED FLOOR IS ACTUALLY WIRED — same means/gap, only the seed spread changes, flips
    the verdict between ``BindingVerdict`` and ``NoDecisiveWinner`` (the whole point of the
    gap fix: ``decide()`` would have silently dropped the seed-noise floor with seed_sem=0).
  * ``NO_DECISIVE_WINNER`` is written verbatim (per-city gap + both floors), never softened.
  * STRICT held-out city-set equality across manifest / floor / real / every ckpt's gen.
  * a ceiling-bound short is EXCLUDED (not raised); a non-ceiling short RAISES.

Fixtures reuse the ``tests/eval/test_bakeoff_decision.py`` uniform-grid grammar (KS exact by
construction; the same screened floor regime).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.realism_eval_decide as decide_cli
from cfm.eval import lane_s_sampler as ls
from cfm.eval.city_aggregate import BindingVerdict, NoDecisiveWinner
from cfm.eval.conditioning_floor import (
    LaneSResult,
    build_floor_artifact_payload,
    freeze_floor_artifact,
    load_verified_floor,
)
from cfm.eval.lane_s_sampler import SamplerCoverageError
from cfm.eval.realism_driver import scoring
from cfm.eval.realism_driver.conditioning import (
    EXPECTED_CENSUS_SHA256,
    EXPECTED_FLOOR_SHA256,
    EXPECTED_N_CELLS,
    EXPECTED_N_STRATA,
    ManifestLineageError,
)
from cfm.eval.realism_driver.scoring import MemorizationHalt

# --------------------------------------------------------------------------- #
# Fixtures: uniform integer grids -> EXACT pairwise KS (the screened regime)
# --------------------------------------------------------------------------- #

_SA = ("R", "S1", 1, "inland")
_M = "building_area_m2"
_HELD = ["d_city", "h_city"]
_TRAIN_CITIES = ["t1_city", "t2_city"]
_SHIFTS = {"d_city": 0, "h_city": 20, "t1_city": 30, "t2_city": 50}


def _grid(shift: int, n: int = 100) -> list[float]:
    return [float(i + shift) for i in range(n)]


def _feat(shift: int) -> dict[tuple[str, tuple], list[float]]:
    return {(_M, _SA): _grid(shift)}


_REAL = {"d_city": _feat(0), "h_city": _feat(20)}
_TRAIN = {"t1_city": _feat(30), "t2_city": _feat(50)}
#: gen := t1's REAL samples for d (the regurgitator) — best excess AND Lane-M FAIL.
_GEN_MEMORIZER = {"d_city": _feat(30), "h_city": _feat(20)}
#: plausible-but-mediocre: far from d AND every training city (Lane-M PASS, excess 0.4 at d).
_GEN_HONEST = {"d_city": _feat(-60), "h_city": _feat(10)}
#: gen := D's own held-out samples (the oracle) — Lane-M PASS, excess 0 everywhere.
_GEN_ORACLE = {"d_city": _feat(0), "h_city": _feat(20)}


def _frozen_floor(tmp_path: Path) -> Path:
    features = {(c, _SA, _M): _grid(s) for c, s in _SHIFTS.items()}
    payload = build_floor_artifact_payload(
        features, release="test", held_out_cities=_HELD, train_cities=_TRAIN_CITIES
    )
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def _lane_s_manifest(
    *, cities: list[str] | None = None, target: int = 50, ceiling: bool = False
) -> dict:
    """A synthetic sealed-shaped Lane-S manifest (the fields assert_city_sets +
    verify_gen_coverage read: held_out_cities, methodology.target_features, strata)."""
    cs = cities or list(_HELD)
    return {
        "held_out_cities": sorted(cs),
        "methodology": {"target_features": target},
        "strata": [
            {
                "city": c,
                "stratum": list(_SA),
                "owed_metrics": [_M],
                "binding_metric": _M,
                "ceiling_bound": ceiling,
            }
            for c in cs
        ],
    }


def _config(floor: Path) -> dict:
    verified = load_verified_floor(floor)
    return {"floor_sha256": verified.payload["floor_sha256"], "commit": "deadbeef"}


def _ckpts(gen_by_backbone: dict[str, dict], *, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """{(backbone, seed): gen} — the SAME deterministic gen for every seed (seed_sem=0)."""
    return {(bb, s): gen for bb, gen in gen_by_backbone.items() for s in seeds}


# --------------------------------------------------------------------------- #
# Torch discipline
# --------------------------------------------------------------------------- #


def test_decide_cli_is_torch_free():
    assert not hasattr(decide_cli, "torch")


# --------------------------------------------------------------------------- #
# THE seed-floor test: same means/gap, spread flips the verdict
# --------------------------------------------------------------------------- #


def _ls(city: str, med: float) -> LaneSResult:
    return LaneSResult(
        city=city,
        per_stratum_excess={},
        median_excess=med,
        p90_excess=med,
        n_qualifying=1,
        n_skipped_thin=0,
    )


def test_seed_sem_drives_no_decisive():
    """WIDE seed spread on B (mean 0.25, values 0.0/0.05/0.70) -> the seed-noise floor
    (~0.226) exceeds the winner-vs-runner-up gap (0.15) even though gap CLEARS the tiny
    resolution floor (~0.043 at n=1000) -> NoDecisiveWinner. Shrinking B's spread
    (0.20/0.25/0.30) with the IDENTICAL means/gap -> BindingVerdict. This proves the
    seed-noise floor is actually wired (decide() with seed_sem=0 would crown both)."""
    n_ref = {"d_city": 1000}
    a = {("A", s): {"d_city": _ls("d_city", 0.10)} for s in (0, 1, 2)}

    wide = {
        ("B", 0): {"d_city": _ls("d_city", 0.0)},
        ("B", 1): {"d_city": _ls("d_city", 0.05)},
        ("B", 2): {"d_city": _ls("d_city", 0.70)},
    }
    verdict = scoring.aggregate_seed_verdict({**a, **wide}, n_reference_by_city=n_ref)
    assert isinstance(verdict, NoDecisiveWinner)
    assert "d_city" in verdict.demoted
    # the SEED floor (not resolution) is what bound the decision:
    assert verdict.seed_noise_floor["d_city"] > verdict.resolution_floor["d_city"]
    assert verdict.gap["d_city"] > verdict.resolution_floor["d_city"]  # gap DID clear resolution
    assert verdict.gap["d_city"] < verdict.seed_noise_floor["d_city"]  # ... but not seed-noise

    tight = {
        ("B", 0): {"d_city": _ls("d_city", 0.20)},
        ("B", 1): {"d_city": _ls("d_city", 0.25)},
        ("B", 2): {"d_city": _ls("d_city", 0.30)},
    }
    verdict2 = scoring.aggregate_seed_verdict({**a, **tight}, n_reference_by_city=n_ref)
    assert isinstance(verdict2, BindingVerdict)
    assert verdict2.winner == "A"  # lower mean excess (0.10 vs 0.25)
    assert verdict2.binding_city == "d_city"


def test_decisive_winner_when_gap_clears_both_floors():
    """Clean separation with tight seeds -> BindingVerdict naming the winner + binding city."""
    n_ref = {"d_city": 100, "h_city": 100}
    a = {  # oracle-like: excess ~0 everywhere
        ("A", s): {"d_city": _ls("d_city", 0.0), "h_city": _ls("h_city", 0.0)} for s in (0, 1, 2)
    }
    b = {  # honest-like: excess 0.4 at d, 0 at h
        ("B", s): {"d_city": _ls("d_city", 0.4), "h_city": _ls("h_city", 0.0)} for s in (0, 1, 2)
    }
    verdict = scoring.aggregate_seed_verdict({**a, **b}, n_reference_by_city=n_ref)
    assert isinstance(verdict, BindingVerdict)
    assert verdict.winner == "A"
    assert verdict.binding_city == "d_city"
    assert verdict.gap == pytest.approx(0.4)


def test_aggregate_requires_two_backbones():
    with pytest.raises(ValueError, match="2 backbone"):
        scoring.aggregate_seed_verdict(
            {("A", 0): {"d_city": _ls("d_city", 0.1)}}, n_reference_by_city={"d_city": 100}
        )


def test_aggregate_refuses_single_seed_per_backbone():
    """Review fix I-1: a single seed per backbone RAISES (never warn-and-proceed) — seed_sem
    would be 0.0, silently re-opening the exact seed-noise-floor hole this task closes."""
    with pytest.raises(ValueError, match="seed-noise floor"):
        scoring.aggregate_seed_verdict(
            {
                ("A", 0): {"d_city": _ls("d_city", 0.1)},
                ("B", 0): {"d_city": _ls("d_city", 0.5)},
            },
            n_reference_by_city={"d_city": 100},
        )


def test_cli_run_shape_refuses_partial_artifact_sets():
    """Review fix I-1 at the CLI boundary: the locked shape is exactly 2 backbones x 3 seeds.
    A 2-artifact partial (1 seed per backbone) is a SystemExit naming found-vs-expected;
    an unbalanced 3/2 split and a 3rd backbone refuse too; the exact locked shape passes."""
    partial = {("A", 0): {}, ("B", 0): {}}
    with pytest.raises(SystemExit, match="3 seeds") as exc:
        decide_cli.assert_locked_run_shape(partial)
    assert "'A': [0]" in str(exc.value)  # names what was found

    unbalanced = {("A", s): {} for s in (0, 1, 2)} | {("B", 0): {}, ("B", 1): {}}
    with pytest.raises(SystemExit, match="'B': 2"):
        decide_cli.assert_locked_run_shape(unbalanced)

    three_backbones = {(bb, s): {} for bb in ("A", "B", "C") for s in (0, 1, 2)}
    with pytest.raises(SystemExit, match="3 backbone"):
        decide_cli.assert_locked_run_shape(three_backbones)

    locked = {(bb, s): {} for bb in ("A", "B") for s in (0, 1, 2)}
    decide_cli.assert_locked_run_shape(locked)  # the locked shape passes silently


def test_aggregate_refuses_unbalanced_seed_counts():
    with pytest.raises(ValueError, match="seed count"):
        scoring.aggregate_seed_verdict(
            {
                ("A", 0): {"d_city": _ls("d_city", 0.1)},
                ("A", 1): {"d_city": _ls("d_city", 0.1)},
                ("B", 0): {"d_city": _ls("d_city", 0.2)},
            },
            n_reference_by_city={"d_city": 100},
        )


# --------------------------------------------------------------------------- #
# MEMORIZATION FIRST: a regurgitator halts before any lane_s_excess
# --------------------------------------------------------------------------- #


def test_memorization_halt_blocks_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A ckpt whose gen := training-city t1's real samples FAILS Lane-M -> MemorizationHalt
    is raised BEFORE any lane_s_excess call. A spy on the excess symbol the core uses proves
    it is never invoked past the halt (the PI-mandated ordering, provable)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    calls: list[str] = []

    def _spy_excess(*args, **kwargs):
        calls.append("lane_s_excess")
        raise AssertionError("lane_s_excess must NOT run past a memorization halt")

    monkeypatch.setattr(decide_cli, "lane_s_excess", _spy_excess)

    gen_by_ckpt = _ckpts({"memorizer": _GEN_MEMORIZER, "honest": _GEN_HONEST})
    with pytest.raises(MemorizationHalt, match="memoriz"):
        decide_cli.score_and_decide(
            gen_by_ckpt=gen_by_ckpt,
            real_by_city=_REAL,
            real_train_by_city=_TRAIN,
            verified=verified,
            manifest=_lane_s_manifest(),
            out_dir=tmp_path / "out",
            config=_config(tmp_path / "conditioning-floor.yaml"),
        )
    assert calls == []  # excess NEVER called
    # the halt wrote memorization.yaml and NO decision.yaml
    assert (tmp_path / "out" / "memorization.yaml").exists()
    assert not (tmp_path / "out" / "decision.yaml").exists()


# --------------------------------------------------------------------------- #
# STRICT held-out city-set equality (decide()'s Tooth-2)
# --------------------------------------------------------------------------- #


def test_city_set_mismatch_raises(tmp_path: Path):
    verified = load_verified_floor(_frozen_floor(tmp_path))
    gen = _ckpts({"a": _GEN_ORACLE, "b": _GEN_HONEST})
    # drop a held-out city from ONE checkpoint's gen
    bad = dict(gen)
    bad[("a", 0)] = {"d_city": _feat(0)}  # h_city missing
    with pytest.raises(ValueError, match="h_city"):
        scoring.assert_city_sets(_lane_s_manifest(), verified, _REAL, bad)


def test_city_set_mismatch_real_features_raises(tmp_path: Path):
    verified = load_verified_floor(_frozen_floor(tmp_path))
    gen = _ckpts({"a": _GEN_ORACLE, "b": _GEN_HONEST})
    with pytest.raises(ValueError, match="real_by_city"):
        scoring.assert_city_sets(_lane_s_manifest(), verified, {"d_city": _feat(0)}, gen)


# --------------------------------------------------------------------------- #
# Coverage: ceiling-bound excluded (not raised); non-ceiling short raises
# --------------------------------------------------------------------------- #


def test_coverage_ceiling_bound_excluded_not_raised():
    """A short binding metric in a CEILING-BOUND stratum is a data limit -> excluded, not
    raised; the SAME short in a non-ceiling stratum is a sampler bug -> SamplerCoverageError."""
    short_gen = {(_M, _SA): [1.0, 2.0]}  # 2 < target 5
    gen_by_ckpt = {("a", 0): {"d_city": short_gen}}

    ceiling_mf = _lane_s_manifest(cities=["d_city"], target=5, ceiling=True)
    reports = decide_cli.check_coverage(gen_by_ckpt, ceiling_mf)
    excluded = reports[("a", 0)].ceiling_bound_excluded
    assert ("d_city", _M, _SA) in excluded

    non_ceiling_mf = _lane_s_manifest(cities=["d_city"], target=5, ceiling=False)
    with pytest.raises(SamplerCoverageError, match="under-sized"):
        decide_cli.check_coverage(gen_by_ckpt, non_ceiling_mf)


# --------------------------------------------------------------------------- #
# End-to-end: decisive writes decision.yaml; NoDecisive is verbatim
# --------------------------------------------------------------------------- #


def test_score_and_decide_end_to_end_decisive(tmp_path: Path):
    """Oracle vs honest, 3 identical seeds each (seed_sem=0): memorization PASSES for both,
    the gap at d (0.4) clears the resolution floor -> DECISIVE, and decision.yaml carries the
    per-(backbone,seed,city) excess table + the per-city aggregation."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    gen_by_ckpt = _ckpts({"oracle": _GEN_ORACLE, "honest": _GEN_HONEST})
    verdict = decide_cli.score_and_decide(
        gen_by_ckpt=gen_by_ckpt,
        real_by_city=_REAL,
        real_train_by_city=_TRAIN,
        verified=verified,
        manifest=_lane_s_manifest(),
        out_dir=tmp_path / "out",
        config=_config(tmp_path / "conditioning-floor.yaml"),
    )
    assert isinstance(verdict, BindingVerdict)
    assert verdict.winner == "oracle"
    assert verdict.binding_city == "d_city"

    record = yaml.safe_load((tmp_path / "out" / "decision.yaml").read_text())
    assert record["verdict"]["verdict"] == "DECISIVE"
    assert record["verdict"]["winner"] == "oracle"
    assert record["config"]["floor_sha256"] == verified.payload["floor_sha256"]
    # per-(backbone, seed, city) table present, one row per ckpt*city (2 bb * 3 seed * 2 city)
    assert len(record["lane_s_median_excess"]) == 12
    honest_d = [
        r
        for r in record["lane_s_median_excess"]
        if r["backbone"] == "honest" and r["city"] == "d_city"
    ]
    assert all(r["median_excess"] == pytest.approx(0.4) for r in honest_d)
    # gen-features audit dumps written per ckpt
    assert (tmp_path / "out" / "gen-features-oracle-seed0.yaml").exists()
    assert (tmp_path / "out" / "summary.md").exists()


def test_no_decisive_is_reported_verbatim(tmp_path: Path):
    """Two backbones with IDENTICAL gen (gap 0 everywhere) -> no city separates ->
    NoDecisiveWinner, written verbatim with the named verdict + per-city (gap,
    resolution_floor, seed_noise_floor)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    gen_by_ckpt = _ckpts({"a": _GEN_HONEST, "b": _GEN_HONEST})  # identical -> gap 0
    verdict = decide_cli.score_and_decide(
        gen_by_ckpt=gen_by_ckpt,
        real_by_city=_REAL,
        real_train_by_city=_TRAIN,
        verified=verified,
        manifest=_lane_s_manifest(),
        out_dir=tmp_path / "out",
        config=_config(tmp_path / "conditioning-floor.yaml"),
    )
    assert isinstance(verdict, NoDecisiveWinner)

    record = yaml.safe_load((tmp_path / "out" / "decision.yaml").read_text())
    v = record["verdict"]
    assert v["verdict"] == "NO_DECISIVE_WINNER"  # named, never softened
    assert set(v["demoted_cities"]) == {"d_city", "h_city"}
    for city in ("d_city", "h_city"):
        assert set(v["per_city"][city]) == {"gap", "resolution_floor", "seed_noise_floor"}
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "NO_DECISIVE_WINNER" in summary


def test_decision_is_write_once(tmp_path: Path):
    """A second decision into the same out-dir refuses (write-once): re-deciding means
    deleting the old directory deliberately."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    gen_by_ckpt = _ckpts({"oracle": _GEN_ORACLE, "honest": _GEN_HONEST})
    kw = dict(
        gen_by_ckpt=gen_by_ckpt,
        real_by_city=_REAL,
        real_train_by_city=_TRAIN,
        verified=verified,
        manifest=_lane_s_manifest(),
        out_dir=tmp_path / "out",
        config=_config(tmp_path / "conditioning-floor.yaml"),
    )
    decide_cli.score_and_decide(**kw)
    with pytest.raises(FileExistsError):
        decide_cli.score_and_decide(**kw)


# --------------------------------------------------------------------------- #
# n_reference / decoded-cells helpers
# --------------------------------------------------------------------------- #


def test_n_reference_counts_only_floored_strata(tmp_path: Path):
    """The honest n_features: floored-strata real feature count only (never all-strata)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    real = {
        "d_city": {(_M, _SA): _grid(0), (_M, ("R", "S2", 1, "inland")): _grid(0)},  # 2nd unfloored
        "h_city": _feat(20),
    }
    n_ref = scoring.n_reference_by_city(verified, real)
    assert n_ref == {"d_city": 100, "h_city": 100}  # not 200 for d


def test_decoded_cells_from_artifact_maps_and_reuses_decode():
    """GenCellRecord -> DecodedCell reuses the aligned blocks/geoms and maps cell_key/density."""
    from cfm.eval.realism_driver.driver import GenCellRecord

    rec = GenCellRecord(
        cell_key=("glasgow", 2, 5, 0, 1),
        density_bucket=3,
        tokens=[1, 2, 3],
        blocks=[[1, 2]],
        geoms=[{"type": "x"}],
        self_terminated=True,
    )
    decoded = scoring.decoded_cells_from_artifact({"release": "r"}, [rec], release="r")
    assert len(decoded) == 1
    d = decoded[0]
    assert (d.city, d.tile_i, d.tile_j, d.cell_density_bucket) == ("glasgow", 2, 5, 3)
    assert d.blocks == [[1, 2]]
    assert d.geoms == [{"type": "x"}]


# --------------------------------------------------------------------------- #
# CLI arg surface
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# F-1: the PRODUCTION scored path pins the Lane-S manifest lineage
# --------------------------------------------------------------------------- #


def _sealed_manifest(path: Path, *, floor_sha: str, census_sha: str) -> None:
    """A validly-SEALED Lane-S manifest at ``path`` (seal check passes); lineage fields
    are caller-controlled so a wrong-sha manifest can be fed to the production loader."""
    payload = {
        "sampler_schema_version": ls.SAMPLER_SCHEMA_VERSION,
        "release": "eu.test",
        "floor_sha256": floor_sha,
        "census_sha256": census_sha,
        "methodology": {"target_features": 50, "headroom": 2.0, "seed": 7},
        "held_out_cities": ["glasgow"],
        "strata": [{"city": "glasgow", "stratum": [i, 0, 0, 0]} for i in range(EXPECTED_N_STRATA)],
        "cells": [
            {
                "city": "glasgow",
                "tile_i": 0,
                "tile_j": 0,
                "cell_i": i // 8,
                "cell_j": i % 8,
                "density_bucket": 0,
            }
            for i in range(EXPECTED_N_CELLS)
        ],
    }
    ls.seal_manifest(payload, path)


def test_production_path_refuses_wrong_sha_manifest(tmp_path: Path):
    """F-1: ``main()`` (the scored production path) loads the manifest through the PINNED
    ``load_verified_manifest_or_raise`` — a validly-sealed but wrong-floor-sha manifest is
    refused with ``ManifestLineageError`` BEFORE any gen artifact is read, so a
    differently-sealed manifest can never slip past the decide step. (The dry-run path keeps
    the bare loader and is not pinned — covered by test_dryrun.)"""
    floor = _frozen_floor(tmp_path)
    manifest_path = tmp_path / "sampler-manifest.yaml"
    _sealed_manifest(manifest_path, floor_sha="deadbeef", census_sha=EXPECTED_CENSUS_SHA256)
    with pytest.raises(ManifestLineageError, match="floor_sha256"):
        decide_cli.main(
            [
                "--gen-artifact",
                "never-read.json",  # manifest raises first; gen artifacts are never loaded
                "--real-features",
                "never-read-real.yaml",
                "--floor-artifact",
                str(floor),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_production_path_accepts_pinned_lineage_manifest(tmp_path: Path):
    """A manifest whose lineage MATCHES the pinned constants passes the loader (the gate lets
    the true Lane-S lineage through); scoring then proceeds past the manifest load and fails
    later on the placeholder gen artifact, proving the pin itself did not reject it."""
    floor = _frozen_floor(tmp_path)
    manifest_path = tmp_path / "sampler-manifest.yaml"
    _sealed_manifest(
        manifest_path, floor_sha=EXPECTED_FLOOR_SHA256, census_sha=EXPECTED_CENSUS_SHA256
    )
    # NOT ManifestLineageError: the pin accepts it, so failure comes later (unreadable gen).
    with pytest.raises(Exception) as exc:  # asserting NOT the lineage error, so any is fine
        decide_cli.main(
            [
                "--gen-artifact",
                str(tmp_path / "missing-gen.json"),
                "--real-features",
                str(tmp_path / "missing-real.yaml"),
                "--floor-artifact",
                str(floor),
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not isinstance(exc.value, ManifestLineageError)


def test_arg_parser_collects_repeated_gen_artifacts():
    args = decide_cli.build_arg_parser().parse_args(
        [
            "--gen-artifact",
            "a.json",
            "--gen-artifact",
            "b.json",
            "--real-features",
            "real.yaml",
            "--floor-artifact",
            "floor.yaml",
            "--manifest",
            "m.yaml",
            "--out-dir",
            "out/",
        ]
    )
    assert args.gen_artifacts == ["a.json", "b.json"]
    assert args.min_n is None
    assert args.release == decide_cli.DEFAULT_RELEASE
