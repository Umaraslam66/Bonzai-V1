"""Tests for scripts/run_conditioning_floor.py (Task 25 step 1; spec §8).

The runner is a thin driver over ``cfm.eval.conditioning_floor``: extract
held-out (optionally + training) real features, compute the real-real pair
table, run the integrity halts, derive floors + discriminating strata, and
freeze the sha-stamped write-once artifact. All tests are synthetic: the
extraction fn is monkeypatched (no real corpus locally, no Leonardo).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from cfm.eval.conditioning_discrimination import ExtractionResult, TileCoverage
from cfm.eval.conditioning_floor import (
    FLOOR_ARTIFACT_LOCK_NAME,
    FloorCollapseError,
    load_verified_floor,
)

_REPO = Path(__file__).resolve().parents[2]
_RELEASE = "2026-04-15.0"

_SA = ("R", "S1", 1, "inland")
_M = "building_area_m2"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_conditioning_floor", _REPO / "scripts" / "run_conditioning_floor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grid(shift: int, n: int = 100) -> list[float]:
    return [float(i + shift) for i in range(n)]


def _coverage(cities: list[str]) -> dict[str, TileCoverage]:
    return {
        c: TileCoverage(n_tiles_expected=10, n_tiles_read=10, n_tiles_skipped=0, n_bref_excluded=2)
        for c in cities
    }


def _fake_extraction(shifts: dict[str, int], n: int = 100):
    """An extraction stand-in returning the uniform-grid fixture (exact KS)."""

    def _extract(release: str, cities) -> ExtractionResult:
        assert release == _RELEASE
        assert sorted(cities) == sorted(shifts)
        features = {(c, _SA, _M): _grid(s, n=n) for c, s in shifts.items()}
        return ExtractionResult(features=features, tile_coverage=_coverage(list(shifts)))

    return _extract


def _write_manifest(tmp_path: Path, held_out: list[str] | None) -> Path:
    path = tmp_path / "holdout_manifest.yaml"
    data: dict = {"manifest_schema_version": "2.0"}
    if held_out is not None:
        data["held_out_cities"] = held_out
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_e2e_artifact_lands_verifies_and_summary_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Tooth 7: runner e2e on the synthetic healthy fixture — exit 0, artifact +
    lock marker on disk, loads VERIFIED, floors are the strict-min values."""
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 20, "t2_city": 40}
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))
    manifest = _write_manifest(tmp_path, ["d_city", "t1_city", "t2_city"])
    out = tmp_path / "conditioning-floor.yaml"

    rc = mod.main(
        [
            "--release",
            _RELEASE,
            "--holdout-manifest",
            str(manifest),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()
    verified = load_verified_floor(out)
    floors = {(r["city"], r["metric"], tuple(r["stratum"])): r for r in verified.payload["floors"]}
    d_floor = floors[("d_city", _M, _SA)]
    assert d_floor["floor"] == pytest.approx(0.2)  # strict min(0.2, 0.4)
    assert d_floor["floor_median_context"] == pytest.approx(0.3)
    assert verified.payload["tile_coverage"]["d_city"]["n_bref_excluded"] == 2
    assert verified.payload["train_cities"] == []
    summary = capsys.readouterr().out
    assert "conditioning-floor" in summary
    assert str(out) in summary


def test_held_out_cities_read_is_strict_never_get(tmp_path: Path) -> None:
    """Correction #12: a manifest without held_out_cities is REFUSED loudly —
    never .get(..., []) silently scoping zero cities."""
    mod = _load_module()
    manifest = _write_manifest(tmp_path, held_out=None)
    with pytest.raises(ValueError, match="held_out_cities"):
        mod.main(
            [
                "--release",
                _RELEASE,
                "--holdout-manifest",
                str(manifest),
                "--out",
                str(tmp_path / "floor.yaml"),
            ]
        )


def test_unsupported_zero_qualifying_pairs_is_loud_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 20}
    monkeypatch.setattr(
        mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts, n=10)
    )
    manifest = _write_manifest(tmp_path, ["d_city", "t1_city"])
    out = tmp_path / "floor.yaml"
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        mod.main(["--release", _RELEASE, "--holdout-manifest", str(manifest), "--out", str(out)])
    assert not out.exists()
    assert not (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_missing_held_out_city_halts_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-25 spec review #3 at the runner seam: a held-out city whose samples
    never qualify (zero pairs -> no floor) halts BEFORE any artifact byte —
    no artifact, no lock marker — instead of freezing a write-once artifact
    that silently shrinks the Lane-S worst-case max domain weeks later."""
    mod = _load_module()
    cities = ["d_city", "t1_city", "ghost_city"]

    def _extract(release: str, extract_cities) -> ExtractionResult:
        assert release == _RELEASE
        assert sorted(extract_cities) == sorted(cities)
        features = {
            ("d_city", _SA, _M): _grid(0),
            ("t1_city", _SA, _M): _grid(20),
            ("ghost_city", _SA, _M): _grid(0, n=10),  # thin: zero qualifying pairs
        }
        return ExtractionResult(features=features, tile_coverage=_coverage(cities))

    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _extract)
    manifest = _write_manifest(tmp_path, cities)
    out = tmp_path / "floor.yaml"
    with pytest.raises(ValueError, match="ghost_city"):
        mod.main(["--release", _RELEASE, "--holdout-manifest", str(manifest), "--out", str(out)])
    assert not out.exists()
    assert not (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_collapse_halt_fires_before_any_artifact_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integrity halt BEFORE artifact bytes: nothing lands on disk on collapse."""
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 1, "t2_city": 2}  # median KS 0.01 < 0.049
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))
    manifest = _write_manifest(tmp_path, ["d_city", "t1_city", "t2_city"])
    out = tmp_path / "floor.yaml"
    with pytest.raises(FloorCollapseError):
        mod.main(["--release", _RELEASE, "--holdout-manifest", str(manifest), "--out", str(out)])
    assert not out.exists()
    assert not (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_write_once_second_run_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 20, "t2_city": 40}
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))
    manifest = _write_manifest(tmp_path, ["d_city", "t1_city", "t2_city"])
    out = tmp_path / "floor.yaml"
    argv = ["--release", _RELEASE, "--holdout-manifest", str(manifest), "--out", str(out)]
    assert mod.main(argv) == 0
    with pytest.raises(FileExistsError):
        mod.main(argv)


def test_include_train_cities_routes_through_the_union_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--include-train-cities resolves training cities via verify_union_manifests
    (the Gate-2 source) and extracts held-out + training together; the artifact
    records both city lists. The flag is OPTIONAL so the gated Leonardo run can
    stage held-out-only first."""
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 20, "t2_city": 40}
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))

    union_calls: list[dict] = []

    def _fake_union(release: str, *, g4_rollup, holdout_manifest) -> list[str]:
        union_calls.append(
            {"release": release, "g4_rollup": g4_rollup, "holdout_manifest": holdout_manifest}
        )
        return ["t1_city", "t2_city"]

    monkeypatch.setattr(mod, "verify_union_manifests", _fake_union)
    manifest = _write_manifest(tmp_path, ["d_city"])
    out = tmp_path / "floor.yaml"
    rc = mod.main(
        [
            "--release",
            _RELEASE,
            "--holdout-manifest",
            str(manifest),
            "--include-train-cities",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert len(union_calls) == 1
    assert union_calls[0]["release"] == _RELEASE
    verified = load_verified_floor(out)
    assert verified.payload["held_out_cities"] == ["d_city"]
    assert verified.payload["train_cities"] == ["t1_city", "t2_city"]


def test_default_g4_rollup_is_the_one_source_constant() -> None:
    """The --g4-rollup default derives from build_shards.DEFAULT_G4_ROLLUP
    (correction #12: one-sourced, never a hand-copied path literal)."""
    import scripts.run_conditioning_floor as rcf
    from cfm.data.training.build_shards import DEFAULT_G4_ROLLUP

    assert rcf._DEFAULT_G4_ROLLUP == DEFAULT_G4_ROLLUP
