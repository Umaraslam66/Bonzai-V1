"""Tests for scripts/run_conditioning_floor.py (Task 25 step 1; spec §8).

The runner is a thin driver over ``cfm.eval.conditioning_floor``: extract
held-out (optionally + training) real features, build the TWO BH families
(family 1 = D-D held-out pairwise, the determinism anchor + halts; family 2 =
D-T cross with its own BH; no T-T pairs — PI call 2026-06-11), derive the
two-variant floors + cross-selected discriminating strata, and freeze the
sha-stamped write-once artifact. All tests are synthetic: the extraction fn
is monkeypatched (no real corpus locally, no Leonardo).
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
from cfm.eval.holdout.lineage_audit import HoldoutLeakError
from cfm.eval.holdout.manifest import manifest_sha256

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
    """A properly FROZEN synthetic manifest: ``manifest_sha256`` stamped with the
    real freeze grammar + ``_EVAL_SET_LOCKED`` beside it — the runner now refuses
    anything less (Task-25 quality review #2: the F9 verified read)."""
    path = tmp_path / "holdout_manifest.yaml"
    data: dict = {"manifest_schema_version": "2.0"}
    if held_out is not None:
        data["held_out_cities"] = held_out
    data["manifest_sha256"] = manifest_sha256(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "_EVAL_SET_LOCKED").touch()
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
    assert d_floor["floor_heldout"] == pytest.approx(0.2)  # strict min(0.2, 0.4)
    assert d_floor["floor_heldout_median_context"] == pytest.approx(0.3)
    assert d_floor["floor_all"] == d_floor["floor_heldout"]  # no training cities
    assert verified.payload["tile_coverage"]["d_city"]["n_bref_excluded"] == 2
    assert verified.payload["train_cities"] == []
    assert verified.payload["cross_pairs"] == []  # held-out-only run: empty cross side
    assert verified.payload["cross_median_ks"] is None
    summary = capsys.readouterr().out
    assert "conditioning-floor" in summary
    assert "n/a (no cross pairs)" in summary
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


def test_tampered_manifest_is_refused_before_any_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-25 quality review #2 RED tooth: a post-freeze edit of
    held_out_cities (sha left stale) must be refused LOUDLY — via the holdout
    guard's F9 verifier — BEFORE any extraction runs, so a tampered manifest
    can never define the D-vs-T boundary the floor artifact freezes."""
    mod = _load_module()

    def _never_extract(release: str, cities) -> ExtractionResult:
        raise AssertionError("extraction must never run against a tampered manifest")

    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _never_extract)
    manifest = _write_manifest(tmp_path, ["d_city"])
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["held_out_cities"] = ["d_city", "attacker_city"]  # post-freeze edit; sha stale
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    out = tmp_path / "floor.yaml"
    with pytest.raises(HoldoutLeakError, match="sha mismatch"):
        mod.main(["--release", _RELEASE, "--holdout-manifest", str(manifest), "--out", str(out)])
    assert not out.exists()
    assert not (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_unsealed_manifest_missing_lock_marker_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The freeze seal travels with the file: sha intact but no _EVAL_SET_LOCKED
    beside the manifest -> refused before any extraction."""
    mod = _load_module()

    def _never_extract(release: str, cities) -> ExtractionResult:
        raise AssertionError("extraction must never run against an unsealed manifest")

    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _never_extract)
    manifest = _write_manifest(tmp_path, ["d_city"])
    (tmp_path / "_EVAL_SET_LOCKED").unlink()
    with pytest.raises(HoldoutLeakError, match="_EVAL_SET_LOCKED"):
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """--include-train-cities resolves training cities via verify_union_manifests
    (the Gate-2 source) and extracts held-out + training together; the artifact
    records both city lists, the cross family is populated with its own BH (no
    T-T pairs), floor_all tightens where a training city is closest, and the
    summary reports the cross median + the tightened-row count. The flag is
    OPTIONAL so the gated Leonardo run can stage held-out-only first."""
    mod = _load_module()
    # held-out {d:0, h:20}; train {t1:30, t2:50}: family-1 KS(d,h)=0.2;
    # cross KS = (d,t1) 0.3 / (d,t2) 0.5 / (h,t1) 0.1 / (h,t2) 0.3
    shifts = {"d_city": 0, "h_city": 20, "t1_city": 30, "t2_city": 50}
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))

    union_calls: list[dict] = []

    def _fake_union(release: str, *, g4_rollup, holdout_manifest) -> list[str]:
        union_calls.append(
            {"release": release, "g4_rollup": g4_rollup, "holdout_manifest": holdout_manifest}
        )
        return ["t1_city", "t2_city"]

    monkeypatch.setattr(mod, "verify_union_manifests", _fake_union)
    manifest = _write_manifest(tmp_path, ["d_city", "h_city"])
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
    assert verified.payload["held_out_cities"] == ["d_city", "h_city"]
    assert verified.payload["train_cities"] == ["t1_city", "t2_city"]
    # family 1 holds ONLY the held-out pair; cross holds the 4 D-T pairs
    assert [(r["city_a"], r["city_b"]) for r in verified.payload["pairs"]] == [("d_city", "h_city")]
    assert len(verified.payload["cross_pairs"]) == 4
    assert verified.payload["cross_median_ks"] == pytest.approx(0.3)
    # h's closest city is the TRAINING city t1 (KS 0.1): floor_all tightened
    floors = {(r["city"], r["metric"], tuple(r["stratum"])): r for r in verified.payload["floors"]}
    h_floor = floors[("h_city", _M, _SA)]
    assert h_floor["floor_heldout"] == pytest.approx(0.2)
    assert h_floor["floor_all"] == pytest.approx(0.1)
    d_floor = floors[("d_city", _M, _SA)]
    assert d_floor["floor_all"] == d_floor["floor_heldout"]  # d's closest stays held-out
    summary = capsys.readouterr().out
    assert "family-2 (D-T) pairs     : 4" in summary
    assert "cross median KS          : 0.3000" in summary
    assert "floor_all < floor_heldout: 1 rows" in summary  # h tightened, d not


def test_default_out_is_a_dedicated_per_release_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-25 quality review #4: the default artifact path is
    reports/conditioning_floor/<release>/conditioning-floor.yaml — a DEDICATED
    per-release directory, so the per-directory _CONDITIONING_FLOOR_LOCKED
    marker can never be a stale seal left beside unrelated reports/ artifacts."""
    mod = _load_module()
    shifts = {"d_city": 0, "t1_city": 20, "t2_city": 40}
    monkeypatch.setattr(mod, "extract_features_by_city_stratum_metric", _fake_extraction(shifts))
    monkeypatch.setattr(mod, "_REPO", tmp_path)  # the relative default resolves here
    manifest = _write_manifest(tmp_path, ["d_city", "t1_city", "t2_city"])

    rc = mod.main(["--release", _RELEASE, "--holdout-manifest", str(manifest)])  # no --out
    assert rc == 0
    expected = tmp_path / "reports" / "conditioning_floor" / _RELEASE / "conditioning-floor.yaml"
    assert expected.exists()
    assert (expected.parent / FLOOR_ARTIFACT_LOCK_NAME).exists()
    assert load_verified_floor(expected).payload["release"] == _RELEASE


def test_default_g4_rollup_is_the_one_source_constant() -> None:
    """The --g4-rollup default derives from build_shards.DEFAULT_G4_ROLLUP
    (correction #12: one-sourced, never a hand-copied path literal)."""
    import scripts.run_conditioning_floor as rcf
    from cfm.data.training.build_shards import DEFAULT_G4_ROLLUP

    assert rcf._DEFAULT_G4_ROLLUP == DEFAULT_G4_ROLLUP
