"""Sub-F Singapore integration tests (T13) — the consolidated real-cache home.

ONE integration file for every real-cache-gated sub-F leg. This file SUPERSEDES
four per-module skip-stubs (deleted in the same commit):
  - test_pipeline.py::test_derive_region_against_real_singapore            (close-checklist #18)
  - test_pipeline_writer.py::test_encode_tile_against_real_sub_e_singapore (close-checklist #22)
  - test_validator_cross_tile.py::test_validate_cross_tile_against_real_region_singapore (#26)
  - test_per_axis_determinism.py::..._real_singapore_same_and_fresh_process (#25)
Singapore integration is CROSS-module by definition (it tests composition), so a
single home keeps the close-checklist 1:1 and makes the sub-E un-skip mechanical.

FAIL-LOUD, NOT SKIP (sub-E precedent at tests/data/sub_e/test_singapore_integration.py):
every test is `@pytest.mark.slow` (deselected from the default fast suite) and, when
run under `-m slow`, FAILS LOUD via `pytest.fail(<missing path>)` if a cache is
absent — it does NOT skip-and-continue. T13 is the de-risk gate before sub-F
close; a run that silently skips the real-cache legs and ships green would look
like "Singapore integration passed" while proving nothing. A gate that auto-passes
when its preconditions aren't met is not a gate (`feedback_gate_must_distinguish_regimes`).
So today (sub-E cache absent) `-m slow` for this file is RED by design; the default
`uv run pytest` stays green because these are deselected as slow.

When the caches regenerate, `-m slow` runs them as-is — mechanical un-skip preserved.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import yaml
from shapely.geometry import LineString, Polygon
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.boundary_contract import SubEContractViolation, load_boundary_contract
from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature
from cfm.data.sub_f.pipeline import PipelineConfig, derive_region
from cfm.data.sub_f.token_cost import chunked_segment_pairs
from cfm.data.sub_f.validator_cross_tile import validate_cross_tile
from cfm.data.sub_f.validator_inline import validate_inline

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[3]
_RELEASE = "2026-04-15.0"
# Cache-path convention is the sub-E precedent's
# (data/processed/sub_X/<release>/singapore), NOT the drifted §5.5-stub docstring
# (sub_e/2024-04-16-beta.3 / sub_c without a release). sub-C + sub-D caches exist
# locally at this release today; sub-E does not (project_sub_e_cache_absent...).
_SUB_C_REGION = _REPO / "data" / "processed" / "sub_c" / _RELEASE / "singapore"
_SUB_D_REGION = _REPO / "data" / "processed" / "sub_d" / _RELEASE / "singapore"
_SUB_E_REGION = _REPO / "data" / "processed" / "sub_e" / _RELEASE / "singapore"

# Round-trip thresholds — Halt-2 RE-LOCK (N=360, 2026-05-29). Position p99.9 was
# measured 3.7m, angle p95 measured 3.0°; gates set at the locked ceilings.
_POSITION_P999_MAX_M = 4.8
_ANGLE_P95_MAX_DEG = 4.0
_PINNED_UTC = "2026-05-30T00:00:00Z"


# ---------------------------------------------------------------------------
# Fail-loud cache gate + layer-3 subset fixture (mirrors the sub-E precedent)
# ---------------------------------------------------------------------------


def _require_caches() -> None:
    """Fail LOUD (not skip) if any required cache is absent, naming the path.

    derive_region reads: sub-D _SUCCESS (marker only), sub-E _SUCCESS + tile dirs
    (the authoritative tile set), and sub-C features.parquet per tile. So the
    three things that must exist are the sub-C region, the sub-D _SUCCESS, and
    the sub-E _SUCCESS. sub-E is the one absent today.
    """
    for label, marker in (
        ("sub-C region", _SUB_C_REGION),
        ("sub-D _SUCCESS", _SUB_D_REGION / "_SUCCESS"),
        ("sub-E _SUCCESS", _SUB_E_REGION / "_SUCCESS"),
    ):
        if not marker.exists():
            pytest.fail(
                f"Singapore {label} missing at {marker} — regenerate the upstream "
                f"cache (sub-A→…→sub-E at release {_RELEASE}) before running the "
                f"T13 de-risk gate. Fail-loud per sub-E precedent: do NOT skip-and-"
                f"continue (a silently-skipped de-risk run proves nothing)."
            )


def _layer3_subset_tiles() -> list[tuple[int, int]]:
    """Layer-3 9-tile subset from sub-D's locked macro_plan_vocab.yaml (sub-E
    precedent). A representative subset, not the full region: full Singapore in a
    test is enormous and adds no coverage over the subset."""
    vocab = yaml.safe_load(
        (_REPO / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml").read_text()
    )
    return [(t["tile_i"], t["tile_j"]) for t in vocab["selected_layer3_tiles"]]


def _build_filtered_inputs(work: Path) -> SimpleNamespace:
    """Symlink the layer-3 subset's sub-C + sub-E tile dirs into a filtered region
    tree (+ markers + the sub-E manifest). derive_region reads no sub-C / sub-D
    manifest, but DOES read the sub-E manifest for region_crs (spec §8), plus the
    markers, the sub-E tile dirs, and per-tile sub-C features."""
    sub_c = work / "sub_c" / "singapore"
    sub_d = work / "sub_d" / "singapore"
    sub_e = work / "sub_e" / "singapore"
    for d in (sub_c, sub_d, sub_e):
        d.mkdir(parents=True, exist_ok=True)
    (sub_d / "_SUCCESS").touch()
    (sub_e / "_SUCCESS").touch()
    # sub-F reads region_crs from the sub-E manifest (spec §8); symlink the real
    # one (carries region_crs: EPSG:3414) alongside the filtered tile dirs.
    (sub_e / "manifest.yaml").symlink_to(_SUB_E_REGION / "manifest.yaml")
    for ti, tj in _layer3_subset_tiles():
        tile = f"tile=EPSG3414_i{ti}_j{tj}"
        (sub_c / tile).symlink_to(_SUB_C_REGION / tile, target_is_directory=True)
        (sub_e / tile).symlink_to(_SUB_E_REGION / tile, target_is_directory=True)
    return SimpleNamespace(sub_c=sub_c, sub_d=sub_d, sub_e=sub_e)


@pytest.fixture(scope="module")
def derived_singapore(tmp_path_factory) -> SimpleNamespace:
    """Derive the layer-3 subset once for the module. Fail-loud if caches absent."""
    _require_caches()
    work = tmp_path_factory.mktemp("sub_f_singapore_intg")
    inp = _build_filtered_inputs(work)
    out = work / "sub_f" / "singapore"
    derive_region(
        PipelineConfig(
            release=_RELEASE,
            region="singapore",
            sub_c_region_dir=inp.sub_c,
            sub_d_region_dir=inp.sub_d,
            sub_e_region_dir=inp.sub_e,
            output_region_dir=out,
            extracted_utc=_PINNED_UTC,
            run_alpha_drop_report=False,  # never write into the committed reports/ tree
        )
    )
    return SimpleNamespace(out=out, sub_c=inp.sub_c, sub_d=inp.sub_d, sub_e=inp.sub_e)


# ---------------------------------------------------------------------------
# 1. End-to-end derive (was test_pipeline #18)
# ---------------------------------------------------------------------------


def test_singapore_end_to_end_derive(derived_singapore: SimpleNamespace) -> None:
    out = derived_singapore.out
    assert (out / "_SUCCESS").exists()
    assert (out / "manifest.yaml").exists()
    tiles = sorted(out.glob("tile=*"))
    assert tiles, "no tiles derived"
    shas: set[str] = set()
    for tile in tiles:
        assert (tile / "cells.parquet").exists()
        prov = yaml.safe_load((tile / "provenance.yaml").read_text())
        shas.add(prov["provenance_sha256"])
    assert len(shas) == len(tiles), "tile provenance shas must be distinct"


# ---------------------------------------------------------------------------
# 2 + 3. Determinism: same-process and fresh-process (was test_per_axis #25)
# cells.parquet carries no timestamp, so byte-identity holds without pinning
# the clock; we pin it anyway for fully reproducible provenance.
# ---------------------------------------------------------------------------


def _cells_shas(region: Path) -> dict[str, str]:
    return {
        p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(region.glob("tile=*/cells.parquet"))
    }


def test_singapore_determinism_same_process(
    derived_singapore: SimpleNamespace, tmp_path: Path
) -> None:
    first = _cells_shas(derived_singapore.out)
    assert first, "fixture produced no cells.parquet"
    rerun = tmp_path / "rerun" / "sub_f" / "singapore"
    derive_region(
        PipelineConfig(
            release=_RELEASE,
            region="singapore",
            sub_c_region_dir=derived_singapore.sub_c,
            sub_d_region_dir=derived_singapore.sub_d,
            sub_e_region_dir=derived_singapore.sub_e,
            output_region_dir=rerun,
            extracted_utc=_PINNED_UTC,
            run_alpha_drop_report=False,
        )
    )
    second = _cells_shas(rerun)
    assert first == second, f"same-process cells.parquet byte drift: {first.keys() ^ second.keys()}"


def test_singapore_determinism_fresh_process(
    derived_singapore: SimpleNamespace, tmp_path: Path
) -> None:
    """Two cold-Python derive.py subprocesses (PYTHONHASHSEED=random) → byte-
    identical cells.parquet (T5b cold-subprocess + hash-seed discipline, at the
    region scale)."""
    derive_py = _REPO / "scripts" / "sub_f" / "derive.py"
    out_shas: list[dict[str, str]] = []
    for run in ("a", "b"):
        out = tmp_path / run / "sub_f" / "singapore"
        subprocess.run(
            [
                sys.executable,
                str(derive_py),
                "--release",
                _RELEASE,
                "--region",
                "singapore",
                "--sub-c-region-dir",
                str(derived_singapore.sub_c),
                "--sub-d-region-dir",
                str(derived_singapore.sub_d),
                "--sub-e-region-dir",
                str(derived_singapore.sub_e),
                "--output-region-dir",
                str(out),
                "--extracted-utc",
                _PINNED_UTC,
                "--no-alpha-drop-report",
            ],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": "random"},
            timeout=600,
        )
        out_shas.append(_cells_shas(out))
    assert out_shas[0] and out_shas[0] == out_shas[1], "fresh-process cells.parquet byte drift"


# ---------------------------------------------------------------------------
# 4. Round-trip re-measure: honest position p99.9 + right-angle p95 (close-checklist #6 / §5.5)
#
# Uses the REAL locked encode_feature/decode_feature (360-bin) + the current
# token_cost.chunked_segment_pairs for the chunk-aware source->decoded vertex
# mapping. NOTE: it deliberately does NOT reuse scripts/sub_f/scope_halt2_*.py —
# both are hardcoded to the OLD 48-bin direction count (BIN_DEG_48 / DIR_COUNT=48)
# and would re-measure at the rejected quantization (recorded in close-checklist #6).
# ---------------------------------------------------------------------------


def _corner_angle_deg(prev, cur, nxt) -> float | None:
    """Interior angle at `cur` in degrees (pure geometry; inlined to avoid a
    dependency on the 48-bin-stale scope scripts)."""
    ax, ay = prev[0] - cur[0], prev[1] - cur[1]
    bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cosv))


def _single_part_ring_or_line(geom):
    """(coords, closed) for a single-part LineString/Polygon, else None."""
    gt = geom.geom_type
    if gt == "LineString":
        return list(geom.coords), False
    if gt == "Polygon":
        return list(geom.exterior.coords), True
    return None  # Point / Multi* skipped (encode_cell splits Multi* per part)


def _source_to_decoded_index(src_coords) -> list[int]:
    """Cumulative decoded index of each SOURCE vertex. decode_feature appends one
    vertex per (dir,mag) pair, and the encoder emits chunked_segment_pairs(seg)
    pairs per source segment; chunk-intermediate (collinear) vertices sit between
    source vertices and are admitted per spec §3.8."""
    idx = [0]
    cum = 0
    for k in range(1, len(src_coords)):
        seg = math.hypot(
            src_coords[k][0] - src_coords[k - 1][0], src_coords[k][1] - src_coords[k - 1][1]
        )
        cum += chunked_segment_pairs(seg)
        idx.append(cum)
    return idx


def test_singapore_roundtrip_position_and_angle(derived_singapore: SimpleNamespace) -> None:
    position_linf: list[float] = []
    angle_dev: list[float] = []  # non-catastrophic (<45°) deviations at right-angle corners

    for ti, tj in _layer3_subset_tiles():
        fp = derived_singapore.sub_c / f"tile=EPSG3414_i{ti}_j{tj}" / "features.parquet"
        if not fp.exists():
            continue
        for row in pq.ParquetFile(fp).read().to_pylist():
            parsed = _single_part_ring_or_line(canonicalize_geometry(wkb_loads(row["geometry"])))
            if parsed is None:
                continue
            src, closed = parsed
            geom = Polygon(src) if closed else LineString(src)
            ef = encode_feature(canonicalize_geometry(geom), semantic_tag="highway=residential")
            decoded = decode_feature(ef.tokens)["coordinates"]
            mapping = _source_to_decoded_index(src)
            if mapping[-1] >= len(decoded):
                continue  # defensive: grammar/decoder mismatch would surface in tests 1/5

            for k, sidx in enumerate(mapping):
                dx, dy = decoded[sidx]
                position_linf.append(max(abs(src[k][0] - dx), abs(src[k][1] - dy)))

            # Right-angle corners: input corner within 5° of 90° → post-round-trip deviation.
            n = len(src)
            for k in range(1, n - 1):
                in_ang = _corner_angle_deg(src[k - 1], src[k], src[k + 1])
                if in_ang is None or abs(in_ang - 90.0) > 5.0:
                    continue
                rt_ang = _corner_angle_deg(
                    decoded[mapping[k - 1]], decoded[mapping[k]], decoded[mapping[k + 1]]
                )
                if rt_ang is None:
                    continue
                dev = abs(rt_ang - 90.0)
                if dev < 45.0:  # non-catastrophic basis (the Halt-2 angle metric)
                    angle_dev.append(dev)

    assert position_linf, "no single-part geometries measured — caches or subset empty?"
    pos = sorted(position_linf)
    p999 = pos[min(len(pos) - 1, math.ceil(0.999 * len(pos)) - 1)]
    assert p999 <= _POSITION_P999_MAX_M, (
        f"round-trip position p99.9 = {p999:.3f}m > {_POSITION_P999_MAX_M}m "
        "(Halt-2 N=360 re-lock regressed on real data — surface as Halt-2-revisit-2)"
    )
    if angle_dev:
        ang = sorted(angle_dev)
        a95 = ang[min(len(ang) - 1, math.ceil(0.95 * len(ang)) - 1)]
        assert a95 <= _ANGLE_P95_MAX_DEG, (
            f"right-angle-corner post-deviation p95 = {a95:.3f}° > {_ANGLE_P95_MAX_DEG}° "
            "(the angle gate has NO synthetic proxy — this is its only binding check)"
        )


# ---------------------------------------------------------------------------
# 5. Encode/T8 layer on real sub-E: first-real-read + inline well-formedness
#    (was test_pipeline_writer #22; also the close-checklist #15(a) first-real-read)
# ---------------------------------------------------------------------------


def test_singapore_encode_layer_real_sub_e(derived_singapore: SimpleNamespace) -> None:
    """The source-derived sub-E reader must consume real sub-E without raising
    SubEContractViolation (close-checklist #15(a) — only verifiable on real
    sub-E), and every derived tile must pass the inline contract.

    (close-checklist #15(b) T3c stage-4 ratio + #15(c) motorway/MultiLineString
    spot-checks remain data-MEASUREMENT obligations on the checklist, not pass/fail
    gates here.)"""
    for tile in sorted(derived_singapore.sub_e.glob("tile=*")):
        try:
            load_boundary_contract(tile / "boundary_contract.parquet")
        except SubEContractViolation as e:
            pytest.fail(
                f"real sub-E {tile.name} violated the source-derived contract: {e} "
                "— sub-E source drifted since T8.5 plan-write; update boundary_contract.py."
            )
    for cells in sorted(derived_singapore.out.glob("tile=*/cells.parquet")):
        validate_inline(cells)  # raises InlineValidationError on any per-tile violation


# ---------------------------------------------------------------------------
# 6. Validator/T10 layer: cross-tile composite on real data (was test_validator_cross_tile #26)
# ---------------------------------------------------------------------------


def test_singapore_cross_tile_composite(derived_singapore: SimpleNamespace) -> None:
    """BP7 four-test composite + version/sha/all-cells on the real derived region.
    Runs inside derive_region too (before _SUCCESS); asserting directly is the
    explicit composite gate (mirrors the sub-E precedent)."""
    validate_cross_tile(
        derived_singapore.out, derived_singapore.sub_e, derived_singapore.sub_c
    )


# ---------------------------------------------------------------------------
# 7. CLI end-to-end: derive.py then validate.py via subprocess (NEW)
#    Crosses the process boundary where CLI-arg + serialization assumptions
#    can break silently (the T12 OS-pipe lesson, at region scale).
# ---------------------------------------------------------------------------


def test_singapore_cli_end_to_end(tmp_path: Path) -> None:
    _require_caches()
    inp = _build_filtered_inputs(tmp_path / "inputs")
    out = tmp_path / "sub_f" / "singapore"
    derive_py = _REPO / "scripts" / "sub_f" / "derive.py"
    validate_py = _REPO / "scripts" / "sub_f" / "validate.py"

    derive = subprocess.run(
        [
            sys.executable,
            str(derive_py),
            "--release",
            _RELEASE,
            "--region",
            "singapore",
            "--sub-c-region-dir",
            str(inp.sub_c),
            "--sub-d-region-dir",
            str(inp.sub_d),
            "--sub-e-region-dir",
            str(inp.sub_e),
            "--output-region-dir",
            str(out),
            "--extracted-utc",
            _PINNED_UTC,
            "--no-alpha-drop-report",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert derive.returncode == 0, f"derive.py failed: {derive.stderr}"
    assert (out / "_SUCCESS").exists()

    validate = subprocess.run(
        [
            sys.executable,
            str(validate_py),
            "--region-dir",
            str(out),
            "--sub-e-region-dir",
            str(inp.sub_e),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert validate.returncode == 0, f"validate.py failed: {validate.stderr}"
    assert "passed" in validate.stdout.lower()
