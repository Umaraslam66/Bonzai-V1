"""Task 13 tests: extract_region — process-pool parallelization.

Spec §14.5 invariant: "Pool size affects wall-clock; byte output is invariant
under `pool_size ∈ [1, N]` for any N."

Named tests per plan Task 13:
- test_extraction_pool_size_independence
- test_extraction_pool_size_independence_more_workers_than_tiles
- test_workers_receive_shared_densified_polygon_and_sea_polygons

Plus 1 structural test verifying pool_size=1 short-circuits without
constructing a multiprocessing.Pool (avoids pool-construction overhead on
the default sequential path).
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
import yaml

from cfm.data.sub_c.determinism import compute_sha256_excluding
from cfm.data.sub_c.manifest import RegionManifest
from cfm.data.sub_c.pipeline import extract_region

# Reuse the Task 12 synthetic Region builder + constants. The synthetic
# region intentionally spans multiple SVY21 tiles, which is required to
# exercise the pool/dynamic-queue codepaths (a 1-tile region would trivially
# pass any pool_size test).
from tests.data.sub_c.test_pipeline import (  # type: ignore[import-not-found]
    _POLICY_YAML,
    _VOCAB_YAML,
    _make_default_region,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_with_pool_size(output_dir: Path, pool_size: int) -> RegionManifest:
    """Run extract_region with the default synthetic region + fixed
    timestamps + the given pool_size. Fixed timestamps make the manifest
    `*_utc` fields identical across runs; the rest of the byte-stream is
    determinism-tested below.
    """
    region = _make_default_region()
    return extract_region(
        region,
        output_dir,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha="b86c509" + "0" * 33,
        extracted_utc="2026-05-18T00:00:00Z",
        started_utc="2026-05-18T00:00:00Z",
        pool_size=pool_size,
    )


def _assert_per_tile_parquet_bytes_identical(
    manifest_a: RegionManifest,
    out_a: Path,
    manifest_b: RegionManifest,
    out_b: Path,
) -> None:
    """Per-tile cells/features/crossings parquet bytes MUST be identical
    across the two runs. The pool size affects wall-clock only.
    """
    keys_a = sorted((t["tile_i"], t["tile_j"]) for t in manifest_a.tiles)
    keys_b = sorted((t["tile_i"], t["tile_j"]) for t in manifest_b.tiles)
    assert keys_a == keys_b, f"tile inventories differ: {keys_a} vs {keys_b}"

    for ti, tj in keys_a:
        rel = f"tile=EPSG3414_i{ti}_j{tj}"
        for parquet in ("cells.parquet", "features.parquet", "crossings.parquet"):
            bytes_a = (out_a / rel / parquet).read_bytes()
            bytes_b = (out_b / rel / parquet).read_bytes()
            assert bytes_a == bytes_b, (
                f"parquet bytes differ under pool-size variation: {rel}/{parquet}"
            )


def _assert_manifest_and_provenance_content_shas_identical(
    manifest_a: RegionManifest,
    out_a: Path,
    manifest_b: RegionManifest,
    out_b: Path,
) -> None:
    """Manifest content shas (via compute_sha256_excluding, which already
    strips timestamps + *_sha256 fields) MUST be identical across runs.
    Same for every per-tile provenance.yaml.
    """
    manifest_a_dict = yaml.safe_load((out_a / "manifest.yaml").read_text())
    manifest_b_dict = yaml.safe_load((out_b / "manifest.yaml").read_text())
    sha_a = compute_sha256_excluding(manifest_a_dict, "manifest.yaml")
    sha_b = compute_sha256_excluding(manifest_b_dict, "manifest.yaml")
    assert sha_a == sha_b, "manifest content sha differs under pool-size variation"

    for tile in manifest_a.tiles:
        rel = f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        prov_a = yaml.safe_load((out_a / rel / "provenance.yaml").read_text())
        prov_b = yaml.safe_load((out_b / rel / "provenance.yaml").read_text())
        sha_p_a = compute_sha256_excluding(prov_a, "provenance.yaml")
        sha_p_b = compute_sha256_excluding(prov_b, "provenance.yaml")
        assert sha_p_a == sha_p_b, (
            f"provenance content sha differs under pool-size variation for {rel}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extraction_pool_size_independence(tmp_path: Path) -> None:
    """Spec §14.5: byte output invariant under pool_size variation.

    Run with pool_size=1 (sequential) and pool_size=2 (parallel). Per-tile
    parquet bytes + manifest content sha + per-tile provenance content shas
    MUST all be identical.
    """
    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"

    manifest_seq = _extract_with_pool_size(out_seq, pool_size=1)
    manifest_par = _extract_with_pool_size(out_par, pool_size=2)

    # Sanity: the synthetic region produces >= 2 tiles, otherwise this test
    # is no stronger than the determinism test in test_pipeline.py.
    assert len(manifest_seq.tiles) >= 2, (
        f"need >= 2 tiles to exercise the pool path; got {len(manifest_seq.tiles)}"
    )

    _assert_per_tile_parquet_bytes_identical(manifest_seq, out_seq, manifest_par, out_par)
    _assert_manifest_and_provenance_content_shas_identical(
        manifest_seq, out_seq, manifest_par, out_par
    )


def test_extraction_pool_size_independence_more_workers_than_tiles(tmp_path: Path) -> None:
    """Catches empty-queue worker shutdown bugs: pool_size > tile_count.

    With pool_size=8 and (typically) 4-6 tiles in the synthetic region, the
    extra workers receive nothing from the imap_unordered queue and must
    shut down cleanly. A hang here = process-pool bug; a byte-diff = ordering
    bug (the main process's manifest sort by (i,j) is supposed to absorb
    completion-order non-determinism).
    """
    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"

    manifest_seq = _extract_with_pool_size(out_seq, pool_size=1)
    manifest_par = _extract_with_pool_size(out_par, pool_size=8)

    # Sanity precondition: actually have N > tile_count workers.
    assert 8 > len(manifest_seq.tiles), (
        f"need pool_size > tile_count; got pool_size=8 vs tile_count={len(manifest_seq.tiles)}"
    )

    _assert_per_tile_parquet_bytes_identical(manifest_seq, out_seq, manifest_par, out_par)
    _assert_manifest_and_provenance_content_shas_identical(
        manifest_seq, out_seq, manifest_par, out_par
    )


def test_workers_receive_shared_densified_polygon_and_sea_polygons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §14.5 (workers MUST NOT re-densify or re-derive sea polygons):

    Both `densify_polygon` and `derive_sea_polygons` are once-per-region in
    the main process. Workers receive the SVY21-projected, already-densified
    polygon (WKB) and the SVY21-projected sea polygon union (WKB).

    We assert this by patching the pipeline module's `densify_polygon` and
    `derive_sea_polygons` to count their call count in the main process and
    by patching the worker entry point to raise if either is invoked there.

    Implementation strategy: patch `densify_polygon` and `derive_sea_polygons`
    in the pipeline module. They are called by the main process exactly once
    each (per the spec). Workers call neither — the worker delegates straight
    into `_extract_tile`, which never references either.
    """
    from cfm.data.sub_c import pipeline as pipeline_mod

    real_densify = pipeline_mod.densify_polygon
    real_derive_sea = pipeline_mod.derive_sea_polygons

    densify_calls: list[int] = []
    derive_sea_calls: list[int] = []

    def densify_spy(*args, **kwargs):
        densify_calls.append(1)
        return real_densify(*args, **kwargs)

    def derive_sea_spy(*args, **kwargs):
        derive_sea_calls.append(1)
        return real_derive_sea(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "densify_polygon", densify_spy)
    monkeypatch.setattr(pipeline_mod, "derive_sea_polygons", derive_sea_spy)

    _extract_with_pool_size(tmp_path / "region", pool_size=2)

    # Even with pool_size=2 (multiple workers), the main process invokes
    # densify_polygon exactly once and derive_sea_polygons exactly once.
    # If a worker re-derived/re-densified, this count would only be
    # 1-from-main (workers run in subprocesses, where the spy doesn't exist,
    # so worker invocations wouldn't increment the parent's list). So the
    # invariant we *can* assert from the parent is: parent saw exactly 1
    # call to each.
    assert len(densify_calls) == 1, (
        f"densify_polygon must be called exactly once (in main process); got {len(densify_calls)}"
    )
    assert len(derive_sea_calls) == 1, (
        f"derive_sea_polygons must be called exactly once (in main process); "
        f"got {len(derive_sea_calls)}"
    )

    # Also: the worker function MUST NOT *call* densify_polygon or
    # derive_sea_polygons. Inspect the AST so docstring mentions (which
    # this skill keeps verbatim per spec §14.5) don't trigger the check.
    import ast
    import inspect
    import textwrap

    worker_src = textwrap.dedent(inspect.getsource(pipeline_mod._extract_one_tile))
    tree = ast.parse(worker_src)
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_names.add(func.id)
            elif isinstance(func, ast.Attribute):
                called_names.add(func.attr)
    assert "densify_polygon" not in called_names, (
        "_extract_one_tile must not call densify_polygon — workers receive the "
        "densified admin polygon by WKB (spec §14.5)"
    )
    assert "derive_sea_polygons" not in called_names, (
        "_extract_one_tile must not call derive_sea_polygons — workers receive "
        "sea_polygons by WKB (spec §14.5)"
    )


def test_pool_size_1_falls_back_to_sequential_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pool_size=1` MUST short-circuit to the sequential path; no
    `multiprocessing.Pool` is constructed. Saves the (modest) fork/spawn +
    pickle round-trip overhead on small extractions.
    """
    from cfm.data.sub_c import pipeline as pipeline_mod

    pool_constructions: list[int] = []
    real_pool = multiprocessing.Pool

    def pool_spy(*args, **kwargs):
        pool_constructions.append(1)
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod.mp, "Pool", pool_spy)

    _extract_with_pool_size(tmp_path / "region", pool_size=1)

    assert len(pool_constructions) == 0, (
        f"pool_size=1 must NOT construct multiprocessing.Pool; got "
        f"{len(pool_constructions)} construction(s)"
    )
