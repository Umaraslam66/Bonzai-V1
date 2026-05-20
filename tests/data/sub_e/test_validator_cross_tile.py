from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from cfm.data.sub_e.validator_cross_tile import (
    CrossTileValidationError,
    validate_extraction_cross_tile,
)


def _build_synthetic_region(
    tmp_path: Path,
    *,
    n_tiles: int = 3,
    boundary_vocab_version: str = "1.0",
    boundary_derivation_version: str = "1.0",
    sub_e_schema_version: str = "1.0",
    corrupt_external_at_tile_k: int | None = None,
    duplicate_external_at_tile_k: int | None = None,
) -> Path:
    """Create a minimal sub-E region directory with N consistent tiles.

    The external boundary rows are constructed via ``cell_to_edge_ids``
    so the synthetic data matches what the real per-cell→per-edge path
    produces (spec §10.2 #5). Index-based external rows would
    synthetic-pass the rotation-aware validator while failing on real
    Singapore data — caught by the implementer pre-Task-9-dispatch.

    Two controlled-violation modes for invariant #5 are exposed at
    fixture-build time (so the digest chain remains internally consistent
    with the bad state — corruption introduced after-the-fact via parquet
    mutation would fail invariant #3 first):

    - ``corrupt_external_at_tile_k``: that tile's first axis=1 external
      row has its ``lower_cell_i`` mutated to ``4``. Selection is
      *semantic* (find first axis=1 row), not positional — robust against
      future sort-order changes. Axis=1 externals have
      ``lower_cell_i ∈ {0, 7}`` by construction, so the mutated tuple
      ``(4, lj, 1)`` is guaranteed outside rotation's external set.
      Tuple uniqueness is preserved (no other row has the mutated tuple),
      so the duplicate-tuple branch does NOT fire — only the
      rotation-equality (set-mismatch) branch fires.
    - ``duplicate_external_at_tile_k``: that tile's external row at
      index 1 has its ``(lower_cell_i, lower_cell_j, axis)`` triple
      replaced by row 0's triple. Slot_index stays distinct so the
      OLD weak validator (slot_index uniqueness) would pass; the
      duplicate-tuple branch of the strengthened invariant #5 fires.
    """
    from dataclasses import replace as _dc_replace

    from cfm.data.sub_e.derivation import BoundaryClass
    from cfm.data.sub_e.manifest import (
        SubEManifest,
        SubEManifestConfig,
        SubEManifestExtraction,
        SubEManifestInputs,
        SubEManifestTile,
        SubEManifestVersions,
        write_manifest,
    )
    from cfm.data.sub_e.provenance import (
        SubEInputDigests,
        SubEProvenance,
        SubEVersions,
        provenance_sha256,
        provenance_to_dict,
        write_provenance,
    )
    from cfm.data.sub_e.rotation import GRID_SIZE, EdgeKind, cell_to_edge_ids
    from cfm.data.sub_e.writer import (
        BoundaryContractRow,
        SlotKind,
        write_boundary_contract,
    )

    region = tmp_path / "sub_e_singapore"
    region.mkdir()

    def _external_rows_via_rotation(
        *,
        corrupt_first_axis_1: bool = False,
        duplicate_idx: int | None = None,
    ) -> list[BoundaryContractRow]:
        """Enumerate the 8x8 grid through rotation, collect the unique
        external (lower_cell_i, lower_cell_j, axis) tuples, sort them
        canonically, and emit one parquet row per tuple. Should produce
        exactly 32 rows (24 edge cells x 1 external + 4 corners x 2).

        Selection of the corruption target is *semantic* (find first
        axis=1 row), not positional (corrupt_idx=N). Robust against
        future sort-order changes. The tripwire assert on ``target.axis``
        catches any selection-logic regression that would silently
        re-introduce the duplicate-tuple trap.
        """
        seen: set[tuple[int, int, int]] = set()
        tuples: list[tuple[int, int, int]] = []
        for ci in range(GRID_SIZE):
            for cj in range(GRID_SIZE):
                cell_edges = cell_to_edge_ids(ci, cj)
                for edge in (
                    cell_edges.north,
                    cell_edges.south,
                    cell_edges.west,
                    cell_edges.east,
                ):
                    li, lj, axis, kind = edge
                    if kind is EdgeKind.EXTERNAL:
                        key = (li, lj, axis)
                        if key not in seen:
                            seen.add(key)
                            tuples.append(key)
        tuples.sort()
        assert len(tuples) == 32, (
            f"rotation should produce exactly 32 unique external edges, got {len(tuples)}"
        )
        ext_rows = [
            BoundaryContractRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=li,
                lower_cell_j=lj,
                axis=axis,
                scope_marker=3,
                boundary_class_enum=None,
            )
            for idx, (li, lj, axis) in enumerate(tuples)
        ]

        if corrupt_first_axis_1:
            # Semantic selection: rotation's axis=1 externals have
            # lower_cell_i ∈ {0, 7} by construction (west/east sides).
            # Setting lower_cell_i=4 produces a tuple genuinely outside
            # rotation's external set → set-mismatch branch fires by
            # construction. Positional selection (corrupt_idx=N) was
            # fragile against sort-order changes — a prior version with
            # corrupt_idx=0 hit the axis=0 first tuple (0,0,0), whose
            # mutation (4,0,0) was already in rotation, triggering
            # duplicate-tuple instead of set-mismatch.
            target_idx = next((i for i, r in enumerate(ext_rows) if r.axis == 1), None)
            assert target_idx is not None, (
                "rotation must produce at least one axis=1 external "
                "(west/east boundary); selection logic broken"
            )
            target = ext_rows[target_idx]
            assert target.axis == 1, (
                f"selection tripwire: target.axis={target.axis}, expected 1 "
                f"— selection logic broken, would re-introduce duplicate-tuple trap"
            )
            ext_rows[target_idx] = _dc_replace(target, lower_cell_i=4)

        if duplicate_idx is not None:
            # Copy row[duplicate_idx]'s (lower_cell_i, lower_cell_j, axis)
            # triple onto row[duplicate_idx + 1]. slot_index stays
            # distinct → OLD weak validator (slot_index uniqueness)
            # passes; tuple count != set size → strengthened invariant
            # #5's duplicate-tuple branch fires.
            src = ext_rows[duplicate_idx]
            dst = ext_rows[duplicate_idx + 1]
            ext_rows[duplicate_idx + 1] = _dc_replace(
                dst,
                lower_cell_i=src.lower_cell_i,
                lower_cell_j=src.lower_cell_j,
                axis=src.axis,
            )

        return ext_rows

    def _rows(
        *,
        corrupt_first_axis_1: bool = False,
        duplicate_idx: int | None = None,
    ) -> list[BoundaryContractRow]:
        rows: list[BoundaryContractRow] = []
        for idx in range(112):
            rows.append(
                BoundaryContractRow(
                    slot_kind=SlotKind.INTERNAL_EDGE,
                    slot_index=idx,
                    lower_cell_i=idx % 8,
                    lower_cell_j=idx // 8 % 8,
                    axis=idx % 2,
                    scope_marker=0,
                    boundary_class_enum=int(BoundaryClass.NONE),
                )
            )
        rows.extend(
            _external_rows_via_rotation(
                corrupt_first_axis_1=corrupt_first_axis_1,
                duplicate_idx=duplicate_idx,
            )
        )
        return rows

    tile_records: list[SubEManifestTile] = []
    for k in range(n_tiles):
        tile_dir = region / f"tile=EPSG3414_i{k}_j0"
        tile_dir.mkdir()
        contract = write_boundary_contract(
            tile_dir / "boundary_contract.parquet",
            _rows(
                corrupt_first_axis_1=(k == corrupt_external_at_tile_k),
                duplicate_idx=(0 if k == duplicate_external_at_tile_k else None),
            ),
        )
        contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        prov_path = tile_dir / "provenance.yaml"
        prov_obj = SubEProvenance(
            tile_i=k,
            tile_j=0,
            extraction_commit_sha="a" * 40,
            extracted_utc="2026-05-21T12:00:00Z",
            rerun_count=0,
            rerun_reason="initial",
            inputs=SubEInputDigests(
                release="2026-04-15.0",
                sub_c_manifest_sha256="b" * 64,
                sub_c_features_parquet_sha256="c" * 64,
                sub_c_crossings_parquet_sha256="d" * 64,
                sub_d_manifest_sha256="e" * 64,
                sub_d_macro_core_parquet_sha256="f" * 64,
                boundary_vocab_sha256="0" * 64,
                derivation_config_sha256="1" * 64,
            ),
            versions=SubEVersions(
                sub_e_schema_version=sub_e_schema_version,
                boundary_vocab_version=boundary_vocab_version,
                boundary_derivation_version=boundary_derivation_version,
            ),
            boundary_contract_parquet_sha256=contract_sha,
        )
        write_provenance(prov_path, prov_obj)
        # Self-integrity sha: strip extracted_utc + *_sha256 per
        # SUB_E_EXCLUDED_FROM_SHA so the digest chain survives live-clock
        # reruns. NOT hashlib.sha256(prov_path.read_bytes()) — that would
        # bake extracted_utc into the chain and break determinism (spec §9.2).
        prov_sha = provenance_sha256(provenance_to_dict(prov_obj))
        tile_records.append(SubEManifestTile(tile_i=k, tile_j=0, provenance_sha256=prov_sha))

    write_manifest(
        region / "manifest.yaml",
        SubEManifest(
            manifest_schema_version="1.0",
            sub_e_schema_version=sub_e_schema_version,
            release="2026-04-15.0",
            region="singapore",
            region_crs="EPSG:3414",
            inputs=SubEManifestInputs(
                sub_c_manifest_sha256="b" * 64,
                sub_c_region_dir="data/processed/sub_c/2026-04-15.0/singapore",
                sub_d_manifest_sha256="e" * 64,
                sub_d_region_dir="data/processed/sub_d/2026-04-15.0/singapore",
                boundary_vocab_sha256="0" * 64,
            ),
            versions=SubEManifestVersions(
                boundary_vocab_version=boundary_vocab_version,
                boundary_derivation_version=boundary_derivation_version,
            ),
            config_source="sub_d_manifest.config",
            config=SubEManifestConfig(
                cell_grid=(8, 8), internal_edge_count=112, external_edge_count=32
            ),
            initial_extraction=SubEManifestExtraction(
                commit_sha="a" * 40,
                started_utc="2026-05-21T12:00:00Z",
                completed_utc="2026-05-21T12:05:00Z",
                tile_count=n_tiles,
            ),
            tiles=tile_records,
        ),
    )
    (region / "_SUCCESS").touch()
    return region


def test_valid_region_passes_all_cross_tile_invariants(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    validate_extraction_cross_tile(region)  # should not raise


def test_invariant_1_schema_version_consistency(tmp_path: Path) -> None:
    """Corrupt one tile's provenance to use a different sub_e_schema_version.

    Operates at the YAML-dict layer (load → mutate → dump) rather than
    byte-layer text-replace. canonicalize_yaml emits single-quoted
    strings; double-quote text-replace would be a silent no-op against
    yaml.safe_dump's default quoting (defect found pre-Task-9-dispatch).
    Round-trip via yaml.safe_dump is robust against yaml lib version
    drift and quote-style shifts.
    """
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    data = yaml.safe_load(prov_path.read_text())
    data["versions"]["sub_e_schema_version"] = "2.0"
    prov_path.write_text(yaml.safe_dump(data))
    with pytest.raises(CrossTileValidationError, match="sub_e_schema_version"):
        validate_extraction_cross_tile(region)


def test_invariant_2_vocab_and_derivation_consistency(tmp_path: Path) -> None:
    """Same yaml round-trip pattern as invariant #1 for the same reason."""
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i0_j0" / "provenance.yaml"
    data = yaml.safe_load(prov_path.read_text())
    data["versions"]["boundary_vocab_version"] = "2.0"
    prov_path.write_text(yaml.safe_dump(data))
    with pytest.raises(CrossTileValidationError, match="boundary_vocab_version"):
        validate_extraction_cross_tile(region)


def test_invariant_3_digest_chain_broken_at_parquet(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    # Mutate the parquet bytes so its sha no longer matches what provenance recorded.
    parquet = region / "tile=EPSG3414_i0_j0" / "boundary_contract.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"\x00")
    with pytest.raises(CrossTileValidationError, match="digest chain"):
        validate_extraction_cross_tile(region)


def test_invariant_4_input_digest_drift_across_tiles(tmp_path: Path) -> None:
    region = _build_synthetic_region(tmp_path)
    prov_path = region / "tile=EPSG3414_i1_j0" / "provenance.yaml"
    text = prov_path.read_text()
    prov_path.write_text(
        text.replace(
            "sub_d_manifest_sha256: " + ("e" * 64),
            "sub_d_manifest_sha256: " + ("9" * 64),
        )
    )
    with pytest.raises(CrossTileValidationError, match="input digest"):
        validate_extraction_cross_tile(region)


def test_invariant_5_duplicate_external_tuple(tmp_path: Path) -> None:
    """Two parquet rows share the same (lower_cell_i, lower_cell_j, axis).

    The duplicate is injected at fixture-build time via
    ``duplicate_external_at_tile_k=0`` so the digest chain remains
    internally consistent with the bad parquet (invariant #3 does not
    fire). Slot_index uniqueness is preserved — the OLD weak validator
    (slot_index uniqueness only) would PASS this corruption. Only the
    strengthened invariant #5's duplicate-tuple branch catches it.

    Earlier draft mutated parquet bytes after region build, but that
    broke invariant #3 (digest chain) which fires before invariant #5
    in the per-tile loop. The fixture-build-time injection keeps the
    chain valid so invariant #5 actually gets exercised.
    """
    region = _build_synthetic_region(tmp_path, duplicate_external_at_tile_k=0)
    with pytest.raises(CrossTileValidationError, match="duplicate external"):
        validate_extraction_cross_tile(region)


def test_invariant_5_external_set_mismatch_against_rotation(tmp_path: Path) -> None:
    """The first axis=1 external row has its lower_cell_i mutated to 4.

    This is the load-bearing controlled-violation test for the
    rotation-aware strengthening (spec §10.2 #5). The corruption:

    - Preserves 144-row count (writer-level invariants pass).
    - Preserves slot_index uniqueness (OLD weak validator passes).
    - Preserves tuple uniqueness within the parquet (duplicate-tuple
      branch does NOT fire — by construction: axis=1 mutations to
      lower_cell_i=4 produce tuples like (4, lj, 1) that do not match
      any other parquet row's tuple).
    - Shifts the parquet's external-tuple SET away from the rotation's
      set: rotation's axis=1 externals have lower_cell_i ∈ {0, 7} by
      construction, so (4, lj, 1) is genuinely outside rotation's set.

    Only the rotation-equality check at the end of invariant #5 fires.
    Selection is semantic (find first axis=1 row), not positional —
    robust against future sort-order changes in the rotation function.
    Earlier draft used corrupt_idx=0 which silently hit the axis=0
    tuple (0,0,0); mutation to (4,0,0) was already in rotation's set,
    triggering duplicate-tuple instead of set-mismatch and defeating
    the load-bearing meta-check that this test specifically exercises
    the strengthening, not legacy uniqueness behavior.
    """
    region = _build_synthetic_region(tmp_path, corrupt_external_at_tile_k=0)
    with pytest.raises(CrossTileValidationError, match="external slot_index set mismatch"):
        validate_extraction_cross_tile(region)
