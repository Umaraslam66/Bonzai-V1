"""Sub-E per-region cross-tile validator (5 invariants per spec §10.2).

Invariants 1 and 2 (version consistency) route through ``compare_version``
per sub-D's mandate at ``src/cfm/data/sub_d/versions.py:6-7`` — bare ``!=``
would silently allow cross-namespace string equality and lose sub-D
known_issue #8's lesson. ``_check_version`` wraps the call so the
sub-E-specific error message format is preserved across the existing
test regex pattern.

Invariant #5 (external-edge consistency) is rotation-aware per spec §10.2 #5:
enumerates the 8x8 grid via ``cell_to_edge_ids`` from Task 3, collects the
external ``(lower_cell_i, lower_cell_j, axis)`` set, and asserts it equals
the parquet's external-tuple set. Catches rotation/parquet skew that the
old uniqueness-only check would silently allow.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from cfm.data.sub_d.errors import VersionMismatchError
from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version
from cfm.data.sub_e.provenance import provenance_sha256
from cfm.data.sub_e.rotation import GRID_SIZE, EdgeKind, cell_to_edge_ids
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_NAMESPACE,
    BOUNDARY_VOCAB_NAMESPACE,
    SUB_E_SCHEMA_NAMESPACE,
)
from cfm.data.sub_e.writer import SlotKind


class CrossTileValidationError(ValueError):
    """Raised when a sub-E region fails any cross-tile invariant."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_version(
    namespace: VersionNamespace,
    field_name: str,
    expected: str,
    actual: str,
    tile_coords: tuple[int, int],
) -> None:
    """Wrap ``compare_version`` for cross-tile version invariants.

    ``compare_version`` is the single sanctioned equality path per sub-D's
    versions.py docstring; bare ``!=`` would silently allow cross-namespace
    string equality (sub-D known_issue #8). The wrapper catches
    ``VersionMismatchError`` and re-raises as ``CrossTileValidationError``
    with the sub-E-specific message so the existing test regex (e.g.
    ``match="sub_e_schema_version"``) still matches.

    Pattern mirrors sub-D ``validator.py:85-97``.
    """
    try:
        compare_version(
            namespace,
            VersionRef(namespace, expected),
            VersionRef(namespace, actual),
        )
    except VersionMismatchError as exc:
        raise CrossTileValidationError(
            f"{field_name} mismatch at tile {tile_coords}: manifest={expected}, provenance={actual}"
        ) from exc


def _rotation_external_tuples() -> set[tuple[int, int, int]]:
    """Enumerate every external edge in the 8x8 grid via the rotation.

    Returns the set of ``(lower_cell_i, lower_cell_j, axis)`` identities
    for external edges. Each tuple should appear in exactly one cell's
    per-cell view by construction (spec §6.2 "vacuous-invariant"); the
    caller asserts that and uses the set as the canonical truth against
    the parquet's external-row set.
    """
    seen_count: dict[tuple[int, int, int], int] = {}
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
                    seen_count[key] = seen_count.get(key, 0) + 1
    # Each external must be owned by exactly one cell (spec §6.2).
    for key, count in seen_count.items():
        if count != 1:
            raise CrossTileValidationError(
                f"rotation external edge {key} appears in {count} cells' "
                f"per-cell views (expected 1) — rotation function bug"
            )
    return set(seen_count.keys())


def validate_extraction_cross_tile(region_dir: Path) -> None:
    manifest_path = region_dir / "manifest.yaml"
    manifest = _load_yaml(manifest_path)

    expected_sub_e_schema = manifest["sub_e_schema_version"]
    expected_vocab = manifest["versions"]["boundary_vocab_version"]
    expected_derivation = manifest["versions"]["boundary_derivation_version"]

    # Gather per-tile provenance + parquet pairs.
    first_input_digests: dict[str, str] | None = None

    # Rotation's external set is invariant across tiles (same 8x8 grid);
    # compute once before the per-tile loop.
    rotation_external = _rotation_external_tuples()

    for tile in manifest["tiles"]:
        tile_dir = region_dir / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        prov_path = tile_dir / "provenance.yaml"
        parquet_path = tile_dir / "boundary_contract.parquet"
        tile_coords = (tile["tile_i"], tile["tile_j"])

        prov = _load_yaml(prov_path)

        # Invariant 1: schema version consistency (DATA_SHAPE namespace).
        _check_version(
            SUB_E_SCHEMA_NAMESPACE,
            "sub_e_schema_version",
            expected_sub_e_schema,
            prov["versions"]["sub_e_schema_version"],
            tile_coords,
        )

        # Invariant 2: vocab + derivation version consistency.
        _check_version(
            BOUNDARY_VOCAB_NAMESPACE,
            "boundary_vocab_version",
            expected_vocab,
            prov["versions"]["boundary_vocab_version"],
            tile_coords,
        )
        _check_version(
            BOUNDARY_DERIVATION_NAMESPACE,
            "boundary_derivation_version",
            expected_derivation,
            prov["versions"]["boundary_derivation_version"],
            tile_coords,
        )

        # Invariant 3: digest chain.
        # The manifest→provenance anchor uses provenance_sha256() — the
        # exclusion-aware self-sha (strips extracted_utc + *_sha256 per
        # SUB_E_EXCLUDED_FROM_SHA, spec §9.2). Raw file-bytes hash would
        # bake extracted_utc into the chain and break determinism on every
        # rerun under live clocks (Task 8 plan-fixup landed this discipline).
        expected_prov_sha = tile["provenance_sha256"]
        actual_prov_sha = provenance_sha256(prov)
        if expected_prov_sha != actual_prov_sha:
            raise CrossTileValidationError(
                f"digest chain broken at tile ({tile['tile_i']}, {tile['tile_j']}): "
                f"manifest→provenance sha mismatch"
            )
        # The provenance→parquet anchor is a raw file-bytes hash: parquet
        # is byte-deterministic by construction (no timestamps, fixed
        # pyarrow schema/sort), so no exclusion needed.
        expected_parquet_sha = prov["outputs"]["boundary_contract_parquet_sha256"]
        actual_parquet_sha = _file_sha256(parquet_path)
        if expected_parquet_sha != actual_parquet_sha:
            raise CrossTileValidationError(
                f"digest chain broken at tile ({tile['tile_i']}, {tile['tile_j']}): "
                f"provenance→parquet sha mismatch"
            )

        # Invariant 4: input digest consistency across tiles.
        input_digests = dict(prov["inputs"])
        # Allow per-tile parquets to differ at the tile level
        # (sub_c_features/crossings can differ per tile is OK in principle but
        # for Singapore single-region the upstream manifests must match).
        anchor_keys = (
            "release",
            "sub_c_manifest_sha256",
            "sub_d_manifest_sha256",
            "boundary_vocab_sha256",
            "derivation_config_sha256",
        )
        if first_input_digests is None:
            first_input_digests = {k: input_digests[k] for k in anchor_keys}
        else:
            for k in anchor_keys:
                if first_input_digests[k] != input_digests[k]:
                    raise CrossTileValidationError(
                        f"input digest drift at tile ({tile['tile_i']}, "
                        f"{tile['tile_j']}): {k} differs from first tile"
                    )

        # Invariant 5: external-edge single-cell membership (spec §10.2 #5).
        # Each external (lower_cell_i, lower_cell_j, axis) identity must
        # appear in exactly one cell's per-cell view per the rotation
        # function (spec §6.2 vacuous-invariant lifted to a real-data
        # regression check). The parquet's external-tuple set must equal
        # the rotation's. Plus: row count must equal set size (no
        # duplicate external rows).
        tbl = pq.ParquetFile(parquet_path).read()
        slot_kinds = tbl.column("slot_kind").to_pylist()
        lower_is = tbl.column("lower_cell_i").to_pylist()
        lower_js = tbl.column("lower_cell_j").to_pylist()
        axes = tbl.column("axis").to_pylist()

        parquet_external_tuples = [
            (li, lj, ax)
            for sk, li, lj, ax in zip(slot_kinds, lower_is, lower_js, axes, strict=True)
            if sk == int(SlotKind.EXTERNAL_EDGE)
        ]
        parquet_external_set = set(parquet_external_tuples)

        if len(parquet_external_tuples) != len(parquet_external_set):
            raise CrossTileValidationError(
                f"duplicate external (lower_cell_i, lower_cell_j, axis) "
                f"at tile {tile_coords}: "
                f"{len(parquet_external_tuples)} rows, "
                f"{len(parquet_external_set)} unique tuples"
            )

        if parquet_external_set != rotation_external:
            only_parquet = sorted(parquet_external_set - rotation_external)
            only_rotation = sorted(rotation_external - parquet_external_set)
            raise CrossTileValidationError(
                f"external slot_index set mismatch at tile {tile_coords}: "
                f"only-in-parquet={only_parquet}, only-in-rotation={only_rotation}"
            )
