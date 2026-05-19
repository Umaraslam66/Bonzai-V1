"""Sub-D sidecar validators (spec §13, Task 13).

The validator is the gate that decides whether ``_SUCCESS`` may be written
for a sub-D region: Task 14's pipeline writes the per-tile and region
artifacts, then calls ``validate_region``; if it returns without raising,
the pipeline writes ``_SUCCESS``. External callers (CLI, re-validation
scripts) can call ``validate_region`` against an already-finalised region
too — the validator does not require ``_SUCCESS`` on the sub-D side because
it IS the gate that authorises that marker.

What it checks:

- ``macro_core.parquet``: exactly 64+112+32 = 208 rows; ``slot_kind`` ∈
  ``{CELL, INTERNAL_EDGE, EXTERNAL_EDGE}``; masked slots have all three
  target columns ``None`` (spec §11.2). It does NOT range-check derived
  bucket boundaries (e.g. cell_density ratio > 1.0); the locked vocab's
  top-bucket ``[0.35, ∞)`` absorbs overflow, and asserting ratio <= 1.0
  here would weaken the vocab contract — known_issue #9.
- ``derivation_evidence.parquet``: slot_kind within {CELL, INTERNAL_EDGE,
  EXTERNAL_EDGE, TILE}; slot_index inside the lattice cardinality for
  its slot_kind (TILE → 0 only).
- ``effective_conditioning.yaml``: schema version matches via
  ``compare_version``; ``sub_c_inputs`` digests match the live sub-C
  tile bytes.
- ``provenance.yaml``: schema version matches via ``compare_version``;
  ``inputs.sub_c_*_sha256`` match the live sub-C tile bytes;
  ``outputs.*_sha256`` match the live sub-D tile output bytes (chain of
  custody — drift in either direction is caught).
- region ``manifest.yaml``: schema version matches via ``compare_version``;
  ``config`` equals ``sub_c_manifest["config"]`` byte-for-byte (B6,
  defense in depth — build_manifest enforced at write time too);
  ``inputs.sub_c_manifest_sha256`` matches the live sub-C manifest bytes;
  each tile's ``provenance_sha256`` equals ``provenance_sha256()``
  recomputed from the on-disk per-tile provenance.yaml.

Every version comparison goes through ``compare_version`` from
``cfm.data.sub_d.versions`` (tension flag B3 — never raw ``==``/``!=`` on
``_version`` fields; the AST meta-test pins this).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.sub_d.conditioning import EFFECTIVE_CONDITIONING_SCHEMA_VERSION
from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.io import (
    read_derivation_evidence_parquet,
    read_macro_core_parquet,
)
from cfm.data.sub_d.lattice import (
    CELL_SLOT_COUNT,
    EXTERNAL_EDGE_SLOT_COUNT,
    INTERNAL_EDGE_SLOT_COUNT,
)
from cfm.data.sub_d.manifest import MANIFEST_SCHEMA_VERSION, read_manifest
from cfm.data.sub_d.provenance import PROVENANCE_SCHEMA_VERSION, provenance_sha256
from cfm.data.sub_d.sub_c_reader import (
    SubCTileInputs,
    iter_sub_c_tile_paths,
    read_sub_c_manifest,
    read_sub_c_tile_inputs,
)
from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version

_EXPECTED_MACRO_CORE_ROW_COUNT: int = (
    CELL_SLOT_COUNT + INTERNAL_EDGE_SLOT_COUNT + EXTERNAL_EDGE_SLOT_COUNT
)

_SLOT_INDEX_LIMIT_BY_KIND: dict[SlotKind, int] = {
    SlotKind.CELL: CELL_SLOT_COUNT,
    SlotKind.INTERNAL_EDGE: INTERNAL_EDGE_SLOT_COUNT,
    SlotKind.EXTERNAL_EDGE: EXTERNAL_EDGE_SLOT_COUNT,
    SlotKind.TILE: 1,
}


def _check_artifact_format_version(
    expected_value: str, actual_value: str, artifact_label: str
) -> None:
    """Compare two artifact-format version strings via ``compare_version``.

    Wraps ``compare_version`` so the call site reads as a single check, but
    keeps the namespace check (``ARTIFACT_FORMAT``) inside the wrapper.
    Raises ``VersionMismatchError`` on value mismatch — that subclass
    (under ``SubDValidationError``) is what the behavioural half of B3's
    test pattern matches on.
    """
    compare_version(
        VersionNamespace.ARTIFACT_FORMAT,
        VersionRef(VersionNamespace.ARTIFACT_FORMAT, expected_value),
        VersionRef(VersionNamespace.ARTIFACT_FORMAT, actual_value),
    )


def validate_tile(
    tile_dir: Path,
    sub_c_inputs: SubCTileInputs,
    macro_vocab: dict,
) -> None:
    """Validate one sub-D tile's per-tile artifacts against its sub-C inputs.

    Parameters:
        tile_dir: path to ``<sub_d_region_dir>/tile=<EPSG_LABEL>_i<i>_j<j>``.
        sub_c_inputs: the matching sub-C tile inputs (parsed parquets +
            yaml + bytes-sha digests). Caller is responsible for picking
            the right tile.
        macro_vocab: locked macro vocab dict (from ``load_macro_vocab``).
            Accepted but not yet exercised — Task 13's named tests do not
            require token-range checks against the vocab. Future expansion
            (Task 14/15) can range-check macro_core target columns against
            ``macro_vocab["locked_buckets"]``.

    Raises ``SubDValidationError`` (or ``VersionMismatchError`` from
    ``compare_version``) on any contract violation.
    """
    _ = macro_vocab  # Reserved for token-range checks; intentionally unused for Task 13.

    macro_core_path = tile_dir / "macro_core.parquet"
    derivation_evidence_path = tile_dir / "derivation_evidence.parquet"
    effective_conditioning_path = tile_dir / "effective_conditioning.yaml"
    provenance_path = tile_dir / "provenance.yaml"

    _validate_macro_core(macro_core_path)
    _validate_derivation_evidence(derivation_evidence_path)
    _validate_effective_conditioning(effective_conditioning_path, sub_c_inputs)
    _validate_provenance(
        provenance_path,
        sub_c_inputs,
        output_paths={
            "macro_core_parquet_sha256": macro_core_path,
            "derivation_evidence_parquet_sha256": derivation_evidence_path,
            "effective_conditioning_yaml_sha256": effective_conditioning_path,
        },
    )


def validate_region(region_dir: Path, sub_c_region_dir: Path) -> None:
    """Validate a sub-D region against its sub-C source region.

    Reads the sub-D manifest, cross-checks region-level invariants
    (config copy via B6 defense-in-depth, sub-C manifest sha anchor,
    artifact-format version), and recurses into each per-tile validation.

    Raises ``SubDValidationError`` (or ``VersionMismatchError``) on any
    contract violation.
    """
    sub_d_manifest = read_manifest(region_dir / "manifest.yaml")

    _check_artifact_format_version(
        MANIFEST_SCHEMA_VERSION,
        str(sub_d_manifest["manifest_schema_version"]),
        "manifest.yaml",
    )

    sub_c_manifest = read_sub_c_manifest(sub_c_region_dir)

    if sub_d_manifest["config"] != sub_c_manifest["config"]:
        raise SubDValidationError(
            "sub-D manifest.config drifted from sub_c_manifest.config "
            "(tension flag B6 — copy-verbatim contract violated). "
            f"sub-C keys: {sorted(sub_c_manifest['config'].keys())}; "
            f"sub-D keys: {sorted(sub_d_manifest['config'].keys())}"
        )

    sub_c_manifest_bytes_sha = compute_sha256((sub_c_region_dir / "manifest.yaml").read_bytes())
    recorded_sub_c_manifest_sha = sub_d_manifest["inputs"]["sub_c_manifest_sha256"]
    if recorded_sub_c_manifest_sha != sub_c_manifest_bytes_sha:
        raise SubDValidationError(
            "sub-D manifest inputs.sub_c_manifest_sha256 does not match the "
            "live sub-C manifest.yaml bytes (sub-C re-extracted since sub-D "
            f"was derived?). recorded={recorded_sub_c_manifest_sha}, "
            f"live={sub_c_manifest_bytes_sha}"
        )

    sub_c_tile_paths_by_key = {
        (p.tile_i, p.tile_j): p for p in iter_sub_c_tile_paths(sub_c_region_dir)
    }

    epsg_label = str(sub_d_manifest["region_crs"]).replace(":", "")

    for tile_entry in sub_d_manifest["tiles"]:
        tile_i = int(tile_entry["tile_i"])
        tile_j = int(tile_entry["tile_j"])
        tile_dir = region_dir / f"tile={epsg_label}_i{tile_i}_j{tile_j}"

        # Chain-of-custody: the manifest's recorded sha must equal
        # provenance_sha256 recomputed from the on-disk provenance.yaml.
        prov_data = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
        recomputed_sha = provenance_sha256(prov_data)
        recorded_sha = tile_entry["provenance_sha256"]
        if recomputed_sha != recorded_sha:
            raise SubDValidationError(
                f"tile ({tile_i},{tile_j}) provenance_sha256 chain-of-custody "
                f"broken: manifest recorded={recorded_sha}, "
                f"recomputed from on-disk provenance.yaml={recomputed_sha}"
            )

        sub_c_paths = sub_c_tile_paths_by_key.get((tile_i, tile_j))
        if sub_c_paths is None:
            raise SubDValidationError(
                f"tile ({tile_i},{tile_j}) is in sub-D manifest but not in "
                "sub-C manifest's tiles[] inventory"
            )
        sub_c_inputs = read_sub_c_tile_inputs(sub_c_paths)

        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


# ---------------------------------------------------------------------------
# Internal per-artifact validators.
# ---------------------------------------------------------------------------


def _validate_macro_core(path: Path) -> None:
    """Check cardinality + slot_kind domain + masked-slot target-class rule.

    Does NOT range-check derived target values against vocab buckets
    (cell_density ratios may exceed 1.0; the locked vocab's [0.35, ∞)
    top bucket absorbs the overflow — known_issue #9). Range-checking
    here would weaken the vocab contract.
    """
    rows = read_macro_core_parquet(path)
    if len(rows) != _EXPECTED_MACRO_CORE_ROW_COUNT:
        raise SubDValidationError(
            f"macro_core.parquet at {path} has {len(rows)} rows; expected "
            f"exactly {_EXPECTED_MACRO_CORE_ROW_COUNT} (64 cells + 112 "
            "internal edges + 32 external edges)"
        )

    tile_rows = [r for r in rows if r.slot_kind == SlotKind.TILE]
    if tile_rows:
        raise SubDValidationError(
            f"macro_core.parquet at {path} has {len(tile_rows)} SlotKind.TILE "
            "row(s); TILE-scoped metrics belong in derivation_evidence.parquet only "
            "(spec §11.2)"
        )

    for row in rows:
        if not row.scope.name == "FULLY_MASKED":
            continue
        bad_targets = []
        if row.zoning_class is not None:
            bad_targets.append(("zoning_class", row.zoning_class))
        if row.cell_density_bucket is not None:
            bad_targets.append(("cell_density_bucket", row.cell_density_bucket))
        if row.road_skeleton_class is not None:
            bad_targets.append(("road_skeleton_class", row.road_skeleton_class))
        if bad_targets:
            raise SubDValidationError(
                f"macro_core.parquet at {path}: row "
                f"slot_kind={row.slot_kind.name} slot_index={row.slot_index} "
                f"has scope=FULLY_MASKED but target(s) populated: {bad_targets}. "
                "Masked slots are out-of-scope by definition; target values "
                "there are undefined (spec §11.2)."
            )


def _validate_derivation_evidence(path: Path) -> None:
    """Range-check slot_kind and slot_index against lattice cardinality."""
    rows = read_derivation_evidence_parquet(path)
    for row in rows:
        limit = _SLOT_INDEX_LIMIT_BY_KIND.get(row.slot_kind)
        if limit is None:
            raise SubDValidationError(
                f"derivation_evidence.parquet at {path}: row has unknown "
                f"slot_kind={int(row.slot_kind)}"
            )
        if not (0 <= row.slot_index < limit):
            raise SubDValidationError(
                f"derivation_evidence.parquet at {path}: row "
                f"slot_kind={row.slot_kind.name} slot_index={row.slot_index} "
                f"is outside the lattice cardinality (limit={limit})"
            )


def _validate_effective_conditioning(path: Path, sub_c_inputs: SubCTileInputs) -> None:
    """Version check + sub-C input digest cross-check."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_artifact_format_version(
        EFFECTIVE_CONDITIONING_SCHEMA_VERSION,
        str(data["effective_conditioning_schema_version"]),
        "effective_conditioning.yaml",
    )

    recorded = data.get("sub_c_inputs") or {}
    expected = {
        "tile_meta_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
        "tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
    }
    for key, expected_value in expected.items():
        actual_value = recorded.get(key)
        if actual_value != expected_value:
            raise SubDValidationError(
                f"effective_conditioning.yaml at {path}: sub_c_inputs.{key} "
                f"mismatch — recorded={actual_value}, live sub-C bytes-sha="
                f"{expected_value}. Was sub-C re-extracted since sub-D was "
                "derived?"
            )


def _validate_provenance(
    path: Path,
    sub_c_inputs: SubCTileInputs,
    output_paths: dict[str, Path],
) -> None:
    """Version check + sub-C input digest cross-check + sub-D output digest
    cross-check.

    ``output_paths`` maps each ``outputs.*_sha256`` key in the provenance YAML
    to its corresponding on-disk file. The validator recomputes the live
    bytes-sha and compares — drift in either direction (someone edited the
    output file after provenance, or someone hand-edited the provenance) is
    caught.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_artifact_format_version(
        PROVENANCE_SCHEMA_VERSION,
        str(data["provenance_schema_version"]),
        "provenance.yaml",
    )

    recorded_inputs = data.get("inputs") or {}
    expected_inputs = {
        "sub_c_cells_parquet_sha256": sub_c_inputs.digests["cells_parquet_sha256"],
        "sub_c_features_parquet_sha256": sub_c_inputs.digests["features_parquet_sha256"],
        "sub_c_crossings_parquet_sha256": sub_c_inputs.digests["crossings_parquet_sha256"],
        "sub_c_meta_yaml_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
        "sub_c_tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
    }
    for key, expected_value in expected_inputs.items():
        actual_value = recorded_inputs.get(key)
        if actual_value != expected_value:
            raise SubDValidationError(
                f"provenance.yaml at {path}: inputs.{key} mismatch — "
                f"recorded={actual_value}, live sub-C bytes-sha={expected_value}. "
                "Was sub-C re-extracted since sub-D was derived?"
            )

    recorded_outputs = data.get("outputs") or {}
    for key, output_path in output_paths.items():
        live_sha = compute_sha256(output_path.read_bytes())
        actual_value = recorded_outputs.get(key)
        if actual_value != live_sha:
            raise SubDValidationError(
                f"provenance.yaml at {path}: outputs.{key} mismatch — "
                f"recorded={actual_value}, live bytes-sha={live_sha}. "
                f"The on-disk file {output_path} has been modified since "
                "provenance was written, or provenance was hand-edited."
            )
