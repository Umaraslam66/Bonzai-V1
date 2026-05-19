"""Tests for the sub-D sidecar validators (Task 13).

The validator is the gate that decides whether ``_SUCCESS`` may be written
for a sub-D region (Task 14's pipeline calls validate_region; if it returns
without raising, the pipeline writes ``_SUCCESS``). It enforces:

- Per-tile schema/lattice/scope invariants on ``macro_core.parquet`` and
  ``derivation_evidence.parquet``.
- The digest chain: every sha sub-D records about its inputs must match the
  live sub-C bytes; every sha sub-D records about its outputs must match
  the live sub-D bytes; every tile entry's ``provenance_sha256`` in the
  region manifest must equal ``provenance_sha256()`` recomputed from the
  on-disk provenance.yaml.
- B6: the region manifest's ``config`` block must equal
  ``sub_c_manifest["config"]`` byte-for-byte (defense in depth — build_manifest
  enforced it at write time, validate_region re-checks).
- B3: every version comparison goes through ``compare_version`` from
  ``cfm.data.sub_d.versions``, never raw ``==``/``!=`` on _version fields.
  Tested two ways:
  (a) behaviourally — a corrupted version field raises VersionMismatchError
      (subclass of SubDValidationError; proves compare_version was used).
  (b) statically — an AST meta-test scans validator.py / validator_*.py for
      ast.Compare nodes whose operands contain "version" identifiers.

The validator must NOT weaken its own invariants. Active example: the
``cell_density`` known-issue (ratio > 1.0 in real Singapore data) is
absorbed by the locked vocab's top bucket ``[0.35, ∞)`` — the validator must
NOT assert ratio <= 1.0.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml, write_parquet
from cfm.data.sub_d.conditioning import (
    build_effective_conditioning,
    write_effective_conditioning,
)
from cfm.data.sub_d.enums import Scope, SlotKind
from cfm.data.sub_d.errors import SubDValidationError, VersionMismatchError
from cfm.data.sub_d.io import (
    MacroCoreRow,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)
from cfm.data.sub_d.lattice import (
    iter_cell_slots,
    iter_external_edge_slots,
    iter_internal_edge_slots,
)
from cfm.data.sub_d.manifest import (
    SUB_D_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
    write_success_marker,
)
from cfm.data.sub_d.provenance import (
    build_tile_provenance,
    write_provenance,
)
from cfm.data.sub_d.sub_c_reader import (
    iter_sub_c_tile_paths,
    read_sub_c_manifest,
    read_sub_c_tile_inputs,
)
from cfm.data.sub_d.validator import validate_region, validate_tile

# ---------------------------------------------------------------------------
# Fixtures: build matched sub-C + sub-D region pairs.
# ---------------------------------------------------------------------------


def _sub_c_config_fixture() -> dict:
    return {
        "cell_grid": [8, 8],
        "cell_size_m": 250,
        "tile_size_m": 2000,
        "internal_edge_count": 112,
        "external_edge_count": 32,
        "sliver_drop_rule": "drop iff area < 0.01",
    }


def _build_fake_sub_c_region(
    root: Path,
    *,
    region: str = "singapore",
    region_crs: str = "EPSG:3414",
    tile_keys: list[tuple[int, int]] | None = None,
    config: dict | None = None,
) -> Path:
    """Build a minimal valid sub-C region directory tree.

    The sub-C contents don't have to be semantically meaningful — sub-D just
    reads the bytes, hashes them, and consumes the manifest's tiles[] inventory.
    """
    if tile_keys is None:
        tile_keys = [(0, 0)]
    if config is None:
        config = _sub_c_config_fixture()

    region_dir = root / region
    region_dir.mkdir(parents=True, exist_ok=True)
    epsg_label = region_crs.replace(":", "")

    tiles_inventory: list[dict] = []
    for tile_i, tile_j in tile_keys:
        tile_dir = region_dir / f"tile={epsg_label}_i{tile_i}_j{tile_j}"
        tile_dir.mkdir()
        # Sub-C tile artifacts: byte content doesn't matter to sub-D's
        # validator beyond hashing. Use small parquet tables and minimal YAML.
        write_parquet(
            pa.table({"cell_i": [tile_i], "cell_j": [tile_j]}),
            tile_dir / "cells.parquet",
        )
        write_parquet(pa.table({"feature_class": [0]}), tile_dir / "features.parquet")
        write_parquet(pa.table({"axis": [0]}), tile_dir / "crossings.parquet")
        meta_content = {
            "schema_version": "1.1",
            "tile_i": tile_i,
            "tile_j": tile_j,
            "aggregates": {"kept_cell_count": 1},
            "config": {"sliver_drop_rule": "drop iff area < 0.01"},
            "conditioning_per_tile": {
                "admin_region": "Central",
                "morphology_class": "Asian-megacity",
                "era_class": "contemporary",
                "coastal_inland_river": 1,
                "population_density_bucket": None,
                "population_density_bucket_owner": "sub-D",
            },
        }
        (tile_dir / "meta.yaml").write_text(canonicalize_yaml(meta_content), encoding="utf-8")
        provenance_content = {
            "schema_version": "1.0",
            "tile_i": tile_i,
            "tile_j": tile_j,
            "extraction": {"extracted_utc": "2026-04-15T00:00:00Z"},
            "outputs": {"cells_parquet_sha256": "deadbeef"},
        }
        (tile_dir / "provenance.yaml").write_text(
            canonicalize_yaml(provenance_content), encoding="utf-8"
        )
        tiles_inventory.append(
            {"tile_i": tile_i, "tile_j": tile_j, "provenance_sha256": "deadbeef"}
        )

    manifest = {
        "schema_version": "1.1",
        "sub_c_schema_version": "1.1",
        "release": "2026-04-15.0",
        "region": region,
        "region_crs": region_crs,
        "config": config,
        "conditioning_defaults": {
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
        "tiles": tiles_inventory,
    }
    (region_dir / "manifest.yaml").write_text(canonicalize_yaml(manifest), encoding="utf-8")
    (region_dir / "_SUCCESS").write_bytes(b"")
    return region_dir


def _build_all_masked_macro_core_rows() -> list[MacroCoreRow]:
    """Build a 208-row macro_core with every slot FULLY_MASKED.

    All-masked is a valid (if degenerate) sub-D output state — a tile that
    is entirely outside admin coverage or entirely sea would produce this.
    For validator tests, what matters is that the invariants hold under any
    valid state, including this one.
    """
    rows: list[MacroCoreRow] = []
    for cell in iter_cell_slots():
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.CELL,
                slot_index=cell.slot_index,
                cell_i=cell.cell_i,
                cell_j=cell.cell_j,
                lower_cell_i=None,
                lower_cell_j=None,
                axis=None,
                scope=Scope.FULLY_MASKED,
                zoning_class=None,
                cell_density_bucket=None,
                road_skeleton_class=None,
            )
        )
    for edge in iter_internal_edge_slots():
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=edge.slot_index,
                cell_i=None,
                cell_j=None,
                lower_cell_i=edge.lower_cell_i,
                lower_cell_j=edge.lower_cell_j,
                axis=edge.axis,
                scope=Scope.FULLY_MASKED,
                zoning_class=None,
                cell_density_bucket=None,
                road_skeleton_class=None,
            )
        )
    for edge in iter_external_edge_slots():
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=edge.slot_index,
                cell_i=None,
                cell_j=None,
                lower_cell_i=edge.lower_cell_i,
                lower_cell_j=edge.lower_cell_j,
                axis=edge.axis,
                scope=Scope.FULLY_MASKED,
                zoning_class=None,
                cell_density_bucket=None,
                road_skeleton_class=None,
            )
        )
    return rows


_HAPPY_VERSIONS_TILE = {
    "sub_d_schema_version": SUB_D_SCHEMA_VERSION,
    "macro_plan_vocab_version": "1.0",
    "zoning_vocab_version": "1.0",
    "zoning_derivation_version": "1.0",
    "cell_density_vocab_version": "1.0",
    "cell_density_derivation_version": "1.0",
    "tile_population_density_vocab_version": "1.0",
    "tile_population_density_derivation_version": "1.0",
    "road_skeleton_vocab_version": "1.0",
    "road_skeleton_derivation_version": "1.0",
}

_HAPPY_VERSIONS_MANIFEST = {
    "macro_plan_vocab_version": "1.0",
    "zoning_vocab_version": "1.0",
    "zoning_derivation_version": "1.0",
    "cell_density_vocab_version": "1.0",
    "cell_density_derivation_version": "1.0",
    "tile_population_density_vocab_version": "1.0",
    "tile_population_density_derivation_version": "1.0",
    "road_skeleton_vocab_version": "1.0",
    "road_skeleton_derivation_version": "1.0",
}


def _build_happy_path_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Build matched sub-C + sub-D region dirs that pass validate_region.

    Returns ``(sub_c_region_dir, sub_d_region_dir)`` with one tile (0, 0).
    Individual tests mutate one artifact to exercise a rejection path.
    """
    sub_c_root = tmp_path / "sub_c"
    sub_c_region_dir = _build_fake_sub_c_region(sub_c_root)

    sub_c_manifest = read_sub_c_manifest(sub_c_region_dir)
    sub_c_paths = iter_sub_c_tile_paths(sub_c_region_dir)[0]
    sub_c_inputs = read_sub_c_tile_inputs(sub_c_paths)
    sub_c_manifest_sha = compute_sha256((sub_c_region_dir / "manifest.yaml").read_bytes())

    sub_d_region_dir = tmp_path / "sub_d" / "singapore"
    epsg_label = "EPSG3414"
    sub_d_tile_dir = sub_d_region_dir / f"tile={epsg_label}_i0_j0"
    sub_d_tile_dir.mkdir(parents=True)

    # Per-tile sub-D artifacts. Order matters: outputs sha computed AFTER the
    # parquet/yaml bytes exist on disk.
    macro_core_path = sub_d_tile_dir / "macro_core.parquet"
    write_macro_core_parquet(_build_all_masked_macro_core_rows(), macro_core_path)

    de_path = sub_d_tile_dir / "derivation_evidence.parquet"
    write_derivation_evidence_parquet([], de_path)

    ec_data = build_effective_conditioning(
        meta=sub_c_inputs.meta,
        manifest=sub_c_manifest,
        population_density_bucket=0,
        versions={
            "sub_c_conditioning_schema_version": "1.1",
            "tile_population_density_vocab_version": "1.0",
            "tile_population_density_derivation_version": "1.0",
        },
        digests={
            "manifest_sha256": sub_c_manifest_sha,
            "tile_meta_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
            "tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
        },
    )
    ec_path = sub_d_tile_dir / "effective_conditioning.yaml"
    write_effective_conditioning(ec_data, ec_path)

    prov_data = build_tile_provenance(
        tile_i=0,
        tile_j=0,
        extraction={
            "commit_sha": "abc" + "0" * 37,
            "extracted_utc": "2026-05-19T12:00:00Z",
            "rerun_count": 0,
            "rerun_reason": "initial",
        },
        inputs={
            "release": "2026-04-15.0",
            "sub_c_manifest_sha256": sub_c_manifest_sha,
            "sub_c_tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
            "sub_c_cells_parquet_sha256": sub_c_inputs.digests["cells_parquet_sha256"],
            "sub_c_features_parquet_sha256": sub_c_inputs.digests["features_parquet_sha256"],
            "sub_c_crossings_parquet_sha256": sub_c_inputs.digests["crossings_parquet_sha256"],
            "sub_c_meta_yaml_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
            "macro_vocab_sha256": "0" * 64,
            "derivation_config_sha256": "0" * 64,
        },
        versions=_HAPPY_VERSIONS_TILE,
        outputs={
            "macro_core_parquet_sha256": compute_sha256(macro_core_path.read_bytes()),
            "derivation_evidence_parquet_sha256": compute_sha256(de_path.read_bytes()),
            "effective_conditioning_yaml_sha256": compute_sha256(ec_path.read_bytes()),
        },
    )
    prov_path = sub_d_tile_dir / "provenance.yaml"
    write_provenance(prov_data, prov_path)

    sub_d_manifest_data = build_manifest(
        release="2026-04-15.0",
        region="singapore",
        region_crs="EPSG:3414",
        sub_c_manifest=sub_c_manifest,
        inputs={
            "sub_c_manifest_sha256": sub_c_manifest_sha,
            "sub_c_region_dir": str(sub_c_region_dir),
        },
        versions=_HAPPY_VERSIONS_MANIFEST,
        config=sub_c_manifest["config"],
        initial_extraction={
            "commit_sha": "abc" + "0" * 37,
            "started_utc": "2026-05-19T12:00:00Z",
            "completed_utc": "2026-05-19T12:10:00Z",
            "tile_count": 1,
        },
        tile_provenances=[prov_data],
    )
    write_manifest(sub_d_manifest_data, sub_d_region_dir / "manifest.yaml")
    write_success_marker(sub_d_region_dir)

    return sub_c_region_dir, sub_d_region_dir


def _tile_dir(sub_d_region_dir: Path, tile_i: int = 0, tile_j: int = 0) -> Path:
    return sub_d_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"


def _sub_c_inputs_for(sub_c_region_dir: Path):
    sub_c_paths = iter_sub_c_tile_paths(sub_c_region_dir)[0]
    return read_sub_c_tile_inputs(sub_c_paths)


def _rewrite_yaml(path: Path, mutate) -> None:
    """Load YAML, apply ``mutate(dict)`` in place, write back canonical."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(canonicalize_yaml(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Happy-path smoke test (companion; not in the plan's 8 named tests).
# Sanity-checks that all the fixture pieces fit before each rejection test
# starts mutating one thing.
# ---------------------------------------------------------------------------


def test_validate_region_passes_on_clean_pair(tmp_path: Path):
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    validate_region(sub_d_region_dir, sub_c_region_dir)


# ---------------------------------------------------------------------------
# Plan-named tests
# ---------------------------------------------------------------------------


def test_validator_rejects_missing_macro_core_rows(tmp_path: Path):
    """macro_core.parquet must contain exactly 64+112+32 = 208 rows (one row
    per lattice slot). A short macro_core means some slots have no row, which
    means consumers cannot index the lattice deterministically.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    # Overwrite macro_core with a short version — only 5 cell rows.
    short_rows = _build_all_masked_macro_core_rows()[:5]
    write_macro_core_parquet(short_rows, tile_dir / "macro_core.parquet")

    with pytest.raises(SubDValidationError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


def test_validator_rejects_target_class_on_masked_slot(tmp_path: Path):
    """Spec §11.2 validation rule: masked slots (scope=FULLY_MASKED) must
    have ``zoning_class=cell_density_bucket=road_skeleton_class=None``. A
    target class on a masked slot is a derivation bug — the masked slot is
    out-of-scope by definition, so any target value there is undefined.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    # Re-write macro_core with one masked cell that has a non-None zoning_class.
    rows = _build_all_masked_macro_core_rows()
    bad_row = rows[0]
    rows[0] = MacroCoreRow(
        slot_kind=bad_row.slot_kind,
        slot_index=bad_row.slot_index,
        cell_i=bad_row.cell_i,
        cell_j=bad_row.cell_j,
        lower_cell_i=bad_row.lower_cell_i,
        lower_cell_j=bad_row.lower_cell_j,
        axis=bad_row.axis,
        scope=Scope.FULLY_MASKED,  # masked
        zoning_class=5,  # non-None target on a masked slot — invariant violation
        cell_density_bucket=None,
        road_skeleton_class=None,
    )
    write_macro_core_parquet(rows, tile_dir / "macro_core.parquet")

    with pytest.raises(SubDValidationError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


def test_validator_rejects_effective_conditioning_digest_mismatch(tmp_path: Path):
    """effective_conditioning.yaml records sha256 of the sub-C inputs that
    fed the conditioning copy. If those sha values do not match the live
    sub-C bytes (e.g. someone re-derived against a different sub-C release
    but kept the old digest), the validator must refuse.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    def _corrupt(data: dict) -> None:
        data["sub_c_inputs"]["tile_meta_sha256"] = "f" * 64

    _rewrite_yaml(tile_dir / "effective_conditioning.yaml", _corrupt)

    with pytest.raises(SubDValidationError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


def test_validator_rejects_sub_c_input_digest_mismatch(tmp_path: Path):
    """provenance.yaml's ``inputs.sub_c_*_sha256`` values are sub-D's
    snapshot of the upstream sub-C bytes at extraction time. If sub-C is
    re-extracted and the bytes change, those digests no longer match and
    the validator must catch the drift.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    def _corrupt(data: dict) -> None:
        data["inputs"]["sub_c_cells_parquet_sha256"] = "f" * 64

    _rewrite_yaml(tile_dir / "provenance.yaml", _corrupt)
    # Recompute the manifest's tiles[].provenance_sha256 so the digest chain
    # at the region level still ties out — we want validate_tile to be the
    # one that catches this, not the region-level chain check.
    _rebuild_manifest_after_provenance_change(sub_c_region_dir, sub_d_region_dir)

    with pytest.raises(SubDValidationError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


def test_validator_uses_compare_version_for_namespace_checks(tmp_path: Path):
    """B3 (behavioural half): every version comparison in the validator uses
    ``compare_version`` from ``cfm.data.sub_d.versions`` — which raises
    ``VersionMismatchError`` (subclass of ``SubDValidationError``) on a
    value mismatch.

    If a future refactor changed the validator to raise a plain
    ``SubDValidationError`` from a raw ``!=`` check, this test would
    still fail at the ``pytest.raises(VersionMismatchError)`` line (because
    ``SubDValidationError`` is the base, not the subclass). That asymmetric
    typing is what proves the code went through compare_version.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    def _corrupt(data: dict) -> None:
        data["provenance_schema_version"] = "9.9"

    _rewrite_yaml(tile_dir / "provenance.yaml", _corrupt)
    _rebuild_manifest_after_provenance_change(sub_c_region_dir, sub_d_region_dir)

    with pytest.raises(VersionMismatchError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


def test_validator_files_do_not_compare_version_strings_directly():
    """B3 (static half): AST scan of ``src/cfm/data/sub_d/validator.py`` and
    any future ``src/cfm/data/sub_d/validator_*.py`` for ``ast.Compare`` nodes
    using ``==``/``!=`` where any operand subtree contains a name, attribute,
    or string subscript key containing "version".

    The only sanctioned version equality path is ``compare_version`` from
    ``cfm.data.sub_d.versions`` — validator files must not compare version
    strings directly.
    """
    import cfm.data.sub_d.validator as validator_mod

    validator_root = Path(validator_mod.__file__).parent
    candidate_paths = [
        validator_root / "validator.py",
        *sorted(validator_root.glob("validator_*.py")),
    ]
    assert candidate_paths, "no validator files found to scan"

    offenders: list[tuple[str, int, str]] = []

    def _contains_version_identifier(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and "version" in child.id.lower():
                return True
            if isinstance(child, ast.Attribute) and "version" in child.attr.lower():
                return True
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if "version" in child.value.lower():
                    return True
        return False

    for path in candidate_paths:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                continue
            operands = [node.left, *node.comparators]
            if any(_contains_version_identifier(op) for op in operands):
                offenders.append((str(path), node.lineno, ast.unparse(node)))

    assert not offenders, (
        "validator files compare version strings directly with ==/!=; use "
        f"compare_version() from cfm.data.sub_d.versions instead. "
        f"Offenders: {offenders}"
    )


def test_validator_rejects_manifest_config_drift_from_sub_c(tmp_path: Path):
    """B6 defense-in-depth: build_manifest already raises on
    ``config != sub_c_manifest['config']`` at write time, but validate_region
    re-checks at validation time. A manifest that was written before the
    sub-C config changed (or hand-edited after) must be caught here too.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)

    # Hand-edit the on-disk sub-D manifest's config to drift from sub-C.
    def _corrupt(data: dict) -> None:
        data["config"]["tile_size_m"] = 9999  # drift

    _rewrite_yaml(sub_d_region_dir / "manifest.yaml", _corrupt)

    with pytest.raises(SubDValidationError):
        validate_region(sub_d_region_dir, sub_c_region_dir)


def test_validator_rejects_provenance_output_sha_mismatch(tmp_path: Path):
    """provenance.yaml records sha256 of the per-tile sub-D output files
    (macro_core.parquet, derivation_evidence.parquet,
    effective_conditioning.yaml). If any one of those is corrupted on disk
    after the provenance was written, the recorded sha no longer matches the
    file bytes and the validator must catch the drift.
    """
    sub_c_region_dir, sub_d_region_dir = _build_happy_path_pair(tmp_path)
    tile_dir = _tile_dir(sub_d_region_dir)
    sub_c_inputs = _sub_c_inputs_for(sub_c_region_dir)

    def _corrupt(data: dict) -> None:
        data["outputs"]["macro_core_parquet_sha256"] = "f" * 64

    _rewrite_yaml(tile_dir / "provenance.yaml", _corrupt)
    _rebuild_manifest_after_provenance_change(sub_c_region_dir, sub_d_region_dir)

    with pytest.raises(SubDValidationError):
        validate_tile(tile_dir, sub_c_inputs, macro_vocab={})


# ---------------------------------------------------------------------------
# Helper: when a test corrupts a tile's provenance.yaml, the region manifest's
# tiles[i].provenance_sha256 anchor still references the OLD bytes. Some
# rejection tests target validate_tile specifically and want the region-level
# chain check to NOT fire first — they rebuild the manifest so the
# provenance_sha256 anchor matches the corrupted on-disk file. This isolates
# the failure mode each test targets.
# ---------------------------------------------------------------------------


def _rebuild_manifest_after_provenance_change(
    sub_c_region_dir: Path, sub_d_region_dir: Path
) -> None:
    sub_c_manifest = read_sub_c_manifest(sub_c_region_dir)
    sub_c_manifest_sha = compute_sha256((sub_c_region_dir / "manifest.yaml").read_bytes())
    tile_dir = _tile_dir(sub_d_region_dir)
    prov_data = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
    # Re-derive the manifest using the (possibly corrupted) provenance dict.
    sub_d_manifest_data = build_manifest(
        release="2026-04-15.0",
        region="singapore",
        region_crs="EPSG:3414",
        sub_c_manifest=sub_c_manifest,
        inputs={
            "sub_c_manifest_sha256": sub_c_manifest_sha,
            "sub_c_region_dir": str(sub_c_region_dir),
        },
        versions=_HAPPY_VERSIONS_MANIFEST,
        config=sub_c_manifest["config"],
        initial_extraction={
            "commit_sha": "abc" + "0" * 37,
            "started_utc": "2026-05-19T12:00:00Z",
            "completed_utc": "2026-05-19T12:10:00Z",
            "tile_count": 1,
        },
        tile_provenances=[prov_data],
    )
    write_manifest(sub_d_manifest_data, sub_d_region_dir / "manifest.yaml")
