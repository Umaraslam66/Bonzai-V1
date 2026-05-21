"""Sub-E derivation pipeline orchestrator.

Reads sub-D + sub-C, derives boundary contracts, writes per-tile artifacts,
validates inline and cross-tile, then writes _SUCCESS. Any validator failure
aborts the run; no partial _SUCCESS.

Lever-3 mode: `lever_3_collapse=True` skips the class-precedence derivation;
all boundary_class_enum values written as null.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from cfm.data.sub_c.enums import FEATURE_CLASS, encode_enum
from cfm.data.sub_e.derivation import derive_boundary_class
from cfm.data.sub_e.io import (
    SubCCrossingRow,
    SubCFeatureRow,
    SubDMacroCoreRow,
    read_sub_c_crossings,
    read_sub_c_features,
    read_sub_d_macro_core,
    require_sub_d_success_marker,
)
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
from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile
from cfm.data.sub_e.validator_inline import validate_boundary_contract
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_VERSION,
    BOUNDARY_VOCAB_VERSION,
    SUB_E_SCHEMA_VERSION,
)
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    release: str
    region: str
    sub_c_region_dir: Path
    sub_d_region_dir: Path
    output_region_dir: Path
    commit_sha: str
    lever_3_collapse: bool = False


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_or_raise(
    parquet_path: Path,
    expected_derivation_version: str,
    provenance_derivation_version: str,
    lever_3_collapse: bool,
) -> None:
    """Indirect call so tests can monkey-patch to simulate failure.

    Both ``expected_derivation_version`` (orchestrator's BOUNDARY_DERIVATION_VERSION
    constant) and ``provenance_derivation_version`` (read back from the
    just-written provenance.yaml on disk) are required by the inline
    validator post Task-7-defect-2. Passing the same constant for both
    would make inline invariant #8 vacuous; the orchestrator's caller
    threads the disk-read value to give the invariant real signal —
    catches divergence between SubEProvenance serialization and the
    in-memory constant if it ever drifts.
    """
    validate_boundary_contract(
        parquet_path,
        expected_derivation_version=expected_derivation_version,
        provenance_derivation_version=provenance_derivation_version,
        lever_3_collapse=lever_3_collapse,
    )


def derive_region(cfg: PipelineConfig) -> None:
    """Run the full sub-E derivation for a region. Halts on any validator
    failure; no _SUCCESS written on error.
    """
    require_sub_d_success_marker(cfg.sub_d_region_dir)
    cfg.output_region_dir.mkdir(parents=True, exist_ok=True)

    sub_d_manifest = yaml.safe_load((cfg.sub_d_region_dir / "manifest.yaml").read_text())
    sub_d_manifest_sha = _file_sha256(cfg.sub_d_region_dir / "manifest.yaml")
    sub_c_manifest_sha = _file_sha256(cfg.sub_c_region_dir / "manifest.yaml")
    boundary_vocab_path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "macro_plan"
        / "v1"
        / "boundary_vocab.yaml"
    )
    boundary_vocab_sha = _file_sha256(boundary_vocab_path)
    # parents[4] climbs src/cfm/data/sub_e/pipeline.py → repo root, matching
    # derivation.py's _VOCAB_PATH convention. Same depth, same index.

    started_utc = _utc_now()
    tile_records: list[SubEManifestTile] = []

    for sub_d_tile in sub_d_manifest["tiles"]:
        tile_i = sub_d_tile["tile_i"]
        tile_j = sub_d_tile["tile_j"]
        sub_d_tile_dir = cfg.sub_d_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        sub_c_tile_dir = cfg.sub_c_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"

        macro_core = read_sub_d_macro_core(sub_d_tile_dir / "macro_core.parquet")
        crossings = read_sub_c_crossings(sub_c_tile_dir / "crossings.parquet")
        features = read_sub_c_features(sub_c_tile_dir / "features.parquet")

        rows = _derive_tile_rows(
            macro_core=macro_core,
            crossings=crossings,
            features=features,
            lever_3_collapse=cfg.lever_3_collapse,
        )

        out_tile_dir = cfg.output_region_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        out_tile_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = write_boundary_contract(out_tile_dir / "boundary_contract.parquet", rows)
        parquet_sha = _file_sha256(parquet_path)

        # Provenance is the canonical record-as-written; the inline validator
        # reads it back from disk to give invariant #8 real signal. Earlier
        # draft validated BEFORE writing provenance and passed the same
        # constant for both kwargs, making #8 vacuous (constant-vs-itself).
        # Spec §9.4: provenance is canonical record of what was produced.
        # Halt-on-validator-fail discipline: if the validator raises, the
        # parquet + provenance exist on disk but no _SUCCESS marker is
        # written for the region — standard "incomplete" semantics
        # (spec §11.8 sub-C precedent). Next run sees no _SUCCESS and
        # re-derives cleanly.
        provenance = SubEProvenance(
            tile_i=tile_i,
            tile_j=tile_j,
            extraction_commit_sha=cfg.commit_sha,
            extracted_utc=started_utc,
            rerun_count=0,
            rerun_reason="initial",
            inputs=SubEInputDigests(
                release=cfg.release,
                sub_c_manifest_sha256=sub_c_manifest_sha,
                sub_c_features_parquet_sha256=_file_sha256(sub_c_tile_dir / "features.parquet"),
                sub_c_crossings_parquet_sha256=_file_sha256(sub_c_tile_dir / "crossings.parquet"),
                sub_d_manifest_sha256=sub_d_manifest_sha,
                sub_d_macro_core_parquet_sha256=_file_sha256(sub_d_tile_dir / "macro_core.parquet"),
                boundary_vocab_sha256=boundary_vocab_sha,
                derivation_config_sha256=boundary_vocab_sha,  # same file for v1
            ),
            versions=SubEVersions(
                sub_e_schema_version=SUB_E_SCHEMA_VERSION,
                boundary_vocab_version=BOUNDARY_VOCAB_VERSION,
                boundary_derivation_version=BOUNDARY_DERIVATION_VERSION,
            ),
            boundary_contract_parquet_sha256=parquet_sha,
        )
        prov_path = write_provenance(out_tile_dir / "provenance.yaml", provenance)

        # Read provenance back from DISK (not from the dataclass) so the
        # validator's invariant #8 catches divergence between the in-memory
        # constant and what serialization actually wrote. Reading the
        # dataclass instead would re-introduce the vacuous-comparison trap.
        prov_dict = yaml.safe_load(prov_path.read_text())
        _validate_or_raise(
            parquet_path,
            expected_derivation_version=BOUNDARY_DERIVATION_VERSION,
            provenance_derivation_version=prov_dict["versions"]["boundary_derivation_version"],
            lever_3_collapse=cfg.lever_3_collapse,
        )

        # Chain anchor uses provenance_sha256() — strips extracted_utc and
        # *_sha256 per SUB_E_EXCLUDED_FROM_SHA so the chain survives live-clock
        # reruns. Raw _file_sha256(prov_path) here would bake extracted_utc
        # into the chain and break determinism on every rerun (spec §9.2).
        tile_records.append(
            SubEManifestTile(
                tile_i=tile_i,
                tile_j=tile_j,
                provenance_sha256=provenance_sha256(provenance_to_dict(provenance)),
            )
        )

    completed_utc = _utc_now()
    write_manifest(
        cfg.output_region_dir / "manifest.yaml",
        SubEManifest(
            manifest_schema_version="1.0",
            sub_e_schema_version=SUB_E_SCHEMA_VERSION,
            release=cfg.release,
            region=cfg.region,
            region_crs="EPSG:3414",
            inputs=SubEManifestInputs(
                sub_c_manifest_sha256=sub_c_manifest_sha,
                sub_c_region_dir=str(cfg.sub_c_region_dir),
                sub_d_manifest_sha256=sub_d_manifest_sha,
                sub_d_region_dir=str(cfg.sub_d_region_dir),
                boundary_vocab_sha256=boundary_vocab_sha,
            ),
            versions=SubEManifestVersions(
                boundary_vocab_version=BOUNDARY_VOCAB_VERSION,
                boundary_derivation_version=BOUNDARY_DERIVATION_VERSION,
            ),
            config_source="sub_d_manifest.config",
            config=SubEManifestConfig(
                cell_grid=(8, 8),
                internal_edge_count=112,
                external_edge_count=32,
            ),
            initial_extraction=SubEManifestExtraction(
                commit_sha=cfg.commit_sha,
                started_utc=started_utc,
                completed_utc=completed_utc,
                tile_count=len(tile_records),
            ),
            tiles=tile_records,
        ),
    )
    # Halt-on-validator-fail discipline (sub-D precedent at
    # src/cfm/data/sub_d/pipeline.py:254-255; spec §11.8 sub-C precedent):
    # cross-tile validator runs BEFORE _SUCCESS is written. If validation
    # raises, the touch never runs and consumers see no green-light marker
    # — disk state is consistent with "this run did not succeed."
    #
    # Earlier draft did write→try-except→unlink. That pattern (a) created a
    # brief window where _SUCCESS existed before validation completed (false
    # green for any polling observer), and (b) handled failure via recovery
    # rather than by-construction non-occurrence — if unlink itself fails
    # (race, permission, signal), disk state is permanently inconsistent.
    # Validate-then-touch has no race window and no failure mode in the
    # failure handler.
    validate_extraction_cross_tile(cfg.output_region_dir)
    (cfg.output_region_dir / "_SUCCESS").touch()


def _derive_tile_rows(
    *,
    macro_core: list[SubDMacroCoreRow],
    crossings: list[SubCCrossingRow],
    features: list[SubCFeatureRow],
    lever_3_collapse: bool,
) -> list[BoundaryContractRow]:
    """Construct the 144-row per-tile boundary contract from sub-D + sub-C.

    Edge rows are keyed by ``(slot_kind, lower_cell_i, lower_cell_j, axis)``
    rather than the 3-tuple ``(li, lj, axis)`` because rotation's per-cell
    enumeration can produce identical ``(li, lj, axis)`` triples for the
    internal and external versions of distinct physical edges (e.g. cell
    (0,0)'s north and cell (0,1)'s north both encode as (0, 0, 0) under
    the lower_lj convention in rotation.py). ``slot_kind`` disambiguates;
    without it the dict collapses keys and the writer rejects the row count.
    Sub-C crossings, conversely, are keyed by the 3-tuple — they apply only
    to active internal edges (external rows are scope=3 and skipped).
    """
    edge_scope: dict[tuple[int, int, int, int], int] = {}
    edge_slot_index: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for r in macro_core:
        if r.slot_kind in (1, 2):  # internal or external edge
            assert r.lower_cell_i is not None
            assert r.lower_cell_j is not None
            assert r.axis is not None
            key = (r.slot_kind, r.lower_cell_i, r.lower_cell_j, r.axis)
            edge_scope[key] = r.scope
            edge_slot_index[key] = (r.slot_kind, r.slot_index)

    # Filter features down to road geometries. feature_class is int8 per sub-C
    # contract (FEATURE_CLASS: {0: "road", 1: "building", 2: "poi", 3: "base"}
    # at sub_c/enums.py:22). Earlier draft compared against the string "road"
    # which silently never matched and made the filter a no-op — see
    # SubCFeatureRow docstring at sub_e/io.py for the postmortem. Comparing
    # symbolically via encode_enum(FEATURE_CLASS, "road") rather than the
    # magic number 0 ties this site directly to sub-C's enum source.
    _road_class_code = encode_enum(FEATURE_CLASS, "road")
    features_by_id: dict[str, str | None] = {
        f.source_feature_id: f.class_raw for f in features if f.feature_class == _road_class_code
    }
    crossings_by_edge: dict[tuple[int, int, int], list[str | None]] = {}
    for c in crossings:
        key = (c.lower_cell_i, c.lower_cell_j, c.axis)
        crossings_by_edge.setdefault(key, []).append(features_by_id.get(c.source_feature_id))

    rows: list[BoundaryContractRow] = []
    for key, scope in edge_scope.items():
        slot_kind_int, i, j, axis = key
        _, slot_idx = edge_slot_index[key]
        is_active_internal = scope == 0 and slot_kind_int == 1
        if is_active_internal and not lever_3_collapse:
            # Pass all crossings (including None entries) through to
            # derive_boundary_class. Per spec §5.1 + derivation.py:84-85,
            # None entries map to the MINOR_ROAD default bucket; filtering
            # them out would change semantics. Earlier draft had
            # `if cr is not None or True` which short-circuited to always
            # True — dead code that obscured intent.
            class_raws = list(crossings_by_edge.get((i, j, axis), []))
            bc = int(derive_boundary_class(class_raws))
        else:
            bc = None
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind(slot_kind_int),
                slot_index=slot_idx,
                lower_cell_i=i,
                lower_cell_j=j,
                axis=axis,
                scope_marker=scope,
                boundary_class_enum=bc,
            )
        )
    return rows
