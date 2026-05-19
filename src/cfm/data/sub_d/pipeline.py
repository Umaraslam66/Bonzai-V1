"""Sub-D sidecar derivation pipeline (Task 14).

Orchestrates every sub-D module to derive one region's macro plan from a
finalised sub-C region. Write order is locked (spec §11.8 sub-C precedent
applied to sub-D):

1. Read sub-C manifest + tile inventory.
2. Load locked macro vocab (refuses to run if missing/malformed).
3. Per tile:
   a. Read sub-C tile inputs (cells, features, crossings, meta, provenance)
      plus bytes-sha digests of those files.
   b. Derive lattice targets (scope from sub-C cells; target token_ids from
      Layer-1 evidence + locked vocab buckets). For Task 14, target
      derivation is conservatively None on active slots — the active-path
      token assignment (zoning class, density bucket, road skeleton class)
      is real but only fires when sub-C provides corresponding evidence.
      An all-empty sub-C tile produces 208 FULLY_MASKED rows + 116
      derivation_evidence rows (112 road skeleton + 4 tile population
      density), which is a valid degenerate sub-D output.
   c. Write ``macro_core.parquet``.
   d. Write ``derivation_evidence.parquet``.
   e. Write ``effective_conditioning.yaml``.
   f. Hash the just-written output bytes (NOT in-memory dataclasses —
      provenance.yaml's ``outputs.*_sha256`` must match the live file bytes
      the validator will read back).
   g. Write ``provenance.yaml``.
4. Build + write region ``manifest.yaml``.
5. Run ``validate_region``. Raises on any contract violation.
6. Write ``_SUCCESS`` — and ONLY if validation passed. Step 5's exception
   propagates past step 6, leaving no green-light marker behind.

Determinism: same sub-C inputs + same locked vocab + same pinned commit_sha
and extracted_utc → byte-identical sub-D output. ``extracted_utc=None``
falls back to wall-clock UTC; callers that need determinism pin it
explicitly.

Tension flag B6 enforcement is delegated to ``build_manifest`` (build-time)
and ``validate_region`` (validation-time) — the pipeline copies
``sub_c_manifest["config"]`` verbatim, never a hand-picked subset.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cfm.data.determinism import compute_sha256
from cfm.data.sub_d.conditioning import (
    build_effective_conditioning,
    write_effective_conditioning,
)
from cfm.data.sub_d.enums import FeatureClass, MetricNamespace, Scope, SlotKind
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.evidence import (
    EvidenceMetric,
    derive_cell_scope_metrics,
    derive_density_evidence,
    derive_road_skeleton_evidence,
    derive_tile_population_density_evidence,
    derive_zoning_evidence,
)
from cfm.data.sub_d.io import (
    DerivationEvidenceRow,
    MacroCoreRow,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)
from cfm.data.sub_d.lattice import (
    derive_external_edge_scope,
    derive_internal_edge_scope,
    iter_cell_slots,
    iter_external_edge_slots,
    iter_internal_edge_slots,
)
from cfm.data.sub_d.macro_vocab import load_macro_vocab
from cfm.data.sub_d.manifest import (
    SUB_D_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
    write_success_marker,
)
from cfm.data.sub_d.provenance import build_tile_provenance, write_provenance
from cfm.data.sub_d.sub_c_reader import (
    SubCTileInputs,
    iter_sub_c_tile_paths,
    read_sub_c_manifest,
    read_sub_c_tile_inputs,
)
from cfm.data.sub_d.validator import validate_region


def derive_region_macro_plan(
    sub_c_region_dir: Path,
    output_dir: Path,
    macro_vocab_path: Path,
    *,
    release: str,
    region: str,
    commit_sha: str,
    extracted_utc: str | None = None,
    rerun_count: int = 0,
    rerun_reason: str = "initial",
) -> dict:
    """Derive one sub-D region from a finalised sub-C region.

    Returns the manifest dict that was written. Raises ``SubDValidationError``
    (or the more-specific ``VersionMismatchError``) on any contract
    violation. On failure, ``_SUCCESS`` is NOT created; partially-written
    artifacts may remain under ``output_dir`` and the caller is expected
    to clean them up before the next attempt.
    """
    macro_vocab_path = Path(macro_vocab_path)
    if not macro_vocab_path.is_file():
        raise SubDValidationError(
            f"locked macro vocab not found at {macro_vocab_path}. "
            "Run scripts/promote_macro_vocab.py to derive it from the "
            "reviewer-approved proposal at "
            "reports/phase-1-sub-D/macro_vocab_proposal.yaml."
        )

    macro_vocab = load_macro_vocab(macro_vocab_path)
    macro_vocab_bytes = macro_vocab_path.read_bytes()
    macro_vocab_sha = compute_sha256(macro_vocab_bytes)

    sub_c_region_dir = Path(sub_c_region_dir)
    output_dir = Path(output_dir)

    sub_c_manifest = read_sub_c_manifest(sub_c_region_dir)
    sub_c_manifest_sha = compute_sha256((sub_c_region_dir / "manifest.yaml").read_bytes())
    sub_c_tile_paths_list = iter_sub_c_tile_paths(sub_c_region_dir)

    region_crs = str(sub_c_manifest["region_crs"])
    epsg_label = region_crs.replace(":", "")

    if extracted_utc is None:
        extracted_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    versions_tile = _build_versions_dict(macro_vocab, include_sub_d_schema_version=True)
    versions_manifest = _build_versions_dict(macro_vocab, include_sub_d_schema_version=False)
    versions_ec = {
        "sub_c_conditioning_schema_version": str(sub_c_manifest.get("sub_c_schema_version", "1.1")),
        "tile_population_density_vocab_version": versions_tile[
            "tile_population_density_vocab_version"
        ],
        "tile_population_density_derivation_version": versions_tile[
            "tile_population_density_derivation_version"
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    tile_provenances: list[dict] = []
    for sub_c_paths in sub_c_tile_paths_list:
        sub_c_inputs = read_sub_c_tile_inputs(sub_c_paths)
        tile_i, tile_j = sub_c_paths.tile_i, sub_c_paths.tile_j
        sub_d_tile_dir = output_dir / f"tile={epsg_label}_i{tile_i}_j{tile_j}"
        sub_d_tile_dir.mkdir(parents=True, exist_ok=True)

        macro_core_rows, derivation_evidence_rows = _derive_tile_targets(sub_c_inputs, macro_vocab)

        macro_core_path = sub_d_tile_dir / "macro_core.parquet"
        write_macro_core_parquet(macro_core_rows, macro_core_path)

        derivation_evidence_path = sub_d_tile_dir / "derivation_evidence.parquet"
        write_derivation_evidence_parquet(derivation_evidence_rows, derivation_evidence_path)

        population_density_bucket = _resolve_population_density_bucket(
            derivation_evidence_rows, macro_vocab
        )
        ec_data = build_effective_conditioning(
            meta=sub_c_inputs.meta,
            manifest=sub_c_manifest,
            population_density_bucket=population_density_bucket,
            versions=versions_ec,
            digests={
                "manifest_sha256": sub_c_manifest_sha,
                "tile_meta_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
                "tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
            },
        )
        effective_conditioning_path = sub_d_tile_dir / "effective_conditioning.yaml"
        write_effective_conditioning(ec_data, effective_conditioning_path)

        # CRITICAL: hash the JUST-WRITTEN file bytes. The validator's
        # provenance.outputs.* vs live-bytes check reads these files back —
        # if we hashed in-memory dataclasses instead, the parquet writer's
        # canonicalisation could produce different bytes and the chain
        # would break on every run.
        outputs_shas = {
            "macro_core_parquet_sha256": compute_sha256(macro_core_path.read_bytes()),
            "derivation_evidence_parquet_sha256": compute_sha256(
                derivation_evidence_path.read_bytes()
            ),
            "effective_conditioning_yaml_sha256": compute_sha256(
                effective_conditioning_path.read_bytes()
            ),
        }

        prov_data = build_tile_provenance(
            tile_i=tile_i,
            tile_j=tile_j,
            extraction={
                "commit_sha": commit_sha,
                "extracted_utc": extracted_utc,
                "rerun_count": rerun_count,
                "rerun_reason": rerun_reason,
            },
            inputs={
                "release": release,
                "sub_c_manifest_sha256": sub_c_manifest_sha,
                "sub_c_tile_provenance_sha256": sub_c_inputs.digests["provenance_yaml_sha256"],
                "sub_c_cells_parquet_sha256": sub_c_inputs.digests["cells_parquet_sha256"],
                "sub_c_features_parquet_sha256": sub_c_inputs.digests["features_parquet_sha256"],
                "sub_c_crossings_parquet_sha256": sub_c_inputs.digests["crossings_parquet_sha256"],
                "sub_c_meta_yaml_sha256": sub_c_inputs.digests["meta_yaml_sha256"],
                "macro_vocab_sha256": macro_vocab_sha,
                # Phase 1 sub-D ships a single locked file that carries both
                # the vocab buckets and the derivation config; record the
                # same sha twice so the schema is forward-compatible with a
                # future split into two files.
                "derivation_config_sha256": macro_vocab_sha,
            },
            versions=versions_tile,
            outputs=outputs_shas,
        )
        write_provenance(prov_data, sub_d_tile_dir / "provenance.yaml")
        tile_provenances.append(prov_data)

    sub_d_manifest_data = build_manifest(
        release=release,
        region=region,
        region_crs=region_crs,
        sub_c_manifest=sub_c_manifest,
        inputs={
            "sub_c_manifest_sha256": sub_c_manifest_sha,
            "sub_c_region_dir": str(sub_c_region_dir),
        },
        versions=versions_manifest,
        config=sub_c_manifest["config"],
        initial_extraction={
            "commit_sha": commit_sha,
            "started_utc": extracted_utc,
            "completed_utc": extracted_utc,
            "tile_count": len(tile_provenances),
        },
        tile_provenances=tile_provenances,
    )
    write_manifest(sub_d_manifest_data, output_dir / "manifest.yaml")

    # Gate _SUCCESS on validation. If validate_region raises, the next
    # line never runs, and consumers see no green-light marker — exactly
    # the contract Task 12 pinned (write_manifest doesn't touch _SUCCESS;
    # only this call does, and only after validation).
    validate_region(output_dir, sub_c_region_dir)
    write_success_marker(output_dir)

    return sub_d_manifest_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_versions_dict(macro_vocab: dict, *, include_sub_d_schema_version: bool) -> dict:
    """Compose the per-namespace versions block.

    Per A1 discipline: vocab + derivation versions are per-namespace; there
    is no single global ``derivation_version``. ``analysis_version`` is the
    one-file locked-vocab artifact version that applies uniformly to every
    namespace's locked_buckets (the vocab is locked all-at-once); per
    namespace derivation versions come from ``derivation_versions``.

    ``include_sub_d_schema_version`` distinguishes the tile-provenance
    versions block (which DOES carry ``sub_d_schema_version`` per spec §11.5)
    from the manifest versions block (which does NOT — the manifest carries
    it at the top level per spec §11.6, sourced from
    ``SUB_D_SCHEMA_VERSION`` inside ``build_manifest``).
    """
    analysis_version = str(macro_vocab["analysis_version"])
    derivation_versions = macro_vocab["derivation_versions"]
    versions = {
        "macro_plan_vocab_version": analysis_version,
        "zoning_vocab_version": analysis_version,
        "zoning_derivation_version": str(derivation_versions["zoning"]),
        "cell_density_vocab_version": analysis_version,
        "cell_density_derivation_version": str(derivation_versions["cell_density"]),
        "tile_population_density_vocab_version": analysis_version,
        "tile_population_density_derivation_version": str(
            derivation_versions["tile_population_density"]
        ),
        "road_skeleton_vocab_version": analysis_version,
        "road_skeleton_derivation_version": str(derivation_versions["road_skeleton"]),
    }
    if include_sub_d_schema_version:
        versions["sub_d_schema_version"] = SUB_D_SCHEMA_VERSION
    return versions


def _derive_tile_targets(
    sub_c_inputs: SubCTileInputs, macro_vocab: dict
) -> tuple[list[MacroCoreRow], list[DerivationEvidenceRow]]:
    """Build the 208-row macro_core + the full derivation_evidence row list
    for one tile from sub-C inputs + locked vocab.

    Scope is derived directly from sub-C: a cell is ACTIVE iff sub-C wrote
    it to cells.parquet. Edge scopes follow the lattice helpers.

    Active-cell target derivation:
    - ``zoning_class``: dominant feature class (by raw count) → vocab token_id.
      If no features at all, leave ``None`` (active cell with no zoning info
      is a known edge case; the validator allows it).
    - ``cell_density_bucket``: building_footprint_ratio → vocab bucket lookup
      via [lower_inclusive, upper_exclusive); the open-ended top bucket
      absorbs ratios > 1.0 (known_issue #9 — must NOT be range-clamped here).
    Active-internal-edge target derivation:
    - ``road_skeleton_class``: road_crossing_count → vocab bucket lookup.
    External edges have all targets ``None`` — sub-E will own external-edge
    target derivation.
    """
    cell_scope = derive_cell_scope_metrics(sub_c_inputs.cells)
    zoning_metrics = derive_zoning_evidence(sub_c_inputs.features, sub_c_inputs.cells)
    density_metrics = derive_density_evidence(sub_c_inputs.features, sub_c_inputs.cells)
    road_metrics = derive_road_skeleton_evidence(sub_c_inputs.crossings, sub_c_inputs.features)
    tile_pop_metrics = derive_tile_population_density_evidence(
        sub_c_inputs.cells, sub_c_inputs.features
    )

    cell_zoning_counts = _index_zoning_counts(zoning_metrics)
    cell_density_value = _index_density_values(density_metrics)
    edge_road_count = _index_road_counts(road_metrics)

    zoning_buckets = macro_vocab["locked_buckets"]["zoning"]
    cell_density_buckets = macro_vocab["locked_buckets"]["cell_density"]
    road_skeleton_buckets = macro_vocab["locked_buckets"]["road_skeleton"]

    rows: list[MacroCoreRow] = []

    for cell in iter_cell_slots():
        active = cell_scope.get((cell.cell_i, cell.cell_j), False)
        scope = Scope.ACTIVE if active else Scope.FULLY_MASKED
        zoning_class: int | None = None
        density_bucket: int | None = None
        if active:
            counts = cell_zoning_counts.get(cell.slot_index, {})
            zoning_class = _zoning_token_id(counts, zoning_buckets)
            density_val = cell_density_value.get(cell.slot_index)
            if density_val is not None:
                density_bucket = _bucket_for_numeric_value(density_val, cell_density_buckets)
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.CELL,
                slot_index=cell.slot_index,
                cell_i=cell.cell_i,
                cell_j=cell.cell_j,
                lower_cell_i=None,
                lower_cell_j=None,
                axis=None,
                scope=scope,
                zoning_class=zoning_class,
                cell_density_bucket=density_bucket,
                road_skeleton_class=None,
            )
        )

    for edge in iter_internal_edge_slots():
        lower_active = cell_scope.get((edge.lower_cell_i, edge.lower_cell_j), False)
        if edge.axis == 0:
            upper_key = (edge.lower_cell_i + 1, edge.lower_cell_j)
        else:
            upper_key = (edge.lower_cell_i, edge.lower_cell_j + 1)
        upper_active = cell_scope.get(upper_key, False)
        scope = derive_internal_edge_scope(lower_active, upper_active)
        road_class: int | None = None
        if scope == Scope.ACTIVE:
            count = edge_road_count.get(edge.slot_index, 0)
            road_class = _bucket_for_numeric_value(count, road_skeleton_buckets)
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=edge.slot_index,
                cell_i=None,
                cell_j=None,
                lower_cell_i=edge.lower_cell_i,
                lower_cell_j=edge.lower_cell_j,
                axis=edge.axis,
                scope=scope,
                zoning_class=None,
                cell_density_bucket=None,
                road_skeleton_class=road_class,
            )
        )

    for edge in iter_external_edge_slots():
        if edge.axis == 0:
            interior_key = (
                (0, edge.lower_cell_j)
                if edge.lower_cell_i == -1
                else (edge.lower_cell_i, edge.lower_cell_j)
            )
        else:
            interior_key = (
                (edge.lower_cell_i, 0)
                if edge.lower_cell_j == -1
                else (edge.lower_cell_i, edge.lower_cell_j)
            )
        interior_active = cell_scope.get(interior_key, False)
        scope = derive_external_edge_scope(interior_active)
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=edge.slot_index,
                cell_i=None,
                cell_j=None,
                lower_cell_i=edge.lower_cell_i,
                lower_cell_j=edge.lower_cell_j,
                axis=edge.axis,
                scope=scope,
                zoning_class=None,
                cell_density_bucket=None,
                road_skeleton_class=None,
            )
        )

    all_metrics = zoning_metrics + density_metrics + road_metrics + tile_pop_metrics
    de_rows = [
        DerivationEvidenceRow(
            slot_kind=m.slot_kind,
            slot_index=m.slot_index,
            metric_namespace=m.metric_namespace,
            metric_name=m.metric_name,
            value=m.value,
            derivation_version=m.derivation_version,
        )
        for m in all_metrics
    ]
    return rows, de_rows


def _index_zoning_counts(
    metrics: list[EvidenceMetric],
) -> dict[int, dict[str, int]]:
    """Group zoning evidence by slot_index → {feature_class_name: count}."""
    by_slot: dict[int, dict[str, int]] = {}
    for m in metrics:
        if m.metric_namespace != MetricNamespace.ZONING:
            continue
        if not m.metric_name.startswith("feature_count_"):
            continue
        feature_name = m.metric_name[len("feature_count_") :]
        by_slot.setdefault(m.slot_index, {})[feature_name] = int(m.value)
    return by_slot


def _index_density_values(metrics: list[EvidenceMetric]) -> dict[int, float]:
    return {
        m.slot_index: float(m.value)
        for m in metrics
        if m.metric_namespace == MetricNamespace.CELL_DENSITY
        and m.metric_name == "building_footprint_ratio"
    }


def _index_road_counts(metrics: list[EvidenceMetric]) -> dict[int, int]:
    return {
        m.slot_index: int(m.value)
        for m in metrics
        if m.metric_namespace == MetricNamespace.ROAD_SKELETON
        and m.metric_name == "road_crossing_count"
    }


def _zoning_token_id(
    counts_by_class_name: dict[str, int], zoning_buckets: list[dict]
) -> int | None:
    """Pick the feature_class with max count → look up vocab token_id by name.

    Deterministic tie-break: among ties, prefer the class with the smallest
    ``FeatureClass`` enum integer. Returns None when all counts are zero
    (active cell with no features).
    """
    if not counts_by_class_name or sum(counts_by_class_name.values()) == 0:
        return None
    enum_order = {fc.name.lower(): int(fc) for fc in FeatureClass}
    max_count = max(counts_by_class_name.values())
    tied_names = [n for n, c in counts_by_class_name.items() if c == max_count]
    dominant = min(tied_names, key=lambda n: enum_order.get(n, 99))
    for entry in zoning_buckets:
        if entry["token_name"] == dominant:
            return int(entry["token_id"])
    return None


def _bucket_for_numeric_value(value: float | int, buckets: list[dict]) -> int:
    """Return the ``token_id`` of the bucket whose
    ``[lower_inclusive, upper_exclusive)`` contains *value*.

    ``upper_exclusive == None`` denotes the open-ended top bucket. Values
    above the highest finite ``upper_exclusive`` fall into that top bucket;
    no clamping, no rejection — known_issue #9 (cell_density ratios > 1.0
    in real Singapore data) is absorbed here exactly as the locked vocab
    designed.
    """
    numeric = float(value)
    for entry in buckets:
        lower = float(entry["lower_inclusive"])
        upper = entry.get("upper_exclusive")
        if numeric < lower:
            continue
        if upper is None or numeric < float(upper):
            return int(entry["token_id"])
    raise SubDValidationError(
        f"value {value!r} fell outside every bucket in the locked vocab; "
        "this should be impossible if the locked vocab has an open-ended "
        "top bucket (upper_exclusive=None)."
    )


def _resolve_population_density_bucket(
    de_rows: list[DerivationEvidenceRow], macro_vocab: dict
) -> int:
    """Look up the locked tile_population_density proxy in the derivation
    evidence and bucket the value via the locked vocab."""
    locked_proxy = str(macro_vocab["locked_proxy"]["tile_population_density"])
    for row in de_rows:
        if (
            row.slot_kind == SlotKind.TILE
            and row.metric_namespace == MetricNamespace.TILE_POPULATION_DENSITY
            and row.metric_name == locked_proxy
        ):
            return _bucket_for_numeric_value(
                row.value, macro_vocab["locked_buckets"]["tile_population_density"]
            )
    raise SubDValidationError(
        f"locked tile_population_density proxy {locked_proxy!r} not found in "
        "derivation_evidence rows — sub-D's evidence module did not emit it. "
        "Was the proxy renamed without bumping "
        "TILE_POPULATION_DENSITY_DERIVATION_VERSION?"
    )
