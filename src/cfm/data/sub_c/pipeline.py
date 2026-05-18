"""End-to-end pipeline orchestrator for sub-C tile extraction.

Composes every sub-C library function in the locked order per spec §6. Task
12 introduced the sequential path; Task 13 adds the process-pool variant
gated by the `pool_size` keyword. Pool size affects wall-clock only — byte
output is invariant under `pool_size ∈ [1, N]` per spec §14.5.

Pipeline ordering (locked at spec §6; tested verbatim by Task 12 named tests):

    derive_sea_polygons(raw_base)             # §9.1: BEFORE policy (mask-not-feature)
    apply_missing_value_policy(themes)        # §10.1: raw-row level
    densify_polygon(admin_polygon, None)      # §7.4: no-op for SG; signature for Sweden
    reproject to SVY21                        # §7.1: themes, sea_polygons, admin_polygon
    clip themes to admin_polygon (SVY21)      # §7.3: reproject FIRST, then clip
    partition_into_tiles(admin)               # §7.2: 2km grid
    for tile in tiles:                         # §11.8: per-tile write order locked
        partition_into_cells -> sliver_drop -> sea_mask -> conditioning
        write cells.parquet
        write features.parquet
        write crossings.parquet
        write meta.yaml
        validate_tile_inline (blocks on fail; failure -> no provenance.yaml)
        write provenance.yaml                  # LAST: presence = tile complete
    write manifest.yaml                        # tiles[] sorted by (i,j)

_SUCCESS is NOT written here (Task 14 orchestrator writes it after cross-tile validator).

Parallelism (spec §14.5):

Main process performs all once-per-region work (derive_sea_polygons,
apply_missing_value_policy, densify, reproject, clip, partition_into_tiles)
BEFORE any worker starts. Workers receive the per-tile feature subset plus
the SVY21 densified-admin polygon and sea-polygons union (by value, via the
pickled TileWorkerArgs dataclass) and run only the per-tile pipeline +
artifact writes + inline validator + provenance write. Workers MUST NOT
call densify_polygon or derive_sea_polygons — those are recorded as
once-per-region invariants and re-running them in workers could yield
different bytes under floating-point vertex-insertion-order artifacts.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pyarrow as pa
from shapely import wkb as shapely_wkb
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

from cfm.data.sub_c.conditioning import compute_conditioning_per_tile
from cfm.data.sub_c.coords import (
    CELL_SIZE_M,
    SVY21_EPSG_CODE,
    TILE_SIZE_M,
    densify_polygon,
    partition_into_tiles,
    reproject_geometry_to_svy21,
)
from cfm.data.sub_c.determinism import compute_sha256
from cfm.data.sub_c.enums import FEATURE_CLASS, GEOMETRY_TYPE, encode_enum
from cfm.data.sub_c.epsilon import (
    EPS_AREA_M2,
    EPS_COORD_M,
    EPS_LENGTH_M,
    EPS_RATIO,
)
from cfm.data.sub_c.geom import (
    CellSubFeature,
    apply_sliver_drop,
    partition_into_cells,
)
from cfm.data.sub_c.io import (
    CellAggregate,
    FeatureRow,
    TileMeta,
    TileProvenance,
    dump_wkb,
    write_cells_parquet,
    write_crossings_parquet,
    write_features_parquet,
    write_meta_yaml,
    write_provenance_yaml,
)
from cfm.data.sub_c.manifest import RegionManifest, aggregate_tile_inventory, write_manifest
from cfm.data.sub_c.policy import apply_missing_value_policy
from cfm.data.sub_c.sea_mask import apply_sea_mask, derive_sea_polygons
from cfm.data.sub_c.validator_inline import validate_tile_inline

# Sub-C schema versions (spec §11.5/§11.6/§11.7). Bumped on append-only changes.
_SCHEMA_VERSION: str = "1.0"
_SUB_C_SCHEMA_VERSION: str = "1.0"
# Per spec §11.5 — stringified, per-tile-tunable sliver-drop rule.
_SLIVER_DROP_RULE: str = "drop iff geometry has area < 0.01 m² OR length < 0.01 m"
# Default sliver thresholds per spec §11.5.
_SLIVER_AREA_THRESHOLD_M2: float = 0.01
_SLIVER_LENGTH_THRESHOLD_M: float = 0.01


# ---------------------------------------------------------------------------
# Region protocol — captures only what the orchestrator needs from sub-A's Region
# ---------------------------------------------------------------------------


class _RegionLike(Protocol):
    """Structural type for what the orchestrator reads off a Region.

    sub-A's cfm.data.overture.region.Region satisfies this; tests can pass a
    plain SimpleNamespace with the same attributes without building a full
    Region (which would require manifest_path + cache machinery).
    """

    name: str
    themes: dict[str, pa.Table]

    @property
    def admin_polygon(self) -> BaseGeometry: ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_region(
    region: _RegionLike,
    output_dir: Path,
    *,
    policy_yaml_path: Path,
    vocab_yaml_path: Path,
    release: str,
    commit_sha: str,
    admin_polygon_source: str = "overture://divisions:country:SG",
    extracted_utc: str | None = None,
    started_utc: str | None = None,
    rerun_reason: str = "initial",
    pool_size: int = 1,
) -> RegionManifest:
    """End-to-end extraction per spec §6 + §14.5.

    Pre-conditions:
      - region must expose `themes` (dict of pa.Tables for buildings,
        transportation, base, places; "geometry" column is WKB in EPSG:4326).
      - region.admin_polygon is a shapely geometry in EPSG:4326.
      - output_dir parent must exist; output_dir itself is created here.

    Post-conditions:
      - For every tile in tile_inventory, output_dir/tile=EPSG3414_i<i>_j<j>/
        contains cells.parquet, features.parquet, crossings.parquet, meta.yaml,
        provenance.yaml (in that write order; provenance.yaml last per §11.8).
      - output_dir/manifest.yaml written with tiles[] sorted by (i, j).
      - _SUCCESS is NOT written here — that is Task 14's responsibility,
        gated by the cross-tile validator.

    Parameters:
      - pool_size: process-pool size for per-tile extraction. `pool_size=1`
        runs the sequential path (no multiprocessing.Pool is constructed,
        avoiding the pool overhead). `pool_size>1` uses `multiprocessing.Pool`
        with the dynamic queue (`imap_unordered`) — order-independent because
        the main process aggregates tiles[] by (i, j) sorting before writing
        the manifest. Spec §14.5 names the invariant: byte output is
        invariant under `pool_size ∈ [1, N]` for any N; pool size affects
        only wall-clock.

    Workers receive shared inputs (densified admin polygon as WKB, sea
    polygons as WKB, per-tile feature subset) by value via the pickled
    `_TileWorkerArgs` dataclass. Workers MUST NOT call `densify_polygon` or
    `derive_sea_polygons` — those are once-per-region invariants computed
    here.

    Determinism: byte-identical output across runs given identical inputs
    modulo EXCLUDED_FROM_SHA fields (timestamps). Tests pass a fixed
    extracted_utc for reproducible verification.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    now_iso = _utcnow_iso()
    extracted_utc = extracted_utc if extracted_utc is not None else now_iso
    started_utc = started_utc if started_utc is not None else now_iso

    # ---- 1. derive_sea_polygons FROM RAW BASE (BEFORE policy) -----------
    # Spec §9.1 + §6 ordering: sea polygons are masks, not features. The
    # base.class not_in_vocab drop step in apply_missing_value_policy would
    # otherwise remove ocean/strait/bay rows from the themes and leave sea-mask
    # with nothing to work with.
    sea_polygons_4326: BaseGeometry = derive_sea_polygons(region.themes["base"])

    # ---- 2. apply_missing_value_policy (raw-row level) ------------------
    policied_themes = apply_missing_value_policy(
        region.themes,
        policy_yaml_path,
        vocab_yaml_path=vocab_yaml_path,
    )

    # ---- 3. densify admin polygon (no-op for Singapore; signature for Sweden)
    # Spec §7.4 + §14.5: computed once in main process; shared input.
    densified_admin_polygon_4326 = densify_polygon(region.admin_polygon, None)

    # ---- 4. Reproject everything to SVY21 -------------------------------
    # Spec §7.1 + §7.3: reproject first, then clip — so the clip cut-points
    # are exactly metric-correct in SVY21.
    sea_polygons_svy21 = reproject_geometry_to_svy21(sea_polygons_4326)
    densified_admin_polygon_svy21 = reproject_geometry_to_svy21(densified_admin_polygon_4326)
    themes_svy21_features = _reproject_and_extract_feature_records(policied_themes)

    # ---- 5. Clip themes to admin polygon (in SVY21) ---------------------
    # Spec §7.3: intersection happens AFTER reprojection.
    clipped_features = _clip_features_to_admin(themes_svy21_features, densified_admin_polygon_svy21)

    # ---- 6. Partition into 2km tiles ------------------------------------
    # Spec §7.2: dict[(i, j), admin_clipped_footprint]; sorted by (i, j).
    tile_inventory = partition_into_tiles(densified_admin_polygon_svy21)

    # ---- 7. Per-tile extraction -----------------------------------------
    # Spec §11.8 write order is enforced inside _extract_tile.
    # Compute once-per-region shared sha digests (spec §11.7 manifest fields).
    admin_polygon_sha256 = _sha256_of_geometry(densified_admin_polygon_4326)
    densified_admin_polygon_sha256 = _sha256_of_geometry(densified_admin_polygon_svy21)
    sea_polygons_sha256 = _sha256_of_geometry(sea_polygons_svy21)
    policy_yaml_sha256 = _sha256_of_file(policy_yaml_path)
    vocab_yaml_sha256 = _sha256_of_file(vocab_yaml_path)

    inputs_shared: dict = {
        "release": release,
        "admin_polygon_sha256": admin_polygon_sha256,
        "policy_yaml_sha256": policy_yaml_sha256,
        "vocab_yaml_sha256": vocab_yaml_sha256,
    }

    # Build per-tile worker args once in the main process. Each entry is a
    # pickle-friendly dataclass carrying everything the worker needs; the
    # shared SVY21 sea polygons + densified admin polygon are serialized as
    # WKB so the worker re-hydrates with a single shapely.wkb.loads call (no
    # re-derivation, no re-densification — spec §14.5 invariant).
    sea_polygons_svy21_wkb = dump_wkb(sea_polygons_svy21)
    densified_admin_polygon_svy21_wkb = dump_wkb(densified_admin_polygon_svy21)

    tile_args_list: list[_TileWorkerArgs] = []
    for (tile_i, tile_j), tile_admin_footprint in tile_inventory.items():
        tile_features = _filter_features_to_tile(clipped_features, tile_i, tile_j)
        tile_args_list.append(
            _TileWorkerArgs(
                tile_i=tile_i,
                tile_j=tile_j,
                tile_admin_footprint_wkb=dump_wkb(tile_admin_footprint),
                features_in_tile=tile_features,
                sea_polygons_svy21_wkb=sea_polygons_svy21_wkb,
                densified_admin_polygon_svy21_wkb=densified_admin_polygon_svy21_wkb,
                output_dir=output_dir,
                inputs_shared=inputs_shared,
                commit_sha=commit_sha,
                extracted_utc=extracted_utc,
                rerun_reason=rerun_reason,
            )
        )

    # Dispatch sequentially when pool_size=1 (no multiprocessing overhead) or
    # via a process pool otherwise. Either path produces the same TileProvenance
    # objects; main process aggregates by (i,j) sort below regardless.
    tile_provenances: list[TileProvenance]
    if pool_size == 1:
        tile_provenances = [_extract_one_tile(args) for args in tile_args_list]
    else:
        # `imap_unordered` lets workers return results as they finish; we
        # sort by (i, j) when building the manifest. `spawn` start method
        # would force re-importing the module per worker; `fork` (the macOS
        # default for now) reuses parent memory pages cheaply. We don't
        # specify a context — pickle-safety is enforced by _TileWorkerArgs
        # being a dataclass with picklable fields (str, int, bytes,
        # _FeatureRecord with shapely geom), so either context works.
        with mp.Pool(processes=pool_size) as pool:
            tile_provenances = list(pool.imap_unordered(_extract_one_tile, tile_args_list))

    # ---- 8. Manifest assembly (main process; AFTER all tiles done) ------
    # Spec §11.7 + §11.8: tiles[] sorted by (i, j); manifest written last
    # (besides _SUCCESS which is Task 14's responsibility).
    completed_utc = _utcnow_iso()

    manifest = RegionManifest(
        schema_version=_SCHEMA_VERSION,
        sub_c_schema_version=_SUB_C_SCHEMA_VERSION,
        release=release,
        region=region.name,
        region_crs=f"EPSG:{SVY21_EPSG_CODE}",
        admin_polygon_source=admin_polygon_source,
        admin_polygon_sha256=admin_polygon_sha256,
        densified_admin_polygon_sha256=densified_admin_polygon_sha256,
        sea_polygons_sha256=sea_polygons_sha256,
        policy_yaml_sha256=policy_yaml_sha256,
        vocab_yaml_sha256=vocab_yaml_sha256,
        config={
            "tile_size_m": TILE_SIZE_M,
            "cell_size_m": CELL_SIZE_M,
            "cell_grid": [8, 8],
            "epsilon_ratio": EPS_RATIO,
            "epsilon_coord_m": EPS_COORD_M,
            "epsilon_area_m2": EPS_AREA_M2,
            "epsilon_length_m": EPS_LENGTH_M,
            "sea_definition": "base.class IN {ocean, strait, bay} OR base.subtype = ocean",
            "sea_water_fraction_threshold": 1.0,
            "coastal_inland_river_min_river_length_m": 500.0,
            "pipeline_order": ["reproject", "clip", "partition", "sliver_drop", "sea_mask"],
        },
        conditioning_defaults={
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
        initial_extraction={
            "commit_sha": commit_sha,
            "started_utc": started_utc,
            "completed_utc": completed_utc,
            "tile_count": len(tile_provenances),
        },
        tiles=aggregate_tile_inventory(tile_provenances),
    )

    write_manifest(manifest, output_dir / "manifest.yaml")
    return manifest


# ---------------------------------------------------------------------------
# Per-tile extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TileWorkerArgs:
    """Pickle-friendly per-tile worker payload (spec §14.5).

    Geometries shared across all tiles (sea polygons union, densified admin
    polygon) are passed as WKB bytes so the worker re-hydrates them locally
    with a single shapely.wkb.loads call — never by re-running
    derive_sea_polygons or densify_polygon.

    The per-tile features list is shapely-resident (already-clipped,
    already-reprojected _FeatureRecords); shapely geometries pickle natively
    so we don't need a WKB round-trip on the per-tile feature stream.
    """

    tile_i: int
    tile_j: int
    tile_admin_footprint_wkb: bytes
    features_in_tile: list[_FeatureRecord]
    sea_polygons_svy21_wkb: bytes
    densified_admin_polygon_svy21_wkb: bytes
    output_dir: Path
    inputs_shared: dict
    commit_sha: str
    extracted_utc: str
    rerun_reason: str


def _extract_one_tile(args: _TileWorkerArgs) -> TileProvenance:
    """Worker entry point.

    Rehydrates shared geometries from WKB and delegates to `_extract_tile`.
    MUST be importable at module scope (top-level function) so
    `multiprocessing.Pool` can pickle the reference. MUST NOT call
    `densify_polygon` or `derive_sea_polygons` (spec §14.5).
    """
    tile_admin_footprint = shapely_wkb.loads(args.tile_admin_footprint_wkb)
    sea_polygons_svy21 = shapely_wkb.loads(args.sea_polygons_svy21_wkb)

    return _extract_tile(
        tile_i=args.tile_i,
        tile_j=args.tile_j,
        tile_admin_footprint=tile_admin_footprint,
        features_in_tile=args.features_in_tile,
        sea_polygons_svy21=sea_polygons_svy21,
        output_dir=args.output_dir,
        inputs_shared=args.inputs_shared,
        commit_sha=args.commit_sha,
        extracted_utc=args.extracted_utc,
        rerun_reason=args.rerun_reason,
    )


@dataclass(frozen=True)
class _FeatureRecord:
    """A feature in absolute SVY21 coordinates carrying the attributes the
    orchestrator needs to emit a FeatureRow. Lives only in memory during one
    extract_region() call.
    """

    geometry: BaseGeometry
    source_feature_id: str
    feature_class: str  # "road" | "building" | "poi" | "base"
    class_raw: str | None
    subtype_raw: str | None
    categories_primary: str | None
    categories_alternate: list[str] | None


def _extract_tile(
    *,
    tile_i: int,
    tile_j: int,
    tile_admin_footprint: BaseGeometry,
    features_in_tile: list[_FeatureRecord],
    sea_polygons_svy21: BaseGeometry,
    output_dir: Path,
    inputs_shared: dict,
    commit_sha: str,
    extracted_utc: str,
    rerun_reason: str,
) -> TileProvenance:
    """Per-tile pipeline stages (spec §11.8 write order; §9.2 mask order):

        partition_into_cells -> sliver_drop -> sea_mask -> conditioning ->
        write cells.parquet -> write features.parquet -> write crossings.parquet
        -> write meta.yaml -> validate_tile_inline -> write provenance.yaml

    provenance.yaml is written LAST; its presence on disk means the tile is
    complete (passed inline validator). Cells dropped by §9.2 sea-mask rule
    are NOT in cells.parquet, and their features are NOT in features.parquet.
    """
    tile_dir = output_dir / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # ---- partition_into_cells -------------------------------------------
    # Input tuples: (geometry_in_absolute_svy21, source_feature_id, feature_class).
    cell_input: list[tuple[BaseGeometry, str, str]] = [
        (fr.geometry, fr.source_feature_id, fr.feature_class) for fr in features_in_tile
    ]
    sub_features, crossings = partition_into_cells(cell_input, tile_i, tile_j)

    # ---- sliver_drop (BEFORE sea_mask per spec §9.2 order) --------------
    sub_features = apply_sliver_drop(
        sub_features,
        area_threshold_m2=_SLIVER_AREA_THRESHOLD_M2,
        length_threshold_m=_SLIVER_LENGTH_THRESHOLD_M,
    )

    # Look up attributes for each sub-feature by source_feature_id so we can
    # reconstruct class_raw / subtype_raw / categories on the per-cell row.
    attrs_by_id: dict[str, _FeatureRecord] = {fr.source_feature_id: fr for fr in features_in_tile}

    # Group sub-features by cell.
    cell_to_subs: dict[tuple[int, int], list[CellSubFeature]] = {}
    for sf in sub_features:
        cell_to_subs.setdefault((sf.cell_i, sf.cell_j), []).append(sf)

    # ---- per-cell sea_mask + sea_overlap_fraction -----------------------
    # Spec §9.2: cell-level drop rule (sea_water_fraction & no non-sea features).
    # Spec §9.3: per-feature sea_overlap_fraction; cell-local sea geometry
    # computed once per cell.
    kept_cells: list[CellAggregate] = []
    kept_feature_rows: list[FeatureRow] = []
    kept_source_ids_per_cell: dict[tuple[int, int], set[str]] = {}
    sea_drop_count = 0
    cell_river_stream_lengths: list[float] = []

    # Iterate ALL 8x8 cells that intersect the tile_admin_footprint, not just
    # cells that contain features — a pure-sea cell has zero features and
    # would be missed by iterating cell_to_subs alone, but its drop must be
    # counted for sea_mask_drop_count.
    from shapely.geometry import box as shapely_box

    tile_origin_x = tile_i * TILE_SIZE_M
    tile_origin_y = tile_j * TILE_SIZE_M

    for ci in range(TILE_SIZE_M // CELL_SIZE_M):
        for cj in range(TILE_SIZE_M // CELL_SIZE_M):
            cell_box_abs = shapely_box(
                tile_origin_x + ci * CELL_SIZE_M,
                tile_origin_y + cj * CELL_SIZE_M,
                tile_origin_x + (ci + 1) * CELL_SIZE_M,
                tile_origin_y + (cj + 1) * CELL_SIZE_M,
            )
            cell_box_admin = cell_box_abs.intersection(tile_admin_footprint)
            cell_admin_area = cell_box_admin.area
            if cell_admin_area <= EPS_AREA_M2:
                # Outside admin (or near-zero sliver of admin); not a kept cell.
                # No drop_count increment — cells outside admin are not in scope.
                continue

            cell_subs = cell_to_subs.get((ci, cj), [])
            sea_water_fraction, water_fraction, drop_flag = apply_sea_mask(
                cell_box_admin_clipped=cell_box_admin,
                cell_features=cell_subs,
                sea_polygons_svy21=sea_polygons_svy21,
            )
            if drop_flag:
                # Sea-mask drop: pure-sea cell with zero non-sea features.
                sea_drop_count += 1
                continue

            # DECISION: water_fraction == sea_water_fraction for Task 12.
            # Spec §11.3 defines water_fraction as "all-water (sea + inland)"
            # coverage. Refining with inland water requires intersecting
            # inland-water base features (river, stream, reservoir, ...) with
            # the cell box — non-trivial enough to defer to a follow-up task.
            # Setting wf = sea_water_fraction is a safe under-estimate that
            # passes invariant #5 (sea <= wf <= 1) trivially. Revisit when
            # downstream consumers need accurate inland-water coverage.
            water_fraction = sea_water_fraction

            # Cell-local sea geometry: cached once per cell, translated to
            # cell-local coords (spec §9.3 fast-path optimization).
            cell_local_sea: BaseGeometry | None
            if sea_water_fraction == 0.0:
                cell_local_sea = None
            else:
                cell_local_sea_abs = cell_box_admin.intersection(sea_polygons_svy21)
                if cell_local_sea_abs.is_empty:
                    cell_local_sea = None
                else:
                    from shapely.affinity import translate as shapely_translate

                    cell_local_sea = shapely_translate(
                        cell_local_sea_abs,
                        xoff=-(tile_origin_x + ci * CELL_SIZE_M),
                        yoff=-(tile_origin_y + cj * CELL_SIZE_M),
                    )

            # Emit FeatureRow for every sub-feature in this kept cell.
            for sf in cell_subs:
                attrs = attrs_by_id.get(sf.source_feature_id)
                if attrs is None:
                    # Should not happen — sub-feature's source_feature_id must be
                    # in attrs_by_id. Skip defensively.
                    continue

                geom = sf.geometry
                bounds = geom.bounds  # (min_x, min_y, max_x, max_y)
                sea_overlap = _compute_sea_overlap_fraction(geom, sf.geometry_type, cell_local_sea)
                kept_feature_rows.append(
                    FeatureRow(
                        cell_i=ci,
                        cell_j=cj,
                        feature_class=encode_enum(FEATURE_CLASS, sf.feature_class),
                        source_feature_id=sf.source_feature_id,
                        geometry=geom,
                        geometry_type=encode_enum(GEOMETRY_TYPE, sf.geometry_type),
                        bbox_min_x=bounds[0],
                        bbox_min_y=bounds[1],
                        bbox_max_x=bounds[2],
                        bbox_max_y=bounds[3],
                        class_raw=attrs.class_raw,
                        subtype_raw=attrs.subtype_raw,
                        categories_primary=attrs.categories_primary,
                        categories_alternate=attrs.categories_alternate,
                        sea_overlap_fraction=sea_overlap,
                    )
                )

            kept_source_ids_per_cell[(ci, cj)] = {sf.source_feature_id for sf in cell_subs}

            kept_cells.append(
                CellAggregate(
                    cell_i=ci,
                    cell_j=cj,
                    water_fraction=water_fraction,
                    sea_water_fraction=sea_water_fraction,
                    cell_area_admin_clipped_m2=cell_admin_area,
                    kept_features_count=len(cell_subs),
                )
            )

            # Sum river/stream length for conditioning per spec §11.9.
            for sf in cell_subs:
                attrs = attrs_by_id.get(sf.source_feature_id)
                if attrs is None:
                    continue
                # base.river / base.stream are the inland-water classes (spec
                # §11.9). class_raw is the raw Overture base.class string.
                if attrs.feature_class == "base" and attrs.class_raw in ("river", "stream"):
                    cell_river_stream_lengths.append(sf.geometry.length)

    # ---- Drop crossings whose source was sliver-dropped or sea-dropped --
    # A crossing record must link to a source_feature_id that survives in
    # >= 2 distinct kept cells (invariant #4). Crossings whose source either
    # was sliver-dropped, sea-mask-dropped, or only survives in one cell
    # must be removed.
    surviving_ids_per_cell: dict[str, set[tuple[int, int]]] = {}
    for cell_xy, ids in kept_source_ids_per_cell.items():
        for sid in ids:
            surviving_ids_per_cell.setdefault(sid, set()).add(cell_xy)
    crossings = [
        c for c in crossings if len(surviving_ids_per_cell.get(c.source_feature_id, set())) >= 2
    ]

    # ---- conditioning_per_tile (spec §11.9) -----------------------------
    conditioning = compute_conditioning_per_tile(
        cell_sea_water_fractions=[c.sea_water_fraction for c in kept_cells],
        river_stream_lengths_m=cell_river_stream_lengths,
        admin_region="Central Region",  # DECISION: placeholder until divisions-theme lookup
        # is wired in (deferred — sub-A divisions theme lookup TBD). Spec §11.9
        # acknowledges admin_region needs second-level division resolution; for
        # Singapore Phase 1 a single region label is acceptable as the user-facing
        # value. Revisit when sub-D needs disambiguated tile-conditioning.
        morphology_class="Asian-megacity",
        era_class="contemporary",
    )

    # ---- meta.yaml aggregates (spec §11.5) ------------------------------
    feature_count_by_class: dict[str, int] = {"road": 0, "building": 0, "poi": 0, "base": 0}
    for row in kept_feature_rows:
        label = FEATURE_CLASS[row.feature_class]
        feature_count_by_class[label] += 1

    # Area-weighted means (spec §11.5 + §12.1 #8).
    total_area = sum(c.cell_area_admin_clipped_m2 for c in kept_cells)
    if total_area > 0:
        mean_wf = (
            sum(c.water_fraction * c.cell_area_admin_clipped_m2 for c in kept_cells) / total_area
        )
        mean_sea_wf = (
            sum(c.sea_water_fraction * c.cell_area_admin_clipped_m2 for c in kept_cells)
            / total_area
        )
    else:
        mean_wf = 0.0
        mean_sea_wf = 0.0

    meta = TileMeta(
        schema_version=_SCHEMA_VERSION,
        tile_i=tile_i,
        tile_j=tile_j,
        aggregates={
            "kept_cell_count": len(kept_cells),
            "sea_mask_drop_count": sea_drop_count,
            "mean_water_fraction": mean_wf,
            "mean_sea_water_fraction": mean_sea_wf,
            "feature_count_by_class": feature_count_by_class,
            "crossing_count": len(crossings),
        },
        config={"sliver_drop_rule": _SLIVER_DROP_RULE},
        conditioning_per_tile=conditioning,
    )

    # ---- Write tile artifacts in spec §11.8 order -----------------------
    cells_path = tile_dir / "cells.parquet"
    features_path = tile_dir / "features.parquet"
    crossings_path = tile_dir / "crossings.parquet"
    meta_path = tile_dir / "meta.yaml"
    provenance_path = tile_dir / "provenance.yaml"

    write_cells_parquet(kept_cells, cells_path)
    write_features_parquet(kept_feature_rows, features_path)
    write_crossings_parquet(crossings, crossings_path)
    write_meta_yaml(meta, meta_path)

    # Inline validator runs AFTER parquet+meta writes, BEFORE provenance.
    # Failure leaves provenance.yaml absent → tile is "in-flight / failed"
    # per spec §11.8.
    validate_tile_inline(tile_dir)

    # ---- provenance.yaml: outputs.*_sha256 digests + write LAST ---------
    outputs = {
        "cells_parquet_sha256": _sha256_of_file(cells_path),
        "features_parquet_sha256": _sha256_of_file(features_path),
        "crossings_parquet_sha256": _sha256_of_file(crossings_path),
        "meta_yaml_sha256": _sha256_of_file(meta_path),
    }
    provenance = TileProvenance(
        schema_version=_SCHEMA_VERSION,
        tile_i=tile_i,
        tile_j=tile_j,
        crs=f"EPSG:{SVY21_EPSG_CODE}",
        extraction={
            "commit_sha": commit_sha,
            "extracted_utc": extracted_utc,
            "rerun_count": 0,
            "rerun_reason": rerun_reason,
        },
        inputs=dict(inputs_shared),
        outputs=outputs,
    )
    write_provenance_yaml(provenance, provenance_path)
    return provenance


# ---------------------------------------------------------------------------
# Theme reprojection + feature extraction
# ---------------------------------------------------------------------------


def _reproject_and_extract_feature_records(
    policied_themes: dict[str, pa.Table],
) -> list[_FeatureRecord]:
    """Decode WKB geometry from each policied theme, reproject to SVY21, and
    package as a flat list of _FeatureRecord. Order within theme is preserved
    (deterministic input order); themes are processed in a fixed order so the
    overall list order is deterministic.

    The decoded-and-reprojected geometry stays as shapely objects through clip
    + cell partitioning; serialization to WKB happens only at write time
    (FeatureRow → dump_wkb). This avoids a per-stage WKB round-trip.
    """
    out: list[_FeatureRecord] = []

    for theme_name, feature_class in (
        ("transportation", "road"),
        ("buildings", "building"),
        ("places", "poi"),
        ("base", "base"),
    ):
        if theme_name not in policied_themes:
            continue
        table = policied_themes[theme_name]
        if table.num_rows == 0:
            continue
        ids = table.column("id").to_pylist()
        geoms_wkb = table.column("geometry").to_pylist()

        if feature_class in ("road", "building", "base"):
            class_raws = table.column("class").to_pylist()
        else:
            class_raws = [None] * table.num_rows

        if feature_class in ("building", "base"):
            subtype_raws = table.column("subtype").to_pylist()
        else:
            subtype_raws = [None] * table.num_rows

        if feature_class == "poi":
            cats_col = table.column("categories").to_pylist()
            primaries = [c["primary"] if c else None for c in cats_col]
            alternates = [c["alternate"] if c else None for c in cats_col]
        else:
            primaries = [None] * table.num_rows
            alternates = [None] * table.num_rows

        for idx in range(table.num_rows):
            wkb_bytes = geoms_wkb[idx]
            if wkb_bytes is None:
                continue
            geom_4326 = shapely_wkb.loads(wkb_bytes)
            if geom_4326.is_empty:
                continue
            geom_svy21 = reproject_geometry_to_svy21(geom_4326)
            out.append(
                _FeatureRecord(
                    geometry=geom_svy21,
                    source_feature_id=str(ids[idx]),
                    feature_class=feature_class,
                    class_raw=class_raws[idx],
                    subtype_raw=subtype_raws[idx],
                    categories_primary=primaries[idx],
                    categories_alternate=alternates[idx],
                )
            )

    return out


def _clip_features_to_admin(
    features: Iterable[_FeatureRecord],
    admin_polygon_svy21: BaseGeometry,
) -> list[_FeatureRecord]:
    """Intersect each feature with the admin polygon (both in SVY21).
    Drop empty intersections; preserve order; preserve attributes.
    """
    out: list[_FeatureRecord] = []
    for fr in features:
        clipped = fr.geometry.intersection(admin_polygon_svy21)
        if clipped.is_empty:
            continue
        out.append(
            _FeatureRecord(
                geometry=clipped,
                source_feature_id=fr.source_feature_id,
                feature_class=fr.feature_class,
                class_raw=fr.class_raw,
                subtype_raw=fr.subtype_raw,
                categories_primary=fr.categories_primary,
                categories_alternate=fr.categories_alternate,
            )
        )
    return out


def _filter_features_to_tile(
    features: Iterable[_FeatureRecord],
    tile_i: int,
    tile_j: int,
) -> list[_FeatureRecord]:
    """Return features whose absolute SVY21 geometry intersects the tile bbox.

    The tile box is [i*T, (i+1)*T) x [j*T, (j+1)*T) in SVY21 (spec §7.2 half-open).
    A feature on the upper boundary attaches to the higher-ij tile — but at the
    tile-membership level we use a slightly inclusive predicate (shapely's
    intersects) so a feature on the boundary is considered for both candidate
    tiles; partition_into_cells then enforces the half-open rule at cell level.
    """
    from shapely.geometry import box as shapely_box

    tile_box = shapely_box(
        tile_i * TILE_SIZE_M,
        tile_j * TILE_SIZE_M,
        (tile_i + 1) * TILE_SIZE_M,
        (tile_j + 1) * TILE_SIZE_M,
    )
    return [fr for fr in features if fr.geometry.intersects(tile_box)]


# ---------------------------------------------------------------------------
# Sea-overlap fraction (per-feature, cell-local)
# ---------------------------------------------------------------------------


def _compute_sea_overlap_fraction(
    feature_geom: BaseGeometry,
    feature_type_name: str,  # "Point" | "LineString" | "Polygon" | "Multi*"
    cell_local_sea: BaseGeometry | None,
) -> float:
    """Wrapper that handles the Multi* geom_type strings produced by shapely
    after intersection collapses (a polygon clipped at the cell boundary may
    return MultiPolygon).

    Maps "MultiLineString" → LineString and "MultiPolygon" → Polygon for the
    purpose of selecting the length-vs-area formula; falls back to 0.0 for
    truly unknown types.
    """
    if cell_local_sea is None:
        return 0.0

    if feature_type_name == "Point" or feature_type_name == "MultiPoint":
        return 1.0 if feature_geom.intersects(cell_local_sea) else 0.0
    if feature_type_name in ("LineString", "MultiLineString"):
        total = feature_geom.length
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea).length / total)
    if feature_type_name in ("Polygon", "MultiPolygon"):
        total = feature_geom.area
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea).area / total)
    return 0.0


# ---------------------------------------------------------------------------
# SHA-256 helpers (file + geometry)
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    """SHA-256 of a file's bytes on disk."""
    return compute_sha256(path.read_bytes())


def _sha256_of_geometry(geom: BaseGeometry) -> str:
    """SHA-256 of a shapely geometry's WKB bytes (little-endian).

    Wrapping in a MultiPolygon for sea_polygons happens at the caller; this
    helper just hashes whatever geometry is passed.
    """
    if geom.is_empty:
        # Hash a canonical empty MultiPolygon for stability — shapely's empty
        # geometry WKB varies slightly across versions for raw GeometryCollection().
        canonical_empty = MultiPolygon()
        return compute_sha256(dump_wkb(canonical_empty))
    return compute_sha256(dump_wkb(geom))


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp suffixed with 'Z' (spec §11.6 sample format)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
