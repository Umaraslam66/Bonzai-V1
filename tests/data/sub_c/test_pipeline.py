"""Task 12 tests: extract_region — sequential pipeline orchestrator.

Named tests per plan Task 12:
- test_pipeline_runs_derive_sea_polygons_before_apply_missing_value_policy
- test_pipeline_reproject_runs_before_clip
- test_pipeline_clip_runs_before_partition_into_tiles
- test_pipeline_sliver_drop_runs_before_sea_mask
- test_pipeline_extract_tile_produces_complete_directory_artifacts

Plus complementary direct-API tests:
- test_extract_region_emits_full_tile_directory_for_hand_built_region
- test_extract_region_manifest_aggregates_all_tiles_in_sorted_order
- test_extract_region_byte_deterministic_modulo_excluded_fields
- test_extract_region_inline_validator_passes_per_tile
- test_extract_region_uses_pre_policy_sea_polygons
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, Point, Polygon

from cfm.data.sub_c.determinism import compute_sha256_excluding
from cfm.data.sub_c.manifest import RegionManifest
from cfm.data.sub_c.pipeline import extract_region

# ---------------------------------------------------------------------------
# Synthetic Region builder
# ---------------------------------------------------------------------------
#
# Admin polygon is a small 4326 box in SG-area lonlat. In SVY21 it spans:
#   easting:  ~28743 .. ~33807  (tiles i=14, 15, 16)
#   northing: ~31372 .. ~32976  (tiles j=15, 16)
# Total up to 6 tiles; typically 4-6 depending on admin trim.
# Repo's existing configs/data/missing_value_policy.yaml and
# configs/tokenizer/vocab_phase1.yaml are used directly.

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"

# Admin polygon corners (4326). 6km E-W by 1.5km N-S near (103.84, 1.30).
_ADMIN_LON_MIN: float = 103.84
_ADMIN_LON_MAX: float = 103.8855  # 0.0455° lon ≈ 5km at SG latitude
_ADMIN_LAT_MIN: float = 1.30
_ADMIN_LAT_MAX: float = 1.3145  # 0.0145° lat ≈ 1.6km


def _admin_polygon_4326() -> Polygon:
    return Polygon(
        [
            (_ADMIN_LON_MIN, _ADMIN_LAT_MIN),
            (_ADMIN_LON_MAX, _ADMIN_LAT_MIN),
            (_ADMIN_LON_MAX, _ADMIN_LAT_MAX),
            (_ADMIN_LON_MIN, _ADMIN_LAT_MAX),
        ]
    )


def _wkb(geom) -> bytes:
    """Serialize geometry to WKB matching Overture's binary column convention."""
    return shapely_wkb.dumps(geom, hex=False, byte_order=1)


def _make_buildings_table(rows: list[dict]) -> pa.Table:
    """Minimal buildings.parquet schema subset matching apply_missing_value_policy.

    apply_missing_value_policy only reads the "class" column on buildings;
    the orchestrator additionally reads "id", "geometry", "subtype". Other
    columns are not needed for Task 12 tests.
    """
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "class": [r.get("class") for r in rows],
            "subtype": [r.get("subtype") for r in rows],
            "geometry": [_wkb(r["geometry"]) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("subtype", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_transportation_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "class": [r.get("class") for r in rows],
            "geometry": [_wkb(r["geometry"]) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_base_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "class": [r.get("class") for r in rows],
            "subtype": [r.get("subtype") for r in rows],
            "geometry": [_wkb(r["geometry"]) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("class", pa.string()),
                pa.field("subtype", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_places_table(rows: list[dict]) -> pa.Table:
    cats_type = pa.struct(
        [
            pa.field("primary", pa.string()),
            pa.field("alternate", pa.list_(pa.string())),
        ]
    )
    return pa.table(
        {
            "id": [r["id"] for r in rows],
            "categories": [
                {
                    "primary": r.get("primary"),
                    "alternate": r.get("alternate", []),
                }
                for r in rows
            ],
            "geometry": [_wkb(r["geometry"]) for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("categories", cats_type),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )


def _make_default_region() -> SimpleNamespace:
    """Build a SimpleNamespace satisfying the orchestrator's _RegionLike protocol.

    Themes contain a handful of features positioned to ensure every tile
    receives at least one feature (so kept_cell_count > 0 and the inline
    validator passes the kept_cell_rule on every tile).
    """
    admin = _admin_polygon_4326()

    # Pepper buildings across the admin region. Tiny polygons (~ a few m wide
    # in 4326; in SVY21 these are ~50m-wide squares that easily exceed sliver
    # area threshold of 0.01 m²).
    buildings = []
    for k in range(12):
        lx = _ADMIN_LON_MIN + 0.003 + (k * 0.004) % 0.040
        ly = _ADMIN_LAT_MIN + 0.002 + (k * 0.0023) % 0.012
        poly = Polygon(
            [
                (lx, ly),
                (lx + 0.0003, ly),
                (lx + 0.0003, ly + 0.0003),
                (lx, ly + 0.0003),
            ]
        )
        buildings.append(
            {
                "id": f"bldg_{k:03d}",
                "class": "residential",
                "subtype": "residential",
                "geometry": poly,
            }
        )

    # A road as a horizontal LineString crossing all tiles.
    roads = [
        {
            "id": "road_001",
            "class": "primary",
            "geometry": LineString(
                [
                    (_ADMIN_LON_MIN + 0.001, _ADMIN_LAT_MIN + 0.007),
                    (_ADMIN_LON_MAX - 0.001, _ADMIN_LAT_MIN + 0.007),
                ]
            ),
        },
    ]

    # Sprinkle some POIs.
    pois = []
    for k in range(6):
        lon = _ADMIN_LON_MIN + 0.005 + k * 0.008
        lat = _ADMIN_LAT_MIN + 0.005
        pois.append(
            {
                "id": f"poi_{k:03d}",
                "primary": "restaurant",
                "alternate": [],
                "geometry": Point(lon, lat),
            }
        )

    # base: a couple of inland-water features (no ocean/strait/bay; those are
    # added in the sea-polygon test only).
    base_rows = [
        {
            "id": "base_river_001",
            "class": "river",
            "subtype": "water",
            "geometry": LineString(
                [
                    (_ADMIN_LON_MIN + 0.002, _ADMIN_LAT_MIN + 0.011),
                    (_ADMIN_LON_MAX - 0.002, _ADMIN_LAT_MIN + 0.011),
                ]
            ),
        },
    ]

    themes = {
        "buildings": _make_buildings_table(buildings),
        "transportation": _make_transportation_table(roads),
        "places": _make_places_table(pois),
        "base": _make_base_table(base_rows),
    }

    region = SimpleNamespace(
        name="synthetic_region",
        themes=themes,
        admin_polygon=admin,
        geometry=SimpleNamespace(admin_polygon=admin),
        projected_crs="EPSG:3414",
    )
    return region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pa.Table:
    """Read a parquet file without Hive partition column inference.

    Saved memory `feedback_pyarrow_hive_partition_inference.md`: tile dirs
    are named `tile=EPSG3414_iN_jN`, which pyarrow would otherwise read as a
    Hive partition. ParquetFile(path).read() bypasses that.
    """
    return pq.ParquetFile(path).read()


def _extract_default(output_dir: Path, **kwargs) -> RegionManifest:
    """Run extract_region with the default synthetic region + fixed timestamps."""
    region = _make_default_region()
    return extract_region(
        region,
        output_dir,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha="b86c509" + "0" * 33,  # 40-char canonical
        extracted_utc="2026-05-18T00:00:00Z",
        started_utc="2026-05-18T00:00:00Z",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_region_emits_full_tile_directory_for_hand_built_region(tmp_path: Path) -> None:
    """For every tile in the inventory: cells.parquet + features.parquet +
    crossings.parquet + meta.yaml + provenance.yaml must exist + manifest.yaml.
    """
    out = tmp_path / "region"
    manifest = _extract_default(out)

    # At least one tile must be produced from the synthetic region.
    assert len(manifest.tiles) >= 1, "synthetic region should produce >= 1 tile"

    # manifest.yaml present
    assert (out / "manifest.yaml").exists()

    # Every tile in the manifest has all 5 artifacts on disk
    for tile in manifest.tiles:
        tile_dir = out / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        assert tile_dir.is_dir(), f"tile_dir missing: {tile_dir}"
        for artifact in (
            "cells.parquet",
            "features.parquet",
            "crossings.parquet",
            "meta.yaml",
            "provenance.yaml",
        ):
            path = tile_dir / artifact
            assert path.exists(), f"missing artifact {path}"

    # _SUCCESS is NOT written by extract_region (Task 14 writes it)
    assert not (out / "_SUCCESS").exists(), (
        "_SUCCESS must not be written by extract_region (Task 14's responsibility)"
    )


def test_extract_region_manifest_aggregates_all_tiles_in_sorted_order(tmp_path: Path) -> None:
    """manifest.tiles[] must be sorted by (tile_i, tile_j) for byte-determinism."""
    out = tmp_path / "region"
    manifest = _extract_default(out)

    keys = [(t["tile_i"], t["tile_j"]) for t in manifest.tiles]
    assert keys == sorted(keys), f"tiles[] not sorted by (i,j): {keys}"

    # Provenance shas must be 64-hex strings (SHA-256)
    for tile in manifest.tiles:
        sha = tile["provenance_sha256"]
        assert isinstance(sha, str) and len(sha) == 64, sha


def test_extract_region_byte_deterministic_modulo_excluded_fields(tmp_path: Path) -> None:
    """Two runs with identical inputs + fixed extracted_utc produce:
    - byte-identical per-tile parquet files
    - byte-identical manifest content shas (computed via compute_sha256_excluding)
    """
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    manifest_a = _extract_default(out_a)
    manifest_b = _extract_default(out_b)

    assert len(manifest_a.tiles) == len(manifest_b.tiles)

    # Per-tile parquet bytes identical
    for tile in manifest_a.tiles:
        rel = f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        for parquet in ("cells.parquet", "features.parquet", "crossings.parquet"):
            bytes_a = (out_a / rel / parquet).read_bytes()
            bytes_b = (out_b / rel / parquet).read_bytes()
            assert bytes_a == bytes_b, f"parquet bytes differ: {rel}/{parquet}"

    # Manifest content shas identical (timestamps and *_sha256 fields are
    # already excluded by compute_sha256_excluding via EXCLUDED_FROM_SHA).
    manifest_a_dict = yaml.safe_load((out_a / "manifest.yaml").read_text())
    manifest_b_dict = yaml.safe_load((out_b / "manifest.yaml").read_text())
    sha_a = compute_sha256_excluding(manifest_a_dict, "manifest.yaml")
    sha_b = compute_sha256_excluding(manifest_b_dict, "manifest.yaml")
    assert sha_a == sha_b, "manifest content shas (modulo EXCLUDED_FROM_SHA) differ"

    # Per-tile provenance content shas identical too
    for tile in manifest_a.tiles:
        rel = f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        prov_a = yaml.safe_load((out_a / rel / "provenance.yaml").read_text())
        prov_b = yaml.safe_load((out_b / rel / "provenance.yaml").read_text())
        assert compute_sha256_excluding(prov_a, "provenance.yaml") == compute_sha256_excluding(
            prov_b, "provenance.yaml"
        ), f"provenance content shas differ for {rel}"


def test_extract_region_inline_validator_passes_per_tile(tmp_path: Path) -> None:
    """No TileValidationError raised on the synthetic clean tile.

    extract_region calls validate_tile_inline for every tile BEFORE writing
    provenance.yaml. If validation failed, provenance.yaml would be missing
    AND the exception would propagate. We assert both: provenance.yaml exists
    everywhere, AND no exception was raised (asserted by extract_region
    returning normally).
    """
    out = tmp_path / "region"
    manifest = _extract_default(out)  # would raise on validator failure

    for tile in manifest.tiles:
        prov_path = out / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}" / "provenance.yaml"
        # Presence = tile complete (per spec §11.8). Pre-validator failure
        # leaves provenance.yaml absent.
        assert prov_path.exists()


def test_extract_region_uses_pre_policy_sea_polygons(tmp_path: Path) -> None:
    """Add an ocean base row + an inland-water (river) base row. Verify:
    - The ocean polygon contributes to sea_polygons_sha256 (i.e., changes the sha).
    - The ocean row does NOT appear in any tile's features.parquet (policy drops it
      from feature emission; sea polygons are masks, not features).
    - The inland-water (river) row remains in features.parquet (policy keeps it).
    """
    # Build a region with an ocean polygon to the SOUTH of admin (it won't end
    # up as a feature, but its existence affects derive_sea_polygons + sha).
    region_no_ocean = _make_default_region()

    region_with_ocean = _make_default_region()

    # Add a small ocean polygon that does NOT overlap admin (so its addition
    # only affects sea_polygons_sha256, not the per-tile feature stream).
    ocean_geom = Polygon(
        [
            (_ADMIN_LON_MIN + 0.001, _ADMIN_LAT_MIN - 0.005),
            (_ADMIN_LON_MIN + 0.005, _ADMIN_LAT_MIN - 0.005),
            (_ADMIN_LON_MIN + 0.005, _ADMIN_LAT_MIN - 0.001),
            (_ADMIN_LON_MIN + 0.001, _ADMIN_LAT_MIN - 0.001),
        ]
    )
    base_existing = region_with_ocean.themes["base"]
    base_with_ocean = pa.concat_tables(
        [
            base_existing,
            _make_base_table(
                [
                    {
                        "id": "base_ocean_001",
                        "class": "ocean",
                        "subtype": "ocean",
                        "geometry": ocean_geom,
                    }
                ]
            ),
        ]
    )
    region_with_ocean.themes["base"] = base_with_ocean

    out_no = tmp_path / "no_ocean"
    out_yes = tmp_path / "with_ocean"

    manifest_no = extract_region(
        region_no_ocean,
        out_no,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha="b86c509" + "0" * 33,
        extracted_utc="2026-05-18T00:00:00Z",
        started_utc="2026-05-18T00:00:00Z",
    )
    manifest_yes = extract_region(
        region_with_ocean,
        out_yes,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha="b86c509" + "0" * 33,
        extracted_utc="2026-05-18T00:00:00Z",
        started_utc="2026-05-18T00:00:00Z",
    )

    # sea_polygons_sha256 must DIFFER (adding an ocean row changes the
    # derived pre-policy sea-polygon view).
    assert manifest_no.sea_polygons_sha256 != manifest_yes.sea_polygons_sha256, (
        "adding an ocean base row must change sea_polygons_sha256"
    )

    # ocean row must NOT appear in any tile's features.parquet — apply_missing_value_policy
    # drops sea-defining base rows (class IN {ocean, strait, bay}) per spec §10.2.
    ocean_in_features = False
    for tile in manifest_yes.tiles:
        rel = f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        features = _read_parquet(out_yes / rel / "features.parquet")
        ids = features.column("source_feature_id").to_pylist()
        if "base_ocean_001" in ids:
            ocean_in_features = True
            break
    assert not ocean_in_features, (
        "ocean polygon must not appear in features.parquet (policy drop_row applied)"
    )

    # The pre-existing inland-water river feature should still be present
    # somewhere (policy keeps it; it's in vocab as BASE_river per
    # configs/tokenizer/vocab_phase1.yaml).
    river_in_features = False
    for tile in manifest_yes.tiles:
        rel = f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        features = _read_parquet(out_yes / rel / "features.parquet")
        ids = features.column("source_feature_id").to_pylist()
        if "base_river_001" in ids:
            river_in_features = True
            break
    assert river_in_features, "inland-water river feature must remain in features.parquet"


# ---------------------------------------------------------------------------
# Pipeline-ordering tests (call-order assertions via monkeypatch)
#
# These are the plan-Task-12 named tests verifying the locked spec §6
# pipeline order. Each test patches the relevant library function to record
# its call moment, then asserts the relative order via a shared call-log list.
# ---------------------------------------------------------------------------


def test_pipeline_runs_derive_sea_polygons_before_apply_missing_value_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §9.1 + §6: derive_sea_polygons MUST run on raw themes BEFORE
    apply_missing_value_policy. Reversed order would yield an empty sea-polygon
    set (the base.class not_in_vocab drop would have removed ocean/strait/bay).
    """
    calls: list[str] = []

    from cfm.data.sub_c import pipeline as pipeline_mod

    real_derive = pipeline_mod.derive_sea_polygons
    real_policy = pipeline_mod.apply_missing_value_policy

    def derive_spy(*args, **kw):
        calls.append("derive_sea_polygons")
        return real_derive(*args, **kw)

    def policy_spy(*args, **kw):
        calls.append("apply_missing_value_policy")
        return real_policy(*args, **kw)

    monkeypatch.setattr(pipeline_mod, "derive_sea_polygons", derive_spy)
    monkeypatch.setattr(pipeline_mod, "apply_missing_value_policy", policy_spy)

    _extract_default(tmp_path / "region")

    assert calls.index("derive_sea_polygons") < calls.index("apply_missing_value_policy"), (
        f"derive_sea_polygons must run BEFORE apply_missing_value_policy; got order {calls}"
    )


def test_pipeline_reproject_runs_before_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §7.3: reproject themes/admin to SVY21 FIRST, then clip in SVY21.

    We spy on partition_into_tiles (which receives the SVY21 admin polygon)
    and assert it runs only after the reprojection helper has been called at
    least once. _clip_features_to_admin is internal; we use partition as the
    public next-stage proxy.
    """
    calls: list[str] = []

    from cfm.data.sub_c import pipeline as pipeline_mod
    from cfm.data.sub_c.coords import RegionCoords

    # Reprojection now flows through RegionCoords.reproject_geometry (region-bound),
    # not a module-level pipeline function — spy on the method.
    real_reproject = RegionCoords.reproject_geometry
    real_partition = pipeline_mod.partition_into_tiles
    real_clip = pipeline_mod._clip_features_to_admin

    def reproject_spy(self, geom):
        calls.append("reproject")
        return real_reproject(self, geom)

    def clip_spy(*args, **kw):
        calls.append("clip")
        return real_clip(*args, **kw)

    def partition_spy(*args, **kw):
        calls.append("partition_into_tiles")
        return real_partition(*args, **kw)

    monkeypatch.setattr(RegionCoords, "reproject_geometry", reproject_spy)
    monkeypatch.setattr(pipeline_mod, "_clip_features_to_admin", clip_spy)
    monkeypatch.setattr(pipeline_mod, "partition_into_tiles", partition_spy)

    _extract_default(tmp_path / "region")

    # At least one reproject call before the first clip call.
    first_reproject = calls.index("reproject")
    first_clip = calls.index("clip")
    assert first_reproject < first_clip, f"reproject must run BEFORE clip; got order {calls[:10]}"


def test_pipeline_clip_runs_before_partition_into_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §7.3 + §7.2: clip to admin BEFORE partitioning into tiles."""
    calls: list[str] = []

    from cfm.data.sub_c import pipeline as pipeline_mod

    real_clip = pipeline_mod._clip_features_to_admin
    real_partition = pipeline_mod.partition_into_tiles

    def clip_spy(*args, **kw):
        calls.append("clip")
        return real_clip(*args, **kw)

    def partition_spy(*args, **kw):
        calls.append("partition_into_tiles")
        return real_partition(*args, **kw)

    monkeypatch.setattr(pipeline_mod, "_clip_features_to_admin", clip_spy)
    monkeypatch.setattr(pipeline_mod, "partition_into_tiles", partition_spy)

    _extract_default(tmp_path / "region")

    assert calls.index("clip") < calls.index("partition_into_tiles"), (
        f"clip must run BEFORE partition_into_tiles; got order {calls}"
    )


def test_pipeline_sliver_drop_runs_before_sea_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §9.2 explicit pipeline order: sliver_drop → sea_mask.

    Rationale (spec §9.2 verbatim): "sliver-drop runs first so a 1 cm road
    sliver crossing into a sea cell is removed before the sea-mask runs,
    leaving the cell genuinely feature-empty and droppable."
    """
    calls: list[str] = []

    from cfm.data.sub_c import pipeline as pipeline_mod

    real_sliver = pipeline_mod.apply_sliver_drop
    real_sea = pipeline_mod.apply_sea_mask

    def sliver_spy(*args, **kw):
        calls.append("apply_sliver_drop")
        return real_sliver(*args, **kw)

    def sea_spy(*args, **kw):
        calls.append("apply_sea_mask")
        return real_sea(*args, **kw)

    monkeypatch.setattr(pipeline_mod, "apply_sliver_drop", sliver_spy)
    monkeypatch.setattr(pipeline_mod, "apply_sea_mask", sea_spy)

    _extract_default(tmp_path / "region")

    # For every tile, sliver_drop must precede the first sea_mask call. The
    # synthetic region has multiple tiles; assert the FIRST occurrence of each
    # interleaving keeps the order.
    first_sliver = calls.index("apply_sliver_drop")
    first_sea = calls.index("apply_sea_mask")
    assert first_sliver < first_sea, f"sliver_drop must run BEFORE sea_mask; got order {calls[:10]}"


def test_pipeline_extract_tile_produces_complete_directory_artifacts(tmp_path: Path) -> None:
    """Per-tile dir contains all 5 artifacts (cells, features, crossings,
    meta, provenance) AND each meta.yaml has well-formed aggregates/config/
    conditioning_per_tile sections (spec §11.5).
    """
    out = tmp_path / "region"
    manifest = _extract_default(out)

    for tile in manifest.tiles:
        tile_dir = out / f"tile=EPSG3414_i{tile['tile_i']}_j{tile['tile_j']}"
        meta = yaml.safe_load((tile_dir / "meta.yaml").read_text())

        assert meta["schema_version"] == "1.1"  # bumped in v1.1 for Multi* enum extension
        assert meta["tile_i"] == tile["tile_i"]
        assert meta["tile_j"] == tile["tile_j"]

        # aggregates block
        agg = meta["aggregates"]
        for key in (
            "kept_cell_count",
            "sea_mask_drop_count",
            "mean_water_fraction",
            "mean_sea_water_fraction",
            "feature_count_by_class",
            "crossing_count",
        ):
            assert key in agg, f"missing aggregates.{key} in {tile_dir}/meta.yaml"

        # feature_count_by_class uses string labels per Task 12 spec note
        fc = agg["feature_count_by_class"]
        assert set(fc.keys()) == {"road", "building", "poi", "base"}

        # config + conditioning_per_tile
        assert "sliver_drop_rule" in meta["config"]
        for key in (
            "admin_region",
            "morphology_class",
            "era_class",
            "coastal_inland_river",
            "population_density_bucket",
            "population_density_bucket_owner",
        ):
            assert key in meta["conditioning_per_tile"]

        # provenance.yaml structure
        prov = yaml.safe_load((tile_dir / "provenance.yaml").read_text())
        assert prov["tile_i"] == tile["tile_i"]
        assert prov["tile_j"] == tile["tile_j"]
        assert prov["crs"] == "EPSG:3414"
        for key in ("commit_sha", "extracted_utc", "rerun_count", "rerun_reason"):
            assert key in prov["extraction"]
        for key in ("release", "admin_polygon_sha256", "policy_yaml_sha256", "vocab_yaml_sha256"):
            assert key in prov["inputs"]
        for key in (
            "cells_parquet_sha256",
            "features_parquet_sha256",
            "crossings_parquet_sha256",
            "meta_yaml_sha256",
        ):
            assert key in prov["outputs"]
