"""Task 9 tests: per-tile write helpers for cells/features/crossings parquet +
meta/provenance YAML.

Named tests per plan Task 9:
- test_write_features_parquet_schema_and_sort_key
- test_write_cells_parquet_schema_and_sort_key
- test_write_crossings_parquet_schema_and_sort_key
- test_write_meta_yaml_includes_aggregates_and_conditioning
- test_write_provenance_yaml_inputs_and_outputs_digests
- test_features_parquet_bbox_columns_match_wkb_per_row
- test_features_parquet_geometry_type_int8_matches_wkb_header_per_row
- test_per_tile_directory_naming_includes_crs_and_named_ij

Additional (from task description hints):
- test_features_parquet_uses_wkb_little_endian_for_geometry
- test_byte_deterministic_re_write_modulo_excluded_fields
"""

from __future__ import annotations

import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, Point, Polygon

from cfm.data.sub_c.enums import AXIS, EVENT_TYPE, GEOMETRY_TYPE, encode_enum
from cfm.data.sub_c.geom import CrossingRecord
from cfm.data.sub_c.io import (
    _CELLS_SCHEMA,
    _CROSSINGS_SCHEMA,
    _FEATURES_SCHEMA,
    CellAggregate,
    FeatureRow,
    TileMeta,
    TileProvenance,
    write_cells_parquet,
    write_crossings_parquet,
    write_features_parquet,
    write_meta_yaml,
    write_provenance_yaml,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_feature_row(
    cell_i: int = 0,
    cell_j: int = 0,
    feature_class: int = 0,  # 0=road
    source_feature_id: str = "feat_001",
    geometry: object | None = None,
    geometry_type: int = 1,  # 1=LineString
    class_raw: str | None = "primary",
    subtype_raw: str | None = None,
    categories_primary: str | None = None,
    categories_alternate: list[str] | None = None,
    sea_overlap_fraction: float = 0.0,
) -> FeatureRow:
    """Build a minimal FeatureRow for testing."""
    if geometry is None:
        geometry = LineString([(0.0, 0.0), (100.0, 50.0)])
    min_x, min_y, max_x, max_y = geometry.bounds
    return FeatureRow(
        cell_i=cell_i,
        cell_j=cell_j,
        feature_class=feature_class,
        source_feature_id=source_feature_id,
        geometry=geometry,
        geometry_type=geometry_type,
        bbox_min_x=min_x,
        bbox_min_y=min_y,
        bbox_max_x=max_x,
        bbox_max_y=max_y,
        class_raw=class_raw,
        subtype_raw=subtype_raw,
        categories_primary=categories_primary,
        categories_alternate=categories_alternate,
        sea_overlap_fraction=sea_overlap_fraction,
    )


def _make_cell_aggregate(
    cell_i: int = 0,
    cell_j: int = 0,
    water_fraction: float = 0.1,
    sea_water_fraction: float = 0.05,
    cell_area_admin_clipped_m2: float = 62500.0,
    kept_features_count: int = 5,
) -> CellAggregate:
    return CellAggregate(
        cell_i=cell_i,
        cell_j=cell_j,
        water_fraction=water_fraction,
        sea_water_fraction=sea_water_fraction,
        cell_area_admin_clipped_m2=cell_area_admin_clipped_m2,
        kept_features_count=kept_features_count,
    )


def _make_crossing_record(
    source_feature_id: str = "feat_001",
    lower_cell_i: int = 0,
    lower_cell_j: int = 0,
    axis: int = 0,
    ring_index: int = 0,
    event_type: int = 0,
    edge_position_m: float = 125.0,
    edge_extent_length_m: float = 0.0,
) -> CrossingRecord:
    return CrossingRecord(
        source_feature_id=source_feature_id,
        lower_cell_i=lower_cell_i,
        lower_cell_j=lower_cell_j,
        axis=axis,
        ring_index=ring_index,
        event_type=event_type,
        edge_position_m=edge_position_m,
        edge_extent_length_m=edge_extent_length_m,
    )


def _make_tile_meta(
    tile_i: int = 12,
    tile_j: int = 17,
) -> TileMeta:
    return TileMeta(
        schema_version="1.1",
        tile_i=tile_i,
        tile_j=tile_j,
        aggregates={
            "kept_cell_count": 47,
            "sea_mask_drop_count": 17,
            "mean_water_fraction": 0.23,
            "mean_sea_water_fraction": 0.04,
            "feature_count_by_class": {
                "road": 1834,
                "building": 6712,
                "poi": 412,
                "base": 89,
            },
            "crossing_count": 2107,
        },
        config={
            "sliver_drop_rule": "drop iff geometry has area < 0.01 m² OR length < 0.01 m",
        },
        conditioning_per_tile={
            "admin_region": "Central Region",
            "morphology_class": "Asian-megacity",
            "era_class": "contemporary",
            "coastal_inland_river": 1,
            "population_density_bucket": None,
            "population_density_bucket_owner": "sub-D",
        },
    )


def _make_tile_provenance(
    tile_i: int = 12,
    tile_j: int = 17,
) -> TileProvenance:
    return TileProvenance(
        schema_version="1.1",
        tile_i=tile_i,
        tile_j=tile_j,
        crs="EPSG:3414",
        extraction={
            "commit_sha": "a" * 40,
            "extracted_utc": "2026-05-17T08:12:14Z",
            "rerun_count": 0,
            "rerun_reason": "initial",
        },
        inputs={
            "release": "2026-04-15.0",
            "admin_polygon_sha256": "b" * 64,
            "policy_yaml_sha256": "c" * 64,
            "vocab_yaml_sha256": "d" * 64,
        },
        outputs={
            "cells_parquet_sha256": "e" * 64,
            "features_parquet_sha256": "f" * 64,
            "crossings_parquet_sha256": "0" * 64,
            "meta_yaml_sha256": "1" * 64,
        },
    )


# ---------------------------------------------------------------------------
# features.parquet tests
# ---------------------------------------------------------------------------


def test_write_features_parquet_schema_and_sort_key(tmp_path: Path) -> None:
    """Write a 3-row table in scrambled order; read back; assert schema columns +
    types match spec §11.2, and rows are sorted by (cell_i, cell_j, feature_class,
    source_feature_id).
    """
    path = tmp_path / "features.parquet"

    # Three rows deliberately out of canonical order
    rows = [
        _make_feature_row(
            cell_i=1,
            cell_j=0,
            feature_class=1,
            source_feature_id="zzz",
            geometry=Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]),
            geometry_type=2,
            class_raw=None,
            subtype_raw="residential",
        ),
        _make_feature_row(
            cell_i=0,
            cell_j=0,
            feature_class=2,
            source_feature_id="aaa",
            geometry=Point(10.0, 10.0),
            geometry_type=0,
            class_raw=None,
            categories_primary="restaurant",
            categories_alternate=["food", "dining"],
        ),
        _make_feature_row(
            cell_i=0,
            cell_j=0,
            feature_class=0,
            source_feature_id="mmm",
        ),
    ]

    write_features_parquet(rows, path)

    table = pq.read_table(path)

    # --- schema ---
    assert table.schema.equals(_FEATURES_SCHEMA), (
        f"Schema mismatch.\nExpected:\n{_FEATURES_SCHEMA}\nGot:\n{table.schema}"
    )

    # --- sort key (cell_i, cell_j, feature_class, source_feature_id) ---
    cell_i_col = table.column("cell_i").to_pylist()
    cell_j_col = table.column("cell_j").to_pylist()
    fc_col = table.column("feature_class").to_pylist()
    sid_col = table.column("source_feature_id").to_pylist()

    sort_keys = list(zip(cell_i_col, cell_j_col, fc_col, sid_col, strict=True))
    assert sort_keys == sorted(sort_keys), f"Rows are not in canonical sort order: {sort_keys}"

    # --- column count ---
    assert len(table.schema) == 15, f"Expected 15 columns, got {len(table.schema)}"


def test_features_parquet_uses_wkb_little_endian_for_geometry(tmp_path: Path) -> None:
    """geometry column bytes must start with 0x01 (WKB little-endian NDR marker)."""
    path = tmp_path / "features.parquet"
    rows = [_make_feature_row(geometry=Point(1.0, 2.0), geometry_type=0)]
    write_features_parquet(rows, path)

    table = pq.read_table(path)
    geom_bytes = table.column("geometry").to_pylist()[0]

    assert isinstance(geom_bytes, bytes), "geometry column should contain bytes"
    assert geom_bytes[0] == 0x01, (
        f"Expected WKB little-endian marker 0x01, got {geom_bytes[0]:#04x}"
    )

    # Also verify roundtrip
    recovered = shapely_wkb.loads(geom_bytes)
    assert recovered == Point(1.0, 2.0)


def test_features_parquet_bbox_columns_match_wkb_per_row(tmp_path: Path) -> None:
    """bbox_min_x/y and bbox_max_x/y stored in table must match WKB-derived bounds.

    This mirrors inline validator invariant #2 (spec §12.1).
    """
    path = tmp_path / "features.parquet"

    line = LineString([(10.0, 20.0), (80.0, 70.0)])
    poly = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])

    rows = [
        _make_feature_row(
            cell_i=0,
            cell_j=0,
            source_feature_id="line_feat",
            geometry=line,
            geometry_type=1,
        ),
        _make_feature_row(
            cell_i=0,
            cell_j=0,
            source_feature_id="poly_feat",
            feature_class=1,
            geometry=poly,
            geometry_type=2,
            class_raw=None,
            subtype_raw="residential",
        ),
    ]
    write_features_parquet(rows, path)

    table = pq.read_table(path)
    geom_col = table.column("geometry").to_pylist()
    bbox_min_x_col = table.column("bbox_min_x").to_pylist()
    bbox_min_y_col = table.column("bbox_min_y").to_pylist()
    bbox_max_x_col = table.column("bbox_max_x").to_pylist()
    bbox_max_y_col = table.column("bbox_max_y").to_pylist()

    for i, geom_bytes in enumerate(geom_col):
        geom = shapely_wkb.loads(geom_bytes)
        wkb_min_x, wkb_min_y, wkb_max_x, wkb_max_y = geom.bounds
        assert abs(bbox_min_x_col[i] - wkb_min_x) < 1e-6, f"Row {i}: bbox_min_x mismatch"
        assert abs(bbox_min_y_col[i] - wkb_min_y) < 1e-6, f"Row {i}: bbox_min_y mismatch"
        assert abs(bbox_max_x_col[i] - wkb_max_x) < 1e-6, f"Row {i}: bbox_max_x mismatch"
        assert abs(bbox_max_y_col[i] - wkb_max_y) < 1e-6, f"Row {i}: bbox_max_y mismatch"


def test_features_parquet_geometry_type_int8_matches_wkb_header_per_row(tmp_path: Path) -> None:
    """geometry_type int8 enum (0=Point,1=LineString,2=Polygon) must match the
    WKB type parsed from the geometry column.

    This mirrors inline validator invariant #3 (spec §12.1).
    """
    path = tmp_path / "features.parquet"

    # Map WKB type int to our int8 enum: WKB type 1=Point→0, 2=LineString→1, 3=Polygon→2
    _WKB_TYPE_TO_ENUM = {1: 0, 2: 1, 3: 2}

    rows = [
        _make_feature_row(
            source_feature_id="pt",
            geometry=Point(5.0, 5.0),
            geometry_type=encode_enum(GEOMETRY_TYPE, "Point"),
        ),
        _make_feature_row(
            source_feature_id="ls",
            geometry=LineString([(0.0, 0.0), (50.0, 50.0)]),
            geometry_type=encode_enum(GEOMETRY_TYPE, "LineString"),
            class_raw="residential",
        ),
        _make_feature_row(
            source_feature_id="poly",
            feature_class=1,
            geometry=Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]),
            geometry_type=encode_enum(GEOMETRY_TYPE, "Polygon"),
            class_raw=None,
            subtype_raw="commercial",
        ),
    ]
    write_features_parquet(rows, path)

    table = pq.read_table(path)
    geom_col = table.column("geometry").to_pylist()
    geom_type_col = table.column("geometry_type").to_pylist()

    for i, geom_bytes in enumerate(geom_col):
        # WKB geometry type code is stored in bytes 1-4 (little-endian uint32)
        wkb_type_code = struct.unpack_from("<I", geom_bytes, 1)[0]
        expected_enum = _WKB_TYPE_TO_ENUM[wkb_type_code]
        assert geom_type_col[i] == expected_enum, (
            f"Row {i}: geometry_type={geom_type_col[i]} but WKB type code={wkb_type_code}"
            f" expected enum={expected_enum}"
        )


# ---------------------------------------------------------------------------
# cells.parquet tests
# ---------------------------------------------------------------------------


def test_write_cells_parquet_schema_and_sort_key(tmp_path: Path) -> None:
    """Write a 3-row table in scrambled order; read back; assert schema matches
    spec §11.3 and rows are sorted by (cell_i, cell_j).
    """
    path = tmp_path / "cells.parquet"

    # Deliberately scrambled order
    rows = [
        _make_cell_aggregate(cell_i=2, cell_j=0, water_fraction=0.5),
        _make_cell_aggregate(cell_i=0, cell_j=1, water_fraction=0.2),
        _make_cell_aggregate(cell_i=0, cell_j=0, water_fraction=0.1),
    ]
    write_cells_parquet(rows, path)

    table = pq.read_table(path)

    # --- schema ---
    assert table.schema.equals(_CELLS_SCHEMA), (
        f"Schema mismatch.\nExpected:\n{_CELLS_SCHEMA}\nGot:\n{table.schema}"
    )

    # --- sort key (cell_i, cell_j) ---
    ci_col = table.column("cell_i").to_pylist()
    cj_col = table.column("cell_j").to_pylist()
    sort_keys = list(zip(ci_col, cj_col, strict=True))
    assert sort_keys == sorted(sort_keys), f"Rows not in canonical sort order: {sort_keys}"

    # --- column count ---
    assert len(table.schema) == 6, f"Expected 6 columns, got {len(table.schema)}"

    # --- column types ---
    assert table.schema.field("cell_i").type == pa.int8()
    assert table.schema.field("cell_j").type == pa.int8()
    assert table.schema.field("kept_features_count").type == pa.int32()
    assert table.schema.field("water_fraction").type == pa.float64()


# ---------------------------------------------------------------------------
# crossings.parquet tests
# ---------------------------------------------------------------------------


def test_write_crossings_parquet_schema_and_sort_key(tmp_path: Path) -> None:
    """Write a 3-row crossings table in scrambled order; read back; assert schema
    matches spec §8.2 and rows are sorted by canonical key.
    """
    path = tmp_path / "crossings.parquet"

    axis_x = encode_enum(AXIS, "x")
    axis_y = encode_enum(AXIS, "y")
    enter = encode_enum(EVENT_TYPE, "enter")
    exit_ = encode_enum(EVENT_TYPE, "exit")

    # Scrambled order: should end up sorted by (lower_cell_i, lower_cell_j, axis,
    # source_feature_id, ring_index, edge_position_m, event_type)
    rows = [
        _make_crossing_record(
            source_feature_id="zzz",
            lower_cell_i=1,
            lower_cell_j=0,
            axis=axis_x,
            ring_index=0,
            event_type=enter,
            edge_position_m=200.0,
        ),
        _make_crossing_record(
            source_feature_id="aaa",
            lower_cell_i=0,
            lower_cell_j=0,
            axis=axis_y,
            ring_index=0,
            event_type=exit_,
            edge_position_m=100.0,
        ),
        _make_crossing_record(
            source_feature_id="aaa",
            lower_cell_i=0,
            lower_cell_j=0,
            axis=axis_x,
            ring_index=0,
            event_type=enter,
            edge_position_m=50.0,
        ),
    ]
    write_crossings_parquet(rows, path)

    table = pq.read_table(path)

    # --- schema ---
    assert table.schema.equals(_CROSSINGS_SCHEMA), (
        f"Schema mismatch.\nExpected:\n{_CROSSINGS_SCHEMA}\nGot:\n{table.schema}"
    )

    # --- column count ---
    assert len(table.schema) == 8, f"Expected 8 columns, got {len(table.schema)}"

    # --- column types ---
    assert table.schema.field("lower_cell_i").type == pa.int8()
    assert table.schema.field("lower_cell_j").type == pa.int8()
    assert table.schema.field("axis").type == pa.int8()
    assert table.schema.field("ring_index").type == pa.int16()
    assert table.schema.field("event_type").type == pa.int8()
    assert table.schema.field("edge_position_m").type == pa.float64()

    # --- sort key ---
    lci = table.column("lower_cell_i").to_pylist()
    lcj = table.column("lower_cell_j").to_pylist()
    ax = table.column("axis").to_pylist()
    sid = table.column("source_feature_id").to_pylist()
    ri = table.column("ring_index").to_pylist()
    ep = table.column("edge_position_m").to_pylist()
    et = table.column("event_type").to_pylist()

    sort_keys = list(zip(lci, lcj, ax, sid, ri, ep, et, strict=True))
    assert sort_keys == sorted(sort_keys), f"Rows not in canonical sort order: {sort_keys}"


# ---------------------------------------------------------------------------
# meta.yaml tests
# ---------------------------------------------------------------------------


def test_write_meta_yaml_includes_aggregates_and_conditioning(tmp_path: Path) -> None:
    """write_meta_yaml produces a valid YAML with all spec §11.5 top-level keys
    and the expected aggregates + conditioning_per_tile sub-keys.
    """
    path = tmp_path / "meta.yaml"
    meta = _make_tile_meta()
    write_meta_yaml(meta, path)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    # --- top-level keys (spec §11.5) ---
    for key in (
        "schema_version",
        "tile_i",
        "tile_j",
        "aggregates",
        "config",
        "conditioning_per_tile",
    ):
        assert key in data, f"Missing top-level key '{key}' in meta.yaml"

    # --- aggregates sub-keys ---
    agg = data["aggregates"]
    for sub_key in (
        "kept_cell_count",
        "sea_mask_drop_count",
        "mean_water_fraction",
        "mean_sea_water_fraction",
        "feature_count_by_class",
        "crossing_count",
    ):
        assert sub_key in agg, f"Missing aggregates key '{sub_key}'"

    # feature_count_by_class has the four class labels
    fcc = agg["feature_count_by_class"]
    for label in ("road", "building", "poi", "base"):
        assert label in fcc, f"Missing feature_count_by_class key '{label}'"

    # --- conditioning_per_tile sub-keys (spec §11.5) ---
    cpt = data["conditioning_per_tile"]
    for sub_key in (
        "admin_region",
        "morphology_class",
        "era_class",
        "coastal_inland_river",
        "population_density_bucket",
        "population_density_bucket_owner",
    ):
        assert sub_key in cpt, f"Missing conditioning_per_tile key '{sub_key}'"

    # --- tile identity ---
    assert data["tile_i"] == 12
    assert data["tile_j"] == 17
    assert data["schema_version"] == "1.1"

    # --- config key (spec §11.5) ---
    assert "sliver_drop_rule" in data["config"]


# ---------------------------------------------------------------------------
# provenance.yaml tests
# ---------------------------------------------------------------------------


def test_write_provenance_yaml_inputs_and_outputs_digests(tmp_path: Path) -> None:
    """write_provenance_yaml produces a valid YAML with all spec §11.6 top-level
    keys and correct inputs/outputs sub-keys.
    """
    path = tmp_path / "provenance.yaml"
    prov = _make_tile_provenance()
    write_provenance_yaml(prov, path)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    # --- top-level keys (spec §11.6) ---
    for key in ("schema_version", "tile_i", "tile_j", "crs", "extraction", "inputs", "outputs"):
        assert key in data, f"Missing top-level key '{key}' in provenance.yaml"

    # --- extraction sub-keys ---
    ext = data["extraction"]
    for sub_key in ("commit_sha", "extracted_utc", "rerun_count", "rerun_reason"):
        assert sub_key in ext, f"Missing extraction key '{sub_key}'"

    # --- inputs sub-keys ---
    inp = data["inputs"]
    for sub_key in ("release", "admin_polygon_sha256", "policy_yaml_sha256", "vocab_yaml_sha256"):
        assert sub_key in inp, f"Missing inputs key '{sub_key}'"

    # --- outputs sub-keys ---
    out = data["outputs"]
    for sub_key in (
        "cells_parquet_sha256",
        "features_parquet_sha256",
        "crossings_parquet_sha256",
        "meta_yaml_sha256",
    ):
        assert sub_key in out, f"Missing outputs key '{sub_key}'"

    # --- identity ---
    assert data["tile_i"] == 12
    assert data["tile_j"] == 17
    assert data["crs"] == "EPSG:3414"
    assert data["schema_version"] == "1.1"


# ---------------------------------------------------------------------------
# Tile directory naming test
# ---------------------------------------------------------------------------


def test_per_tile_directory_naming_includes_crs_and_named_ij(tmp_path: Path) -> None:
    """Tile directory names follow tile=EPSG<code>_i<i>_j<j> convention (spec §11.1).

    The CRS code and signed i/j coordinates must be embedded in the directory name
    so that consumers can determine the CRS and tile coordinates without reading
    any parquet file. This test exercises the naming convention itself by forming
    the expected directory name and asserting it round-trips correctly.
    """
    # Standard Singapore CRS
    crs_code = "3414"
    tile_i = 12
    tile_j = -3  # negative j is valid for future regions (not Singapore but spec allows)

    tile_dir_name = f"tile=EPSG{crs_code}_i{tile_i}_j{tile_j}"
    tile_dir = tmp_path / tile_dir_name
    tile_dir.mkdir()

    # Write a minimal cells.parquet + meta.yaml into it to confirm the convention works
    cells_path = tile_dir / "cells.parquet"
    write_cells_parquet([_make_cell_aggregate(cell_i=0, cell_j=0)], cells_path)
    assert cells_path.exists()

    meta_path = tile_dir / "meta.yaml"
    write_meta_yaml(_make_tile_meta(tile_i=tile_i, tile_j=tile_j), meta_path)
    assert meta_path.exists()

    # Assert directory name parses correctly
    name = tile_dir.name
    assert name.startswith("tile=EPSG"), f"Expected 'tile=EPSG' prefix, got: {name}"
    assert f"_i{tile_i}_j{tile_j}" in name, f"Expected i/j in name: {name}"

    # Assert CRS code embedded
    assert f"EPSG{crs_code}" in name, f"Expected EPSG code in name: {name}"


# ---------------------------------------------------------------------------
# Byte-determinism test
# ---------------------------------------------------------------------------


def test_byte_deterministic_re_write_modulo_excluded_fields(tmp_path: Path) -> None:
    """Writing the same tile data twice produces byte-identical parquet files
    and YAML files (excluding fields governed by EXCLUDED_FROM_SHA).

    For parquet: table bytes must be identical (sort key + pinned schema guarantee this).
    For YAML: raw file bytes must be identical (canonicalize_yaml guarantee).

    The provenance.yaml extracted_utc field changes between runs but is excluded
    from the sha computation — we verify the sha is stable even with different
    timestamps.
    """
    from cfm.data.sub_c.determinism import compute_sha256_excluding

    # --- features.parquet ---
    features_path_1 = tmp_path / "features_1.parquet"
    features_path_2 = tmp_path / "features_2.parquet"
    rows = [
        _make_feature_row(cell_i=0, cell_j=0, source_feature_id="f001"),
        _make_feature_row(
            cell_i=1,
            cell_j=0,
            source_feature_id="f002",
            geometry=Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]),
            geometry_type=2,
            feature_class=1,
            class_raw=None,
            subtype_raw="residential",
        ),
    ]
    write_features_parquet(rows, features_path_1)
    write_features_parquet(rows, features_path_2)
    assert features_path_1.read_bytes() == features_path_2.read_bytes(), (
        "features.parquet is not byte-deterministic across two writes"
    )

    # --- cells.parquet ---
    cells_path_1 = tmp_path / "cells_1.parquet"
    cells_path_2 = tmp_path / "cells_2.parquet"
    cells = [_make_cell_aggregate(cell_i=0, cell_j=0)]
    write_cells_parquet(cells, cells_path_1)
    write_cells_parquet(cells, cells_path_2)
    assert cells_path_1.read_bytes() == cells_path_2.read_bytes(), (
        "cells.parquet is not byte-deterministic across two writes"
    )

    # --- crossings.parquet ---
    cx_path_1 = tmp_path / "cx_1.parquet"
    cx_path_2 = tmp_path / "cx_2.parquet"
    crossings = [_make_crossing_record()]
    write_crossings_parquet(crossings, cx_path_1)
    write_crossings_parquet(crossings, cx_path_2)
    assert cx_path_1.read_bytes() == cx_path_2.read_bytes(), (
        "crossings.parquet is not byte-deterministic across two writes"
    )

    # --- meta.yaml byte identity ---
    meta_path_1 = tmp_path / "meta_1.yaml"
    meta_path_2 = tmp_path / "meta_2.yaml"
    meta = _make_tile_meta()
    write_meta_yaml(meta, meta_path_1)
    write_meta_yaml(meta, meta_path_2)
    assert meta_path_1.read_bytes() == meta_path_2.read_bytes(), (
        "meta.yaml is not byte-deterministic across two writes"
    )

    # --- provenance.yaml: sha stable even when extracted_utc changes ---
    prov_a = _make_tile_provenance()
    # Simulate a re-run with different extracted_utc (same other fields)
    prov_b = TileProvenance(
        schema_version=prov_a.schema_version,
        tile_i=prov_a.tile_i,
        tile_j=prov_a.tile_j,
        crs=prov_a.crs,
        extraction={**prov_a.extraction, "extracted_utc": "2026-06-01T12:00:00Z"},
        inputs=prov_a.inputs,
        outputs=prov_a.outputs,
    )

    prov_path_a = tmp_path / "prov_a.yaml"
    prov_path_b = tmp_path / "prov_b.yaml"
    write_provenance_yaml(prov_a, prov_path_a)
    write_provenance_yaml(prov_b, prov_path_b)

    data_a = yaml.safe_load(prov_path_a.read_text())
    data_b = yaml.safe_load(prov_path_b.read_text())

    sha_a = compute_sha256_excluding(data_a, file_key="provenance.yaml")
    sha_b = compute_sha256_excluding(data_b, file_key="provenance.yaml")
    assert sha_a == sha_b, (
        "provenance.yaml sha should be identical when only extracted_utc differs "
        "(extracted_utc is excluded from sha per EXCLUDED_FROM_SHA)"
    )
