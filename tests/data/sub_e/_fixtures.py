"""Sub-E test fixtures.

Synthetic sub-D + sub-C region builder used by Task 10 pipeline tests.
Mirrors the data shapes sub-E reads from real sub-D output without depending
on sub-D's full pipeline.
"""

from __future__ import annotations

from pathlib import Path


def _build_synthetic_sub_d_and_sub_c(tmp_path: Path) -> Path:
    """Build minimum-viable sub-D + sub-C region pair for sub-E to read.

    Returns the sub-D region directory; the sub-C region directory sits
    alongside under the same tmp_path tree.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    sub_c = tmp_path / "sub_c" / "singapore"
    sub_d = tmp_path / "sub_d" / "singapore"
    sub_c.mkdir(parents=True)
    sub_d.mkdir(parents=True)

    tiles = [(0, 0), (1, 0)]

    sub_d_tile_records: list[dict] = []
    sub_c_tile_records: list[dict] = []

    # Enumerate the 8x8 grid via cell_to_edge_ids and collect unique internal
    # + external (lower_cell_i, lower_cell_j, axis) triples. This mirrors
    # what real sub-D produces because sub-D's lattice IS rotation's lattice
    # (spec §4.1: sub-E inherits sub-D's lattice verbatim).
    #
    # Earlier draft generated triples via modulo arithmetic (idx % 8 etc.)
    # which produced (a) duplicate keys within internals (e.g. idx=0 and
    # idx=64 both yield (0, 0, 0)) and (b) externals outside rotation's
    # set (axis=1 with lower_cell_i in {1..6} which rotation never emits).
    # Both defects would have broken Task 10's happy path before any
    # validator ran. Pattern source: Task 9's _external_rows_via_rotation.
    from cfm.data.sub_e.rotation import EdgeKind, cell_to_edge_ids

    internal_set: set[tuple[int, int, int]] = set()
    external_set: set[tuple[int, int, int]] = set()
    for ci in range(8):
        for cj in range(8):
            cell_edges = cell_to_edge_ids(ci, cj)
            for edge in (
                cell_edges.north,
                cell_edges.south,
                cell_edges.west,
                cell_edges.east,
            ):
                li, lj, axis, kind = edge
                if kind is EdgeKind.INTERNAL:
                    internal_set.add((li, lj, axis))
                else:
                    external_set.add((li, lj, axis))
    internal_triples = sorted(internal_set)
    external_triples = sorted(external_set)
    assert len(internal_triples) == 112, (
        f"rotation should produce 112 unique internal triples, got {len(internal_triples)}"
    )
    assert len(external_triples) == 32, (
        f"rotation should produce 32 unique external triples, got {len(external_triples)}"
    )

    for ti, tj in tiles:
        sub_d_tile = sub_d / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_c_tile = sub_c / f"tile=EPSG3414_i{ti}_j{tj}"
        sub_d_tile.mkdir()
        sub_c_tile.mkdir()

        # Synthetic sub-D macro_core: 64 cell rows + 112 internal edge rows +
        # 32 external edge rows. Cell rows carry scope=0; internal-edge rows
        # carry scope=0 (active); external-edge rows carry scope=3
        # (external_deferred). Total: 64+112+32 = 208 rows.
        slot_kinds, slot_indices = [], []
        cell_is, cell_js = [], []
        lower_is, lower_js, axes, scopes = [], [], [], []
        zoning, density, road = [], [], []
        for idx in range(64):
            slot_kinds.append(0)  # cell
            slot_indices.append(idx)
            cell_is.append(idx % 8)
            cell_js.append(idx // 8)
            lower_is.append(None)
            lower_js.append(None)
            axes.append(None)
            scopes.append(0)
            zoning.append(0)
            density.append(1)
            road.append(None)
        for idx, (li, lj, axis) in enumerate(internal_triples):
            slot_kinds.append(1)  # internal_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(li)
            lower_js.append(lj)
            axes.append(axis)
            scopes.append(0)  # active
            zoning.append(None)
            density.append(None)
            road.append(0)
        for idx, (li, lj, axis) in enumerate(external_triples):
            slot_kinds.append(2)  # external_edge
            slot_indices.append(idx)
            cell_is.append(None)
            cell_js.append(None)
            lower_is.append(li)
            lower_js.append(lj)
            axes.append(axis)
            scopes.append(3)  # external_deferred
            zoning.append(None)
            density.append(None)
            road.append(None)
        macro_core_table = pa.table(
            {
                "slot_kind": pa.array(slot_kinds, type=pa.int8()),
                "slot_index": pa.array(slot_indices, type=pa.int16()),
                "cell_i": pa.array(cell_is, type=pa.int8()),
                "cell_j": pa.array(cell_js, type=pa.int8()),
                "lower_cell_i": pa.array(lower_is, type=pa.int8()),
                "lower_cell_j": pa.array(lower_js, type=pa.int8()),
                "axis": pa.array(axes, type=pa.int8()),
                "scope": pa.array(scopes, type=pa.int8()),
                "zoning_class": pa.array(zoning, type=pa.int16()),
                "cell_density_bucket": pa.array(density, type=pa.int16()),
                "road_skeleton_class": pa.array(road, type=pa.int16()),
            }
        )
        pq.write_table(macro_core_table, sub_d_tile / "macro_core.parquet")

        # Synthetic sub-C: one primary-road crossing on edge (0, 0, axis=0).
        crossings_table = pa.table(
            {
                "lower_cell_i": pa.array([0], type=pa.int8()),
                "lower_cell_j": pa.array([0], type=pa.int8()),
                "axis": pa.array([0], type=pa.int8()),
                "source_feature_id": pa.array(["F-primary"], type=pa.string()),
            }
        )
        pq.write_table(crossings_table, sub_c_tile / "crossings.parquet")

        features_table = pa.table(
            {
                "source_feature_id": pa.array(["F-primary"], type=pa.string()),
                "feature_class": pa.array(["road"], type=pa.string()),
                "class_raw": pa.array(["primary"], type=pa.string()),
            }
        )
        pq.write_table(features_table, sub_c_tile / "features.parquet")

        sub_d_tile_records.append({"tile_i": ti, "tile_j": tj, "provenance_sha256": "0" * 64})
        sub_c_tile_records.append({"tile_i": ti, "tile_j": tj, "provenance_sha256": "0" * 64})

    # Minimal sub-C manifest + _SUCCESS.
    (sub_c / "manifest.yaml").write_text(
        yaml.safe_dump({"region": "singapore", "tiles": sub_c_tile_records})
    )
    (sub_c / "_SUCCESS").touch()

    # Minimal sub-D manifest + _SUCCESS. Sub-E's pipeline reads `tiles[]` and
    # `inputs.sub_c_region_dir`; the rest can be skeletal.
    (sub_d / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "region": "singapore",
                "tiles": sub_d_tile_records,
                "inputs": {"sub_c_region_dir": str(sub_c)},
            }
        )
    )
    (sub_d / "_SUCCESS").touch()

    return sub_d
