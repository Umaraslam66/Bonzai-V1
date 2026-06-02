"""Building-ring promotion -- the single authority (Phase-2 bake-off Task 1.5).

The sealed sub-F decoder returns a closed ring with no brefs as ``type: "LineString"``
BY DOCUMENTED CONTRACT (``decoder.py:145-157``): a closed ring is ambiguous between a
building polygon and a closed-road roundabout at decode time, so the decoder defers the
promotion to the consumer. The scaffold's metrics never did the promotion, so every
building-geometry metric (n_polygons, right-angle, building-area KS, the emergence floor)
silently read 0 on real data -- the dominant cause of the probe's ``n_polygons=0`` (NOT
under-training). The 6th contract/regime catch of the bake-off.

This module is the ONE promotion authority that slice_metrics + emergence (and, on its
output, realism) share. Promotion is keyed on CONSTRUCTION IDENTITY (§9): a feature is a
building iff its block carries a building-class token (the SAME 77-id authority Task 1's
``building_token_ids`` uses) -- NEVER bare ring-closure, which would wrongly promote
closed-road roundabouts (the exact ambiguity the decoder names).
"""

from __future__ import annotations

from typing import Any

from cfm.eval.emergence import building_token_ids

_MIN_RING_COORDS = 4  # >=3 distinct vertices + the closing repeat


def _is_building_block(block: list[int]) -> bool:
    """Construction-identity: the block carries a building-class token (one authority)."""
    ids = building_token_ids()
    return any(t in ids for t in block)


def _is_closed_ring(geom: dict[str, Any]) -> bool:
    coords = geom.get("coordinates")
    return isinstance(coords, list) and len(coords) >= _MIN_RING_COORDS and coords[0] == coords[-1]


def promote_building_rings(
    blocks: list[list[int]], geoms: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return geoms with building-class CLOSED LineStrings re-typed as Polygon.

    A decoded feature is promoted iff its block is a building feature-class (§9
    construction identity) AND its decoded ring is closed. Roads -- including closed
    roundabouts -- are NOT promoted (they stay LineString), which is exactly what
    distinguishes feature-class keying from bare ring-closure. Non-LineString geoms and
    non-building features pass through unchanged.
    """
    out: list[dict[str, Any]] = []
    for block, geom in zip(blocks, geoms, strict=True):
        if geom.get("type") == "LineString" and _is_building_block(block) and _is_closed_ring(geom):
            out.append({"type": "Polygon", "coordinates": [geom["coordinates"]]})
        else:
            out.append(geom)
    return out


def holdout_polygons_per_active_cell(*, release: str, region: str) -> float:
    """The holdout's real polygons-per-active-cell -- the one source of the emergence
    floor's ``holdout_polys_per_cell``.

    Round-trips real holdout cells through the SEALED sub-F decoder + sub-G splitter, then
    PROMOTES building closed rings to polygons (Task 1.5) before counting -- without the
    promotion this reads a vacuous 0.0 (buildings decode as LineString by decoder contract).
    Requires the real sub-F tile data (Leonardo ``$WORK``); the calling test is slow.
    """
    import yaml

    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_g.readers import read_sub_f_cells
    from cfm.data.sub_g.seam_decodability import split_cell_into_features
    from cfm.eval.holdout.paths import holdout_manifest_path, sub_f_region_dir, tile_dirname

    manifest = yaml.safe_load(holdout_manifest_path(release).read_text(encoding="utf-8"))
    tiles = manifest["regions"][region]["tiles"]
    sub_f_dir = sub_f_region_dir(release, region)

    n_polygons = 0
    n_active_cells = 0
    for tile in tiles:
        cells_path = sub_f_dir / tile_dirname(tile["tile_i"], tile["tile_j"]) / "cells.parquet"
        if not cells_path.exists():
            continue
        for tokens in read_sub_f_cells(cells_path).values():
            if not tokens:
                continue
            n_active_cells += 1
            blocks = split_cell_into_features(tokens)
            geoms = [decode_feature(b) for b in blocks]
            for geom in promote_building_rings(blocks, geoms):
                if geom.get("type") in ("Polygon", "MultiPolygon"):
                    n_polygons += 1
    return n_polygons / n_active_cells if n_active_cells else 0.0
