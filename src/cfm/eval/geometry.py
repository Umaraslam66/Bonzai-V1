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

#: DECISION (defect (a) fix, 2026-07-19): closure is FLOAT-DRIFT-tolerant, not exact.
#: The sealed decoder's rotation arithmetic leaves ~1e-14 m residue on the closing vertex
#: (cos(radians(90)) = 6.1e-17 scale), so exact `==` failed ~52% of the encoder's OWN clean
#: round-trips (eyeball probe, reports/_eyeball_probe/SUMMARY.md). 1e-6 m sits ~1e8 above
#: that drift and far below the coordinate quantum, so the MODEL's ~1-quantum closing misses
#: (defect (b)) still fail — the epsilon absorbs float noise, never model imprecision.
#: Structural-boundary epsilon per the project discipline; user thresholds stay strict.
_CLOSURE_EPS_M = 1e-6


def _is_building_block(block: list[int]) -> bool:
    """Construction-identity: the block carries a building-class token (one authority)."""
    ids = building_token_ids()
    return any(t in ids for t in block)


def _is_closed_ring(geom: dict[str, Any]) -> bool:
    coords = geom.get("coordinates")
    if not (isinstance(coords, list) and len(coords) >= _MIN_RING_COORDS):
        return False
    first, last = coords[0], coords[-1]
    if not (isinstance(first, (list, tuple)) and isinstance(last, (list, tuple))):
        return first == last
    if len(first) != len(last):
        return False
    return all(abs(a - b) <= _CLOSURE_EPS_M for a, b in zip(first, last, strict=True))


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
    from cfm.eval.holdout.paths import (
        epsg_label_for_region,
        holdout_manifest_for_region,
        sub_f_region_dir,
        tile_dirname,
    )

    # REGION-AWARE (obligation (a)): singapore -> SG manifest (1.0); EU held-out cities
    # -> multiregion manifest (2.0). The tile-data round-trip below is Leonardo-only.
    manifest = yaml.safe_load(
        holdout_manifest_for_region(release, region).read_text(encoding="utf-8")
    )
    tiles = manifest["regions"][region]["tiles"]
    sub_f_dir = sub_f_region_dir(release, region)
    # Per-tile dir names embed the REGION's CRS label (e.g. EPSG25832), NOT the Singapore
    # EPSG3414 default — without this an EU region's tiles resolve to a non-existent
    # SG-named dir and every tile is silently skipped, returning a VACUOUS 0.0 (Task-9
    # step-0 eval-side fix, the twin of the Task-8 build_shards fix).
    epsg_label = epsg_label_for_region(region)

    n_polygons = 0
    n_active_cells = 0
    for tile in tiles:
        cells_path = (
            sub_f_dir / tile_dirname(tile["tile_i"], tile["tile_j"], epsg_label) / "cells.parquet"
        )
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
