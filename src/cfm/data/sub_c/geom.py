"""Cell partitioning, split-at-boundaries, crossing-record derivation, sliver drop.

Per spec §8 + §8.3 edge cases. Tokenizer (encode.py:_require_in_bounds) requires
per-cell features to fit in cell-local coordinates; split-at-boundaries (§8.1) is
how multi-cell features are made tokenizable.

Crossing records (§8.2) are the raw input from which sub-E derives PRD-§5
boundary contracts; the 8-column schema + canonical sort key are locked here.

Edge-case handling (§8.3):
  - Single-cell features: one sub-feature, zero crossings.
  - Multi-cell features (linear, span N edge-adjacent cells): N pieces, N-1
    crossings on shared edges. enter/exit alternate per (lower_cell_i, axis)
    or (lower_cell_j, axis) bucket so a zigzag road's events alternate
    correctly across consecutive edge-id segments on the same axis-line.
  - Corner-crossing: a LineString passing through a cell corner where 4 cells
    meet emits TWO records, one per axis (axis=x and axis=y), both with the
    lower-left corner anchor as (lower_cell_i, lower_cell_j). Polygons whose
    boundary does NOT touch the corner (the corner is interior) do NOT emit
    corner records.
  - Polygon interior rings: each interior ring's transversal crossings emit
    records with ring_index >= 1 (exterior shell is ring_index = 0). The
    body chord (polygon ∩ edge) becomes an interval event with ring_index = 0.
  - Co-linear-entirety: a feature lying exactly on a cell boundary attaches to
    the higher-ij cell via half-open derivation (spec §7.2). The lower cell
    receives no piece, no crossing is emitted (touch-but-not-cross).
  - Touch-but-not-cross: a non-Point source whose intersection with the cell
    box collapses to a Point lies in only one cell. The candidate piece is
    skipped; no crossing record is generated.
  - Partial co-linearity: when source ∩ edge_line yields a LineString
    (polygon body chord, or LineString co-linear with edge for some span),
    emit one interval event with edge_extent_length_m = length.
  - Multi-crossing same axis-line: enter/exit alternate globally across all
    crossings on the same (lower_cell_i, axis) or (lower_cell_j, axis)
    bucket, so a zigzag road producing crossings in multiple cell-pair
    segments still alternates monotonically when sorted by edge_position_m.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from shapely.affinity import translate as shapely_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from cfm.data.sub_c.coords import CELL_SIZE_M, TILE_SIZE_M
from cfm.data.sub_c.enums import AXIS, EVENT_TYPE, encode_enum
from cfm.data.sub_c.epsilon import EPS_COORD_M

_CELLS_PER_TILE_EDGE: int = TILE_SIZE_M // CELL_SIZE_M  # 8 cells per side


@dataclass(frozen=True)
class CellSubFeature:
    """A piece of a source feature that fits in a single cell.

    Per spec §11.2 features.parquet row corresponds to one CellSubFeature.
    Geometry is in CELL-LOCAL coordinates (translated so the cell's lower-left
    corner is (0, 0) and the cell spans [0, CELL_SIZE_M]).
    """

    cell_i: int
    cell_j: int
    source_feature_id: str
    feature_class: str  # "road" | "building" | "poi" | "base"
    geometry: BaseGeometry
    geometry_type: str  # "Point" | "LineString" | "Polygon"


@dataclass(frozen=True)
class CrossingRecord:
    """Per spec §8.2 — 8-column schema.

    Canonical sort key (spec §8.2 + §14.2):
    (lower_cell_i, lower_cell_j, axis, source_feature_id, ring_index,
     edge_position_m, event_type).
    """

    source_feature_id: str
    lower_cell_i: int
    lower_cell_j: int
    axis: int  # int8 enum: 0=x, 1=y
    ring_index: int  # 0 for polygon exterior shell; >=1 for interior rings
    event_type: int  # int8 enum: 0=enter, 1=exit, 2=interval
    edge_position_m: float  # raw SVY21 meter along edge (un-quantized)
    edge_extent_length_m: float  # 0 for point crossings; > 0 for intervals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def partition_into_cells(
    features: Iterable[tuple[BaseGeometry, str, str]],
    tile_i: int,
    tile_j: int,
) -> tuple[list[CellSubFeature], list[CrossingRecord]]:
    """Partition features into per-cell sub-features and crossing records.

    Args:
      features: iterable of (geometry, source_feature_id, feature_class) tuples
        in SVY21 absolute coordinates; expected to lie within this tile.
      tile_i, tile_j: tile coordinates.

    Returns:
      Tuple (sub_features, crossings). sub_features are in cell-local
      coordinates; crossings are in raw SVY21 absolute meters along the
      relevant edge axis.

    Sorting:
      sub_features sorted by (cell_i, cell_j, feature_class, source_feature_id).
      crossings sorted by (lower_cell_i, lower_cell_j, axis,
                           source_feature_id, ring_index,
                           edge_position_m, event_type).
    """
    tile_origin_x = tile_i * TILE_SIZE_M
    tile_origin_y = tile_j * TILE_SIZE_M

    sub_features: list[CellSubFeature] = []
    crossings: list[CrossingRecord] = []

    for geom, source_id, fclass in features:
        rel_geom = shapely_translate(geom, xoff=-tile_origin_x, yoff=-tile_origin_y)
        if rel_geom.is_empty:
            continue

        per_cell_pieces = _partition_geometry_into_cells(rel_geom, source_geom_type=geom.geom_type)

        for (ci, cj), piece in per_cell_pieces.items():
            cell_local = shapely_translate(piece, xoff=-ci * CELL_SIZE_M, yoff=-cj * CELL_SIZE_M)
            sub_features.append(
                CellSubFeature(
                    cell_i=ci,
                    cell_j=cj,
                    source_feature_id=source_id,
                    feature_class=fclass,
                    geometry=cell_local,
                    geometry_type=cell_local.geom_type,
                )
            )

        if len(per_cell_pieces) >= 2:
            crossings.extend(
                _derive_crossings_for_feature(
                    rel_geom=rel_geom,
                    per_cell_pieces=per_cell_pieces,
                    source_feature_id=source_id,
                    tile_origin_x=tile_origin_x,
                    tile_origin_y=tile_origin_y,
                )
            )

    crossings.sort(
        key=lambda c: (
            c.lower_cell_i,
            c.lower_cell_j,
            c.axis,
            c.source_feature_id,
            c.ring_index,
            c.edge_position_m,
            c.event_type,
        )
    )
    sub_features.sort(key=lambda s: (s.cell_i, s.cell_j, s.feature_class, s.source_feature_id))
    return sub_features, crossings


def apply_sliver_drop(
    sub_features: Sequence[CellSubFeature],
    *,
    area_threshold_m2: float = 0.01,
    length_threshold_m: float = 0.01,
) -> list[CellSubFeature]:
    """Drop features whose geometry is smaller than the sliver thresholds.

    Per spec §4.3 / §11.5: strict comparison (β user-threshold, no EPSILON).
    Polygon/MultiPolygon use area_threshold_m2; LineString/MultiLineString use
    length_threshold_m. Points are never slivers under area/length thresholds.
    """
    kept: list[CellSubFeature] = []
    for sf in sub_features:
        g = sf.geometry
        if g.geom_type in ("Polygon", "MultiPolygon"):
            if g.area < area_threshold_m2:
                continue
        elif g.geom_type in ("LineString", "MultiLineString"):
            if g.length < length_threshold_m:
                continue
        kept.append(sf)
    return kept


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partition_geometry_into_cells(
    rel_geom: BaseGeometry,
    source_geom_type: str,
) -> dict[tuple[int, int], BaseGeometry]:
    """Intersect rel_geom (in tile-local coords) with every cell box that its
    bbox touches; return non-empty pieces keyed by (ci, cj).

    Half-open boundary handling (spec §7.2): a piece that lies entirely on a
    cell's upper-x or upper-y boundary is dropped — the higher-ij cell already
    has the canonical piece, and keeping the lower one would double-count or
    misattribute a co-linear feature.

    Touch-but-not-cross (spec §8.3): a non-Point source whose intersection
    with a cell box collapses to a 0-d Point is discarded — the feature is
    wholly in another cell.
    """
    min_x, min_y, max_x, max_y = rel_geom.bounds
    min_ci = max(0, math.floor(min_x / CELL_SIZE_M))
    min_cj = max(0, math.floor(min_y / CELL_SIZE_M))
    # NOTE: do NOT subtract EPS_COORD_M from the bbox max — that would
    # exclude a cell whose lower edge coincides with the bbox max (the
    # half-open case where the feature attaches to the higher-ij cell).
    # We rely on the upper-boundary filter below to drop the symmetric
    # lower-side duplicate that floor() may include.
    max_ci = min(_CELLS_PER_TILE_EDGE - 1, math.floor(max_x / CELL_SIZE_M))
    max_cj = min(_CELLS_PER_TILE_EDGE - 1, math.floor(max_y / CELL_SIZE_M))

    per_cell: dict[tuple[int, int], BaseGeometry] = {}
    for ci in range(min_ci, max_ci + 1):
        for cj in range(min_cj, max_cj + 1):
            cell_box = shapely_box(
                ci * CELL_SIZE_M,
                cj * CELL_SIZE_M,
                (ci + 1) * CELL_SIZE_M,
                (cj + 1) * CELL_SIZE_M,
            )
            piece = rel_geom.intersection(cell_box)
            if piece.is_empty:
                continue
            # Touch-but-not-cross: source is non-Point, intersection collapsed
            # to a Point on the cell box boundary — feature is in another cell.
            if isinstance(piece, Point) and source_geom_type != "Point":
                continue
            # Co-linear half-open: drop pieces entirely on this cell's upper
            # boundaries (max-x or max-y) so they don't double up with the
            # higher-ij cell's piece.
            if _piece_on_upper_boundary(piece, ci, cj):
                continue
            per_cell[(ci, cj)] = piece
    return per_cell


def _piece_on_upper_boundary(piece: BaseGeometry, ci: int, cj: int) -> bool:
    """True if `piece` lies entirely on the cell's max-x or max-y edge.

    Such a piece belongs to cell (ci+1, cj) or (ci, cj+1) per half-open
    convention (spec §7.2); the bbox-driven loop will have produced the
    canonical piece in the higher-ij cell, so we drop the duplicate here.
    """
    if piece.is_empty:
        return False
    upper_x = (ci + 1) * CELL_SIZE_M
    upper_y = (cj + 1) * CELL_SIZE_M
    pmin_x, pmin_y, pmax_x, pmax_y = piece.bounds
    if pmax_x - pmin_x <= EPS_COORD_M and abs(pmin_x - upper_x) <= EPS_COORD_M:
        # entirely on max-x line
        return True
    if pmax_y - pmin_y <= EPS_COORD_M and abs(pmin_y - upper_y) <= EPS_COORD_M:
        # entirely on max-y line
        return True
    return False


def _derive_crossings_for_feature(
    *,
    rel_geom: BaseGeometry,
    per_cell_pieces: dict[tuple[int, int], BaseGeometry],
    source_feature_id: str,
    tile_origin_x: float,
    tile_origin_y: float,
) -> list[CrossingRecord]:
    """Derive crossing records for one feature.

    Three categories:
      1. Edge-adjacent cell pairs (sharing an x- or y-axis edge): intersect
         source with the full axis-line (restricted to tile range) and assign
         each piece to an edge_id; emit enter/exit (alternating per
         (lower_cell_i, axis) or (lower_cell_j, axis) line) or interval events.
      2. Diagonal-adjacent cell pairs (no edge-pair piece): if the source's
         boundary passes through the shared corner, emit TWO records (one per
         axis) per spec §8.3 corner-crossing rule.
      3. Polygon interior rings: for each hole, intersect the hole's
         LineString with each crossed line and emit records with
         ring_index >= 1.
    """
    records: list[CrossingRecord] = []

    # ---- 1. Edge-adjacent crossings (body / shell + co-linear intervals) ----
    # Collect crossed vertical and horizontal lines (which cell-boundary indices
    # actually have pieces straddling them).
    occupied_ci: set[int] = {ci for ci, _ in per_cell_pieces}
    occupied_cj: set[int] = {cj for _, cj in per_cell_pieces}

    # Vertical lines (axis = x): x = k * CELL_SIZE_M for k in [1, 7]
    for k in range(1, _CELLS_PER_TILE_EDGE):
        if (k - 1) not in occupied_ci or k not in occupied_ci:
            # Body of feature doesn't span this line on either side
            continue
        records.extend(
            _derive_axis_line_crossings(
                rel_geom=rel_geom,
                axis_name="x",
                line_position_local=k * CELL_SIZE_M,
                per_cell_pieces=per_cell_pieces,
                source_feature_id=source_feature_id,
                tile_origin_x=tile_origin_x,
                tile_origin_y=tile_origin_y,
            )
        )

    # Horizontal lines (axis = y)
    for k in range(1, _CELLS_PER_TILE_EDGE):
        if (k - 1) not in occupied_cj or k not in occupied_cj:
            continue
        records.extend(
            _derive_axis_line_crossings(
                rel_geom=rel_geom,
                axis_name="y",
                line_position_local=k * CELL_SIZE_M,
                per_cell_pieces=per_cell_pieces,
                source_feature_id=source_feature_id,
                tile_origin_x=tile_origin_x,
                tile_origin_y=tile_origin_y,
            )
        )

    # ---- 2. Corner crossings (diagonal-adjacent cell pairs) ----
    records.extend(
        _derive_corner_crossings(
            rel_geom=rel_geom,
            per_cell_pieces=per_cell_pieces,
            source_feature_id=source_feature_id,
            tile_origin_x=tile_origin_x,
            tile_origin_y=tile_origin_y,
        )
    )

    return records


def _derive_axis_line_crossings(
    *,
    rel_geom: BaseGeometry,
    axis_name: str,  # "x" or "y"
    line_position_local: float,
    per_cell_pieces: dict[tuple[int, int], BaseGeometry],
    source_feature_id: str,
    tile_origin_x: float,
    tile_origin_y: float,
) -> list[CrossingRecord]:
    """Intersect rel_geom with a full tile-spanning axis line and emit one
    record per piece, alternating enter/exit across all point-crossings on
    that line (so zigzag features alternate correctly across cell-pair
    segments)."""

    if axis_name == "x":
        # vertical line at x = line_position_local, y in [0, TILE_SIZE_M]
        line = LineString([(line_position_local, 0.0), (line_position_local, TILE_SIZE_M)])
        axis_code = encode_enum(AXIS, "x")
    else:
        line = LineString([(0.0, line_position_local), (TILE_SIZE_M, line_position_local)])
        axis_code = encode_enum(AXIS, "y")

    # Body-chord intersection (for polygons this gives the chord LineString;
    # for LineStrings this gives points or co-linear segments). ring_index=0.
    body_pieces = _flatten_pieces(rel_geom.intersection(line))
    body_records = _records_from_pieces(
        pieces=body_pieces,
        ring_index=0,
        axis_name=axis_name,
        axis_code=axis_code,
        per_cell_pieces=per_cell_pieces,
        source_feature_id=source_feature_id,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
    )

    # Per-interior-ring crossings (polygon holes). ring_index >= 1.
    ring_records: list[CrossingRecord] = []
    if isinstance(rel_geom, Polygon):
        for idx, interior in enumerate(rel_geom.interiors):
            ring_line = LineString(list(interior.coords))
            ring_pieces = _flatten_pieces(ring_line.intersection(line))
            ring_records.extend(
                _records_from_pieces(
                    pieces=ring_pieces,
                    ring_index=idx + 1,
                    axis_name=axis_name,
                    axis_code=axis_code,
                    per_cell_pieces=per_cell_pieces,
                    source_feature_id=source_feature_id,
                    tile_origin_x=tile_origin_x,
                    tile_origin_y=tile_origin_y,
                )
            )
    elif isinstance(rel_geom, MultiPolygon):
        # DECISION: ring_index restarts at idx+1 per sub-polygon, matching the
        # Polygon case. For a MultiPolygon where multiple sub-polygons have holes
        # crossing the same axis line, distinct sub-polygon holes can share
        # ring_index values. Spec §8.2 requires ring_index >= 1 for interior rings
        # but does not require global uniqueness across sub-polygons. Revisit if
        # sub-E needs per-sub-polygon disambiguation (would require adding a
        # sub_polygon_index column to CrossingRecord).
        # multi-polygon source: iterate component polygons + their interior rings
        for sub in rel_geom.geoms:
            if not isinstance(sub, Polygon):
                continue
            for idx, interior in enumerate(sub.interiors):
                ring_line = LineString(list(interior.coords))
                ring_pieces = _flatten_pieces(ring_line.intersection(line))
                ring_records.extend(
                    _records_from_pieces(
                        pieces=ring_pieces,
                        ring_index=idx + 1,
                        axis_name=axis_name,
                        axis_code=axis_code,
                        per_cell_pieces=per_cell_pieces,
                        source_feature_id=source_feature_id,
                        tile_origin_x=tile_origin_x,
                        tile_origin_y=tile_origin_y,
                    )
                )

    # Alternate enter / exit globally per ring_index on this axis line.
    all_records = body_records + ring_records
    _assign_alternating_event_types(all_records)
    return all_records


def _flatten_pieces(intersection: BaseGeometry) -> list[BaseGeometry]:
    """Decompose a shapely intersection result into a flat list of Point /
    LineString components, dropping empties and degenerate types."""
    if intersection.is_empty:
        return []
    if isinstance(intersection, (Point, LineString)):
        return [intersection]
    if isinstance(intersection, (MultiPoint, MultiLineString, GeometryCollection)):
        out: list[BaseGeometry] = []
        for g in intersection.geoms:
            if g.is_empty:
                continue
            if isinstance(g, (Point, LineString)):
                out.append(g)
        return out
    # Polygon/MultiPolygon intersection with a line wouldn't normally appear,
    # but if it does we ignore it (no edge-crossing semantics).
    return []


def _records_from_pieces(
    *,
    pieces: list[BaseGeometry],
    ring_index: int,
    axis_name: str,
    axis_code: int,
    per_cell_pieces: dict[tuple[int, int], BaseGeometry],
    source_feature_id: str,
    tile_origin_x: float,
    tile_origin_y: float,
) -> list[CrossingRecord]:
    """Build crossing records from line-intersection pieces.

    Touch-but-not-cross filter: only emit a record if both cells flanking the
    derived (lower_cell_i, lower_cell_j) edge actually contain a piece of the
    feature in `per_cell_pieces`. Otherwise the feature only TOUCHES the edge,
    it doesn't CROSS it (spec §8.3 row 4).
    """
    out: list[CrossingRecord] = []
    for piece in pieces:
        if isinstance(piece, Point):
            pos_local, lower_ci, lower_cj = _point_to_edge_id(piece, axis_name)
            if not _both_cells_present(per_cell_pieces, lower_ci, lower_cj, axis_name):
                continue
            pos_abs = pos_local + (tile_origin_y if axis_name == "x" else tile_origin_x)
            out.append(
                CrossingRecord(
                    source_feature_id=source_feature_id,
                    lower_cell_i=lower_ci,
                    lower_cell_j=lower_cj,
                    axis=axis_code,
                    ring_index=ring_index,
                    event_type=encode_enum(EVENT_TYPE, "enter"),  # placeholder; alternated later
                    edge_position_m=float(pos_abs),
                    edge_extent_length_m=0.0,
                )
            )
        elif isinstance(piece, LineString):
            # Interval crossing: a polygon body chord, or a LineString segment
            # truly co-linear with the cell-boundary line. The piece may span
            # multiple cell-pair edge segments along the line; split it per
            # cell-pair so each emitted interval lives on exactly one edge_id.
            for split_part in _split_linestring_by_cells(piece, axis_name):
                start_perp, end_perp, lower_ci, lower_cj = split_part
                if not _both_cells_present(per_cell_pieces, lower_ci, lower_cj, axis_name):
                    continue
                pos_local_min = min(start_perp, end_perp)
                extent_local = abs(end_perp - start_perp)
                pos_abs = pos_local_min + (tile_origin_y if axis_name == "x" else tile_origin_x)
                out.append(
                    CrossingRecord(
                        source_feature_id=source_feature_id,
                        lower_cell_i=lower_ci,
                        lower_cell_j=lower_cj,
                        axis=axis_code,
                        ring_index=ring_index,
                        event_type=encode_enum(EVENT_TYPE, "interval"),
                        edge_position_m=float(pos_abs),
                        edge_extent_length_m=float(extent_local),
                    )
                )
    return out


def _point_to_edge_id(point: Point, axis_name: str) -> tuple[float, int, int]:
    """Map a Point on the axis-line to (perpendicular-coord, lower_ci, lower_cj).

    For axis=x (vertical line), perpendicular coord is the point's y; the line
    sits at x = (lower_ci + 1) * CELL_SIZE_M.
    """
    if axis_name == "x":
        # vertical line at x = (lower_ci + 1) * CELL_SIZE_M; lower_cj = floor(y/cs)
        lower_ci = round(point.x / CELL_SIZE_M) - 1
        cj_floor = math.floor(point.y / CELL_SIZE_M)
        # half-open: a point exactly at y = cj * cs is in cell cj (higher side);
        # the LOWER cell for that edge is cj - 1 when the y is on a boundary.
        # But for a generic crossing point inside a cell-pair edge segment,
        # lower_cj = floor(y/cs). A boundary y means the point is on a corner —
        # treated by corner logic, not here.
        return point.y, lower_ci, cj_floor
    else:
        lower_cj = round(point.y / CELL_SIZE_M) - 1
        ci_floor = math.floor(point.x / CELL_SIZE_M)
        return point.x, ci_floor, lower_cj


def _both_cells_present(
    per_cell_pieces: dict[tuple[int, int], BaseGeometry],
    lower_ci: int,
    lower_cj: int,
    axis_name: str,
) -> bool:
    if axis_name == "x":
        return (lower_ci, lower_cj) in per_cell_pieces and (
            lower_ci + 1,
            lower_cj,
        ) in per_cell_pieces
    return (lower_ci, lower_cj) in per_cell_pieces and (
        lower_ci,
        lower_cj + 1,
    ) in per_cell_pieces


def _split_linestring_by_cells(
    line: LineString,
    axis_name: str,
) -> list[tuple[float, float, int, int]]:
    """Split a LineString on the axis-line into per-edge-id segments.

    For axis=x (vertical line, parallel to y): returns segments parameterized
    by their start/end y-coordinates along with (lower_ci, lower_cj). The
    LineString is assumed to lie on the line, so all points share the same
    perpendicular coordinate (the x-value at axis=x).
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return []
    if axis_name == "x":
        # parameter along the line is y
        x_on_line = coords[0][0]
        lower_ci = round(x_on_line / CELL_SIZE_M) - 1
        params = sorted({c[1] for c in coords})
    else:
        y_on_line = coords[0][1]
        lower_cj = round(y_on_line / CELL_SIZE_M) - 1
        params = sorted({c[0] for c in coords})

    # Insert every cell-grid crossing between the bbox extremes
    p_min, p_max = params[0], params[-1]
    crossings = sorted(
        {
            k * CELL_SIZE_M
            for k in range(_CELLS_PER_TILE_EDGE + 1)
            if p_min < k * CELL_SIZE_M < p_max
        }
    )
    breakpoints = [p_min, *crossings, p_max]

    out: list[tuple[float, float, int, int]] = []
    for a, b in pairwise(breakpoints):
        if b - a <= EPS_COORD_M:
            continue
        midpoint = (a + b) / 2.0
        if axis_name == "x":
            lower_cj = math.floor(midpoint / CELL_SIZE_M)
            out.append((a, b, lower_ci, lower_cj))
        else:
            lower_ci = math.floor(midpoint / CELL_SIZE_M)
            out.append((a, b, lower_ci, lower_cj))
    return out


def _assign_alternating_event_types(records: list[CrossingRecord]) -> None:
    """For point records (edge_extent_length_m == 0) on the same axis line,
    sort by edge_position_m and assign enter (0) / exit (1) alternately.

    Operates per ring_index so a polygon's exterior body-chord points and its
    interior ring points alternate independently.

    Mutates `records` IN PLACE by replacing CrossingRecord entries with new
    instances that have the updated event_type. (CrossingRecord is frozen.)
    """
    # Group indices by ring_index for independent alternation
    enter_code = encode_enum(EVENT_TYPE, "enter")
    exit_code = encode_enum(EVENT_TYPE, "exit")

    by_ring: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        if r.edge_extent_length_m != 0.0:
            continue
        by_ring.setdefault(r.ring_index, []).append(i)

    for idxs in by_ring.values():
        # sort by edge_position_m
        idxs_sorted = sorted(idxs, key=lambda i: records[i].edge_position_m)
        for n, i in enumerate(idxs_sorted):
            r = records[i]
            new_event = enter_code if n % 2 == 0 else exit_code
            records[i] = CrossingRecord(
                source_feature_id=r.source_feature_id,
                lower_cell_i=r.lower_cell_i,
                lower_cell_j=r.lower_cell_j,
                axis=r.axis,
                ring_index=r.ring_index,
                event_type=new_event,
                edge_position_m=r.edge_position_m,
                edge_extent_length_m=r.edge_extent_length_m,
            )


def _derive_corner_crossings(
    *,
    rel_geom: BaseGeometry,
    per_cell_pieces: dict[tuple[int, int], BaseGeometry],
    source_feature_id: str,
    tile_origin_x: float,
    tile_origin_y: float,
) -> list[CrossingRecord]:
    """Emit corner-crossing records per spec §8.3.

    A corner-crossing occurs when the source feature's boundary passes
    EXACTLY through a cell-corner point (where 4 cells meet) AND the feature
    has presence in at least one pair of diagonally-opposite cells across
    that corner, without an edge-adjacent piece bridging them.

    Each such corner emits TWO records (axis=x and axis=y) anchored at the
    lower-left of the 4 corner cells.
    """
    records: list[CrossingRecord] = []
    enter_code = encode_enum(EVENT_TYPE, "enter")
    seen_corners: set[tuple[int, int]] = set()

    # Inspect every diagonal pair we can find in per_cell_pieces
    cells = list(per_cell_pieces.keys())
    for a in cells:
        for b in cells:
            if a >= b:
                continue
            di = b[0] - a[0]
            dj = b[1] - a[1]
            if abs(di) != 1 or abs(dj) != 1:
                continue
            # diagonal pair (a, b). The shared corner is the cell-corner of
            # min(ci_a, ci_b)+1 by min(cj_a, cj_b)+1.
            lower_ci = min(a[0], b[0])
            lower_cj = min(a[1], b[1])
            corner_x = (lower_ci + 1) * CELL_SIZE_M
            corner_y = (lower_cj + 1) * CELL_SIZE_M
            # Edge-adjacent piece bridging would defeat the corner-only case:
            # if both edge-adjacent neighbours of `a` toward `b` exist
            # (i.e., (a[0], b[1]) and (b[0], a[1]) both present), it's not a
            # pure corner crossing — already handled by edge-line records.
            edge_neighbour_1 = (a[0], b[1])
            edge_neighbour_2 = (b[0], a[1])
            if edge_neighbour_1 in per_cell_pieces or edge_neighbour_2 in per_cell_pieces:
                continue
            corner_point = Point(corner_x, corner_y)
            if not _feature_boundary_touches(rel_geom, corner_point):
                continue
            corner_key = (lower_ci, lower_cj)
            if corner_key in seen_corners:
                continue
            seen_corners.add(corner_key)

            # Position is in raw SVY21 meters along the edge axis.
            # axis=x edge between (lower_ci, lower_cj) and (lower_ci+1, lower_cj)
            #   — perpendicular coord is y (the corner y).
            # axis=y edge between (lower_ci, lower_cj) and (lower_ci, lower_cj+1)
            #   — perpendicular coord is x (the corner x).
            pos_y_abs = corner_y + tile_origin_y
            pos_x_abs = corner_x + tile_origin_x

            records.append(
                CrossingRecord(
                    source_feature_id=source_feature_id,
                    lower_cell_i=lower_ci,
                    lower_cell_j=lower_cj,
                    axis=encode_enum(AXIS, "x"),
                    ring_index=0,
                    event_type=enter_code,
                    edge_position_m=float(pos_y_abs),
                    edge_extent_length_m=0.0,
                )
            )
            records.append(
                CrossingRecord(
                    source_feature_id=source_feature_id,
                    lower_cell_i=lower_ci,
                    lower_cell_j=lower_cj,
                    axis=encode_enum(AXIS, "y"),
                    ring_index=0,
                    event_type=enter_code,
                    edge_position_m=float(pos_x_abs),
                    edge_extent_length_m=0.0,
                )
            )
    return records


def _feature_boundary_touches(geom: BaseGeometry, corner: Point) -> bool:
    """True iff the geometry's effective boundary passes through the corner.

    LineString / Point: uses geom.distance(corner) directly (any contact).
    Polygon / MultiPolygon: uses the boundary rings (a corner strictly INSIDE
    the polygon body — i.e., not on any ring — is NOT a boundary touch and
    should not yield corner records).
    """
    if isinstance(geom, (Polygon, MultiPolygon)):
        boundary = geom.boundary
        if boundary.is_empty:
            return False
        return boundary.distance(corner) <= EPS_COORD_M
    return geom.distance(corner) <= EPS_COORD_M
