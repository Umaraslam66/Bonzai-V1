from __future__ import annotations

from dataclasses import dataclass

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
)
from cfm.tokenizer.vocabulary import TokenId, Vocabulary

GeoJSON = dict

DYADIC_STEPS_M: tuple[int, ...] = (32, 16, 8, 4, 2, 1)
CARDINAL: tuple[tuple[int, int, str], ...] = (
    (0, 1, "N"),
    (1, 0, "E"),
    (0, -1, "S"),
    (-1, 0, "W"),
)


@dataclass(frozen=True)
class CellTokens:
    """Token sequence for a single cell, plus its spatial frame."""

    tokens: tuple[TokenId, ...]
    cell_origin: tuple[float, float]
    cell_size_m: float


def encode_cell(
    geojson: GeoJSON,
    *,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> CellTokens:
    out: list[TokenId] = [vocab.token_to_id["BOS"], vocab.token_to_id["CELL"]]
    for feature in geojson["features"]:
        out.extend(_encode_feature(feature, cell_origin, cell_size_m, vocab))
    out.extend([vocab.token_to_id["END_CELL"], vocab.token_to_id["EOS"]])
    return CellTokens(tokens=tuple(out), cell_origin=cell_origin, cell_size_m=cell_size_m)


def _encode_feature(
    feature: dict,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    cls = feature["properties"]["class"]
    if cls not in vocab.token_to_id:
        raise UnsupportedFeatureClass(f"unknown class {cls!r}")
    geom = feature["geometry"]
    gtype = geom["type"]
    if gtype == "Point":
        body = _encode_point(geom["coordinates"], cell_origin, cell_size_m, vocab)
    elif gtype == "Polygon":
        body = _encode_polygon(cls, geom["coordinates"], cell_origin, cell_size_m, vocab)
    else:
        raise UnsupportedGeometry(f"Phase 0 does not yet handle geometry type {gtype!r}")
    return [
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id[cls],
        *body,
        vocab.token_to_id["FEATURE_END"],
    ]


def _encode_point(
    coords: list[float],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    x_local, y_local = _to_cell_local(coords[0], coords[1], cell_origin)
    _require_in_bounds(x_local, y_local, cell_size_m)
    return [
        vocab.token_to_id[f"ANCHOR_X_{round(x_local)}"],
        vocab.token_to_id[f"ANCHOR_Y_{round(y_local)}"],
    ]


def _encode_polygon(
    cls: str,
    coordinates: list[list[list[float]]],
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    # Phase 0: ignore interior rings, exterior only.
    ring = coordinates[0]
    if ring[0] != ring[-1]:
        raise UnsupportedGeometry("polygon ring not closed (first != last)")
    vertices = [tuple(p) for p in ring[:-1]]  # drop the closing duplicate
    vertices_local = [_to_cell_local(x, y, cell_origin) for (x, y) in vertices]
    for x, y in vertices_local:
        _require_in_bounds(x, y, cell_size_m)
    vertices_local = _drop_collinear(vertices_local)
    if cls.startswith("B_") and len(vertices_local) != 4:
        raise UnsupportedGeometry(
            "Phase 0 buildings must be axis-aligned rectangles (4 vertices); "
            f"got {len(vertices_local)}"
        )
    return _encode_closed_path(vertices_local, vocab)


def _encode_closed_path(
    vertices_local: list[tuple[float, float]],
    vocab: Vocabulary,
) -> list[TokenId]:
    # First vertex becomes the anchor.
    ax, ay = vertices_local[0]
    body: list[TokenId] = [
        vocab.token_to_id[f"ANCHOR_X_{round(ax)}"],
        vocab.token_to_id[f"ANCHOR_Y_{round(ay)}"],
    ]
    # For each subsequent vertex (including a virtual return to the anchor for closure),
    # emit cardinal moves that sum to the segment delta.
    closed = [*vertices_local, vertices_local[0]]
    for i in range(len(vertices_local)):
        x0, y0 = closed[i]
        x1, y1 = closed[i + 1]
        body.extend(_encode_axis_aligned_segment(x1 - x0, y1 - y0, vocab))
    return body


def _encode_axis_aligned_segment(
    dx: float,
    dy: float,
    vocab: Vocabulary,
) -> list[TokenId]:
    if dx != 0 and dy != 0:
        raise UnsupportedGeometry(
            f"Phase 0 supports axis-aligned segments only; got dx={dx}, dy={dy}"
        )
    if dx == 0 and dy == 0:
        return []  # degenerate; drop_collinear should have removed it
    if dx == 0:
        direction = "N" if dy > 0 else "S"
        length = round(abs(dy))
    else:
        direction = "E" if dx > 0 else "W"
        length = round(abs(dx))
    return [vocab.token_to_id[f"MOVE_{direction}_{step}"] for step in _dyadic_decomposition(length)]


def _dyadic_decomposition(n: int) -> list[int]:
    """Greedy decomposition of `n` into the dyadic step set {32, 16, 8, 4, 2, 1}."""
    if n < 1:
        raise UnsupportedGeometry(f"segment length must be a positive integer metre; got {n}")
    out: list[int] = []
    remaining = n
    for step in DYADIC_STEPS_M:
        while remaining >= step:
            out.append(step)
            remaining -= step
    if remaining != 0:
        raise UnsupportedGeometry(f"segment length {n} not expressible in dyadic steps")
    return out


def _drop_collinear(verts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove vertices that lie on the same axis-aligned segment as their neighbours."""
    if len(verts) < 3:
        return verts
    n = len(verts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        prev = verts[(i - 1) % n]
        cur = verts[i]
        nxt = verts[(i + 1) % n]
        dx_in, dy_in = cur[0] - prev[0], cur[1] - prev[1]
        dx_out, dy_out = nxt[0] - cur[0], nxt[1] - cur[1]
        same_axis = (dx_in == 0 and dx_out == 0) or (dy_in == 0 and dy_out == 0)
        same_sign = (dx_in * dx_out + dy_in * dy_out) > 0
        if not (same_axis and same_sign):
            out.append(cur)
    return out


def _to_cell_local(x: float, y: float, cell_origin: tuple[float, float]) -> tuple[float, float]:
    return x - cell_origin[0], y - cell_origin[1]


def _require_in_bounds(x: float, y: float, cell_size_m: float) -> None:
    if not (0 <= x <= cell_size_m and 0 <= y <= cell_size_m):
        raise FeatureOutOfBounds(f"point ({x}, {y}) outside [0, {cell_size_m}]^2")
