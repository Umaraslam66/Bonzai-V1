from __future__ import annotations

from dataclasses import dataclass

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
)
from cfm.tokenizer.vocabulary import TokenId, Vocabulary

GeoJSON = dict


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
    else:
        # Line and Polygon handlers arrive in later tasks.
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


def _to_cell_local(x: float, y: float, cell_origin: tuple[float, float]) -> tuple[float, float]:
    return x - cell_origin[0], y - cell_origin[1]


def _require_in_bounds(x: float, y: float, cell_size_m: float) -> None:
    if not (0 <= x <= cell_size_m and 0 <= y <= cell_size_m):
        raise FeatureOutOfBounds(f"point ({x}, {y}) outside [0, {cell_size_m}]^2")
