from __future__ import annotations

from cfm.tokenizer.encode import CellTokens
from cfm.tokenizer.errors import UnsupportedGeometry, VocabularyMismatch
from cfm.tokenizer.vocabulary import Vocabulary

GeoJSON = dict


def decode_cell(tokens: CellTokens, *, vocab: Vocabulary) -> GeoJSON:
    names = [_lookup(tid, vocab) for tid in tokens.tokens]
    cursor = _Cursor(names)
    cursor.expect("BOS")
    cursor.expect("CELL")
    features: list[dict] = []
    while cursor.peek() != "END_CELL":
        features.append(_decode_feature(cursor, tokens.cell_origin, tokens.cell_size_m))
    cursor.expect("END_CELL")
    cursor.expect("EOS")
    return {"type": "FeatureCollection", "features": features}


def _lookup(tid: int, vocab: Vocabulary) -> str:
    if not 0 <= tid < len(vocab):
        raise VocabularyMismatch(f"token id {tid} out of vocabulary range")
    return vocab.id_to_token[tid]


class _Cursor:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self._i = 0

    def peek(self) -> str:
        if self._i >= len(self._names):
            raise VocabularyMismatch("unexpected end of token sequence")
        return self._names[self._i]

    def take(self) -> str:
        name = self.peek()
        self._i += 1
        return name

    def expect(self, expected: str) -> None:
        got = self.take()
        if got != expected:
            raise VocabularyMismatch(f"expected {expected!r}, got {got!r}")


def _decode_feature(
    cursor: _Cursor,
    cell_origin: tuple[float, float],
    cell_size_m: float,
) -> dict:
    cursor.expect("FEATURE_START")
    cls = cursor.take()
    body: list[str] = []
    while cursor.peek() != "FEATURE_END":
        body.append(cursor.take())
    cursor.expect("FEATURE_END")
    return _materialise_feature(cls, body, cell_origin, cell_size_m)


def _materialise_feature(
    cls: str,
    body: list[str],
    cell_origin: tuple[float, float],
    cell_size_m: float,
) -> dict:
    # Phase 0 dispatch by class prefix.
    has_exit = body and body[-1] == "EXIT"
    if has_exit:
        body = body[:-1]
    anchor_x, anchor_y, rest = _read_anchor(body, cell_origin)
    if not rest and not has_exit and cls.startswith(("POI_",)):
        return _point_feature(cls, anchor_x, anchor_y)
    if cls.startswith(("R_",)):
        return _line_feature(cls, anchor_x, anchor_y, rest, has_exit, cell_size_m, cell_origin)
    if cls.startswith(("B_", "L_")):
        if has_exit:
            raise UnsupportedGeometry(f"<EXIT> not valid for class {cls}")
        return _polygon_feature(cls, anchor_x, anchor_y, rest, cell_origin)
    raise UnsupportedGeometry(f"unknown class prefix for {cls!r}")


def _read_anchor(
    body: list[str],
    cell_origin: tuple[float, float],
) -> tuple[float, float, list[str]]:
    if len(body) < 2 or not body[0].startswith("ANCHOR_X_") or not body[1].startswith("ANCHOR_Y_"):
        raise UnsupportedGeometry("feature body must start with anchor X/Y pair")
    x = float(body[0].removeprefix("ANCHOR_X_")) + cell_origin[0]
    y = float(body[1].removeprefix("ANCHOR_Y_")) + cell_origin[1]
    return x, y, body[2:]


def _point_feature(cls: str, x: float, y: float) -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def _line_feature(
    cls: str,
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
    has_exit: bool,
    cell_size_m: float,
    cell_origin: tuple[float, float],
) -> dict:
    coords = _apply_moves_as_polyline(anchor_x, anchor_y, moves)
    if has_exit:
        end_x, end_y = coords[-1]
        local_x = end_x - cell_origin[0]
        local_y = end_y - cell_origin[1]
        if not _on_cell_boundary(local_x, local_y, cell_size_m):
            raise UnsupportedGeometry("<EXIT> token but final vertex not on cell boundary")
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": [list(p) for p in coords]},
    }


def _polygon_feature(
    cls: str,
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
    cell_origin: tuple[float, float],
) -> dict:
    coords = _apply_moves_as_polyline(anchor_x, anchor_y, moves)
    # Closure check: final cursor must equal anchor (within 1m grid).
    if (round(coords[-1][0]) != round(anchor_x)) or (round(coords[-1][1]) != round(anchor_y)):
        raise UnsupportedGeometry("polygon move sequence does not return to anchor")
    # GeoJSON polygon ring repeats the first vertex; drop the moves' trailing duplicate.
    ring = [list(p) for p in coords]
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


_DIR_TO_DELTA: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}


def _apply_moves_as_polyline(
    anchor_x: float,
    anchor_y: float,
    moves: list[str],
) -> list[tuple[float, float]]:
    """Apply move tokens, collapsing consecutive same-direction moves into one segment.

    Returns the vertex list starting with the anchor.
    """
    coords: list[tuple[float, float]] = [(anchor_x, anchor_y)]
    if not moves:
        return coords
    cur_x, cur_y = anchor_x, anchor_y
    seg_dir: str | None = None
    seg_len = 0
    for tok in moves:
        if not tok.startswith("MOVE_"):
            raise UnsupportedGeometry(f"expected MOVE_ token, got {tok!r}")
        _, direction, step_s = tok.split("_")
        if direction not in _DIR_TO_DELTA:
            raise UnsupportedGeometry(f"Phase 0 supports cardinal moves only; got {direction!r}")
        step = int(step_s)
        if seg_dir is None or direction == seg_dir:
            seg_dir = direction
            seg_len += step
        else:
            # Close the previous segment.
            dx, dy = _DIR_TO_DELTA[seg_dir]
            cur_x += dx * seg_len
            cur_y += dy * seg_len
            coords.append((cur_x, cur_y))
            seg_dir = direction
            seg_len = step
    # Close the trailing segment.
    if seg_dir is not None:
        dx, dy = _DIR_TO_DELTA[seg_dir]
        cur_x += dx * seg_len
        cur_y += dy * seg_len
        coords.append((cur_x, cur_y))
    return coords


def _on_cell_boundary(x: float, y: float, cell_size_m: float) -> bool:
    return x == 0 or x == cell_size_m or y == 0 or y == cell_size_m
