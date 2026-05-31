"""Seam 2: sub-E boundary contract <-> sub-F cell tokens. Transcription bijection.

Independence (design rule 2 / spec Decision 3b): the EXPECTED bref multiset is
recomputed from sub-E's raw parquet + sub-C geometry, never via
``sub_f.boundary_contract.load_boundary_contract`` or
``sub_f.encoder._classify_feature_for_bref``. The ACTUAL multiset is parsed from
sub-F tokens. A bijection mismatch is a TRANSCRIPTION failure (sub-F dropped or
invented a bref). Semantic class-correctness (is MAJOR right?) is OUT of scope —
deferred per sub-G design §8 (motorway-tiering trigger).

Provenance chain (T5 Step-0, 2026-05-31): class <- sub-E enum spec
(sub_e/derivation.py:19-23); active/NULL <- sub-E invariant
(sub_e/validator_inline.py:169-178); only-MAJOR/MINOR-emit <- BP7 vocab
(boundary_reference_vocab.yaml + sub-F design §3.7); edge <- road endpoint on the
250m lattice (sub-C geometry + PRD §5 stage four). NOT circular-by-provenance
(unlike SI-3): the correspondence rule is a written design clause implemented
here independently, with a 0.5m edge tolerance (BP5 quantum) chosen here — NOT
copied from sub-F's 1e-6.

bref IDs (boundary_reference_vocab.yaml:29-68):
  1500 N_MAJOR 1501 E_MAJOR 1502 S_MAJOR 1503 W_MAJOR
  1504 N_MINOR 1505 E_MINOR 1506 S_MINOR 1507 W_MINOR
feature split (encoder.py:214-215): <feature>=509 <feature_end>=510.
"""

from __future__ import annotations

from collections import Counter

from shapely.geometry.base import BaseGeometry
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_e.rotation import EdgeKind
from cfm.data.sub_f.rotation import cell_edge_directions  # pure lattice geometry (cites sub-E)
from cfm.data.sub_g.diagnostics import Diagnostic
from cfm.data.sub_g.readers import SubEContractRow

_BREF_LO, _BREF_HI = 1500, 1507
_EDGE_TOL_M = 0.5  # absorbs canonicalization (BP5 magnitude quantum); NOT 1e-6
_CELL_EXTENT_M = 250.0

_BREF_ID_MAP: dict[int, tuple[str, str]] = {
    1500: ("N", "MAJOR_ROAD"),
    1501: ("E", "MAJOR_ROAD"),
    1502: ("S", "MAJOR_ROAD"),
    1503: ("W", "MAJOR_ROAD"),
    1504: ("N", "MINOR_ROAD"),
    1505: ("E", "MINOR_ROAD"),
    1506: ("S", "MINOR_ROAD"),
    1507: ("W", "MINOR_ROAD"),
}


def bref_id_to_dir_class(token_id: int) -> tuple[str, str]:
    return _BREF_ID_MAP[token_id]


def parse_actual_brefs_per_cell(token_sequence: list[int]) -> list[tuple[str, str]]:
    """Collect every bref token's (dir, class) from a cell's flat token sequence."""
    return [_BREF_ID_MAP[t] for t in token_sequence if _BREF_LO <= t <= _BREF_HI]


def _endpoint_edge(
    x: float, y: float, extent: float = _CELL_EXTENT_M, tol: float = _EDGE_TOL_M
) -> str | None:
    if abs(x) <= tol:
        return "W"
    if abs(x - extent) <= tol:
        return "E"
    if abs(y) <= tol:
        return "S"
    if abs(y - extent) <= tol:
        return "N"
    return None


def build_cell_contracts(rows: list[SubEContractRow]) -> dict[tuple[int, int], dict[str, str]]:
    """Independent (cell)->{dir->class} map from raw sub-E rows.

    Class resolution is sub-G's own (SubEContractRow.class_label, derived from the
    sub-E enum spec) — NOT sub_f.boundary_contract.load_boundary_contract. Only the
    lattice join (cell-dir -> edge id) reuses cell_edge_directions, a pure
    lattice-geometry helper (plan-permitted; cites sub-E).
    """
    join: dict[tuple[int, int, int, int], str] = {}
    for r in rows:
        join[(r.slot_kind, r.lower_cell_i, r.lower_cell_j, r.axis)] = r.class_label() or "NONE"

    contracts: dict[tuple[int, int], dict[str, str]] = {}
    for cell_i in range(8):
        for cell_j in range(8):
            edges = cell_edge_directions(cell_i, cell_j)
            cell: dict[str, str] = {}
            for direction in ("N", "E", "S", "W"):
                lower_i, lower_j, axis, kind = edges[direction]
                slot_kind = 1 if kind is EdgeKind.INTERNAL else 2
                cell[direction] = join.get(
                    (slot_kind, int(lower_i), int(lower_j), int(axis)), "NONE"
                )
            contracts[(cell_i, cell_j)] = cell
    return contracts


def _road_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Yield LineString parts of a road geometry (mirrors encode_cell Multi* split)."""
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    return []


def predict_expected_brefs_per_cell(
    features: list[dict], cell_contract: dict[str, str]
) -> list[tuple[str, str]]:
    """For each road LineString endpoint on an edge whose contract class is
    MAJOR/MINOR, expect a (dir, class) bref. Mirrors encode_cell's per-part split.
    """
    expected: list[tuple[str, str]] = []
    for f in features:
        if int(f["feature_class"]) != 0:  # ROAD only emits brefs
            continue
        for part in _road_parts(wkb_loads(bytes(f["geometry"]))):
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            for endpoint in (coords[0], coords[-1]):
                d = _endpoint_edge(endpoint[0], endpoint[1])
                if d is None:
                    continue
                cls = cell_contract.get(d, "NONE")
                if cls in ("MAJOR_ROAD", "MINOR_ROAD"):
                    expected.append((d, cls))
    return expected


def check_cell_bijection(
    tile_id: str,
    cell: tuple[int, int],
    expected: list[tuple[str, str]],
    actual: list[tuple[str, str]],
) -> list[Diagnostic]:
    """Compare expected vs actual bref multisets for one cell (both directions)."""
    ce, ca = Counter(expected), Counter(actual)
    if ce == ca:
        return []
    missing = sorted(str(x) for x in (ce - ca).elements())  # sub-E said, sub-F didn't emit
    extra = sorted(str(x) for x in (ca - ce).elements())  # sub-F emitted, unjustified
    if missing and not extra:
        signature = "bref missing (sub-F dropped)"
    elif extra and not missing:
        signature = "bref extra (sub-F invented)"
    else:
        signature = "bref multiset mismatch (both missing+extra)"
    return [
        Diagnostic(
            tile_id=tile_id,
            invariant_name="bref_bijection_contract_vs_tokens",
            artifact_left=f"predicted(sub_e+sub_c) cell={cell}",
            observed_left=missing,
            artifact_right="emitted(sub_f tokens)",
            observed_right=extra,
            expected_relationship="per-cell expected bref multiset == emitted bref multiset",
            spec_clause_citation="PRD §5 + boundary_reference_vocab.yaml + sub_e/writer.py:38-48",
            signature=signature,
        )
    ]
