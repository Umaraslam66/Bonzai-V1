"""Sub-F cross-tile validator (Task 10).

Per spec §4.7 + §8.1 BP7 row: the BP7 four-test composite
(cross-reference, symmetry, non-road non-emission, coverage) PLUS
version-manifest consistency across tiles PLUS sha-uniqueness PLUS
all-64-distinct-cells-present per tile.

ALL-OF discipline (per BP2 fix 1 protocol-level lesson): every leg must
pass independently; a single failure halts with a leg-specific message.

LEG DECOMPOSITION (T9 rule-isolation standard, per
`feedback_gate_must_distinguish_regimes`): each BP7 leg is one private
function (`_check_cross_reference`, `_check_symmetry`,
`_check_non_road_non_emission`, `_check_coverage`,
`_check_version_consistency`, `_check_sha_uniqueness`,
`_check_all_cells_present`). This lets a negative test target ONE leg
in isolation and confirm the TARGET leg fires, not an incidentally-earlier
check. Each leg raises `CrossTileValidationError` with a distinct
leg-name substring in the message:
  - "cross-reference"
  - "symmetry"
  - "non-road"
  - "coverage"
  - "version manifest"
  - "sha" + "unique"
  - "cell" + "present" (or "distinct")

WHAT THIS MODULE READS (never writes — provenance/manifest/_SUCCESS are
Task 11):
  - sub-F `cells.parquet` per tile (token sequences) under
    `sub_f_region_dir/tile=*/cells.parquet`.
  - sub-F `provenance.yaml` per tile (version axes) — READ ONLY.
  - sub-E `boundary_contract.parquet` per tile under
    `sub_e_region_dir/tile=*/boundary_contract.parquet`, via
    `load_boundary_contract` (the source-derived reader; it raises
    `SubEContractViolation` on a malformed contract before this validator
    ever sees it).

WHAT THIS MODULE DOES NOT DO (Task 9 / Task 11 scope — do not duplicate):
  - per-cell schema / row-count / derivation / token-range / sha checks
    (Task 9 `validator_inline`);
  - pipeline orchestration, `_SUCCESS` touch, provenance/manifest writing
    (Task 11).

CARRY-FORWARD (do NOT let green BP7 tests imply this is validated):
sub-E tiers `motorway` as MINOR for v1 (a SCOPED accept, not a
data-validated decision). Cross-reference passes whenever encoder and
contract AGREE on MINOR-for-motorway; it does NOT validate that MINOR is
semantically correct. See close-checklist.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cfm.data.sub_f.boundary_contract import load_boundary_contract
from cfm.data.sub_f.encoder import endpoint_edge_direction
from cfm.data.sub_f.rotation import DIRECTION_ORDER
from cfm.data.sub_f.vocab import ROAD_L1_KEY, semantic_tag_to_l1_key, vocab_tag_to_id

# Sub-C feature_class for road (highway) features (sub_c/enums.py FEATURE_CLASS
# {0: "road", ...}; pipeline_writer._FEATURE_CLASS_TO_KEY maps 0 -> "highway").
# Only these emit brefs in the encoder, so only these populate road_edge_presence.
_ROAD_FEATURE_CLASS = 0

# Structural sentinels — must match encoder._FEATURE_TOKEN_ID /
# _FEATURE_END_TOKEN_ID. Sourced from encoder to keep a single point of
# truth; re-declared here as ints to avoid importing private encoder names
# into the validator's public surface.
_FEATURE_TOKEN_ID = 509
_FEATURE_END_TOKEN_ID = 510

# BP7 boundary-reference token range (LOCKED 1500-1507 per
# configs/sub_f/boundary_reference_vocab.yaml id_block). Used only for the
# fast range test below; the (id -> direction, class) decode is resolved
# from the LIVE vocab via vocab_tag_to_id() so a future relocation surfaces
# as a test failure rather than a silent pass.
_BREF_MIN_ID = 1500
_BREF_MAX_ID = 1507

# Direction order for the symmetric-pair (opposite-direction) adjacency.
# E <-> W share an axis-0 edge; S <-> N share an axis-1 edge (verified
# empirically against cell_to_edge_ids: A(0,0).east == B(1,0).west and
# A(0,0).south == B(0,1).north).
_OPPOSITE_DIRECTION: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Emitting classes (the only two that carry a <bref> token, per spec §3.7).
_EMITTING_CLASSES = frozenset({"MAJOR_ROAD", "MINOR_ROAD"})

# Semantic-tag key that denotes a road (LineString) feature. Only road
# features emit brefs (spec §1.4: buildings/POIs clipped at geometry layer,
# never at the token layer). Aliases the shared vocab authority so the encoder's
# emission gate and the validator's non-road leg agree on what counts as a road.
_ROAD_KEY = ROAD_L1_KEY


class CrossTileValidationError(ValueError):
    """Raised when a sub-F region fails any cross-tile invariant."""


def _bref_id_to_dir_class() -> dict[int, tuple[str, str]]:
    """Build {token_id -> (direction, CLASS)} for the 8 live BP7 tokens.

    Resolved from the LIVE vocab (vocab_tag_to_id) so a vocab relocation
    surfaces here, not as a silent miss. Tag form is `<bref_DIR_SHORT>`
    where DIR in {N,E,S,W} and SHORT in {MAJOR,MINOR}; mapped to the
    sub-E class label {MAJOR_ROAD, MINOR_ROAD}.
    """
    short_to_class = {"MAJOR": "MAJOR_ROAD", "MINOR": "MINOR_ROAD"}
    out: dict[int, tuple[str, str]] = {}
    for tag, token_id in vocab_tag_to_id().items():
        if not tag.startswith("<bref_") or not tag.endswith(">"):
            continue
        # "<bref_E_MAJOR>" -> ["E", "MAJOR"]
        inner = tag[len("<bref_") : -1]
        direction, short = inner.split("_", 1)
        out[token_id] = (direction, short_to_class[short])
    return out


def _semantic_id_to_tag() -> dict[int, str]:
    """Build {token_id -> tag} for all vocab slots (reverse of vocab_tag_to_id)."""
    return {token_id: tag for tag, token_id in vocab_tag_to_id().items()}


def _split_into_feature_chunks(token_sequence: list[int]) -> list[list[int]]:
    """Split a cell token sequence into per-feature chunks.

    Each chunk includes its <feature>(509) ... <feature_end>(510) markers.
    Mirrors the helper in tests/data/sub_f/test_pipeline_writer.py.
    """
    chunks: list[list[int]] = []
    i = 0
    n = len(token_sequence)
    while i < n:
        if token_sequence[i] == _FEATURE_TOKEN_ID:
            j = i + 1
            while j < n and token_sequence[j] != _FEATURE_END_TOKEN_ID:
                j += 1
            if j < n:
                chunks.append(token_sequence[i : j + 1])
                i = j + 1
            else:
                break  # malformed: no matching end (Task 9 catches token-range)
        else:
            i += 1
    return chunks


def _emitted_brefs_by_cell(
    cells_rows: list[dict],
    bref_decode: dict[int, tuple[str, str]],
    sem_id_to_tag: dict[int, str],
) -> dict[tuple[int, int], list[tuple[str, str, str]]]:
    """Extract emitted brefs per cell.

    Returns {(cell_i, cell_j): [(direction, CLASS, feature_key), ...]} where
    feature_key is the semantic-tag's key (e.g. "highway", "building"). One
    entry per emitted bref token. feature_key is the chunk's semantic tag
    (chunk[1]); it lets the non-road leg attribute a bref to a feature type.
    """
    out: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
    for r in cells_rows:
        cell = (int(r["cell_i"]), int(r["cell_j"]))
        emitted: list[tuple[str, str, str]] = []
        for chunk in _split_into_feature_chunks(list(r["token_sequence"])):
            # chunk[0] == <feature>; chunk[1] == semantic tag id.
            feature_key = "<malformed>"
            if len(chunk) >= 2:
                sem_tag = sem_id_to_tag.get(chunk[1], "")
                feature_key = semantic_tag_to_l1_key(sem_tag)
            for tok in chunk:
                if tok in bref_decode:
                    direction, cls = bref_decode[tok]
                    emitted.append((direction, cls, feature_key))
        if emitted:
            out[cell] = emitted
    return out


# --------------------------------------------------------------------------
# Leg 1: cross-reference
# --------------------------------------------------------------------------


def _check_cross_reference(
    tile_label: str,
    emitted_by_cell: dict[tuple[int, int], list[tuple[str, str, str]]],
    contract: dict[tuple[int, int], dict[str, str]],
) -> None:
    """Every emitted <bref_DIR_CLASS> must match sub-E's class for cell x edge.

    A cell that emits <bref_E_MAJOR> while sub-E's contract says MINOR_ROAD
    (or NONE) for that cell's East edge is a cross-reference failure.
    """
    for cell, emitted in emitted_by_cell.items():
        cell_edges = contract.get(cell, {})
        for direction, cls, _feature_key in emitted:
            contract_cls = cell_edges.get(direction, "NONE")
            if contract_cls != cls:
                raise CrossTileValidationError(
                    f"BP7 cross-reference failure at {tile_label} cell {cell} "
                    f"edge {direction}: emitted <bref_{direction}_{_SHORT(cls)}> "
                    f"(class {cls}) but sub-E contract says {contract_cls}."
                )


# --------------------------------------------------------------------------
# Leg 2: symmetry
# --------------------------------------------------------------------------


def _check_symmetry(
    tile_label: str,
    emitted_by_cell: dict[tuple[int, int], list[tuple[str, str, str]]],
    road_edge_presence: set[tuple[tuple[int, int], str]],
) -> None:
    """Paired adjacent cells must agree on their shared internal edge — CONDITIONED
    on road presence (validator v1.2 relax).

    For an internal shared edge, cell A's view (direction DIR) and neighbour B's
    view (opposite direction) reference the SAME sub-E row. The pre-1.2 leg
    required A and B to emit identically. That premise is FALSE for a road that
    TERMINATES exactly at an internal cell boundary (spec §8.3 touch-not-cross):
    the road is present (endpoint-on-edge) only on A's side, so A emits and B —
    which has no road endpoint there — correctly emits nothing. The asymmetry is
    faithful to the geometry, not a defect. The pre-1.2 leg false-positived on
    this and failed 8 batch-2 cities on 14 edges (see
    reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md).

    v1.2 conditions the leg on `road_edge_presence` — the set of (cell, direction)
    that carry a road LineString with an endpoint on that edge, derived from the
    sub-C geometry (the road authority, independent of the encoder). Disagreement
    rules:
      - both cells emit, different CLASS  -> raise (genuine class mismatch).
      - A emits, B silent, B HAS a road endpoint on the edge -> raise
        (under-emission: B should have emitted; the genuine defect-catching power
        the relax must preserve — the must-distinguish twin, fixture b).
      - A emits, B silent, B has NO road endpoint on the edge -> ALLOW
        (legit one-sided termination, fixture a).

    NOTE: this leg no longer catches a sub-C clip-DROP (a true crossing whose B
    fragment was dropped) — that is INVISIBLE here (B looks like a termination).
    The drop mode is covered by the sub-C lossless-clip length invariant
    (cfm.data.multiregion.lossless_clip) and the independent source-trace corpus
    gate; see the report.

    Adjacency within the 8x8 tile grid:
      DIR=E of (i,j)  <-> DIR=W of (i+1,j)
      DIR=N of (i,j)  <-> DIR=S of (i,j-1)
    (Direction labels are cell-local; N is -j, S is +j per sub-E rotation.)
    """
    # Aggregate the per-cell emitted classes into {(cell, dir): CLASS}.
    by_cell_dir: dict[tuple[tuple[int, int], str], set[str]] = {}
    for cell, emitted in emitted_by_cell.items():
        for direction, cls, _fk in emitted:
            by_cell_dir.setdefault((cell, direction), set()).add(cls)

    for (cell, direction), classes in by_cell_dir.items():
        opp = _OPPOSITE_DIRECTION[direction]
        neighbour = _neighbour_cell(cell, direction)
        if neighbour is None:
            # Tile-boundary (external) edge: no in-tile neighbour to pair
            # with. Cross-tile-stitch symmetry across separate tiles is a
            # v2/region-stitch concern; within-tile symmetry is the v1 gate.
            continue
        neighbour_classes = by_cell_dir.get((neighbour, opp), set())
        if classes == neighbour_classes:
            continue
        if neighbour_classes:
            # Both sides emit but disagree on CLASS -> genuine mismatch.
            raise CrossTileValidationError(
                f"BP7 symmetry failure at {tile_label}: cell {cell} edge "
                f"{direction} emits {sorted(classes)} but paired neighbour "
                f"{neighbour} edge {opp} emits {sorted(neighbour_classes)} "
                f"— shared-edge views disagree on class."
            )
        # Neighbour emits nothing on the shared edge. v1.2: a defect iff the
        # neighbour genuinely carries a road endpoint there (under-emission);
        # otherwise a legitimate §8.3 termination (road wholly on this side).
        if (neighbour, opp) in road_edge_presence:
            raise CrossTileValidationError(
                f"BP7 symmetry failure at {tile_label}: cell {cell} edge "
                f"{direction} emits {sorted(classes)} but paired neighbour "
                f"{neighbour} edge {opp} emits [] despite carrying a road "
                f"endpoint on the shared edge — under-emission."
            )
        # else: legit one-sided termination (§8.3) -> not a symmetry violation.


def _neighbour_cell(cell: tuple[int, int], direction: str) -> tuple[int, int] | None:
    """Return the in-tile neighbour across `direction`, or None if off-grid.

    Per sub-E rotation: N = (i, j-1), S = (i, j+1), W = (i-1, j), E = (i+1, j).
    """
    i, j = cell
    if direction == "N":
        nb = (i, j - 1)
    elif direction == "S":
        nb = (i, j + 1)
    elif direction == "W":
        nb = (i - 1, j)
    elif direction == "E":
        nb = (i + 1, j)
    else:  # pragma: no cover - DIRECTION_ORDER is N/E/S/W
        raise ValueError(f"unknown direction {direction!r}")
    if 0 <= nb[0] < 8 and 0 <= nb[1] < 8:
        return nb
    return None


# --------------------------------------------------------------------------
# Leg 3: non-road non-emission
# --------------------------------------------------------------------------


def _check_non_road_non_emission(
    tile_label: str,
    emitted_by_cell: dict[tuple[int, int], list[tuple[str, str, str]]],
) -> None:
    """Building / POI features must emit zero <bref> tokens.

    Only road (highway-keyed) features carry brefs (spec §1.4). A bref
    emitted from a feature whose semantic key is NOT `highway` is a
    non-road non-emission failure.
    """
    for cell, emitted in emitted_by_cell.items():
        for direction, cls, feature_key in emitted:
            if feature_key != _ROAD_KEY:
                raise CrossTileValidationError(
                    f"BP7 non-road non-emission failure at {tile_label} cell "
                    f"{cell}: a feature with key {feature_key!r} emitted "
                    f"<bref_{direction}_{_SHORT(cls)}> — only road (highway) "
                    f"features may emit boundary-ref tokens."
                )


# --------------------------------------------------------------------------
# Leg 4: coverage (AND symmetry on that emission)
# --------------------------------------------------------------------------


def _check_coverage(
    tile_label: str,
    emitted_by_cell: dict[tuple[int, int], list[tuple[str, str, str]]],
    contract: dict[tuple[int, int], dict[str, str]],
    road_edge_presence: set[tuple[tuple[int, int], str]],
) -> None:
    """An active edge with a road ENDPOINT in THIS cell must emit >=1 bref (v1.2).

    For each cell's active (MAJOR/MINOR) edge, if THIS cell has a road feature
    with an endpoint on that edge (the edge is in `road_edge_presence`), the cell
    must emit a bref on it.

    v1.2 CHANGE (validator 1.2): the condition is THIS cell's own road-endpoint
    presence, NOT the pre-1.2 `road_here OR road_neighbour` over road_cells. The
    old condition demanded emission from a road that TERMINATES on the neighbour's
    side (the neighbour is a road cell, so coverage required THIS empty side to
    emit a bref it cannot produce) — the §8.3 termination false positive that, with
    the symmetry leg, failed 8 batch-2 cities. Conditioning on the cell's own
    endpoint both removes that FP (termination side has no endpoint here) AND keeps
    the teeth: a cell that HAS a road endpoint on an active edge but emits nothing
    is a genuine under-emission and still raises (fixture cov-b). See
    reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md.
    """
    for cell, cell_edges in contract.items():
        emitted_dirs = {d for (d, _c, _fk) in emitted_by_cell.get(cell, [])}
        for direction in DIRECTION_ORDER:
            edge_class = cell_edges.get(direction, "NONE")
            if edge_class not in _EMITTING_CLASSES:
                continue  # NONE edge: nothing to cover
            if (cell, direction) not in road_edge_presence:
                # No road endpoint in THIS cell on this edge -> nothing to emit
                # (e.g. a road terminating on the neighbour's side). Do not over-fire.
                continue
            if direction not in emitted_dirs:
                raise CrossTileValidationError(
                    f"BP7 coverage failure at {tile_label} cell {cell}: edge "
                    f"{direction} is active ({edge_class}) with a road endpoint in "
                    f"this cell, but no <bref> emitted on that edge."
                )


# --------------------------------------------------------------------------
# Leg 5: version-manifest consistency
# --------------------------------------------------------------------------

_VERSION_AXES: tuple[str, ...] = (
    "sub_f_artifact_format_version",
    "sub_f_schema_version",
    "sub_f_vocab_version",
    "sub_f_derivation_version",
    "sub_f_validator_version",
    "sub_f_source_version",
)


def _check_version_consistency(tile_provenances: dict[str, dict]) -> None:
    """Every tile's provenance.yaml must carry an identical 6-axis version tuple.

    Uses the REAL version keys from src/cfm/data/sub_f/versions.py /
    manifest.py (verified at T10 write time): sub_f_artifact_format_version,
    sub_f_schema_version, sub_f_vocab_version, sub_f_derivation_version,
    sub_f_validator_version, sub_f_source_version.
    """
    versions_seen: dict[tuple, list[str]] = {}
    for tile_label, prov in tile_provenances.items():
        version_tuple = tuple(_freeze(prov.get(axis)) for axis in _VERSION_AXES)
        versions_seen.setdefault(version_tuple, []).append(tile_label)
    if len(versions_seen) != 1:
        groups = {
            ", ".join(sorted(labels)): dict(zip(_VERSION_AXES, vt, strict=True))
            for vt, labels in versions_seen.items()
        }
        raise CrossTileValidationError(f"version manifest inconsistent across tiles: {groups}")


def _freeze(value: object) -> object:
    """Make a provenance value hashable for set membership (dict -> sorted tuple)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


# --------------------------------------------------------------------------
# Leg 6: sha-uniqueness across tiles
# --------------------------------------------------------------------------


def _check_sha_uniqueness(tile_provenances: dict[str, dict]) -> None:
    """Every tile's provenance.yaml must carry a DISTINCT `provenance_sha256`.

    A real tile's provenance sha is content-derived (provenance.py
    `provenance_sha256`), so two tiles with genuinely distinct provenance
    content always produce distinct shas. A duplicate sha either means two
    tiles with identical provenance content (a pipeline bug) or a stub/
    placeholder sha that was not replaced (also a pipeline bug).

    Raises CrossTileValidationError with "sha" and "unique" in the message
    if any two tiles share the same sha, or if any tile's provenance.yaml
    is missing the `provenance_sha256` field entirely.
    """
    seen: dict[str, str] = {}  # sha -> first tile_label that carries it
    for tile_label, prov in tile_provenances.items():
        sha = prov.get("provenance_sha256")
        if sha is None:
            raise CrossTileValidationError(
                f"sha unique check failed: tile {tile_label} provenance.yaml is "
                f"missing the required `provenance_sha256` field — every tile "
                f"manifest must carry its self-integrity sha."
            )
        if sha in seen:
            raise CrossTileValidationError(
                f"sha not unique across tiles: {tile_label} and {seen[sha]} share "
                f"identical provenance_sha256 {sha!r} — each tile must have a "
                f"distinct content-derived sha."
            )
        seen[sha] = tile_label


# --------------------------------------------------------------------------
# Leg 7: all-64-distinct-cells-present per tile
# --------------------------------------------------------------------------


def _check_all_cells_present(tile_label: str, cells_table: pa.Table) -> None:
    """The tile's cells.parquet must contain exactly the 64 distinct (cell_i,
    cell_j) pairs that span the full 8x8 grid (every (i,j) with 0<=i,j<8).

    Raises CrossTileValidationError with "cell" and "present" in the message if:
    - Any (cell_i, cell_j) pair is missing from the grid.
    - Any (cell_i, cell_j) pair is duplicated (two rows for the same cell).

    This is the region-level complement to Task 9's inline row-count==64 check:
    T9 catches count!=64, but a table with exactly 64 rows can still have a
    duplicate cell and a missing cell — those row-count-correct anomalies are
    what this leg catches.
    """
    cell_i_col = cells_table.column("cell_i").to_pylist()
    cell_j_col = cells_table.column("cell_j").to_pylist()

    expected: set[tuple[int, int]] = {(i, j) for i in range(8) for j in range(8)}
    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []

    for i, j in zip(cell_i_col, cell_j_col, strict=True):
        pair = (int(i), int(j))
        if pair in seen:
            duplicates.append(pair)
        seen.add(pair)

    missing = sorted(expected - seen)
    if duplicates or missing:
        parts: list[str] = []
        if duplicates:
            parts.append(f"duplicate cells {sorted(set(duplicates))}")
        if missing:
            parts.append(f"missing cells {missing}")
        raise CrossTileValidationError(
            f"cell not present: tile {tile_label} cells.parquet has "
            f"{'; '.join(parts)} — all 64 distinct (cell_i, cell_j) pairs "
            f"(0..7 x 0..7) must be present exactly once."
        )


def _SHORT(class_label: str) -> str:
    """Short BP7 token suffix for a sub-E class label (for messages only)."""
    return "MAJOR" if class_label == "MAJOR_ROAD" else "MINOR"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def validate_cross_tile(
    sub_f_region_dir: Path, sub_e_region_dir: Path, sub_c_region_dir: Path
) -> None:
    """Validate a sub-F region against all cross-tile invariants.

    Raises CrossTileValidationError on the first leg that fails, with a
    leg-specific substring in the message.

    DECISION (validator v1.2): added a third argument `sub_c_region_dir`. The
    symmetry (leg 2) and coverage (leg 4) legs are now road-presence-conditioned
    — they need to know, for each cell-edge, whether THIS cell carries a road
    feature with an endpoint on that edge (`road_edge_presence`). That signal is
    derived from sub-C `features.parquet` (the CLEAN artifact — 0-d/sliver clips
    are discarded before write, so a §8.3 touch-as-cross road is absent there).
    It is deliberately NOT derived from sub-C `crossings.parquet`, whose
    `_both_cells_present`/`per_cell_pieces` path (pre-discard) carries the spurious
    touch-as-cross records this fix tolerates. The independent source-trace gate
    stays the separate drop backstop. See
    reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md.

    Per feedback_pyarrow_hive_partition_inference: cells.parquet are read
    via load_boundary_contract / pq.ParquetFile(path).read(), never bare
    pq.read_table() on the partitioned parent.
    """
    tile_paths = sorted(sub_f_region_dir.glob("tile=*/cells.parquet"))
    if not tile_paths:
        raise CrossTileValidationError(f"no tiles under {sub_f_region_dir}")

    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()

    # --- version-manifest consistency (leg 5) ---
    tile_provenances: dict[str, dict] = {}
    for tile_path in tile_paths:
        prov_path = tile_path.parent / "provenance.yaml"
        if not prov_path.exists():
            raise CrossTileValidationError(
                f"missing provenance.yaml for tile {tile_path.parent.name}"
            )
        tile_provenances[tile_path.parent.name] = yaml.safe_load(prov_path.read_text())
    _check_version_consistency(tile_provenances)
    _check_sha_uniqueness(tile_provenances)  # leg 6: distinct content-derived shas

    # --- per-tile BP7 four-test composite (legs 1-4) + all-cells-present (leg 7) ---
    for tile_path in tile_paths:
        tile_label = tile_path.parent.name
        sub_e_contract_path = sub_e_region_dir / tile_path.parent.name / "boundary_contract.parquet"
        if not sub_e_contract_path.exists():
            raise CrossTileValidationError(
                f"missing sub-E boundary_contract.parquet for tile {tile_label} "
                f"at {sub_e_contract_path}"
            )
        contract = load_boundary_contract(sub_e_contract_path)

        cells_table = pq.ParquetFile(tile_path).read()
        _check_all_cells_present(tile_label, cells_table)  # leg 7: full 8x8 grid

        cells_rows = cells_table.to_pylist()
        emitted_by_cell = _emitted_brefs_by_cell(cells_rows, bref_decode, sem_id_to_tag)
        road_edge_presence = _build_road_edge_presence(sub_c_region_dir / tile_label)

        _check_cross_reference(tile_label, emitted_by_cell, contract)
        _check_symmetry(tile_label, emitted_by_cell, road_edge_presence)
        _check_non_road_non_emission(tile_label, emitted_by_cell)
        _check_coverage(tile_label, emitted_by_cell, contract, road_edge_presence)


def _build_road_edge_presence(
    sub_c_tile_dir: Path,
) -> set[tuple[tuple[int, int], str]]:
    """Set of (cell, direction) where a ROAD (feature_class==0) LineString has an
    endpoint on that cell edge — i.e. where the encoder's gate WOULD emit a bref.

    Recomputed from sub-C `features.parquet` (the CLEAN artifact: 0-d Point
    collapses and <0.01 m slivers are discarded before write per geom.py:200-202 /
    224-225, so a §8.3 touch-as-cross road is ABSENT here). Deliberately NOT from
    `crossings.parquet`, which carries the spurious touch-as-cross records (its
    `_both_cells_present` tests `per_cell_pieces`, pre-discard). Endpoint ->
    direction routes through the shared `encoder.endpoint_edge_direction`
    authority so the validator and the encoder cannot drift on the N/S convention.

    Used by the symmetry (leg 2) and coverage (leg 4) legs to tell a legit
    one-sided termination (neighbour has no road endpoint on the edge -> allow)
    from an under-emission (neighbour HAS one but didn't emit -> raise). A missing
    sub-C tile dir / features.parquet yields the empty set (the legs then treat all
    one-sided emission as termination — the drop mode is backstopped separately by
    the sub-C length invariant + source-trace gate).
    """
    from shapely import wkb as _wkb

    feats_path = sub_c_tile_dir / "features.parquet"
    if not feats_path.exists():
        return set()
    presence: set[tuple[tuple[int, int], str]] = set()
    for r in pq.ParquetFile(feats_path).read().to_pylist():
        if int(r["feature_class"]) != _ROAD_FEATURE_CLASS:
            continue  # only highway features emit brefs (§1.4)
        geom = _wkb.loads(r["geometry"])
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            continue
        cell = (int(r["cell_i"]), int(r["cell_j"]))
        for line in parts:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for px, py in (coords[0], coords[-1]):
                direction = endpoint_edge_direction(px, py)
                if direction is not None:
                    presence.add((cell, direction))
    return presence
