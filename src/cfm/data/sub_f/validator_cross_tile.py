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
from cfm.data.sub_f.rotation import DIRECTION_ORDER
from cfm.data.sub_f.vocab import ROAD_L1_KEY, semantic_tag_to_l1_key, vocab_tag_to_id

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
) -> None:
    """Paired adjacent cells must agree on their shared internal edge.

    For an internal shared edge, cell A's view (direction DIR) and the
    neighbour cell B's view (opposite direction) reference the SAME sub-E
    row, so any bref A emits on DIR must be matched by B emitting the same
    CLASS on the opposite direction. A break that survives cross-reference
    is reachable only via an adjacency-mapping bug — this leg validates the
    E<->W / S<->N opposite-direction mapping independently.

    Adjacency within the 8x8 tile grid:
      DIR=E of (i,j)  <-> DIR=W of (i+1,j)
      DIR=N of (i,j)  <-> DIR=S of (i,j-1)
    (Direction labels are cell-local; N is -j, S is +j per sub-E rotation.)
    """
    # Aggregate the per-cell emitted classes into {(cell, dir): CLASS}.
    # If a cell emits >1 class on one direction (encoder bug), that is itself
    # a defect — but symmetry compares against the neighbour, so we collapse
    # to the set and require a single shared class on each side.
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
        if classes != neighbour_classes:
            raise CrossTileValidationError(
                f"BP7 symmetry failure at {tile_label}: cell {cell} edge "
                f"{direction} emits {sorted(classes)} but paired neighbour "
                f"{neighbour} edge {opp} emits {sorted(neighbour_classes)} "
                f"— shared-edge views disagree."
            )


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
    road_cells: set[tuple[int, int]],
) -> None:
    """Active road edges with a road feature in either neighbour emit >=1 bref.

    For each cell's active (MAJOR/MINOR) edge, if a road feature is present
    in this cell OR the in-tile neighbour across that edge, then this cell
    must emit at least one bref on that edge. We do NOT fire on active edges
    where neither this cell nor the neighbour carries a road feature (no
    feature to attach a crossing to). Per the BP7-fix-3 reviewer note,
    coverage is an explicit conjunction: emission present AND symmetry holds
    — symmetry is enforced by the dedicated leg above (run before coverage),
    so coverage here asserts the emission-present half on edges that demand it.
    """
    for cell, cell_edges in contract.items():
        emitted_dirs = {d for (d, _c, _fk) in emitted_by_cell.get(cell, [])}
        for direction in DIRECTION_ORDER:
            edge_class = cell_edges.get(direction, "NONE")
            if edge_class not in _EMITTING_CLASSES:
                continue  # NONE edge: nothing to cover
            neighbour = _neighbour_cell(cell, direction)
            road_here = cell in road_cells
            road_neighbour = neighbour is not None and neighbour in road_cells
            if not (road_here or road_neighbour):
                # Active edge but no road feature on either side to attach a
                # crossing to — do not over-fire.
                continue
            if direction not in emitted_dirs:
                raise CrossTileValidationError(
                    f"BP7 coverage failure at {tile_label} cell {cell}: edge "
                    f"{direction} is active ({edge_class}) with a road feature "
                    f"in this cell or its neighbour, but no <bref> emitted on "
                    f"that edge."
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


def validate_cross_tile(sub_f_region_dir: Path, sub_e_region_dir: Path) -> None:
    """Validate a sub-F region against all cross-tile invariants.

    Raises CrossTileValidationError on the first leg that fails, with a
    leg-specific substring in the message.

    DECISION: chose a two-argument signature
    `(sub_f_region_dir, sub_e_region_dir)` over the master-plan stub's
    single-argument form because the cross-reference + coverage legs MUST
    compare emitted brefs against the sub-E boundary contract, which lives
    in the sub-E region tree — the single-arg stub could not implement
    those legs. Judged LOCAL (not load-bearing): T10 only reads, and the
    sole caller is the Task 11 orchestrator, which already holds both region
    roots. Revisit if a caller appears that has only the sub-F dir.

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
        road_cells = _road_cells(cells_rows, sem_id_to_tag)

        _check_cross_reference(tile_label, emitted_by_cell, contract)
        _check_symmetry(tile_label, emitted_by_cell)
        _check_non_road_non_emission(tile_label, emitted_by_cell)
        _check_coverage(tile_label, emitted_by_cell, contract, road_cells)


def _road_cells(
    cells_rows: list[dict],
    sem_id_to_tag: dict[int, str],
) -> set[tuple[int, int]]:
    """Cells that contain at least one road (highway-keyed) feature.

    Used by the coverage leg's "road feature in either neighbour" condition.
    A road feature is a feature chunk whose semantic tag (chunk[1]) has key
    `highway`.
    """
    out: set[tuple[int, int]] = set()
    for r in cells_rows:
        cell = (int(r["cell_i"]), int(r["cell_j"]))
        for chunk in _split_into_feature_chunks(list(r["token_sequence"])):
            if len(chunk) < 2:
                continue
            sem_tag = sem_id_to_tag.get(chunk[1], "")
            key = semantic_tag_to_l1_key(sem_tag)
            if key == _ROAD_KEY:
                out.add(cell)
                break
    return out
