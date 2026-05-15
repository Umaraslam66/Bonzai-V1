# Phase 0 — Tokenizer Round-Trip Design

- **Date:** 2026-05-15
- **Phase:** 0 (Setup, per PRD §11)
- **Status:** Draft, pending user review
- **Owner:** umar

## 1. Goal

Build the smallest end-to-end deliverable that proves Phase 0 is "done" per PRD §11: a new contributor can clone the repo, run `pytest`, and watch a hand-built GeoJSON cell encode to tokens and decode back to a geometrically equivalent GeoJSON.

The tokenizer round-trip is the single most load-bearing contract in the project. If encode → decode is lossy, every downstream model is trained on a lie. We get this contract right before touching real Overture data.

This spec covers **only Phase 0**. Real Overture data, multi-cell tiles, boundary contracts, macro plans, stitching, and the full ~100-class vocabulary all land in Phase 1.

## 2. Scope (in / out)

**In scope for Phase 0:**

- A single-cell, hand-built GeoJSON fixture (cell-local metric coordinates).
- A small Python package `cfm.tokenizer` exposing `encode_cell`, `decode_cell`, `Vocabulary`, `CellTokens`, and a typed error hierarchy.
- A vocabulary spec at `docs/tokenizer/vocabulary.md` plus the canonical YAML at `configs/tokenizer/vocab_phase0.yaml`.
- A `geometric_equal` helper using `shapely` for round-trip assertions.
- Three Phase-0 tests: geometric round-trip, deterministic encoding, loud failures on bad input.
- Repo scaffolding sufficient to run the tests on the user's Mac: `pyproject.toml` managed by `uv`, `ruff`, `pytest`, a CI workflow that runs lint + tests on PRs.

**Out of scope for Phase 0:**

- Overture loading or any real-world data.
- Multi-cell tiles, boundary contracts, deterministic stitching.
- Macro-plan generation.
- The full ~100-class semantic catalog (Phase-0 vocabulary is a deliberately curated subset).
- Any model training, even a toy one.
- Leonardo job submission. The Phase-0 smoke runs on the user's laptop; Leonardo wiring waits until Phase 2.

## 3. Load-bearing decisions

These are deliberately called out so they're easy to revisit when Phase 1 hits.

1. **Single-cell scope.** "Road crosses a cell boundary" is interpreted at Phase 0 as "road exits the cell," encoded by an `<EXIT>` control token at the road's last in-cell vertex. Multi-cell stitching, which would require boundary-contract derivation and a deterministic stitcher, is deferred to Phase 1. Rationale: PRD §11 places those in Phase 1 and the "small before big" principle (CLAUDE.md) says we don't pre-build them.
2. **Cell-local metric coordinates.** Fixtures are authored directly in metres with the cell's south-west corner at `(0, 0)`. No reprojection in Phase 0. Lat/lon → metric reprojection becomes a Phase-1 data-pipeline concern.
3. **Anchor grid at 1 m on a 250 m cell.** Anchors are emitted as a pair `<ANCHOR_X_n> <ANCHOR_Y_n>` rather than a single 62 500-token index. Splitting x and y keeps the absolute-position vocabulary at 500 tokens, leaving headroom for rare classes during training.
4. **Move tokens: 8 directions × 6 dyadic steps = 48.** Cardinal-direction frequency in training data is expected to produce the right-angle bias the PoC observed (95 % perfect right angles). Diagonals are reachable but uncommon.
5. **Geometric equivalence, not byte-identical, is the round-trip contract.** GeoJSON key order, whitespace, and number precision finer than the 1 m anchor grid must not fail the test. Class labels match exactly; geometries match within `tol_m` (default 0.5 m).
6. **Errors are loud.** Unsupported input raises a specific `TokenizerError` subclass rather than producing silent garbage. Three negative fixtures cover the three failure modes.

## 4. Token vocabulary — Phase 0 sketch

Reference doc: `docs/tokenizer/vocabulary.md`. Canonical IDs: `configs/tokenizer/vocab_phase0.yaml`. The vocabulary is frozen for Phase 0 and reopened in Phase 1 against Overture frequency analysis (PRD §5).

| Category | Count | Examples |
|---|---|---|
| Control | 8 | `<PAD>`, `<BOS>`, `<EOS>`, `<CELL>`, `<END_CELL>`, `<FEATURE_START>`, `<FEATURE_END>`, `<EXIT>` |
| Hierarchy (reserved) | 4 | `<MACRO>`, `<END_MACRO>`, `<MICRO>`, `<END_MICRO>` |
| Feature class | ~30 | Roads: `R_motorway`, `R_primary`, `R_secondary`, `R_residential`, `R_service`. Buildings: `B_residential`, `B_commercial`, `B_industrial`. POIs: `POI_restaurant`, `POI_school`, `POI_retail`, `POI_park_amenity`, `POI_transit_stop`. Land use: `L_residential`, `L_commercial`, `L_industrial`, `L_park`, `L_water`, `L_agricultural`. |
| Anchor X | 250 | `<ANCHOR_X_0>` … `<ANCHOR_X_249>` |
| Anchor Y | 250 | `<ANCHOR_Y_0>` … `<ANCHOR_Y_249>` |
| Move | 48 | 8 compass directions × 6 dyadic step sizes {1, 2, 4, 8, 16, 32 m}, e.g. `MOVE_E_8`, `MOVE_NW_2`. |
| **Total** | **~590** | within PRD's 1 000–3 000 budget |

Schematic token sequence for a cell:

```
<BOS> <CELL>
  <FEATURE_START> R_residential
    <ANCHOR_X_0> <ANCHOR_Y_125>
    MOVE_E_32 MOVE_E_32 MOVE_E_32 ... <EXIT>
  <FEATURE_END>
  <FEATURE_START> B_residential
    <ANCHOR_X_40> <ANCHOR_Y_40>
    MOVE_E_16 MOVE_N_16 MOVE_W_16 MOVE_S_16
  <FEATURE_END>
  <FEATURE_START> POI_restaurant
    <ANCHOR_X_50> <ANCHOR_Y_80>
  <FEATURE_END>
  <FEATURE_START> L_residential
    <ANCHOR_X_100> <ANCHOR_Y_0>
    MOVE_E_32 MOVE_E_32 ... MOVE_N_32 ... MOVE_W_32 ... MOVE_S_32 ...
  <FEATURE_END>
<END_CELL> <EOS>
```

A polygon is closed implicitly by the decoder when its move sequence returns to the anchor; the decoder validates closure and raises if it does not. A line carrying an `<EXIT>` token must end on a cell edge; the decoder validates and raises if it does not.

## 5. Tokenizer interface

File: `src/cfm/tokenizer/__init__.py`.

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

TokenId = int
GeoJSON = dict  # RFC 7946 FeatureCollection

@dataclass(frozen=True)
class Vocabulary:
    id_to_token: tuple[str, ...]
    token_to_id: dict[str, TokenId]

    @classmethod
    def load(cls, path: Path) -> "Vocabulary": ...
    def __len__(self) -> int: ...

@dataclass(frozen=True)
class CellTokens:
    tokens: tuple[TokenId, ...]
    cell_origin: tuple[float, float]   # SW corner, tile-local metres
    cell_size_m: float                  # 250.0 in Phase 0

class TokenizerError(ValueError): ...
class UnsupportedFeatureClass(TokenizerError): ...
class UnsupportedGeometry(TokenizerError): ...
class FeatureOutOfBounds(TokenizerError): ...
class VocabularyMismatch(TokenizerError): ...

def encode_cell(
    geojson: GeoJSON, *,
    cell_origin: tuple[float, float],
    cell_size_m: float,
    vocab: Vocabulary,
) -> CellTokens: ...

def decode_cell(
    tokens: CellTokens, *,
    vocab: Vocabulary,
) -> GeoJSON: ...

def geometric_equal(
    a: GeoJSON, b: GeoJSON, *,
    tol_m: float = 0.5,
) -> bool: ...
```

Notes on semantics:

- `encode_cell` raises `UnsupportedFeatureClass` if any feature's class is not in the vocabulary, `UnsupportedGeometry` for geometries we don't yet handle (e.g. non-rectangular buildings in Phase 0), and `FeatureOutOfBounds` for features whose geometry extends outside the cell except as a deliberate boundary exit.
- `decode_cell` is the structural inverse. It raises `VocabularyMismatch` for unknown token IDs and `UnsupportedGeometry` for structurally invalid sequences (e.g. a polygon whose move sequence does not return to its anchor, or a line carrying `<EXIT>` that does not end on a cell edge).
- `geometric_equal` matches features greedily within each class: for class `C` with `n` features in each input, it pairs each feature in `a` with the unmatched feature in `b` of minimum distance, then checks every pair is within `tol_m`. Returns `False` if class counts differ. Hausdorff distance is used for lines and polygons, Euclidean for points. Tolerance defaults to 0.5 m (half the 1 m anchor grid) so quantisation can never fail the assertion. The Phase-0 fixture has exactly one feature per class; the greedy rule still applies but is trivial.
- All public names get `from __future__ import annotations` and full type hints per CLAUDE.md code style.

## 6. Fixture structure

```
tests/fixtures/
├── single_cell/
│   ├── README.md          # describes the fixture, why each feature exists
│   ├── input.geojson      # hand-built, cell-local metres
│   └── expected.yaml      # class counts and key geometric facts for sanity tests
└── degenerate/
    ├── non_rectangular_building.geojson   # triggers UnsupportedGeometry
    ├── unknown_class.geojson              # triggers UnsupportedFeatureClass
    └── out_of_bounds.geojson              # triggers FeatureOutOfBounds
```

`single_cell/input.geojson` contains, in a 250 m × 250 m cell (cell-local metres, SW corner at origin):

| Feature | Geometry | Class |
|---|---|---|
| Road (exits E edge) | LineString `(0, 125) → (250, 125)` | `R_residential` |
| Rectangular building | Polygon `(40, 40) – (60, 40) – (60, 60) – (40, 60)` | `B_residential` |
| POI | Point `(50, 80)` | `POI_restaurant` |
| Land use | Polygon `(100, 0) – (250, 0) – (250, 150) – (100, 150)` | `L_residential` |

Small enough to read in your head. Exercises a line with `<EXIT>`, a closed rectangular polygon, a point, and a multi-vertex polygon — the four geometric primitives the tokenizer must handle in Phase 0.

## 7. Tests

All under `tests/tokenizer/`.

1. **`test_round_trip_geometric_equivalence`** — load fixture, `encode_cell`, `decode_cell`, assert `geometric_equal(orig, decoded)`.
2. **`test_token_sequence_is_deterministic`** — encoding the same fixture twice yields identical `CellTokens.tokens` tuples.
3. **`test_degenerate_inputs_raise_clear_errors`** — parametrised over the three negative fixtures, asserts the exact `TokenizerError` subclass.

Plus supporting unit tests:

- `test_vocabulary_load_roundtrip` — load YAML, dump back, identical.
- `test_geometric_equal_basics` — sanity: equal geometries equal, nudged-by-0.6 m geometries not equal.

Every test must run in < 5 s (CLAUDE.md testing convention).

## 8. Repo scaffolding required for Phase 0

```
Bonzai-OSM/
├── PRD.md
├── CLAUDE.md
├── README.md                         # quick-start
├── pyproject.toml                    # ruff, pytest, project metadata
├── uv.lock                           # pinned dependencies
├── .gitignore
├── .github/workflows/ci.yml          # lint + tests on PR
├── configs/
│   └── tokenizer/
│       └── vocab_phase0.yaml         # canonical Phase-0 vocabulary
├── docs/
│   ├── LEONARDO_REFERENCE.md
│   ├── tokenizer/
│   │   └── vocabulary.md             # human-readable vocab reference
│   └── superpowers/specs/
│       └── 2026-05-15-phase-0-tokenizer-roundtrip-design.md  # this file
├── src/cfm/
│   ├── __init__.py
│   └── tokenizer/
│       ├── __init__.py               # public API
│       ├── vocabulary.py             # Vocabulary class
│       ├── encode.py                 # encode_cell
│       ├── decode.py                 # decode_cell
│       └── errors.py                 # TokenizerError hierarchy
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── single_cell/
│   │   └── degenerate/
│   └── tokenizer/
│       ├── test_round_trip.py
│       ├── test_determinism.py
│       ├── test_errors.py
│       ├── test_vocabulary.py
│       └── test_geometric_equal.py
└── scripts/
    └── smoke.py                      # convenience CLI: load fixture, round-trip, print result
```

Dependencies (pinned in `uv.lock`):

- runtime: `shapely`, `pydantic`, `pyyaml`
- dev: `pytest`, `ruff`, `mypy` (optional, evaluate after Phase 0)

`README.md` documents the Phase-0 smoke: `uv sync && uv run pytest && uv run python scripts/smoke.py`.

## 9. Decision point at end of Phase 0

Per PRD §11, Phase 0 ends on: *can a new contributor clone the repo and reproduce a tiny end-to-end run?*

We say yes when:

- `git clone … && cd Bonzai-OSM && uv sync && uv run pytest` is green on a clean macOS or Linux machine.
- `uv run python scripts/smoke.py` prints a deterministic summary including the token IDs of the round-tripped fixture and the result of `geometric_equal`.
- CI is green on the main branch.

## 10. Open questions / explicitly deferred

- **Real anchor resolution.** 1 m on a 250 m cell is a Phase-0 placeholder. Phase 1 may show that 0.5 m, 2 m, or a hierarchical (coarse + offset) scheme is better suited to actual building/road sizes.
- **Move-token set.** Dyadic steps {1, 2, 4, 8, 16, 32} m are a guess. Phase 1 frequency analysis on real Overture geometries will determine the right set.
- **Polygon ordering convention.** Phase 0 codes polygons CCW starting from the south-west-most vertex. If real data convention differs (Overture follows RFC 7946: exterior CCW, interior CW), the encoder normalises before tokenising.
- **Macro tokens / boundary contracts.** Stubbed in vocabulary, not exercised. Designed in Phase 1.
- **Reprojection.** Lat/lon → local metres lives in `cfm.data`, not `cfm.tokenizer`. Phase 1.

## 11. Risks specific to this phase

- **Vocabulary IDs drift between commits.** Mitigation: canonical YAML in `configs/tokenizer/vocab_phase0.yaml` is the source of truth, and a test asserts `Vocabulary.load(...)` returns the exact same IDs as the previous commit's fixture-derived token tuple.
- **`geometric_equal` too forgiving.** Mitigation: explicit unit tests that perturb geometry by just above and just below the tolerance and assert the expected boolean.
- **Polygon closure semantics.** Decoders that auto-close polygons can mask encoder bugs. Mitigation: decoder validates closure explicitly and raises `UnsupportedGeometry` if a polygon's move sequence does not return to the anchor.
