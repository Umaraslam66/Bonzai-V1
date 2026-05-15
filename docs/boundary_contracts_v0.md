# Boundary Contracts — Hypothesis (v0)

> **This is a hypothesis, not a spec.** Captured 2026-05-15, while Phase 1 sub-project A (data infrastructure) is in flight. To be re-evaluated against real Overture geometry when Phase 1 sub-project E (boundary contracts) starts. Anything in this document may change. Anything that holds up against real data graduates to the spec.

## 1. What is a boundary contract for

PRD §4 calls boundary contracts the *key innovation* that enables parallel cell generation without losing coherence: every shared cell-edge gets a specification of exactly what crosses it, and both neighbouring cells receive that specification as conditioning. The macro planner emits contracts; the micro generator honours them; a deterministic stitcher then merges cells.

In Phase 0 we deferred this entirely. A single cell with an `<EXIT>` token is the placeholder. Phase 1 replaces `<EXIT>` with structured contracts.

## 2. Hypothesised shape of a contract

For a single cell-edge (one of the four edges of a 250 m × 250 m cell), a contract is **an ordered list of crossings**. Each crossing has:

| Field | Type | Notes |
|---|---|---|
| `feature_class` | vocab token | e.g. `R_residential`, `R_motorway`, `L_park`. Same vocabulary used by feature tokens. |
| `crossing_position` | int 0–249 | quantised offset along the edge, in metres from the SW corner of the edge. |
| `extent_class` | small enum | discrete bucket capturing width (roads) or extent (land use boundaries). Values TBD by frequency analysis. |
| `feature_type` | shape token | `LINE` or `POLYGON`. (`POINT` features don't cross edges.) |

Open: do we need a "direction" or "side" indicator (does the road continue on the *other* side, or terminate at the edge)? Probably implicit from the symmetry of contracts on both adjacent cells — but worth verifying with real cells where a road ends in a cul-de-sac right at a tile boundary.

## 3. Edge identity and orientation

Each cell has four edges. Naming convention proposal:

```
N (north): y = cell_size_m, x in [0, cell_size_m]
E (east):  x = cell_size_m, y in [0, cell_size_m]
S (south): y = 0,           x in [0, cell_size_m]
W (west):  x = 0,           y in [0, cell_size_m]
```

For a shared edge between cells `A` and `B`:
- The edge is `A.E` and `B.W` (same line in tile-local coordinates).
- The 1D parameterisation must agree on both sides: we hypothesise *both cells use the same 1D coordinate origin* (e.g., the SW corner of the shared edge in tile-local frame). This avoids a "mirror flip" bug where cell A thinks position 50 is north of the centre and cell B thinks position 50 is south.

## 4. Tile-edge cells

Cells at the outside of the 8×8 grid don't have neighbours on one or two sides. Open question: do they get a contract on those tile-edge sides? Three options:

- **(i)** No contract — the macro planner emits `<TILE_EDGE>` on those sides, and the micro generator is free.
- **(ii)** Identical contract format, populated by the macro planner using its global view of the tile (so cross-tile generation later can pick up the same contracts).
- **(iii)** Contract derived later when this tile is composed with neighbours.

Our hypothesis: **(ii)**. It keeps the cell-generation protocol uniform (every cell sees four contracts, no special cases) and pre-prepares for inter-tile stitching in a later phase.

## 5. Serialisation: tokens vs. JSON

Two options for the in-pipeline representation:

- **(a) Token sequence** — contracts are part of the model's input vocabulary. New control tokens (`<CONTRACT_N>`, `<CONTRACT_E>`, `<CONTRACT_S>`, `<CONTRACT_W>`, `<CONTRACT_END>`, `<CROSSING>`) plus existing class/anchor/extent tokens. The model conditions on contracts the same way it conditions on macro plans.
- **(b) Structured side-channel** — contracts live as JSON on disk and as cross-attention conditioning at training time, not in the autoregressive token stream.

Our hypothesis: **(a)**. The PRD's "macro plan as token sequence" framing implies contracts ride alongside macro tokens. Token-stream uniformity also simplifies inference. We'd add ~8 control tokens + the existing extent vocabulary; total impact on vocab size is small.

## 6. Generation order

We hypothesise per-cell contract sections appear in a fixed order: **N, E, S, W**. Within each edge, crossings appear in **ascending `crossing_position` order**. This gives a deterministic canonical form, useful for both training data and round-trip tests.

## 7. Mutual-consistency check

The "single source of truth" for contracts is the macro-plan derivation step that runs over the full tile. From there, contracts are *copied* to both adjacent cells. A validation pass asserts: for every internal cell-edge in the tile, the contract held by cell A on edge `E` is identical to the contract held by cell B on edge `W`. Any mismatch is a data-pipeline bug.

## 8. Stitching algorithm sketch

When 64 generated cells are reassembled into a tile:

1. For each internal cell-edge, fetch the contract.
2. For each crossing in the contract: connect the feature on cell A's side to the feature on cell B's side at exactly the recorded position.
3. Connect = update the feature's GeoJSON geometry so the LineString or Polygon ring crosses the edge at the contract position. Both cells' tokens were generated to honour this; the stitcher mostly just verifies and merges.

The stitcher is a **deterministic algorithm**, not a model. If it can't connect cleanly (e.g., generated geometry doesn't reach the contract position), the cell is regenerated or the failure is logged.

## 9. Open questions for sub-project E

These are the questions we'll answer with real data:

1. How many crossings does a typical cell-edge have? (1–4 is the rough guess; verify with Singapore data.)
2. Is `extent_class` discrete enough that a small bucket vocabulary (5–10 values) covers >95 % of real cases?
3. Does ordering crossings by position introduce ambiguity (e.g., two crossings at the same exact metre)?
4. Are there feature types we haven't anticipated (e.g., rivers, railway tracks) that need their own crossing semantics?
5. Tile-edge contracts: option (ii) works, but is it expensive to derive globally?
6. Token-stream representation (option (a)): does it inflate sequence lengths uncomfortably, or is it fine?

## 10. What would invalidate this hypothesis

- Real Overture data shows >10 crossings per cell-edge on dense urban edges (would force a different representation, maybe a per-edge feature graph).
- `extent_class` has a long tail (would force continuous width tokens, like the dyadic anchor scheme).
- Adjacent cells naturally disagree about an edge's contract (would force a different "single source of truth" — maybe contracts derived at stitch time, not macro-planning time).

If any of these fire when sub-project E starts, this document gets rewritten and a real spec replaces it.
