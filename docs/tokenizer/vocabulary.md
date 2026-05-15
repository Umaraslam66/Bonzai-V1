# Phase 0 token vocabulary

Reference companion to `configs/tokenizer/vocab_phase0.yaml`. The YAML is the source of truth; this doc explains what the tokens *mean*.

## Why the vocabulary is the way it is

The Phase-0 vocabulary is a deliberately small starter set. It exists to lock the encode → decode round-trip contract before Phase 1's full Overture-driven frequency analysis. New tokens are append-only across phases so checkpoints stay readable. See spec `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md`.

Counts: 8 control + 4 hierarchy + 19 feature class + 500 anchor + 48 move = **579 tokens**.

## Control (8)

| Token | Role |
|---|---|
| `PAD` | sequence padding |
| `BOS` | beginning of sequence |
| `EOS` | end of sequence |
| `CELL` | start of a cell's contents |
| `END_CELL` | end of a cell's contents |
| `FEATURE_START` | start of a feature |
| `FEATURE_END` | end of a feature |
| `EXIT` | preceding line ends on a cell boundary |

## Hierarchy (4, reserved)

`MACRO`, `END_MACRO`, `MICRO`, `END_MICRO`. Reserved for Phase 1 macro/micro split. IDs frozen now so Phase-1 additions don't renumber.

## Feature classes (19)

- Roads (5): `R_motorway`, `R_primary`, `R_secondary`, `R_residential`, `R_service`
- Buildings (3): `B_residential`, `B_commercial`, `B_industrial`
- POIs (5): `POI_restaurant`, `POI_school`, `POI_retail`, `POI_park_amenity`, `POI_transit_stop`
- Land use (6): `L_residential`, `L_commercial`, `L_industrial`, `L_park`, `L_water`, `L_agricultural`

A feature's `properties.class` value must match exactly one of these.

## Anchor (500)

Absolute positions in cell-local metres, split as `ANCHOR_X_n` + `ANCHOR_Y_n` for `n` in `0..249`. Always emitted as a pair (X then Y). One pair marks the starting vertex of every feature.

## Move (48)

8 cardinal/diagonal directions × 6 dyadic step sizes {1, 2, 4, 8, 16, 32} m.

- Directions: `N, NE, E, SE, S, SW, W, NW`
- Sizes (m): `1, 2, 4, 8, 16, 32`
- Names: `MOVE_<dir>_<step>`, e.g. `MOVE_E_8`, `MOVE_NW_32`.

**Phase 0 emits cardinal moves only** (`N`, `E`, `S`, `W`). The 32 diagonal tokens are reserved; the encoder raises `UnsupportedGeometry` on non-axis-aligned segments. Phase 1 enables diagonals.

## Sequence shape (Phase 0)

```
<BOS> <CELL>
  <FEATURE_START> <class> <anchor_pair> [<move> ...] [<EXIT>] <FEATURE_END>
  ...
<END_CELL> <EOS>
```

For polygons, the move sequence closes (cumulative delta returns to the anchor). For lines that terminate on a cell edge, the move sequence is followed by `<EXIT>` before `<FEATURE_END>`. Vertex boundaries within a polygon or line are inferred by direction change (consecutive moves in the same cardinal direction collapse into one segment).
