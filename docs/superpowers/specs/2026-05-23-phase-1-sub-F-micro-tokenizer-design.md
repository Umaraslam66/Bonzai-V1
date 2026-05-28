# Phase 1 sub-F micro-tokenizer design

Date: 2026-05-23

Status: design spec for review

> **This spec is hand-maintained.** The §2 paired-check inventory, §3 citation
> index, §10 governance + halt-points table, §11 task DAG, §12 deferral ledger,
> and §13 revision ledger are not regenerated from any source file. Edits land
> directly in this document and the spec self-review checklist runs as a
> manual pass before user review.

> **Operating discipline.** Sub-F brainstorm, plan-write, dispatch, and close
> all run against the sub-project planning protocol at
> `docs/protocols/sub-project-planning-protocol-v1.md`. The six gates +
> lint-cosmetic exemption + audit-after-fixup pattern + four supporting
> principles + diagnostic heuristic all apply. Sub-F is the first sub-project
> to operate under the protocol as active discipline (sub-E close was the
> protocol's derivation source).

---

## 1. Charter + scope locks

### 1.1 Charter

Sub-F is the per-cell micro-tokenizer (PRD stage four). For each 250m × 250m
cell, sub-F emits a token sequence describing the geometry inside the cell
(roads, buildings, POIs) as moves from anchor points. Ships a v1 micro-vocab,
encoder/decoder pair, and round-trip correctness gate.

Cells generate independently — PRD page 49 anchor: "cells generate in parallel;
each cell is independently generatable; the boundary contracts ensure they fit
together." Cross-cell coherence is via boundary-reference tokens to sub-E's
pre-derived contracts (§3.7, BP7), NOT via sequence concatenation.

### 1.2 Inputs (locked upstream; sub-F does not modify)

- Sub-C tile-extracted features at `data/processed/sub_c/<release>/<region>/tile=*/features.parquet`.
- Sub-D macro plan + lattice; canonical axis convention at `src/cfm/data/sub_d/lattice.py:5-17`.
- Sub-E boundary contracts at `data/processed/sub_e/<release>/<region>/tile=*/boundary_contract.parquet`; `BoundaryClass` enum at `src/cfm/data/sub_e/derivation.py:19-23`.
- Sub-A Overture pinning — single source of truth at `configs/data/overture_release.yaml` (build-time read for `SUB_F_SOURCE_VERSION`); policy reference at `docs/data/overture_pinning_policy.md`.

### 1.3 Outputs (new with sub-F-v1)

- Per-tile parquet, one row per cell: `(cell_id, token_sequence: list<int16>, feature_count: int16, provenance columns)` at `data/processed/sub_f/<release>/<region>/tile=*/cells.parquet`.
- Vocab YAML union: BP1 semantic + BP2 encoding primitives + BP4 `<unknown_*>` family + on-disk sentinel inventory.
- Encoder + decoder Python API with same-process AND fresh-process byte-identity contract.
- Round-trip correctness gate covering four geometry classes (§3.8).
- Per-tile `provenance.yaml` with four-axis version manifest (§6); region-level `<region>/manifest.yaml` with `vocab_sources` block (taginfo CSV + wiki revision IDs).
- **Determinism test suite** as durable lock artifact (§5.5): per-axis unit tests + same-process + fresh-process integration tests against real cached Singapore.

### 1.4 Scope locks (explicit OUT)

- **Tokenizer-fix** (Phase 0 round-trip surface; only currently-xfailed test in the suite). Separate sub-project queued after sub-F.
- **Training-scaffold.** Consumes sub-F output; separate sub-project.
- **Position-aware boundary contracts.** Sub-E v1 contract is class-only; exact crossing positions deferred to sub-E-v2 + sub-F-v2 joint sub-project (§12 #1). Sub-F-v1 ships class-only `<bref_dir_class>` tokens.
- **Runtime stitching gate** (sub-E §15 #10) — deferred. Sub-F-v1 consumes sub-E's `boundary_contract.parquet` as authoritative; no cross-cell consistency check at encode time. Sub-E's own §10.x invariants are the upstream guarantee. Runtime stitching gate that would verify sub-E's claims against sub-F's emissions is deferred to sub-E-v2/sub-F-v2 joint (§12 #2).
- **Non-road cross-cell features.** Buildings/POIs spanning two cells are clipped at geometry layer in v1. Token layer represents roads only for cross-cell references. Defers to sub-F-v2 + matching sub-E-v2 (§12 #3).
- **Cross-environment determinism.** Within-env contract locks here (§5); cross-env (darwin/aarch64 ↔ Leonardo linux/x86_64) verification deferred to end-of-Phase-1; first sub-F Leonardo run is the trigger (§12 #4). Same residual as sub-D §15 #7 and sub-E §15 #7.
- **Sweden + Sri Lanka region coverage.** V1 vocab + encoder calibrated against Singapore-only. F (frequency floor) per BP1 option (c) uses global taginfo with Singapore-prioritized must-appears. Sweden ingest TODO at `docs/known_issues.md:178-190` remains deferred. Multi-region vocab revision when Sweden lands (§12 #5).

### 1.5 Branch + commit pattern

Sub-F develops on `phase-1-sub-F-micro-tokenizer` branch per `project_branch_pattern`. Commit task-by-task; merge to main at sub-project end via reviewer-initiated merge. No PR flow. Branch created at design-write time.

---

## 2. Vocab union

Sub-F's on-disk vocab is the union of four families: BP1 semantic + BP2 encoding primitives + BP4 `<unknown_*>` family + on-disk sentinel inventory. Total estimated count surfaces in §2.5. Token IDs occupy contiguous range `[0, N-1]`; post-N block `[N, ∞)` reserved by name in `sentinel_inventory.yaml` for training-scaffold sentinels.

### 2.1 BP1 semantic vocab

**Vocab pinning context.** F (frequency floor) computed against global `taginfo.openstreetmap.org` distribution (option c per BP1 fix 5), Singapore-prioritized must-appears layered on top. Threshold-pairing: empirical cut (taginfo) + structural enumerate-from check (OSM wiki `Map_features`). v1 calibrated against Singapore; Sweden / Sri Lanka coverage gaps are accepted-cost residuals (§12 #5).

**Encoding.** One OSM `(key=value)` pair = one vocab slot. Examples: `highway=primary`, `building=residential`, `amenity=restaurant`. Separate key/value tokens rejected (combinatorial blow-up + positional invariant the model has to learn from scratch).

**F denominator.** Fraction-of-feature-bearing-elements, normalized within OSM element type (way / node / relation independently). Way-tags and node-tags are different populations; cross-type normalization inflates rare tags.

**Granularity.** Locked at Task 1 reviewer-halt (Halt 1) via marginal-cost-of-cut curve over:
- L1: top-level keys (~15 must-appears — highway, building, amenity, landuse, natural, water, waterway, leisure, shop, place, boundary, route, public_transport, barrier, man_made).
- L2: (key, primary-value) pairs (~80).
- L3: all wiki-documented pairs (~500+).

For each level: compute `(F_min, vocab_size, must-appears_admitted)`. Plot `Δvocab_size / Δmust-appears` between adjacent levels. Pick elbow with documented exception list for sub-floor wiki must-appears chosen to drop. NOT "pick L3 by default" — that's the trap BP1 fix 3 calls out.

**Singapore-prioritized X-threshold** (BP1 fix C). Option (c)'s additional structural check requires: any tag with `Singapore-frequency ≥ X` must appear above F. Candidate framings at Task 1 halt:
- Singapore's own elbow-derived F-equivalent (run elbow analysis on Singapore alone for X, on global for F).
- Fixed multiplier above F_global (e.g. 10 × F_global) with documented justification.

X's value AND X's paired §2 structural check both surface at Halt 1.

**Snapshot artifacts** (BP1 fix B; per `feedback_provenance_scope_placement` at config scope):
- `configs/sub_f/taginfo/2026-04-15.0.csv` — vendored taginfo dump at lock time.
- `configs/sub_f/wiki_map_features/2026-04-15.0.revision_id` — MediaWiki revision ID.
- `configs/sub_f/wiki_map_features/2026-04-15.0.wikitext` — raw wikitext (deterministic source; HTML hashing is fragile due to embedded timestamps and view-state).
- `configs/sub_f/wiki_map_features/2026-04-15.0.sha256` — hash over wikitext.

**Estimated slot count.** Depends on Task 1 elbow. L1 ≈ 50–100, L2 ≈ 150–250, L3 ≈ 500+. Concrete vocab size locks post-Task 1.

### 2.2 BP2 encoding-primitive vocab

| Slot family | Count est. | Notes |
|---|---|---|
| `<direction_*>` | 16 default (8 or 24 alternatives at Halt 2) | 22.5° resolution at 16; admits 45° right angles + 30°/60° diagonals; default-toward-16 per BP2 fix 4 cheap-to-keep rule (dropping to 8 makes future fine-angle data unrecoverable). |
| `<magnitude_*>` | ~64 | 0.5m quantum default (revised at Halt 2 over 0.25m / 0.5m / 1.0m surface). |
| Anchor coords | ~1000 (flat) or ~96 (hierarchical) | Halt 2 produces BOTH vocab-size AND mean-sequence-length-per-cell for both options. |
| Structural sentinels (on-disk; 6 named) | 6 | `<feature>`, `<feature_end>`, `<anchor_x>`, `<anchor_y>`, `<direction>` marker, `<magnitude>` marker. `<cell_start>` / `<cell_end>` dropped per BP2 → BP4 revision (§13.1); cell boundaries via parquet row structure. |

**Total encoding-primitive vocab: ~700–1100** depending on anchor scheme.

**§3 citation for axis convention** (per BP2 fix 3): `src/cfm/data/sub_d/lattice.py:5-17` (canonical) + `src/cfm/data/sub_c/enums.py:23 AXIS` (upstream-most source). `src/cfm/data/sub_e/rotation.py:50-62` is the per-cell transform, NOT the convention source.

### 2.3 BP4 `<unknown_*>` family

**Count.** ~15 per-key slots (one per BP1 must-appear key). Invariant under BP1 granularity level — per-key is the natural floor regardless of L1/L2/L3 lock per §2.3 cross-decision pre-commit.

**Enumeration anchor.** Derived from BP1's locked Gate 6 wiki Map_features must-appears. NOT derived from sub-F's empirical Singapore frequency — same anti-trap as BP1's structural check.

**Cost-asymmetry rationale.** Both lock options (per-type vs global+typed-escape) have symmetric data-re-tokenization costs for axis migration. The differentiator is runtime cost: per-type is 1 token per unknown occurrence vs global+escape's 2 tokens; per-type has direct training-time observability; decoder simplicity favors per-type. Per-type wins (option i per Fix 2).

**Snapshot artifact.** `configs/sub_f/unknown_family.yaml`, frozen at sub-F-close. References BP1's wiki snapshot revision ID in its derivation comment.

### 2.4 On-disk sentinel inventory + ID-space layout

**On-disk (in sub-F vocab):**
- BP2 structural sentinels (6 named).
- BP1 semantic tags.
- BP2 direction + magnitude + anchor coordinate tokens.
- BP4 `<unknown_*>` family (~15).
- BP7 `<bref_*_*>` road-crossing tokens (8: 4 directions × 2 active classes — §3.7).

**Dataloader-side only (NEVER in `token_sequence`):**
- `<pad>`, `<eos>`, `<bos>` — training-frame sentinels.
- `<cell_start>`, `<cell_end>` — training-scaffold inserts at training time if batch-packing concatenates cells.

**ID-space layout** (per §2 fix 1, option a). Per-family reserved blocks within `[0, N-1]` with gaps for append-only growth. Reserve sizes locked at Task 1 + Task 2 reviewer halts alongside curve elbow + encoding-primitive decisions. Reserved blocks (sizes TBD at halts):

- BP1 semantic: `[0, K1)`
- BP2 encoding primitives: `[K1, K2)`
- BP4 `<unknown_*>`: `[K2, K3)`
- BP7 boundary-refs: `[K3, K4)`
- BP2 structural sentinels: `[K4, N)`

**Post-N reserved block** `[N, ∞)` reserved by name in `sentinel_inventory.yaml` for training-scaffold sentinels (BP4 fix 1) without locking specific IDs. Training-scaffold picks specific IDs when it lands; sub-F append-only vocab cannot collide.

### 2.5 Total vocab size estimate

| Family | Count est. |
|---|---|
| BP1 semantic | Task 1 output (L1 ≈ 50–100, L2 ≈ 150–250, L3 ≈ 500+) |
| BP2 encoding primitives | ~700–1100 |
| BP4 unknown family | ~15 |
| BP7 boundary-ref | 8 |
| **Total** | **~775–1400** (lower bound at L1, upper at L3 anchor-flat) |

Total vocab size is a Task 1 + Task 2 joint output; precise value at sub-F-close.

### 2.6 Vocab artifacts

| Artifact | Purpose |
|---|---|
| `configs/sub_f/semantic_vocab.yaml` | BP1 `(key=value)` slots + frozen elbow + exception list |
| `configs/sub_f/encoding_primitives.yaml` | BP2 direction count, magnitude quantum, anchor scheme |
| `configs/sub_f/unknown_family.yaml` | BP4 per-key `<unknown_*>` slots |
| `configs/sub_f/boundary_reference_vocab.yaml` | BP7 8-token road-crossing vocab |
| `configs/sub_f/sentinel_inventory.yaml` | full vocab manifest + on-disk vs dataloader split + reserved post-N block |
| `configs/sub_f/taginfo/2026-04-15.0.csv` | snapshot artifact, BP1 frequency input |
| `configs/sub_f/wiki_map_features/2026-04-15.0.{wikitext, sha256, revision_id}` | snapshot artifact, BP1 Gate 6 source |
| `configs/sub_f/vocab_floor_analysis.yaml` | Task 1 curve + chosen elbow + exception list |

`provenance.yaml` (per-tile) + region manifest (with `vocab_sources` block) covered in §6 (version manifest semantics).

All YAMLs versioned with sub-F's `<release>` per the same release pattern as sub-D / sub-E.

---

## 3. Encoder grammar

Per-feature token sequence shape, per-cell layout, coordinate frame, direction / magnitude conventions, boundary-ref placement. Composes BP2 encoding primitives + BP7 boundary-ref shape; consumes BP1 semantic vocab (the `<semantic_tag>` slot below) without re-deriving its construction.

### 3.1 Coordinate frame

**Per-cell SW corner = (0, 0).** Aligns with sub-D lattice cell-local frame (`src/cfm/data/sub_d/lattice.py:5-17`).

**Units: projected meters** in sub-C's pinned projection per region. Singapore = EPSG:3414 SVY21 (verified from sub-E handoff line 80 tile path scheme `tile=EPSG3414_i{ti}_j{tj}/`). Sweden / Sri Lanka projection TBD per region's sub-C pinning when those regions land — accepted-cost residual.

**Anchor coords are absolute within-cell.** Range [0, 250] m. Quantized to 0.5m default (Halt 2 may revise to 0.25m or 1.0m over the joint surface).

**Direction + magnitude are relative per-vertex** from the previous vertex (or from the anchor for the first vertex per the convention in §3.2).

### 3.2 Per-feature sequence shape (four cases)

**Anchor convention (locked uniform per §7 fix 1).** Anchor IS vertex 1 in ALL cases (A, B, C, D). `V` = geometry vertex count. For Cases A, B (no inbound `<bref>`): anchor's coordinates equal vertex 1's coordinates by encoder convention; `(V−1)` dir+mag pairs reach vertices 2..V. For Cases C, D (inbound `<bref>` prepended): anchor IS vertex 1 by the inbound boundary constraint (vertex 1 on entry edge); `(V−1)` or `(V−2)` dir+mag pairs reach the remaining vertices.

Trigger between conventions is presence/absence of prepended `<bref>` at sequence start — decoder determines convention from token stream alone.

**Case A — Uncrossed feature (road, building, or POI fully within cell):**

```
<feature> <semantic_tag>
  <anchor_x_q> <anchor_y_q>
  <direction_d> <magnitude_q>           ×(V−1) pairs
<feature_end>
```

Token count: `1 + 1 + N_anchor + 2(V−1) + 1 = 3 + N_anchor + 2(V−1)`.

**Case B — Road outbound (terminates by exiting one cell edge):**

```
<feature> <semantic_tag>
  <anchor_x_q> <anchor_y_q>
  <direction_d> <magnitude_q>           ×(V−2) inner pairs
  <bref_dir_class>                       replaces final direction+magnitude; implicit position on edge
<feature_end>
```

Token count: `1 + 1 + N_anchor + 2(V−2) + 1 + 1 = 4 + N_anchor + 2(V−2)`.

**Case C — Road inbound (enters from one cell edge):**

```
<feature> <semantic_tag>
  <bref_dir_class>                       prepended before anchor
  <anchor_x_q> <anchor_y_q>              entry vertex coordinates (on edge)
  <direction_d> <magnitude_q>           ×(V−1) pairs
<feature_end>
```

Token count: `1 + 1 + 1 + N_anchor + 2(V−1) + 1 = 4 + N_anchor + 2(V−1)`.

**Case D — Road through-cell (both inbound and outbound):**

```
<feature> <semantic_tag>
  <bref_dir_class>                       inbound
  <anchor_x_q> <anchor_y_q>              entry vertex (on inbound edge)
  <direction_d> <magnitude_q>           ×(V−2) inner pairs
  <bref_dir_class>                       outbound (replaces final direction+magnitude)
<feature_end>
```

Token count: `1 + 1 + 1 + N_anchor + 2(V−2) + 1 + 1 = 5 + N_anchor + 2(V−2)`.

**Pattern:** pairs count = `(V−1) − outbound_bref_count`; inbound bref is a marker (no vertex emission); outbound bref replaces the last dir+mag pair.

**Non-road cross-cell features** (buildings, POIs spanning two cells): clipped at geometry layer per §1.4 scope lock. Last vertex lands exactly on edge with normal `<direction> <magnitude>` tokens; no `<bref>` token emitted. Cross-cell building / POI handling deferred to sub-F-v2 + sub-E-v2 (§12 #3).

### 3.3 Per-cell layout

Cell-level sequence is a flat concatenation of per-feature sequences. No `<cell_start>` / `<cell_end>` on-disk (per BP2 → BP4 revision §13.1); cell boundaries via parquet row structure (one row = one cell, with `token_sequence: list<int16>` column).

**Feature ordering within cell:** sub-C row order from `features.parquet`. Sub-F does NOT re-sort. Inherits sub-C's canonical row determinism per BP5 per-axis discipline (§5.2).

**Sub-C unknown semantic mapping.** Sub-C-normalized `class_raw` sentinels such as `unknown`, `__UNK__` values, and `B_*` building sentinels are NOT BP1 semantic vocab slots. When encountered by the encoder, they map to BP4's `<unknown_*>` family for the relevant parent key. Singapore X-threshold derivation filters these sentinels before pass-list computation (cascade #7) so sub-F does not encode sub-C normalization artifacts as first-class OSM semantics.

**Vertex iteration order within feature:** deferred to §5 (BP5 per-axis discipline + Task 5a verification commitment for Overture → sub-A → sub-C vertex-order chain stability).

### 3.4 Direction encoding

**16 directions at 22.5° resolution.** ID ordering follows mathematical convention (counterclockwise from +x axis) to match sub-D lattice axes (axis=0 = x = E-W; axis=1 = y = N-S). So `<direction_0> = E (0°)`, `<direction_4> = N (90°)`, `<direction_8> = W (180°)`, `<direction_12> = S (270°)`, with 22.5° steps between.

**Tie-breaking at exact bin boundary: round to LOWER bin index** (BP5 per-axis lock). Documented and tested with explicit boundary inputs (per BP5 plan-write refinement: boundary inputs constructed, not real-data-derived).

**Default 16 may revise to 8 or 24 at Halt 2** via the joint surface over `(direction_count × magnitude_quantum)`; default-toward-16 per BP2 fix 4 cheap-to-keep rule.

### 3.5 Magnitude encoding

**~64 magnitude tokens at 0.5m quantum, range 0.5m to 32m per step.** Magnitudes beyond 32m broken into multiple direction+magnitude pairs at the same direction (e.g., a 50m straight stretch becomes `<direction_0> <magnitude_64>` followed by `<direction_0> <magnitude_36>` — same-direction stretches concatenate at the encoder's discretion).

**Default 0.5m may revise to 0.25m or 1.0m at Halt 2** via the same joint surface as direction count.

### 3.6 Anchor scheme

**v1 default: flat dx + dy** (2 tokens per anchor, ~1000 anchor-coord slots). Hierarchical (4 tokens per anchor, ~96 anchor-coord slots) is the alternative revisited at Halt 2. Task 2 produces BOTH options' vocab-size AND mean-sequence-length-per-cell so the trade-off is decidable on numbers (per BP2 fix 4 minor).

### 3.7 Boundary-ref token semantics (BP7 lock summary)

**Direction labels (N/E/S/W) are cell-local view post sub-E rotation** (`src/cfm/data/sub_e/rotation.py:50-62`).

**Composite 8 tokens:**

```
<bref_N_MAJOR>  <bref_E_MAJOR>  <bref_S_MAJOR>  <bref_W_MAJOR>
<bref_N_MINOR>  <bref_E_MINOR>  <bref_S_MINOR>  <bref_W_MINOR>
```

**Class set: {MAJOR_ROAD, MINOR_ROAD}** only. NONE = non-emitting (no road on edge). BOUNDARY_NOT_APPLICABLE never on-disk per sub-E sentinel precedent (`src/cfm/data/sub_e/derivation.py:20`).

**Class assignment semantics (cascade #9, Halt 7 approved).** sub-F-v1 consumes sub-E's `boundary_contract.parquet` BoundaryClass as authoritative and tokenizes it verbatim; sub-F does not own a local `highway=*` → BoundaryClass override. Halt 7 surfaced and accepted a sub-E inherited limitation for v1: values omitted from sub-E's grouping default to `MINOR_ROAD` when present (`motorway` therefore emits but is under-tiered; non-vehicular or ambiguous values such as `path`, `pedestrian`, `track`, and `subway` also emit as MINOR). Correctness criterion for sub-F-v1 is faithful passthrough of sub-E's class per edge. Tiering refinement is a sub-E-v2 candidate, not a sub-F-v1 blocker.

**Multi-class collapse:** sub-E hierarchy `MAJOR > MINOR` (`src/cfm/data/sub_e/derivation.py:27-31` `_HIERARCHY`). Cell with two roads exiting same edge of different classes → both emit `<bref_dir_MAJOR>` per sub-E's contract derivation rule.

**Diagonal crossings** (one feature crosses two different edges, e.g., enters from N, exits at E): Case D covers (one inbound + one outbound at different edges). Multi-edge-exit features (Y-road originating in cell, exiting two edges) are represented per sub-C's feature-splitting convention as multiple separate features each with their own Case A/B/C/D — sub-C splitting convention is a §9.6 verify-before-lock pending reference (Task 7).

**Design-space audit rationale** (BP7 plan-write): composite (8 vocab, 1 token per ref) chosen over pair B (3 vocab, 3 tokens per ref) and intermediate (2 vocab `<bref_MAJOR>` / `<bref_MINOR>` + `<direction>`, 2 tokens per ref). Rationale rests on rotation-in-vocab structural value (cell-local view stays a first-class vocab object), not token-count math alone.

### 3.8 Right-angle inductive prior + round-trip structural check

Discrete-direction grammar encodes the right-angle preference into the vocab structure, not into learned weights. POC's 95% right-angle result is the empirical anchor (PRD page 67).

**Round-trip §2 paired check (four cases, ALL must pass independently per BP2 fix 1):**

1. **Closed polygon** (building footprint with arbitrary angles).
2. **Multi-vertex open polyline** (road centerline with curves).
3. **Right-angle building footprint** (the inductive-prior-load-bearing case).
4. **Road crossing cell boundary** (BP7 → BP2 expansion per §13.1; admits larger L_∞ tolerance for position quantization at the edge; class agreement remains strict).

**Both metrics required** (per BP2 fix 2): L_∞ vertex error AND 95th-percentile angle on input-right-angled corners. Tolerances locked at Halt 2.

**Round-trip equivalence definition** (per §3 fix 2): geometric — original vertices must be present in decoded output within L_∞ tolerance; additional collinear vertices on straight segments are admitted as a consequence of magnitude chunking (§3.5's >32m split) and do not constitute round-trip failure. Halt 2 produces both the per-vertex tolerance AND the collinearity admission threshold (max perpendicular deviation from the straight line through neighbors).

### 3.9 Cross-references

- Full vocab union: §2.
- Per-axis discipline locks for direction tie-breaking, anchor quantization, vocab iteration: §5.
- Per-tile `provenance.yaml` embedding the encoder version: §6.
- Boundary-ref four-test structural check (cross-reference + symmetry + non-road non-emission + coverage): §8 paired-check inventory, BP7 row.
- BP1 → sub-E class mapping standalone verification: §8 paired-check inventory, BP7 row, standalone test.
- Sub-C feature-splitting verification (informs §3.7 multi-outbound case necessity): §9.6 Task 7 verification.

---

## 4. Storage + sentinels

Per-tile parquet schema, region-level layout, sentinel split (storage perspective; vocab content in §2; version manifest embedding in §6).

### 4.1 Region-level path scheme

Mirrors sub-E (handoff line 80) with sub-F-specific naming:

```
data/processed/sub_f/<release>/<region>/
  manifest.yaml            region manifest (tile list + vocab_sources block + manifest_sha256)
  _SUCCESS                 empty marker; written ONLY after cross-tile validator passes
  tile=EPSG3414_i{ti}_j{tj}/
    cells.parquet          per-tile cell micro-tokens
    provenance.yaml        per-tile sha-chain anchor + four-axis version manifest (§6)
```

Projection embedded in tile directory name per sub-E precedent (Singapore = EPSG:3414). For Phase 1 de-risk run: `data/processed/sub_f/2026-04-15.0/singapore/`.

### 4.2 Per-tile parquet schema (`cells.parquet`)

Row-per-cell, 64 rows per tile (8×8 lattice from sub-D `lattice.py:5-17`). Pinned `pa.schema` with explicit types and nullable flags per sub-E precedent at `src/cfm/data/sub_e/writer.py:33` ("Pinned schema: explicit pa.schema with nullable flags").

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `cell_i` | `int8` | no | 0–7 |
| `cell_j` | `int8` | no | 0–7 |
| `cell_slot_index` | `int8` | no | `cell_i * 8 + cell_j` per sub-D `lattice.py:5`; int8 fits 0–63 (max 127) |
| `token_sequence` | `list<int16>` | no | variable-length; empty cells = `[]` not null; total vocab ~775–1400 fits int16 (max 32767) with substantial headroom |
| `feature_count` | `int16` | no | cardinality summary; BP3 stage-1 attribution input; if Task 3a surfaces a cell exceeding int16, bump and document |
| `provenance_sha256` | `string` (64-char hex) | no | pinned-length hash; sha chain anchor |

Inherits sub-D `_MACRO_CORE_SCHEMA` pattern (`src/cfm/data/sub_d/io.py:_MACRO_CORE_SCHEMA`): store all canonical identifiers (even derivable ones); correctness via pinned schema + inline-validator-as-derivation-check (§4.7), NOT via column omission.

`token_sequence` carries only on-disk vocab IDs. Dataloader-side sentinels (`<pad>`, `<eos>`, `<bos>`, `<cell_start>`, `<cell_end>`) are NEVER in `token_sequence`; added by training-scaffold dataloader at training time.

### 4.3 Cell row ordering

Cells sorted by `(cell_i, cell_j)` row-major. `cell_slot_index` matches sub-D's slot_index convention exactly (verified against `src/cfm/data/sub_d/lattice.py:5` — `slot_index = cell_i * 8 + cell_j`). No re-derivation; sub-F consumes sub-D's lattice convention as-is.

### 4.4 Empty cells

Cells with zero geometric features (over water, parks, agriculture) emit `token_sequence = []` (empty list, not null). The cell still occupies a row. Empty cells contribute the floor of BP3's per-cell distribution and the floor of `feature_count` (= 0). The dataloader is responsible for handling empty-cell entropy at training time — sub-F's contract is "every cell has a row, even if empty."

### 4.5 On-disk vs dataloader sentinel split (storage perspective)

**On-disk in `token_sequence`** (token IDs from sub-F vocab):
- BP2 structural sentinels (6 named per §2.2).
- BP1 semantic tags.
- BP4 `<unknown_*>` family.
- BP7 boundary-ref tokens.
- BP2 direction + magnitude + anchor coordinate tokens.

**NEVER in `token_sequence`** (dataloader inserts at training time):
- `<pad>`, `<eos>`, `<bos>` — batch-frame sentinels.
- `<cell_start>`, `<cell_end>` — if training-scaffold concatenates cells into longer sequences.

Sentinel ID space layout per §2.4: per-family reserved blocks within `[0, N-1]`. Post-N block `[N, ∞)` reserved by name in `sentinel_inventory.yaml` for training-scaffold without locking specific IDs.

### 4.6 Atomic write + `_SUCCESS` discipline (sub-E precedent)

Mirrors sub-E (handoff line 198, fixup `fd53fdd` — validate-then-touch ordering):

1. Per tile: write `cells.parquet` → run inline validator on parquet → write `provenance.yaml` (atomic via `path.write_text(canonicalize_yaml(data), encoding="utf-8")`, sub-E convention at handoff line 188 + fixup `4ffa101`).
2. Region-level: run cross-tile validator across all tiles in `<region>/`; write region `manifest.yaml` with `vocab_sources` block.
3. `_SUCCESS` touched ONLY after cross-tile validator passes. No try / except / unlink dance.
4. **Halt-on-validator-fail:** sub-E precedent (handoff line 199); no partial sub-F output produced if any validator fails.

### 4.7 Inline + cross-tile validators (sub-F adaptation)

**Inline validator** (per-cell, runs at write time):

- Schema conformance: every row has all declared columns at declared types.
- Empty-cell representation: empty `token_sequence` is `[]` not null.
- `feature_count == 0 ⟺ token_sequence == []`.
- **`cell_slot_index == cell_i * 8 + cell_j`** (load-bearing derivation check per §4.2; this is the protection the §4 fix 1 reviewer asked about for the three-column store-all-three pattern).
- Token IDs in `token_sequence` are all in `[0, N-1]` (sub-F vocab range).
- `provenance_sha256` is 64-char lowercase hex.

**Cross-tile validator** (per-region, runs at region close before `_SUCCESS`):

- All 64 cells present per tile (no missing `(cell_i, cell_j)` combinations).
- All tile manifests' `provenance_sha256` are unique (no accidental tile duplication).
- Version manifest consistency across tiles (BP6 paired field-correctness check; structure in §6).
- BP7 four-test structural check (cross-reference + symmetry + non-road non-emission + coverage) — §8 BP7 row, applied here since cross-cell symmetry requires multiple tiles.

### 4.8 §3 citations

- Sub-E pinned schema pattern: `src/cfm/data/sub_e/writer.py:33`.
- Sub-E per-tile output path scheme: handoff line 80.
- Sub-D lattice cell ordering: `src/cfm/data/sub_d/lattice.py:5` (`slot_index = cell_i * 8 + cell_j`).
- Sub-D store-all-three-identifiers precedent: `src/cfm/data/sub_d/io.py:_MACRO_CORE_SCHEMA`.
- Sub-E `_SUCCESS` validate-then-touch: handoff line 198 + fixup `fd53fdd`.
- Sub-E halt-on-validator-fail discipline: handoff line 199.
- Sub-E manifest pattern: handoff line 77.
- Sub-E `canonicalize_yaml` idiom: handoff line 188 (fixup `4ffa101`).

### 4.9 Residuals

- **`pa.field` per column** verified at Task 8 (writer task) by reading sub-E's actual pinned-schema declaration; pattern inherited verbatim if API matches.
- **Pyarrow Hive partition inference on `tile=...` dirs** — sub-C / sub-E precedent (memory `feedback_pyarrow_hive_partition_inference`): use `pq.ParquetFile(path).read()` for per-tile reads, never bare `pq.read_table()` on the parent directory (injects spurious `tile` column). Sub-F readers follow.
- **`provenance_sha256` excluded-from-sha mechanism** inherited from sub-E (`SUB_E_EXCLUDED_FROM_SHA` pattern, handoff line 188); sub-F emits `SUB_F_EXCLUDED_FROM_SHA` covering `extracted_utc` and any equivalent live-clock timestamp fields so live-clock reruns don't break the digest chain.

---

## 5. Determinism contract

Per-axis discipline + lock artifact + Task 5a verification commitment. BP5 scope: within-env only; cross-env deferred to end-of-Phase-1 per §1.4 scope locks.

### 5.1 Top-level contract (three commitments)

1. **Same-process encode determinism.** `encode(features) == encode(features)` byte-for-byte across two invocations in one Python process.
2. **Fresh-process encode determinism.** `encode(features)` byte-identical across cold Python starts in the same environment (matches sub-E precedent at `tests/data/sub_e/test_singapore_integration.py::test_layer3_deterministic_rerun_same_process`).
3. **Round-trip determinism.** `decode(encode(features))` byte-identical across re-runs. NOT a round-trip correctness guarantee (that's §3.8); just repeatability of the round-trip.

Symmetric commitments for `decode()`.

### 5.2 Per-axis discipline table

| Axis | Lock | Source / cite |
|---|---|---|
| Coordinate quantization arithmetic | Integer-only after `int(round(coord_m / quantum))`; no float division in quantization step | bit-stable integer ops |
| Round tie-breaking | **Pending Task 5a verification of sub-D's actual rounding mechanism. Assumed default: Python `round()` round-half-to-even per PEP 3141. Final lock at Halt 5.** | Per `feedback_verify_before_lock_not_after`; sub-D wins; cascade per §9.6.1 if mismatch |
| Direction bin tie-breaking at boundary | Round to LOWER bin index | arbitrary but explicit; per-axis unit test with constructed boundary input |
| OSM feature iteration order | sub-C row order from `features.parquet`; verify at Task 5a against `src/cfm/data/sub_c/io.py` | Pending verification per `feedback_verify_before_lock_not_after` |
| Vertex iteration within feature | sub-A → sub-C → sub-F chain stability per Task 5a verification (three branches a/b/c per `feedback_ambiguous_third_branch_in_verification`); if ambiguous or unguaranteed, canonicalize via lex-min polygon-ring rotation | BP5 BLOCKING fix; sub-A pinning policy + sub-C reader |
| Vocab `(key=value) → id` lookup | Insertion-order dict populated from YAML (Python 3.7+ guarantee); YAML load via `yaml.safe_load` deterministic | `feedback_external_source_of_truth_gate` discipline |
| `<unknown_*>` fallback path | Same insertion-order discipline | symmetric to vocab lookup |
| Cross-cell coherence iteration (BP7) | Per-tile sub-E parquet sort order `(slot_kind, slot_index)` | sub-E handoff line 80 |

### 5.3 Decode byte-identity definition

Decode output is geometry, not bytes. **Lock: canonical serialization to GeoJSON with pinned formatter options** (`sort_keys=True`, `indent=None`, `ensure_ascii=True`). Byte-identity is then defined over the canonical-GeoJSON byte stream. Alternative (structural equality on geometry objects) rejected for simpler test infrastructure — byte comparison is trivial to assert, structural equality requires custom comparator.

### 5.4 Per-axis unit test discipline

Each per-axis unit test constructs inputs **at exact bin edges**, not real-data-derived. Examples:

- Vertex at exactly 11.25° (half a bin) for direction tie-breaking.
- Coord at exactly `quantum/2` for round tie-breaking.
- **Vocab dict iteration order matches YAML file order, verified under `PYTHONHASHSEED=random` across N cold pytest invocations** (per `feedback_pythonhashseed_dict_iteration_test`).

Real Singapore data won't reliably land on boundaries; synthetic edge-case inputs do.

### 5.5 Lock artifact

Sub-F's determinism passes iff the per-axis unit test suite + same-process integration test + fresh-process integration test all pass against real cached Singapore. **The test suite IS the durable contract artifact — not a YAML.** Adding `configs/sub_f/determinism.yaml` would be redundant with code-as-contract.

### 5.6 Task 5a vertex-order verification (BLOCKING)

Task 5a step 1: read current source of Overture pinning + sub-A reader + sub-C row-emission code; verify vertex-order stability across the sub-A → sub-C → sub-F chain. Three outcomes per `feedback_ambiguous_third_branch_in_verification`:

- **(a) Chain guarantees stable vertex order:** inheritance is valid; document with file:line cites in encoder comment; no canonicalization needed.
- **(b) Chain documents absence of guarantee:** sub-F canonicalizes via lex-min polygon-ring rotation (open polylines: lex-smaller of forward / reverse traversal; closed rings: rotate to start at lex-min vertex). Document the canonicalization choice; add per-axis test.
- **(c) Ambiguous (docs don't guarantee, empirical sample shows stability):** canonicalize anyway. Cheap insurance; eliminates the defect class. Empirical stability under sampling is NOT a guarantee — same logic as sub-E's `feature_class` defect.

Default for (c) is (b)'s treatment, NOT (a)'s. Decision lands at Halt 5 per `feedback_subagent_branch_pattern` (no silent inline fix).

### 5.7 Alternatives (rejected)

- Cross-environment byte-identity in v1: deferred per §1.4; matches sub-D / E.
- Threshold-based equivalence ("≥99% of tokens match"): weakens contract; same anti-pattern as sub-E's empirical-gate-without-structural-check.
- Same-process only (drop fresh-process): sub-E already cleared fresh-process; sub-F lowering the bar is unjustified.

### 5.8 §3 citations

- Sub-E same-process + fresh-process tests: `tests/data/sub_e/test_singapore_integration.py::test_layer3_deterministic_rerun_same_process`.
- Sub-D byte-identity precedent: `src/cfm/data/sub_d/io.py` write_parquet routing through `cfm.data.io.write_parquet` with pinned `PARQUET_WRITE_KWARGS`.
- Sub-C row-order convention: verify at Task 5a against `src/cfm/data/sub_c/io.py`.
- Overture pinning policy: `docs/data/overture_pinning_policy.md`.
- Sub-A release lock: `configs/data/overture_release.yaml`.

### 5.9 Residuals

- Cross-env deferred (end-of-Phase-1 trigger: first sub-F Leonardo run; §12 #4).
- `torch.compile` non-determinism out of scope (sub-F encoder / decoder are pre-training, no torch).
- Multi-threaded execution: sub-F single-threaded; if training-scaffold parallelizes encode across cells, the parallelism layer (dataloader workers) carries its own determinism contract — sub-F's contract is the per-cell unit.
- Pytest fixture seeding: deterministic seeds documented in test convention; no new precedent vs sub-E.

---

## 6. Version manifest semantics + on-disk embedding

Four-axis version manifest, bump rules per axis, `compare_version` extension at sub-D, on-disk embedding in `provenance.yaml` + parquet metadata + region manifest's `vocab_sources` block, migration semantics.

### 6.1 Six-axis manifest

> **Plan-write audit revision (2026-05-23 cascade):** original spec assumed sub-D had 3 namespaces (DATA_SHAPE / VOCAB / DERIVATION) and sub-F would extend to 4 by adding SOURCE. Pre-dispatch audit at plan-write time read `src/cfm/data/sub_d/versions.py` directly and surfaced that sub-D's `VersionNamespace` enum has 5 members (`ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR`). Per §9.6.1 cascade (sub-D wins; lock value updates; revision ledger entry mandatory): sub-F adopts all 5 sub-D namespaces + adds SOURCE = **6 axes total**. See §13.1 for ledger entry; §13.3 for plan-write spec-revision context.

| Axis | Constant in `src/cfm/data/sub_f/versions.py` | v1 value | Governs |
|---|---|---|---|
| ARTIFACT_FORMAT | `SUB_F_ARTIFACT_FORMAT_VERSION` | `"1.0"` | on-disk format identifier (parquet writer kwargs, list<int16> serialization, GeoJSON canonical formatter options); separate evolution from DATA_SHAPE so format-only changes (e.g., parquet compression algorithm) don't trigger schema bump |
| DATA_SHAPE | `SUB_F_SCHEMA_VERSION` | `"1.0"` | parquet schema (cell-row layout, column types) + YAML structure (provenance, manifest, vocab files) |
| VOCAB | `SUB_F_VOCAB_VERSION` | `"1.0"` | union of (BP1 semantic + BP2 encoding-primitive + BP4 `<unknown_*>` family + on-disk sentinels + BP7 boundary-refs) |
| DERIVATION | `SUB_F_DERIVATION_VERSION` | `"1.0"` | encoder logic — direction count, magnitude quantum, anchor scheme, vertex canonicalization (if BP5-added), round-trip behavior, multi-class collapse rule |
| VALIDATOR | `SUB_F_VALIDATOR_VERSION` | `"1.0"` | sub-F's inline + cross-tile validator code version; v1 ships single shared constant (option a at Task 6 halt). Split into VALIDATOR_INLINE + VALIDATOR_CROSS_TILE deferred to sub-F-v2 if validators evolve independently |
| SOURCE | `SUB_F_SOURCE_VERSION` | read at build time from `configs/data/overture_release.yaml` | upstream sub-A pinning (e.g., `"2026-04-15.0"`) |

**ARTIFACT_FORMAT vs DATA_SHAPE distinction.** DATA_SHAPE governs WHAT data is stored (column names, types, semantics). ARTIFACT_FORMAT governs HOW it's encoded on disk (parquet kwargs, byte-order, compression, YAML canonicalization options). Format-only changes (e.g., adding zstd compression to PARQUET_WRITE_KWARGS) bump ARTIFACT_FORMAT without DATA_SHAPE; schema changes (e.g., adding a column) bump DATA_SHAPE without ARTIFACT_FORMAT.

**VALIDATOR scope.** v1 ships single shared constant covering both validators (`validator_inline.py` + `validator_cross_tile.py`). At Task 6 halt, reviewer locks (a) shared or (b) split into VALIDATOR_INLINE + VALIDATOR_CROSS_TILE (7 axes total). Default recommendation: (a) shared — simpler; deferral to sub-F-v2 if empirical evidence shows validators evolve at different rates.

**SOURCE source-of-truth** (BP6 fix 2): read at build time from `configs/data/overture_release.yaml`. Sub-F does NOT duplicate the constant. Single source of truth; sub-A re-pinning cannot silently invalidate sub-F vocab.

### 6.2 SemVer canonical form

Lock: **`"X.Y"` two-component form** (matches sub-D / E precedent at handoff line 64–69). `compare_version` normalizes parsing. Three-component `"X.Y.Z"` rejected — sub-D / E have no patch-level concept; sub-F has no patch-level use case at v1.

### 6.3 Per-axis bump rules

| Axis | Major bump (`X.0 → Y.0`) | Minor bump (`1.0 → 1.1`) |
|---|---|---|
| ARTIFACT_FORMAT | parquet kwargs incompatible change (e.g., compression algorithm change); WKB byte-order change; canonical-GeoJSON formatter option change | compression-level tuning (backward-readable); pyarrow-version metadata update |
| DATA_SHAPE | column rename / remove; column type change | column add (nullable) |
| VOCAB | slot reorder; slot remove; BP1 granularity-level change (L1↔L2↔L3); `<unknown_*>` family axis change | slot append (per `feedback_append_only_vocab_safety`) |
| DERIVATION | direction count change; magnitude quantum change; anchor scheme change; vertex canonicalization change | new feature-type support (existing-type tokens unchanged) |
| VALIDATOR | invariant added or removed; structural check semantics change | refactor without invariant change (e.g., split helper function) |
| SOURCE | any sub-A re-pinning (Overture extract change) | (no minor; sub-A pinning is opaque to sub-F) |

### 6.4 `compare_version` extension

**Lock: extend sub-D's `compare_version` in-place to accept four axes, backward-compatible** (three-axis callers continue to work; no sub-D version bump). Pending Task 6 verification that sub-D's current implementation supports a clean backward-compatible extension (per `feedback_verify_before_lock_not_after`).

**Ordered fallback chain** (per BP6 fix 1 + §6 fix 3, in priority order):

1. **(a) In-place extension** (default). Add `source_version: str | None = None` kwarg; existing call sites unchanged.
2. **(c) Variable-axis-count** if (a) backward-compat blocker surfaces — accept variable axis-count; add SOURCE to call sites only.
3. **(b) `compare_version_v2`** if (c) also problematic — sub-F ships its own helper, keeps sub-D's `compare_version` unchanged.

§3 verification at Task 6: read `compare_version` source by file:line (cite at lock time); confirm extension shape (e.g., adding `source_version` kwarg) is backward-compatible against all current call sites in sub-D + sub-E.

### 6.5 Sub-A `release_id` semantic verification

§3 verification at Task 6: read `configs/data/overture_release.yaml` and sub-A code by file:line; determine which semantic `release_id` carries:

- **(a) Source-data-pinning identifier** (Overture extract hash + date) → SOURCE-bump-on-this-change is correct.
- **(b) Sub-A output hash** → also acceptable; SOURCE-bump-on-output-change.
- **(c) Sub-A code version** → PROBLEMATIC; code changes that don't affect output would force sub-F bumps. If verification surfaces (c), restructure: sub-F's SOURCE tracks a derived data-pinning ID, not sub-A's code version.

Three branches per `feedback_ambiguous_third_branch_in_verification`: if `release_id` semantic is ambiguous, default to deriving sub-F's own data-pinning ID from `(overture_release_date, overture_extract_sha256)` and citing both — defend rather than trust.

### 6.6 On-disk embedding

**Per-tile `provenance.yaml`** (per `feedback_provenance_scope_placement` — only tile-specific fields):

```yaml
sub_f_artifact_format_version: "1.0"
sub_f_schema_version: "1.0"
sub_f_vocab_version: "1.0"
sub_f_derivation_version: "1.0"
sub_f_validator_version: "1.0"
sub_f_source_version: "2026-04-15.0"

extracted_utc: "2026-05-23T14:30:00Z"        # excluded from provenance_sha256 per SUB_F_EXCLUDED_FROM_SHA
provenance_sha256: "<64-hex>"                 # over all fields except extracted_utc
```

**Region manifest** (`<region>/manifest.yaml`) carries the shared `vocab_sources` block per `feedback_provenance_scope_placement` (release-level shared metadata, not per-tile):

```yaml
sub_f_artifact_format_version: "1.0"
sub_f_schema_version: "1.0"
sub_f_vocab_version: "1.0"
sub_f_derivation_version: "1.0"
sub_f_validator_version: "1.0"
sub_f_source_version: "2026-04-15.0"

vocab_sources:
  taginfo_release: "2026-04-15.0"             # project-release-anchored (§6 fix 1)
  taginfo_csv_sha256: "<64-hex>"
  wiki_revision_id: <integer>                 # MediaWiki revision ID per BP1 fix B
  wiki_wikitext_sha256: "<64-hex>"

tiles:                                        # sub-E precedent
  - tile_dir: "tile=EPSG3414_i0_j0"
    provenance_sha256: "<64-hex>"
  # ... per tile

manifest_sha256: "<64-hex>"                   # self-integrity, sub-E precedent
```

`extracted_utc` excluded from `provenance_sha256` per `SUB_F_EXCLUDED_FROM_SHA` mechanism (sub-E precedent at handoff line 188, fixup `4ffa101`). Sub-F emits a `SUB_F_EXCLUDED_FROM_SHA` constant covering `extracted_utc` and any equivalent live-clock timestamp fields.

**Per-tile parquet metadata** carries the same four-axis manifest in the pyarrow.parquet metadata kvstore (key-value bytes; UTF-8 encoded YAML or JSON). Locks identical to `provenance.yaml` for cross-format consistency.

### 6.7 Migration semantics (v1 → v2 sub-project trigger)

Mirrors sub-E §15 #1 pattern:

- **All-axes minor bumps:** in-place edit within sub-F branch; vocab YAML appends; no new sub-project.
- **Any-axis major bump:** new sub-project (sub-F-v2). Old artifacts deprecate at old release path; new artifacts under new release path. Migration plan part of sub-F-v2's spec.
- **Asymmetric forward-compat:** v2 reader can consume v1 data ONLY when v2's bumps are all minor. Major bump in any axis breaks both directions.

### 6.8 §2 paired structural check

Four-part check per BP6 §2 decomposition:

1. **Field presence:** all SIX axes appear in every artifact (parquet metadata + provenance.yaml + region manifest). Plus `vocab_sources` block in region manifest.
2. **Field correctness:** test imports `SUB_F_*_VERSION` constants directly + asserts they appear verbatim in latest artifact. NO bump-and-revert dance (which would require source mutation in tests).
3. **Cross-axis coupling sanity:** SOURCE-only bump (sub-A re-pinning, sub-F code unchanged) produces correctly-updated SOURCE while DERIVATION stays at v1.0. DERIVATION-only bump (encoder logic change, sub-A unchanged) produces correctly-updated DERIVATION while SOURCE stays. Both tested. Test extends to ARTIFACT_FORMAT and VALIDATOR axes (each independently bumpable).
4. **Parse correctness:** all SIX axes parse correctly via `compare_version` (sub-D's existing helper, backward-compatible enum extension per Plan Revision 1) across the SemVer canonical form.

### 6.9 §3 citations

- Sub-D / sub-E three-axis precedent: `src/cfm/data/sub_d/versions.py`, `src/cfm/data/sub_e/versions.py`.
- `compare_version` helper: sub-D source, file:line at Task 6 (pending verify-before-lock).
- Sub-A pinning policy: `docs/data/overture_pinning_policy.md`.
- Sub-A release lock (build-time read): `configs/data/overture_release.yaml`.
- `feedback_append_only_vocab_safety` — minor-vs-major bump rationale for VOCAB.
- Sub-E provenance pattern + `EXCLUDED_FROM_SHA` mechanism: handoff line 188 + fixup `4ffa101`.
- Sub-E atomic-write: `path.write_text(canonicalize_yaml(data), encoding="utf-8")` idiom.

### 6.10 Residuals

- **Sub-D / sub-E manifests don't surface their own sub-A dependency** (§12 #8). Retrospective concern; protocol-bump candidate.
- **`compare_version` extension blocker.** If Task 6 surfaces backward-compatible extension is non-trivial, fallback chain advances per §6.4.
- **`release_id` semantic surfacing (c).** If sub-A's `release_id` is code-versioned, sub-F derives its own data-pinning ID per `feedback_ambiguous_third_branch_in_verification`'s defend-not-trust default.

---

## 7. Per-cell sequence-length budget

BP3 four-stage decomposition, Task 3a / 3b / 3c split with dependencies, joint 4D distribution requirement, marginal-cost surface for budget elbow, retention defaults, empty-cell-as-floor, architecture-coupling residual.

### 7.1 Assumption: one sequence per cell

PRD page 49 confirms: one token sequence PER CELL (cells generate in parallel, independently generatable). Sub-F does NOT stitch sequences across cells in the sequence layer — cross-cell coherence is via BP7 boundary-reference tokens (§3.7), NOT sequence concatenation. Closes any "stitching tokens" temptation upfront.

### 7.2 Four-stage decomposition

| Stage | What | Source |
|---|---|---|
| 1 | Per-cell feature counts by type (mean, variance, P95, P99, empty-cell fraction) | sub-C Singapore extracts at `data/processed/sub_c/2026-04-15.0/singapore/` |
| 2 | Per-feature vertex-count distribution by type | sub-A Overture Singapore → sub-C |
| 3 | Tokens per geometry element | derived from §3 encoder grammar (BP2 locked values) |
| 4 | Cross-cell coordination overhead per cell | sub-E boundary contracts at `data/processed/sub_e/2026-04-15.0/singapore/` + §3.7 boundary-ref token shape |

**Stage-3 formula** (parametrized by §3.4–§3.6 locks; anchor convention per §3.2):

```
Case A uncrossed:  3 + N_anchor + 2(V−1)
Case B outbound:   4 + N_anchor + 2(V−2)
Case C inbound:    4 + N_anchor + 2(V−1)
Case D through:    5 + N_anchor + 2(V−2)

  where N_anchor = 2 (flat) or 4 (hierarchical), 
        V = geometry vertex count,
        leading constants account for <feature>, <semantic_tag>, <bref>s, <feature_end>.

tokens_per_cell = Σ_feat tokens_per_feature + stage_4_overhead
                  (no <cell_start>/<cell_end>; row-per-cell parquet per §4.5)
```

**Stage-4 formula** (BP7 → BP3 correction per §13.1, asymmetric):

```
stage_4_overhead ≈ inbound_crossings_per_cell × 1 token   (inbound <bref> prepended, +1 each)
                 + outbound_crossings_per_cell × 0 tokens (outbound <bref> replaces tail direction+magnitude, net 0)
                 ≈ 0.7 × 1 ≈ 0.7 tokens/cell on Singapore rough estimate
```

Outbound is token-neutral (replaces tail); inbound is +1 per crossing.

### 7.3 Compound is joint 4D, not sum of marginals

Task 3a output is `P(feature_count, vertex_count | cell, feature_type)` joint per type — NOT separate `P(feature_count | cell)` × `P(vertex_count | feature)` marginals. Stages 1 and 2 are correlated (dense areas tend to have systematically simpler geometry per feature due to zoning regularity); treating as independent inflates tail prediction. Budget comes from the joint's chosen quantile, NOT from mean × mean × mean × mean.

### 7.4 Budget surface (NOT autonomous P100 default)

Task 3c outputs a marginal-cost-of-cut SURFACE over (quantile ∈ {P99, P99.5, P99.9, P99.99, P100}) with three axes:

| Axis | Meaning |
|---|---|
| `data_loss_per_feature_type` | retention rate per BP1 feature type at this quantile |
| `sequence_length` | budget value in tokens at this quantile |
| `padding_overhead_estimate` | expected padding cost per cell for transformer-arch packing |

Reviewer picks elbow on the surface at Halt 4, NOT on data-loss column alone. Transformer self-attention is O(n²); moving P99.9 → P100 can be 3–10× compute waste even when data-loss difference is <0.1%. Cost-asymmetry per `feedback_schema_vs_data_cost_asymmetry`: data loss expensive to recover, but compute cost is the ceiling.

### 7.5 Per-feature-type retention thresholds (defaults)

| Feature type | Default minimum retention rate |
|---|---|
| Roads (`highway=*`) | ≥99.9% (AV use case, downstream criticality) |
| Buildings (`building=*`) | ≥99.0% |
| POIs (`amenity=*`, `shop=*`, etc.) | ≥99.0% |
| Landuse-class (combining `landuse=*` and `natural=*` — both are area-feature types with comparable downstream criticality) | ≥99.0% |

Reviewer may override per-type at Halt 4 if Task 3c data shows a specific class needs different treatment. Defaults anchor; overrides documented per-type in the Halt 4 report. Landuse + natural split into separate rows at Halt 4 if data shows asymmetric retention behavior.

### 7.6 Retention definition

A feature counts as **retained iff (tokenized AND BP2-round-trip-passing)**. Truncated features count as retention loss. Non-round-tripping features (per §3.8's four-case structural check) ALSO count as retention loss — truncation and non-round-tripping have identical downstream impact.

### 7.7 Long-cell diagnostic

Diagnostic threshold = **sequence length corresponding to (chosen_quantile − 0.5pp) on the joint distribution.** E.g., if budget locks at the P99.5 quantile producing ~N tokens, diagnostic fires at the token length corresponding to P99.0 quantile, which is < N tokens. The subtraction is in percentile space; the threshold value lives in token space. Diagnostic fires BEFORE the budget cuts — surfaces cells approaching the truncation boundary so reviewer can attribute drift before retention degrades.

### 7.8 Empty-cell handling

Cells with zero geometric features emit `token_sequence = []` (per §4.4 storage spec). Stage-1 distribution enumerates empty-cell fraction as its own bucket; budget calculation treats empty as the floor of the per-cell distribution, NOT as outliers to truncate.

### 7.9 Truncation strategy (three options at Halt 4)

- **(α) Tail-cell rejection.** Cells exceeding the elbow quantile removed from training set; logged with per-stage attribution. Data loss per-cell, not per-feature-within-cell. Preferred if elbow tail is bounded.
- **(β) Feature-priority tail-drop within cell.** Cell stays, low-priority features truncated. Per `feedback_schema_vs_data_cost_asymmetry`: schema cheap to change, data expensive to recover — last-resort.
- **(γ) Return to BP2 / §3 for encoding compression.** Tail-pathological cells force encoder revision (smaller direction count or hierarchical anchor). Only if both (α) and (β) costs unacceptable.

Decision at Halt 4 with per-stage attribution + budget-surface elbow visible.

### 7.10 Task 3 split with dependencies

| Subtask | Inputs | Unblocked by | Output |
|---|---|---|---|
| **Task 3a** | sub-C Singapore extracts (cached) | (no upstream BP) | joint `P(feature_count, vertex_count \| cell, feature_type)` per type |
| **Task 3b** | Task 3a output + BP2 encoder lock | Halt 2 | stage-3 compound: per-cell length distribution sans cross-cell overhead |
| **Task 3c** | Task 3b output + sub-E boundary contracts (cached) + §7.2 stage-4 formula (BP7 design lock, brainstorm output — NOT Task 7 implementation) | Halt 2 + BP7 design lock | full 4D joint + budget surface + retention table + truncation proposal |

Task 3a runs as soon as sub-F branch opens — no upstream BP dependencies. Task 3c is the Halt 4 gate; YAML locks `configs/sub_f/sequence_length_analysis.yaml` only after reviewer approves elbow + truncation + per-type thresholds.

### 7.11 Architecture cost coupling (residual)

Sequence-length-squared cost in transformer self-attention vs sequence-length-linear in Mamba. Sub-F's budget choice affects total training FLOPs differently per architecture. **Out of scope for sub-F sub-project; surfaces in training-scaffold sub-project.** Sub-F's budget recommendation feeds the bake-off comparator without committing to one architecture.

### 7.12 Snapshot artifact

`configs/sub_f/sequence_length_analysis.yaml` — frozen at sub-F-close. Contains: joint 4D distribution histograms; budget surface table; chosen elbow; per-type retention thresholds + overrides; truncation strategy (α / β / γ); long-cell diagnostic threshold; per-stage attribution model.

### 7.13 §3 citations

- sub-C extracts schema: `src/cfm/data/sub_c/io.py` (verify file:line at Task 3a per `feedback_verify_before_lock_not_after`).
- sub-A Overture pinning policy: `docs/data/overture_pinning_policy.md` + `configs/data/overture_release.yaml`.
- sub-E boundary contracts: `data/processed/sub_e/2026-04-15.0/singapore/tile=*/boundary_contract.parquet` (handoff line 80).
- §3 encoder grammar formulas: §3.2 + §3.4–§3.6.
- §3 boundary-ref token shape: §3.7.

### 7.14 Residuals

- **Sweden / Sri Lanka deferral.** Stages 1, 2, 4 distributions Singapore-only at v1; multi-region budget revision when Sweden lands (accepted cost per §1.4; §12 #5).
- **Stage 1 – Stage 2 correlation.** Captured in joint distribution (per §7.3); flag if joint shows substantially different shape than independence assumption — informs BP3-v2 prediction model refinement.
- **Architecture coupling out of scope** per §7.11.
- **Truncation choice (α vs β vs γ)** deferred to Halt 4 based on actual surface shape.

---

## 8. §2 paired-check inventory

Every threshold-based verdict in sub-F has a paired structural-correctness check per §2 (protocol). This section enumerates the complete inventory. Each row: verdict mechanism, broken-but-in-range shape it cannot detect alone, paired structural check, implementation site.

### 8.1 Inventory table

| BP / location | Threshold-based verdict | Broken-but-in-range shape | Paired structural check | Implemented in |
|---|---|---|---|---|
| **BP1 / §2.1** | F (frequency floor) admits all tags ≥ F on taginfo | Floor admits tags by frequency only; wiki-canonical major categories silently absent | Hand-enumerate OSM wiki Map_features must-appears (taxonomy snapshot); every must-appear must be a first-class slot in vocab | Task 1 (`vocab_floor_analysis.yaml`) |
| **BP1 / §2.1** | Singapore-prioritized X-threshold ("Singapore-frequency ≥ X must appear above F") | X admits the wrong tail; X-paired structural check absent | Singapore X-threshold paired with its own enumerate-from check; candidate framings at Halt 1 | Halt 1 (sub-decision within `vocab_floor_analysis.yaml`) |
| **BP2 / §3.8** | L_∞ vertex-error tolerance for round-trip | Mean L_∞ passes while right-angle geometries skew to parallelograms (angles drift) | 95th-percentile angle on input-right-angled corners must clear threshold; **ALL four** geometry classes round-trip independently (closed polygon, multi-vertex polyline, right-angle building footprint, road crossing cell boundary) | Halt 2 + integration test (Task 13) |
| **BP3 / §7.4** | Budget quantile (P99/P99.5/…) on joint distribution | Aggregate passes while one feature type silently truncates | Per-feature-type retention rates **ALL of** clear thresholds independently (roads ≥99.9%; buildings / POIs / landuse-class ≥99.0%); fraction of cells with any truncation event clears threshold; per-stage attribution on truncated cells reported | Halt 4 |
| **BP4 / §2.3** | Vocab includes `<unknown_*>` family of correct count (~15) | Family present but never fires (over-permissive floor) OR family dominates (under-permissive floor) | Each `<unknown_*>` slot's Singapore occurrence count reported; zero-firing slots flagged as over-permissive-floor candidates (BP1 revisit); over-firing slots flagged as under-permissive-floor candidates | Halt 3 |
| **BP5 / §5.2** | Integration test passes (same-process + fresh-process byte-identity) | Test happens to land in a deterministic codepath while encoder is non-deterministic elsewhere | **ALL** per-axis unit tests (one per row of §5.2 discipline table) pass AND same-process integration test passes AND fresh-process integration test passes — single failure halts the determinism contract; per-axis tests use **constructed boundary inputs** per §5.4, not real-data-derived | §5.4 + Task 5b |
| **BP6 / §6.8** | Four-axis version manifest validity | Writer remembers fields but values don't reflect running code's actual constants | (1) Field presence (all 4 axes in parquet metadata + provenance.yaml + region manifest); (2) field correctness via direct constant import (no bump-and-revert); (3) cross-axis coupling sanity (SOURCE-only bump leaves DERIVATION at v1.0, and vice versa); (4) parse correctness via extended `compare_version` | Task 6 |
| **BP7 / §3.7 + §4.7** | `<bref_*_*>` tokens emitted in encoded cells | Tokens emitted but disagree with sub-E's boundary contract for the cell × edge | Four-test composite — **ALL of** must pass: (a) cross-reference (every emitted `<bref>` matches sub-E parquet for that cell × edge); (b) symmetry (paired cell views agree on shared edge, opposite directions); (c) non-road non-emission (buildings / POIs emit zero `<bref>`); (d) coverage (active road edges with road features in either neighbor emit at least one `<bref>`, AND symmetry passes on that emission); PLUS standalone BP1 → sub-E class mapping test (hand-enumerate `highway=*` tokens → MAJOR / MINOR via sub-E `derivation.py:47-50` grouping; assert sub-F encoder map matches) | Task 7 + cross-tile validator (Task 10) |

### 8.2 Discipline invariants

Three principles surface repeatedly:

- **"ALL of" not "at least one of"** (BP2 fix 1 protocol-level lesson). Every structural check enumerating multiple cases requires ALL cases to pass independently. Aggregate-passes-with-one-failure is exactly the broken-but-in-range trap §2 exists to close.
- **Hand-enumeration from external source** (Gate 6). Wiki Map_features, sub-E `BoundaryClass` enum, sub-D `lattice.py` axis convention, sub-C `enums.py:23 AXIS` — all hand-enumerated. Assertion logic does NOT use sub-F's own derivation in expected-value computation.
- **Constructed inputs not real-data sampling** (BP5 plan-write refinement). Per-axis tests hit exact bin edges with synthetic inputs. Real data won't reliably land on boundaries.

### 8.3 §3 citations for paired-check sources

- OSM wiki Map_features taxonomy snapshot: `configs/sub_f/wiki_map_features/2026-04-15.0.{wikitext, sha256, revision_id}` (BP1).
- taginfo frequency snapshot: `configs/sub_f/taginfo/2026-04-15.0.csv` (BP1).
- sub-E `BoundaryClass` enum: `src/cfm/data/sub_e/derivation.py:19-23` (BP7).
- sub-E class-grouping rule: `src/cfm/data/sub_e/derivation.py:47-50` (BP7 standalone class-mapping test).
- sub-E rotation: `src/cfm/data/sub_e/rotation.py:50-62` (BP7 symmetry test).
- sub-D lattice convention: `src/cfm/data/sub_d/lattice.py:5-17` (BP7 direction interpretation).
- sub-E per-tile parquet sort order: handoff line 80 (BP7 cross-reference + coverage tests).

### 8.4 Cross-references

- BP5 per-axis unit tests defined in §5.2 + §5.4; this section indexes.
- BP7 four-test composite defined in §3.7 + §4.7 cross-tile validator; this section indexes.
- BP6 four-part check defined in §6.8; this section indexes.

---

## 9. §3 citation index

Consolidated table of every upstream contract reference by file:line, indexed by upstream module and external source. Used by Task implementers as the authoritative cite list per `feedback_external_source_of_truth_gate` (read source, don't infer from naming) and `feedback_verify_before_lock_not_after` (verify file:line is current before locking the cite).

### 9.0 BP1 external sources

| Cite | Purpose | Snapshot artifact | Sections that consume |
|---|---|---|---|
| `taginfo.openstreetmap.org` | Empirical global tag frequency input for BP1 floor F | `configs/sub_f/taginfo/2026-04-15.0.csv` | §2.1, §8.1 BP1 |
| `wiki.openstreetmap.org/wiki/Map_features` | Canonical taxonomy for Gate 6 must-appears enumeration | `configs/sub_f/wiki_map_features/2026-04-15.0.{wikitext, sha256, revision_id}` | §2.1, §8.1 BP1, §8.1 BP7 (standalone class-mapping) |

External sources are URL-pinned at snapshot time; reproducibility lives in local snapshot artifacts (URLs are time-varying per BP1 fix B).

### 9.1 Sub-A (Overture pinning + source data)

| Cite | Purpose | Sections that consume |
|---|---|---|
| `configs/data/overture_release.yaml` | Build-time read for `SUB_F_SOURCE_VERSION`; single source of truth (BP6 fix 2) | §1.2, §6.1, §6.5 |
| `docs/data/overture_pinning_policy.md` | §3 policy reference for pinning semantics | §1.2, §6.5 |
| Sub-A reader source (file:line at Task 6) | Verify `release_id` semantic per §6.5 — three-branch outcome (source-data-pinning ID / output hash / code version — defend if ambiguous per `feedback_ambiguous_third_branch_in_verification`) | §6.5 |
| Sub-A vertex-order guarantee (chain through Overture → sub-A → sub-C, file:line at Task 5a) | Verify before encoder inheritance; three-branch outcome per `feedback_ambiguous_third_branch_in_verification`; default canonicalize for ambiguous case | §5.6 |

### 9.2 Sub-C (tile-extracted features)

| Cite | Purpose | Sections that consume |
|---|---|---|
| `src/cfm/data/sub_c/io.py` (file:line at Task 5a + Task 3a) | Sub-C row-order convention for feature iteration determinism; sub-C extracts schema for BP3 stage-1 distribution | §3.3, §5.2, §7.13 |
| `src/cfm/data/sub_c/enums.py:23 AXIS` | Upstream-most axis convention source (axis=0 = x = E-W; axis=1 = y = N-S); sub-D lattice inherits | §3.4 |
| Sub-C feature-splitting convention (cross-ref to §9.6 verify-before-lock pending) | Informs §3.7 multi-outbound case necessity (per §3 fix 4 deferral) | §3.7, §9.6 |
| `data/processed/sub_c/<release>/<region>/tile=*/features.parquet` | Per-tile feature data for sub-F encoder + BP3 Task 3a empirical input | §1.2, §3.3, §7.2 stage 1 |

### 9.3 Sub-D (lattice + macro plan + versioning precedent)

| Cite | Purpose | Sections that consume |
|---|---|---|
| `src/cfm/data/sub_d/lattice.py:5-17` | Canonical 8×8 cell-grid axis convention; sub-F coordinate frame + BP7 direction interpretation inherit | §3.1, §3.4, §3.7 |
| `src/cfm/data/sub_d/lattice.py:5` | `slot_index = cell_i * 8 + cell_j` row-major convention for per-cell ordering | §4.2, §4.3 |
| `src/cfm/data/sub_d/io.py:_MACRO_CORE_SCHEMA` | Pinned `pa.schema` precedent (store all derivable identifiers + inline-validator-as-derivation-check) | §4.2 |
| `src/cfm/data/sub_d/io.py` write_parquet routing | Sub-D byte-identity precedent via `cfm.data.io.write_parquet` + pinned `PARQUET_WRITE_KWARGS` | §5.2 |
| `src/cfm/data/sub_d/versions.py` | Three-axis versioning precedent (DATA_SHAPE / VOCAB / DERIVATION); sub-F extends to four-axis adding SOURCE | §6.1 |
| Sub-D `compare_version` (file:line at Task 6) | Helper to extend in-place for four-axis (BP6 fix 1, option a default); ordered fallback to variable-axis-count then `compare_version_v2` | §6.4 |
| Sub-D rounding mechanism (file:line at Task 5a) | Sub-F inherits whatever sub-D uses for FP round tie-breaking (BP5 verify-before-lock per `feedback_verify_before_lock_not_after`); assumed default `round()` round-half-to-even pending verify | §5.2 |

### 9.4 Sub-E (boundary contracts + storage precedents)

| Cite | Purpose | Sections that consume |
|---|---|---|
| `src/cfm/data/sub_e/derivation.py:19-23` | `BoundaryClass(IntEnum)` exact values; `BOUNDARY_NOT_APPLICABLE=0` sentinel-never-on-disk discipline | §2.3, §3.7, §4.5, BP7 |
| `src/cfm/data/sub_e/derivation.py:27-31` | `_HIERARCHY` multi-class collapse rule (MAJOR > MINOR > NONE) | §3.7 |
| `src/cfm/data/sub_e/derivation.py:47-50` | `class_grouping_map` for BP1 → sub-E class mapping standalone test (§8.1 BP7 row) | §8.1 |
| `src/cfm/data/sub_e/writer.py:23-25` | `SlotKind(IntEnum)` INTERNAL_EDGE=1, EXTERNAL_EDGE=2 | §3.7 |
| `src/cfm/data/sub_e/writer.py:28-30` | Per-tile parquet structure (112 internal + 32 external = 144 rows) — sub-F mirrors with 64 cells / tile | §4.2 |
| `src/cfm/data/sub_e/writer.py:33` | Pinned `pa.schema` with explicit nullable flags precedent (sub-F mirrors) | §4.2 |
| `src/cfm/data/sub_e/rotation.py:50-62` | Per-cell rotation `cell_to_edge_ids` (post-Task-14 fix commit `8e90869`); direction-label semantic for boundary refs | §3.7 |
| `data/processed/sub_e/<release>/<region>/tile=*/boundary_contract.parquet` | Per-tile boundary contracts consumed by BP7 cross-reference test + BP3 stage-4 | §1.2, §7.10 (Task 3c), §8.1 BP7 |
| `tests/data/sub_e/test_singapore_integration.py::test_layer3_deterministic_rerun_same_process` | Same-process + fresh-process byte-identity precedent | §5.1, §5.8 |
| Sub-E handoff line 77 | Region manifest pattern (`<region>/manifest.yaml` with `manifest_sha256` self-integrity) | §4.1, §6.6 |
| Sub-E handoff line 80 | Per-tile output path scheme `tile=EPSG3414_i{ti}_j{tj}/` + sort order `(slot_kind, slot_index)` | §3.1, §4.1, §5.2 |
| Sub-E handoff line 188 + fixup `4ffa101` | `canonicalize_yaml` idiom + `EXCLUDED_FROM_SHA` mechanism for live-clock-excluded provenance fields | §4.6, §6.6 |
| Sub-E handoff line 198 + fixup `fd53fdd` | `_SUCCESS` validate-then-touch ordering discipline | §4.6 |
| Sub-E handoff line 199 | Halt-on-validator-fail discipline (no partial sub-F output) | §4.6 |
| `configs/macro_plan/v1/boundary_vocab.yaml` | Locked sub-E vocab + class-grouping map source; sub-F's BP1 → sub-E mapping test consumes | §8.1 BP7 |
| `src/cfm/data/sub_e/versions.py` | Three-axis versioning twin to sub-D | §6.1 |

### 9.5 Cross-cutting protocol

| Cite | Purpose | Sections that consume |
|---|---|---|
| `docs/protocols/sub-project-planning-protocol-v1.md` | **Operational meta-contract** — gates, principles, exemptions, audit-after-fixup. Governs HOW sub-F operates, not WHAT it cites. | header note, §13 ledger |

### 9.5b Memory entries (constitutive principles)

Memory entries operate as discipline lenses; failure to apply produces the defect class the memory exists to close. Cited not as data sources but as reasoning anchors.

| Entry | Discipline lens | Sections that apply |
|---|---|---|
| `feedback_external_source_of_truth_gate` | Sixth gate — cross-reference new abstractions against existing module's docs / source; assertion logic does not use new abstraction in expected-value computation | §8.2, all Gate-6 cross-ref tests |
| `feedback_append_only_vocab_safety` | Vocab YAML append-only constraint informs BP6 minor-bump rules + BP4 unknown family permanence | §2.4, §6.3 |
| `feedback_marginal_cost_of_cut` | Elbow-finding framework for BP1 curve + BP3 budget surface | §7.4, Task 1 |
| `feedback_schema_vs_data_cost_asymmetry` | Schema cheap, data expensive — informs BP4 symmetric-re-tokenization analysis + BP3 truncation default | §2.3, §7.9 |
| `feedback_pyarrow_hive_partition_inference` | Per-tile reads use `pq.ParquetFile(path).read()`; bare `pq.read_table()` injects spurious `tile` column | §4.9 |
| `feedback_brainstorm_gate_discipline` | Topic-by-topic gating during sub-project brainstorm; surfaced cross-bite-point revisions in §13.1 | §13 |
| `feedback_subagent_branch_pattern` | Implementer halts on first defect; no silent inline fixes | §5.6, §10.3 |
| `feedback_consult_planning_protocol_before_brainstorm` | Read protocol end-to-end before brainstorm; cite during topic-by-topic gating; sub-project close decides whether to bump | header note, §13 |
| `feedback_pythonhashseed_dict_iteration_test` | Vocab dict iteration test uses `PYTHONHASHSEED=random` across cold pytest invocations | §5.4 |
| `feedback_ambiguous_third_branch_in_verification` | §3 verifications plan three branches (yes / no / ambiguous); ambiguous defaults to defend | §5.6, §6.5, §9.1, §9.2 |
| `feedback_verify_before_lock_not_after` | Spec-level locks citing upstream must verify FIRST; lock is "pending verification" until Task N closes | §5.2, §6.4, §9.3, §9.4 |
| `feedback_provenance_scope_placement` | Per-tile carries only tile-specific provenance; shared metadata at region / release / config scope | §6.6 |

### 9.6 Verify-before-lock pending references

Three cites with assumed defaults pending Task-time verification per `feedback_verify_before_lock_not_after`:

| Cite | Assumed default | Verified at | Cascade on mismatch |
|---|---|---|---|
| Sub-D `round()` rounding mechanism | Python `round()` round-half-to-even per PEP 3141 | Task 5a | §9.6.1 |
| Sub-D `compare_version` API extensibility | In-place four-axis extension is backward-compatible | Task 6 | §9.6.1 — fallback chain advance (in-place extension → variable-axis-count → `compare_version_v2`) |
| Sub-C feature-splitting convention | Y-roads split into single-row-per-branch features (so §3.7 multi-outbound case is unnecessary) | Task 7 | §9.6.1 — §3.7 multi-outbound case added per §3 fix 4 deferral; impacts BP7 vocab estimate by +N tokens for multi-outbound shape |

#### 9.6.1 On verification mismatch

When a verify-before-lock check returns a value that contradicts the assumed default, the resolution discipline is:

1. **Sub-D wins by default.** Sub-F is the downstream consumer; sub-D (or whichever upstream is cited) is the canonical convention source. Sub-F inherits sub-D's actual behavior verbatim.
2. **Lock value updates in the originating upstream-section.** §5.2 row for `round()` updates to whatever sub-D actually uses; §6.4 row for `compare_version` advances by one step in the fallback chain (in-place extension → variable-axis-count → `compare_version_v2`); §3.7 multi-outbound case added if §9.0 sub-C feature-splitting mismatch.
3. **§2 paired check re-validates.** §5.2 mismatch → determinism integration test re-runs against the updated lock; §6.4 mismatch → field-correctness test re-runs against the updated `compare_version` path; §9.0 mismatch → §3.7 multi-outbound case added with paired structural check extended to cover it.
4. **§13 revision ledger entry mandatory.** Verification timestamp + assumed default + actual upstream behavior + cascade triggered + sub-F update applied. Documents the inheritance event for future audit.

Mismatch is the expected outcome of a fraction of verify-before-lock checks per `feedback_verify_before_lock_not_after`. The cascade is the operational path from "verification surfaces mismatch" to "spec + tests consistent again."

---

## 10. Governance + halt-points

Sub-F's operational backbone: seven reviewer-halt gates across Tasks 1–7. Each halt has the same shape — implementer surfaces specific data, reviewer approves on specific criteria, specific artifacts lock after approval. Halts are NOT optional; per `feedback_subagent_branch_pattern` and `feedback_verify_before_lock_not_after`, no autonomous lock is permitted on any of these.

### 10.1 Halt-points table

| # | Halt name | Task | Data surfaced by implementer | Reviewer approves | Artifacts that lock after approval |
|---|---|---|---|---|---|
| 1 | **BP1 vocab floor elbow** | Task 1 | **L1 (full, 28 keys) + L2 (load-bearing per cascade #4 Singapore-X scope: highway + building) + L3 deferred per §12.** Marginal-cost curve over the in-scope levels (level, vocab_size, must-appears_admitted); X-threshold candidates for Singapore-prioritized must-appears (highway + building only). Other 26 keys' L2 elbows deferred — if reviewer wants broader after Halt 1, separate Halt 1b spawns scoped to specific keys (recursive marginal-cost-of-cut). | Granularity level + sub-floor exception list + X-threshold value + X-threshold paired structural check framing | `configs/sub_f/vocab_floor_analysis.yaml`, `configs/sub_f/semantic_vocab.yaml` |
| 2 | **BP2 encoder primitives + round-trip thresholds** | Task 2 | Joint (direction_count × magnitude_quantum) surface against Overture turn-angle + vertex-spacing distributions; anchor scheme {flat, hierarchical} reported with BOTH vocab size AND mean-sequence-length-per-cell; round-trip L_∞ vertex-error + 95th-pct angle-on-right-angled-corners proposals; collinearity-admission threshold proposal | Direction count, magnitude quantum, anchor scheme, both round-trip threshold values, collinearity threshold | `configs/sub_f/encoding_primitives.yaml` |
| 3 | **BP4 unknown family slot list** | Task 4 | Per-key `<unknown_*>` slot enumeration derived from BP1 Gate 6 + Singapore occurrence distribution per slot + §2 over-firing / zero-firing threshold proposals | Slot list + occurrence thresholds | `configs/sub_f/unknown_family.yaml`, `configs/sub_f/sentinel_inventory.yaml` |
| 4 | **BP3 sequence-length budget surface** | Task 3c | 4D joint distribution per feature type + budget surface (quantile × data_loss_per_type × sequence_length × padding_overhead) + per-type retention proposal table + truncation strategy proposal (α / β / γ) + long-cell diagnostic threshold. Note: Tasks 3a (joint distribution by type) and 3b (per-cell length sans cross-cell) ship outputs to 3c without intermediate reviewer-halt; only the 3c-aggregate halt has an approval gate. | Budget quantile elbow + per-type retention overrides + truncation strategy + long-cell diagnostic | `configs/sub_f/sequence_length_analysis.yaml` |
| 5 | **BP5 vertex-order + rounding verifications** | Task 5a | Vertex-order chain (Overture → sub-A → sub-C) verification outcome (branch a/b/c per §5.6); sub-D `round()` rounding mechanism verification outcome | Inheritance-vs-canonicalize choice (default canonicalize if ambiguous per `feedback_ambiguous_third_branch_in_verification`) + rounding lock | §5.2 per-axis test suite (durable artifact, no YAML per BP5) |
| 6 | **BP6 SOURCE semantic + `compare_version` extensibility** | Task 6 | Sub-A `release_id` semantic verification outcome (branch a/b/c per §6.5); `compare_version` extensibility outcome (in-place / variable-axis-count / `compare_version_v2` per fallback chain); region-vs-tile provenance scope precedent verification | SOURCE-bump semantic + `compare_version` path + provenance scope schema | `src/cfm/data/sub_f/versions.py`, region manifest schema in `<region>/manifest.yaml`, post-N reserved block confirmation in `configs/sub_f/sentinel_inventory.yaml` |
| 7 | **BP7 boundary-ref vocab + sub-C feature-splitting** | Task 7 | Boundary-ref 8-token vocab (verified against sub-E enums by file:line); sub-C feature-splitting verification outcome (single-row-per-branch vs branched-multi-row) | Vocab lock + decision on §3.7 multi-outbound case necessity | `configs/sub_f/boundary_reference_vocab.yaml`; §3.7 grammar update if multi-outbound case required |

### 10.2 Halt-point dependencies

Halts run partially in parallel; dependency edges constrain ordering:

- **Halt 1 (Task 1) → Halt 3 (Task 4).** `<unknown_*>` family is per-key derived from BP1's locked Gate 6 enumeration.
- **Halt 2 (Task 2) → Halt 4 (Task 3c).** Stage-3 token formula requires encoder lock.
- **Halt 7 (Task 7) → Halt 4 (Task 3c).** Stage-4 cross-cell overhead requires BP7 vocab lock.
- **Halts 5, 6 are externally orthogonal** (determinism vs version axes) **but internally-conflicting if both surface sub-D source modifications.** Serialize on sub-D edit conflict: Halt 6 (`compare_version` in-place extension) precedes Halt 5 (rounding inheritance), since version manifest extension is broader-scope than rounding lock and likelier to require sub-D source edits.

Full Task-level DAG with subtask split (Task 3a/3b/3c, Task 5a/5b) in §11.

### 10.3 Halt protocol

Every halt follows the same gate discipline:

1. Implementer reaches halt point; surfaces required data + proposals in a single report.
2. Implementer commits report under `reports/2026-MM-DD-phase-1-sub-F-task-N-halt.md`.
3. Implementer waits — does NOT auto-lock artifacts. No silent inline fixes per `feedback_subagent_branch_pattern`.
4. Reviewer applies approval criteria (per §10.1 column 5).
5. Reviewer either approves → implementer locks artifacts → next task unblocks; OR reviewer pushes back → implementer revises → re-halt.
6. Mismatch on verify-before-lock items triggers §9.6.1 cascade.

### 10.4 Halt-point telemetry for protocol-bump candidates

Halts that surface defects revising prior decisions get logged in §13 revision ledger. Recurring revision patterns (e.g., "every Task N halt surfaces the same kind of upstream-contract drift") are protocol-bump candidates per `feedback_consult_planning_protocol_before_brainstorm` — flag in sub-F-close handoff for potential protocol v2 derivation.

### 10.5 Halt cost telemetry

Each halt records three metrics in its report:

- **Implementer-time-to-data-surface** — wall-clock from task start to halt report committed.
- **Reviewer-time-to-decision** — wall-clock from halt report visible to approval / pushback decision.
- **Total-halt-duration** — wall-clock from halt start to next task unblock.

Aggregated at sub-F close. Halts with anomalous duration (>2 standard deviations from sub-F's per-halt mean) are protocol-bump candidates — either halt-data-surface protocol needs refinement (implementer-side cost) or reviewer-decision criteria need sharpening (reviewer-side cost). Surface in sub-F-close handoff per `feedback_consult_planning_protocol_before_brainstorm`.

---

## 11. Task DAG

Fifteen-task sequence with dependency edges; mirrors sub-E's 15-task shape (handoff lines 117–161). Task numbering reflects topological order of brainstorm bite-points first, then implementation tasks (writer → validator → orchestrator → CLI → tests → handoff).

### 11.1 Task list with blockers

| # | Task | Blocked by | Halt | Artifact(s) produced |
|---|---|---|---|---|
| 1 | BP1 vocab floor analysis (curve + elbow proposal) | — | **Halt 1** | `configs/sub_f/vocab_floor_analysis.yaml`, `configs/sub_f/semantic_vocab.yaml`, `configs/sub_f/taginfo/2026-04-15.0.csv`, `configs/sub_f/wiki_map_features/2026-04-15.0.{wikitext, sha256, revision_id}` |
| 2 | BP2 encoder primitives + joint surface + round-trip thresholds | — | **Halt 2** | `configs/sub_f/encoding_primitives.yaml` |
| 3a | Stage-1+2 joint distribution by feature type (on sub-C Singapore) | — | — | (intermediate; feeds Task 3b) |
| 3b | Stage-3 compound (per-cell length sans cross-cell overhead) | Task 2, Task 3a | — | (intermediate; feeds Task 3c) |
| 3c | Stage-4 compound + 4D joint + budget surface + retention table + truncation proposal | Task 3b, **BP7 design lock** (brainstorm output, NOT Task 7 implementation per §7.10) | **Halt 4** | `configs/sub_f/sequence_length_analysis.yaml` |
| 4 | BP4 unknown family enumeration + Singapore occurrence distribution | Task 1 | **Halt 3** | `configs/sub_f/unknown_family.yaml`, `configs/sub_f/sentinel_inventory.yaml` |
| 5a | BP5 verifications (vertex-order chain + sub-D `round()` mechanism) | — | **Halt 5** | verification outcomes; lock decisions in §5.2 table |
| 5b | BP5 per-axis test suite implementation | Task 5a, encoder code from Task 8 | — | per-axis test suite under `tests/data/sub_f/test_per_axis_*.py` |
| 6 | BP6 version manifest + `compare_version` extension + sub-A `release_id` semantic verification | — | **Halt 6** | `src/cfm/data/sub_f/versions.py`, region manifest schema update, post-N reserved block confirmation |
| 7 | BP7 boundary-ref vocab lock + sub-C feature-splitting verification | — | **Halt 7** | `configs/sub_f/boundary_reference_vocab.yaml`, §3.7 grammar update if multi-outbound case required |
| 8 | Writer (per-tile parquet + per-tile `provenance.yaml` + region `manifest.yaml` with `vocab_sources` block) | Tasks 1, 2, 4, 5a, 6, 7 | — | `src/cfm/data/sub_f/writer.py` + integration with `cfm.data.io.write_parquet` |
| 9 | Inline validator (per-cell schema + token-ID-range + derivation `cell_slot_index == cell_i*8 + cell_j` check) | Task 8 | — | `src/cfm/data/sub_f/validator_inline.py` |
| 10 | Cross-tile validator (BP7 four-test composite + cross-axis coupling + version manifest consistency) | Task 9, Task 7 impl | — | `src/cfm/data/sub_f/validator_crosstile.py` |
| 11 | Pipeline orchestrator (sub-E precedent: writer → inline validator → cross-tile validator → `_SUCCESS` validate-then-touch; halt-on-validator-fail) | Tasks 8, 9, 10 | — | `src/cfm/data/sub_f/pipeline.py` |
| 12 | CLI scripts (`derive`, `validate`, `encode`, `decode`) | Task 11 | — | `scripts/sub_f/{derive, validate, encode, decode}.py` |
| 13 | Integration tests against cached sub-D / sub-E / sub-C Singapore (per-axis + same-process + fresh-process + four-test composite) | Task 11, Task 5b | — | `tests/data/sub_f/test_singapore_integration.py` |
| 14 | Empirical gate + round-trip correctness against real cached Singapore (BP2 four-case round-trip + BP3 retention + BP7 Gate-6 cross-reference on real data) | Task 13, Task 3c | (terminal verification gate) | `tests/golden/sub_f/round_trip/<...>.yaml`, golden round-trip distributions |
| 15 | Handoff document at sub-F close | Task 14 | — | `docs/handoffs/2026-MM-DD-end-of-sub-F.md` |

### 11.2 Critical path

```
Task 2 ──┐
         ├─→ Task 3b ──→ Task 3c ──→ Halt 4 ──→ Task 14
Task 3a ─┘                                       │
                                                 │
Task 1 ──→ Halt 1 ──→ Task 4 ──→ Halt 3 ──┐      │
                                          │      │
Task 7 ──→ Halt 7 ─────────────────────────┤      │
                                          │      │
Task 5a ──→ Halt 5 ────────────────────────┤      │
                                          │      │
                                          ↓      │
Tasks 1,2,4,5a,6,7 ─→ Task 8 ─→ Task 9 ─→ Task 10 ──→ Task 11 ─→ Task 12
                                                          │      │
                                                          │      └─→ Task 13 ──→ Task 14 ──→ Task 15
                                                          │             ↑
                                                Task 5b (after 5a + 8) ─┘
Task 6 ──→ Halt 6 ─────────────────────────────────────────┘ (version manifest into writer)
```

### 11.3 Parallelism opportunities at branch open

At T0 (sub-F branch open), the following tasks can start concurrently — all independent of upstream lock tasks:

- **Task 1** (BP1 vocab floor) — only needs taginfo + wiki snapshots.
- **Task 2** (BP2 encoder primitives) — needs cached sub-A / sub-C distributions (already present).
- **Task 3a** (Stage 1+2 joint) — only needs cached sub-C Singapore data.
- **Task 5a** (BP5 verifications) — verifies upstream sources; reads-only.
- **Task 6** (BP6 version manifest) — reads sub-A release lock; extends `compare_version` in sub-D. **Sub-D edit serialization vs Task 5a per §10.2** (Halt 6 before Halt 5 on sub-D edit conflict).
- **Task 7** (BP7 boundary-ref vocab) — reads sub-E enums by file:line; verifies sub-C feature-splitting.

Six tasks in parallel at branch open. Halts 1, 2, 5, 6, 7 fire approximately concurrently — reviewer should expect to handle multiple halts in a single review session at the start of sub-F implementation.

### 11.4 Halt-to-task mapping summary

| Halt | Task | Unblocks |
|---|---|---|
| Halt 1 | Task 1 | Task 4, Task 8 |
| Halt 2 | Task 2 | Task 3b, Task 8 |
| Halt 3 | Task 4 | Task 8 |
| Halt 4 | Task 3c | Task 14 |
| Halt 5 | Task 5a | Task 8 (vertex-order outcome shapes encoder), Task 5b |
| Halt 6 | Task 6 | Task 8 |
| Halt 7 | Task 7 | Task 8 (vocab inclusion), Task 10 (validator), Task 3c (stage-4 formula via brainstorm lock, not implementation) |

### 11.5 Task estimated complexity (rough)

Drawing from sub-E's actual commit count + LOC pattern (sub-E handoff line 121–161 shows 41 commits across 15 tasks):

- **Light tasks** (1–3 commits, <300 LOC): Tasks 1, 4, 5a, 6, 7, 15.
- **Medium tasks** (3–6 commits, 300–800 LOC): Tasks 2, 3a, 3b, 3c, 5b, 8, 9, 12.
- **Heavy tasks** (6+ commits, 800+ LOC): Tasks 10, 11, 13, 14.

Total estimate: ~50 commits across 15 tasks. Plan-fixup commits per sub-E pattern (~20 expected) ride on top, surfaced via Halts + per-task pre-dispatch audits.

**Sub-F plan-fixup count vs sub-E's ~20 baseline is a protocol-effectiveness measurement.** If sub-F ships substantially fewer (<15), evidence the protocol is reducing defect-surface as designed. If sub-F ships similar or more (≥20), the protocol is not reducing defect-surface as expected — protocol-bump candidate. Surface at sub-F close in handoff under a "protocol effectiveness" subsection (parallel to sub-E handoff's "Discipline observations" at line 209).

### 11.6 Cross-references

- Critical path diagram: §11.2.
- Halt details: §10.1.
- Halt serialization: §10.2.
- Halt cascades on verify-before-lock mismatches: §9.6.1.
- Halt cost telemetry for protocol-bump candidates: §10.5.
- Spec deferral ledger (what sub-F-v1 does NOT ship): §12.

---

## 12. Deferral ledger

Fifteen entries enumerating what sub-F-v1 does NOT ship — to be reopened by a future sub-project. Within-sub-F verify-before-lock pending references (sub-C feature-splitting, `compare_version` fallback, sub-D `round()`) are inventoried at §9.6, not here. §12 is cross-sub-project deferrals only.

### 12.1 Deferral inventory

| # | Deferred item | Trigger to re-open | Target sub-project | Source |
|---|---|---|---|---|
| 1 | **Position-aware boundary contracts.** Sub-E v1 is class-only (`{MAJOR_ROAD, MINOR_ROAD}`); sub-F-v1 boundary-ref tokens carry direction + class but NOT crossing position. Position is implicit via feature's last vertex coordinate. | sub-E §15 #9 trigger fires (PRD §5 stage-3 exact contracts needed) | sub-E-v2 + sub-F-v2 joint | §1.4, §3.7, BP7 |
| 2 | **Runtime stitching gate.** Sub-F-v1 consumes sub-E's `boundary_contract.parquet` as authoritative; no cross-cell consistency check at encode time. Sub-E §10.x invariants are the upstream guarantee. | sub-E §15 #10 trigger fires (stitching test as 4th sub-bar) | sub-E-v2 + sub-F-v2 joint | §1.4, §3.7 |
| 3 | **Non-road cross-cell features.** Buildings / POIs that span cell boundaries are clipped at geometry layer in v1 (last vertex on edge, no `<bref>` token). Sub-E v1 boundary contract is roads-only. | sub-E-v2 ships expanded boundary contract surface covering non-road feature types | sub-F-v2 (downstream of sub-E-v2) | §1.4, §3.7 |
| 4 | **Cross-environment determinism.** Sub-F's within-env contract locks per §5; cross-env (darwin/aarch64 ↔ Leonardo linux/x86_64) deferred to end-of-Phase-1 — same residual as sub-D §15 #7 and sub-E §15 #7. | First sub-F Leonardo run | end-of-Phase-1 verification (not its own sub-project — a verification pass) | §1.4, §5 |
| 5 | **Multi-region (Sweden + Sri Lanka) coverage.** V1 vocab + encoder calibrated against Singapore only. F (frequency floor) uses global taginfo with Singapore-prioritized must-appears (BP1 option c); Sweden + Sri Lanka are accepted-cost residuals. Sweden ingest TODO at `docs/known_issues.md:178-190` remains deferred. | Multi-region cold-fetch lands (~8h per country per `project_overture_cold_fetch_slow`) | sub-F-multi-region (potential sub-project) or sub-F-v2 if combined with other v2 work | §1.4, §2.1, §7.14 |
| 6 | **Anchor inbound redundancy collapse.** With explicit `<bref_W_MAJOR>` at sequence start, `anchor_x_q = 0` is structurally implied for west edge (analogously per direction). V1 keeps redundancy for decode simplicity (no edge-coord substitution branch). Savings ~0.7 tokens/cell if collapsed. | sub-F-v2 sequence-length-compression review | sub-F-v2 | §3.7 residual, BP7 note 2 |
| 7 | **Sub-D / sub-E manifests missing SOURCE axis.** Retrospective: sub-D and sub-E both transitively depend on sub-A but their version manifests don't surface SOURCE. Sub-F establishes the four-axis precedent; sub-D / E manifests stay three-axis at v1. | Protocol v2 derivation (sub-F close + future sub-project close) considers retrospective patching | sub-D-v2 or sub-E-v2 (separate sub-projects); not sub-F's scope | §6.10 residual |
| 8 | **BP2 cheap-to-keep v2 candidates.** Three encoder parameters default-toward-conservative under cheap-to-keep rule: direction count (default 16; v2 may revise to 24 if curved-road precision matters more than vocab compaction); magnitude quantum (default 0.5m; v2 may revise to 0.25m if sub-meter precision matters); anchor scheme (default flat; v2 may revise to hierarchical if sequence-length is binding constraint). | Empirical evidence from training-scaffold or downstream that one of the three parameters is binding | sub-F-v2 | §2.2, §3.4, §3.5, §3.6 |
| 9 | **`<unknown_*>` family granularity revision.** v1 ships ~15 per-key slots (per BP1 lock). | Training-scaffold loss analysis surfaces a feature class where per-pair `<unknown_*>` granularity (vs current per-key) would meaningfully improve coverage on Singapore-frequency tags below F. Specifically: model perplexity gap > threshold on a feature class where per-key collapse is provably the limiting factor. | sub-F-v2 | §2.3, BP4 |
| 10 | **L3 (all wiki-documented pairs) curve deferred entirely from Halt 1** (cascade #5 outcome). v1 ships L1 (full 28 keys) + L2 (highway + building only per cascade #4 scope) curve points. L3 enumeration would require fetching + parsing 28 wiki template pages (~1500+ pairs) at plan-author time; marginal benefit for v1 de-risk is zero given Singapore X-threshold's narrow scope. | Training-scaffold loss analysis surfaces an L3-needed feature class for highway or building (i.e., a value below the L2 primary set is provably the limiting factor for some downstream task). | sub-F-v2 OR scoped Halt 1b extension per recursive marginal-cost-of-cut framing | §10.1 row 1, §13.1 cascade #5 entry |
| 11 | **POI + base Singapore X-threshold scoping deferred** (cascade #4 outcome). Sub-C `feature_class=2` (poi) has NULL `class_raw` and lumps amenity+shop+leisure; `feature_class=3` (base) lumps water+landuse+natural with ambiguous parent key. v1 Singapore-prioritized must-appears computation scoped to highway + building only. | Sub-C category disambiguation work lands (parse `categories_primary` for POI key→amenity/shop/leisure mapping; disambiguate base sub-keys via geometry+subtype). | sub-F-v2 (combined with L3 extension if both triggered together) | §10.1 row 1, §13.1 cascade #4 entry |
| 12 | **Value-tail capped at first 999 results where applicable for non-cascade-#4-scope L1 keys** (cascade #6 outcome). Value rows for 26 non-cascade-#4-scope L1 keys are captured up to the first 999 taginfo results per key. Fourteen of these have >999 total upstream values and are intentionally capped in sub-F-v1: `amenity, barrier, craft, healthcare, historic, landuse, leisure, man_made, natural, office, route, shop, tourism, water`. `highway` fits single-page coverage (534 values); `building` is cascade-#4 scope and is paginated. | sub-F-v2 expands L3 enumeration to include rare values for any over-999 capped key. | sub-F-v2 OR scoped Halt 1b extension per recursive marginal-cost-of-cut framing | §10.1 row 1, §13.1 cascade #6 entry |
| 13 | **sub-E highway tiering inherited by BP7.** sub-F-v1 faithfully tokenizes sub-E `boundary_contract.parquet`; sub-E's MINOR-default can under-tier `motorway` as MINOR and over-emit non-vehicular or ambiguous ways as MINOR. | sub-E-v2 revisits grouping map semantics (`motorway` -> MAJOR; explicit decision on non-vehicular way handling). | sub-E-v2; sub-F consumes the revised contract downstream | §3.7, Halt 7 cascade #9 |
| 14 | **sub-E per-edge MultiLineString collapse inherited by BP7.** same-edge MultiLineString / multi-part road crossings collapse to sub-E's single per-edge `BoundaryClass`; sub-F emits one `<bref>` per edge and does not ship multi-outbound grammar in v1. | sub-E-v2 evaluates whether boundary contracts need richer multi-crossing representation than one class per edge. | sub-E-v2; possible sub-F-v2 only if sub-E schema changes | §3.7, Halt 7 cascade B |
| 15 | **BP7 sub-E output verification debt.** Task 7 lacked local `data/processed/sub_e/.../boundary_contract.parquet`, so motorway-only and same-edge MultiLineString emission behavior is code-inferred from sub-E writer/pipeline, not parquet-observed. | sub-E output is regenerated or restored for training-scaffold / sub-F integration; spot-check actual parquet emission against Task 7 inference and update limitation docs if it diverges. | sub-F-close or training-scaffold verification pass | Halt 7 close checklist |

### 12.2 Trigger condition discipline

Each deferral carries a SPECIFIC trigger condition, not a vague "if needed later." Triggers are either:

- **Upstream event** (e.g., sub-E §15 #9 fires; Sweden cold-fetch lands) — passively waiting on another sub-project's status.
- **Downstream evidence** (e.g., training-scaffold finds parameter binding) — waiting on a future sub-project's data.
- **Verification pass** (e.g., first sub-F Leonardo run) — actively scheduled at known infrastructure event.

No deferral is open-ended. If a trigger cannot be specified, the item is either in scope for v1 or should be re-evaluated for relevance.

### 12.3 Deferral classification

| Class | Count | Examples |
|---|---|---|
| Sub-E / F-v2 joint sub-project | 3 | #1, #2, #3 |
| Sub-F-v2 standalone | 6 | #6, #8, #9, #10, #11, #12 |
| Other-sub-project (sub-D-v2 / sub-E-v2) | 3 | #7, #13, #14 |
| Verification pass | 1 | #15 |
| Multi-region expansion | 1 | #5 |
| End-of-Phase-1 verification pass | 1 | #4 |

Six of twelve (50%) lock sub-F-v2's scope upfront (#6, #8, #9, #10, #11, #12) — exactly the kind of forward-looking residual list `feedback_append_only_vocab_safety` + `feedback_schema_vs_data_cost_asymmetry` argue should be tracked at v1 close, not rediscovered at v2 open. Entries #10 + #11 surfaced from cascade #4 + #5 at plan-write; entry #12 surfaced from cascade #6 at reviewer-side redispatch check. Defense-in-depth pattern catches deferral candidates earlier in the cycle.

### 12.4 Cross-references

- Cross-bite-point revision ledger (what was REVISED during brainstorm, distinct from deferred): §13.
- Halt-point cascades on verify-before-lock mismatches: §9.6.1.
- Sub-E precedent for deferral-ledger discipline: sub-E spec §15.1 (handoff lines 302–363).

---

## 13. Revision ledger

Meta-discipline section. Documents decisions revised DURING brainstorm (cross-bite-point + plan-write + section-level) plus durable memory entries generated. Distinct from §12 deferrals (what sub-F-v1 doesn't ship) and from sub-E's "Reviewer-confirmed design decisions during implementation" (implementation-time, not brainstorm-time). §13 is sub-F's innovation — a brainstorm-time revision audit trail.

### 13.1 Cross-bite-point revisions

Decisions where a later bite-point's lock modified an earlier bite-point's text. These are the highest-stakes revisions because they prove the brainstorm's topic-by-topic discipline catches inter-decision interactions that batch-decision would miss.

| Revision | Source | Effect | Spec section affected |
|---|---|---|---|
| BP2 → BP4: `<cell_start>` / `<cell_end>` dropped from on-disk vocab | BP4 sentinel co-decision recognized parquet row-per-cell structure makes cell sentinels redundant | Encoding-primitive vocab estimate decreased from ~10 to ~8 structural sentinels (now 6 named + per-family reserve gap) | §2.2, §4.5, §3.3 |
| BP5 → BP6: SOURCE axis surfaced as required component of version manifest | BP5 vertex-order verification realized sub-F's determinism is transitive through sub-A; manifest must surface dependency | Sub-F adopts four-axis versioning (DATA_SHAPE / VOCAB / DERIVATION / SOURCE); sub-D / E three-axis precedent retroactively flagged for protocol-bump consideration | §6.1, §12 #7 |
| BP6 → BP1: snapshot IDs explicitly tracked in provenance | BP6 fix 4 required taginfo + wiki revision IDs as auditable provenance, not just config-level pinning | `vocab_sources` block added to region manifest (NOT per-tile per §6 fix 2 + `feedback_provenance_scope_placement`) | §2.1, §6.6 |
| BP7 → BP2: round-trip structural check expanded from 3 to 4 cases | BP7 fix 2 recognized boundary-crossing position loss needs its own §2-paired check with larger L_∞ tolerance | BP2 round-trip test now requires ALL of {polygon, polyline, right-angle building, road crossing cell boundary}; tolerance for case 4 locked at Halt 2 | §3.8, §8.1 BP2 row |
| BP7 → BP3: stage-4 cross-cell overhead formula corrected | BP7 fix 1 lock made inbound `<bref>` prepended (+1) and outbound `<bref>` replaces-tail (net 0); asymmetric | Stage-4 estimate dropped from 1.4 tokens / cell to 0.7 tokens / cell | §7.2 |
| **Plan-write → BP6:** sub-D has 5 namespaces, not 3; manifest expands 4 → 6 axes | Pre-dispatch audit at plan-write read `src/cfm/data/sub_d/versions.py` directly and surfaced `VersionNamespace` enum has `ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR`. Brainstorm Gate 6 cited sub-E handoff lines 57-69 as authority for sub-D's state instead of reading sub-D source. §9.6.1 cascade outcome: sub-F adopts all 5 sub-D namespaces + SOURCE = 6 axes. | §6.1, §6.3, §6.6, §6.8 updated to six-axis. Plan Revisions 1+2 in plan document the audit-time cascade. | §6 entire section + §13.3 |
| **Plan-write → BP6:** `compare_version` mechanism is enum-add, not kwarg-add | Pre-dispatch audit found sub-D uses `(VersionNamespace, VersionRef, VersionRef)` signature, not kwarg shape. Extension is trivial enum-add (Python enum semantics: adding a member is backward-compatible). §6.4 fallback chain (a/c/b) becomes moot. | §6.4 fallback chain superseded by enum-add mechanism. | §6.4 + plan §6 step 1 |
| **Plan-write → BP1:** Singapore X-threshold scope narrowed to highway + building (cascade #4) | Pre-dispatch audit of Task 1's `FEATURE_CLASS_TO_KEY` mapping found sub-C `feature_class=2` (poi) has NULL `class_raw` and lumps amenity+shop+leisure; `feature_class=3` (base) lumps water+landuse+natural with ambiguous parent key. Singapore-frequency mapping for poi+base requires `categories_primary`/`categories_alternate` parsing + category→key disambiguation not in v1 scope. | Singapore X-threshold scoped to highway + building only at Halt 1; poi + base Singapore-prioritized must-appears deferred to follow-up per §12. | §10.1 row 1 + §12 new entry |
| **Plan-write → BP1:** L1 enumeration corrected from 15 keys to 28 keys (cascade #5) | Pre-dispatch fetch of wikitext at `configs/sub_f/wiki_map_features/2026-04-15.0.wikitext` found Map_features `==Primary features==` enumerates 28 keys: 15 originally proposed + `aerialway, aeroway, craft, emergency, geological, healthcare, historic, military, office, power, railway, telecom, tourism`. Original 15-key list was reviewer-supplied from memory at BP1 brainstorm and locked without Gate 6 verification against wikitext. | L1 must-appears expanded to 28 keys. L3 deferred entirely; L2 scoped to highway + building only (cascade #4 alignment). | §10.1 row 1 + §13.5 |
| **Plan-write → BP1:** taginfo value fetch revised from `rp=1000` to `rp=999` + building-only pagination (cascade #6) | Reviewer-side check before Task 1 redispatch hit taginfo with the implementation's exact parameters and found `rp=1000` is rejected (`results per page must be integer between 0 and 999`). Follow-up diagnostics over all 28 L1 keys showed `highway` has 534 values (single page complete), `building` has 8759 values (pagination required), and 14 non-cascade-#4-scope L1 keys exceed 999 values but do not affect sub-F-v1 Halt 1 outputs because L2 scope is highway + building and L3 is deferred. | `snapshot_taginfo.py` uses `rp=999`, paginates `building` only per cascade #4 Singapore X scope, and caps non-scope L1 value rows at first 999 results where applicable. `test_vocab.py` asserts `building` has `>= 8000` value rows. Non-scope value-tail coverage deferred to §12 #12. | §10.1 row 1 + §12 #12 + §13.5 |
| **Halt 1 review → BP1/BP4:** Singapore X-threshold filters sub-C unknown sentinels before pass-list derivation (cascade #7) | Halt 1 pass-list surfaced `building=B__UNK__` and `highway=unknown` as high-frequency X candidates. These are sub-C normalization sentinels, not OSM values. Cascade #4 correctly narrowed scope to highway + building but assumed scoped `class_raw` values were raw OSM values; sub-C emits sentinels for unknown/sub-floor cases. | `floor_analysis.py` filters `unknown`, `__UNK__`, and `B_*` values before `derive_x_threshold()`. §3 documents encoder behavior: sub-C unknown sentinels map to BP4 `<unknown_*>` family, not dedicated BP1 semantic slots. X-threshold lock deferred until filtered A'/B' candidates are reviewed. | §3.3 + §10.1 row 1 + §13.5 |
| **Halt 7 review → BP7:** sub-E grouping under-covered locked BP1 highway vocab (cascade #9) | Halt 7 audit found sub-E's grouping omitted 15 locked BP1 `highway=*` values. Follow-up architecture check found sub-F consumes sub-E `boundary_contract.parquet` as authoritative, and sub-E defaults omitted but present `class_raw` values to `MINOR_ROAD`, not NONE. | The earlier sub-F-local override resolution is discarded. The real cascade is upstream: sub-E's MINOR-default under-tiers `motorway` and over-emits non-vehicular or ambiguous values as MINOR. Halt 7 accepts this as a sub-E-inherited v1 limitation; sub-F BP7 locks as faithful passthrough. | §3.7 + §8.1 BP7 row + §13.5 |

Twelve cross-bite-point revisions; five surfaced during brainstorm topic-by-topic gating, seven more surfaced at plan-write pre-dispatch audit + prompt-derivation review + reviewer-side redispatch check + Halt 1 review + Halt 7 integration review. The seven post-brainstorm revisions are defense-in-depth evidence: brainstorm Gate 6 + plan-write audit + prompt-derivation reviewer pass + reviewer-side redispatch check + halt read-through = five-layer redundant coverage on upstream-contract verification. See §13.5 protocol-bump candidate aggregation.

### 13.2 Plan-write refinements

Decisions flagged during brainstorm for explicit treatment at spec-write or implementation time, but not visible at brainstorm-time. These are NOT revisions; they're commitments that get codified during the spec writing phase.

| Refinement | Source | Spec/code site |
|---|---|---|
| BP2 encoder correctness tests must surface unknown-collapse bug class | BP4 close residual | §3.8 round-trip test set + BP2 plan |
| BP5 per-axis unit tests construct inputs at exact bin edges, not real-data | BP5 plan-write refinement 3 | §5.4 |
| BP5 decode byte-identity definition explicit (canonical GeoJSON, not structural equality) | BP5 plan-write refinement 4 | §5.3 |
| BP5 determinism lock artifact is the test suite, not a YAML | BP5 plan-write minor | §5.5 |
| BP6 SemVer canonical form `"X.Y"` locked + `compare_version` normalizes | BP6 plan-write | §6.2 |
| BP6 §2 field-correctness test imports constants directly (no bump-and-revert) | BP6 plan-write | §6.8 point 2 |
| BP6 `provenance.yaml` atomic-write inherited from sub-E `canonicalize_yaml` idiom | BP6 plan-write | §4.6, §6.6 |
| BP7 design-space audit: composite-vs-pair-vs-intermediate rationale rests on rotation-in-vocab structural value, not token-count math alone | BP7 plan-write | §3.7 alternatives |

### 13.3 Section-level fixes during integrated design presentation

Revisions made during §1–§12 reviewer iteration. These shaped the spec but did not revise brainstorm-locked decisions; they corrected the spec's articulation of those decisions. Compact summary:

- **§1:** sub-A pinning input split (build-time read + policy doc); determinism test suite added as durable output; vocab pinning context moved to §2.
- **§2:** per-family ID reserved blocks (sizes at Task 1 + 2 halt); structural sentinels named explicitly; `provenance.yaml` cross-referenced to §6.
- **§3:** anchor convention Read 1 uniform (anchor IS vertex 1 in all cases); round-trip equivalence modulo synthetic-vertex insertion; vertex iteration cross-ref to §5; two-outbound wording removed (deferred per §9.6 sub-C verify).
- **§4:** store-all-three-identifiers + inline-validator-derivation-check pattern (sub-D `_MACRO_CORE_SCHEMA` precedent); int8 / int16 downsizing.
- **§5:** PYTHONHASHSEED=random test framing; ambiguous-third-outcome branch; verify-before-lock for `round()` mechanism.
- **§6:** taginfo_release project-release-anchored; `vocab_sources` lifted to region scope per `feedback_provenance_scope_placement`; `compare_version` fallback ordering (a → c → b).
- **§7:** anchor convention reconciled with §3 + formulas include `<feature_end>`; long-cell diagnostic restated in token space; Task 3c on BP7 design lock (not Task 7 implementation); landuse + natural collapsed into one retention row.
- **§8:** BP6 verdict column widened to "four-axis validity"; BP5 "all of" invariant explicit; BP1 X-threshold row implementation clarified.
- **§9:** sub-C feature-splitting moved to §9.6 with cascade; protocol entry purpose elevated to "operational meta-contract"; memory entries split to §9.5b; BP1 external sources added (§9.0); verification mismatch cascade (§9.6.1).
- **§10:** Halt 4 Task 3a/3b/3c sequencing note; Halts 5/6 sub-D edit-conflict serialization; halt cost telemetry (§10.5).
- **§11:** Task 5 split into 5a (verifications, blocks Task 8) + 5b (test suite, parallel to writer); plan-fixup count as protocol-effectiveness metric.
- **§12:** deferral #11 trigger sharpened; deferrals #7 + #9 moved to §9.6; Sweden + Sri Lanka collapsed into multi-region.
- **§6 (plan-write revision, 2026-05-23):** four-axis manifest revised to six-axis after pre-dispatch audit at plan-write surfaced sub-D's `VersionNamespace` enum has 5 members (not 3 as inferred from sub-E handoff). §6.1/§6.3/§6.6/§6.8 all updated to six-axis. §6.4 fallback chain superseded by enum-add mechanism (Plan Revisions 1+2 in plan document). Audit-after-fixup discipline (protocol §8) applied — spec updated NOW to prevent spec-vs-plan drift; one extra commit on sub-F branch, atomic and discoverable.

### 13.4 Durable memory entries generated during sub-F brainstorm

Four feedback memory entries created (in addition to pre-existing memory consulted). Durable for all future sub-projects:

| Entry | Source bite-point / section | Discipline lens |
|---|---|---|
| `feedback_pythonhashseed_dict_iteration_test` | §5.4 fix | Vocab / dict determinism: vary `PYTHONHASHSEED` across cold processes, not insertion order within process |
| `feedback_ambiguous_third_branch_in_verification` | §5.6 fix | §3 verifications have THREE outcomes (yes / no / ambiguous); ambiguous defaults to defend |
| `feedback_verify_before_lock_not_after` | §5.2 fix | Spec locks citing upstream as source-of-truth must verify FIRST, not lock-then-confirm |
| `feedback_provenance_scope_placement` | §6 fix | Classify provenance fields by scope (config / release / region / tile); per-tile carries only tile-specific facts |

All four indexed in `MEMORY.md` and cross-referenced under §9.5b as constitutive principles. These four are sub-F's contribution to the institutional capital — the analog of sub-E's `feedback_external_source_of_truth_gate` (which was THE protocol-derivation driver).

### 13.5 Protocol-bump candidates surfaced during sub-F brainstorm

Items that suggest the sub-project planning protocol may need v1 → v2 revision after sub-F close:

- **Sub-D / sub-E retrospective SOURCE axis gap** (§12 #7) — if pattern repeats in tokenizer-fix or training-scaffold, evidence that protocol's version-manifest discipline needs a "surface ALL transitive dependencies" principle.
- **Halt-cost-telemetry pattern** (§10.5) — sub-F is the first sub-project to instrument halt timing. If aggregate data shows clear halt-protocol refinements at sub-F close, protocol bump candidate.
- **Verify-before-lock as standalone gate** — three sub-F brainstorm fixes (§5.2, §6.4, §9.0) all reduced to "verify-before-lock not after" — pattern suggests this may warrant promotion from a §3 modifier to a standalone seventh gate.
- **Cross-bite-point revision ledger** (§13.1) — sub-F is the first sub-project to maintain a brainstorm-time revision ledger. If §13.1's eleven revisions (five brainstorm-time + six post-brainstorm audit/review-time revisions) had been silently absorbed (sub-E pattern), specs would be less auditable. Protocol bump candidate: "maintain explicit cross-bite-point revision ledger in spec, including plan-write-time audit cascades."
- **Defense-in-depth working as designed** (NEW protocol-bump evidence) — plan-write pre-dispatch audit caught 3 brainstorm Gate 6 misses (`compare_version` mechanism, sub-D namespace count 5 vs assumed 3, sub-C sort key concreteness). The misses were caused by Gate 6 citing **transitive documentation** (sub-E handoff) as authority for sub-D's state instead of reading sub-D source directly. Proposed protocol-v2 sharpening: Gate 6 trigger phrases like "extending X's existing helper" or "inheriting X's convention" require **direct source read of X's defining file**, not citation of a downstream document that references X. Hand-enumeration assertion (e.g., "sub-D's `VersionNamespace` has N members named A, B, C, …") forces the read to be complete, not sampled. Captures defense-in-depth = brainstorm Gate 6 + plan-write audit on upstream-contract verification.
- **Validator-axis split as sub-F-v2 candidate.** Plan-write surfaced sub-F has two validators (inline + cross-tile) but v1 ships single `SUB_F_VALIDATOR_VERSION`. If empirical evidence at sub-F-close shows validators evolved at different rates (e.g., cross-tile got 3 invariant additions while inline stayed unchanged), v2 candidate is splitting VALIDATOR → VALIDATOR_INLINE + VALIDATOR_CROSS_TILE (7-axis manifest).
- **(i) Transitive-documentation citing forbidden** (proposed protocol-v2 addition). Gate 6 trigger phrases like "extending X's existing helper" or "inheriting X's convention" require direct source read of X's defining file, NOT citation of a downstream document that references X. Sub-D's `compare_version` brainstorm escape cited sub-E handoff (a downstream document representing sub-E's state) as authority for sub-D's state; direct read of `src/cfm/data/sub_d/versions.py` would have surfaced the 5-namespace shape immediately. Operational rule: "read upstream source" is satisfied only by reading the upstream module's own defining file, never by reading a downstream document that summarizes upstream.
- **(ii) Hand-enumeration with complete-count assertion** (proposed protocol-v2 addition). Hand-enumerations against external sources require an EXPLICIT per-section count assertion that is hand-derived independently from the enumeration content. Sub-F's L1 enumeration shipped at 15 keys without a "count = N" assertion against the wiki page's `==Primary features==` section. A `test_l1_must_appears_count == 28` test derived from independently counting the wikitext section's transclusion lines would have caught the gap. Operational rule: every hand-enumeration ships paired with an independently-derived count assertion (e.g., per-section row counts, transclusion counts).
- **(iii) Reviewer-supplied lists as untrusted input** (proposed protocol-v2 addition). When a reviewer supplies a content list (e.g., "the must-appears are highway, building, …") during brainstorm or plan-write, that list is NOT a Gate 6 source — it's input that requires Gate 6 verification against the canonical upstream source regardless of authorial role. Sub-F's 15-key L1 list was reviewer-supplied at BP1 brainstorm; both reviewer and implementer treated it as canonical without wikitext verification. Operational rule: any list of upstream values, regardless of who supplied them, requires Gate 6 verification against the canonical external source before locking.
- **(iv) Dispatch-prompt audit steps reuse implementation call/code path** (proposed protocol-v2 addition). Audit steps in dispatch prompts must reuse the exact API call or code path the implementation uses, not separately-derived shorthand. Pre-dispatch audit's job is to verify the implementation's actual contract holds, not a parallel one. Surfaced by sub-F Task 1 prompt-derivation defect (cascade-equivalent #6): the prompt's wiki audit used deprecated `action=raw` while `snapshot_wiki.py` correctly used `action=query&prop=revisions`.
- **(v) Exact-parameter upstream diagnostics** (proposed protocol-v2 addition). Pre-dispatch audit should diagnostic-call upstream API endpoints with the exact parameters the implementation uses, not just URL patterns. Sub-F cascade #6 proper surfaced because `rp=1000` passed URL-pattern verification but failed taginfo parameter-range validation (`rp <= 999`).
- **(vi) Reviewer-supplied parameter values as untrusted input** (proposed protocol-v2 addition). Reviewer-supplied parameter values are untrusted input requiring upstream-contract verification before plan revision. The reviewer-side check layer caught the reviewer's own contract assumption: `rp=999` was safe for the API range but not sufficient for all 28 L1 keys. Defense-in-depth applies to reviewer assertions, not just agent assertions.
- **(vi-b) Reviewer-supplied premise verification repeats at Halt 7** (additional evidence for candidate vi). BP7 cascade #9 initially accepted reviewer-supplied "drivable-only" and "missing means NONE" premises without grounding them in the pre-existing §3.7 text, sub-E derivation architecture, or sub-E default behavior. Both premises had to be pulled back after direct source checks showed architecture (b) and MINOR-default. For any cascade resolution proposing sub-F local behavior against an upstream, verify derivation ownership first: does sub-F own this derivation, or consume upstream output as authoritative?
- **(vii) Singapore-frequency pass-lists filter normalization sentinels** (proposed protocol-v2 addition). Pass-lists from Singapore-frequency analysis must filter upstream normalization sentinels before X-threshold derivation. Sub-F cascade #7 surfaced sub-C `__UNK__` / `unknown` values admitted to the X-threshold pass-list at high frequency (`building=B__UNK__` at 0.43 before filtering), which would have encoded sub-C normalization artifacts as sub-F semantic vocab slots.
- **(viii) Audit anchor paths verify actual code structure at audit-step-write time** (proposed protocol-v2 addition). Audit anchor paths must verify against actual code structure at audit-step-write time, not inferred from cascade-fix discussion language. Sub-F Task 4 prompt-v1 audit referenced `pipeline.py` because cascade #7 discussion context mentioned pipeline flow; the actual sentinel emission site is `policy.py`, and the more robust dispatch audit verifies sentinel presence in cached sub-C output data instead of grepping an inferred source path.
- **(ix) Hypothesis-falsification surfacing** (proposed protocol-v2 addition). When diagnostic measurement contradicts prior hypothesis classification, surface "hypothesis falsified" explicitly rather than reframing the result to preserve the hypothesis. Sub-F Task 2 continuation surfaced chunking-as-lever falsified by identical L_∞ across four chunk thresholds; classification should have been "chunking is no-op on this sample" rather than "default chunk retained."
- **(x) Late-stage integration composition audits** (proposed protocol-v2 addition). Late-stage integrating sub-projects should budget explicit composition-audit tasks against each upstream they consume because per-upstream correctness does not compose automatically. Sub-F has now surfaced multiple upstream-composition gaps: cascade #7 (sub-C normalization sentinels admitted to BP1 X pass-list), cascade #9 (sub-E MINOR-default under/over-tiering when consumed by BP7), and the Task 7 MultiLineString surface (sub-C MultiLineString rows collapsing into sub-E's single boundary row per edge). Pattern: sub-F as the late-stage consumer pays the composition tax for locally-correct upstream artifacts.
- **Defense-in-depth frequency observation.** Sub-F has now surfaced defects at every defense-in-depth layer: brainstorm Gate 6, plan-write audit, prompt-derivation review, reviewer-side check before redispatch, and Halt 1 read-through. This pattern is evidence that protocol redundancy is load-bearing, not over-engineering.

Surface at sub-F close in handoff under "Protocol-bump candidates" subsection (parallel to sub-E handoff's pattern of surfacing the sixth-gate derivation). Particular attention to defense-in-depth framing — sub-F is the empirical evidence that Gate 6 alone is insufficient; the brainstorm + plan-write audit + prompt-derivation review + reviewer-side redispatch check + halt read-through five-layer combination is the load-bearing redundancy.

### 13.6 Cross-references

- Spec deferral ledger (what sub-F-v1 does NOT ship): §12.
- Verify-before-lock pending references (within-sub-F): §9.6.
- Halt-point cost telemetry for protocol-bump aggregation: §10.5.
- Protocol operating discipline: `docs/protocols/sub-project-planning-protocol-v1.md`.
