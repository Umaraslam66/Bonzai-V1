# sub-G T11 BP7 symmetry halt — root cause: sub-F encoder N/S vs the BP7 direction authority

**Date:** 2026-05-31 · **Branch:** `phase-1-sub-G-cross-artifact-validator` · **Status:** root cause proven; fix needs human approval (touches merged sub-F + sub-G).

> **Correction note:** an earlier draft of this report named `sub_e.rotation.cell_to_edge_ids`
> as the bug and recommended fixing sub-E. That was wrong — it rested on a corrupted/hallucinated
> read of a "geographic N=+y" comment in `boundary_reference_vocab.yaml` that **does not exist**.
> The clean facts below put the fix in **sub-F (encoder) + sub-G (seam-2)**, which is also what the
> fix-simulation showed (swapping the *encoder* N/S → 0 failures). See "Process notes".

## TL;DR

The BP7 cross-tile symmetry halt is **not** F1 (over-strict check) or F2 (sub-C
clipping). Root cause: the sub-F **encoder** (`encoder._direction_of_endpoint`)
and, latently, **sub-G seam-2** (`seam_contract_tokens._endpoint_edge`) label a
cell's faces by **geographic compass** (cell-local **y=250 → "N"**). But the BP7
direction **authority** — the LOCKED `configs/sub_f/boundary_reference_vocab.yaml`,
whose `source_references` defer the meaning of N/S/E/W to
`sub_e.rotation.cell_to_edge_ids` — defines a cell's **north edge as the one shared
with `(i, j-1)`, i.e. the LOW-y edge**. So the encoder emits/looks-up the contract
on the **opposite j-edge**, dropping or mislabeling N/S brefs. **100% of the 2728
symmetry failures are on the N/S axis; reconciling the encoder → 0 failures.**

## Evidence (all clean, reproduced)

| Source | N/S convention | Role |
|---|---|---|
| `boundary_reference_vocab.yaml` (LOCKED Halt 7) | no geographic def; `source_references` defer N/S/E/W to `cell_to_edge_ids` | **authority** |
| `sub_e/rotation.py::cell_to_edge_ids` (code + docstring agree) | north of `(i,j)` = edge shared with `(i, j-1)` = **low-y**; `(4,2).N=(4,1,1)`, `(4,2).S=(4,2,1)` (reproduced 5×) | defines the convention |
| `sub_f/validator_cross_tile.py::_neighbour_cell` | pairs `(i,j).N ↔ (i,j-1).S` — **matches** `cell_to_edge_ids` 56/56 | ✅ consistent |
| `sub_f/encoder.py::_direction_of_endpoint` | cell-local **y=250 → "N"** (geographic) | ❌ **outlier (the bug)** |
| `sub_g/seam_contract_tokens.py::_endpoint_edge` | cell-local **y=extent → "N"** (geographic) | ❌ latent outlier (sub-G hasn't run) |
| `sub_f/decoder.py` | bref vertex position is **v2-scoped / dropped** in v1 (placeholder) | not N/S-sensitive |

- **Axis split (shipped `cells.parquet`):** `{NS: 2728, EW: 0}` over 306/494 tiles. N/S-only is the signature of an N/S convention mismatch (a clip defect would be ~axis-symmetric).
- **Re-derivation:** current convention reproduces the failures (2857 N/S); swapping the **encoder's** N/S → **0**.
- **Worked example — road `e7be7863`, tile `i10_j10`, shared edge `(4,2,1)`:** raw geometry is **symmetric** (reaches cell-local y=250 in (4,2) and y=0 in (4,3), same x=131.279). (4,2) emitted `[E,W,W]` — the crossing bref was **dropped**: the encoder labels the y=250 endpoint "N", looks up `contract[(4,2)]["N"]` = `cell_to_edge_ids(4,2).N` = edge `(4,1,1)` = NONE → nothing emitted. The road physically lies on `(4,2,1)`, which `cell_to_edge_ids` calls the cell's **S** edge.

## Why it survived all sub-F tests

- Fixtures are internally self-consistent (built with the encoder's geographic convention).
- The cross-reference leg passes: the encoder uses the same (geographic) label for BOTH the bref it emits and the contract class it checks → self-consistent.
- Only the symmetry leg fails — it pairs the encoder's emitted direction against `_neighbour_cell` (which follows `cell_to_edge_ids`), the one place the two conventions meet on the same axis.
- sub-G seam-2 would NOT catch it either: its `_endpoint_edge` uses the same geographic convention as the encoder, so its prediction and the encoder's emission agree (both wrong together). Same defect **class** as `feedback_external_source_of_truth_gate` (sub-E/sub-D axis mismatch, 2026-05-20).

## Which convention is "correct"? (the preflight question)

- **By the locked authority:** the BP7 vocab defers to `cell_to_edge_ids`, so "north = the (i,j-1) / low-y edge" is the defined meaning. The encoder + seam-2 are the outliers. → **fix sub-F + sub-G.**
- **Geographically:** SVY21 Northing increases with +y, and higher `cell_j` = higher y (confirmed from the data: (4,3) sits at higher y than (4,2)). So `cell_to_edge_ids`'s "north = lower j" is geographically **inverted** (its "north" points geographic south). The encoder's geographic label is the intuitive one.
- **Resolution:** bref direction labels are **internal stitching tags** — never surfaced in output (v1 decoder drops bref vertex position; output GeoJSON geometry comes from the coordinate/anchor/dir/mag tokens, which are correct). So geographic-label correctness is **cosmetic for v1**. The minimal, authority-honoring fix is to make the encoder + seam-2 conform to `cell_to_edge_ids`. Making the labels geographically true would mean changing `cell_to_edge_ids`/sub-D (large blast radius, contradicts the locked vocab) — a **v2 consideration**, flagged not taken.

## Fix scope (needs approval — merged sub-F + sub-G)

1. **`sub_f/encoder.py::_direction_of_endpoint`** — swap N/S: cell-local `y≈0 → "N"`, `y≈250 → "S"` (match `cell_to_edge_ids`). *(core fix)*
2. **`sub_g/seam_contract_tokens.py::_endpoint_edge`** — same swap (else seam-2's bijection starts failing once the encoder is fixed).
3. **Tests / fixtures** pinning the geographic mapping — update in the same commit (lock-and-guards): `tests/data/sub_f/test_encoder*`, seam-2 tests, any fixture asserting N/S brefs from a y-position.
4. **Re-derive sub-F** (re-encode all tiles; N/S brefs in the cache are wrong). sub-E cache UNCHANGED.
5. **Re-validate** → expect 0 symmetry failures (fix-sim).

**No change needed:** `cell_to_edge_ids`, `_neighbour_cell`, sub-F decoder, sub-E code, **sub-E cache** (boundary contract is edge_id-keyed and direction-agnostic — confirmed in writer schema + derivation).

## sub-G coverage gap (record in T12 close handoff)

Seam-2's independence-by-construction ("filtering rule traces to BP7 + T8.5, not
sub-F's classifier") **cannot catch a convention mismatch that the predictor and
the emitter share** — both `seam_contract_tokens._endpoint_edge` and the encoder
use geographic N=high-y, so the bijection looks consistent while both disagree
with the contract authority. Disambiguating a convention dispute needs a **third
independent source** (here: `cell_to_edge_ids` / the vocab `source_references`).
Generalizes the both-sides-agree-on-the-same-wrong-thing failure mode from values
to **naming conventions**.

## Downstream cost & convention debt (record in T12 close handoff §8)

The "cosmetic for v1" call is right for the **output geometry**, but two costs
attach downstream and must be named (neither changes the v1 recommendation —
conform the outliers to the authority now):

1. **Trained-token retrain cost.** `<bref_N_*>` and `<bref_S_*>` are distinct
   trained token IDs. Conforming-to-authority now means **v1 models train on the
   lower-j-is-north convention**. A future geographic-true fix (changing
   `cell_to_edge_ids`/sub-D) would relabel these tokens and **invalidate every
   v1-trained model** — a breaking change requiring retrain. The convention
   choice is therefore not free-to-defer in the way "cosmetic" implies.
2. **Geographic-inversion convention debt.** `cell_to_edge_ids` is geographically
   inverted (its "north" = lower-j = geographic **south** in SVY21). Conforming
   locks a geographically-inverted bref naming into v1. Defensible (bref labels
   are internal stitching tags, never surfaced in output), but record as **KNOWN
   CONVENTION DEBT** with an explicit **v2 trigger:** *if bref direction labels
   ever surface in output (v2 decoder retains bref vertex position) OR an external
   consumer reads the boundary-contract direction semantics, the inversion must be
   fixed at `cell_to_edge_ids`/sub-D — invalidating v1-trained models.*

## Implementation discipline for the fix (when approved)

- **Test-first, external-source-of-truth:** write the failing test BEFORE the
  swap — assert `_direction_of_endpoint` (and seam-2 `_endpoint_edge`) agree with
  `cell_to_edge_ids` (the authority), so the bug is caught by a test that traces
  to the authority, not to the encoder's own logic.
- **Fixtures encoded the bug — correct them, don't "make them pass."** sub-F
  fixtures passed *vacuously* (encoder + decoder + fixture all shared the flipped
  convention). Each fixture update must be verified against `cell_to_edge_ids`,
  not against old encoder output.
- **Decoder is a premise to verify, not assume.** "v1 decoder drops bref vertex
  position, so no decoder change" must be confirmed by reading the decoder's bref
  handling: if it reads the N/S label for ANY reconstruction step, it needs the
  symmetric swap.
- **seam-2 is sub-G fixing its OWN bug.** The geographic convention error was
  baked into seam-2 at T5, independently of sub-F. Independence-by-construction
  prevented seam-2 from reusing sub-F's classifier, but both independently made
  the same `+y=north` assumption — so the bijection was blind to it. The defense
  is the third authority (`cell_to_edge_ids`) both sides trace to. See
  `feedback_independence_misses_shared_assumptions`.

## Reproduce

`uv run python scripts/sub_g/t11_symmetry_diagnosis.py` (read-only). Expect:
all symmetry failures on the N/S axis; current convention reproduces them;
encoder-N/S-swapped re-derivation → 0.

## Process notes

This investigation repeatedly hit silent tool-output corruption (missing
`timeout` on macOS; stdout interleaving from a background job; a self-written
drill with unreliable vertex selection; **hallucinated file content** — a
nonexistent "geographic N=+y" vocab comment, and a corrupted grep that made the
`cell_to_edge_ids` code look like it contradicted its docstring). **Three**
intermediate claims were wrong and retracted after clean, sentinel-terminated,
reproduced reads: (a) a `_neighbour_cell` inversion, (b) a "2680/48" axis split,
(c) "fix sub-E `cell_to_edge_ids`". Every fact in the final version rests on clean
reads; the fix-simulation (swap encoder → 0) is the load-bearing corroboration.
See `feedback_tool_output_trustworthiness_layer`. **Given the error rate, the fix
should be independently verified before any code change.**
