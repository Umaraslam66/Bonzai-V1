# Readiness-closure + conditioning-enrichment — SUB-PROJECT CLOSE (2026-06-11)

**Branch:** `phase-2-readiness-closure` (off main@`77fdb6d`), local-only, merge to main
`--no-ff` on Umar's explicit word. **Suite at close: 1703 passed / 2 skipped /
1 xfailed** (the one remaining xfail is the pre-existing Phase-0 ENTRY marker; the
Task-12 `--config` xfail was deliberately flipped to a passing test at Task 26).

## Why this sub-project existed

2026-06-10: the Phase-2 bake-off HALTED at gate-(i) — within identical conditioning,
held-out cities' real feature distributions differed, so a per-city eval miss was
ambiguous ("didn't generalize" vs "wasn't told the city" vs "cities just differ").
The readiness audit (17 failure classes) became a locked spec + 27-task/10-phase plan:
close the gap register (Mission A) and make the cross-city eval valid (Mission B).

## The arc

1. **Segments 1–2 (Tasks 0–21):** value-bearing conditioning delivery (F6), EU-training
   path (region/release CLI, sbatch de-SG, union caller, CRS checker, emergence-floor
   artifact machinery, token-length contract), F8 resume + F17 isolation, reader-side
   integrity (sha/lock at read, TRAIN_TOKENS guard, gate-(i) coverage counters).
2. **Segment 3 (Tasks 22–23):** δ=0.15 recalibrated verdict + bref exclusion-by-identity;
   the localization diagnostic (3 Leonardo CPU runs, bit-identical reproductions, V1b/V1d
   attribution + V4) ELIMINATED every cheap conditioning dim; the residual recon proved
   cross-city character real, broad, shape-dominated, and irreducible by bucketed
   conditioning (kitchen-sink bound ~30% / 781 significant pairs).
3. **The re-scope (PI-approved 2026-06-11, spec §8):** the T5 bar's own REOPEN branch had
   fired; the bar became floor-judged generalization — Lane S (identity-ablated excess
   over the MEASURED same-conditioning floor, strict min over all real cities, worst-case
   across held-out cities) + Lane M (nearest-training-city memorization discriminator,
   all 38, hard halt) + Lane D (seen-city diagnostic). Knobs locked: strict min_T /
   all-38 / median+p90 / explosion 0.5.
4. **Tasks 24a/24b:** city-identity floor via a sha-locked append-only 49-city registry
   (the live string-hash had 11 collisions incl. madrid=rome — caught at Gate-2);
   per-cell continuous character carrier `CellPayload.character_stats` (7 floats,
   recon-parity-pinned derivation, Linear projection prefix position, ablations compose;
   the §8 `macro_tokens` wording deviation PI-blessed — tile-level tokens were
   wrong-shape twice). Scheme tag `value-char-v1`.
5. **Task 25:** the conditioning-floor artifact — two BH families (D-D determinism
   anchor + D-T Lane-M family; the joint-family trap caught before the run), produced in
   two staged Slurm runs and verified by recomputation (D-D family bit-identical across
   stages; floor_all 0 recompute mismatches, tightened 255/265 rows by training cities;
   Lane-M strata 152/152 (D,T) combos). **T5 CLOSED as re-scoped** (`d592dac`;
   `reports/2026-06-11-task25-t5-closure.md`). En-route: the training-city tile-inventory
   integration defect (extractor was holdout-only; mock-tested seam) found by the real
   run and fixed one-source.
6. **Task 26:** the decision layer — `decide()` (excess-over-floor, basis enum, strict
   completeness + coherence, floor-sha refusal pre-KS) and `pick_winner()` with
   `memorization_check_ok` consulted FIRST over ALL candidates: a regurgitator with the
   BEST fidelity in the pool is refused by name (mutation-proven both directions, plus
   the loser-memorizer integrity tooth). Original obligations landed: `--config` loader,
   `feature_samples` promotion (closed-ring-without-blocks raises both metrics — the
   building-as-roads class), Point semantics, the 1.358 one-source, compile-outcome,
   sbatch dry-run vetting the TRAINED backbone.

## The gated measurement steps (each on Umar's per-step word; all CPU on lrd_all_serial)

- **12.5 CRS consistency — CLEAN.** 44/44 regions PASS; three-way coherence re-derived
  from raw rows + every `config_crs` cross-checked against the local region configs
  (0 mismatches); 7 EU UTM zones + SG 3414, geographically sane.
  (`reports/2026-06-11-crs-consistency-42cities.yaml`)
- **13.5 EU emergence floors — CLEAN.** Four writer-emitted floors with full provenance
  (munich 2.41 > glasgow 1.71 > krakow 1.06 > eisenhüttenstadt 0.375 — density-ordered);
  independent recompute matched to the last digit on raw integers (eisenhüttenstadt
  38,554 polygons / 25,702 active cells); no 1.96 literal survives any resolution path;
  SG seed intact. (`configs/eval/emergence_floors.yaml`)
- **15.5 token lengths — GATE FIRED, RESOLVED-DIRECTION, NUMBER DEFERRED-BY-DESIGN.**
  11/38 cities exceed 0.5% over the 5,760 budget (barcelona 8.0% … manchester 0.63%);
  alarm verified by independent recompute (almere 1,093/24,511 to the last digit).
  The PI-ordered investigation (`reports/2026-06-11-token-length-investigation.yaml`)
  PROVED the cause is **genuine density, not digitization redundancy**: tokens-per-
  building flat ~15 everywhere; NL source vertices-per-building LOWER than controls;
  the driver is feature count (median over-length rotterdam cell = 406 real features,
  roads 80% of tokens); a 0.5 m shape-preserving simplification leaves fracs 2–8× over
  the gate. **Resolution recorded (Umar, 2026-06-11): de-densify is RULED OUT with this
  evidence; the direction is raise `DEFAULT_MAX_CELL_TOKENS`/max_len toward ~p99.9 of
  the dense cities (input range: ~8.5–10k for p99, ~12–13k for p99.9) with recorded
  per-city tail-drop for the extreme tail (valencia max 19k makes full coverage
  unreasonable). The EXACT number is ONE coupled decision with the model max_len
  parameter and the parked 2048-vs-5760 commensurability question — set together at
  SCORED-RUN PLANNING.** Open item with a determined direction, not an unresolved
  question. (`reports/2026-06-11-cell-token-lengths-38cities.yaml`)
- **18.5 resume proof — PARKED.** The first post-renewal GPU job, additionally gated on
  T0 closure; untouched until the allocation renewal, on its own word.

## Operational lessons banked

Login nodes stall long CPU extractions (faulthandler-diagnosed `ParquetFile` park);
ALL CPU extractions now run as `lrd_all_serial` Slurm jobs. Compute nodes don't share
login `/tmp`. The mock-only-seam class (the extractor's holdout-only assumption) is
closed at the wiring layer where it bit.

## Recorded backlog (carried openly, never silently)

1. **Token-budget coupled decision** (15.5 resolution): budget/max_len/commensurability
   set together at scored-run planning; investigation YAML is the input.
2. **Shard-derivation caching decision** (every training job start pays the ~40 min
   features walk; cache format needs its own sha/version design — PI design, not a patch).
3. `locked_yaml.py` extraction (third instance of the freeze grammar; rule of three met;
   lands with its own cross-instance regression test, never alongside a feature).
4. `_has_outbound_bref` public re-export (chore sweep).
5. Stale "held-out" wording on the now-shared extraction halt messages.
6. `holdout/sizing.py` + `holdout/pipeline.py` carry independent 1.358 literals
   (value-equal, not one-sourced).
7. Legacy lint sweep (`sub_c/io.py`, `analyze_geometry_primitives.py`, ~120 pre-existing).

## What merges (on Umar's word, `--no-ff`)

`phase-2-readiness-closure`: the full task arc (segments 1–3 + re-scope + 24–26 + gated
steps), the frozen conditioning-floor artifact, the emergence floors, the gated-step
reports, the investigation instruments, and this summary. The bake-off resumes on a
valid, instrumented, memorization-proof eval the moment post-renewal checkpoints exist.
