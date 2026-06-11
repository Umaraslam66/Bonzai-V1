# T5 CLOSES as re-scoped (Task 25 step 3 — 2026-06-11)

**Verdict: T5 — the multi-region eval bar — is CLOSED under the re-scoped definition
(spec §8, PI-approved 2026-06-11).** The bar is valid by construction: a per-city miss
is now decomposable into "didn't generalize" (excess over the measured floor) vs "city
idiosyncrasy" (the floor itself), and memorization is failed directly by a discriminator
with measured power. The instrument backing the bar exists, is sha-locked, and survived
a two-stage verification with every number independently recomputed.

## The chain of evidence (all on branch `phase-2-readiness-closure`)

1. **Why the old bar died** — gate-(i) FAIL (2026-06-10) → localization diagnostic
   (V0–V4 + attribution 2×2; three bit-identical reproductions) → residual recon
   (`742d0bb`): cross-city character at δ=0.15 is real, broad, shape-dominated, and
   bounded below ~30% significance under ANY bucketed conditioning. Zero-difference
   PASS was structurally unreachable; the T5 spec's own REOPEN branch fired.
2. **The re-scope** (`a0624f8`, PI + Umar): floor-judged Lane S (identity-ablated,
   excess-over-floor, worst-case across held-out cities, binding-city power gate
   retained) + Lane-M nearest-training-city discriminator (all 38 training cities,
   hard halt at crowning) + Lane-D seen-city diagnostic. Four knobs locked.
3. **The instrument** (Tasks 24a/24b/25 step 1; `e05dfe5`…`8f9fc45`): city-identity
   registry (sha-locked, collision-proof), continuous character carrier
   (mutation-proven teeth), floor machinery with two BH families (D-D determinism
   anchor + D-T Lane-M family; PI call after the joint-family trap was caught at
   Gate-2), every named tooth red-proven.
4. **The artifact** (`0026283`): `reports/conditioning_floor/2026-04-15.0/`
   (schema 2.0, sha `95abb88bfaf0a79d…`, write-once + lock marker). Two-stage
   verified: D-D family bit-identical across stages (321 pairs / δ-ladder / 265
   floor_heldout rows, exact equality); floor_all recomputed with 0 mismatches
   (tightened by training cities on 255/265 rows); Lane-M strata 152/152 (D,T)
   combos, selection recomputed 0 mismatches; coverage 42/42 cities zero skipped;
   sanity halts clean (family-1 median KS 0.1379).

## What T5 now asks, exactly

For each held-out city D (identity ablated, character carrier live): generated
geometry must sit within `floor_all(D, metric, stratum)` of D's real distributions
(strict min over all 41 other real cities — the closest-real-city-period bar),
aggregated median + p90 per city, judged worst-case across the 4 cities with the
binding-city power gate; AND must match D strictly better than every one of the 38
training cities on the measured discriminating strata (Lane M) — a regurgitator
fails by construction. Lane S/Lane M refuse to run against an unverified floor
artifact (sha/lock/version), and the discriminating-strata selection provably reads
real data only.

## What this does NOT claim

No model has been scored. T5's closure means the BAR is valid and instrumented —
the bake-off can now measure generalization against it. The first scored numbers
arrive when trained checkpoints exist (post-renewal GPU, separately gated).

## Remaining in the sub-project

Task 26 (decision layer: excess-over-floor quantity, `memorization_check_ok`
pairing, Lane-M must-fire fixtures at the pick_winner seam) + the four older
gated steps (12.5 CRS / 13.5 EU floors / 15.5 token lengths / 18.5 resume proof).
Backlog (open, not silent): locked_yaml.py extraction (rule of three),
shard-derivation caching decision at scored-run planning, `_has_outbound_bref`
re-export, stale "held-out" wording on the now-shared extraction halt messages.
