# Phase-2 Multi-Region Corpus — Close-out (MERGED 2026-06-08)

**This file is the corpus's provenance record / source of truth.** It documents what
shipped, why, the #19 root-fix, and what is explicitly out of scope.

## Outcome

The validated **42-city** multi-region EU corpus — **670,030,892 tokens, 23,971 tiles**,
every city at sub-F **DERIVATION 1.2** — passed the four-part G4 DoD and merged to `main`.
This corpus is the training input for the held-out-city eval + the architecture bake-off,
which are **separate, following efforts** (see "Out of scope").

## Four-part G4 DoD result (`build_g4_rollup.py`, measured on REAL counts)

| gate | result | detail |
|---|---|---|
| (a) tokens ≥ 550M | **True** | raw total 670,030,892 (clears by +120M) |
| (b) per-city floor | **True** | below_floor=[], not_validated=[] |
| (c) axis-coverage | **True** | uncovered=[] — morphology/density/geography all covered, incl. geography=NL |
| (d) sha-coherence | **True** | version_skew=[] — all 42 counted cities verified at DERIVATION 1.2, not a stale marker |
| **DoD PASS** | **True** | two independent runs (manual + completion poll) produced identical verdicts |

The 550M floor is the PI-ratified v1 floor (reset from the stale 600M pre-measurement
heuristic; real EU yield ~24k tok/tile, not the canary-skewed 56k).

## Shipped 42 vs the originally-ratified ~40-city plan

**5 EXCLUDED (never-extracted / dropped — no sub_f exists):**
`paris`, `lyon`, `madrid` (pathological extraction walls ~30–104 h; FR/ES already covered),
`rome` (not extracted; IT already covered), `welwyn` (unprocessed; GB + modernist-sprawl/
sparse already covered). These were never part of the shipped corpus and do not count
against any gate.

**6 RECOVERED by the #19 de-densify fix** (excluded for inflation in the 2026-06-06
B1-simple close-out, recovered + validated here): amsterdam 60.4M, rotterdam 44.6M,
warsaw 32.6M, almere 26.3M, lodz 12.9M, a_coruna 8.1M.

**7 add-cities** (all validated): eindhoven 26.0M, tilburg 24.4M, linz 12.0M,
wolfsburg 8.7M, telford 7.6M, szczecin 7.1M, debrecen 4.9M.

Net: the corpus grew from the ~40-ratified plan to 42 shipped — the inflation exclusions
were reversed at root by #19, and NL coverage (amsterdam/almere/eindhoven/tilburg/rotterdam)
is closed.

## #19 — root-fix (the central work of this sub-project)

The 0.5 m magnitude quantum inflated over-densified Overture polygons: `encode_feature`
floored every sub-quantum segment to one quantum (`max(1, quantize_coord_m(seg))`), so a
ring of ~0.04 m segments decoded ~12× longer and tripped sub-G's
`decoded_vertex_within_cell_bound`. Fix: `encoder.dedensify_coords` — radial-distance
simplification at tolerance = the quantum (drops sub-quantum micro-vertices the codec
cannot represent, preserves anchor/closure/bref endpoints; the budget twin matches).
Teeth-proof: `tests/data/sub_f/test_dedensify.py` (over-densified building 6.80×/372 m →
~1.0×/within-bound; over-simplify guard + discrimination teeth). DERIVATION 1.1→1.2 bump
forced a whole-corpus re-derive (sub-F only — sub_c/d/e byte-untouched, confirmed at the
eindhoven gate) and blocks pre/post-fix version skew. Shrink was correct removal of
sub-resolution noise: NL ~3% (worst), most cities <1.5%, many <0.5%.

## #20 — OVERTURNED (rotterdam / warsaw re-admitted)

The "degraded source, do-not-re-run" verdict was correct given the encoder of the time,
but the degradation was inflation **severity** on a few buildings (small token-share),
**not a different corruption class** — and #19 removes it at root. After the de-densify
re-derive both pass sub_g clean AND the path-length spot-check
(`spotcheck_pathlength.py`): undistorted vs the accepted eindhoven baseline
(mean ≈1.0×, **zero** buildings >1.5×; worst real building ~1.3× = the codec ceiling, see
below). PI re-admitted them; +77.1M, counted in the 670M. Validator-clean alone was held
insufficient — the spot-check against an accepted baseline was the bar.

## Codec ceiling — distinct from #19 (do NOT mistake for unfixed inflation)

After the fix, per-building decoded/source path-length is centered at ~1.0× (mean
0.997–0.999×, p99 ~1.05×) with a thin tail: the **worst single building is ~1.3×
(max ~1.6×) in EVERY city, including accepted ones** (eindhoven worst-large 1.314×). This
is the inherent encode/decode quantization ceiling (0.5 m magnitude + 1° direction bins),
governed by the BP2 round-trip L_inf lock — **NOT residual #19** (which was pervasive and
would lift the mean/p99). A future reader must not read the ~1.3× tail as unfixed inflation.

## Byte-compare proof — the fix's reach (condition #4)

`bytecompare_confined.py` on the two least-changed cities proved the fix changed **only
over-densified geometry**: umea 854/854 changed features had a sub-quantum segment;
toledo 12,322/12,322; **0 inverse-failures (changed-without-sub-quantum) across ~145k
features.** The over-simplify guard's per-ring claim holds on real data.

## Peer-median sanity outliers — pre-existing, NOT regressions

The groups=0-not-sufficient check flagged 3 cities for low tile-count vs morphology/density
peers: barcelona (72 tiles), milton_keynes (42), munich (171). These are **pre-existing
extent / box-sizing characteristics** — the sub-F-only re-derive preserves tile counts by
construction (the tile set comes from unchanged sub_c/sub_e), so the counts are identical
pre/post-fix. They passed validation (groups=0); surfaced for review, not gate failures.

## Marker-trust safeguard (gate_d)

This stretch saw a 3rd orchestration false-completion (a chunking driver wrote a false DONE
on control-flow). Principle generalized: **no completion marker is authoritative unless it
verified the actual end-state.** Enforced in the merge gate as gate_d (sha-coherence) — a
city counts only if its `_PHASE1_VALIDATED` marker AND its sub_f is at the current
DERIVATION sha; a stale marker fails the gate. The false-DONE driver was deleted; the
recovery driver verifies 0-pending before writing DONE. See `known_issues` and
the `feedback-no-marker-without-endstate-verify` memory.

## Out of scope (deferred — NOT addressed by this sub-project)

- **#13** sub_c admin_region lookup hardcodes country_code='SG' → None for non-SG tiles.
- **#14** admin_region granularity not comparable across countries (subtype='region').
  (#13/#14 are a ⛔ HARD GATE before any value-bearing conditioning / Task 7 / bake-off.)
- **#15** sub_c tiles the fallback bbox, not the real Overture admin polygon.
- **#16** the v1.2 relax's drop-guard `assert_lossless_clip` is TESTED-BUT-UNWIRED
  (no production caller).
- **Eval-set-generation** (held-out-city split, macro-plan-coherence metric, de-Singapore
  generalizations) — scoped in `reports/2026-06-06-eval-set-gen-scoping.md`, not built.
- **The architecture bake-off** — the next, separate phase.

## Provenance

- Branch `phase-2-corpus-completion`, merged to `main` `--no-ff` 2026-06-08.
- Corpus lives on Leonardo `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed`
  (release `2026-04-15.0`), single-copy; sub_f at DERIVATION 1.2.
- G4 report: `reports/2026-06-05-phase-2-g4-corpus-dod.yaml` (regenerated at merge).
- Key commits: de-densify fix + teeth-proof; per-city lock; fan-out harness;
  re-admission bookkeeping; gate_d marker-trust safeguard.
