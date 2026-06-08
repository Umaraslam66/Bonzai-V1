# Phase-2 multi-region corpus completion — close-out (2026-06-06)

**Status:** add-cities extraction in flight; final G4 verdict section below is PENDING those
landing. Decisions, shipped-list composition, and exclusions are FINAL.

## Shipped corpus vs the PI-ratified 40

Ratified set = `canary_v1` (5) + `batch2_v1` (35) = **40 EU cities**. Shipped corpus =
**29 validated + 7 add-cities = 36**, minus 11 excluded. (singapore/berlin are Phase-1
artifacts, outside the G4 600M scope.)

### Added (7, the spec's "if short, add cities" lever) — `addcities_v1.yaml`

eindhoven (NL — **closes the only open coverage axis**), tilburg (NL), wolfsburg (DE),
telford (GB), szczecin (PL), linz (AT), debrecen (HU). All moderate/sprawl (high tok/tile,
NOT dense cores); all pass the single-UTM-zone gate (0 straddlers). Projected ~+150M tokens.

### Excluded (11), by reason bucket

**A. Dropped — zero coverage at extreme cost (3):** paris (FR), lyon (FR), madrid (ES) —
geographies already covered (cergy / barcelona+toledo+a_coruna), pathological ~30/56/104h
extraction walls. Dropping them avoided ~190 wall-hours for zero coverage gain.

**B. Not extracted — coverage-redundant (1):** rome (IT covered, ~26h+ wall) — the 4th of the
13 timeouts not run in the overnight-9.

**C. Inflation-excluded (6)** — all failed sub-G `decoded_vertex_within_cell_bound`, root-caused
to the **#19 quantum-inflation defect** (0.5 m magnitude floor bloating over-densified Overture
polygons). The 6-city prevalence gate (`reports/2026-06-06-inflation-prevalence-the6.txt`) split
them by their OWN building-inflation rate vs the 29-validated reference (≥2× = 0.05%):

  - **C1 — corpus-normal, edge-tripped (3):** lodz (0.03%), a_coruna (0.04%), almere (0.04%) —
    indistinguishable from the 29; they tripped only because a tail-building landed near a cell
    edge. **Recoverable in principle** (see recover-3 below); excluded under B1-simple to avoid
    new validator surface near the deadline.
  - **C2 — degraded source data (2):** rotterdam (~13×), warsaw (~12×) — abnormally over-densified
    Overture buildings (rotterdam ≥4× token-share = 0.50%, 50× the norm). An OVERTURE source-data
    problem (#20), NOT a pipeline bug — a re-run reproduces it.
  - **C3 — elevated, borderline (1):** amsterdam (~3×) — excluded as not-needed once add-cities
    cover the gap.

**D. Unprocessed — non-load-bearing (1):** welwyn (GB + modernist-sprawl/sparse already covered).

## Why B1-simple over recover-3 (B2)

A blanket construction-identity exclusion (recover-3 / B2) would re-validate the 6, but the
6-city gate proved it would MASK the degraded C2 subset (rotterdam/warsaw): their bound-crossing
IS the inflation, so B2's displacement-teeth would not fire — the touch-as-cross "relax masks a
real subset" failure. B2 is also new validator surface, and validator changes were the session's
biggest risk-generator (each needed a teeth-proof; two nearly masked real defects). The ~2-extra
add-city cost is cheap and diversity-positive. **recover-3 (a per-city-prevalence-gated structural
exclusion, with a mandatory teeth-proof) is BANKED for the post-deadline regen window**, where a
validator change gets unhurried review — alongside the #19 de-densify fix.

## Path A (full re-derive) — correctly OFF

The pre-committed rule (logged before measuring) put Path A's trigger at ≥2× building token-share
**≥5%** (or ≥4× ≥1%, or a dense-core morphology ≥10%). Measured corpus-wide: **≥2× = 0.05%,
≥4× = 0.01%, uniform across morphologies** — ~100× under the bar. The inflation is a documented v1
limitation (#19), not a blocker. No re-derive of the v1.2-blessed corpus.

## Guards / process honored

Guarded sub-C path (per-city flock + atomic write); compute-node sbatch only; continue-but-loud;
26-city resume-state re-stamp before driver runs. Nothing pushed — held for the PI's merge word
on the final G4 verdict.

## FINAL G4 VERDICT — PENDING (fill when the 7 add-cities land)

- [ ] Shipped validated total (29 + 7 add-cities), measured directly
- [ ] Gap to 550M closed
- [ ] Full axis coverage confirmed (esp. NL via eindhoven)
- [ ] Per-city table + three-part DoD verdict (`build_g4_rollup.py`, EXCLUDED set applied)
