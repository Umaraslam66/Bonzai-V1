# START: Phase-2 bake-off RESUME — first scored numbers (boot 2026-06-11)

**This is a NEW-PHASE handoff, not a continuation.** The readiness-closure +
conditioning-enrichment sub-project is SEALED (merged + pushed). Read pointers,
don't re-derive.

## 1. State

- **main @ `f9c797f`** (merge `--no-ff`, pushed; origin/main == local). Suite
  **1703 passed / 2 skipped / 1 xfailed** (the xfail is the pre-existing Phase-0
  ENTRY marker — leave it).
- **No GPU spent. NO MODEL SCORED YET.** The eval instrument is proven; every
  scored number is still in the future.
- **Leonardo repo at `8f9fc45`** (pre-close) — bundle the close commits across
  before any work there (bundle-only deploy; no GitHub creds on the login node).
- Allocation: renewal pending; nothing GPU until it lands.

## 2. What exists now (capability, one line each — details at the pointers)

Conditioning is DELIVERED (was a blank prefix), with a per-cell continuous
character carrier (`CellPayload.character_stats`) and a sha-locked append-only
city-identity registry; the Phase-2 eval is RE-SCOPED and instrumented — Lane S
(floor-judged generalization, identity-ablated, excess over the measured
same-conditioning floor) / Lane M (nearest-training-city memorization
discriminator, hard halt) / Lane D (seen-city diagnostic); knobs locked
strict-min_T / all-38 / median+p90 / explosion-0.5; the conditioning-floor
artifact is FROZEN (`reports/conditioning_floor/2026-04-15.0/`, schema 2.0,
sha-verified reads only); the decision layer makes a memorizer structurally
uncrownable (`memorization_check_ok` consulted first, over all candidates).
**T5 is CLOSED as re-scoped.**

**Authoritative detail — read these, in order, before acting:**
1. `reports/2026-06-11-readiness-closure-sub-project-close.md` (the arc + backlog)
2. Spec §8: `docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md`
3. `reports/2026-06-11-task25-t5-closure.md` (what the bar now asks, exactly)

## 3. The mission

**Resume the Phase-2 architecture bake-off on this eval and produce the FIRST
SCORED NUMBERS** — does the character bet yield generalization? Train the
candidate backbones (post-renewal), score via `scripts/run_bakeoff_decision.py`
(`cfm.eval.bakeoff_decision.decide`/`pick_winner` against the frozen floor
artifact). The instrument is proven; the result is unknown. That asymmetry is
the whole point — report whatever the numbers say.

## 4. Gates BEFORE any scored GPU-hour (ALL parked; each on Umar's explicit word; NONE assumed)

1. **Allocation renewal** must land.
2. **Step 18.5 resume proof** — kill→resubmit on a short job; the FIRST
   post-renewal GPU job; additionally gated on T0 closure.
3. **Token-budget coupled decision** — DEFAULT_MAX_CELL_TOKENS + model max_len +
   the 2048-vs-5760 commensurability question are ONE decision at scored-run
   planning (15.5's gate fired on GENUINE density; de-densify ruled out with
   evidence; input = `reports/2026-06-11-token-length-investigation.yaml`,
   range ~8.5–13k + recorded tail-drop). Do not set any of the three alone.
4. **Shard-caching design** — every training job start currently pays the
   ~40-min in-memory shard derivation; the cache format (own sha/version
   discipline) comes to Umar as a DESIGN, not a patch.
5. **mamba-lock verify-before-lock** before any scored GPU-hour.

## 5. Backlog

Seven items, recorded in the close summary §"Recorded backlog" — carry forward,
never silently drop. (Token-budget coupled decision is #1 and is gate 4.3 above.)

## 6. Discipline (carried, non-negotiable)

Verified-end-state by recomputation, never exit codes. Stop-before-commit at
gates/forks. Teeth on every gate (red-on-divergence; a guard that passes in both
regimes guards nothing). Halt-on-defect — no improvisation past a fired gate.
ALL CPU extractions run as `lrd_all_serial` Slurm jobs (login node stalls long
walks; compute nodes don't share login `/tmp`). Subagent-driven where building:
implementer ≠ reviewer, two-stage review, TDD red shown, ruff UNPIPED. No
GPU/Leonardo run/merge/push without Umar's explicit word; merges `--no-ff`;
main is shared — NEVER force-push. Umar is the chat-reviewer: bring him
decisions, gate results, and anything that fires — with the evidence.
