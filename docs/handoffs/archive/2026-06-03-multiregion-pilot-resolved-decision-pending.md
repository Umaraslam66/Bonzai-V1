# Handoff — multi-region pilot RESOLVED, one decision pending (2026-06-03)

**Open here.** The multi-region (Europe) question is fully de-risked — every technical
unknown measured on real data. What remains is a single **strategy decision (PI's call)**, below.
Do NOT re-derive any of this; it's done and committed.

**Branch** `phase-2-multiregion-crs-utm` (off `main`), tip **`c6253ec`**, synced to Leonardo
(`/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, same tip). 8 commits = 6 Q1 + 1 pilot fix + 1 pilot
scripts/doc. Full record: memory [[project_multiregion_feasibility_audit]] +
`docs/handoffs/2026-06-02-multiregion-feasibility-audit.md` (canonical: audit verdict + bounded
re-cost + Q1 execution + pilot findings).

## What's DONE

**Q1 — CRS parameterization (per-city conformal UTM, one zone from centroid, EPSG-prefixed tile
labels).** Singapore stays `EPSG:3414`, byte-identical. Gated: full fast suite **1038 passed** +
slow Singapore real-data byte-identity **4 passed**. sub_d/sub_f/sub_g were already
multi-region-ready; only sub_c + sub_e needed threading. Plus a `confirm=True` COUNT(*) fetch
optimization.

**Pilot — Berlin (`EPSG:25833`), COMPLETE.** Run on Leonardo (fetch on login/tmux; processing as
Slurm jobs). Resolved numbers:

| Measurement | Value |
|---|---|
| tiles/city | **465** (≈ Singapore 494) → **~44 cities** for a 30M ladder; ~148 for 100M |
| fetch cost | **13.3 min/city** (COUNT-opt), NOT the 8h Singapore cold-fetch |
| CRS code on real EU data | reprojection, large-UTM tile indices (i≈184–208, j≈2900–2918), labels — all correct |
| tokenizer chain sub_d→sub_g | **PASSES, zero code changes**; sub_g cross-artifact validator green → locked macro-vocab + sub_f vocab transfer to EU |
| vocab unknown-rate | **Berlin 41.64% < Singapore 54.77%** (better) → minimal-floor regime, NOT EU under-coverage → **ship vocab as-is**, no re-derivation |

**2 fixes found + fixed (the pilot's job):**
1. *Orchestration:* full-city sub_c OOM-kills on the login node → run as a 120G/32-cpu Slurm job
   (`scripts/berlin_extract.sbatch`). Fetch is fine on login; processing needs a compute node (no
   egress — reads cache).
2. *Real code bug:* `sub_c/validator_cross_tile.py` hardcoded `tile=EPSG3414_*` in all 4 checks (I
   missed it threading; all-Singapore suite was sample-regime-blind). → fixed to derive from
   `manifest.region_crs` (`c1ae046`); regression test exercises a non-Singapore CRS so it can't
   silently re-pass.

**Egress (gate #1):** Leonardo login/serial nodes have S3 egress (Overture `http 200`; GitHub too —
handoff's "can't fetch GitHub" was stale). `dcgp_usr_prod` compute-node egress untested — only
matters past ~148 cities; **skip until/unless the extract grows toward the 100M ceiling.**

## The ONE pending decision (PI's call — pick before next session starts work)

The more-data path is no longer "more-data vs re-frame" hand-waving — it's **measured and bounded.**

- **Option 1 — build the bounded-extract orchestrator + run the ~44-city extract.** ~2–3 weeks,
  fits the renewed allocation window (June-11 is SOFT: renewal confident ~couple days, `$WORK`
  ckpts survive). Preserves the compute-optimal methodology (the nine brainstorm topics) at a 30M
  ceiling that cuts production extrapolation **2400× → ~58×**. The pilot removed every technical
  unknown — it's now a scope/timeline call, not feasibility. **3 known work-items:** thread
  `--commit-sha` to sub_d; handle `_SUCCESS`-after-fix (a stage failure leaves no `_SUCCESS`,
  blocking downstream); fetch+process fan-out (fetch login/tmux, process Slurm).
- **Option 3 — bank the de-risking, re-frame the bake-off to data-efficiency** on current Singapore
  (14.4M-token) data for the near term. A different decision axis than Topic-1 locked.

## NEXT SESSION

Start the **chosen** sub-project with full context:
- **If Option 1:** brainstorm → spec → plan the bounded-extract orchestrator (consult
  `docs/protocols/sub-project-planning-protocol-v*.md` first). Pilot scripts
  (`scripts/berlin_chain.sbatch`, `scripts/diag_berlin_vocab.py`, etc.) are the seed.
- **If Option 3:** re-frame the bake-off; reopen spec §4 (Topic-2 compute axis) for the
  data-limited regime.

**Held pending the decision:** all scored bake-off runs + Task 5 (mamba-ssm verify-before-lock).
Note: the prior session's diagnostic job `44006640` may still be on Leonardo (was on a long eval
pass) — unrelated to this branch.
