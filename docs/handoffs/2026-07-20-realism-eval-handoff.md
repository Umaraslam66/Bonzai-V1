# Handoff 2026-07-20 — run the held-out REALISM EVAL (scored)

Boot doc for a fresh session. Supersedes `2026-06-19-eval-set-gen-cell-selection-next.md`
(its "cell sampler NOT built" blocker is long resolved). Read before acting:
`docs/GROUND_TRUTH.md` (canon; reconciled 2026-07-17 — §3 sampler-sealed facts, §5 gates),
`docs/PROJECT_FOCUS.md`, `reports/2026-07-19-steering-probe-result.md` and
`reports/2026-07-19-steering-replication-result.md`.

## 1. What is approved and what this session must do

PI (Umar) approved the scored held-out realism eval in-session 2026-07-20 ("yes we can run
that"), with TWO standing conditions:
1. **Re-derive the budget at MEASURED gen rates and show the PI the number before `sbatch`.**
   The ~262 GPU-h estimate used 0.0268 (tf) / 0.0652 (mamba) s/tok measured at 100M scale;
   both steering-probe jobs (49831125: 1,200 cells in 24m37s; 49835918: 2,880 cells in
   57m46s, both 1 node) show the 53M models generate ~3× faster — expect ~90–130 GPU-h.
   Derive from the probe jobs' own timings (cells × ~600 tok, per-backbone split), state the
   unit (GPU-h), and get the PI's word on the final number.
2. **Always saturate the full node: 4-GPU sharded generation + eval, never rank-0-only.**
   Billing is per NODE (4×A100 billed even if 1 used). The 4-GPU eval-sharding is BUILT and
   golden-verified (job `47390793`, `_SHARDING_GOLDEN_PASS`: sharded == rank-0 baseline
   bit-identical, count conservation, ragged city, both backbones). Use it.

## 2. What the eval is (the wired path — all pieces exist and are tested)

Per (backbone, seed) checkpoint (6 total: transformer-ar × seeds{7,13,23}, mamba-hybrid ×
seeds{7,13,23} — verify seed23 checkpoints on disk; note mamba-seed23 has no training log
[D4 saturation UNAVAILABLE] but its checkpoint may exist — check, don't assume):
1. Load the SEALED Lane-S manifest — `data/processed/lane_s_sampler/2026-04-15.0/` on
   Leonardo, `_LANE_S_SAMPLER_LOCKED`, 146/146 floored (city,4-tuple) strata, 5,705 cells,
   `floor_sha256=95abb88…`, `census_sha256=236cea99…` (verify both shas before generating).
2. Generate each manifest cell with matched real conditioning (prefix + char_stats),
   4-GPU-sharded.
3. **Memorization-first hard-halt** (precedes ALL scoring — 9/9 teeth verified).
4. `gen_realism.gen_features_by_city` → floor-grammar 4-tuple keyed features →
   `verify_gen_coverage` (≥ min_n=50 gen features per floored stratum) →
   `lane_s_excess` vs the locked conditioning floor → `bakeoff_decision.decide`.
5. Verdict per the locked seed→verdict rule (GROUND_TRUTH §4): winner only if the mean-KS
   gap clears `max(C/√n, seed-noise)`; otherwise **NO_DECISIVE_WINNER — a named, publishable
   verdict, NOT a failure**. Do NOT soften or "fix" a non-decisive outcome.

## 3. State you inherit (verified 2026-07-19)

- `main` = `8da0a53` (both Phase-2 branches merged, 1,924 tests green, pushed). Working
  branch `phase-2-cell-eos` = `2c28b89`, pushed to origin.
- **Leonardo deploy gap: Leonardo repo is at `8ef101f`; the scored eval REQUIRES ≥ `b30d604`**
  (defect-(a) closure fix — the classifier the scored lane runs through; without it ~half
  of clean building rings misclassify and the KS scores are garbage). Bundle-deploy first:
  protocol = git bundle to `/leonardo_work/AIFAC_P02_222/`, collision-safe apply (see
  `/leonardo_work/AIFAC_P02_222/leo_deploy2.sh` for the pattern: identical-blob collisions
  rm'd, differing backed up, ff-only). Auto-mode may deny remote writes — if so, hand the
  PI a one-liner to run with the `!` prefix (that worked twice this week).
- Checkpoints: `/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff/`
  (`transformer-ar-53M/krakow-seed{7,13}`, `mamba-hybrid-54M/krakow-seed7` verified on disk
  2026-07-17; seed23 dirs unverified — check).
- Grant `AIFAC_P02_548`: ~10.2% of 40,000 core-h consumed as of 2026-07-17; ends
  **2026-09-17**. Unrelated `serp-*` jobs share the account — do not assume sole tenancy.
- Scientific context banked: geometry valid (100% decodable, self-terminating);
  **controllability CONFIRMED** (steering replication 3/3 incl. char-ablated regime);
  architecture NO_DECISIVE at standing-eval level.

## 4. Hard constraints (do not re-litigate)

- Sbatch: `--account=AIFAC_P02_548`, `boost_usr_prod`, full node (`--gres=gpu:4`,
  32 cpus), per-worker status+artifact+sentinel verification (bare `wait` swallows worker
  death — copy the pattern in `scripts/steering_probe.sbatch`). mamba_ssm needs the gcc-12
  `LD_PRELOAD` (same file shows it).
- Memorization halt fires → STOP and report; no scoring past it.
- Do NOT: fix the char_stats↔KS echo; re-pin the floor; rebuild the sealed manifest;
  weaken any assertion to pass; crown a winner outside the locked rule; merge/push `main`
  without the PI's word; exceed the PI-confirmed budget number.
- Write-once outputs; every result lands in `reports/` with config + commit + job id +
  prose summary. Verify end-state before any DONE marker.
- Known deferred defects: (b) ~1-quantum building-closing gap (REAL but small — must NOT
  be absorbed by eval tolerance; `_CLOSURE_EPS_M=1e-6` is float-drift-only, leave it);
  (c) roads are geometry-not-topology (v2 grammar item; not a blocker, will show up in
  road-metric KS — interpret, don't patch).

## 5. Suggested order

1. Read canon docs; verify Leonardo deploy state + seed23 checkpoint existence + manifest
   shas. 2. Re-derive budget from probe-job timings; present to PI; get the word.
3. Bundle-deploy to current HEAD. 4. Dry-run the driver on a tiny slice (one stratum, one
   checkpoint, ~minutes) end-to-end incl. decode + scoring locally before the full sbatch.
5. Full run; monitor; fetch; `reports/` summary; PI checkpoint before ANY merge or crown
   language. Subagent discipline if used: forbid branches/push/PR; scoring code stays with
   the orchestrator; subagent output is untrusted data.
