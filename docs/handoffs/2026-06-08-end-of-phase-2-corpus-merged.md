# Handoff — Phase-2 multi-region corpus MERGED; next is eval-set-gen → bake-off (2026-06-08)

You are a fresh agent starting cold. Read this whole file before doing anything.

---

## ⛳ STATE (what is true right now)

- **`main` @ `29bbda0`** — a `--no-ff` merge commit (parents `cdc263b` + `75ddc1c`),
  **pushed to origin** (`github.com/Umaraslam66/Bonzai-V1`); `origin/main == local main`,
  verified by fetch.
- **The corpus is DONE:** validated **42-city / 670,030,892-token / 23,971-tile**
  multi-region EU corpus, every city at sub-F **DERIVATION 1.2**, **four-part G4 DoD PASS**
  (tokens ≥550M · per-city floor · full axis coverage incl. geography=NL · sha-coherence).
- **Provenance / source of truth:** `reports/2026-06-08-phase-2-multiregion-corpus-closeout.md`.
  Read it — it records the shipped set, the #19 fix, #20 overturn, the codec ceiling, the
  deferred items. The G4 numbers are in `reports/2026-06-05-phase-2-g4-corpus-dod.yaml`.
- **Tests green:** `uv run pytest` → **1145 passed**, 2 skipped (Leonardo-only berlin
  integration), 1 xfail (pre-existing Phase-0 marker), on `main`.
- **Corpus data** lives only on Leonardo: `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed`
  (release `2026-04-15.0`), single-copy, sub_f at DERIVATION 1.2.
- **Leonardo repo is at `212e3ed`** (last deployed bundle, on branch
  `phase-2-corpus-completion`). The merge commit `29bbda0` + the close-out are on origin
  but **not yet on Leonardo** (they add no code — just the close-out doc + the merge node).
  **Step 0 of any Leonardo work: sync the repo to `main` @ 29bbda0** (bundle or pull).

## 🚧 HARD BOUNDARY — corpus merged ≠ model trained

Nothing has been trained. This sub-project produced the *training input*. The next work
is **eval-set-generation, then the 30M-param architecture bake-off** — a **separate
effort**. Do not treat "corpus merged" as "ready to train": there is a hard gate in the
way (admin_region, see DEFERRED). Do not reopen, re-derive, or "improve" the corpus
without an explicit instruction from Umar — it is shipped and verified.

---

## NEXT WORK — eval-set-generation (scoped, NOT built)

Full scope: `reports/2026-06-06-eval-set-gen-scoping.md`. The eval machinery EXISTS
(`src/cfm/eval/holdout/*`, `slice_metrics.py`, `perplexity_gap.py`, `paths.py`) but is
**Singapore-shaped** — this is a GENERALIZE job, not a rebuild. Three deliverables:

**1. Held-out CITY split (cities, NOT tiles).** Proposed: **tallinn** (medieval-organic,
EE, z35, 5.6M), **glasgow** (planned-grid, GB, z30, 14.7M), **eisenhuttenstadt**
(modernist-sprawl, DE, z33, 4.3M), **munich** (mixed, DE, z32, 10.1M). Rule: every
held-out morphology is ALSO in train, so a held-out miss reads **"didn't generalize"**,
never "never saw this type" — that is the whole point, vs the old Singapore-tile overfit.
Distinct UTM zones exercise the multi-CRS path Singapore never tested. (munich→lisbon
PT/z29 if you want 4 distinct geographies at +11M.)

  - **TRAIN-FLOOR TENSION — RE-CHECK, likely resolved.** The scoping doc (written when the
    corpus was ~551M) concluded "hold out cities → train dips under the 550M r=20 floor →
    add ~3–4 cities." **The final corpus is 670M**, so train = 670M − ~35M holdout ≈ **~635M**,
    already clear of 550M. **Re-verify with the real held-out token sum before adding any
    cities — you almost certainly do NOT need to.** (The floor is a TRAIN-split budget, not
    the corpus total — do not re-conflate them; that was the "56k" class of error.)

**2. Macro-plan-coherence metric — NET-NEW (the one genuinely-new bake-off bar).** A
**shuffle-gated coherence gap** (mirrors `perplexity_gap`'s matched−shuffled): per held-out
tile, neighbour-agreement over the 4-neighbour cell lattice on sub-D macro attributes
(zoning/road-skeleton categorical; density-bucket ordinal ≈ discrete Moran's-I), reported
as `score(true arrangement) − score(spatially-shuffled)`. **Teeth (must be proven before
it gates anything):** (a) shuffle teeth — real tile must beat its shuffle with margin
(else it measures the marginal mix, not arrangement → vacuous); (b) two-sided — also
require generated diversity within a band of the real tile's, so a UNIFORM plan (perfect
agreement, unreal) FAILS; (c) real-vs-random separation on a held-out sample. Threshold is
set AFTER measuring, never guessed.

**3. Mechanical de-Singapore** (cheap plumbing): `paths.py` (per-city CRS label, not
hardcoded `EPSG3414`); `labels.py` (drop the Singapore `morphology_class` constant — read
per-tile sub-C/sub-D values); `resolution.py` (recompute the gap-floor on the **TRAIN split
ONLY** — never let held-out cities into the floor's calibration, that leaks test into the
threshold). Reusable AS-IS: `lineage_audit.py` (region-keyed), the manifest schema,
`perplexity_gap.py` (region-agnostic).

**Three bake-off bars:** loss-decreases ✓ (exists), macro→micro perplexity-gap ✓ (exists),
**coherence ✗ (net-new — deliverable #2 above).**

---

## ⚠️ DEFERRED known_issues (with their gates) — `docs/known_issues.md`

- **#13 + #14 — admin_region: ⛔ HARD GATE.** #13 = sub_c admin_region lookup hardcodes
  `country_code='SG'` → **None for all non-SG (i.e. all EU) tiles**; #14 = admin_region
  granularity not comparable across countries. **You must re-derive + reopen the corpus to
  fix admin_region BEFORE any value-bearing conditioning (Task 7 / a bake-off candidate that
  conditions on admin_region). DO NOT train a value-bearing model on the current all-None EU
  admin_region.** This is the gate that sits between "corpus merged" and "train value-bearing."
- **#15 — polygon extent:** sub_c tiles the fallback bbox, not the real Overture admin
  polygon (deferred; revisit before extent-dependent work).
- **#16 — unwired drop-guard:** the v1.2 relax's `assert_lossless_clip` is TESTED-BUT-UNWIRED
  (no production caller). Wire it at the next regen, which needs sub_c to persist
  `source_clipped_length_m`.
- **Resolved this phase (do not re-open):** **#19** quantum-inflation FIXED at root
  (`encoder.dedensify_coords`); **#20** rotterdam/warsaw "degraded source" **OVERTURNED**
  (it was inflation severity, fixed at root — re-admitted). **Codec-ceiling note:** the
  per-building decoded/source path-length tail of ~1.3× (max ~1.6×) is encode/decode
  **quantization present in accepted cities too — NOT residual #19.** Do not mistake it for
  unfixed inflation.

---

## 📏 STANDING RULES (non-negotiable)

- **Merges + pushes need Umar's explicit word, and use `--no-ff`. The agent NEVER
  auto-merges/pushes** — not even when everything is green. Present the package, await the word.
- **Every destructive / content-changing re-derive uses `scripts/multiregion/guarded_rederive.py`**
  (per-city flock lock + temp-dir + atomic swap + halt-on-byte-change; `--allow-content-change`
  to authorize intentional regen). Never hand-roll an in-place re-derive loop.
- **NO completion marker (`DONE`/`_SUCCESS`/`_PHASE1_VALIDATED`/"complete") is authoritative
  unless its writer verified the actual end-state.** A false DONE is worse than a false RC —
  the pipeline trusts markers as ground truth, so a false one poisons future sessions.
  Generalize this to anything you build. The G4's **gate_d (sha-coherence)** now defends the
  merge gate against a stale marker (a city counts only if marker AND sub_f at the current
  DERIVATION sha). See the `feedback-no-marker-without-endstate-verify` memory.
- **Verify against SOURCE, not markers/projections.** Re-run and read ground truth; don't
  trust an exit code or a status line (the false-DONE lesson; the lisbon mid-validate
  false-positive; the 0/0 spot-check artifact — all caught by re-reading source).
- **Every gate/axis must have a must-distinguish teeth-proof** — confirm it FAILS in the
  regime it targets, else it guards nothing (e.g. the de-densify over-simplify guard; the
  spot-check calibrated to an accepted baseline; the coherence metric's shuffle teeth).

---

## 🧯 ORCHESTRATION SOFT SPOT (read before any long Slurm run)

Hand-rolled Slurm chunking/poll drivers caused **3 false-completions this phase** (a false
DRAINED/DONE from an unguarded empty-jid submit; the earlier double-nohup near-miss; a
stale-sacct scare). **All were caught and none corrupted data** (the per-city guards held),
but this is the **fragile layer**. The recovery driver `wave2b_driver.sh` has the right
shape (sbatch-success guard on empty jid; wait-for-queue-empty before each chunk; double-
confirmed drain; **0-pending verify before writing DONE**) — but **harden or replace this
layer with something robust before the bake-off's long unattended GPU jobs.** `gate_d`
closes the marker-trust hole at the merge gate; it does not fix the drivers themselves.

---

## 🖥️ INFRA — Leonardo (CINECA), account `AIFAC_P02_222`

- **`lrd_all_serial`** is the free CPU workhorse: **2 concurrent jobs** (QOS `MaxJobsPU=2`,
  `MaxSubmitPU=10`, `cpu=8`, `mem≈30800M`/user, **4 h** wall cap). Size jobs so 2 fit
  (≤4 cpu / ≤~15G each) or you get throttled to 1 lane (`QOSMaxMemoryPerUser`). A >10-task
  array is rejected (`QOSMaxSubmitJobPerUserLimit`) → chunk it.
- **`dcgp` is NOT available to this account** (verified: `invalid account or expired budget`;
  this is a GPU/boost allocation with no dcgp QOS, and dcgp is whole-node-exclusive anyway).
- **`boost_usr_prod` = GPU** (4×A100/node, per-node billing, `--authorized-boost-override`).
  Reserved for the **~4,800 GPU-h bake-off (post top-up)** — do NOT burn it on CPU work.
  Boost compute nodes have **no S3 egress** → any new-city fetch must be pre-fetched on the
  egress-capable login node first.
- **Deploy:** git **bundle** is the proven path (login node historically has no GitHub creds).
  Now that `main` is on origin, an `origin` pull may also work — verify. Either way, **bring
  Leonardo from `212e3ed` to `main` @ 29bbda0** at the start of the next Leonardo work.
- **SSH:** a user-authed ControlMaster socket (`Host leonardo`, user `uaslam00`). **It dies
  on laptop sleep** (it is down right now — `Permission denied (publickey)`). Re-auth:
  `step ssh login '<email>' --provisioner cineca-hpc` then `ssh -fN leonardo` (Umar runs
  these via `!`). Slurm jobs + `setsid`-detached scripts survive socket drops; login-node
  loops do not — wrap them in `setsid`/tmux.

---

## ▶️ One-liner to start the next session

> Phase-2 multi-region corpus is DONE and merged (`main` @ `29bbda0`, pushed; 42-city / 670M-token / four-part-DoD-PASS; close-out at `reports/2026-06-08-phase-2-multiregion-corpus-closeout.md`). Start **eval-set-generation** per `reports/2026-06-06-eval-set-gen-scoping.md`: held-out CITY split (tallinn/glasgow/eisenhuttenstadt/munich), the net-new shuffle-gated coherence metric (teeth-proven), and the mechanical de-Singapore of paths/labels/resolution. First re-verify the train-floor against the 670M corpus (likely no add-cities needed), and remember the admin_region HARD GATE (#13/#14) blocks any value-bearing conditioning. Read `docs/handoffs/2026-06-08-end-of-phase-2-corpus-merged.md` first. Do not reopen the corpus; do not merge/push without my word.
