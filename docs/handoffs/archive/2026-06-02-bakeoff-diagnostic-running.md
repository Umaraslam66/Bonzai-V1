# Handoff — Phase-2 bake-off, diagnostic running (2026-06-02)

**Cold-resume safety net.** Written mid-run so a fresh session can pick up if the socket/session dies. Branch `phase-2-bakeoff` (tip `5182791`), synced to Leonardo (`/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, same tip). Local fast suite **1089 passed**, 1 pre-existing Phase-0 xfail.

## LIVE HEADLINE (preliminary — confirm with the full curve at completion)

**The diagnostic is revealing SEVERE OVERFITTING: data, not compute, is the binding constraint.** At step 74,949/110,000 (**epoch 152**), `train_loss ≈ 0.02` (collapsed from 0.86) while `val_loss = 10.6` (risen from 4.07). The ~362-tile training set is **~8M unique tokens** (~493 steps/epoch × 16,384 tok); r≈40 = ~223 passes → massive memorization. **This challenges the ladder's compute-optimal D∝N premise**: you cannot Chinchilla-train a ≥90M model on 8M unique tokens. The real useful r is the **val-loss minimum (early)**, not r≈20. **Do NOT proceed to the scored ladder before resolving the data-size constraint** (options below).

## ⇒ NEXT SESSION: multi-region data-ingestion FEASIBILITY AUDIT (PI direction)

**PI lean (2026-06-02): solve the data wall with MORE DATA, not a methodology re-frame** — more data preserves the nine-topic methodology AND Phase-4 production needs global data regardless. Candidate: **extract all of Europe** (thousands of cities → easily enough unique tokens). **But everything was built + frozen around single-region Singapore (SVY21 / EPSG:3414).** Before committing, AUDIT whether the pipeline can actually ingest multi-region/multi-CRS data.

**This is a FEASIBILITY AUDIT, not a go. Do NOT extract anything. READ the real code — do not estimate optimistically.** Report which parts are Singapore-specific vs parameterized. Four questions:

1. **Coordinate systems (likeliest hard blocker).** Pipeline is anchored to SVY21/EPSG:3414 (tiles are `tile=EPSG3414_iN_jM`). Europe spans dozens of national CRSs. Is CRS a per-region parameter, or is 3414 baked into coordinate handling, the 2km tile-grid origin, and the **bref/anchor coordinate encoding** (sub-F)? Read: `src/cfm/data/sub_a/` (loader/`load_region`, CRS), `src/cfm/data/sub_c/` (tile extraction + grid origin + CRS), `src/cfm/data/sub_d/` (macro/8×8 cell grid, density buckets), `src/cfm/data/sub_f/encoder.py`+`decoder.py` (are `<anchor_*>`/`<direction_*>`/`<magnitude_*>` CRS-relative or grid-relative?), `src/cfm/eval/holdout/paths.py` (`_EPSG_LABEL`, `tile_dirname`), `configs/` (region/CRS/bbox params). Be concrete about which files assume the projection. Verdict: config change vs real rework.
2. **Tile-grid + tokenizer generalization.** Were the CRS-origin-anchored 2km grid (8×8), the **sub-F vocab (1508, Singapore-derived from `configs/tokenizer/vocab_phase1.yaml`)**, the conditioning derivation, or the density buckets tuned to Singapore's extent/density/feature-mix in ways that distort on European cities? Does Europe have feature classes absent in Singapore → would the **append-only vocab discipline** ([[feedback_append_only_vocab_safety]]) force a vocab phase-transition / re-lock?
3. **Extraction cost + storage at scale.** Singapore = 494 tiles, ~8h cold-fetch ([[project_overture_cold_fetch_slow]]), ~171M. The 30M rung alone needs ~27k tiles (~50 cities); full Europe is far more. Realistic wall-clock/region + total, storage, and does any meaningful slice fit before June-11? Is extraction parallelizable across regions on Leonardo, or serial?
4. **Overture uniformity.** Is Overture's schema + coverage uniform across Europe (same layers/quality as Singapore), or per-country gaps/schema differences the pipeline would choke on?

**Required output:** a straight verdict — is "extract Europe" **(a) config-and-run** (pipeline already generalizes), **(b) moderate adaptation** (CRS parameterization + revalidation), or **(c) its own engineering sub-project** (SVY21 baked deep / tokenizer Singapore-tuned)? Days vs weeks. **That decides:** more-data-now (option 1) if days; re-frame the bake-off to **data-efficiency on current data** (option 3) if weeks. Pending this, spec §4 (Topic-2 compute axis) and possibly §11/Topic-1 are a re-open; ALL scored runs + Task 5 are HELD.

## The running diagnostic

- **Job `44006640`** on Leonardo, `scripts/bakeoff_diagnostic.sbatch`, `boost_qos_lprod`, 4×A100, ~90M (d_model 768/12L/12H), value-AGNOSTIC conditioning (Topic-3 stage 2), `--max-steps 110000` (r≈40), `--eval-cells 64 --eval-max-new 2048`, `--emergence-floor 1.96`, `--ckpt-every-n-steps 25000`, `--no-compile`.
- Logs: `logs/bakeoff-diagnostic-90m-44006640.{out,err}`; clean loss in `reports/logs/training-scaffold/version_14/metrics.csv` (use this, NOT the `\r`-laden .err). Report writes to `reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed-scaleup-*.md`.
- A completion monitor (background task `bktf2j1ba`) polls until the job ends, then dumps the report + metrics.

## FOUR reads when it completes (in order)

1. **geometry-r, MEASURED-AT-90M.** From `metrics.csv`: where does **val_loss MINIMIZE** (not where train_loss flattens — train is overfitting to ~0). That val-min step is the data-limited useful r. **Flag the 90M→1B application as an assumption** (§15); the 1.96 floor is scale-invariant, geometry-r is not.
2. **Emergence verdict + building-token presence.** Did generated output clear **1.96 polys/active-cell**? Did generated streams contain building tokens at all (`n_cells_with_building_tokens` — stage-1 truncation discriminator)? Resolves the staged truncation→training→conditioning question.
3. **Measured eval cost.** `cost.eval_seconds` / `eval_node_h_per_cell` — the first real price on the binding pass; sizes the trajectory follow-on + the ladder.
4. **Train/val gap → memorization vs generation (the load-bearing interpretation).** The gap is large and widening (overfitting). So if buildings clear the floor, **do NOT auto-read it as "the architecture learns to generate buildings."** Distinguish memorization from generation: is the generated building **distribution** close to holdout (per the per-feature KS-realism metric, `cfm.eval.realism.ks_distance`), or are buildings merely *present*? The floor is overfit-invariant (real-holdout-derived); the *reading* is not. Overfitting can fake "present" without "plausible."

## Data-size constraint — the 7th catch, VERIFIED (the decisive finding; PI decision pending)

**Verified on Leonardo 2026-06-02:** `sub_f` = `sub_d` = **494 tiles = ALL of validated Singapore** (not a subset of more extracted Singapore), **14.4M tokens total** (29.2k/tile; 362-tile training set ≈ 10.6M). Compute-optimal r=20 ⇒ all-of-Singapore trains only **~720k params**; the ladder needs **27k–896k tiles (54×–1,814× all of Singapore)** for 30M–1B. **Singapore is EXHAUSTED**; a second de-risking region adds ~1 city (negligible vs the need). The D∝N compute-optimal ladder to 1B is INFEASIBLE on de-risking data — this invalidates the **Topic-2 premise**, not just a number (the 7th catch). The diagnostic confirmed it empirically: val_loss minimized very early, then rose monotonically (4.07→10.6→12.4) = data-starved overfitting from near the start.

Three options (PI call — do NOT let an agent pick; needs the data-strategy decision):
1. **MORE DATA (preserves methodology) — but global-scale.** Needs tens (30M) to thousands (1B) of cities, ~8h+/region. A data-pipeline phase, likely beyond June-11. Second region was triple-duty (resolution+generalization+compliance); it does NOT fix training data (one city ≈ negligible vs 27k+ tiles).
2. **CAP THE LADDER at the data-feasible scale.** ~14.4M tokens caps compute-optimal at <1M params (or ~30M if ~50 cities extracted). Extrapolating sub-1M→1B production is hopeless; may not support a production decision.
3. **RE-FRAME the bake-off to the data-limited regime** — compare which architecture generalizes best / overfits least from limited data, NOT compute-optimal scaling curves. A DIFFERENT decision axis than Topic 1 locked (a real re-open).

Original (pre-verification) framing follows:
- **(a) Train to the val-loss minimum (data-limited early-stop), not r≈40.** Compare architectures at the data-limited regime. Simplest; honest about the constraint; changes the "compute axis" (spec §4) from compute-optimal to data-limited.
- **(b) More training data** — the second-region extraction (Sweden/Sri Lanka, the Topic-6 escalation) becomes necessary for DATA, not just KS resolution. Expensive (~8h+/region, its own sub-project).
- **(c) More Singapore tiles** — the 362 were `494 validated − 132 holdout`. Is there more validated Singapore beyond 494? (Check sub-G output; the eval-set/holdout is locked, but more training tiles may exist or be extractable.)
- This is a load-bearing course-correction: **flag to the PI before the ladder**; it likely updates spec §4 (compute axis) and §11 (ladder sizing). PRD §13 anticipated it ("prefer smaller-but-converged over larger-but-undertrained"; data may bind before compute).

## What's DONE + committed (do NOT redo)

CPU-testable batch, all green, on `phase-2-bakeoff`:
- **T1** `eval/emergence.py` (building-token detector by `building` L1-key = 77 ids; holdout-density floor). **T2** `slice_metrics` §2 emergence guard + `eval/realism.py` per-feature KS (hand-rolled, no scipy). **T3** `eval/feature_resolution.py` (per-feature resolution, winner-vs-runner-up 3-tier seam). **T6** `data/training/conditioning.py` value-bearing builder (identity-locked). **T9** `training/deviation_log.py` (fails-to-train valve) + grad-accum. **T10** `training/resume.py` (across-job $WORK resume) + `scripts/bakeoff_run.sbatch`. **T13** `eval/curve.py` (bootstrap-CI fit + §13 tie-break). **T7/T8 CPU** `models/backbone.py` (swappable; mamba/diffusion gated `BackboneNotYetBuilt` behind Task 5) + `models/diffusion/mask.py`.
- **T1.5 (6th catch)** `eval/geometry.py::promote_building_rings` — buildings decode as closed-ring LineString by decoder contract (`decoder.py:145-157`); metrics never promoted → `n_polygons=0` on real data (the DOMINANT cause of the probe's 0, not under-training). Fixed + validated on real Leonardo data: 40,376 building polygons, **7.85/active-cell**, floor **1.96**.
- **T4 wiring** `train_scaffold.py` (emergence verdict + building-token presence + timed eval) + `_build_parser()` + CLI contract guard test. `_param_count` uses `build_backbone` (one source).

## Hard-gate sequence (unchanged)

Diagnostic (T4, RUNNING) → **resolve the data-size constraint (NEW gate)** → **T5 mamba-ssm verify-before-lock** (`scripts/verify_mamba_lock.sbatch` to write; re-lock-ALL if torch changes; `mamba-ssm`/`causal-conv1d` NOT yet installed — pins_freeze has torch 2.5.1+cu121/lightning 2.6.5/pydantic 2.13.4/triton 3.1.0) → backbone GPU builds (T7/T8) → T11 pilot → T12 ladder → T13 decide → T14 report.

## Leonardo access (CINECA)

- SSH master socket: `ssh -S ~/.ssh/cm-leonardo uaslam00@login.leonardo.cineca.it '<cmd>'`. If expired the PI re-opens: `ssh -fN -M -S ~/.ssh/cm-leonardo -o ControlPersist=8h uaslam00@login.leonardo.cineca.it` (2FA).
- Sync: `git bundle create /tmp/x.bundle phase-2-bakeoff --not <leonardo-HEAD>` → `rsync -e "ssh -S ~/.ssh/cm-leonardo" …` → on Leonardo `git fetch /tmp/x.bundle phase-2-bakeoff && git merge --ff-only FETCH_HEAD`. Leonardo CANNOT fetch GitHub.
- Env: `module load python/3.11.7 cuda/12.2`; `source .venv/bin/activate`. Login node has no GPU (CPU torch + `assert_training_env_locked()` pass there).
- Slurm: partition `boost_usr_prod`, account `AIFAC_P02_222`, qos `boost_qos_dbg` (30-min)/`boost_qos_lprod`. 4-GPU = `--nodes=1 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8`; pre-build manifest ONCE in the preamble before `srun`.

## The 6 regime/contract catches (spec §15 meta-pattern)

(1) NLL exact-vs-ELBO across families; (2) emergence-confound (probe under-trained 190×); (3) resolution unit (per-cell 0.076 ≠ per-feature); (4) 6ND across backbones; (5) budget magnitude (80×→7.6×); (6) building-polygon contract (decoder returns building rings as LineString; metric never promoted). **Audit every inherited number AND every consumed contract.** A 7th is now emerging: the **D∝N compute-optimal premise assumes abundant unique data** — false on 362 tiles.

## Authoritative

Spec `docs/superpowers/specs/2026-06-02-phase-2-bakeoff-design.md`; plan `docs/superpowers/plans/2026-06-02-phase-2-bakeoff.md`; predecessor handoff `docs/handoffs/2026-06-02-bakeoff-resume.md`; memory `[[project_bakeoff_cpu_phase_done]]`.
