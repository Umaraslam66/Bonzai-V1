# Handoff — Phase-2 bake-off, diagnostic running (2026-06-02)

**Cold-resume safety net.** Written mid-run so a fresh session can pick up if the socket/session dies. Branch `phase-2-bakeoff` (tip `5182791`), synced to Leonardo (`/leonardo_work/AIFAC_P02_222/Bonzai-OSM`, same tip). Local fast suite **1089 passed**, 1 pre-existing Phase-0 xfail.

## LIVE HEADLINE (preliminary — confirm with the full curve at completion)

**The diagnostic is revealing SEVERE OVERFITTING: data, not compute, is the binding constraint.** At step 74,949/110,000 (**epoch 152**), `train_loss ≈ 0.02` (collapsed from 0.86) while `val_loss = 10.6` (risen from 4.07). The ~362-tile training set is **~8M unique tokens** (~493 steps/epoch × 16,384 tok); r≈40 = ~223 passes → massive memorization. **This challenges the ladder's compute-optimal D∝N premise**: you cannot Chinchilla-train a ≥90M model on 8M unique tokens. The real useful r is the **val-loss minimum (early)**, not r≈20. **Do NOT proceed to the scored ladder before resolving the data-size constraint** (options below).

## The running diagnostic

- **Job `44006640`** on Leonardo, `scripts/bakeoff_diagnostic.sbatch`, `boost_qos_lprod`, 4×A100, ~90M (d_model 768/12L/12H), value-AGNOSTIC conditioning (Topic-3 stage 2), `--max-steps 110000` (r≈40), `--eval-cells 64 --eval-max-new 2048`, `--emergence-floor 1.96`, `--ckpt-every-n-steps 25000`, `--no-compile`.
- Logs: `logs/bakeoff-diagnostic-90m-44006640.{out,err}`; clean loss in `reports/logs/training-scaffold/version_14/metrics.csv` (use this, NOT the `\r`-laden .err). Report writes to `reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed-scaleup-*.md`.
- A completion monitor (background task `bktf2j1ba`) polls until the job ends, then dumps the report + metrics.

## FOUR reads when it completes (in order)

1. **geometry-r, MEASURED-AT-90M.** From `metrics.csv`: where does **val_loss MINIMIZE** (not where train_loss flattens — train is overfitting to ~0). That val-min step is the data-limited useful r. **Flag the 90M→1B application as an assumption** (§15); the 1.96 floor is scale-invariant, geometry-r is not.
2. **Emergence verdict + building-token presence.** Did generated output clear **1.96 polys/active-cell**? Did generated streams contain building tokens at all (`n_cells_with_building_tokens` — stage-1 truncation discriminator)? Resolves the staged truncation→training→conditioning question.
3. **Measured eval cost.** `cost.eval_seconds` / `eval_node_h_per_cell` — the first real price on the binding pass; sizes the trajectory follow-on + the ladder.
4. **Train/val gap → memorization vs generation (the load-bearing interpretation).** The gap is large and widening (overfitting). So if buildings clear the floor, **do NOT auto-read it as "the architecture learns to generate buildings."** Distinguish memorization from generation: is the generated building **distribution** close to holdout (per the per-feature KS-realism metric, `cfm.eval.realism.ks_distance`), or are buildings merely *present*? The floor is overfit-invariant (real-holdout-derived); the *reading* is not. Overfitting can fake "present" without "plausible."

## Data-size constraint — options to weigh (NEW, surfaced by the diagnostic)

The bake-off assumed compute-optimal D∝N (Chinchilla r≈20). With ~8M unique training tokens that overfits even 90M. Before the ladder:
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
