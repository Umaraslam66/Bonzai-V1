# Handoff — training-scaffold GPU phase resume (2026-06-02)

**Resume cold from here.** Foundation + dependency setup + Leonardo bring-up are DONE and verified. What remains is the GPU phase: the model, DDP DataModule, training loop, inference, slice-eval, and the end-to-end run.

Spec: `docs/superpowers/specs/2026-06-01-phase-1-training-scaffold-design.md`. Plan: `docs/superpowers/plans/2026-06-01-phase-1-training-scaffold.md` (12 tasks + amended Task 0). Branch `phase-1-training-scaffold`; **origin + Leonardo both at `3d0be05`** (re-sync Leonardo to the latest origin tip on resume — see "Leonardo access").

## Done + verified (do NOT redo)

- **Task 1 (#4) is SETTLED via Branch A (clean)** — verified on the live sub-F path (cascade #7 buckets unknowns to `<unknown_KEY>`); `known_issues #4` annotated-closed; Phase-0 `encode.py` knowingly-unfixed-but-non-training-reachable. **Do not re-litigate.** This already discharged the precondition that gated the tier-1 locks.
- **Tasks 2,3,4,11** — tier-1 shard schema; one-source conditioning (`_derive_tile_conditioning` identity-locked) + append-only id-block; `build_training_shards` (set-by-ID from frozen manifest, stamped lineage, byte-deterministic, count==362==marker `training_residual`); resolution seam (marker-sourced, fail-closed, two failure kinds).
- **Task 0 (deps)** — `torch==2.5.1+cu121`, `lightning==2.6.5`, `pydantic==2.13.4`. A100-verified (`matmul_ok`, driver 535, CUDA 12.2). **Both enforcement adds landed and confirmed:** (1) run-start version assert `cfm.training.env_lock.assert_training_env_locked()` (passes on the real Leonardo venv); (2) full resolved lockfiles committed — `configs/training/env-lock-leonardo-cu121.txt` (pip freeze, 63 deps) + `uv.lock` (Mac dev lock). The pins are the bake-off **comparability lock** (identical across all 12 runs).
- **Task 5 core** — `holdout_guard.py` (pure, Lightning-free): 4 regime tests + must-pass twins (F1/F2 inject, F4 no-synthesis, clean non-zero count, stamped-integrity).

## Remaining — GPU phase (Tasks 6,7,8,9,10,12)

**Validation split (do not silently downgrade):**
- **Login CPU-torch (fast, no Slurm):** Task 7 (model forward/loss), Task 9 (decode via sealed sub-F decoder), Task 10 core (slice metrics; the `holdout-not-monitored` test needs Lightning).
- **Slurm GPU jobs (`--gres=gpu:4`, DDP):** Task 6 (`CellDataModule`: setup-halt-before-batch-0 on ALL ranks, val-split disjoint from the 132 holdout, seeded `DistributedSampler`, bit-identical 4→4 resume), Task 8 (loop + 30-min checkpoint + bit-identical resume), Task 12 (`fast_dev_run` devices=4 → short run → `reports/`).
- **DDP tests MUST assert `world_size == 4`** — a 4→4 resume or all-ranks-halt test that silently ran on 1 rank is the vacuous pass we've avoided everywhere. Assert the rank count.

## Leonardo access (CINECA; `docs/LEONARDO_REFERENCE.md`)

- Drive from local Bash via the shared SSH master socket: `ssh -S ~/.ssh/cm-leonardo uaslam00@login.leonardo.cineca.it '<cmd>'`. If expired, the user re-opens it: `! ssh -fN -M -S ~/.ssh/cm-leonardo -o ControlPersist=8h uaslam00@login.leonardo.cineca.it` (2FA).
- **Leonardo cannot fetch GitHub** (no creds). Sync code: `git bundle create /tmp/x.bundle <base>..phase-1-training-scaffold` → `rsync -e "ssh -o ControlPath=~/.ssh/cm-leonardo"` → on Leonardo `git fetch $WORK/x.bundle phase-1-training-scaffold && git merge --ff-only FETCH_HEAD`.
- Env: `module load python/3.11.7 cuda/12.2`; `source $WORK/Bonzai-OSM/.venv/bin/activate`. Login node has NO GPU.
- Slurm: partition `boost_usr_prod`, account `AIFAC_P02_222`, qos `boost_qos_dbg` (quick checks) / `boost_qos_lprod` / `boost_qos_bprod`. Run long work in `tmux`. Reuse `extract_singapore.sbatch` (and `~/extract_singapore.sbatch.bak`) as a job template.

## Budget — TIME-SENSITIVE (post-slice decision to surface)

~**1000 node-hours expiring 2026-06-06** (user-stated 2026-06-02; reconcile with `project_allocation` memory: AIFAC_P02_222, 40k core-h, ends 2026-06-11). The thin slice itself is minimal (de-risking, not compute-bound — a tiny model on ~362 tiles' cells, <1 GPU-h). **But once the slice closes clean, the larger runs that need the pre-6th window must be front-loaded, not deferred past expiry.** Surface this as an explicit decision the moment the slice is green — do not let the window lapse silently.
