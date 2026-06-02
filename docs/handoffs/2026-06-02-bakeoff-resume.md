# Handoff — Phase-2 bake-off resume (2026-06-02)

**Resume cold from here.** The Phase-1 training scaffold (first end-to-end loop) and a 300M de-risking probe are DONE, merged to `main`, and pushed. The next sub-project is the **architecture bake-off** (4 candidates × 3 scales). It has NOT been started — it needs its own brainstorm → spec → plan.

**Repo state:** `main` at `39b2ce2` (origin + local + Leonardo all in sync; Leonardo's checkout is on branch `phase-1-scaleup-probe` whose tip == `main`). Suite: **1020 passed**, 35 slow deselected, 1 pre-existing Phase-0 xfail. 120 tree-wide ruff errors are PRE-EXISTING in untouched `sub_c/d/f` modules (present on the pre-work commit) — not in scope.

## Done + verified (do NOT redo)

- **Training scaffold (12 tasks).** `src/cfm/models/micro_ar.py` (decoder-only AR, prefix+pad-masked loss, head = sub-F range 1508, n_cond=8 field-slot conditioning), `src/cfm/data/training/datamodule.py` (`CellDataModule`: fail-closed holdout audit halting before batch 0, tile-level seeded split disjoint from holdout, padded collate, seeded `DistributedSampler`), `src/cfm/training/{config,lit_module,train}.py` (`ScaffoldLit` + `build_trainer`: DDP/bf16/30-min ckpt/`WorldSizeGuard`/CSVLogger), `src/cfm/inference/generate.py` (generate + sealed-decoder decode), `src/cfm/eval/slice_metrics.py` (decodability/OGC-valid/right-angle + `n_polygons`/`n_corners`; bref via the shared D3 instrument, reported-not-gated), `scripts/train_scaffold.py` (run_smoke/run_short).
- **Validated on Leonardo 4×A100 (non-vacuous, `world_size==4` asserted):** all-ranks audit-halt 4/4; 4→4 resume functionally identical (max|diff| 1.19e-07 = float32 ε; exact bit-identity is unachievable under DDP NCCL — see [[feedback_ddp_determinism_gotchas]]); short run (job 43947093) decodability 0.922 / OGC-valid 0.983. Report: `reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed.md`.
- **300M de-risking probe (job 43957846).** Report: `reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed-scaleup-308M.md`. Reusable bake-off plumbing landed: `build_trainer(max_time=)`, `run_short` per-step-node-h cost reporting, CLI `--d-model/--n-layers/--n-heads/--batch-size/--max-time/--no-compile/--eval-cells/--eval-max-new`, `scripts/scaffold_scaleup_probe.sbatch`.

## The bake-off — open the brainstorm from the EVAL-COST REFRAME (PI directive)

PRD §6 ladder = **30M / 100M / 300M** × 4 architectures (AR transformer, hierarchical transformer, Mamba/SSM via `mamba-ssm`, discrete diffusion). PRD §6.4 budgets it at ~1,500 GPU-h (~375 node-h). The probe relocated the binding constraint — lead the scope with this, not with training compute:

1. **(HEADLINE) Training is NOT the constraint — eval is.** 12 runs of *training* ≈ **4.7 node-h** vs the 375 envelope (~80× under). The probe answered the original worry (do 12 large runs fit the window?) — yes, enormously. So the bake-off must size + budget the **EVAL pass** (generate → decode → score across 12 runs × the holdout) as the binding cost. Autoregressive eval generation is per-token forwards and is slow at scale (64×512 @ 300M overran a 30-min slot). Keep `--eval-cells`/`--eval-max-new` small at scale; budget eval separately from training.
2. **Building-emergence floor (probe-grounded).** 300M / 2000 steps / value-AGNOSTIC conditioning emitted roads but **0 polygons** (no buildings; `n_polygons=0` disambiguated the `right_angle_rate=0.0`). Each bake-off run needs enough steps AND value-bearing conditioning for buildings to appear — else the comparison is roads-only and the geometry-fidelity metrics can't discriminate architectures.
3. **Slice carry-forward — unscored becomes scored.** The named-but-UNSCORED follow-ons (tile cell-to-cell coherence, value-bearing conditioning, conditioning-compliance scoring, the learned right-angle bar) must become EXPLICIT unscored→scored items in the bake-off spec, not silently assumed working.
4. **Trigger-3 goes LIVE.** ≥2 architectures ⇒ the resolution seam fires: wire `cfm.eval.resolution.assert_resolution_sufficient` against real architecture-to-architecture KS gaps; if a needed gap comes in finer than **0.076** (the frozen resolved gap), surface the **second-region extraction** escalation (single-region hard floor = 0.049). The deferred eval-harness depth (KS/Wasserstein vs model output, tokenizer-on-model R2, sim-viability, model-scoring orchestration) is owed here too — real-side baselines + holdout + guards already exist in `src/cfm/eval/holdout/`.

## Comparability lock (carry into every bake-off run)

torch 2.5.1+cu121 / lightning 2.6.5 / pydantic 2.13.4, asserted at GPU entrypoints via `cfm.training.env_lock.assert_training_env_locked()`. Identical across all 12 runs (that's what makes the scaling curves comparable). DDP gotchas codified in [[feedback_ddp_determinism_gotchas]]: seed before model init; `save_checkpoint` is collective (all ranks); keep `WorldSizeGuard`.

## Leonardo access (CINECA; `docs/LEONARDO_REFERENCE.md`)

- SSH master socket: `ssh -S ~/.ssh/cm-leonardo uaslam00@login.leonardo.cineca.it '<cmd>'`. If expired the user re-opens it: `! ssh -fN -M -S ~/.ssh/cm-leonardo -o ControlPersist=8h uaslam00@login.leonardo.cineca.it` (2FA).
- Leonardo CANNOT fetch GitHub. Sync code via `git bundle create /tmp/x.bundle <base>..<branch>` → rsync over the socket → `git fetch /tmp/x.bundle <branch> && git merge --ff-only FETCH_HEAD`. Job-written reports are untracked → `rm` them before an ff-merge that adds the tracked copy.
- Env: `module load python/3.11.7 cuda/12.2`; `source $WORK/Bonzai-OSM/.venv/bin/activate`. Login node has NO GPU (but CPU-torch imports + `assert_training_env_locked()` pass there — the lock is version-tag based).
- Slurm: partition `boost_usr_prod`, account `AIFAC_P02_222`, qos `boost_qos_dbg` (quick, 30-min cap) / `boost_qos_lprod` / `boost_qos_bprod`. 4-GPU DDP sbatch = `--nodes=1 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8`; pre-build the manifest ONCE in the preamble (single process) before `srun` (no rank write-race). Run long work in `tmux`.

## Budget (live `saldo`, reconciled)

Deadline **2026-06-11** (the prior handoff's "June 6" was WRONG — live accounting + `[[project_allocation]]` both say June-11). 902/40,000 core-h spent (2.3%) → ~**1,222 node-h** remaining; June monthly cap 13,043 core-h (~408 node-h), 0 spent. Training is cheap (probe: ~0.39 node-h per 10k-step 300M run) so the bake-off can afford long/large runs — the constraints to manage are the EVAL pass and (later) the ~2,000 GPU-h production run.

## Process discipline

Read `docs/protocols/sub-project-planning-protocol-v3.md` end-to-end FIRST; open the bake-off brainstorm with the §-by-§ topic→gate preamble; gate topics one per message ([[feedback_consult_planning_protocol_before_brainstorm]], [[feedback_brainstorm_gate_discipline]]). Branch `phase-2-bakeoff-...` off main, commit task-by-task, local-first (merge to main + push at sub-project end once suite-green + report-written; PR optional). Authoritative: `PRD.md` §6 (candidates) + §6.4 (budget); [[project_training_scaffold_handoff]] (full GPU-phase + probe status).
