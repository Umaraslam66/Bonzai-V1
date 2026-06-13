# GPU-WAIT CPU DRAIN — COMPLETE. Boot here on renewal day. (2026-06-12)

**STATE:** main @ this handoff commit (on drain tip `477ff50`); **pushed —
origin == local == Leonardo**. Suite **1763 passed / 2 skipped / 1 xfailed**,
ruff zero findings, all 472 files format-clean. **EuroHPC grant APPROVED
2026-06-12; allocation lands ~2026-06-15 (2–3 days out).** Old allocation
`AIFAC_P02_222` end-dated 2026-06-11 at 9.9% consumed; GPU rejects until the
renewal, `lrd_all_serial` accepts real jobs; $WORK + $SCRATCH read-write.
**NO MODEL SCORED YET** — the eval instrument is proven; every scored number is
still in the future. That asymmetry is the whole point.

## What landed (W1–W8, per-item --no-ff merges, each reviewed by Umar)

1. **W1 — the 13,312 budget lock EXECUTED** (`7e8a559`):
   `DEFAULT_MAX_CELL_TOKENS = ScaffoldConfig.max_len = 13_312`;
   `--scored-run` entrypoint gate (`assert_scored_commensurate`: max_len ==
   lock AND eval_max_new >= it; wired in `bakeoff_run.sbatch`); recorded
   tail-drop at 13,312 (job 46009513: union drop 44/1,046,162 = 0.0042%,
   exactly the 5 memo-predicted max-tail cities, worst valencia 5x under the
   0.005 contract). The sub-F alpha-drop 5760 is a DIFFERENT lock (padded
   budget 6016-256) and was deliberately untouched.
2. **W4 — mamba-lock CPU half RESOLVES** (`085e302`): candidate pin
   torch 2.5.1+cu121 UNCHANGED / triton 3.1.0 / causal-conv1d 1.6.2.post1 /
   mamba-ssm 2.3.1. THREE GPU-half preconditions in
   `reports/2026-06-12-mamba-candidate-pin.md` — (1) gcc/12.2.0 +
   CC/CXX/CUDAHOSTCXX at build; (2) constraints file pinning torch+triton on
   every install (unconstrained pip SILENTLY upgraded torch to 2.12/cu130 —
   caught by the probe's torch_matches_lock tooth); (3) gcc-12 libstdc++
   (GLIBCXX_3.4.29) visible at import. Probe venv kept at
   `/leonardo_work/AIFAC_P02_222/envs/mamba-cpu-probe` (built wheels cached).
3. **W2 — `locked_yaml.py` extracted** (`1d94eee`): the shared freeze grammar
   (sha-excluding-itself / stamp-and-seal / generic verified read), three
   instances delegated with behavior-preservation proven on the SEALED floor
   artifact itself (95abb88b… recomputes through both paths; precedence pins
   observed green pre-refactor; red-on-divergence mutation-proven).
4. **W3 — the shard cache, BUILT AND SEALED** (`117a5b8`): per-city parquet +
   sealed manifest (4th locked_yaml instance) + `_SHARD_CACHE_VALID`, on
   $WORK at `data/processed/training_cache/2026-04-15.0/` (761 MB, 38 cities,
   22,019 tiles, 1,409,216 cells). **PAYOFF: ~47-min walk -> 6.9-min verified
   read per job start** (~2.2 GPU-h idle A100 reclaimed per start). Tier (b')
   read verification MEASURED (tier (a) 26.8 min rejected; stat-all 59 s +
   seeded 8/city re-hash). `iter_verified_shard_cache` STREAMS per city (the
   union-resident dict peaks >25 GB — would have OOMed 4-rank training);
   datamodule consumes cache AND walk through one streaming loop; stale cache
   HALTS (ShardCacheStale naming the component), NO fallback to the walk.
   Use: `--shard-cache data/processed/training_cache` on train_scaffold.
5. **W5/W6/W7 — tinies** (`75784df`, 3 commits): `KS_C_ALPHA_05` one-sourced
   (anchors: `_gap_from_cells(2)==1.358`, `ks_two_sample_floor(0.08)==577`);
   shared-extraction halt says "manifest tiles"; `has_outbound_bref` public.
6. **W8 — lint zero** (`477ff50`): 15 B905 individually verdicted (13
   strict=True, 2 strict=False — the pairwise zip(xs, xs[1:]) idiom);
   UP042 noqa'd (StrEnum would change sealed sub-D str() contract);
   the deliberate `PARQUET_WRITE_KWARGS` re-export restored after --fix
   dropped it (suite caught it).

## GPU landmines the drain defused on CPU (the pattern that earned it)

DropRateExceeded would have refused the first scored run at 5760 (W1);
pip's silent torch upgrade would have invalidated the 3-architecture
comparison (W4); the union-resident cache read would have OOMed the first
4-rank training job (W3). Each cost minutes of serial CPU instead of scarce
GPU debugging time.

## THE PARKED SEQUENCE — in order, each on Umar's explicit per-step word

**(a) Step 18.5 resume proof** — kill->resubmit short job; the FIRST
post-renewal GPU job (additionally gated on T0 closure).
**(b) mamba-lock GPU half** — fwd/bwd kernel numerics on A100 = the actual
verify-before-lock verdict; carry the THREE preconditions above VERBATIM (or the
silent-torch-upgrade / build-fail recurs); env_lock.py pin extension lands ONLY
at that verdict; mamba-hybrid stays behind `BackboneNotYetBuilt` until it
passes. Re-lock-all contingency stands.
**(c) The scored bake-off** — the first scored numbers, the answer to whether
the character bet generalizes. Via `scripts/run_bakeoff_decision.py` against the
frozen conditioning-floor artifact (`reports/conditioning_floor/2026-04-15.0/`,
sha-verified reads). Scored runs: `--scored-run` + `--shard-cache` +
eval_max_new >= 13,312.

## Carried backlog (small, open, not silent)

- Cache-read deserialization (~6.9 min is CellPayload reconstruction of 1.4M
  cells) — possible future optimization, deliberately not gold-plated.
- The §9.x content-anchored-cites sweep (old backlog, still open).
- Confirm CINECA data-retention dates with support (saldo exposes none;
  policy suggests ~6 months $WORK retention ≈ Dec 2026).

## Discipline carried

Verified-end-state by recomputation; stop-before-commit at gates; teeth on
every gate; halt-on-defect; ALL CPU extractions as lrd_all_serial; no
GPU/scored runs/push/merge without Umar's word + --no-ff; main is shared —
NEVER force-push.
