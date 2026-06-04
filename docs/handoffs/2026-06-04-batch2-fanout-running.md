# Handoff — batch-2 fanout RUNNING on boost (cold-start, 2026-06-04 ~22:00 local)

**Open here.** A new agent session resumes from this. The PI shut the laptop after launching
the batch-2 corpus build; the work runs autonomously on Leonardo. Your job: re-establish access,
check progress, collect the corpus, run the **G4 three-part DoD gate**, report, then await the
merge decision. **Nothing is merged/pushed; the PI approves G4 before any merge.**

---

## 0. TL;DR — where we are

Phase-2 multi-region. **G2 (5-city canary) DONE + validated. G3 (proceed gate) DONE.** Now building
the **40-city diversity corpus** (5 canary + **35 batch-2**) so the model isn't Singapore-overfit.
At handoff, the 35 batch-2 cities were **fetching (4 parallel login-node tmux lanes) + processing on
boost** (auto-fanout watcher submits each to `boost_usr_prod` as its fetch lands). **Expected complete
~5–7 am local 2026-06-05.** By morning it should be done or nearly so → collect + run **G4 DoD**.

- Local branch `phase-2-multiregion-extract`, tip **`e5fd0fe`** (run `git log --oneline -5`). Leonardo
  is ff'd to the same. **Unmerged + unpushed.** Validated-corpus consistency baseline = the HEAD the
  cities processed at (`e5fd0fe`); canary was processed at `5bdcf05` and is composition-valid forward
  (no stage source-glob changed since — verify with the dry-run in §3).
- Spec: `docs/superpowers/specs/2026-06-03-phase-2-multiregion-extract-orchestrator-design.md`
  (see the 2026-06-04 §7 UPDATE: diversity sizing, 600M DoD, admin_region HARD GATE).
- Prior handoff: `docs/handoffs/2026-06-03-end-of-phase-2-build.md` (G2/G3 detail).

## 1. Re-establish Leonardo access (the laptop sleeping killed the SSH socket)

Claude runs on the **Mac**, not Leonardo. Drive Leonardo via a shared SSH ControlMaster socket the
PI authenticates. After a shutdown the socket + cert are gone — ask the PI to run **two** commands
(they handle the CINECA 2FA), then paste the output:

```
step ssh login 'aslamumar016@gmail.com' --provisioner cineca-hpc
ssh -fN leonardo && ssh leonardo 'echo SOCKET_UP; squeue -u $USER -o "%.10i %.14P %.12j %.2t %.9M"'
```

The `leonardo` host alias is in `~/.ssh/config` (User `uaslam00`, ControlMaster, persist 12h). Once
`SOCKET_UP` prints, every `ssh leonardo '<cmd>'` reuses it with no further auth. **Slurm jobs + tmux
on Leonardo kept running regardless of the laptop** — only your observation paused. Memory:
[[reference_leonardo_claude_ssh_socket]]. **Do NOT poll in tight blocking loops — the PI dislikes it;
check, report, move on.**

## 2. Leonardo facts

- User `uaslam00`, login `login.leonardo.cineca.it`, account `AIFAC_P02_222`.
- Repo on Leonardo: **`/leonardo_work/AIFAC_P02_222/Bonzai-OSM`** (it's the `$WORK` project dir).
- Deploy code Mac→Leonardo via **git bundle** (branch is unpushed): `git bundle create /tmp/x.bundle
  <oldsha>..phase-2-multiregion-extract` → `scp` → `ssh leonardo 'cd <repo> && git pull --ff-only
  /path/x.bundle phase-2-multiregion-extract'`. Don't push to origin.
- Release pin: `2026-04-15.0`. Processed artifacts: `data/processed/sub_{c,d,e,f,g}/2026-04-15.0/<city>/`.
  Fetch cache: `data/cache/overture/2026-04-15.0/<city>/` (cache keys on **name, not bbox** — beware).

## 3. Check pipeline progress (first thing after the socket is up)

```bash
ssh leonardo 'bash -s' <<'EOF'
REPO=/leonardo_work/AIFAC_P02_222/Bonzai-OSM; cd $REPO; REL=2026-04-15.0
echo "=== tmux (fetch lanes b2f1-4, boost watcher b2boost) ==="; tmux ls 2>/dev/null
echo "=== fetch lanes done-count ==="
for L in 1 2 3 4; do echo "lane$L: $(grep -ac 'DONE rc=0' logs/b2-fetch-lane$L.log) done; $(grep -c LANE${L}_COMPLETE logs/b2-fetch-lane$L.log) complete"; done
echo "=== boost watcher: submitted / all-submitted? ==="; grep -c SUBMITTED logs/b2-boost-watcher.log; grep ALL_SUBMITTED logs/b2-boost-watcher.log
echo "=== boost jobs still running ==="; squeue -u $USER -o "%.10i %.14P %.12j %.2t %.9M" | head -40
echo "=== _PHASE1_VALIDATED markers (NOTE: includes 7 pre-existing — 5 canary +"
echo "    singapore (Phase-1) + berlin (pilot); neither singapore nor berlin is in"
echo "    the 40-city G4 corpus. The 35 batch-2 names are listed below.) ==="
ls data/processed/sub_g/$REL/*/_PHASE1_VALIDATED 2>/dev/null | wc -l
echo "=== any FAILED process jobs? ==="; sacct -X --starttime=2026-06-04 -n -o JobID,JobName%14,State | grep -iE "mr-proc-boost" | grep -viE "COMPLETED|RUNNING|PENDING" | head
EOF
```

The 35 batch-2 cities: vienna lyon bologna tallinn krakow edinburgh bruges ljubljana turin valencia
mannheim glasgow lodz helsinki a_coruna karlsruhe almere rotterdam cergy tychy eisenhuttenstadt espoo
welwyn paris madrid rome amsterdam hamburg warsaw budapest lisbon copenhagen manchester malmo toledo.
(toledo's first smoke cache was cleaned; it re-does in the fanout.) **If fetch lanes died** (login
node reboot), re-launch them: `/leonardo_work/AIFAC_P02_222/g_fetch.sh` is the lane script; the boost
watcher is `/leonardo_work/AIFAC_P02_222/b2_boost_watcher.sh` (auto-submits unsubmitted+fetched cities,
idempotent via `logs/b2-boost-submitted.txt`). **If a city's process FAILED** (continue-but-loud):
read its `logs/mr-proc-boost-<jid>.out`; a surfaced data-regime needs a §9 construction-identity guard
+ must-distinguish twin (halt-and-report, don't improvise) — but most likely it just needs a resubmit.

## 4. Budget division (local-h = core-hours; boost bills PER-CORE: billing=cpu, confirmed)

- Allocation `AIFAC_P02_222`: **40,000 local-h**, window to **2026-06-11**, **tops up ~then**.
- Consumed pre-batch-2: **955 local-h** (training-scaffold). Remaining at handoff: **~39,045**.
- **Batch-2 process on boost ≈ ~1,020 local-h** (35 cities × 8 cores × wall; per-core `billing=8`
  VERIFIED on a real boost job 2026-06-04 — NOT whole-node 32). ~2.6% of remaining. Check actual spend:
  `saldo -b`.
- Bake-off training needs **~4,800 GPU-h ≈ ~38,400 local-h** — but that's a **POST-top-up** draw, not
  due now. Plus a SMALL post-corpus overfit-recheck (a few hundred local-h). Both pre-top-up costs fit
  in 39,045 comfortably.
- **HALT RULE (PI policy):** training has priority. If the top-up slips, **stop submitting batch-2**
  (per-city resumable) — do not let prep eat hours the validation run needs.
- **Boost is a PI-AUTHORIZED EXCEPTION for batch-2 prep ONLY** (2026-06-04). The guard
  `assert_cpu_partition` stays **default-deny**; boost is allowed only via the explicit
  `--authorized-boost-override` flag (logged loudly), which `multiregion_process_boost.sbatch` passes.
  **Do NOT weaken/delete the guard. Do NOT use boost for anything else.** The canary/default sbatch
  stays `lrd_all_serial`.

## 5. THE G4 DoD GATE — run this when the corpus is in (anti-traps baked in)

Driver: **`scripts/multiregion/build_g4_rollup.py`** (committed; reads canary_v1 + batch2_v1; read-only).
```
ssh leonardo 'cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM && ./.venv/bin/python scripts/multiregion/build_g4_rollup.py'
```
It enforces the PI-locked **three-part DoD** (a threshold alone is a trap):
- **(a)** `total_validated_tokens >= 600M` — measured DIRECTLY (NOT tiles×29,150; that Singapore
  constant is ~½ the EU rate, superseded — see spec §7 UPDATE).
- **(b)** PER-CITY FLOOR — every validated city contributes non-trivial tiles/tokens (floor = umea
  canary low, 36 tiles / 0.8M tokens). **A city that "passed" with a near-empty box (empty-sea/rural
  or silent clip; per-city counts are known-soft, fallback bbox / known_issues #15) is a SILENT DUD
  and must surface, not hide under the sum.**
- **(c)** axis-coverage matrix green (morphology/density/geography, 0 uncovered).
- AND **groups=0 is necessary NOT sufficient** — it prints peer-median outlier **sanity flags** (a city
  whose tiles/tokens are wildly off its morphology/density peers, even if groups=0).

**Report the PER-CITY TABLE the driver prints, not just the total.** Investigate any `<<BELOW-FLOOR`,
`<<NOT-VALIDATED`, or `⚠` sanity flag before declaring DoD pass. Also run the **§5.1 composition check**
to confirm canary+batch-2 share one validated baseline:
`./.venv/bin/python scripts/extract_region_batch.py --cities <all 40> --partition lrd_all_serial --dry-run`
→ every city must say "nothing — up to date".

## 6. Sequence for the new agent

1. Get the socket up (§1). 2. Check progress (§3). 3. If still running, let it finish (don't tight-poll;
report status + an ETA). 4. Handle any FAILED/below-floor/flagged city (§3, §5). 5. When all 40 validate,
run `build_g4_rollup.py` (§5) + the composition dry-run. 6. **Bring the PI: the per-city table + the
three-part DoD verdict + sanity flags + a close-out** (`reports/` summary). 7. **Then the merge decision
— PI approves G4 first.** On approval, the merge is `phase-2-multiregion-extract` → `main` (local merge,
per the project's local-first flow; confirm with PI).

## 7. Standing carry-forwards / do-not-trip

- **admin_region HARD GATE** (known_issues #13/#14, spec §7): `admin_region` is `None` on every EU tile
  (SG-hardcoded). INERT today (value-agnostic 8-slot conditioning), but a systematic SG-vs-EU confound
  the instant value-bearing conditioning is enabled. **Before ANY value-bearing conditioning (Task 7 /
  bake-off): re-derive admin_region + reopen the corpus.** Do NOT train value-bearing conditioning on it.
- **Polygon extent** (known_issues #15): tiles use the fallback bbox, not the real admin polygon —
  over-includes fringe; fine for a diversity corpus; per-city counts are soft (hence the §5 per-city
  floor). Deferred.
- Boxes are GENEROUS (over-include; le_havre/vallingby were dropped as UTM-zone straddlers,
  eisenhuttenstadt backfills modernist×z33). 40-city corpus covers all 4 morphologies / 3 densities /
  7 UTM zones (29–35), 18 countries. Labels are pre-data hypotheses (PI-ratified) — don't over-defend one.
- Memory to read: [[project_multiregion_g2_canary_done]], [[reference_leonardo_claude_ssh_socket]],
  [[feedback_tool_output_silence_is_not_pass]], [[feedback_structural_exclusion_not_magnitude]].

## 8. Key commits (since baseline 76b068d)

`cd14ac7` admin_region/polygon defer + hard gate · `ce25e2b`/`91938db`/`4846789` batch-2 configs
(generous boxes, generator `scripts/multiregion/build_batch2_configs.py`) · `e5fd0fe` boost override
(guard intact) + `multiregion_process_boost.sbatch` + this G4 driver/handoff. **Run `git log --oneline`
to see the exact tip.**
