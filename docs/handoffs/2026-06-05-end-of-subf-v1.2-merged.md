# Handoff — sub-F validator v1.2 is MERGED + PUSHED; next work is CORPUS COMPLETION (2026-06-05)

**Read this first.** The previous handoff (`2026-06-05-subf-v1.2-resume.md`) said "finish the v1.2
fix." That is DONE. This handoff's job is the opposite: make unmistakable that **v1.2 is merged and
pushed**, and that the real next effort is **corpus completion + G4 DoD** — a SEPARATE body of work.

## 0. State — DONE and on origin

- **`main` @ `fda9678`** (a `--no-ff` merge commit), **pushed to `origin/main`**
  (`github.com/Umaraslam66/Bonzai-V1`). Verified: `origin/main` == local `main` == `fda9678`.
- What landed: the **sub-F validator v1.2 fix** (§8.3-termination relax — road-presence-conditioned
  symmetry+coverage) **+ a coherent v1.2 EU corpus** (26 EU cities re-derived in place to v1.2 +
  `_SUCCESS`; singapore/berlin left at v1.1 as Phase-1 artifacts). **364 tests green** on main
  (`pytest tests/data/multiregion/ tests/data/sub_f/`).
- Evidence (all in-tree): close-out `reports/2026-06-05-subf-v1.2-revalidation-closeout.md` +
  `reports/2026-06-05-subf-v1.2-revalidation/` (census, source-trace, byte-identity, re-validation
  logs). Drop evidence: census **anomaly=0 corpus-wide** (teeth-proven non-vacuous) + source-trace
  **`len_in_MISS=0`** on all 14 SYM8 disagreement edges. Byte-identity proven at scale (bruges
  180/180, ljubljana 672/672, turin 696/696 raw-sha256 identical → encoder change is a pure refactor).
- Key commits in main: `87e2a49` fix · `8573ec0` census teeth-proof · `edcb1b8` guarded re-derive
  tool · `d4f1689` close-out+verification · `fda9678` merge.

## 1. THE HARD BOUNDARY — do not misread this

**v1.2 merged ≠ corpus complete.** The corpus is **25/40 processable**. The **G4 three-part DoD is
UNCHECKED**. The merge certified the VALIDATOR FIX + a coherent v1.2 corpus, NOTHING about
completeness or token-budget sufficiency. Do not let "v1.2 merged" read as "ready to train."

## 2. The open work (corpus completion — scope fresh; each framing intact)

- **13 sub-C TIMEOUTS** (amsterdam, budapest, hamburg, helsinki, lisbon, lyon, madrid, paris, rome,
  rotterdam, valencia, vienna, warsaw): big-metro boxes don't finish sub-C inside boost's 8h wall.
  This is a **box-sizing problem, NOT a resubmit** — re-running the same boxes just times out again
  (the prior attempt burned ~850 local-h producing nothing). **Decide tighter boxes vs a
  longer-wall/different partition BEFORE re-running.** These re-runs hit **sub_c (the EXPENSIVE
  extraction layer)** — see §3.
- **almere**: passed sub-F but **failed sub-G** validation; **42.7% of buildings dropped**
  (alpha-drop band). **GUILTY-UNTIL-PROVEN** — trace against the unclipped Overture source to decide
  reclaimed-land-legit vs a real data-loss bug (same third-authority source-trace method that cleared
  the 14 SYM8 edges). Own bucket.
- **welwyn**: fetched (Overture cache present), **never processed**. Just needs a processing run.
- **G4 three-part DoD** (driver `scripts/multiregion/build_g4_rollup.py`): (a) **≥600M validated
  tokens measured directly**, (b) **per-city floor** (no silent dud hiding under the sum), (c)
  **axis-coverage green**. Run it ONCE the corpus is complete. NOTE from G3: EU runs ~56k tok/tile
  (~1.9× Singapore's 29,150) — the per-tile token budget over-yields, so a tile-count target
  under-counts tokens; size by the DIRECT 600M measure, not tile counts.

## 3. STANDING RULE — every destructive re-derive uses the guarded tool

`scripts/multiregion/guarded_rederive.py` (commit `edcb1b8`, 10 tests). It enforces **lockfile**
(`fcntl.flock` — a second invocation refuses) + **atomic temp-swap** (derive into temp; replace live
only on full success; live untouched during the kill-prone derive) + **halt-on-non-identical**
(compare temp vs live before swap; HALT with live untouched unless `--allow-content-change`).

> Use it for ALL destructive re-derives — **especially the 13 sub-C timeout re-runs**, which rewrite
> the EXPENSIVE sub_c layer (not cheaply-regenerable sub-F). This session had **two near-misses**
> doing exactly this with hand-rolled shell (a login-node-watchdog kill mid-`a_coruna` leaving a
> partial city, and a double-`nohup` that would have run two concurrent in-place re-derives of the
> same dirs — saved only by wait-loop timing, NOT a safeguard). No hand-rolled loops against the
> single-copy corpus. Known-issue **#18** is the standing mandate.

  Usage: `python -m scripts.multiregion.guarded_rederive --release 2026-04-15.0 --city <c> [...]`
  (add `--allow-content-change` ONLY for intentional content-changing regen, e.g. the #17 fix).

## 4. Regen-era prerequisites (known_issues — read before any sub_c regen)

- **#16** — the relax's intended recurring drop-guard `assert_lossless_clip` is **TESTED-BUT-UNWIRED**
  (no production caller in any path). Until wired, source-trace + census-anomaly are the ONLY running
  drop checks. Wiring needs sub-C to **persist `len(source ∩ bbox)` per feature at clip time** (a
  sub-C clip-path change + re-run). Do this as part of the sub_c regen.
- **#17** — sub-C `_both_cells_present` records a §8.3 touch-at-boundary AS a crossing (the
  `per_cell_pieces`→crossing path runs BEFORE `apply_sliver_drop` in `geom.py`; the old `:557` cite
  was stale). Spec-violating, TOLERATED under v1.2, MUST fix at next regen. **When fixed, the corpus
  SHIFTS at ~54–65/187 symmetric touch-as-cross edges** (they lose their spurious bref) — NOT
  byte-stable, so re-derive with `--allow-content-change` and re-bless. Not benign — tolerated.

## 5. admin_region HARD GATE still stands (#13/#14)

Before enabling ANY value-bearing conditioning (Task 7 / a bake-off candidate): `admin_region` MUST
be re-derived with a deliberate cross-country granularity choice (#14) and the corpus reopened. EU is
currently all-`None`, SG hardcoded. Do NOT train value-bearing conditioning on existing admin_region.

## 6. Leonardo access + deploy

Repo: `/leonardo_work/AIFAC_P02_222/Bonzai-OSM`. Drive from the Mac via the user-auth SSH
ControlMaster socket (`Host leonardo`, user `uaslam00`). Dies on laptop sleep — re-auth:
`step ssh login 'aslamumar016@gmail.com' --provisioner cineca-hpc` then `ssh -fN leonardo`. Memory
[[reference_leonardo_claude_ssh_socket]]. Deploy code via git bundle (branch is unpushed-from-Mac
historically; main is now on GitHub so `git pull` on Leonardo from origin also works). Heavy CPU work
runs on `lrd_all_serial` (budget-free, 8cpu/30G/4h cap) or PI-authorized `boost_usr_prod` (per-core
billed, explicit `--authorized-boost-override`); NEVER run heavy derives on the login node (the
watchdog kills them — this session's `a_coruna` lesson).

## 7. One-liner to start the next session

> Corpus completion for Phase-2 multiregion: v1.2 is merged+pushed (`main` @ `fda9678`) — now resolve
> the 13 sub-C timeouts (box-sizing, not resubmit), adjudicate almere (sub-G 42.7% drop,
> guilty-until-proven vs source), process welwyn, then run `build_g4_rollup.py` for the G4 three-part
> DoD; EVERY destructive re-derive must use `guarded_rederive.py`. Read
> `docs/handoffs/2026-06-05-end-of-subf-v1.2-merged.md`.
