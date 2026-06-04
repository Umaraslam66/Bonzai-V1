# Handoff — end of Phase-2 orchestrator BUILD (2026-06-03)

**Open here.** The bounded multi-region extract orchestrator is **built and green**. What remains is
the **operational extract (Phase G)** — Leonardo-only and gated. This doc is the source of truth a
cold session resumes from.

---

## UPDATE 2026-06-04 — G2 + G3 DONE; HALTED before G4 (batch-2)

**G2 (canary extract): COMPLETE + clean.** All 5 canary cities fetched on the Leonardo login node
(tmux, ~45min) then processed via `multiregion_process.sbatch` on `lrd_all_serial` and **validated
end-to-end** — each `passed=True groups=0 sanity_floor_violated=False`. Tiles/tokens: prague 253/12.2M,
munich 171/10.1M, barcelona 72/7.24M, milton_keynes 42/1.85M, umea 36/0.81M (**574 tiles, 32.2M tokens**).

**Canary caught a real bug:** committed sbatch asked 32cpu/120G → rejected (`QOSMaxCpuPerUserLimit`).
`lrd_all_serial` QoS caps cpu=8/mem=30800M/4h, and **cpu=8 is per-USER → per-city jobs run SERIAL**.
Fixed to 8cpu/30G/4h + added `test_sbatch_fits_lrd_all_serial_qos` must-distinguish guard (proven: fails
on old 32cpu/120G). **Committed `b98a20b`; deployed to Leonardo (ff from 5bdcf05).**

**Validated baseline = `b98a20b`** (NOT 5bdcf05 — its committed sbatch is the broken one). §5.1
composition check ran at b98a20b: all 5 cities "nothing — up to date" (b98a20b invalidates no stage;
the 5bdcf05-run artifacts are current). Single clean committed-sha baseline confirmed.

**G3 (proceed gate): all three parts satisfied** — `reports/2026-06-04-phase-2-g3-canary-rollup.yaml`
(driver `scripts/multiregion/build_g3_rollup.py`):
1. **Regime gate** — 5/5 validated, **0 data-regime fixes needed**; the one fix (sbatch) has a proven
   must-distinguish guard. Structural axis-coverage gate `ready_for_next_batch=True` (morphology 4 /
   density 3 / geography 5; 0 uncovered).
2. **Sizing gate** — advisory proxy measured: geometry_redundancy 0.516 vs Singapore 0.537
   (rel_sg −0.039, no anomaly); **language anchor DEFERRED** (no language-token corpus pinned; spec §7
   forbids inventing one — advisory, doesn't gate budget). Budget sizes up to **30,900 tiles**,
   r-unresolved flag set (bake-off = sole r authority).
3. **Cost-model gate — KEY FINDING:** EU **~56,060 tokens/tile vs Singapore's pinned 29,150 (~1.9×;
   barcelona dense 100,538 = 3.4×, umea sparse 22,377 = 0.77×)**. Method reproduces Singapore exactly
   (29,201), so the delta is real. **The 29,150/tile assumption does NOT generalize to Europe** — a
   tile-budget corpus over-yields tokens ~2× (safe per §10.1). Clipped fallback-bbox tile counts are
   **lower bounds** (barcelona dense=72 < moderate munich=171); barcelona is an unreliable
   tiles/morphology point. `lrd_all_serial` is 8-CPU serial-per-user → batch-2 throughput must weigh
   `dcgp_usr_prod` (parallel, BILLS) vs free-but-serial.

**State:** local tip `20c79e7` (3 ahead of 5bdcf05: sbatch fix + G3 driver/report). **Still UNMERGED +
UNPUSHED** — merge waits for G4. (PI considering a branch-only push as off-laptop backup — not a merge.)

**NEXT = G4 (PI decision-gated).** Ratify the batch-2 list to fill axis gaps, **factoring the ~2×
tok/tile finding into sizing** (tile-budget vs token-target trade-off is a PI call per §10.1; high
variance 0.77×–3.4× argues against naive token-sizing) and the serial-vs-dcgp throughput trade-off.
Then fetch+process batch-2, merge into the roll-up, assert validated tokens ≥ budget, write the close-out
— **THEN the merge decision.** Deploy/access mechanics: see memory `reference_leonardo_claude_ssh_socket`.

---

## Merge status — READ FIRST (do not misread)

- **`main` has the CRS+pilot FOUNDATION only** (`4c0bf14`, merged + pushed earlier this session with
  explicit authorization). `main` does **NOT** have the orchestrator. (`main == origin/main == 4c0bf14`.)
- **The orchestrator is NOT merged or pushed.** It lives entirely on branch
  **`phase-2-multiregion-extract`** (tip `4864387`), **16 commits ahead of `main`** (spec, plan, A1,
  B1–B2, C1, D1–D2, E1–E3, F1–F3, canary configs, labels manifest) + this handoff.
- **Merge waits until the canary proves the orchestrator end-to-end (true DoD = G4).** Do NOT merge or
  push the orchestrator before then. Rationale: it's green on synthetic fixtures + Singapore Gate-6,
  but its actual job (many heterogeneous cities, multi-CRS) is **unproven** — "passed on synthetic ≠
  safe multi-region." Nothing on `main` needs it yet (Phase G is the only consumer, on this branch).

---

## State — what's DONE

- **Orchestrator build (Phases A–F): complete + green.** 54 multiregion tests pass; full fast suite
  **1098 passed / 0 failed**; ruff clean. Modules under `src/cfm/data/multiregion/`:
  - `stages.py` — 6-stage table (fetch→sub_c→sub_d→sub_e→sub_f→validate) with **traced** source globs
    + the verified `validate` invocation (`scripts/sub_g/validate_phase1_region.py`).
  - `state.py` — sha-stamped completions + **invalidate-on-fix cascade** (real-git-repo tested).
  - `partition.py` — hard **non-boost** assertion (rejects the `boost_*` GPU family).
  - `driver.py` — per-city chain: sha-gated, **returncode-distrusting** (rc 0 + marker, else fail),
    **continue-but-loud** isolation.
  - `selection.py` — single-UTM-zone filter (rejects on FULL bbox, not centroid) + region-config
    writer + `load_canary_manifest`.
  - `rollup.py` — roll-up + **structural** axis-coverage gate (count can't hide a clustered-failure
    uncovered axis).
  - `proxy.py` — **ADVISORY** redundancy proxy (see carry-forwards).
  - CLI `scripts/extract_region_batch.py` (dry-run + non-boost guard) + `scripts/multiregion_process.sbatch`
    (CPU template, guard-tested).
- **In-scope sub_f fix** (`ad64111`): sub_f manifest now carries `region_crs`
  (`SUB_F_ARTIFACT_FORMAT_VERSION` 1.0→1.1; provenance/consistency, **not** bug-prevention — sub_f is
  CRS-agnostic; non-Singapore guard test added).
- **Foundation close-out:** `reports/2026-06-03-phase-2-multiregion-crs-pilot.md` (the CRS+pilot work
  that is on `main`).
- **Canary list RATIFIED + configs committed** (`f96439f`) + **labels manifest** (`4864387`).

## The 5 canary cities (LOCKED, ratified 2026-06-03)

| City | Morphology | Density | `projected_crs` | UTM zone |
|---|---|---|---|---|
| `prague` | medieval-organic | dense-core | `EPSG:25833` | 33 |
| `barcelona` | planned-grid | dense-core | `EPSG:25831` | 31 |
| `milton_keynes` | modernist-sprawl | moderate | `EPSG:25830` | 30 |
| `munich` | mixed | moderate | `EPSG:25832` | 32 |
| `umea` | planned-grid | sparse | `EPSG:25834` | 34 |

- **5 distinct CRS** (zones 30–34, one path each — every projection path exercised once); morphology
  {medieval-organic, planned-grid, modernist-sprawl, mixed} + density {dense-core, moderate, sparse}
  **fully covered**. All verified single-UTM-zone on the full bbox; none cross-border-admin.
- **Region configs** (pipeline input): `configs/data/regions/{prague,barcelona,milton_keynes,munich,umea}.yaml`.
- **Axis labels have a machine-readable home:** `configs/multiregion/canary_v1.yaml`, read via
  `selection.load_canary_manifest`. The Phase-G roll-up builds `CityRecord` axis labels from THIS —
  do NOT re-guess them (re-guessed labels silently break the G3/G4 coverage gate). A test asserts
  manifest `projected_crs` == region config == `single_utm_zone_ok(bbox)` for all 5.

---

## NEXT — Phase G (Leonardo-only, gated)

- **G2 — run the canary.** Fetch the 5 cities on the Leonardo **login node in tmux** (a
  `load_region(name, confirm=True)` loop → populates `data/cache/overture/`; egress is on login), then
  submit `scripts/multiregion_process.sbatch` per city (CPU partition; run_city's fetch cache-hits, no
  egress). The CLI is `scripts/extract_region_batch.py --cities <name> --partition lrd_all_serial`
  (dry-run first to see the plan). Collect `CityResult`s.
  - **Per failed city:** fix the surfaced regime as a **§9 construction-identity guard WITH its
    must-distinguish twin** (the guard fires on the EU regime AND on a synthetic version lacking the
    construction identity). Halt-and-report each regime; do not improvise patches.
- **G3 — three-part proceed gate** (all three real, not "canary green"):
  1. **Regime gate** — every canary city sub_g-validated; every regime fix has a *proven*
     must-distinguish guard.
  2. **Sizing gate** — the **advisory** proxy measured + recorded on the canary corpus (budget ALWAYS
     sizes up + the r-unresolved flag is ALWAYS set; the bake-off is the sole r authority).
  3. **Cost-model gate** — tiles/city + fetch-cost **re-priced per morphology** (Berlin priced only one
     dense-metro point; a sparse/sprawl city may differ sharply) → feeds the batch-2 count.
  - Then the **§5.1 composition check**: the canary is fully green under the **final post-fix shas**
    (canary + batch 2 share one validated sha baseline) before batch 2 launches.
- **G4 — batch 2 + DoD + close-out.** Ratify the batch-2 list (fill axis gaps shown by the canary's
  coverage matrix), fetch+process, merge into the roll-up; assert validated tokens ≥ budget; write the
  `reports/` close-out — **THEN the merge decision.**

---

## Locked carry-forwards

- **Labels are pre-data hypotheses.** morphology/density are guessed on coarse fallback bboxes; the
  precise admin polygon + measured morphology come from Overture downstream. Do NOT litigate
  Munich-as-"mixed" / Umeå-as-"planned-grid" until the data is in. The canary validates the
  orchestrator **mechanically** across CRS/regimes — not the label taxonomy.
- **The proxy is ADVISORY** (spec §7). It records a diagnostic verdict label + `rel_lang`/`rel_sg`, but
  the budget ALWAYS sizes up to `base·(1+Y)` and the r-unresolved flag is ALWAYS set. The **bake-off is
  the sole authority on r** — a data-only proxy does not clear the bar to gate the budget down (the only
  down-gate path is also the only *unrecoverable* under-provision path, per protocol §10.1).
- **Merge waits for G4.** Nothing from `phase-2-multiregion-extract` merges or pushes to `main` until
  the canary proves the orchestrator end-to-end.

## Key references

- Spec: `docs/superpowers/specs/2026-06-03-phase-2-multiregion-extract-orchestrator-design.md`
- Plan: `docs/superpowers/plans/2026-06-03-phase-2-multiregion-extract-orchestrator.md`
- Foundation close-out: `reports/2026-06-03-phase-2-multiregion-crs-pilot.md`
- Canary labels: `configs/multiregion/canary_v1.yaml`
- Branch: `phase-2-multiregion-extract` (tip `4864387`)
