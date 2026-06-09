# Handoff — START of the architecture bake-off (2026-06-09)

You are a fresh, context-free agent starting the **architecture bake-off**. The eval-set-generation
sub-project is **DONE and MERGED**. This file is forward-looking: what's in `main`, the obligations that
**fire the moment a model exists**, and the decisions you must NOT relitigate. (For the full record of how
the eval set was built, read `docs/handoffs/2026-06-09-end-of-eval-set-gen-execution.md`.)

---

## ⛳ STATE

- **eval-set-generation MERGED to `main` at `0833ac7`** (`--no-ff`, local; **not pushed** — push is a
  separate act on Umar's word). In `main` now:
  - **Frozen 4-city EU held-out set** — `data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml`
    (schema 2.0, `manifest_sha256=ae4d5af6…`, write-once, sha verified stored==recomputed) + `_EVAL_SET_LOCKED`.
    glasgow/eisenhuttenstadt/munich/krakow, whole-city, 46.13M held-out / 623.90M train tokens; usable-n
    523/579/156/601 (`reports/2026-06-08-usable-n.yaml`).
  - **Coherence metric** — `src/cfm/eval/holdout/coherence.py` (S1: continuity + giant-component + active-active
    zoning, interior-permutation shuffle-gap, NO density); reference at `reports/2026-06-08-coherence-reference.yaml`
    (n_shuffle=200, both measures + dense-core flags).
  - **Leak guard** — `lineage_audit.py` city-identity guard (whole_city) + `holdout_guard.py` fail-closed
    schema-2.0 backstop (`run_holdout_audit(..., expected_schema_version="2.0")`).
  - **§7 power gate SHAPE** — `resolution.py::assert_coherence_power_sufficient` (consumes resolved_gap + usable_n
    + opaque `model_vs_real_effect`; owns the munich→manchester swap; dormant until first model).

## 1. THREE obligations — now ACTIVE bake-off-setup tasks (they fire the moment a model exists)

**(a) Datamodule re-point + DDP schema flips — REQUIRED before ANY bake-off training run.** Construct the
bake-off `CellDataModule` with `multiregion_holdout_manifest_path("2026-04-15.0")` (default
`expected_holdout_schema="2.0"` backstops a forgotten re-point — fails loud, never silently audits EU against
the SG manifest). When reusing `scripts/train_scaffold.py` / `scripts/ddp_resume_check.py` for EU, flip their
explicit `expected_holdout_schema="1.0"` → `"2.0"` AND re-point the path — else they silently audit EU against
the SG holdout (#16 one layer over). The 3 `"1.0"` sites carry inline comments saying so. Do this **deliberately**;
the backstop is a net, not a substitute.

**(b) DEFINE `model_vs_real_effect` — still OPEN, with a binding anti-leak constraint, and it needs its own teeth.**
The §7 gate consumes this effect at first model; its computation is NOT pinned (`resolution.py:91-95` + spec §7).
**Binding constraint:** whatever you define it as, a model that merely ECHOES the handed tile-mode conditioning
(scoring high *absolute* continuity/giant without generating real structure) MUST FAIL it — the absolute band
ALONE is conditioning-contaminated (the shuffle-gap exists to subtract the echo; munich is the cautionary case —
highest absolute band 0.92/0.98, lowest gap, would rank BEST exactly where it can't discriminate). **munich's
saturated shuffle-gap cannot be munich's reference.** Both measures are recorded so you can choose anti-leak-soundly.
When you define it, write its **own teeth** proving a conditioning-echoing model fails it (red-before/green-after).

**(c) munich→manchester swap reserve + EU-train-split resolved-gap recompute — feed the §7 gate at first model.**
The swap is the §7 POWER reserve (fires only if, at first model, usable_n can't resolve the effect) — distinct from
the metric-validation saturation in §2 below (which did NOT trigger a swap). And `assert_resolution_sufficient`
still reads the **SG** marker's KS fields; the EU-train-split resolved-gap recompute (resolution.py on the TRAIN
split, §5/§7) is a first-model obligation — the EU `_EVAL_SET_LOCKED` intentionally carries no KS number (§2.3).

## 2. LOCKED decisions — do NOT relitigate

- **Held-out set = glasgow/eisenhuttenstadt/munich/krakow** (whole-city, country+CRS-zone in train; a miss must read
  "didn't generalize", never "never saw this style").
- **All-moderate density cut** — scope cut, revisitable only at a phase transition (PI-ratified), not a permanent blind spot.
- **Density-coherence term DROPPED** — per-cell `cell_density_bucket` conditioning → circular. **PAIRED NON-LEAK:**
  `perplexity_gap` treats the same field oppositely and correctly (a use-test) — do NOT "fix" it.
- **Cross-tile seam coherence — NEVER an architecture bar** (rules-based stitcher, identical across architectures);
  one-time validation when sub-E lands, not a per-architecture discriminator.
- **Within-regime ≠ PRD §9.113** — v1's held-out test is necessary-but-not-sufficient for the cross-region claim;
  literal region-holdout is a separate, separately-reported probe.
- **#13 + #22 are ONE bundled HARD GATE** (admin_region all-None + morphology_class/country/climate SG constants):
  fixed in the SAME single corpus re-derive, never separately, before any value-bearing conditioning.
- **munich dense-core saturation (#21)** — munich's held-out tiles are dense-core (mean 47.8 road-edges > 40 = 2/3 of
  the 60-edge interior capacity), so the shuffle-null saturates → tooth-3 0.43. This is a metric-VALIDATION limitation,
  NOT a power failure (the §7 gate doesn't consume the shuffle-gap). munich STAYS; structural exclusion (by edges, not
  name) gates tooth-3 ≥0.70 on the moderate strata only. No swap, no re-freeze.

## 3. Execution discipline (carried forward)

Verified-end-state, never exit codes (re-read the artifact, recompute the sha — 3 prior false-DONEs this project).
Teeth are HALT-gates, not reports. Long unattended bake-off runs are the orchestration soft-spot — guard markers,
double-confirm cluster state. **No merge/push without Umar's explicit word + `--no-ff`.** Never force-push/rewrite main.

## 4. Infra (Leonardo, CINECA, `AIFAC_P02_222`)

- The bake-off is the **~4,800 GPU-h** run on the **boost/GPU partitions** (post allocation top-up); bill per node
  (4×A100), saturate the full node (4-GPU DDP), bf16, checkpoint every 30 min.
- Corpus frozen at `/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed` (DERIVATION 1.2, release `2026-04-15.0`).
- **`eval-set-gen-wt` clone still on disk** (`/leonardo_work/AIFAC_P02_222/eval-set-gen-wt`) — cleanup pending Umar's
  word. Main Leonardo repo pristine on `phase-2-corpus-completion @ 212e3ed`. SSH: user-authed ControlMaster socket
  `Host leonardo` (re-auth on laptop sleep). Ad-hoc python in the clone needs `PYTHONPATH=<clone>/src`.

## ▶️ One-liner to start the bake-off session

> eval-set-generation is MERGED to `main @ 0833ac7` (not pushed): frozen 4-city EU held-out set (manifest sha
> ae4d5af6…), coherence metric (S1 + active-active zoning, shuffle-gap), leak guard (city-identity + schema-2.0
> backstop), §7 power-gate shape — all in main. Three obligations are now ACTIVE bake-off-setup tasks (read §1):
> (a) datamodule re-point to multiregion + flip the 2 DDP scripts' schema 1.0→2.0 BEFORE any run; (b) DEFINE §7
> `model_vs_real_effect`, still OPEN + anti-leak-constrained (echo-model must fail; absolute-band-alone contaminated;
> munich's shuffle-gap saturates), with its own teeth; (c) munich→manchester reserve + EU resolved-gap recompute feed
> the gate at first model. Do NOT relitigate the §2 locked decisions (incl. munich dense-core #21 stay+record). Bake-off
> = ~4,800 GPU-h on Leonardo boost. No merge/push without Umar's word + `--no-ff`. Read this handoff + the execution
> handoff (`2026-06-09-end-of-eval-set-gen-execution.md`) first.
