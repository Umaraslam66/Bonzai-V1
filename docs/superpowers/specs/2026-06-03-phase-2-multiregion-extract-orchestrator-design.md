# Phase-2 bounded multi-region extract orchestrator — design

**Date:** 2026-06-03
**Branch:** `phase-2-multiregion-extract` (off `main` @ `4c0bf14`, which now contains the CRS-parameterization
+ Berlin-pilot foundation).
**Status:** design locked via brainstorm (2026-06-03), pending plan.
**Foundation:** `reports/2026-06-03-phase-2-multiregion-crs-pilot.md` (CRS code + pilot, byte-identity proven).
**Strategy decision:** Option 1 (build orchestrator + run the bounded extract), 2026-06-03. The extract is the
Phase-4 **production corpus**; the compute-optimal bake-off falls out of it for free — that dual purpose is why
this is worth ~2–3 weeks, not a marginal bake-off upgrade.

---

## 1. Purpose & Definition of Done

Build a **testable Python orchestrator** and use it to extract a **diversity-spanning, validated multi-region
corpus** sized for a 30M-ceiling compute-optimal ladder at r=20.

**Done =**
- Orchestrator built (the state machine below), with its full test suite green.
- Cities extracted to a **~20,600-tile budget** (30M ceiling × r=20 = 600M tokens ÷ 29,150 tokens/tile measured).
- **Each city sub_g-validated** (cross-artifact validator green).
- A **roll-up manifest** recording, per city: identity, axis labels, region CRS, tile count, fetch cost, stage
  shas, release, validation status, token count — plus corpus totals and an **axis-coverage matrix**.
- Corpus token-count **confirmed ≥ 30M-ceiling need at r=20**.
- A **data-only redundancy proxy** measured and reported (see §7) as (a) an early r-ballpark signal and (b) a
  guard on the extract's own sizing.

**Explicitly NOT in this sub-project** (carried to the bake-off):
- The **compute-optimal r** (training-measured; the redundancy proxy here is *not* it).
- All **scored bake-off training runs**.

**Sizing conservatism (protocol §10.1):** size at r=20 because a corpus can be *shrunk* but not *grown* without
re-fetching. Over-provision is recoverable; under-provision is not.

---

## 2. The per-city pipeline (data flow)

Each city passes through five stages, each gated on the previous stage's `_SUCCESS` (a stage writes `_SUCCESS`
only after its own cross-tile validator passes; downstream refuses to start without it). The orchestrator reuses
the existing stage scripts as **subprocess calls** (the pattern `sub_g/pipeline.py` already uses):

| # | Stage | Invocation (existing script) | Where | Output dir | Marker |
|---|---|---|---|---|---|
| 1a | fetch | `load_region(region, confirm=True)` (`cfm.data.overture`) | **login/tmux (egress)** | `data/cache/overture/<rel>/<region>/` | `manifest.yaml` (cache) |
| 1b | sub_c | `scripts/extract_tiles.py --region R [--release REL] [--pool-size N]` | **Slurm CPU (cache-hit, no egress)** | `data/processed/sub_c/<rel>/<region>/` | `_SUCCESS` + `manifest.yaml` (`region_crs`) |
| 2 | sub_d | `scripts/derive_macro_plan.py --region R --release REL --sub-c-dir … --output-dir … --macro-vocab … --commit-sha SHA` | Slurm CPU | `data/processed/sub_d/<rel>/<region>/` | `_SUCCESS` |
| 3 | sub_e | `scripts/derive_boundary_contracts.py --release REL --region R --sub-c-region-dir … --sub-d-region-dir … --output-region-dir …` | Slurm CPU | `…/sub_e/<rel>/<region>/` | `_SUCCESS` |
| 4 | sub_f | `scripts/sub_f/derive.py --release REL --region R --sub-c-region-dir … --sub-d-region-dir … --sub-e-region-dir … --output-region-dir …` | Slurm CPU | `…/sub_f/<rel>/<region>/` | `_SUCCESS` |
| 5 | sub_g validate | `cfm.data.sub_g.cli:validate_main([--region R --release REL --sub-c-region-dir … --sub-d-region-dir … --sub-e-region-dir … --sub-f-region-dir … --output-dir …])` | Slurm CPU | `…/sub_g/<rel>/<region>/` | `_PHASE1_VALIDATED` |

Notes from the foundation map:
- **Stage 1 splits across the egress boundary.** `extract_tiles.py` *internally* does both fetch and sub_c, but
  fetch needs S3 egress and sub_c is CPU-only. So the orchestrator runs **1a fetch on login/tmux first**
  (`load_region(region, confirm=True)` populates `data/cache/overture/`), then **1b `extract_tiles.py` on a
  Slurm CPU node**, where `load_region` **cache-hits** (no egress) and only the sub_c processing runs. This is
  exactly the pilot's two-step split (`berlin_pilot.sh` step 1 fetch + `berlin_extract.sbatch` step 2 process).
- **`--commit-sha` is required only on sub_d** (others capture/default it). The orchestrator always passes
  `git rev-parse HEAD`. This is "work-item #1".
- `region_crs` threads cleanly sub_c→sub_d→sub_e via manifests; tile labels (`tile=EPSG25833_*`) are
  deterministic from it. **sub_f's manifest lacks `region_crs`** — fixed in-scope here (§8).
- All stages after 1a are CPU-only and read cache/disk — they carry the **non-boost assertion** (§3.2).

---

## 3. Orchestrator architecture (the state machine)

A **thin Python driver**, because the orchestrator *is* a state machine (per-city × per-stage `_SUCCESS`,
idempotent resume, invalidate-on-fix, roll-up) and a state machine wants testable code, not bash.

**Module layout (proposed; plan refines):**
- `src/cfm/data/multiregion/` — the testable core:
  - `state.py` — per-city/per-stage state model + `_SUCCESS`/sha bookkeeping + invalidate-on-fix logic.
  - `stages.py` — the stage table (§2): for each stage, its invocation builder and its **source-path globs**
    (for invalidate-on-fix, §3.2).
  - `rollup.py` — roll-up manifest read/write + axis-coverage matrix + fail-loud gate.
  - `proxy.py` — the redundancy proxy (§7).
  - `selection.py` — candidate-city generation spanning the diversity axes (§4); the *named list* is a plan
    artifact the PI ratifies.
- `scripts/extract_region_batch.py` — thin CLI over the core (run a batch of cities, resume, report).
- Slurm submission templates for the CPU process chain (one per-city process job; fetch handled separately).

### 3.1 Compute topology
- **FETCH** on a **Leonardo login node, wrapped in tmux** (egress proven in the pilot; ~13 min/city with the
  `COUNT(*)` optimization; modest concurrency, bandwidth-bound). Populates `data/cache/overture/`.
- **PROCESS** (stages 2–5; stage 1's sub_c step is CPU too) on a **CPU Slurm partition** (`dcgp_usr_prod` or
  budget-free `lrd_all_serial`), reading cache — **never `boost_usr_prod`** (which bills a 4×A100 node for
  CPU-only work and would silently eat the training GPU-budget).

### 3.2 Pin 1 — hard non-boost assertion
The process-submit path **asserts** the target partition is not `boost_usr_prod` and **fails loud** if
misconfigured. Not a convention or a doc note — a runtime assertion, because a copy-pasted sbatch header six
cities in would otherwise quietly burn the training budget.

### 3.3 Pin 2 — sha-based invalidate-on-fix (+ mandatory test)
Each city/stage's completion is recorded with the **commit-sha it was produced under** (this subsumes
work-item #1's `--commit-sha` threading). On any re-run:
- For each stage, the orchestrator checks whether that stage's **source paths** changed between the recorded
  sha and current `HEAD` (`git diff --quiet <recorded_sha> HEAD -- <stage source globs>`).
- If yes, that stage's `_SUCCESS` is **invalidated** and the stage **and everything downstream** re-runs;
  upstream stages whose code is unchanged do **not** re-run.
- Stage→source-path mapping is **conservative**: a stage's globs include its own module/script **and shared
  modules it depends on** (e.g. `sub_c/coords.py`, common io). **Over-inclusion is safe** (an unnecessary
  re-run); **under-inclusion is the bug** (stale post-fix artifacts) — so bias toward over-inclusion.

**Mandatory test (protocol §2 threshold-pairing analogue):** change a file in stage N's source list → assert
stage N **and downstream** are invalidated and re-run, and that **upstream stages are not**. Without this test,
"idempotent resume" can silently serve stale post-fix artifacts — the exact failure mode byte-determinism
guards against.

### 3.4 Pin 3 — per-city failure isolation: continue-but-loud
- A city hitting a new regime is marked **`failed-needs-attention`** in the roll-up and the **batch continues**
  (one sprawl city's regime must not block 43 good cities mid-batch).
- A failed city is **never silently dropped** from the corpus accounting. A corpus quietly missing the 6 sparse
  cities that all hit one regime is a **silent diversity hole**, and diversity is the whole point (§4).
- The **proceed-to-batch-2 gate** (§5) requires the roll-up to have **zero unaddressed failures**.
- Principle: *isolation that continues, not isolation that hides.*

---

## 4. City selection

**Sizing basis = tile budget**, not a fixed city count. Target ≈ **20,600 tiles** (30M × r=20 ÷ 29,150
tokens/tile). The "~44 cities" figure is only what that budget buys *if* cities are Berlin/Singapore-class dense
(~465–494 tiles each); smaller cities raise the count. City count **falls out** of the budget. The budget
**flexes up** if the redundancy proxy (§7) signals r>20.

**Selection criterion = diversity-first, by explicit named axes** (named *before* any list is generated, so
coverage is auditable-by-construction, not a post-hoc justification):
- **morphology**: medieval-organic / planned-grid / modernist-sprawl / mixed
- **density**: dense-core / moderate / sparse
- **geography**: regional spread (for CRS-zone variety) + Overture-coverage variation

**Two hard filters:**
1. City fits within a **single UTM zone** (no zone-straddle distortion; the CRS picks one zone from the centroid).
2. **Defer cross-border-admin cities** (admin polygon spanning a national border) as a named, later regime.

**List materialization:** `selection.py` generates a candidate list from the **Overture divisions** theme that
**spans** the axes; **the PI ratifies the named list in the plan** (auditable, not guessed). The roll-up reports
the achieved axis coverage.

**Budgeted validation tail (not smuggled):** diversity ⇒ more new regimes hitting sub_g per city (a medieval
core, a planned grid, a sprawl city each stress different parts of the geometry tokenizer). Expect **~2–4 regime
fixes across the first diverse (canary) batch, declining as the morphology space fills**. Each is handled with
the **§9 construction-identity guard discipline** (a structural build-fact exclusion + a regime-distinguishing
twin), **never a one-off patch**.

---

## 5. Execution staging: canary → remainder

**Batch 1 (canary) ≈ 5 cities**, chosen to **maximally span the morphology axes** (one medieval-organic, one
planned-grid, one modernist-sprawl, one sparse, +1 mixed) — because regimes are *caused* by morphological
variety, a max-spread canary surfaces the most regimes in the fewest cities.

**Three-part proceed-to-batch-2 gate** (all three must be real, not "canary is green"):
1. **Regime gate.** Every canary city sub_g-validated; each surfaced regime fixed as a **§9 construction-identity
   guard with its must-distinguish twin** — the guard fires on the European regime **and** still fires on a
   synthetic version that lacks the construction identity (the Berlin validator-hardcode lesson: a fix that only
   makes the real city pass is structurally blind). The bar is "the guards added during canary are *proven* to
   catch the genuine defect," not "canary passed."
2. **Sizing gate.** The **redundancy proxy measured on the canary corpus**, and the tile budget re-confirmed (or
   sized up) **before batch 2's fetch commits**. Batch 2's city count is **not finalized** until the canary proxy
   lands; "~44" stays provisional through the canary.
3. **Cost-model gate.** The canary **re-prices** extraction across morphologies — Berlin gave one dense-metro
   point (465 tiles / 13 min); a sparse/sprawl city may differ sharply. Measured tiles/city and fetch-cost
   **across the morphology range** feed the batch-2 count.

**Batch 2** fills the *re-confirmed* tile budget; validation is near-automated (the morphology space is mostly
covered by the canary's guards).

### 5.1 Composition: invalidate-on-fix (§3.3) × staging
When the canary surfaces a regime and a stage (say sub_e) is fixed, invalidate-on-fix correctly re-runs every
**canary** city's sub_e-and-downstream. Batch 2 isn't fetched yet, so batch-2 cities are built **fresh under the
post-fix sha** and never need invalidation. **The gate must therefore require that the canary is fully green
under the *final post-fix code state*** — i.e. batch 2 launches only after the canary re-run-after-fixes is
green under the same shas batch 2 will run, so **canary and batch 2 share one validated sha baseline**.
Otherwise batch 2 would run under code the canary never validated.

---

## 6. Roll-up manifest

The reproducibility record (config + commit + data snapshot, per CLAUDE.md). One entry **per city**:
- `name`, axis labels (`morphology`/`density`/`geography`), `region_crs` + UTM zone, `tile_count`,
  `fetch_seconds`, **per-stage commit-sha**, `release`, `validation_status`
  (`validated` / `failed`+regime / `pending`), `token_count`.

**Corpus totals:** total tiles, total tokens, **axis-coverage matrix** (the load-bearing field — it *proves*
diversity-by-construction and closes the §4 loop, rather than asserting it), the **redundancy-proxy result**
(§7), and **tile-budget vs achieved**.

A failed city appears with its regime, loud (§3.4) — never absent.

---

## 7. Redundancy proxy (data-only r signal + sizing guard)

**Framing (deliberate):** a **relative** comparison, *not* an absolute entropy→r map (no established theory maps
token entropy to compute-optimal r; inventing one just relocates the borrowed-constant error into a second
made-up number). Measure the geometry token stream's redundancy (compression ratio + per-token/n-gram entropy)
and **compare** it to (a) a language-token baseline (the r=20 anchor) and (b) the Singapore corpus.

**Pre-committed decision rule (stated before measuring, so the result can't be read to fit the city count we
already have):**
- If geometry redundancy is **within X% of the language baseline** → treat **r=20 as confirmed**; the ~20,600-tile
  budget stands.
- If geometry is **materially *less* redundant** than the language baseline (→ more tokens/param, r>20) → **size
  up** the tile budget by Y (add cities while it's cheap — mid-extraction adds are cheap; growing a frozen corpus
  later is not).
- *(X and Y are set as concrete numbers in the plan; the rule's shape is locked here.)*

**Limit (stated up front):** the proxy is **directional**. An **ambiguous** result (close to the language
baseline) is **not** read as "r=20 confirmed." The safe default is **conservative: size at r=20 AND raise an
explicit `r-unresolved-until-bakeoff` flag**, so the true compute-optimal r is resolved by the bake-off's
training measurement — never by an over-read of an ambiguous proxy. Ambiguous ⇒ conservative sizing + explicit
unresolved flag, not false confirmation.

---

## 8. sub_f `region_crs` gap — fixed in-scope

sub_f's manifest currently omits `region_crs` (Berlin's sub_g still passed, so it's non-blocking *today*). We
**fix it in-scope** rather than carry it: "non-blocking because one city passed" is the **regime-blindness
pattern** — one morphology/zone passing does not prove a `region_crs`-dependent sub_f path is harmless across
all zones; a carried gap could quietly enable a future-zone bug. The fix is small and multi-region means sub_f
*should* know its region's CRS. **Add it with a test that exercises a non-Singapore `region_crs`** (the same
guard shape as the Berlin validator-hardcode regression).

---

## 9. Testing (protocol v3)

- **Unit:** invalidate-on-fix (§3.3 mandatory test — sha bump on a stage's source → downstream invalidated +
  re-runs, upstream not); non-boost partition assertion fails loud (§3.2); roll-up fail-loud on a city failure
  (§3.4 — a failed city is present-and-loud, never dropped); tile-budget arithmetic; proxy decision-rule
  thresholds.
- **Integration:** reuse **Berlin as the real end-to-end fixture** for a full-chain pass through the orchestrator
  (the pilot already proved the chain; this proves the *orchestrator* drives it).
- **Gate 6 (external-source-of-truth):** cross-reference the orchestrator's model of each stage's
  `_SUCCESS`/manifest contract against the **stage module's actual writes** (do not infer the contract from
  field names — the §3-§4 protocol discipline). Where the orchestrator's `stages.py` encodes a stage's output
  dir / marker / args, a test asserts those match the stage script's real behavior.
- **sub_f `region_crs`:** the non-Singapore-CRS test from §8.

---

## 10. Out of scope (and where it goes)

- Compute-optimal **r measurement** and all **scored bake-off runs** → bake-off sub-project.
- **dcgp compute-node S3 egress** verification → only matters past ~148 cities; deferred unless the extract
  grows toward the 100M ceiling.
- **Full EU vocab frequency re-derivation** → Phase-4 (the locked Singapore vocab transfers; BP4
  `<unknown_*>` handles EU-frequent tags gracefully; pilot showed Berlin's unknown-rate is *lower* than
  Singapore's).
- **Extending the ceiling to 100M/300M** → a measured-numbers decision after the 30M slice + bake-off's r.

## 11. Open items for the plan (not load-bearing; resolved at plan-write)

- Concrete X / Y constants for the proxy decision rule (§7).
- The exact stage→source-path glob mapping for invalidate-on-fix (§3.3), erring toward over-inclusion.
- The named canary city list spanning the axes (§4/§5) — generated by `selection.py`, **PI-ratified**.
- Slurm submission ergonomics (array vs per-city job; resume bookkeeping location).
