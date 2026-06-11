# Task 24b mini-spec — per-cell continuous character carrier (DRAFT for PI review)

**Status: APPROVED 2026-06-11 (PI + Umar), all four knobs as recommended:** (A) fixed
log-transforms, no normalizer artifact; (B) scheme tag `"value-char-v1"` flips AT 24b;
(C) per-cell `CellPayload.character_stats` field — the deviation from §8's literal
`macro_tokens` wording explicitly blessed (tile-level tokens are wrong-shape twice: V1
re-collapse + V2/V4 re-bucket); (D) road = median length only. Carried obligation: the
presence-flag teeth must PROVE zero-vs-absent distinctness red-on-divergence (the
zero-means-absent aliasing class). The wholesale shard rebuild is a SEPARATE gated word
AFTER wiring+teeth are green. Parent: spec §8 + revised plan Task 24b. Evidence base:
`reports/2026-06-11-residual-character-recon.md`.

---

## 0. One deviation from §8's wording, surfaced first

§8 says the carrier rides "the provisioned-but-empty `macro_tokens` channel." Gate-2 read of
the live schema (`src/cfm/data/training/shard_schema.py`): `TrainingShard.macro_tokens` is
**tile-level `tuple[int, ...]`** — integer macro-plan tokens provisioned for bake-off
candidates 1/2. That is the wrong shape twice over: tile-level re-collapses per-cell signal
(the V1 lesson) and integer tokens re-bucket it (the V2/V4 lesson — bucketing is the proven
losing move; continuity is the point). **Proposal: the carrier lands as a NEW per-cell field
`CellPayload.character_stats` (append to the tier-1 schema), and `macro_tokens` stays
untouched for candidates 1/2.** Schema-append is the cheap layer (schema-vs-data asymmetry);
the expensive layer (wholesale shard rebuild) is already mandated by the re-scope. This is a
deliberate, PI-visible deviation from §4.4/§8's literal channel name, not a silent one.

## 1. The carrier (data side)

- **`CellPayload.character_stats: tuple[float, ...]`** — 7 floats per cell, derived at
  shard-build time from sub-C artifacts already on disk (the V4-reader walk; NO sub-C regen):
  1. `log10(median building footprint area m²)` — V4's stat, now continuous
  2. `log10(building-size IQR m² + 1)` — spread (recon: 70% of residual is shape)
  3. `log10(p90/p50 building-size ratio)` — tail weight
  4. `log10(building count + 1)` — density texture beyond the bucket
  5. `log10(median road segment length m + 1)` — the road fine-location thread (54%)
  6. `buildings_present` ∈ {0,1} — explicit absence flag (a zero stat must never alias
     "no buildings"; structural-boundary discipline)
  7. `roads_present` ∈ {0,1}
  Absent layer ⇒ its stats are 0.0 AND its flag is 0 (the flag disambiguates).
- **Fixed transforms, NO data-derived normalization constants** (recommended): log10 +
  documented clip ranges. Rationale: z-scoring needs train-union statistics frozen as another
  write-once sha artifact + a gated derivation run; fixed transforms are deterministic,
  artifact-free, and sufficient for a linear projection to absorb scale. [PI knob A: fixed
  transforms (recommended) vs frozen z-score artifact.]
- **Derivation source:** building polygons + road linestrings from sub-C `features.parquet`
  per tile (4-column projected read, BUILDING/ROAD filter before WKB decode — the proven V4
  reader pattern; per-region UTM CRS ⇒ meters). External-source-of-truth tooth: the
  shard-build derivation is pinned EQUAL to the recon script's independent computation on a
  shared fixture (`investigate_residual_character._read_cell_building_areas` + stats).
- **Wholesale shard rebuild** [LEONARDO — GATED, CPU]: all 38+4 cities' shards regenerate at
  one uniform defect level; no partial fix. Reader-side: shard reader refuses a shard missing
  `character_stats` once the schema lands (version-skew kills, fail-closed).

## 2. The model side (the continuous prefix position)

Today (`backbone.py`): `ids = [9 conditioning prefix ids | cell tokens]` through ONE embedding
table; `n_cond = conditioning_id_span() = 576` rows; positions `= max_len +
CONDITIONING_PREFIX_LEN (9)`.

- **One new prefix POSITION (the 10th), continuous:** its input embedding is
  `Linear(7 → d_model)` of `character_stats`, not a table lookup. Token-id axis UNCHANGED
  (span stays 576 — no new vocabulary ids). **Axis separation named to avoid the n_cond
  conflation (carried correction #9):** `CONDITIONING_PREFIX_LEN` stays `len(_CONDITIONING_FIELDS)
  = 9` (id positions); new constant `CHARACTER_PREFIX_POSITIONS = 1`; positional capacity
  becomes `max_len + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS`. Reverse-lock
  sweep #3 for the 9/2057/capacity-41 pins just written by 24a (rides the 24b commit,
  named).
- **Ablation `"no_character"` becomes functional:** zero the stats vector AND the presence
  flags before projection (the all-zeros input is the learned-nothing signal; bucket-0
  analog). The 24a `NotImplementedError` flips to implemented — a named reverse-lock on the
  24a test that pins the raise.
- **Loss masking:** prefix_len grows by 1 for every example; the existing mask logic keys on
  `prefix_len` per batch — verify, don't assume (Gate-2 at dispatch: `micro_ar.training_loss`).
- **`conditioning_scheme` tag:** bump to `"value-char-v1"` when 24b lands — ONE bump covering
  24a+24b (version-fold: no blessed checkpoint exists; the 24a reviewer's flagged question
  resolves here). The scheme tag then genuinely separates pre/post-carrier checkpoints.
  [PI knob B: tag string.]

## 3. Teeth (named now, proven at execution)

1. Round-trip: shard write→read preserves `character_stats` exactly; reader refuses absent
   field (version-skew fail-closed).
2. Derivation parity vs the recon's independent computation on a shared on-disk fixture
   (external source of truth — not self-consistency).
3. Presence-flag discipline: a zero-building cell and a cell whose stats happen to log-fold
   to 0.0 are distinguishable (flag bit) — regime-distinguishing pair.
4. Ablation: `"no_character"` zeroes ONLY the continuous position (9 id positions
   bit-identical to full); `"no_city"` continues to zero ONLY slot 8; the two compose.
5. Constant-column guard extends to character stats (constant across ≥2 cities → loud;
   spec §4.5 surviving guard); all-zeros-region guard (a region whose every cell has zero
   flags → loud).
6. Capacity: generation-at-exact-positional-capacity test re-derives from the new live
   constants (the production-build path, never a hand-built model).
7. TDD red-first throughout; two-stage review; mutation evidence on the new guards
   (the house bar set by 24a's reviews).

## 4. Execution shape + cost

- CPU-only wiring + tests locally; the shard REBUILD is the only Leonardo step
  [GATED, CPU, ~hours over 42 cities]; no GPU anywhere in 24b.
- Touches: `shard_schema.py` (append field), `build_shards.py` (derivation + guards),
  `datamodule.py` (thread float tensor through flatten/collate), `conditioning.py`
  (ablation flip + constants), `backbone.py`/`micro_ar.py` (projection + masking),
  `train_scaffold.py` (plumbing), `ScaffoldConfig` (nothing new — `conditioning_ablation`
  exists), tests across all.
- After 24b: Task 25 (floor artifact — unchanged by the carrier; the floor measures REAL
  data only) then Task 26.

## 5. PI knobs in this mini-spec

A. Normalization: fixed log10+clip transforms (recommended) vs frozen z-score artifact.
B. Scheme tag string at 24b: `"value-char-v1"` (recommended).
C. The §0 channel deviation itself (per-cell `character_stats` field instead of the
   tile-level `macro_tokens` tuple) — needs explicit blessing since §8 names macro_tokens.
D. Road stat: median segment length only (recommended, recon-backed) vs adding a road-count
   channel (not recon-backed; YAGNI default is NO).
