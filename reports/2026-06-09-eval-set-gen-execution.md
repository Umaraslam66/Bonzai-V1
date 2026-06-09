# Eval-set-generation (multi-region EU) — execution summary (2026-06-09)

**Sub-project:** the held-out evaluation set + macro-plan-coherence metric + de-Singapore
generalizations for the Phase-2 multi-region EU corpus, feeding the architecture bake-off.
**Branch:** `phase-2-eval-set-gen` (29 commits off `main`, tip `4cb74dd`) — final review **READY TO
MERGE**, not merged (gated on PI word + `--no-ff`). **Config:** corpus release `2026-04-15.0`,
DERIVATION 1.2, 42-city / 670M-token EU corpus. **Spec:** `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md`.

## What was done

12 tasks, subagent-driven (fresh implementer + spec-compliance review + code-quality review per task;
PI as chat reviewer). De-Singapore plumbing (T1–T3) → write-once multi-region manifest (T4–T6) →
holdout-leak guard + schema-2.0 backstop (T7–T8.5) → macro-plan-coherence metric + teeth + reference
(T9–T11) → first-model power gate (T12). Code + synthetic tests local; manifest build, usable-n, and
the coherence reference measured on Leonardo (verified-end-state, transferred byte-exact, committed).

## Metrics (committed artifacts)

- **Held-out set** (frozen, `manifest_sha256=ae4d5af6…`): glasgow / eisenhuttenstadt / munich / krakow,
  whole-city, 46,130,102 held-out tokens / 623,900,790 train. **usable-n:** 523 / 579 / 156 / 601.
- **Coherence reference** (n_shuffle=200, `reports/2026-06-08-coherence-reference.yaml`), per stratum
  recording BOTH the absolute band and the shuffle-gap:

  | stratum | mean road-edges | dense-core | tooth-3 (real-vs-permuted) | abs continuity / giant |
  |---|---|---|---|---|
  | eisenhuttenstadt | 24.7 | no | 0.945 | 0.752 / 0.912 |
  | glasgow | 29.2 | no | 0.811 | 0.769 / 0.923 |
  | krakow | 36.3 | no | 0.879 | 0.816 / 0.961 |
  | munich | 47.8 | **yes** | **0.429** (reported, not gated) | 0.915 / 0.982 |

## What was decided (rationale)

- **Schema constants independent by construction** (SG `1.0` / multiregion `2.0`) — bumping a shared
  constant would have broken the frozen SG manifest's sha (§2.3). Caught pre-build.
- **Schema-2.0 leak-guard backstop default `"2.0"`**, with `"1.0"` reachable ONLY at 3 commented legacy-SG
  sites — an opt-in backstop only guards when you remember; the default-2.0 catches the forgotten EU re-point
  (the #16 failure the task closes).
- **munich STAYS, no swap** (load-bearing, PI ruling): munich's tooth-3 0.43 is **saturation** (dense-core
  #21, 47.8 road-edges > the 40 = 2/3-of-60-capacity threshold), not weakness — and the §7 gate consumes
  resolution's KS number + an opaque first-model effect, NOT the shuffle-gap. So it's a metric-VALIDATION
  limitation, handled by **structural exclusion** (by mean-road-edges, not city name), with the full 0.43
  recorded. Swapping to manchester (itself dense at 40 edges) would cost a write-once re-freeze + z-diversity
  for a cosmetic number. Spec §7 amended to correct the "pre-model trigger CLEARED" claim (aggregate-hides-subsets).
- **`model_vs_real_effect` left OPEN** with a binding anti-leak constraint — defining the first-model gate on the
  absolute band alone would reintroduce conditioning-echo contamination (a model can echo the handed tile-mode
  skeleton to score high absolute coherence). Both measures recorded so first-model can choose anti-leak-soundly.

## Tested vs not

- **Tested (164 passed, local + synthetic):** all term/assertion units; HALT-gates (leak-guard 4-case, coherence
  3-way fragmentation non-redundancy, tooth-3 separation) — each mutation-/red-before-proven non-vacuous; the
  schema-2.0 backstop bidirectionally against the real frozen manifests; write-once freeze orchestration.
- **Measured on Leonardo (verified-end-state):** usable-n, the coherence reference. **Deferred to first-model**
  (4 `@slow` tests + the dormant obligations): the EU-train-split resolved-gap recompute, `model_vs_real_effect`
  definition + its anti-leak teeth, the munich→manchester power reserve, the bake-off datamodule re-point.

## Next

Merge on PI word (`--no-ff`), then the bake-off. The 3 dormant first-model obligations are consolidated in
`docs/handoffs/2026-06-09-end-of-eval-set-gen-execution.md` §4 — read before wiring the bake-off.
