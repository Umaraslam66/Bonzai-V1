# Phase 1 sub-D — macro plan derivation proposal artifacts

This directory holds the reviewer-facing artifacts that go through **Gate 2**
before sub-D's macro vocab is locked into
`configs/macro_plan/v1/macro_plan_vocab.yaml`.

## The 4+1 artifact layout

When `scripts/analyse_macro_plan_frequencies.py --proposal-only` runs against
a real sub-C region, this directory receives **four namespace files** plus
**one index file**:

| File | Purpose | Reviewer-edited? |
|---|---|---|
| `zoning_analysis.yaml` | Feature-class distribution + candidate cut strategies for the zoning namespace. | No (content-pinned). |
| `cell_density_analysis.yaml` | Building-footprint-ratio distribution + candidate bucket strategies. | No (content-pinned). |
| `tile_population_density_analysis.yaml` | Distributions and candidate strategies for each candidate proxy (mean / area-weighted / median / p75 of building footprint ratio). | No (content-pinned). |
| `road_skeleton_analysis.yaml` | Active-edge road-crossing-count distribution + candidate bucket strategies. | No (content-pinned). |
| `macro_vocab_proposal.yaml` | **Index file.** Carries `status`, per-namespace `locked_buckets`, `locked_proxy` for tile_population_density, `namespace_files` (filename + sha256 pins), `per_tile_evidence`, `zoning_orthogonality`, `input_digests`, and `selected_layer3_tiles`. | Yes — reviewer hand-edits `locked_buckets` / `locked_proxy` here. |

Namespace files are **content-pinned** by their sha256 in the index's
`namespace_files` array. Any edit to a namespace file invalidates that pin
and the validator notices. The reviewer therefore only edits the index file
at Gate 2; the four namespace files are immutable post-write.

The default mode (no `--proposal-only`) writes a single
`frequency_analysis.yaml` consolidated file for ad-hoc inspection. It is
**not** the Gate 2 artifact — only the 4+1 layout is.

## Gate 2 review process

1. **Open `macro_vocab_proposal.yaml` first** — this is the index, the
   entry point.
2. **Drill into namespace files** for `candidate_strategies` detail. Each
   candidate carries its full bucket definition (`kept_tokens` /
   `merged_tokens` for zoning; `bucket_boundaries` for density and
   tile_population_density; `bucket_lower_bounds` for road skeleton), plus
   `categories`, `coverage`, and `marginal_cost` relative to the prior
   strategy.
3. **Decide which cut to lock for each namespace.** The default
   `locked_buckets[namespace]` in the index is pre-filled to the most-
   granular candidate. To lock a different cut, **edit
   `locked_buckets[namespace]` in `macro_vocab_proposal.yaml`** (NOT in
   the namespace file). For `tile_population_density` also choose a
   `locked_proxy` from the candidate proxies the analysis emitted.
   `git diff` on the index makes the choice auditable.
4. **Cross-check the Layer-3 subset.** `selected_layer3_tiles[*].rationale`
   should mention at least one of: zoning, density, road skeleton, scope,
   coastal/inland/riverside. The subset is up to 12 tiles by default
   (`--max-subset-tiles` overrides). Tiles with `active_cell_count == 0`
   are filtered out by the eligibility predicate before any dimension
   ranking applies — they carry no derivation evidence regardless of how a
   meta dimension might rank them.
5. **Verify input digests** in the index's `input_digests` match the
   sub-C release you intend to lock against. Sub-D records bytes-sha over
   the sub-C files it read; if these change, re-derive before promoting.
6. **Inspect `candidate_strategies` marginal-cost monotonicity** in each
   namespace file. Non-decreasing marginal costs are the typical heavy-
   tail pattern but not universal — a non-monotonic series is a signal to
   investigate, not a hard failure. Sub-D does not enforce monotonicity in
   `validate_frequency_analysis`.

When the proposal is approved, run
`scripts/promote_macro_vocab.py --proposal macro_vocab_proposal.yaml --output configs/macro_plan/v1/macro_plan_vocab.yaml`
(Task 8). The promote script flips `status: proposal` → `status: locked`
on the **index file** and writes the locked artifact; it is the only
sanctioned edit between the reviewed proposal and the locked vocab. The
byte-identity-modulo-status-marker test verifies no other edits leaked in.

## Gate 2 cannot close without real Singapore sub-C output

Synthetic-data fixtures exercise the analysis machinery in unit tests, but
the vocab cuts the reviewer locks must be derived from *real* sub-C output
under `data/processed/sub_c/2026-04-15.0/singapore`. There is no
synthetic-data shortcut for vocab approval: if real sub-C output is
unavailable, halt Gate 2 until it is generated.
