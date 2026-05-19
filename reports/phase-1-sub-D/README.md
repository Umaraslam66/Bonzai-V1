# Phase 1 sub-D — macro plan derivation proposal artifacts

This directory holds the reviewer-facing artifacts that go through **Gate 2**
before sub-D's macro vocab is locked into
`configs/macro_plan/v1/macro_plan_vocab.yaml`.

## Expected artifacts

When `scripts/analyse_macro_plan_frequencies.py --proposal-only` runs against
a real sub-C region, this directory receives:

- `macro_vocab_proposal.yaml` — primary reviewer document. Top-level keys:
  - `status: proposal` (Task 8's promote-script flips this to `locked`).
  - `analysis_version`, `derivation_version`, `tile_count`, `input_digests`.
  - `per_tile_evidence` — small per-tile summary used by subset selection.
  - `zoning_proposal`, `cell_density_proposal`, `road_skeleton_proposal` —
    each section carries `locked_buckets`, `candidate_strategies`, and a
    distribution summary.
  - `zoning_orthogonality` — Pearson correlation of zoning vs density.
  - `selected_layer3_tiles` — the deterministic Layer-3 subset with
    `rationale` per tile. Defaults to up to 12 tiles
    (`--max-subset-tiles` overrides). Tiles with `active_cell_count == 0`
    are filtered out by the eligibility predicate before any dimension
    ranking applies — they carry no derivation evidence regardless of how
    a meta dimension (e.g. `coastal_inland_river`) might rank them.

Default mode (without `--proposal-only`) writes `frequency_analysis.yaml`,
which is the analysis dict only — no `status`, no subset. The proposal file
is the one Gate 2 reviews.

## Gate 2 review process

1. **Inspect `macro_vocab_proposal.yaml` first** — this is the entry point.
2. **Read `candidate_strategies` in each proposal section.** Each candidate
   carries its full bucket definition (`kept_tokens`/`merged_tokens` for
   zoning; `bucket_boundaries` for density; `bucket_lower_bounds` for road
   skeleton), plus `categories`, `coverage`, and `marginal_cost` relative to
   the prior strategy.
3. **Decide which strategy to lock for each section.** The default
   `locked_buckets` is pre-filled to the most-granular candidate. If you
   prefer a different cut, hand-edit `locked_buckets` in this YAML to match
   one of the other candidates' bucket definitions. `git diff` on the
   proposal makes the choice auditable.
4. **Cross-check the Layer-3 subset.** `selected_layer3_tiles[*].rationale`
   should mention at least one of: zoning, density, road skeleton, scope,
   coastal/inland/riverside. A tile with no rationale-grade evidence (e.g.
   an empty masked tile) MUST NOT appear in the subset.
5. **Verify input digests** match the sub-C release you intend to lock
   against. Sub-D records bytes-sha over the sub-C files it read; if these
   change, re-derive before promoting.
6. **Cross-check `derivation_evidence`/`marginal_cost_of_cut` monotonicity.**
   Non-decreasing marginal costs are the typical heavy-tail pattern but not
   universal — a non-monotonic series is a signal to investigate, not a
   hard failure. Sub-D does not enforce monotonicity in
   `validate_frequency_analysis`.

When the proposal is approved, run
`scripts/promote_macro_vocab.py --proposal <this-file> --output configs/macro_plan/v1/macro_plan_vocab.yaml`
(Task 8). The promote script flips `status: proposal` -> `status: locked`
and writes the locked artifact; it is the only sanctioned edit between the
reviewed proposal and the locked vocab.

## Gate 2 cannot close without real Singapore sub-C output

Synthetic-data fixtures exercise the analysis machinery in unit tests, but
the vocab cuts the reviewer locks must be derived from *real* sub-C output
under `data/processed/sub_c/2026-04-15.0/singapore`. There is no
synthetic-data shortcut for vocab approval: if real sub-C output is
unavailable, halt Gate 2 until it is generated.
