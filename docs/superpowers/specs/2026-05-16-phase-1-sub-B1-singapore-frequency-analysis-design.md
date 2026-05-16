# Phase 1 sub-project B1 — Singapore frequency analysis design

- **Date:** 2026-05-16
- **Phase:** 1, sub-project B1 (vocabulary frequency analysis)
- **Status:** Draft, pending user review
- **Owner:** umar

## 1. Goal

Produce a reviewable markdown report that characterises the categorical-field distributions in Singapore's Overture data (release `2026-04-15.0`) at the granularity required to inform Phase 1 vocabulary decisions. The report exposes coverage, rank-frequency shape, and the effect of a small lattice of floor strategies for nine categorical fields across five themes. B1 emits **no vocabulary YAML** — that is B2's responsibility, gated by user review of this report.

The report is auto-generated, byte-reproducible from a code commit + the cached parquets, and explicitly marked **provisional pending Sweden** so reviewers do not freeze vocabulary on single-region data.

## 2. Scope (in / out)

**In scope for B1:**

- A library `cfm.data.frequency` with pure functions for coverage, rank-frequency, floor-strategy application, and list-length distribution.
- A thin CLI script `scripts/analyse_singapore_frequencies.py` that loads the cached Singapore region, runs the library against nine fields, and writes the markdown report + sibling PNG plots.
- Nine analyses across the five Overture themes (two categorical fields per theme except `divisions`):
  - `buildings.class`, `buildings.subtype`
  - `transportation.class`, `transportation.subclass`
  - `base.subtype`, `base.class`
  - `places.categories.primary`, `places.categories.alternate`
  - `divisions.country`
- A locked five-strategy floor lattice, presented per-field in a cut-behavior table and as horizontal threshold lines on each log-log plot.
- Coverage summary across all fields at the top of the report.
- Methodology section that defines coverage, denominators, alternate-counting semantics, and floor strategies before any field tables appear.
- Implications-for-B2 section that enumerates downstream decisions without making them.
- Reproducibility metadata: HTML-comment header (first lines of the file) plus a numbered Reproducibility section at the end.
- Pytest unit tests on the library functions (hand-crafted Counter-like inputs) plus one shape-only integration test that runs the script against sub-A's existing fixtures via `LocalFixtureBackend`.

**Out of scope for B1** (deferred):

- **Vocabulary YAML.** B2's job. B1 ships analysis, not decisions.
- **Per-field floor selection.** B1 ships the lattice; B2 picks one strategy per field after user review.
- **Missing-value handling decision.** B1 enumerates three options (emit `<unknown>`, drop missing-class features, infer class from context). B2 chooses.
- **`places.categories.alternate` max-alternates cap.** B1 emits the list-length distribution; B2 decides the cap.
- **Sweden frequency analysis.** A follow-up B1' will rerun the same library against Sweden once added. The library is written to support this without modification.
- **Fixture changes.** B1 must not add to or modify `tests/fixtures/overture_mini/`. Hand-crafted test data lives in B1's own test directory.
- **Geometric, spatial, or topological analyses.** This is categorical-field frequency only.
- **Numeric-field distribution analyses.** No building-area histograms, no road-length distributions. Those belong in later sub-projects if needed.

## 3. Load-bearing decisions

1. **Coverage and frequency are separated.** Each field's section reports `Coverage: X% non-null (N_present / N_total)` at the top, then rank tables and log-log plots over present values only. NULL is never plotted as a category. Rationale: the power-law shape that drives floor decisions is the *category* distribution; mixing missingness signal into the plot hides the curve on fields like `buildings.class` where NULLs may dominate. Coverage and `<unknown>`-token handling are explicit B2 decisions.
2. **Per-field floor lattice, not a single floor.** Field populations span ~10² to ~10⁵, so a single absolute floor would behave incommensurably across fields. Floor formula: `effective_floor = max(percentage × N_present, hard_min)`. Five named strategies span a ~100× range of pressure (see §6). Each field's cut-behavior table shows all five.
3. **`places.categories.alternate` uses occurrence counts, not row counts.** A row whose alternates are `["restaurant", "cafe"]` contributes +1 to "restaurant" and +1 to "cafe" in the alternate frequency table. Denominators differ from primary fields and the methodology section calls this out explicitly so per-field tables are scannable without ambiguity.
4. **Library + thin CLI, fully auto-generated report.** Code lives in `src/cfm/data/frequency.py` (testable pure functions) and `scripts/analyse_singapore_frequencies.py` (load + glue + render). The .md file is a build artefact: same commit + same cache + same matplotlib pin → byte-identical report and plots. To change wording, edit the script's template strings and re-run. The HTML-comment header at the top of the .md file states this rule directly.
5. **Provisional-pending-Sweden is a content-validity status, not just a label.** The status line in the report states the consequence: "Floor strategies and per-field cuts derived from one region may shift when Sweden is added. B2 should not freeze vocabulary on these numbers alone." Same wording appears in §4 of the report.
6. **Sub-project boundaries stay clean.** B1 does not touch `src/cfm/data/overture/` (sub-A's surface), does not modify `tests/fixtures/overture_mini/`, and does not write any config under `configs/tokenizer/`. The only files B1 creates or modifies live under `src/cfm/data/frequency.py`, `scripts/analyse_singapore_frequencies.py`, `tests/data/test_frequency.py`, `tests/data/test_analyse_singapore_frequencies.py`, and the generated `reports/2026-05-16-...` artefacts.

## 4. Public API (library)

```python
from cfm.data.frequency import (
    FieldFrequencyResult,
    FloorStrategy,
    CutBehaviorRow,
    ListLengthDistribution,
    compute_field_frequencies,
    apply_floor_strategy,
    compute_list_length_distribution,
    render_field_section,
    render_report,
)
```

Key dataclasses (frozen):

```python
@dataclass(frozen=True)
class FieldFrequencyResult:
    field: str                       # e.g. "buildings.class"
    n_total: int                     # total rows in the theme table
    n_present: int                   # rows where the field is non-null (or non-empty for list fields)
    counts: dict[str, int]           # category -> count; for alternate, sums across rows
    is_list_field: bool              # True for places.categories.alternate; False elsewhere
    total_occurrences: int           # = sum(counts.values()); == n_present for non-list fields,
                                     # >= n_present for list fields

@dataclass(frozen=True)
class FloorStrategy:
    name: str                        # "Very lenient", "Lenient", "Moderate", "Strict", "Very strict"
    percentage: float | None         # 0.0001 = 0.01%; None for Very lenient (min only)
    hard_min: int                    # absolute floor regardless of percentage

@dataclass(frozen=True)
class CutBehaviorRow:
    strategy: FloorStrategy
    effective_floor: int             # max(percentage * N_present, hard_min)
    n_total_categories: int          # before cutting
    n_kept: int
    n_dropped: int
    coverage_retained_pct: float     # fraction of (occurrences over kept categories) / total_occurrences

@dataclass(frozen=True)
class ListLengthDistribution:
    field: str                       # "places.categories.alternate"
    n_total_rows: int                # total rows in the places theme
    buckets: dict[str, tuple[int, float]]  # "0" -> (count, pct of n_total_rows); keys: "0","1","2","3","4","5+"
```

Functions:

```python
def compute_field_frequencies(
    table: pyarrow.Table,
    column_path: str,                # path within `table`, e.g. "class" or "categories.primary"
    *,
    label: str | None = None,        # stored in FieldFrequencyResult.field; defaults to column_path
    is_list_field: bool = False,
) -> FieldFrequencyResult: ...

def apply_floor_strategy(
    result: FieldFrequencyResult,
    strategy: FloorStrategy,
) -> CutBehaviorRow: ...

def compute_list_length_distribution(
    table: pyarrow.Table,
    column_path: str,                # e.g., "categories.alternate"
    *,
    label: str | None = None,        # stored in ListLengthDistribution.field
) -> ListLengthDistribution: ...

def render_field_section(
    result: FieldFrequencyResult,
    cut_rows: list[CutBehaviorRow],
    plot_relative_path: str,
    list_length: ListLengthDistribution | None = None,
    binds_to_prd_framing_only: bool = False,
) -> str: ...

def render_report(
    region_name: str,
    overture_release: str,
    manifest_path: pathlib.Path,
    field_sections: list[str],
    coverage_summary_rows: list[tuple[str, int, int, float]],
    commit_sha: str,
    run_timestamp_utc: datetime,
    rerun_reason: str,                # "initial" on first run
) -> str: ...
```

**Naming convention.** `column_path` is the in-table path passed to pyarrow operations (e.g., `"class"` against the `buildings` table, or `"categories.primary"` against the `places` table — struct access via dot syntax, traversed internally). `label` is the human-readable display name stored in the dataclass and shown in the report (e.g., `"buildings.class"`, `"places.categories.alternate"`). The CLI is responsible for constructing the label as `f"{theme_name}.{column_path}"` and passing both. Library tests can omit `label` and let it default to `column_path` for brevity.

## 5. CLI script — `scripts/analyse_singapore_frequencies.py`

Thin glue:

1. Parse `--rerun-reason <str>` (default `"initial"`) and `--output-dir <path>` (default `reports/`).
2. Resolve current git commit sha via `subprocess.run(["git", "rev-parse", "HEAD"])`.
3. Capture `datetime.now(timezone.utc)` once at start; reuse for both header and footer.
4. `region = load_region("singapore")` (sub-A's API; cache-hit, ~1 s).
5. Resolve the report and plot directory paths:
   - `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md`
   - `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis_plots/`
   - Date is **fixed at 2026-05-16** across re-runs (not derived from `today()`).
6. Iterate the nine fields. For each:
   - Compute `FieldFrequencyResult`.
   - Compute `CutBehaviorRow` for each of the five floor strategies.
   - Detect "binds to PRD-framing-only" for Very strict if `n_kept ≤ 3`; flag for §3.x annotation.
   - Render log-log rank-frequency PNG with threshold lines (see §7).
   - For `places.categories.alternate`, also compute `ListLengthDistribution`.
   - Render the field-section markdown via `render_field_section`.
7. Render the coverage summary table from all nine results.
8. Render the full report via `render_report` and write to the resolved path.
9. Print summary to stdout: report path, plot paths, total wall-clock.

The script does **not** invoke a re-fetch. If the Singapore cache is missing, `load_region` will fall through to the cold path — that is sub-A's behavior and is documented in `docs/known_issues.md` #1. B1 does not work around this; B1 expects the cache present (the handoff confirms it is).

## 6. Floor-strategy lattice

Five named strategies, fixed at the library level:

| Name | Percentage | Hard min |
|---|---|---|
| Very lenient | — | 10 |
| Lenient | 0.03% | 30 |
| Moderate | 0.1% | 100 |
| Strict | 0.3% | 300 |
| Very strict | 1% | 1000 |

Effective floor per field: `max(percentage × N_present, hard_min)`. Very lenient has no percentage component (acts as a pure min-10 cut).

**PRD-framing flag.** For any field where the Very strict strategy retains ≤3 categories, the field's section opens with a single annotation line: `Very strict floor binds to fewer than 4 categories on Singapore data; this row is shown for PRD §5 framing only and should not be selected as a Phase 1 cut.` The row stays in the cut-behavior table for comparison context with PRD §5's global 10,000-instance floor, but its impracticality at single-region scale is called out so B2 does not accidentally select it.

## 7. Plot specification

One PNG per field, written under the `_plots/` sibling directory.

- **Axes:** log-log. X = rank (1 = most common). Y = count.
- **Markers:** single colour, single style; small dots connected by a thin line.
- **Threshold lines:** five horizontal dashed lines at each strategy's effective floor for that field. Each line labelled with the strategy name at the right margin. Lines styled distinctly enough to read at PNG resolution (e.g., different dash patterns + a small text label).
- **Title:** `<field> rank-frequency (N_present = ...)`.
- **Determinism.** The script:
  - Pins `matplotlib.use("Agg")` before importing pyplot.
  - Calls `matplotlib.rcParams['svg.hashsalt'] = '0'` and `matplotlib.rcParams['pdf.use14corefonts'] = True` for cross-run stability.
  - Sets `figure(figsize=(8, 5), dpi=100)` explicitly and `tight_layout()` after plot composition.
  - Sorts categories by `(-count, name)` before plotting so ties resolve deterministically.
  - Does **not** call any RNG.
  - Saves with `savefig(path, format='png', metadata={'Software': '', 'Creator': ''})` to strip the matplotlib-version footer that would otherwise drift between environments.
- **Determinism caveat.** If matplotlib's internal PNG encoder produces non-byte-identical output across patch versions despite the above, the methodology section documents the exception explicitly so a no-op re-run that shows plot diffs in `git status` does not confuse a future contributor. The pinning is best-effort; the script's behavior is deterministic.

## 8. Report skeleton (the deliverable)

The exact `.md` structure produced by `render_report`:

```
<!-- Auto-generated by scripts/analyse_singapore_frequencies.py
     Commit:        <sha>
     Overture:      2026-04-15.0
     Run UTC:       <iso>
     Re-run reason: <reason>           (default "initial")
     Do not edit by hand — edit the script and re-run. -->

# Phase 1 sub-B1 — Singapore frequency analysis

**Status: provisional. Floor strategies and per-field cuts derived from one region may shift when Sweden is added. B2 should not freeze vocabulary on these numbers alone.**

## 1. Methodology

(Fields analysed, coverage definitions, denominators, floor-strategy lattice, plot interpretation, data source. ~250 words of prose.)

## 2. Coverage summary

| Field | N_total | N_present | Coverage |
|---|---|---|---|
| buildings.class | ... |
| buildings.subtype | ... |
| ... |
| divisions.country | ... |

## 3. Field analyses

### 3.1 buildings.class
(optional PRD-framing-flag line)
![rank-frequency](_plots/buildings_class.png)
| Strategy | Effective floor | Total categories | Kept | Dropped | % coverage retained |
| Very lenient | ... |
| Lenient | ... |
| Moderate | ... |
| Strict | ... |
| Very strict | ... |

### 3.2 buildings.subtype
(same shape)

### 3.3 transportation.class
### 3.4 transportation.subclass
### 3.5 base.subtype
### 3.6 base.class
### 3.7 places.categories.primary
### 3.8 places.categories.alternate

(opens with the list-length distribution table)

| List length | Count | % of all places rows |
| 0 | ... | ... |
| 1 | ... | ... |
| 2 | ... | ... |
| 3 | ... | ... |
| 4 | ... | ... |
| 5+ | ... | ... |

(then the plot and cut-behavior table on per-occurrence counts)

### 3.9 divisions.country

## 4. Implications for B2 (provisional)

- **Missing-value handling.** B1 does not decide. Three options for B2:
  - emit `<unknown>` token for missing-class rows;
  - drop missing-class features from training entirely;
  - infer class from context (geometry, neighbouring features).
- **Per-field floor selection.** B2 picks one strategy per field from §3's lattice (or computes a hybrid). Strategies flagged as PRD-framing-only should not be picked unless intentional.
- **`places.categories.alternate` max-alternates cap.** Decision input is the list-length distribution in §3.8. B1 does not propose a cap.
- **All B2 decisions are provisional pending Sweden.** Re-run B1 against Sweden when sub-A's cold-fetch issue is resolved and a Swedish cache exists. Compare distributions; widen the lattice or change strategy names if needed.

## 5. Reproducibility

- Overture release: 2026-04-15.0
- Cache manifest: `data/cache/overture/2026-04-15.0/singapore/manifest.yaml`
- Per-theme cache sha256s:
  - buildings: <sha>
  - places: <sha>
  - transportation: <sha>
  - base: <sha>
  - divisions: <sha>
- Code commit: <sha>
- Run timestamp (UTC): <iso>
- Re-run reason: <reason>
- Generated by: `scripts/analyse_singapore_frequencies.py`
```

## 9. Module and file layout

```
src/cfm/data/
└── frequency.py                     # all library functions and dataclasses

scripts/
└── analyse_singapore_frequencies.py # CLI; thin glue + matplotlib calls

tests/data/
├── test_frequency.py                # unit tests on library functions
└── test_analyse_singapore_frequencies.py  # shape-only integration test

reports/
├── 2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md
└── 2026-05-16-phase-1-sub-B1-singapore-frequency-analysis_plots/
    ├── buildings_class.png
    ├── buildings_subtype.png
    ├── transportation_class.png
    ├── transportation_subclass.png
    ├── base_subtype.png
    ├── base_class.png
    ├── places_categories_primary.png
    ├── places_categories_alternate.png
    └── divisions_country.png
```

`src/cfm/data/__init__.py` already exists from sub-A; no changes required there.

## 10. Coverage and counting semantics — methodology details

Definitions that appear in §1 of the report and are enforced by the library:

- **`coverage`** for a primary (scalar) field: `n_present / n_total`, where `n_present` is the count of rows where the field is not null.
- **`coverage`** for `places.categories.alternate`: `n_present / n_total`, where `n_present` is the count of rows where the `alternate` list is non-null *and* non-empty. A row whose `alternate` is `null` or `[]` counts as not present.
- **`% coverage retained` (cut-behavior table)** for a primary field: `sum_of_counts_over_kept_categories / total_occurrences`, where `total_occurrences == n_present` for primary fields.
- **`% coverage retained` (cut-behavior table)** for `places.categories.alternate`: `sum_of_counts_over_kept_categories / total_occurrences`, where `total_occurrences == sum(len(row.alternate)) over non-empty rows` and exceeds `n_present`.

The library functions enforce these via the `is_list_field` flag on `compute_field_frequencies`. The CLI passes `is_list_field=True` only for `categories.alternate`.

## 11. Tests

**Library unit tests** (`tests/data/test_frequency.py`, fast suite):

- `test_compute_field_frequencies_basic` — Counter-like input; assert counts, n_present, total_occurrences.
- `test_compute_field_frequencies_all_null` — all-NULL field; coverage = 0%, counts empty, n_present = 0, no exception.
- `test_compute_field_frequencies_single_value` — every row has the same category; every floor keeps the one value; coverage_retained = 100%.
- `test_compute_field_frequencies_list_field_basic` — list field with mixed list lengths; total_occurrences > n_present; counts sum correctly.
- `test_compute_field_frequencies_list_field_all_empty` — every row has `[]` alternate; n_present = 0; counts empty.
- `test_apply_floor_strategy_effective_floor_formula` — assert `max(pct * N, hard_min)` for a few hand-picked tuples.
- `test_apply_floor_strategy_percentage_dominates` — large N, large percentage; percentage component wins.
- `test_apply_floor_strategy_hard_min_dominates` — small N; hard_min wins.
- `test_apply_floor_strategy_coverage_retained` — known counts; assert retention percentage.
- `test_compute_list_length_distribution_buckets` — hand-crafted list lengths spanning 0..7; assert 5+ bucket aggregates lengths ≥5.
- `test_compute_list_length_distribution_denominator_is_total_rows` — confirm denominator is `n_total_rows`, not `n_present`.
- `test_render_field_section_includes_prd_framing_flag` — when `binds_to_prd_framing_only=True`, assert annotation line appears.

**Integration test** (`tests/data/test_analyse_singapore_frequencies.py`, fast suite):

- `test_script_runs_against_fixtures_and_produces_well_formed_report` — invokes the script with `LocalFixtureBackend` (or via subprocess with an env var that the script reads to swap backends; design choice locked at implementation time), points output at `tmp_path`, asserts:
  - the .md file exists at the expected path under `tmp_path`,
  - all nine `### 3.x` section headers are present,
  - the methodology section (§1) appears,
  - the coverage summary table (§2) has 9 data rows,
  - the implications-for-B2 section (§4) appears,
  - the reproducibility section (§5) appears,
  - one PNG file exists per field in the `_plots/` directory.
- **Shape-only assertions.** No assertions about specific category counts on the fixture data — that would couple B1's tests to sub-A's fixture content and break sub-project isolation.

Run with `uv run pytest tests/data/test_frequency.py tests/data/test_analyse_singapore_frequencies.py`. Should complete in under 5 seconds.

## 12. Reproducibility metadata — HTML comment header

First four lines of every generated report (rendered as an HTML comment so GitHub hides them but any editor shows them):

```
<!-- Auto-generated by scripts/analyse_singapore_frequencies.py
     Commit:        <40-char sha>
     Overture:      2026-04-15.0
     Run UTC:       2026-05-16T14:32:11Z
     Re-run reason: initial
     Do not edit by hand — edit the script and re-run. -->
```

`Re-run reason` is a single-line free-form string passed via `--rerun-reason`. Defaults to `"initial"` if omitted. Examples for future re-runs: `"sweden-added"`, `"floor-lattice-revised"`, `"matplotlib-pin-bump"`, `"bug-fix-list-length-bucket"`. The audit trail in `git log` of the report file shows what changed and why across the report's life.

## 13. Errors

B1 raises standard Python exceptions (no custom error class hierarchy needed at this scope):

- `ValueError` from `compute_field_frequencies` if `field` doesn't exist in the table's schema.
- `ValueError` from `apply_floor_strategy` if `result.n_present < 0` (defensive).
- `RuntimeError` from the CLI if `git rev-parse HEAD` fails (e.g., not in a git repo) — the script aborts rather than writing a report with a bogus commit sha.

## 14. Done criteria

B1 is done when:

- `uv run pytest tests/data/test_frequency.py tests/data/test_analyse_singapore_frequencies.py` passes (fast suite).
- `uv run python scripts/analyse_singapore_frequencies.py` produces `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md` and nine PNGs under the sibling `_plots/` directory in ≤ 60 seconds wall-clock.
- The generated report has all five numbered sections, all nine field subsections, the HTML-comment header, and the provisional-status line.
- A second invocation with `--rerun-reason "rerun-test"` produces a byte-identical `.md` file except for the run-timestamp and rerun-reason lines (and ideally byte-identical PNGs; if not, the deviation is documented in methodology).
- The user has reviewed the report and approved (or requested changes that are then applied).
- No changes have been made to `src/cfm/data/overture/`, `tests/fixtures/overture_mini/`, or any sub-A surface.

## 15. Risks specific to B1

- **`places.categories` is a struct with non-trivial pyarrow access patterns.** The sub-A spec noted that `places.categories` is typed as `string` in `schema.py` but is actually a struct in real Overture data. The library must traverse `pa.Table → categories struct → primary/alternate fields` correctly. Mitigation: unit test against a hand-crafted pyarrow Table that mirrors the real struct shape.
- **`n_total` for `places.categories.alternate`'s list-length distribution is the full `places` row count, not non-null `alternate` count.** Easy to get wrong; the user explicitly called this out. Mitigation: dedicated unit test `test_compute_list_length_distribution_denominator_is_total_rows`.
- **Singapore building-class coverage may be unusually high or low compared to the 94% global missing rate.** Either outcome is fine — the report shows what it shows — but if Very strict happens to bind on a field nobody expected, the PRD-framing-flag must surface it cleanly. Mitigation: the flag is computed mechanically (`n_kept ≤ 3`), not field-gated.
- **Plot byte-identity across matplotlib patch versions.** Best effort; documented as an exception in methodology if it fails. Not a blocker for B1 sign-off.
- **The integration test running the CLI script may need a backend-injection seam.** The script defaults to the real `S3DuckDBBackend` via `load_region`. For the integration test to use `LocalFixtureBackend`, either (a) the script accepts a `--backend fixture` flag, or (b) the test monkeypatches `load_region`. Implementation chooses; (a) is cleaner if the seam is cheap.

## 16. Out-of-scope deferrals — what B2 picks up

- **B2** reviews this report, selects one floor strategy per field (or constructs a hybrid), decides missing-value handling for each field, decides the `places.categories.alternate` max-alternates cap, and emits `configs/tokenizer/vocab_phase1.yaml`. B2 may also propose lattice changes if the five strategies do not bracket the right decision space.
- **B1' (Sweden)** re-runs the same library against a Swedish Overture cache once that exists. Comparison of the two reports informs whether Phase 1 vocabulary needs region-conditional treatment.
- **Numeric-field distributions** (building areas, road lengths, POI density per km²) are deferred. If they turn out to matter for vocabulary, they live in a separate B1.5 or are folded into C's tile-extraction validator.
