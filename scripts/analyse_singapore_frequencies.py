#!/usr/bin/env python
"""Phase 1 sub-B1 — Singapore frequency analysis report generator.

Reads the cached Singapore Overture region, computes categorical-field
frequency distributions for nine fields across five themes, and writes a
markdown report + log-log rank-frequency PNGs.

The report is a build artefact. To change it, edit this script and re-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Third-party imports. matplotlib.pyplot MUST come after matplotlib.use("Agg") —
# the remaining imports are suppressed for E402 because they must follow that call.
import matplotlib
import yaml

matplotlib.use("Agg")  # Pin backend BEFORE pyplot import. Required for headless determinism.
matplotlib.rcParams["svg.hashsalt"] = "0"
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt  # noqa: E402

from cfm.data.frequency import (  # noqa: E402
    CutBehaviorRow,
    FloorStrategy,
    apply_floor_strategy,
    compute_field_frequencies,
    compute_list_length_distribution,
    render_field_section,
    render_report,
)
from cfm.data.overture import load_region  # noqa: E402
from cfm.data.overture.backend import LocalFixtureBackend, S3DuckDBBackend  # noqa: E402

REPORT_DATE = "2026-05-16"  # FIXED; Sweden re-run produces a separate dated report.
REPORT_NAME = f"{REPORT_DATE}-phase-1-sub-B1-singapore-frequency-analysis"

FLOOR_STRATEGIES: tuple[FloorStrategy, ...] = (
    FloorStrategy(name="Very lenient", percentage=None, hard_min=10),
    FloorStrategy(name="Lenient", percentage=0.0003, hard_min=30),
    FloorStrategy(name="Moderate", percentage=0.001, hard_min=100),
    FloorStrategy(name="Strict", percentage=0.003, hard_min=300),
    FloorStrategy(name="Very strict", percentage=0.01, hard_min=1000),
)


@dataclass(frozen=True)
class FieldPlan:
    theme: str  # name of the table in region.themes
    column_path: str  # in-table column path; dot syntax for struct fields
    is_list_field: bool  # only places.categories.alternate
    plot_basename: str  # without extension; the PNG filename
    section_number: str  # "3.1", "3.2", ...


FIELDS: tuple[FieldPlan, ...] = (
    FieldPlan("buildings", "class", False, "buildings_class", "3.1"),
    FieldPlan("buildings", "subtype", False, "buildings_subtype", "3.2"),
    FieldPlan("transportation", "class", False, "transportation_class", "3.3"),
    FieldPlan("transportation", "subclass", False, "transportation_subclass", "3.4"),
    FieldPlan("base", "subtype", False, "base_subtype", "3.5"),
    FieldPlan("base", "class", False, "base_class", "3.6"),
    FieldPlan("places", "categories.primary", False, "places_categories_primary", "3.7"),
    FieldPlan("places", "categories.alternate", True, "places_categories_alternate", "3.8"),
    FieldPlan("divisions", "country", False, "divisions_country", "3.9"),
)


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "could not resolve git commit sha; refusing to write a report with bogus provenance"
        ) from e
    return result.stdout.strip()


def _render_plot(
    field_label: str,
    counts: dict[str, int],
    cut_rows: list[CutBehaviorRow],
    out_path: Path,
) -> None:
    """Log-log rank-frequency PNG with horizontal threshold lines per strategy."""
    if not counts:
        # Single annotation; nothing to plot.
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.text(
            0.5,
            0.5,
            f"no data for {field_label}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, format="png", metadata={"Software": "", "Creator": ""})
        plt.close(fig)
        return

    sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks = list(range(1, len(sorted_counts) + 1))
    values = [c for _, c in sorted_counts]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.loglog(ranks, values, marker="o", markersize=3, linewidth=0.8, color="black")
    ax.set_xlabel("rank (1 = most common)")
    ax.set_ylabel("count")
    ax.set_title(f"{field_label} rank-frequency (N_present = {sum(values):,})")

    # Horizontal threshold lines per strategy. Use legend per spec §7 fallback.
    # Linestyles distinguish strategies visually; legend pins the names.
    line_styles = [":", "--", "-.", "-", (0, (3, 1, 1, 1))]
    for style, row in zip(line_styles, cut_rows, strict=True):
        ax.axhline(
            row.effective_floor,
            linestyle=style,
            linewidth=0.8,
            alpha=0.7,
            label=f"{row.strategy.name} (floor={row.effective_floor:,})",
        )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, format="png", metadata={"Software": "", "Creator": ""})
    plt.close(fig)


def _select_backend(name: str, fixture_dir: Path):
    if name == "real":
        return S3DuckDBBackend()
    if name == "fixture":
        return LocalFixtureBackend(fixtures_dir=fixture_dir)
    raise ValueError(f"unknown --backend {name!r}; expected 'real' or 'fixture'")


def _load_manifest_shas(manifest_path: Path) -> dict[str, str]:
    data = yaml.safe_load(manifest_path.read_text())
    themes = data.get("themes", {})
    return {theme: entry.get("sha256", "") for theme, entry in themes.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 1 sub-B1 Singapore frequency report."
    )
    parser.add_argument(
        "--rerun-reason",
        default="initial",
        help="Free-form one-line audit string written to the report header.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where the report and _plots/ subdirectory are written.",
    )
    parser.add_argument(
        "--backend",
        choices=("real", "fixture"),
        default="real",
        help=(
            "real = S3DuckDBBackend (production); fixture = LocalFixtureBackend (integration test)."
        ),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("tests/fixtures/overture_mini"),
        help="Fixture directory used when --backend=fixture.",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    commit_sha = _git_commit_sha()
    run_ts = datetime.now(UTC).replace(microsecond=0)
    backend = _select_backend(args.backend, args.fixture_dir)

    print(f"[B1] loading singapore via backend={type(backend).__name__}")
    region = load_region("singapore", backend=backend)
    print(f"[B1] release={region.release}  themes={list(region.themes)}")

    out_dir: Path = args.output_dir
    plots_dir = out_dir / f"{REPORT_NAME}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    field_sections: list[str] = []
    coverage_rows: list[tuple[str, int, int, float]] = []

    for plan in FIELDS:
        label = f"{plan.theme}.{plan.column_path}"
        if plan.theme not in region.themes:
            field_sections.append(
                f"### {plan.section_number} {label}\n\n"
                f"*Theme `{plan.theme}` not present in input data — section skipped.*\n"
            )
            coverage_rows.append((label, 0, 0, 0.0))
            continue
        table = region.themes[plan.theme]

        try:
            result = compute_field_frequencies(
                table, plan.column_path, label=label, is_list_field=plan.is_list_field
            )
        except ValueError as e:
            field_sections.append(
                f"### {plan.section_number} {label}\n\n"
                f"*Field not present in input data ({e}). "
                f"Section skipped; PNG not produced.*\n"
            )
            coverage_rows.append((label, table.num_rows, 0, 0.0))
            continue

        cut_rows = [apply_floor_strategy(result, s) for s in FLOOR_STRATEGIES]

        # PRD-framing flag: Very strict (last strategy) retains <= 3 categories.
        binds_to_prd_framing_only = cut_rows[-1].n_kept <= 3

        plot_relative = f"{REPORT_NAME}_plots/{plan.plot_basename}.png"
        _render_plot(label, result.counts, cut_rows, plots_dir / f"{plan.plot_basename}.png")

        list_length = None
        if plan.is_list_field:
            try:
                list_length = compute_list_length_distribution(table, plan.column_path, label=label)
            except ValueError:
                list_length = None

        field_sections.append(
            render_field_section(
                result=result,
                cut_rows=cut_rows,
                plot_relative_path=plot_relative,
                list_length=list_length,
                binds_to_prd_framing_only=binds_to_prd_framing_only,
                section_number=plan.section_number,
            )
        )

        coverage_pct = 100.0 * result.n_present / result.n_total if result.n_total > 0 else 0.0
        coverage_rows.append((label, result.n_total, result.n_present, coverage_pct))

    manifest_shas = _load_manifest_shas(region.manifest_path)
    # Render manifest as a repo-relative path so the report is machine-portable.
    try:
        manifest_for_report = region.manifest_path.relative_to(Path.cwd())
    except ValueError:
        manifest_for_report = region.manifest_path
    report = render_report(
        region_name="singapore",
        overture_release=region.release,
        manifest_path=manifest_for_report,
        per_theme_sha256=manifest_shas,
        field_sections=field_sections,
        coverage_summary_rows=coverage_rows,
        commit_sha=commit_sha,
        run_timestamp_utc=run_ts,
        rerun_reason=args.rerun_reason,
    )
    report_path = out_dir / f"{REPORT_NAME}.md"
    report_path.write_text(report)

    elapsed = time.monotonic() - started
    print(f"[B1] wrote report:  {report_path}")
    print(f"[B1] wrote plots:   {plots_dir}/")
    print(f"[B1] wall clock:    {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
