"""The per-city stage table: how to invoke each stage, where it writes, and
which source paths gate its invalidate-on-fix (spec §2, §3.3).

``source_globs`` is the load-bearing field: a stage re-runs when any tracked file
matching its globs changed since the sha its marker was stamped with. Globs were
derived by **tracing** each stage script's transitive ``cfm.*`` imports (B1 step 1
trace, 2026-06-03), NOT guessed — under-globbing silently leaves stale post-fix
artifacts; over-globbing only costs a needless re-run. Shared modules a stage does
not import directly (e.g. ``sub_c/coords.py`` for sub_d/sub_f/validate) propagate
via the downstream **cascade** in ``state.stages_to_run`` (a coords change flags
sub_c, whose re-run cascades), so they are intentionally absent from those globs.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_PY = sys.executable


@dataclass(frozen=True)
class StageContext:
    """Everything a stage needs to build its invocation, for one city."""

    region: str
    release: str
    repo_root: Path
    commit_sha: str
    sub_c_dir: Path
    sub_d_dir: Path
    sub_e_dir: Path
    sub_f_dir: Path
    sub_g_dir: Path


@dataclass(frozen=True)
class Stage:
    name: str
    #: marker filename that signals this stage completed.
    marker: str
    #: dir whose marker the driver checks (None for fetch: the cache dir, handled
    #: specially by the driver because it is not under data/processed).
    output_dir: Callable[[StageContext], Path] | None
    #: argv builder for subprocess (empty for fetch; fetch is in-process load_region).
    argv: Callable[[StageContext], list[str]]
    #: tracked pathspecs whose change invalidates this stage + everything downstream.
    source_globs: tuple[str, ...]


STAGE_ORDER: tuple[Stage, ...] = (
    Stage(
        name="fetch",
        marker="manifest.yaml",  # the overture cache manifest
        output_dir=None,  # cache dir; driver resolves + checks specially
        argv=lambda c: [],  # in-process load_region; driver handles
        source_globs=("src/cfm/data/overture/",),
    ),
    Stage(
        name="sub_c",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_c_dir,
        argv=lambda c: [
            _PY,
            "scripts/extract_tiles.py",
            "--region",
            c.region,
            "--release",
            c.release,
            "--output-dir",  # explicit so write-dir == the dir we check the marker in
            str(c.sub_c_dir),
        ],
        source_globs=(
            "scripts/extract_tiles.py",
            "src/cfm/data/sub_c/",  # includes coords.py
            "src/cfm/data/overture/",
            "src/cfm/data/io.py",
            "src/cfm/data/determinism.py",
        ),
    ),
    Stage(
        name="sub_d",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_d_dir,
        argv=lambda c: [
            _PY,
            "scripts/derive_macro_plan.py",
            "--region",
            c.region,
            "--release",
            c.release,
            "--sub-c-dir",
            str(c.sub_c_dir),
            "--output-dir",
            str(c.sub_d_dir),
            "--macro-vocab",
            "configs/macro_plan/v1/macro_plan_vocab.yaml",
            "--commit-sha",
            c.commit_sha,  # REQUIRED only on sub_d (work-item #1)
        ],
        source_globs=(
            "scripts/derive_macro_plan.py",
            "src/cfm/data/sub_d/",
            "src/cfm/data/io.py",
            "src/cfm/data/determinism.py",
        ),
    ),
    Stage(
        name="sub_e",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_e_dir,
        argv=lambda c: [
            _PY,
            "scripts/derive_boundary_contracts.py",
            "--release",
            c.release,
            "--region",
            c.region,
            "--sub-c-region-dir",
            str(c.sub_c_dir),
            "--sub-d-region-dir",
            str(c.sub_d_dir),
            "--output-region-dir",
            str(c.sub_e_dir),
        ],
        source_globs=(
            "scripts/derive_boundary_contracts.py",
            "src/cfm/data/sub_e/",
            "src/cfm/data/sub_c/",  # sub_e imports sub_c (features/enums)
            "src/cfm/data/sub_d/",
            "src/cfm/data/io.py",
            "src/cfm/data/determinism.py",
        ),
    ),
    Stage(
        name="sub_f",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_f_dir,
        argv=lambda c: [
            _PY,
            "scripts/sub_f/derive.py",
            "--release",
            c.release,
            "--region",
            c.region,
            "--sub-c-region-dir",
            str(c.sub_c_dir),
            "--sub-d-region-dir",
            str(c.sub_d_dir),
            "--sub-e-region-dir",
            str(c.sub_e_dir),
            "--output-region-dir",
            str(c.sub_f_dir),
        ],
        source_globs=(
            "scripts/sub_f/derive.py",
            "src/cfm/data/sub_f/",
            "src/cfm/data/sub_d/",
            "src/cfm/data/sub_e/",
            "src/cfm/data/io.py",
            "src/cfm/data/determinism.py",
        ),
    ),
    Stage(
        name="validate",
        marker="_PHASE1_VALIDATED",
        output_dir=lambda c: c.sub_g_dir,
        argv=lambda c: [
            _PY,
            "scripts/sub_g/validate_phase1_region.py",
            "--region",
            c.region,
            "--release",
            c.release,
            "--sub-c-region-dir",
            str(c.sub_c_dir),
            "--sub-d-region-dir",
            str(c.sub_d_dir),
            "--sub-e-region-dir",
            str(c.sub_e_dir),
            "--sub-f-region-dir",
            str(c.sub_f_dir),
            "--output-dir",
            str(c.sub_g_dir),
        ],
        source_globs=(
            "scripts/sub_g/validate_phase1_region.py",
            "src/cfm/data/sub_g/",
            "src/cfm/data/sub_d/",
            "src/cfm/data/sub_e/",
            "src/cfm/data/sub_f/",
            "src/cfm/data/io.py",
        ),
    ),
)
