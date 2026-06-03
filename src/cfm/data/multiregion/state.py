"""Per-city orchestration state + invalidate-on-fix (spec §3.3).

A stage's completion is stamped with the commit-sha it was produced under. On a
re-run, a stage is invalidated when any tracked file in its ``source_globs``
changed between its recorded sha and HEAD; an invalidated stage AND all downstream
stages re-run; unchanged upstream stages do not. Shared-module changes propagate
via this cascade: a ``coords`` change flags ``sub_c`` (which globs ``sub_c/``),
whose re-run cascades to every downstream stage.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cfm.data.multiregion.stages import Stage


@dataclass(frozen=True)
class StageCompletion:
    stage: str
    sha: str


@dataclass
class CityState:
    region: str
    completions: dict[str, StageCompletion] = field(default_factory=dict)


def stage_source_changed(
    recorded_sha: str, head_sha: str, source_globs: tuple[str, ...], repo_root: Path
) -> bool:
    """True iff any tracked path in ``source_globs`` differs between the two shas."""
    if recorded_sha == head_sha:
        return False
    proc = subprocess.run(
        ["git", "diff", "--quiet", recorded_sha, head_sha, "--", *source_globs],
        cwd=repo_root,
    )
    if proc.returncode == 0:
        return False  # no diff
    if proc.returncode == 1:
        return True  # diff present
    raise RuntimeError(
        f"git diff failed (rc={proc.returncode}) comparing "
        f"{recorded_sha}..{head_sha} for {source_globs}"
    )


def stages_to_run(
    city: CityState, head_sha: str, stage_order: tuple[Stage, ...], repo_root: Path
) -> list[str]:
    """Stages needing a (re)run, in dependency order.

    A stage runs if: it has no recorded completion, OR its source changed since
    its recorded sha, OR any upstream stage is (re)running (downstream cascade).
    """
    to_run: list[str] = []
    cascade = False
    for stage in stage_order:
        rec = city.completions.get(stage.name)
        if rec is None:
            cascade = True
        elif stage_source_changed(rec.sha, head_sha, stage.source_globs, repo_root):
            cascade = True
        if cascade:
            to_run.append(stage.name)
    return to_run


def load_city_state(path: Path, region: str) -> CityState:
    if not path.exists():
        return CityState(region=region)
    raw = json.loads(path.read_text())
    return CityState(
        region=raw["region"],
        completions={k: StageCompletion(**v) for k, v in raw["completions"].items()},
    )


def save_city_state(path: Path, city: CityState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": city.region,
        "completions": {k: {"stage": v.stage, "sha": v.sha} for k, v in city.completions.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
