"""Per-city chain driver (spec §3). Mirrors sub_g/pipeline.py's subprocess pattern
(``subprocess.run(..., check=True)`` then RECHECK the marker) and adds:

- **sha-aware gating** — only the stages ``state.stages_to_run`` returns run; a
  stage whose code is unchanged since its recorded sha is skipped.
- **returncode is not trusted** — a stage is successful ONLY if it returns 0 AND
  its expected marker was written (exit-0-no-marker is a failure).
- **continue-but-loud** — a failed stage marks the city ``failed`` with the
  reason and stops that city, but ``run_batch`` keeps going and the city is never
  counted as validated or silently dropped.
- **current-sha re-stamping** — each completed stage is stamped with the CURRENT
  head sha so the next run sees it done (and a later code change re-invalidates it).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cfm.data.multiregion import stages, state
from cfm.data.multiregion.stages import Stage, StageContext

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CityResult:
    region: str
    status: str  # "validated" | "failed"
    detail: str = ""


def _head_sha(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_fetch(ctx: StageContext) -> None:
    """Stage 1a: in-process fetch on the egress-capable host (cache-hit is cheap).
    Success requires load_region to write its cache manifest — 'it returned' is
    not trusted; the marker (the cache manifest) must exist on disk."""
    from cfm.data.overture import load_region

    r = load_region(ctx.region, confirm=True, repo_root=ctx.repo_root)
    if not r.manifest_path.exists():
        raise RuntimeError(
            f"fetch {ctx.region}: load_region wrote no cache manifest at {r.manifest_path}"
        )


def _run_stage_subprocess(stage: Stage, ctx: StageContext) -> None:
    """Run a stage and accept it ONLY if returncode==0 AND its marker was written."""
    _log.info("city=%s stage=%s: running", ctx.region, stage.name)
    subprocess.run(stage.argv(ctx), cwd=ctx.repo_root, check=True)  # raises on nonzero
    assert stage.output_dir is not None  # only fetch has None; it is handled separately
    marker = stage.output_dir(ctx) / stage.marker
    if not marker.exists():
        raise RuntimeError(
            f"city={ctx.region} stage={stage.name} returned 0 but wrote no "
            f"{stage.marker} at {marker} (returncode alone is not trusted)"
        )


def run_city(ctx: StageContext, city: state.CityState, repo_root: Path) -> CityResult:
    """Run the stages that need (re)running for one city; continue-but-loud on failure."""
    head = _head_sha(repo_root)
    to_run = state.stages_to_run(city, head, stages.STAGE_ORDER, repo_root)
    if not to_run:
        return CityResult(region=ctx.region, status="validated")
    try:
        for stage in stages.STAGE_ORDER:
            if stage.name not in to_run:
                continue
            if stage.name == "fetch":
                _run_fetch(ctx)
            else:
                _run_stage_subprocess(stage, ctx)
            # Stamp ONLY after the stage is accepted (rc 0 + marker present).
            city.completions[stage.name] = state.StageCompletion(stage=stage.name, sha=head)
    except (subprocess.CalledProcessError, RuntimeError, ValueError, OSError) as exc:
        _log.error("city=%s FAILED-NEEDS-ATTENTION: %s", ctx.region, exc)
        return CityResult(region=ctx.region, status="failed", detail=str(exc))
    return CityResult(region=ctx.region, status="validated")


def run_batch(
    contexts: list[StageContext],
    city_states: dict[str, state.CityState],
    repo_root: Path,
) -> list[CityResult]:
    """Run each city; one city's failure NEVER kills the batch (continue-but-loud).

    Returns one result per city — failures are present-and-loud, never dropped.
    Mutates ``city_states`` in place (completions accumulate); persistence is the
    caller's job."""
    results: list[CityResult] = []
    for ctx in contexts:
        cs = city_states.setdefault(ctx.region, state.CityState(region=ctx.region))
        result = run_city(ctx, cs, repo_root)
        if result.status == "failed":
            _log.error(
                "batch: city=%s failed-needs-attention (%s); continuing with the rest",
                ctx.region,
                result.detail,
            )
        results.append(result)
    return results
