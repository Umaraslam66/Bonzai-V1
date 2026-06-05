"""Orchestrator integration against REAL materialized artifacts.

Dynamic complement to C1's static Gate-6: stages.py's (output_dir, marker)
beliefs must match the artifacts a real run actually produced on disk. Parametrized
over singapore (materialized locally) and berlin (Leonardo-only → skips locally).
Fast (file-existence + git), so unmarked with a skip-if-absent guard rather than
@slow."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.multiregion import driver, state
from cfm.data.multiregion.stages import STAGE_ORDER, StageContext

_REPO = Path(__file__).resolve().parents[3]
_REL = "2026-04-15.0"


def _ctx(region: str) -> StageContext:
    base = _REPO / "data" / "processed"
    return StageContext(
        region=region,
        release=_REL,
        repo_root=_REPO,
        commit_sha="HEAD",
        sub_c_dir=base / "sub_c" / _REL / region,
        sub_d_dir=base / "sub_d" / _REL / region,
        sub_e_dir=base / "sub_e" / _REL / region,
        sub_f_dir=base / "sub_f" / _REL / region,
        sub_g_dir=base / "sub_g" / _REL / region,
    )


def _materialized(ctx: StageContext) -> bool:
    return (ctx.sub_g_dir / "_PHASE1_VALIDATED").exists()


@pytest.mark.parametrize("region", ["singapore", "berlin"])
def test_stage_contract_matches_real_artifacts(region):
    # Dynamic Gate-6: for every processing stage, stages.py's output_dir(ctx)/marker
    # must exist among the artifacts a real run produced — belief vs disk reality.
    ctx = _ctx(region)
    if not _materialized(ctx):
        pytest.skip(f"{region} not fully materialized locally (lives on Leonardo)")
    for stage in STAGE_ORDER:
        if stage.name == "fetch":
            continue  # cache dir, not data/processed
        assert stage.output_dir is not None
        marker = stage.output_dir(ctx) / stage.marker
        assert marker.exists(), (
            f"{region}/{stage.name}: stages.py expects {stage.marker} at {marker}, "
            f"missing on disk — stage-table belief drifted from a real run"
        )


@pytest.mark.parametrize("region", ["singapore", "berlin"])
def test_resume_runs_nothing_when_complete_and_code_unchanged(region):
    # The orchestrator's resume path against real dirs + the real git repo: with
    # every stage stamped at HEAD and no source change, nothing re-runs.
    ctx = _ctx(region)
    if not _materialized(ctx):
        pytest.skip(f"{region} not fully materialized locally (lives on Leonardo)")
    head = driver._head_sha(_REPO)
    cs = state.CityState(
        region=region,
        completions={s.name: state.StageCompletion(stage=s.name, sha=head) for s in STAGE_ORDER},
    )
    assert state.stages_to_run(cs, head, STAGE_ORDER, _REPO) == []
