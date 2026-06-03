"""invalidate-on-fix: a source change to stage N re-runs N + downstream, NOT
upstream. Uses a REAL temp git repo so the git-diff mechanism itself is exercised
(an under-inclusive glob is catchable here, not just the cascade logic)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cfm.data.multiregion import stages, state

# One tracked file per path referenced by STAGE_ORDER's source_globs, so a real
# `git diff -- <glob>` resolves. Mirrors the real repo layout.
_REPO_FILES = (
    "scripts/extract_tiles.py",
    "scripts/derive_macro_plan.py",
    "scripts/derive_boundary_contracts.py",
    "scripts/sub_f/derive.py",
    "scripts/sub_g/validate_phase1_region.py",
    "src/cfm/data/sub_c/coords.py",
    "src/cfm/data/sub_c/x.py",
    "src/cfm/data/sub_d/x.py",
    "src/cfm/data/sub_e/x.py",
    "src/cfm/data/sub_f/x.py",
    "src/cfm/data/sub_g/x.py",
    "src/cfm/data/overture/x.py",
    "src/cfm/data/io.py",
    "src/cfm/data/determinism.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    for rel in _REPO_FILES:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _all_complete(sha: str) -> state.CityState:
    return state.CityState(
        region="x",
        completions={
            s.name: state.StageCompletion(stage=s.name, sha=sha) for s in stages.STAGE_ORDER
        },
    )


def _commit_change(repo: Path, rel: str, msg: str) -> str:
    (repo / rel).write_text("v2 (changed)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_changing_sub_e_source_reruns_sub_e_and_downstream_not_upstream(tmp_path):
    repo = _make_repo(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = _all_complete(sha_a)
    sha_b = _commit_change(repo, "src/cfm/data/sub_e/x.py", "fix sub_e regime")

    to_run = state.stages_to_run(cs, sha_b, stages.STAGE_ORDER, repo)
    assert to_run == ["sub_e", "sub_f", "validate"], (
        f"expected sub_e + downstream to re-run, upstream untouched; got {to_run}"
    )


def test_unchanged_head_reruns_nothing(tmp_path):
    repo = _make_repo(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = _all_complete(sha_a)
    assert state.stages_to_run(cs, sha_a, stages.STAGE_ORDER, repo) == []


def test_missing_completion_runs_that_stage_and_downstream(tmp_path):
    repo = _make_repo(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = state.CityState(
        region="x",
        completions={
            s.name: state.StageCompletion(stage=s.name, sha=sha_a)
            for s in stages.STAGE_ORDER
            if s.name not in {"sub_f", "validate"}
        },
    )
    assert state.stages_to_run(cs, sha_a, stages.STAGE_ORDER, repo) == ["sub_f", "validate"]


def test_changing_shared_coords_cascades_to_all_processing_stages(tmp_path):
    # coords.py lives under sub_c/ — sub_c globs it, so a coords change flags sub_c
    # and the cascade re-runs every downstream processing stage. fetch (overture
    # only) is NOT flagged. This is the shared-module-via-cascade guarantee.
    repo = _make_repo(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = _all_complete(sha_a)
    sha_b = _commit_change(repo, "src/cfm/data/sub_c/coords.py", "coords change")
    to_run = state.stages_to_run(cs, sha_b, stages.STAGE_ORDER, repo)
    assert to_run == ["sub_c", "sub_d", "sub_e", "sub_f", "validate"]


def test_changing_overture_flags_fetch_and_everything(tmp_path):
    # overture is fetch's source AND sub_c's; a change flags fetch (the earliest),
    # cascading through the entire chain.
    repo = _make_repo(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = _all_complete(sha_a)
    sha_b = _commit_change(repo, "src/cfm/data/overture/x.py", "overture change")
    to_run = state.stages_to_run(cs, sha_b, stages.STAGE_ORDER, repo)
    assert to_run == ["fetch", "sub_c", "sub_d", "sub_e", "sub_f", "validate"]


def test_city_state_roundtrips(tmp_path):
    p = tmp_path / "s.json"
    cs = state.CityState(
        region="berlin",
        completions={"sub_c": state.StageCompletion(stage="sub_c", sha="abc")},
    )
    state.save_city_state(p, cs)
    assert state.load_city_state(p, "berlin") == cs
