"""Driver behaviors, all must-fail-proven:
- exit-0-without-marker registers as FAILURE (returncode alone is not trusted);
- a stage failure marks the city failed (NOT validated) and is not stamped;
- a successful stage is re-stamped with the CURRENT head sha;
- run_batch isolates one city's failure and keeps going (continue-but-loud).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.multiregion import driver, extract_lock, state
from cfm.data.multiregion.stages import Stage, StageContext


def _ctx(root: Path, region: str = "x") -> StageContext:
    return StageContext(
        region=region,
        release="2026-04-15.0",
        repo_root=root,
        commit_sha="SHA",
        sub_c_dir=root / "c",
        sub_d_dir=root / "d",
        sub_e_dir=root / "e",
        sub_f_dir=root / "f",
        sub_g_dir=root / "g",
    )


def test_no_stages_to_run_returns_validated_without_running(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: [])
    called: list[str] = []
    monkeypatch.setattr(driver, "_run_fetch", lambda ctx: called.append("fetch"))
    monkeypatch.setattr(driver, "_run_stage_subprocess", lambda s, c: called.append(s.name))
    cs = state.CityState(region="x")
    result = driver.run_city(_ctx(tmp_path), cs, repo_root=tmp_path)
    assert result.status == "validated"
    assert called == []


def test_successful_stage_is_stamped_with_current_head_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEADX")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])
    monkeypatch.setattr(driver, "_run_stage_subprocess", lambda s, c: None)  # "succeeds"
    cs = state.CityState(region="x")
    result = driver.run_city(_ctx(tmp_path), cs, repo_root=tmp_path)
    assert result.status == "validated"
    assert cs.completions["sub_c"].sha == "HEADX"  # re-stamped with the CURRENT sha


def test_stage_failure_marks_city_failed_not_validated_and_unstamped(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])

    def boom(stage, ctx):
        raise RuntimeError("regime: multipolygon explosion")

    monkeypatch.setattr(driver, "_run_stage_subprocess", boom)
    cs = state.CityState(region="x")
    result = driver.run_city(_ctx(tmp_path), cs, repo_root=tmp_path)
    assert result.status == "failed"
    assert result.status != "validated"
    assert "regime" in result.detail
    assert "sub_c" not in cs.completions  # a failed stage is NOT stamped


def test_exit_zero_without_marker_is_a_failure(tmp_path, monkeypatch):
    # returncode==0 is NOT trusted: a stage that returns 0 but writes no marker
    # must raise → the city is failed, never silently 'validated'.
    out = tmp_path / "out"
    out.mkdir()
    stage = Stage(
        name="sub_c",
        marker="_SUCCESS",
        output_dir=lambda c: out,
        argv=lambda c: ["true"],
        source_globs=("x",),
    )
    # subprocess.run returns (rc 0) without raising AND without writing the marker.
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="returned 0 but wrote no"):
        driver._run_stage_subprocess(stage, _ctx(tmp_path))


def test_run_city_refuses_when_another_extract_holds_the_lock(tmp_path, monkeypatch):
    # The double-nohup near-miss class: a second concurrent extract of the SAME
    # city must refuse IMMEDIATELY and run NO stages (never touch the corpus).
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])
    ran: list[str] = []
    monkeypatch.setattr(driver, "_run_stage_subprocess", lambda s, c: ran.append(s.name))

    held = extract_lock.acquire_city_lock(extract_lock.city_lock_path(tmp_path, "x"))
    try:
        result = driver.run_city(_ctx(tmp_path), state.CityState(region="x"), repo_root=tmp_path)
    finally:
        held.close()

    assert result.status == "failed"
    assert "lock" in result.detail.lower() or "concurrent" in result.detail.lower()
    assert ran == []  # NO stage ran while another extract held the lock


def test_run_city_releases_lock_after_success(tmp_path, monkeypatch):
    # A successful run must RELEASE the lock so the next run can re-acquire it.
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEADX")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])
    monkeypatch.setattr(driver, "_run_stage_subprocess", lambda s, c: None)  # "succeeds"
    result = driver.run_city(_ctx(tmp_path), state.CityState(region="x"), repo_root=tmp_path)
    assert result.status == "validated"
    # re-acquire proves the lock was released (raises if still held)
    extract_lock.acquire_city_lock(extract_lock.city_lock_path(tmp_path, "x")).close()


def test_run_city_releases_lock_after_stage_failure(tmp_path, monkeypatch):
    # Even on stage failure the lock must be released (finally), not leaked.
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])

    def boom(stage, ctx):
        raise RuntimeError("regime: explosion")

    monkeypatch.setattr(driver, "_run_stage_subprocess", boom)
    result = driver.run_city(_ctx(tmp_path), state.CityState(region="x"), repo_root=tmp_path)
    assert result.status == "failed"
    extract_lock.acquire_city_lock(extract_lock.city_lock_path(tmp_path, "x")).close()  # released


def test_run_batch_isolates_failure_and_continues(tmp_path, monkeypatch):
    def fake_run_city(ctx, cs, repo_root):
        status = "failed" if ctx.region == "bad" else "validated"
        return driver.CityResult(
            region=ctx.region, status=status, detail="regime" if status == "failed" else ""
        )

    monkeypatch.setattr(driver, "run_city", fake_run_city)
    ctxs = [_ctx(tmp_path, "good1"), _ctx(tmp_path, "bad"), _ctx(tmp_path, "good2")]
    results = driver.run_batch(ctxs, {}, repo_root=tmp_path)

    by = {r.region: r.status for r in results}
    assert by == {"good1": "validated", "bad": "failed", "good2": "validated"}
    # the bad city did NOT stop good2, and it is present-and-loud (not dropped).
    assert any(r.region == "bad" and r.status == "failed" for r in results)
    assert sum(1 for r in results if r.status == "validated") == 2
