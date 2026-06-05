from __future__ import annotations

from pathlib import Path

from cfm.data.multiregion import stages
from cfm.data.multiregion.stages import StageContext


def _ctx() -> StageContext:
    return StageContext(
        region="r",
        release="2026-04-15.0",
        repo_root=Path("."),
        commit_sha="SHA",
        sub_c_dir=Path("c"),
        sub_d_dir=Path("d"),
        sub_e_dir=Path("e"),
        sub_f_dir=Path("f"),
        sub_g_dir=Path("g"),
    )


def test_stage_order_is_the_full_chain():
    assert [s.name for s in stages.STAGE_ORDER] == [
        "fetch",
        "sub_c",
        "sub_d",
        "sub_e",
        "sub_f",
        "validate",
    ]


def test_each_stage_has_nonempty_source_globs():
    for s in stages.STAGE_ORDER:
        assert s.source_globs, f"{s.name} has empty source_globs (invalidate-on-fix blind)"


def test_all_source_globs_point_at_real_paths():
    # A typo'd glob silently never matches → under-globbing (a fix wouldn't
    # re-trigger the stage). Assert every glob resolves to a real file/dir.
    repo = Path(__file__).resolve().parents[3]
    for s in stages.STAGE_ORDER:
        for g in s.source_globs:
            assert (repo / g).exists(), (
                f"{s.name} source_glob {g!r} does not exist under {repo} — typo'd "
                f"globs silently never match (under-globbing)"
            )


def test_coords_covered_by_sub_c_glob_in_importing_stages():
    # coords.py lives under sub_c/. sub_c and sub_e import sub_c, so a coords change
    # flags them directly; sub_d/sub_f/validate get it via the cascade (see test_state).
    for name in ("sub_c", "sub_e"):
        s = next(x for x in stages.STAGE_ORDER if x.name == name)
        assert any("sub_c" in g for g in s.source_globs), (
            f"{name} must glob src/cfm/data/sub_c/ so a coords change flags it"
        )


def test_io_in_every_processing_stage_glob():
    # Every processing stage (not fetch) imports io; the earliest importer must be
    # flagged on a shared-io change so the cascade re-runs everything downstream.
    for name in ("sub_c", "sub_d", "sub_e", "sub_f", "validate"):
        s = next(x for x in stages.STAGE_ORDER if x.name == name)
        assert any("io.py" in g for g in s.source_globs), f"{name} missing io.py glob"


def test_validate_uses_wrapper_script_not_dash_m():
    # sub_g/cli.py has no __main__/subcommand router; the thin wrapper is the entrypoint.
    validate = next(s for s in stages.STAGE_ORDER if s.name == "validate")
    argv = validate.argv(_ctx())
    assert "scripts/sub_g/validate_phase1_region.py" in argv
    assert "-m" not in argv


def test_commit_sha_only_on_sub_d():
    ctx = _ctx()
    for s in stages.STAGE_ORDER:
        present = "--commit-sha" in s.argv(ctx)
        assert present == (s.name == "sub_d"), (
            f"{s.name}: --commit-sha present={present}, expected only on sub_d"
        )
