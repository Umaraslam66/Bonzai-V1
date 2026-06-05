"""Gate 6 (protocol v3): cross-reference the orchestrator's stage table against
each stage module's ACTUAL writes — marker filename, output dir, and invocation
form. Expected values are HAND-ENUMERATED from the upstream modules (file:line
cited), NOT read from stages.py — so a wrong marker/dir/arg in stages.py fails
here, before any city run. Reading the writer is the point: "_SUCCESS" for
sub_d/e/f was VERIFIED from the literals, not assumed by convention.
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.multiregion import stages
from cfm.data.multiregion.stages import StageContext

# (1) MARKER FILENAMES — hand-enumerated from each writer:
#   fetch    : overture/loader.py:71   cache_dir / "manifest.yaml"
#   sub_c    : sub_c/manifest.py:114   region_dir / "_SUCCESS"  (write_success_marker)
#   sub_d    : sub_d/manifest.py:182   (region_dir / "_SUCCESS").write_bytes(b"")
#   sub_e    : sub_e/pipeline.py:269   (output_region_dir / "_SUCCESS").touch()
#   sub_f    : sub_f/pipeline.py:250   (output_region_dir / "_SUCCESS").touch()
#   validate : sub_g/validator.py:190  (output_dir / "_PHASE1_VALIDATED").write_text(...)
EXPECTED_MARKERS = {
    "fetch": "manifest.yaml",
    "sub_c": "_SUCCESS",
    "sub_d": "_SUCCESS",
    "sub_e": "_SUCCESS",
    "sub_f": "_SUCCESS",
    "validate": "_PHASE1_VALIDATED",
}

# (3) OUTPUT-DIR flag per stage — hand-enumerated from each script's argparse:
#   sub_c    : extract_tiles.py:102            --output-dir
#   sub_d    : derive_macro_plan.py:76         --output-dir
#   sub_e    : derive_boundary_contracts.py:33 --output-region-dir
#   sub_f    : sub_f/derive.py:49              --output-region-dir
#   validate : sub_g/cli.py:73                 --output-dir
OUTPUT_FLAG = {
    "sub_c": "--output-dir",
    "sub_d": "--output-dir",
    "sub_e": "--output-region-dir",
    "sub_f": "--output-region-dir",
    "validate": "--output-dir",
}

# validate_main's required args (sub_g/cli.py:66-73 _add_common_args):
VALIDATE_REQUIRED_ARGS = {
    "--region",
    "--release",
    "--sub-c-region-dir",
    "--sub-d-region-dir",
    "--sub-e-region-dir",
    "--sub-f-region-dir",
    "--output-dir",
}


def _ctx() -> StageContext:
    return StageContext(
        region="r",
        release="2026-04-15.0",
        repo_root=Path("."),
        commit_sha="SHA",
        sub_c_dir=Path("/x/c"),
        sub_d_dir=Path("/x/d"),
        sub_e_dir=Path("/x/e"),
        sub_f_dir=Path("/x/f"),
        sub_g_dir=Path("/x/g"),
    )


def test_stage_markers_match_module_writes():
    actual = {s.name: s.marker for s in stages.STAGE_ORDER}
    assert actual == EXPECTED_MARKERS, (
        "stages.py marker beliefs drifted from the stage modules' actual writes; "
        "re-read each writer and fix stages.py (do NOT change the hand-enumerated "
        "expectation — it is the external source of truth)"
    )


def test_output_dir_matches_invocation_output_arg():
    # The dir where the orchestrator CHECKS the marker (output_dir(ctx)) must equal
    # the dir it TELLS the stage to write to (the --output* arg). Otherwise a stage
    # writes its marker where the orchestrator never looks → false "missing".
    ctx = _ctx()
    for s in stages.STAGE_ORDER:
        if s.name == "fetch":
            continue
        argv = s.argv(ctx)
        flag = OUTPUT_FLAG[s.name]
        assert flag in argv, f"{s.name} argv missing its output flag {flag}"
        out_value = argv[argv.index(flag) + 1]
        assert out_value == str(s.output_dir(ctx)), (
            f"{s.name}: argv {flag}={out_value!r} != checked output_dir "
            f"{str(s.output_dir(ctx))!r} — marker would be written where the "
            f"orchestrator never checks"
        )


def test_validate_invocation_supplies_all_required_cli_args():
    # Lock the (previously uncertain) validate invocation against the real script's
    # required args (sub_g/cli.py:66-73 _add_common_args).
    validate = next(s for s in stages.STAGE_ORDER if s.name == "validate")
    argv = validate.argv(_ctx())
    supplied = {a for a in argv if a.startswith("--")}
    assert VALIDATE_REQUIRED_ARGS <= supplied, (
        f"validate argv missing required args: {VALIDATE_REQUIRED_ARGS - supplied}"
    )
    # validate_main has neither --force nor --commit-sha
    assert "--force" not in argv and "--commit-sha" not in argv


def test_each_stage_invocation_script_exists():
    # A wrong entrypoint path = the stage never runs. fetch is in-process (empty
    # argv) — skip it.
    repo = Path(__file__).resolve().parents[3]
    for s in stages.STAGE_ORDER:
        argv = s.argv(_ctx())
        if not argv:
            continue
        script = argv[1]  # argv[0] is the python interpreter
        assert (repo / script).exists(), f"{s.name} invocation script {script!r} missing"
