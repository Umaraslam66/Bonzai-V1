"""sbatch <-> CLI contract guard (bake-off Task 4 regression).

The diagnostic job failed (exit 2, "unrecognized arguments: --backbone") because the
sbatch passed a flag the argparse never grew. This test reads the bake-off sbatch scripts
and asserts every long-flag they pass to train_scaffold.py is a recognized parser option,
so the sbatch<->CLI contract can't silently drift again.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_scaffold_cli", _REPO / "scripts" / "train_scaffold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_parser():
    return _load_module()._build_parser()


def _parser_long_flags() -> set[str]:
    flags: set[str] = set()
    for action in _build_parser()._actions:
        flags.update(opt for opt in action.option_strings if opt.startswith("--"))
    return flags


def _sbatch_flags(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    # the train_scaffold.py invocation (line-continued); grab --flags after it
    m = re.search(r"train_scaffold\.py(.*?)(?:\n[A-Za-z]|\necho|\Z)", text, re.DOTALL)
    assert m, f"no train_scaffold.py invocation in {path.name}"
    return set(re.findall(r"(--[a-z0-9-]+)", m.group(1)))


def test_diagnostic_sbatch_flags_are_all_recognized() -> None:
    known = _parser_long_flags()
    used = _sbatch_flags(_REPO / "scripts" / "bakeoff_diagnostic.sbatch")
    assert used, "expected the diagnostic sbatch to pass some flags"
    unknown = used - known
    assert not unknown, f"sbatch passes flags the CLI does not recognize: {sorted(unknown)}"


def test_backbone_flag_parses_and_overrides() -> None:
    args = _build_parser().parse_args(["--backbone", "mamba-hybrid", "--devices", "4"])
    assert args.backbone == "mamba-hybrid"


# REVERSE-LOCK (Task 26 (e), deliberate): this was the NAMED xfail
# ``test_bakeoff_run_sbatch_config_flag_is_a_known_task12_gap`` — the --config
# yaml loader is now wired into build_config_from_args, so the gap-record flips
# into the positive contract guard it was always holding a seat for.
def test_bakeoff_run_sbatch_flags_are_all_recognized() -> None:
    used = _sbatch_flags(_REPO / "scripts" / "bakeoff_run.sbatch")
    assert used, "expected the bake-off run sbatch to pass some flags"
    assert not (used - _parser_long_flags())


# --- Task 26 (e): --config per-run YAML loader -> ScaffoldConfig round-trip ---


def _write_yaml(tmp_path, payload: dict):
    p = tmp_path / "bakeoff-run.yaml"
    import yaml

    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


def test_config_yaml_round_trips_into_scaffold_config(tmp_path) -> None:
    """The per-run YAML (configs/experiments/bakeoff-*.yaml) carries the locked
    recipe; every field must land in ScaffoldConfig unchanged."""
    mod = _load_module()
    p = _write_yaml(
        tmp_path,
        {
            # region is REQUIRED (no default); a per-run bake-off YAML must name it.
            "region": "singapore",
            "backbone": "mamba-hybrid",
            "d_model": 384,
            "n_layers": 10,
            "n_heads": 12,
            "max_steps": 1234,
            "grad_accum": 4,
            "lr": 1e-4,
            "seed": 11,
        },
    )
    cfg = mod.build_config_from_args(mod._build_parser().parse_args(["--config", str(p)]))
    assert cfg.region == "singapore"  # the YAML region round-trips
    assert cfg.backbone == "mamba-hybrid"
    assert cfg.d_model == 384
    assert cfg.n_layers == 10
    assert cfg.n_heads == 12
    assert cfg.max_steps == 1234
    assert cfg.grad_accum == 4
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.seed == 11


def test_explicit_cli_flag_overrides_the_config_value(tmp_path) -> None:
    mod = _load_module()
    p = _write_yaml(tmp_path, {"region": "singapore", "d_model": 384, "max_steps": 1234})
    args = mod._build_parser().parse_args(["--config", str(p), "--d-model", "512"])
    cfg = mod.build_config_from_args(args)
    assert cfg.d_model == 512  # explicit flag wins
    assert cfg.max_steps == 1234  # untouched config value still lands


def test_config_unknown_key_is_loud_not_ignored(tmp_path) -> None:
    """pydantic ignores extra kwargs-by-default regimes elsewhere; a typo'd
    recipe field silently dropped would train the WRONG experiment — loud."""
    mod = _load_module()
    p = _write_yaml(tmp_path, {"d_modle": 384})  # typo
    args = mod._build_parser().parse_args(["--config", str(p)])
    with pytest.raises(ValueError, match="d_modle"):
        mod.build_config_from_args(args)


def test_config_refuses_topology_owned_keys(tmp_path) -> None:
    """devices/accelerator are owned by the sbatch topology (CLI defaults are
    ALWAYS applied, so a config value would be silently stomped) — refuse."""
    mod = _load_module()
    p = _write_yaml(tmp_path, {"devices": 2})
    args = mod._build_parser().parse_args(["--config", str(p)])
    with pytest.raises(ValueError, match="devices"):
        mod.build_config_from_args(args)


def test_bakeoff_run_sbatch_has_buildability_dry_run_before_srun() -> None:
    """Task 26 / readiness A-3: the preamble must prove the per-run config
    BUILDS (CPU build_backbone dry-run) before any GPU rank launches — an
    unbuildable config fails in the preamble, never after queueing 4 GPUs.

    REVERSE-LOCK (Task-26 spec review #2): the dry-run must route the YAML
    through the CLI's OWN fail-closed loader, ``_load_config_yaml`` — bare
    ``ScaffoldConfig(**yaml)`` silently ignores unknown kwargs, so a typo'd
    recipe key would survive the preamble and fail nowhere (it would train the
    WRONG experiment); a revert to the bare-pydantic dry-run goes red here."""
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    assert "build_backbone" in text, "no build_backbone buildability dry-run in the preamble"
    # "\nsrun " = the actual launch command at line start (comments mention srun earlier)
    assert text.index("build_backbone") < text.index("\nsrun "), "dry-run must precede srun"
    assert "_load_config_yaml" in text, (
        "the preamble dry-run does not use the CLI's fail-closed _load_config_yaml loader"
    )
    assert text.index("_load_config_yaml('${RUN_CONFIG}')") < text.index("\nsrun "), (
        "the fail-closed loader call must run on RUN_CONFIG in the preamble, before srun"
    )
    assert "ScaffoldConfig(**(yaml.safe_load" not in text, (
        "the bare ScaffoldConfig(**yaml) dry-run is back — it silently ignores unknown keys"
    )


def test_bakeoff_run_sbatch_dry_run_vets_the_trained_backbone() -> None:
    """REVERSE-LOCK (Task-26 quality review #1): srun passes --backbone
    "$BACKBONE", an explicit CLI flag that OVERRIDES the YAML value — so the
    dry-run must apply the SAME override ({**yaml, 'backbone': '${BACKBONE}'}).
    A bare ScaffoldConfig(**yaml) dry-run on a YAML that OMITS `backbone`
    builds the default transformer while the job trains mamba — the exact
    missing-kernels regime the gate's comment claims to catch. And a YAML
    naming a DIFFERENT backbone than $BACKBONE is a config/submission
    mismatch: a loud refusal, never a silent stomp."""
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    override = "'backbone': '${BACKBONE}'"
    assert override in text, (
        "the dry-run does not apply the srun --backbone override — a YAML omitting "
        "`backbone` gates the DEFAULT backbone, not the one the job will train"
    )
    assert text.index(override) < text.index("\nsrun "), (
        "the backbone-overridden dry-run must run in the preamble, before srun"
    )
    assert "CONFIG_BACKBONE_MISMATCH" in text, (
        "no loud refusal for a YAML backbone that contradicts $BACKBONE — a "
        "config/submission mismatch must refuse, never silently stomp the YAML"
    )
    assert "ScaffoldConfig(**_load_config_yaml" not in text, (
        "the un-overridden ScaffoldConfig(**yaml) dry-run is back — it builds the "
        "YAML/default backbone, not the trained ($BACKBONE-overridden) one"
    )


# --- Task 10 (readiness-closure): --region/--release CLI + sbatch de-Singaporization ---

_SBATCH_NAMES = ["bakeoff_diagnostic.sbatch", "bakeoff_run.sbatch"]


def test_region_release_flags_reach_scaffold_config() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--region", "krakow", "--release", "2026-04-15.0"])
    cfg = mod.build_config_from_args(args)
    assert cfg.region == "krakow"
    assert cfg.release == "2026-04-15.0"
    # release still defaults when its flag is absent (region supplied so the build proceeds)
    cfg_rel_default = mod.build_config_from_args(
        mod._build_parser().parse_args(["--region", "krakow"])
    )
    assert cfg_rel_default.release == "2026-04-15.0"


def test_no_region_fails_loud_no_silent_singapore_default() -> None:
    # Fail-closed (the only NEW behavior): region has no default, so a run that names
    # no --region (and no region in --config) must die LOUDLY at config-build time with
    # a clear message — never silently fall back to the retired Phase-1 singapore corpus.
    mod = _load_module()
    args = mod._build_parser().parse_args([])
    with pytest.raises(SystemExit, match="region is required"):
        mod.build_config_from_args(args)


@pytest.mark.parametrize("name", _SBATCH_NAMES)
def test_sbatch_has_no_singapore_literal(name: str) -> None:
    text = (_REPO / "scripts" / name).read_text(encoding="utf-8")
    assert "singapore" not in text.lower(), f"{name} still hardcodes singapore"


@pytest.mark.parametrize("name", _SBATCH_NAMES)
def test_sbatch_prebuild_line_is_env_driven(name: str) -> None:
    text = (_REPO / "scripts" / name).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "build_training_shards" in ln]
    assert lines, f"{name} has no build_training_shards pre-build line"
    for line in lines:
        assert re.search(r"\$\{?RELEASE\}?", line), f"{name} pre-build line not RELEASE-driven"
        assert re.search(r"\$\{?REGION\}?", line), f"{name} pre-build line not REGION-driven"


@pytest.mark.parametrize("name", _SBATCH_NAMES)
def test_sbatch_has_region_release_env_guards(name: str) -> None:
    text = (_REPO / "scripts" / name).read_text(encoding="utf-8")
    assert ': "${REGION:?' in text, f"{name} missing the REGION env guard"
    assert ': "${RELEASE:?' in text, f"{name} missing the RELEASE env guard"


@pytest.mark.parametrize("name", _SBATCH_NAMES)
def test_sbatch_srun_passes_region_release_to_cli(name: str) -> None:
    # Env-driven sbatch must forward REGION/RELEASE into the job, else ScaffoldConfig
    # silently falls back to its defaults inside the srun.
    flags = _sbatch_flags(_REPO / "scripts" / name)
    assert "--region" in flags, f"{name} srun does not pass --region"
    assert "--release" in flags, f"{name} srun does not pass --release"


# --- Task 11 (readiness-closure): --train-set CLI + TRAIN_SET-aware run sbatch ---


def test_train_set_flag_reaches_scaffold_config() -> None:
    mod = _load_module()
    # --region supplied (now REQUIRED); this test's subject is the --train-set flag/default
    args = mod._build_parser().parse_args(
        ["--region", "singapore", "--train-set", "eu-train-union"]
    )
    cfg = mod.build_config_from_args(args)
    assert cfg.train_set == "eu-train-union"
    # without the flag, the ScaffoldConfig default must remain single-region
    cfg_default = mod.build_config_from_args(
        mod._build_parser().parse_args(["--region", "singapore"])
    )
    assert cfg_default.train_set == "single"


def test_train_set_flag_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--train-set", "all-of-europe"])


# --- Task 14 (readiness-closure, F15): eval lengths are config, CLI flags override ---


def test_eval_length_flags_reach_scaffold_config() -> None:
    mod = _load_module()
    # --region supplied (now REQUIRED); this test's subject is the eval-length flags/defaults
    args = mod._build_parser().parse_args(
        ["--region", "singapore", "--eval-cells", "32", "--eval-max-new", "5760"]
    )
    cfg = mod.build_config_from_args(args)
    assert cfg.eval_cells == 32
    assert cfg.eval_max_new == 5760
    # without the flags, the ScaffoldConfig defaults rule (argparse defaults are None)
    cfg_default = mod.build_config_from_args(
        mod._build_parser().parse_args(["--region", "singapore"])
    )
    assert cfg_default.eval_cells == 64
    assert cfg_default.eval_max_new == 512


def test_run_sbatch_has_defaulted_train_set_guard_and_forwards_flag() -> None:
    # The Task-10 reviewer's deferred ${TRAIN_SET} guard: DEFAULTED (:=single), not a
    # :? hard guard — single-region remains the default submit path. The srun must
    # forward --train-set or the job silently falls back to the config default.
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    assert ': "${TRAIN_SET:=single}"' in text, "missing the defaulted TRAIN_SET guard"
    flags = _sbatch_flags(_REPO / "scripts" / "bakeoff_run.sbatch")
    assert "--train-set" in flags, "bakeoff_run.sbatch srun does not pass --train-set"


# --- Task 19 (readiness-closure, F8): USR1 forward + verified resubmit + end-state markers ---


def test_run_sbatch_usr1_trap_forwards_signal_to_captured_srun_pid() -> None:
    # Slurm's --signal=B:USR1 delivers ONLY to the batch shell; without an explicit
    # forward the srun ranks never see it and get SIGKILLed at the wall instead of
    # checkpointing + exiting cleanly. The trap must kill -USR1 a CAPTURED srun PID.
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    assert "SRUN_PID=$!" in text, "srun PID not captured ($! after the backgrounded srun)"
    assert re.search(r'kill -USR1 "\$SRUN_PID"', text), "trap does not forward USR1 to srun"


def test_run_sbatch_resubmit_is_verified_not_silent() -> None:
    # A failed `sbatch` in the trap must NOT exit 0 (silent resume-chain break — this
    # project's false-completion class). --parsable makes sbatch emit a bare job id
    # (not "Submitted batch job NNNN" prose), and the check verifies the KIND of yes:
    # a NUMERIC job id, not just any nonempty output. A failure exits 1. --export=ALL
    # so BACKBONE/SCALE/REGION/RELEASE/TRAIN_SET propagate.
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    assert 'JID=$(sbatch --parsable --export=ALL "$0")' in text, (
        "resubmit JID not captured via sbatch --parsable"
    )
    assert '[[ "$JID" =~ ^[0-9]+ ]]' in text, (
        "no numeric-JID check (a bare nonempty check would accept sbatch prose/garbage)"
    )
    assert re.search(r"JOB_FAILED_RESUBMIT.*?\bexit 1\b", text, re.DOTALL), (
        "the resubmit-failure branch does not fail loudly (exit 1)"
    )
    assert 'sbatch "$0"' not in text, "the old unverified resubmit is still present"


def test_run_sbatch_failed_resubmit_still_grace_waits_for_ckpt_flush() -> None:
    # Even when the resubmit is broken, the in-flight last.ckpt flush is exactly what
    # the operator's MANUAL resubmit will resume from; the failure branch must wait for
    # the srun ranks before its exit 1, never kill the flush by exiting immediately.
    m = re.search(
        r"JOB_FAILED_RESUBMIT(.*?)\bexit 1\b",
        (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert m, "no JOB_FAILED_RESUBMIT failure branch found"
    assert 'wait "$SRUN_PID"' in m.group(1), (
        "JOB_FAILED_RESUBMIT branch exits 1 without grace-waiting for the srun ranks"
    )


@pytest.mark.parametrize("name", _SBATCH_NAMES)
def test_sbatch_job_done_only_after_verified_endstate(name: str) -> None:
    # JOB_DONE is a TRUSTED marker (future sessions believe it); it must follow proof
    # that the expected checkpoint + report artifacts exist, not mere control-flow
    # reaching the last line. Failure path prints JOB_FAILED_ENDSTATE and exits 1.
    text = (_REPO / "scripts" / name).read_text(encoding="utf-8")
    assert "JOB_FAILED_ENDSTATE" in text, f"{name} has no loud end-state failure marker"
    assert text.count('"JOB_DONE"') == 1, f"{name} must print JOB_DONE exactly once"
    call = re.search(r"^verify_endstate\s*$", text, re.MULTILINE)
    assert call, f"{name} never CALLS verify_endstate (a definition alone proves nothing)"
    assert call.start() < text.index('echo "JOB_DONE"'), (
        f"{name}: JOB_DONE is printed before the end-state verification"
    )


def test_diagnostic_sbatch_missing_report_fails_loudly() -> None:
    # The `|| echo "(report not found)"` shrug masked a missing report as a passing
    # job; a missing report must fail the job loudly (verify_endstate + set -euo pipefail).
    text = (_REPO / "scripts" / "bakeoff_diagnostic.sbatch").read_text(encoding="utf-8")
    assert "(report not found)" not in text, "missing-report masking is still present (F8)"


# --- Phase-2 bake-off Task 1: gcc/12.2.0 toolchain on the compiled scored path ---


def test_bakeoff_run_sbatch_loads_gcc12_and_preloads_libstdcxx() -> None:
    """The compiled scored path (torch.compile inductor CPU codegen) needs gcc/12.2.0,
    not the RHEL-8 gcc-8.5 that crashed eval at Step 18.5; and the mamba run needs the
    gcc-12 libstdc++ at import. Shared run sbatch (parameterized by --backbone), so one
    fix serves both backbones; the LD_PRELOAD is harmless for transformer-ar."""
    text = (_REPO / "scripts" / "bakeoff_run.sbatch").read_text(encoding="utf-8")
    assert "module load python/3.11.7 cuda/12.2 gcc/12.2.0" in text
    assert "export CC=" in text and "CXX=" in text and "CUDAHOSTCXX=" in text
    assert "LD_PRELOAD" in text and "libstdc++.so.6" in text
