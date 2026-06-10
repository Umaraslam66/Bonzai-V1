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


@pytest.mark.xfail(
    reason="Task-12 TODO: bakeoff_run.sbatch's --config yaml loader is not wired yet"
)
def test_bakeoff_run_sbatch_config_flag_is_a_known_task12_gap() -> None:
    # Records (does not mask) that bakeoff_run.sbatch passes --config, which the CLI does
    # not yet support. When the Task-12 per-run config loader lands, this xpasses -> remove.
    used = _sbatch_flags(_REPO / "scripts" / "bakeoff_run.sbatch")
    assert not (used - _parser_long_flags())


# --- Task 10 (readiness-closure): --region/--release CLI + sbatch de-Singaporization ---

_SBATCH_NAMES = ["bakeoff_diagnostic.sbatch", "bakeoff_run.sbatch"]


def test_region_release_flags_reach_scaffold_config() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--region", "krakow", "--release", "2026-04-15.0"])
    cfg = mod.build_config_from_args(args)
    assert cfg.region == "krakow"
    assert cfg.release == "2026-04-15.0"
    # without the flags, the ScaffoldConfig defaults must remain untouched
    cfg_default = mod.build_config_from_args(mod._build_parser().parse_args([]))
    assert cfg_default.region == "singapore"
    assert cfg_default.release == "2026-04-15.0"


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
    args = mod._build_parser().parse_args(["--train-set", "eu-train-union"])
    cfg = mod.build_config_from_args(args)
    assert cfg.train_set == "eu-train-union"
    # without the flag, the ScaffoldConfig default must remain single-region
    cfg_default = mod.build_config_from_args(mod._build_parser().parse_args([]))
    assert cfg_default.train_set == "single"


def test_train_set_flag_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--train-set", "all-of-europe"])


# --- Task 14 (readiness-closure, F15): eval lengths are config, CLI flags override ---


def test_eval_length_flags_reach_scaffold_config() -> None:
    mod = _load_module()
    args = mod._build_parser().parse_args(["--eval-cells", "32", "--eval-max-new", "5760"])
    cfg = mod.build_config_from_args(args)
    assert cfg.eval_cells == 32
    assert cfg.eval_max_new == 5760
    # without the flags, the ScaffoldConfig defaults rule (argparse defaults are None)
    cfg_default = mod.build_config_from_args(mod._build_parser().parse_args([]))
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
