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


def _build_parser():
    spec = importlib.util.spec_from_file_location(
        "train_scaffold_cli", _REPO / "scripts" / "train_scaffold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_parser()


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
