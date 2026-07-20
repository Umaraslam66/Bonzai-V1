"""Task 3 — generation CLI (``scripts/realism_eval_gen.py``) unit surface.

These tests exercise the torch-FREE surface of the CLI: argument parsing, the
dry-run ``--stratum``/``--limit-cells`` cell filter over synthetic
``ConditionedCell``s, and ablation resolution (checkpoint ablation must match the
``--ablation`` arg, or ``SystemExit``). The real GPU generation path
(``dist.init_process_group`` -> ``score_cell`` -> ``run_generation``) is validated
by the ops dry-run (T8), never here — so importing the module must not pull torch.
"""

from __future__ import annotations

import pytest

import scripts.realism_eval_gen as gen
from cfm.eval.realism_driver.conditioning import ConditionedCell


def _cell(i: int, *, city: str = "glasgow", density_bucket: int = 3) -> ConditionedCell:
    return ConditionedCell(
        cell_key=(city, 0, 0, i, 0),
        density_bucket=density_bucket,
        prefix_ids=tuple(range(10)),
        char_stats=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        real_body_tokens=(11, 12, 13),
    )


def test_module_is_torch_free():
    """Importing the CLI must not pull torch (arg-parse + filter stay GPU-free)."""
    assert not hasattr(gen, "torch")


def test_arg_parser_defaults():
    """Required args parse; the orchestrator-reviewed defaults hold."""
    args = gen.build_arg_parser().parse_args(
        ["--ckpt", "c.ckpt", "--manifest", "m.json", "--out", "o.json"]
    )
    assert args.ckpt == "c.ckpt"
    assert args.manifest == "m.json"
    assert args.out == "o.json"
    assert args.release == gen.DEFAULT_RELEASE
    assert args.base_seed == gen.DEFAULT_BASE_SEED
    # DECISION (orchestrator 2026-07-20): 4096, NOT the 13312 context cap.
    assert args.max_new == 4096
    assert args.ablation == gen.READ_FROM_CKPT
    assert args.dry_run is False
    assert args.stratum is None
    assert args.limit_cells is None


def test_arg_parser_accepts_dry_run_filters():
    args = gen.build_arg_parser().parse_args(
        [
            "--ckpt",
            "c.ckpt",
            "--manifest",
            "m.json",
            "--out",
            "o.json",
            "--dry-run",
            "--stratum",
            "2",
            "--limit-cells",
            "5",
            "--max-new",
            "64",
        ]
    )
    assert args.dry_run is True
    assert args.stratum == "2"
    assert args.limit_cells == 5
    assert args.max_new == 64


def test_filter_by_stratum_selects_density_bucket():
    cells = [
        _cell(0, city="glasgow", density_bucket=1),
        _cell(1, city="glasgow", density_bucket=3),
        _cell(2, city="munich", density_bucket=3),
        _cell(3, city="krakow", density_bucket=1),
    ]
    got = gen.filter_cells(cells, stratum="3")
    assert [c.cell_key for c in got] == [cells[1].cell_key, cells[2].cell_key]


def test_filter_limit_cells_takes_prefix_in_order():
    cells = [_cell(i) for i in range(6)]
    got = gen.filter_cells(cells, limit_cells=4)
    assert got == cells[:4]


def test_filter_stratum_and_limit_compose():
    cells = [
        _cell(0, density_bucket=3),
        _cell(1, density_bucket=1),
        _cell(2, density_bucket=3),
        _cell(3, density_bucket=3),
    ]
    got = gen.filter_cells(cells, stratum="3", limit_cells=2)
    assert got == [cells[0], cells[2]]


def test_filter_no_filters_is_identity():
    cells = [_cell(i) for i in range(3)]
    assert gen.filter_cells(cells) == cells


def test_resolve_ablation_read_from_ckpt_uses_checkpoint():
    assert gen.resolve_ablation(gen.READ_FROM_CKPT, "no_character") == "no_character"


def test_resolve_ablation_match_returns_value():
    assert gen.resolve_ablation("full", "full") == "full"


def test_resolve_ablation_mismatch_raises_systemexit():
    with pytest.raises(SystemExit):
        gen.resolve_ablation("full", "no_character")


@pytest.mark.parametrize("extra", [["--stratum", "3"], ["--limit-cells", "5"]])
def test_filters_without_dry_run_refused(extra):
    """--stratum/--limit-cells are dry-run-only: main() aborts BEFORE any torch import."""
    argv = ["--ckpt", "c.ckpt", "--manifest", "m.json", "--out", "o.json", *extra]
    with pytest.raises(SystemExit, match="dry-run-only"):
        gen.main(argv)


def test_check_world_size_refuses_single_process_scored_run():
    """WORLD_SIZE<2 without --dry-run is a hard abort (full-node discipline)."""
    with pytest.raises(SystemExit, match="WORLD_SIZE<2"):
        gen.check_world_size(1, dry_run=False)


def test_check_world_size_allows_dry_run_and_multiproc():
    gen.check_world_size(1, dry_run=True)  # dry-run smoke: allowed
    gen.check_world_size(4, dry_run=False)  # full node: allowed
    gen.check_world_size(2, dry_run=True)  # multiproc dry-run: allowed
