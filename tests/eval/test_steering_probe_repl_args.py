"""Replication-addendum arguments of ``build_arms`` (2026-07-19 pre-registered addendum).

The replication run passes disjoint seeds (3000..3159) and a contrast filter {C1,C4,C5}.
These tests pin: filter keeps exactly the named contrasts, seeds land verbatim and stay
PAIRED across arms, and the default path is byte-identical to the pre-addendum behavior.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "steering_probe_gen", _REPO / "scripts" / "steering_probe_gen.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()
_CHAR_MEAN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_REPL_SEEDS = tuple(range(3000, 3160))


def test_contrast_filter_keeps_exactly_the_named_contrasts():
    items = _MOD.build_arms(
        "t7",
        char_mean=_CHAR_MEAN,
        seeds=_REPL_SEEDS,
        contrasts_filter=frozenset({"C1", "C4", "C5"}),
    )
    assert {it["contrast"] for it in items} == {"C1", "C4", "C5"}
    # 3 contrasts x 2 arms x 160 seeds
    assert len(items) == 3 * 2 * 160


def test_replication_seeds_land_verbatim_and_stay_paired():
    items = _MOD.build_arms(
        "t7", char_mean=_CHAR_MEAN, seeds=_REPL_SEEDS, contrasts_filter=frozenset({"C5"})
    )
    a = sorted(it["gen_seed"] for it in items if it["arm"] == "A")
    b = sorted(it["gen_seed"] for it in items if it["arm"] == "B")
    assert a == sorted(_REPL_SEEDS)
    assert b == sorted(_REPL_SEEDS)
    # Disjoint from the main run's 2000..2039 block by construction.
    assert min(a) >= 3000


def test_default_call_unchanged_by_addendum():
    assert _MOD.build_arms("t7", char_mean=_CHAR_MEAN) == _MOD.build_arms(
        "t7", char_mean=_CHAR_MEAN, seeds=_MOD.GEN_SEEDS, contrasts_filter=None
    )
    assert len(_MOD.build_arms("t7", char_mean=_CHAR_MEAN)) == 400
