"""Unit tests for scripts/steering_probe_gen.py ``build_arms`` (SPEC 2026-07-17).

The load-bearing guarantee of the steering probe is "everything else identical": the two arms
of a contrast must differ in EXACTLY the swapped conditioning field(s) and nothing else, and
the generation seeds must be PAIRED (identical multiset across arms). These are pure-construction
facts — proven here with NO torch weights and NO GPU (only ``build_value_bearing_prefix`` and
``character_stats_for_cell`` are touched, transitively, via the module import).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
# A deterministic stand-in char_mean (7-dim, matching character_stats_for_cell output width);
# its exact values are irrelevant to the structural assertions below.
_CHAR_MEAN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_CKPT = "transformer-ar-seed7"

# Prefix positions that each contrast is allowed to differ at (0-indexed in the 10-id prefix).
# pos0=pop_density, pos1=zoning, pos2=road_skeleton, pos3=cell_density (conditioning field order).
_EXPECTED_DIFF_POSITIONS = {
    "C1": {2},
    "C2": {3},
    "C3": {0, 2, 3},
    "C4": set(),  # char-only: prefixes identical
    "C5": {2},
}


@pytest.fixture(scope="module")
def arms():
    return _MOD.build_arms(_CKPT, char_mean=_CHAR_MEAN)


def _by_contrast(arms):
    out: dict[str, dict[str, list[dict]]] = {}
    for it in arms:
        out.setdefault(it["contrast"], {}).setdefault(it["arm"], []).append(it)
    return out


def test_total_item_count(arms):
    # 5 contrasts x 2 arms x 40 paired seeds = 400 items per checkpoint (spec: 1200 / 3 ckpts).
    assert len(arms) == 400


def test_every_item_carries_spec_fields(arms):
    for it in arms:
        assert it["ckpt_id"] == _CKPT
        assert it["arm"] in {"A", "B"}
        assert len(it["prefix"]) == 10
        assert it["prefix"][-1] == _MOD.CHARACTER_PLACEHOLDER_ID
        assert isinstance(it["char_stats"], list) and len(it["char_stats"]) == 7
        assert 2000 <= it["gen_seed"] <= 2039


def test_arms_differ_only_at_spec_positions(arms):
    """(a) prefix diff positions AND char_stats equal/differ exactly per spec."""
    by = _by_contrast(arms)
    for contrast, expected_pos in _EXPECTED_DIFF_POSITIONS.items():
        arm_a = sorted(by[contrast]["A"], key=lambda x: x["gen_seed"])
        arm_b = sorted(by[contrast]["B"], key=lambda x: x["gen_seed"])
        # Prefixes are constant within an arm (only the gen_seed varies across the 40 items).
        assert len({tuple(x["prefix"]) for x in arm_a}) == 1
        assert len({tuple(x["prefix"]) for x in arm_b}) == 1
        pa = arm_a[0]["prefix"]
        pb = arm_b[0]["prefix"]
        diff = {i for i in range(len(pa)) if pa[i] != pb[i]}
        assert diff == expected_pos, f"{contrast}: prefix diff {diff} != expected {expected_pos}"

        # char_stats: C4 is the ONLY contrast whose arms differ in char_stats; the macro-swap
        # contrasts (C1/C2/C3/C5) hold char_stats FIXED across their two arms.
        ca = arm_a[0]["char_stats"]
        cb = arm_b[0]["char_stats"]
        if contrast == "C4":
            assert ca != cb, "C4 arms must differ in char_stats"
            # ...and C4 prefixes must be identical (char-only contrast).
            assert diff == set()
        else:
            assert ca == cb, f"{contrast} arms must hold char_stats fixed"


def test_c5_uses_char_mean(arms):
    """C5 is the char-ablated skeleton contrast: both arms carry the injected char_mean."""
    by = _by_contrast(arms)
    for arm in ("A", "B"):
        for it in by["C5"][arm]:
            assert it["char_stats"] == _CHAR_MEAN


def test_paired_seeds_identical_across_arms(arms):
    """(b) gen_seed multiset is identical across arms and equals 2000..2039 for every contrast."""
    by = _by_contrast(arms)
    expected = sorted(range(2000, 2040))
    for contrast in _EXPECTED_DIFF_POSITIONS:
        seeds_a = sorted(x["gen_seed"] for x in by[contrast]["A"])
        seeds_b = sorted(x["gen_seed"] for x in by[contrast]["B"])
        assert seeds_a == expected
        assert seeds_b == expected


def test_deterministic_across_two_calls():
    a = _MOD.build_arms(_CKPT, char_mean=_CHAR_MEAN)
    b = _MOD.build_arms(_CKPT, char_mean=_CHAR_MEAN)
    assert a == b


def test_sharding_is_partition(arms):
    """(c) union of shards 0..N-1 == full list, disjoint, deterministic across two calls."""
    n = 4
    shards = [_MOD.shard_items(arms, k, n) for k in range(n)]

    # Disjoint: no work item appears in two shards (key on the full identity tuple).
    def key(it):
        return (it["ckpt_id"], it["contrast"], it["arm"], it["gen_seed"])

    seen: set = set()
    for sh in shards:
        for it in sh:
            kk = key(it)
            assert kk not in seen, f"item {kk} in more than one shard"
            seen.add(kk)

    # Union reproduces the full list (as a set of identities) and total count.
    assert sum(len(sh) for sh in shards) == len(arms)
    assert seen == {key(it) for it in arms}

    # Balanced: 400 / 4 = 100 items each (exact here since 400 % 4 == 0).
    assert [len(sh) for sh in shards] == [100, 100, 100, 100]

    # Deterministic across two calls.
    again = [_MOD.shard_items(arms, k, n) for k in range(n)]
    assert shards == again


def test_shard_index_out_of_range(arms):
    with pytest.raises(ValueError):
        _MOD.shard_items(arms, 4, 4)
