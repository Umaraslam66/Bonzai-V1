"""Macro-deranged donor (spec §2, D3 refined): the perplexity-gap donor must differ in
the MACRO tuple, still within-city, fixed seed — otherwise a within-city shuffle that
draws an identical-macro donor is a NO-OP and the macro-only gap is vacuously ~0.

§9 teeth: donor same-city + different-macro + != self; deterministic; PYTHONHASHSEED-robust;
and FAIL LOUD (never silently fall back) when a city has < 2 distinct macro tuples.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from cfm.eval.standing.shuffle import macro_deranged_donor


def test_donor_same_city_and_different_macro():
    cities = ["glasgow", "glasgow", "glasgow", "munich", "munich", "glasgow"]
    macros = [(1, 0), (2, 0), (1, 0), (3, 1), (4, 1), (2, 0)]
    donor = macro_deranged_donor(cities, macros, seed=0)
    for i, di in enumerate(donor):
        assert di != i  # different macro => never self
        assert cities[di] == cities[i]  # within-city
        assert macros[di] != macros[i]  # different macro tuple (non-no-op)


def test_deterministic_given_seed():
    cities = ["g"] * 8
    macros = [(i % 3,) for i in range(8)]
    assert macro_deranged_donor(cities, macros, 7) == macro_deranged_donor(cities, macros, 7)


def test_fail_loud_when_city_has_single_macro_tuple():
    """All cells in a city share ONE macro tuple -> cannot macro-derange -> FAIL LOUD,
    never silently fall back to a self/no-op donor that would emit a clean ~0 gap."""
    cities = ["solo", "solo", "solo"]
    macros = [(1, 1), (1, 1), (1, 1)]
    with pytest.raises(ValueError, match=r"(?i)distinct macro|cannot macro-derange"):
        macro_deranged_donor(cities, macros, seed=0)


def test_fail_loud_names_the_offending_city():
    cities = ["ok", "ok", "bad", "bad"]
    macros = [(1,), (2,), (5,), (5,)]  # 'bad' has only one macro tuple
    with pytest.raises(ValueError, match="bad"):
        macro_deranged_donor(cities, macros, seed=0)


def test_pythonhashseed_robust():
    snippet = (
        "from cfm.eval.standing.shuffle import macro_deranged_donor;"
        "print(macro_deranged_donor(['a','a','a','a'], [(0,),(1,),(0,),(1,)], seed=3))"
    )
    outs = []
    for hs in ("0", "1", "12345"):
        r = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hs},
        )
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, outs
