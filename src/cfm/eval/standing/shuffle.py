"""Macro-deranged donor for the perplexity-gap (spec §2, D3 refined).

Each held-out cell gets a DONOR cell that is (a) in the SAME city (within-city control) and
(b) carries a DIFFERENT macro tuple — so the shuffled conditioning actually differs in the
macro buckets. Without (b), a within-city random donor frequently shares the target's macro
(strata cluster within a city), making the shuffle a NO-OP and the macro-only gap vacuously
~0. If a city has < 2 distinct macro tuples it cannot be macro-deranged: FAIL LOUD (never
silently fall back to a self/no-op donor). Deterministic given (inputs, seed) and
PYTHONHASHSEED-robust: grouped in input order (dict preserves insertion order); choices over
input-ordered lists, never hash-ordered sets.
"""

from __future__ import annotations

import random
from collections.abc import Hashable


def macro_deranged_donor(group_keys: list[str], macro_keys: list[Hashable], seed: int) -> list[int]:
    """Donor index per position: same city, different macro tuple. Deterministic.

    Raises ValueError (naming the city) if any city has < 2 distinct macro tuples.
    """
    if len(group_keys) != len(macro_keys):
        raise ValueError("group_keys and macro_keys must be the same length")
    rng = random.Random(seed)
    by_city: dict[str, list[int]] = {}
    for i, g in enumerate(group_keys):
        by_city.setdefault(g, []).append(i)
    donor: list[int] = [-1] * len(group_keys)
    for city, idxs in by_city.items():  # insertion order == input order (hash-independent)
        n_distinct = len({macro_keys[i] for i in idxs})
        if n_distinct < 2:
            raise ValueError(
                f"city {city!r} has {n_distinct} distinct macro tuple(s) (< 2); cannot "
                f"macro-derange — refusing to silently fall back to a no-op donor"
            )
        for i in idxs:
            pool = [j for j in idxs if macro_keys[j] != macro_keys[i]]  # input-ordered
            donor[i] = rng.choice(pool)  # pool non-empty (n_distinct >= 2)
    return donor
