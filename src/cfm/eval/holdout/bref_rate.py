"""The §2 shared bref-placeholder rate - ONE function, two consumers.

C's round-tripped-real ceiling (= 1 - rate) and D's degeneracy-rate judge are
the SAME quantity. It is computed ONCE on round-tripped-real and imported by
both; recomputing it in C's path and D's path separately would resurrect the
reimplementation/drift bug class one-source exists to prevent (spec §2).

Because there is no independent corroborant for this quantity (on real data
sub-G's bijection grounded the bref; here nothing cross-checks it), the guards
on THIS function are the only check on its correctness - they are load-bearing.

Construction-identity exclusion (protocol v2 §9): we import sub-G's predicate by
reference and never reimplement it; the identity-lock test asserts that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfm.data.sub_g.seam_decodability import _is_bref_placeholder_collapse

#: The shared construction-identity predicate, imported by REFERENCE from sub-G.
#: Never reimplement (Gate 6 identity-lock; test asserts `is` identity).
_bref_predicate = _is_bref_placeholder_collapse


@dataclass(frozen=True)
class StratumRate:
    n_total: int
    n_collapse: int

    @property
    def rate(self) -> float:
        return self.n_collapse / self.n_total if self.n_total else 0.0


@dataclass(frozen=True)
class BrefRateResult:
    overall_rate: float
    per_stratum: dict[int, StratumRate]


def bref_placeholder_rate(
    blocks: list[list[int]],
    geoms: list[dict[str, Any]],
    strata: list[int],
) -> BrefRateResult:
    """Stratified bref-placeholder collapse rate over round-tripped-real.

    blocks/geoms/strata are aligned (block i decodes to geom i in stratum i).
    A block is a placeholder collapse iff sub-G's construction-identity predicate
    says so - NEVER a bare zero-length / magnitude test.
    """
    if not (len(blocks) == len(geoms) == len(strata)):
        raise ValueError("blocks, geoms, strata must be the same length")

    totals: dict[int, int] = {}
    collapses: dict[int, int] = {}
    n_total = 0
    n_collapse = 0
    for block, geom, stratum in zip(blocks, geoms, strata, strict=True):
        totals[stratum] = totals.get(stratum, 0) + 1
        n_total += 1
        if _bref_predicate(block, geom):
            collapses[stratum] = collapses.get(stratum, 0) + 1
            n_collapse += 1

    per_stratum = {
        s: StratumRate(n_total=totals[s], n_collapse=collapses.get(s, 0)) for s in totals
    }
    overall = n_collapse / n_total if n_total else 0.0
    return BrefRateResult(overall_rate=overall, per_stratum=per_stratum)
