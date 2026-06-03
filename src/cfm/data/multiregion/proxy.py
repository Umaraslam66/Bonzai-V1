"""Data-only redundancy proxy (spec §7). ADVISORY ONLY — it does NOT gate the
budget (PI decision 2026-06-03). It computes and RECORDS a diagnostic signal (the
geometry redundancy, its relative position to BOTH the language baseline and the
Singapore corpus, and a band-classified verdict label), but the tile budget ALWAYS
sizes up and the r-unresolved flag is ALWAYS set. The bake-off is the sole
authority on r.

Why advisory, not a down-gate: the only place a budget-gating proxy could
DOWN-size (confirm→base) is also the only place it could cause the *unrecoverable*
error — an under-provision discovered mid-bake-off forces a re-fetch+reprocess
inside the timed Leonardo window — and its sole upside is saving cheap CPU cities
that are pre-paid production data anyway. A novel data-only proxy does not clear
the confidence bar to gate the budget DOWN against that asymmetry (protocol §10.1).

NOT the compute-optimal r (that is TRAINING-measured, in the bake-off). The
comparison is RELATIVE — against the language baseline (the r=20 anchor) AND the
Singapore corpus — never an absolute threshold on the EU corpus alone.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

#: Pinned BEFORE any measurement (spec §7). Do not fit to data.
X_AMBIGUOUS_BAND = 0.10  # within ±10% of a reference == "not materially different"
Y_SIZE_UP = 0.5  # +50% tile budget (always applied; proxy is advisory)


def compression_redundancy(token_bytes: bytes) -> float:
    """1 - compressed/raw (gzip). Higher ⇒ more redundant. A cheap, robust proxy."""
    if not token_bytes:
        return 0.0
    return 1.0 - (len(gzip.compress(token_bytes, compresslevel=6)) / len(token_bytes))


@dataclass(frozen=True)
class ProxyVerdict:
    geometry_redundancy: float
    language_baseline: float
    singapore_redundancy: float
    rel_language: float  # relative to the r=20 anchor
    rel_singapore: float  # relative to our own corpus (anomaly cross-reference)
    verdict: str  # DIAGNOSTIC label only — does not gate the budget
    recommended_tile_budget: int  # ALWAYS base*(1+Y)
    r_unresolved_flag: bool  # ALWAYS True


def proxy_decision(
    *,
    geometry_redundancy: float,
    language_baseline: float,
    singapore_redundancy: float,
    base_tile_budget: int,
) -> ProxyVerdict:
    """Record the advisory signal and return the (unconditional) sized-up budget.

    The verdict label classifies the geometry-vs-language relative position into a
    coherent band (diagnostic for the bake-off); ``rel_singapore`` is recorded to
    flag gross anomalies. NEITHER changes the budget: it is always ``base*(1+Y)``
    and the r-unresolved flag is always set.
    """
    rel_lang = (geometry_redundancy - language_baseline) / language_baseline
    rel_sg = (geometry_redundancy - singapore_redundancy) / singapore_redundancy

    # Recorded label ONLY (low-stakes diagnostic; does not gate budget).
    if rel_lang >= X_AMBIGUOUS_BAND:
        verdict = "more_redundant_than_language"  # would-be r<=20 signal
    elif rel_lang <= -X_AMBIGUOUS_BAND:
        verdict = "less_redundant_than_language"  # would-be r>20 signal
    else:
        verdict = "ambiguous"

    # ADVISORY: budget ALWAYS sizes up; flag ALWAYS set. Bake-off is sole r authority.
    budget = round(base_tile_budget * (1.0 + Y_SIZE_UP))
    flag = True

    return ProxyVerdict(
        geometry_redundancy=geometry_redundancy,
        language_baseline=language_baseline,
        singapore_redundancy=singapore_redundancy,
        rel_language=rel_lang,
        rel_singapore=rel_sg,
        verdict=verdict,
        recommended_tile_budget=budget,
        r_unresolved_flag=flag,
    )
