"""Real-side reference distributions + the round-tripped-real ceiling (spec §C).

Two baselines per §9 layer (spec §C):
- core = round-tripped-real (real -> tokens -> decode): the architecture-comparison
  reference (cancels the shared tokenizer ceiling).
- full = raw-real (sub-C original geometry): the absolute-fidelity reference.
- gap (full - core) = the tokenizer's own contribution, reported explicitly.

The geometric-validity CEILING is ``1 - bref_placeholder_rate``, taken from the §2
SHARED BrefRateResult - never recomputed here (one source; spec §2/§D ceiling).

DEFERRED (spec §7): the model-vs-baseline Wasserstein/KS distance. We ship the
reference samples + provenance; the distance is computed against model output in
the eval-harness successor.

Provenance propagation (spec §F): every ReferenceDistribution carries its
source-tile lineage so the fail-closed lineage audit (G-F4) can bind.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfm.eval.holdout.bref_rate import BrefRateResult

#: A (region, tile_dirname) lineage anchor.
TileRef = tuple[str, str]


@dataclass(frozen=True)
class GeometricValidityCeiling:
    overall: float
    per_stratum: dict[int, float]


def geometric_validity_ceiling(shared: BrefRateResult) -> GeometricValidityCeiling:
    """Ceiling = 1 - bref-placeholder-rate, from the §2 shared result (not recomputed)."""
    return GeometricValidityCeiling(
        overall=1.0 - shared.overall_rate,
        per_stratum={s: 1.0 - sr.rate for s, sr in shared.per_stratum.items()},
    )


@dataclass(frozen=True)
class ReferenceDistribution:
    metric: str  # e.g. "building_area_m2", "road_length_m", "cell_density"
    kind: str  # "raw" | "round_tripped"
    stratum: int  # cell_density_bucket
    samples: tuple[float, ...]
    source_tiles: tuple[TileRef, ...]  # lineage - mandatory, non-empty

    def __post_init__(self) -> None:
        if not self.source_tiles:
            raise ValueError(
                "ReferenceDistribution requires non-empty source_tiles lineage "
                "(spec §F: provenance-propagation, or the G-F4 audit cannot bind)"
            )
        if self.kind not in ("raw", "round_tripped"):
            raise ValueError(f"kind must be 'raw' or 'round_tripped'; got {self.kind!r}")


def report_gap(*, full_value: float, core_value: float) -> float:
    """full - core = the tokenizer's own contribution (the H1 229m-residual shape)."""
    return full_value - core_value
