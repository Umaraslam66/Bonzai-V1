"""Fast canonical-tile derivation tests for sub-D (Task 15 fast complement).

Pins the numerical correctness of the pipeline's derivation helpers against
the locked macro vocab. Lives in the fast suite (no ``@pytest.mark.slow``,
no cached-data dependency) so a CI/local-dev push catches a derivation
regression even when the Layer-3 cached Singapore tests are skipped.

What's tested here vs. ``test_singapore_integration.py``:

- HERE: pure-function correctness of ``_zoning_token_id``,
  ``_bucket_for_numeric_value``, ``_resolve_population_density_bucket``.
  Synthetic inputs + assertions against specific token_ids from the locked
  vocab. Fast, deterministic, runs on every push.
- THERE: end-to-end run of ``derive_region_macro_plan`` against the cached
  Singapore extraction, with structural validation + byte-determinism
  across re-runs. Slow, gated on local cache availability.

The helpers under test are private; we import them via the public
``cfm.data.sub_d.pipeline`` module path. If they're ever renamed or
extracted, fix the imports here together with the rename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.sub_d.enums import MetricNamespace, SlotKind
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.io import DerivationEvidenceRow
from cfm.data.sub_d.pipeline import (
    _bucket_for_numeric_value,
    _resolve_population_density_bucket,
    _zoning_token_id,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCKED_VOCAB_PATH = _REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


@pytest.fixture(scope="module")
def locked_vocab() -> dict:
    """Load the real committed locked macro vocab. Module-scoped so the
    repeated reads inside this file don't re-parse the 11k-line YAML."""
    return yaml.safe_load(_LOCKED_VOCAB_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Zoning: dominant feature class → vocab token_id
# Locked vocab zoning (per macro_plan_vocab.yaml lines 3523-3535):
#   building → 0, road → 1, poi → 2, base → 3.
# ---------------------------------------------------------------------------


def test_zoning_picks_road_when_road_dominant(locked_vocab):
    """A cell with more road features than buildings → zoning_class = 1 (road)."""
    counts = {"road": 5, "building": 2, "poi": 1, "base": 0}
    assert _zoning_token_id(counts, locked_vocab["locked_buckets"]["zoning"]) == 1


def test_zoning_picks_building_when_building_dominant(locked_vocab):
    """A cell with more building features than roads → zoning_class = 0."""
    counts = {"road": 1, "building": 5, "poi": 0, "base": 0}
    assert _zoning_token_id(counts, locked_vocab["locked_buckets"]["zoning"]) == 0


def test_zoning_picks_poi_when_poi_dominant(locked_vocab):
    counts = {"road": 0, "building": 1, "poi": 7, "base": 0}
    assert _zoning_token_id(counts, locked_vocab["locked_buckets"]["zoning"]) == 2


def test_zoning_picks_base_when_only_base(locked_vocab):
    counts = {"road": 0, "building": 0, "poi": 0, "base": 3}
    assert _zoning_token_id(counts, locked_vocab["locked_buckets"]["zoning"]) == 3


def test_zoning_tie_break_uses_feature_class_enum_order(locked_vocab):
    """Ties resolve to the class with the smallest ``FeatureClass`` enum int.
    Enum order: ROAD=0, BUILDING=1, POI=2, BASE=3 — so a 4-way tie picks road.
    """
    counts = {"road": 3, "building": 3, "poi": 3, "base": 3}
    # Road wins (smallest enum) — its vocab token_id is 1.
    assert _zoning_token_id(counts, locked_vocab["locked_buckets"]["zoning"]) == 1


def test_zoning_returns_none_on_empty_or_zero_counts(locked_vocab):
    """Active cell with no features → no zoning info — validator allows None."""
    buckets = locked_vocab["locked_buckets"]["zoning"]
    assert _zoning_token_id({}, buckets) is None
    assert _zoning_token_id({"road": 0, "building": 0, "poi": 0, "base": 0}, buckets) is None


# ---------------------------------------------------------------------------
# Cell density: building_footprint_ratio → bucket lookup
# Locked vocab cell_density (per macro_plan_vocab.yaml lines 3472-3488):
#   [0.00, 0.05) → 0
#   [0.05, 0.15) → 1
#   [0.15, 0.35) → 2
#   [0.35, ∞)    → 3   (open-ended; absorbs known_issue #9)
# ---------------------------------------------------------------------------


def test_cell_density_each_bucket_boundary(locked_vocab):
    """Pin the bucket-assignment behaviour at every locked boundary.

    The boundaries are ``[lower_inclusive, upper_exclusive)``: a value
    equal to ``upper_exclusive`` falls into the NEXT bucket, never the
    current one.
    """
    buckets = locked_vocab["locked_buckets"]["cell_density"]
    # Inside each bucket
    assert _bucket_for_numeric_value(0.0, buckets) == 0
    assert _bucket_for_numeric_value(0.04, buckets) == 0
    assert _bucket_for_numeric_value(0.07, buckets) == 1
    assert _bucket_for_numeric_value(0.20, buckets) == 2
    assert _bucket_for_numeric_value(0.50, buckets) == 3
    # On each lower_inclusive boundary
    assert _bucket_for_numeric_value(0.05, buckets) == 1
    assert _bucket_for_numeric_value(0.15, buckets) == 2
    assert _bucket_for_numeric_value(0.35, buckets) == 3
    # Just below each upper_exclusive boundary (within float epsilon of the
    # cut but still inside the lower bucket)
    assert _bucket_for_numeric_value(0.04999999, buckets) == 0
    assert _bucket_for_numeric_value(0.14999999, buckets) == 1
    assert _bucket_for_numeric_value(0.34999999, buckets) == 2


def test_cell_density_top_bucket_absorbs_ratio_above_one(locked_vocab):
    """Known_issue #9: real Singapore data produces cell_density ratios > 1.0.
    The locked vocab's top bucket ``[0.35, ∞)`` absorbs them — the validator
    must NOT clamp or reject, and the bucket lookup must NOT raise.
    """
    buckets = locked_vocab["locked_buckets"]["cell_density"]
    assert _bucket_for_numeric_value(0.95, buckets) == 3
    assert _bucket_for_numeric_value(1.0, buckets) == 3
    assert _bucket_for_numeric_value(1.5, buckets) == 3
    assert _bucket_for_numeric_value(7.42, buckets) == 3  # absurd but admissible


# ---------------------------------------------------------------------------
# Road skeleton: road_crossing_count → bucket lookup
# Locked vocab road_skeleton (per macro_plan_vocab.yaml lines 3489-3505):
#   [0, 1) → 0   (no crossings)
#   [1, 4) → 1
#   [4, 9) → 2
#   [9, ∞) → 3
# ---------------------------------------------------------------------------


def test_road_skeleton_each_bucket_boundary(locked_vocab):
    buckets = locked_vocab["locked_buckets"]["road_skeleton"]
    assert _bucket_for_numeric_value(0, buckets) == 0
    assert _bucket_for_numeric_value(1, buckets) == 1
    assert _bucket_for_numeric_value(3, buckets) == 1
    assert _bucket_for_numeric_value(4, buckets) == 2
    assert _bucket_for_numeric_value(8, buckets) == 2
    assert _bucket_for_numeric_value(9, buckets) == 3
    assert _bucket_for_numeric_value(50, buckets) == 3  # open-ended top


# ---------------------------------------------------------------------------
# Tile-population-density: locked_proxy value → bucket lookup
# Locked vocab tile_population_density (per macro_plan_vocab.yaml lines 3506-3522):
#   [0.00, 0.02) → 0
#   [0.02, 0.15) → 1
#   [0.15, 0.31) → 2
#   [0.31, ∞)    → 3
# locked_proxy: p75_building_footprint_ratio
# ---------------------------------------------------------------------------


def _make_pop_density_evidence_row(proxy_name: str, value: float) -> DerivationEvidenceRow:
    """Build one TILE-scoped tile_population_density evidence row for tests."""
    return DerivationEvidenceRow(
        slot_kind=SlotKind.TILE,
        slot_index=0,
        metric_namespace=MetricNamespace.TILE_POPULATION_DENSITY,
        metric_name=proxy_name,
        value=float(value),
        derivation_version="1.0",
    )


def test_resolve_population_density_bucket_uses_locked_proxy(locked_vocab):
    """``_resolve_population_density_bucket`` reads
    ``locked_vocab["locked_proxy"]["tile_population_density"]`` to pick WHICH
    proxy row to consume, then buckets that value. Other proxy rows present
    in the evidence are ignored — only the locked proxy contributes.
    """
    locked_proxy = locked_vocab["locked_proxy"]["tile_population_density"]
    assert locked_proxy == "p75_building_footprint_ratio"  # spec sanity-check

    # Emit all four candidate proxy rows. Only the locked proxy's value
    # should drive the bucket selection.
    rows = [
        _make_pop_density_evidence_row("mean_building_footprint_ratio", 0.99),
        _make_pop_density_evidence_row("area_weighted_building_density", 0.99),
        _make_pop_density_evidence_row("median_building_footprint_ratio", 0.99),
        _make_pop_density_evidence_row("p75_building_footprint_ratio", 0.01),
    ]
    assert _resolve_population_density_bucket(rows, locked_vocab) == 0  # 0.01 → bucket 0


def test_resolve_population_density_bucket_assigns_each_bucket(locked_vocab):
    """Exercise every locked bucket via the proxy row's value."""
    locked_proxy = locked_vocab["locked_proxy"]["tile_population_density"]
    for value, expected_token_id in [
        (0.00, 0),
        (0.01, 0),
        (0.02, 1),
        (0.10, 1),
        (0.15, 2),
        (0.30, 2),
        (0.31, 3),
        (0.50, 3),
        (0.95, 3),  # absorbed by open-ended top bucket
    ]:
        rows = [_make_pop_density_evidence_row(locked_proxy, value)]
        assert _resolve_population_density_bucket(rows, locked_vocab) == expected_token_id, (
            f"value={value!r}"
        )


def test_resolve_population_density_bucket_raises_when_proxy_missing(locked_vocab):
    """If the locked proxy's row is missing from derivation_evidence, the
    resolver raises rather than silently picking a default — a missing
    proxy means upstream evidence emission is broken, not that the value
    is zero.
    """
    # Emit only NON-locked proxies; the locked one is absent.
    rows = [
        _make_pop_density_evidence_row("mean_building_footprint_ratio", 0.5),
    ]
    with pytest.raises(SubDValidationError):
        _resolve_population_density_bucket(rows, locked_vocab)
