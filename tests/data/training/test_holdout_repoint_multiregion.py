"""Two-sided guard for obligation (a): REGION-AWARE holdout-manifest repoint.

The holdout-manifest readers (`build_shards._holdout_ids`,
`geometry.holdout_polygons_per_active_cell`, the region-parameterized scripts) are
DUAL-REGION: reached with `region="singapore"` (the local test fixture) AND with the
4 EU held-out cities at runtime. Obligation (a) must route them through a region-aware
helper so:

  - `region == "singapore"`        -> the SG single-region manifest (schema 1.0)
  - `region` in the 4 EU cities    -> the multiregion manifest (schema 2.0)

This test drives the manifest-read surface of `build_shards._holdout_ids` ONLY (no
tile-data round-trip, so it runs locally with munich's tile data absent). It asserts
BOTH sides:

  - (EU) `_holdout_ids(..., "munich")` resolves munich's holdout tile-ids. RED before
    the region-aware routing exists (the consumer reads the SG manifest -> `KeyError:
    'munich'`); GREEN after.
  - (SG) `_holdout_ids(..., "singapore")` resolves singapore's holdout tile-ids. GREEN
    both before AND after (proves region-aware routing did not break the SG path).

It NEVER touches `assert_resolution_sufficient` / the SG `_EVAL_SET_LOCKED` marker
(that is obligation (c) / Task 9, not (a)).
"""

from __future__ import annotations

import yaml

from cfm.data.training.build_shards import _holdout_ids
from cfm.eval.holdout.paths import (
    holdout_manifest_path,
    multiregion_holdout_manifest_path,
)

_RELEASE = "2026-04-15.0"


def _expected_ids_from_manifest(manifest_path, region: str) -> set[tuple[int, int]]:
    """Read the holdout tile-ids straight from the on-disk manifest, WITHOUT going
    through the routing under test (independent oracle)."""
    m = yaml.safe_load(manifest_path(_RELEASE).read_text(encoding="utf-8"))
    return {(int(t["tile_i"]), int(t["tile_j"])) for t in m["regions"][region]["tiles"]}


def test_holdout_ids_resolves_eu_city_from_multiregion_manifest() -> None:
    """EU SIDE: `_holdout_ids(release, "munich")` must resolve munich's holdout
    tile-ids from the MULTIREGION manifest. RED before the region-aware routing exists
    (the SG manifest has no `munich` key -> KeyError); GREEN after."""
    got = _holdout_ids(_RELEASE, "munich")
    expected = _expected_ids_from_manifest(multiregion_holdout_manifest_path, "munich")
    assert got == expected
    assert len(got) > 0  # munich is a real held-out city with tiles


def test_holdout_ids_resolves_singapore_from_sg_manifest() -> None:
    """SG SIDE: `_holdout_ids(release, "singapore")` must resolve singapore's holdout
    tile-ids from the SG single-region manifest. GREEN before AND after — proves the
    region-aware routing did not break the (now local-fixture) Singapore path."""
    got = _holdout_ids(_RELEASE, "singapore")
    expected = _expected_ids_from_manifest(holdout_manifest_path, "singapore")
    assert got == expected
    assert len(got) > 0  # the frozen 132-tile Singapore holdout set
