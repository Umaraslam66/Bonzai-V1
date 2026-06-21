# src/cfm/eval/lane_s_sampler.py
"""Lane-S held-out CELL SAMPLER (spec 2026-06-21).

A budget-bounded stratified DOWN-sampler over the held-out cell pool. Picks
which held-out cells the bake-off backbones generate so generated-side feature
distributions clear the conditioning floor's min_n per scored stratum.

UNIT DISCIPLINE (spec, protocol §10.3): the obligation is FEATURES (>= min_n per
floored (city, metric, stratum)); the lever is CELLS (per (city, 4-tuple)). The
scarce floored metric (building_area where owed) binds n_cells.

The artifact is sha-locked write-once via cfm.data.locked_yaml, mirroring the
conditioning floor's grammar (sha excludes itself; a _LANE_S_SAMPLER_LOCKED
marker beside the file; reader refuses absent/unsealed/sha-mismatch/skew).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cfm.data.locked_yaml import stamp_and_seal, verify_sealed_yaml

logger = logging.getLogger(__name__)

SAMPLER_SCHEMA_VERSION = "1.0"
SAMPLER_LOCK_NAME = "_LANE_S_SAMPLER_LOCKED"
SAMPLER_SHA_FIELD = "sampler_sha256"

#: Metric token strings as the floor freezes them (conditioning_discrimination._tile_features).
BUILDING_METRIC = "building_area_m2"
ROAD_METRIC = "road_length_m"

#: LOCK-AND-GUARDS-TRAVEL-TOGETHER (spec invariant, PI 2026-06-21): floor_n is READ from THIS
#: locked floor (sha 95abb88), NEVER recomputed. The build CLI fails loud if the loaded floor's
#: sha differs (a re-derived floor could change n_a/n_b silently); Task 4's external-SoT test
#: RED-flags a change so the guard + this constant update in the SAME commit as the floor.
EXPECTED_FLOOR_SHA256 = "95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6"

#: DECISION: default target = the floor's locked min_n (the obligation unit). Revisit only if
#: the floor's min_n changes (then cells re-derive automatically). Spec §6.
DEFAULT_TARGET_FEATURES = 50
#: DECISION: headroom=2.0 default (spec Gate 5 + R3: 6/119 ceiling-bound at 2.0, glasgow-
#: concentrated, #21 risk negligible). Config knob; refined after first generation. Spec §6.
DEFAULT_HEADROOM = 2.0


class SamplerArtifactError(RuntimeError):
    """The sampler manifest failed verification (absent / unsealed / tampered / skewed)."""


def seal_manifest(payload: dict, path: Path) -> None:
    """Stamp the sha, write canonical YAML ONCE, touch the lock marker."""
    stamp_and_seal(payload, path, sha_field=SAMPLER_SHA_FIELD, lock_name=SAMPLER_LOCK_NAME)


def load_verified_manifest(path: Path) -> dict:
    """Verified read; refuses absent/unsealed/sha-mismatch/version-skew (fail-closed)."""
    return verify_sealed_yaml(
        path,
        sha_field=SAMPLER_SHA_FIELD,
        lock_name=SAMPLER_LOCK_NAME,
        schema_field="sampler_schema_version",
        schema_version=SAMPLER_SCHEMA_VERSION,
        required_key="strata",
        error=SamplerArtifactError,
    )
