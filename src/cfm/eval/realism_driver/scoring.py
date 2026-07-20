"""Realism-eval scoring helpers: torch-free decode + real-features serialization.

Task 5 creates this module with (a) ``decode_tokens_to_cell`` — the SAME
split-into-features + decode-or-None chain ``scripts.train_scaffold.score_cell``
uses, but WITHOUT a model (already-real tokens, no generation), and (b) the
``real-features.yaml`` record serialization keyed to the schema
``scripts/run_bakeoff_decision.py`` loads (``{metric, stratum, samples}`` per
city). Task 6 extends this module with the scored-lane driver.

TORCH DISCIPLINE (verified 2026-07-20; single authority per Task-5 review): the
decode primitives live in torch-free modules — ``split_cell_into_features`` in
``cfm.data.sub_g``'s ``seam_decodability`` and ``try_decode_block`` in
``cfm.data.sub_f.decoder`` (its home since the review fix; ``cfm.inference.generate``
re-exports it unchanged for its torch-side importers). Importing this module never
pulls torch, matching the Task-1/Task-2 import discipline of the realism_driver
package.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

# BOTH torch-free (checked 2026-07-20): importing this module must not pull torch.
from cfm.data.sub_f.decoder import try_decode_block
from cfm.data.sub_g.seam_decodability import split_cell_into_features

logger = logging.getLogger(__name__)

#: {(metric, stratum) -> samples}; stratum is the floor's 4-tuple. Structural
#: mirror of ``gen_realism.GenFeatures`` / ``bakeoff_decision.GenFeatures`` (kept
#: local so importing this module never drags a torch-pulling module in).
GenFeatures = dict[tuple[str, tuple], list[float]]


# --------------------------------------------------------------------------- #
# Decode (real tokens -> aligned kept blocks/geoms), torch-free
# --------------------------------------------------------------------------- #


def decode_tokens_to_cell(tokens: Sequence[int]) -> tuple[list[list[int]], list[dict]]:
    """Split a real cell's token body into feature blocks and decode each, keeping
    only the blocks that decode — returning ALIGNED ``(blocks, geoms)`` exactly as
    ``score_cell`` does (``[b for b, d in decoded if d is not None]`` /
    ``[d for ... if d is not None]``), so a decoded real cell feeds
    ``gen_realism.DecodedCell`` in the identical shape a generated cell does.

    No model, no torch: the tokens are already-real (no generation step). A block
    that fails to decode is skipped (measured rate), never raised."""
    kept_blocks: list[list[int]] = []
    kept_geoms: list[dict] = []
    for block in split_cell_into_features(list(tokens)):
        geom = try_decode_block(block)
        if geom is not None:
            kept_blocks.append(block)
            kept_geoms.append(geom)
    return kept_blocks, kept_geoms


# --------------------------------------------------------------------------- #
# real-features.yaml serialization (run_bakeoff_decision record schema)
# --------------------------------------------------------------------------- #

#: Spec tag stamped into the artifact meta (lineage; bump on layout change).
REAL_FEATURES_SPEC = "realism-eval-real-features-v1"


def _stratum_sort_key(stratum: tuple) -> tuple[str, ...]:
    return tuple(str(x) for x in stratum)


def features_to_records(features: GenFeatures) -> list[dict]:
    """``{(metric, stratum): samples}`` -> ``[{metric, stratum, samples}, ...]`` — the
    schema ``run_bakeoff_decision._features_from_records`` reads. Sorted by
    (metric, str(stratum)) so the artifact is insertion-order independent
    (PYTHONHASHSEED-proof), mirroring the floor artifact's ordering discipline."""
    return [
        {
            "metric": metric,
            "stratum": list(stratum),
            "samples": [float(x) for x in samples],
        }
        for (metric, stratum), samples in sorted(
            features.items(), key=lambda kv: (kv[0][0], _stratum_sort_key(kv[0][1]))
        )
    ]


def features_from_records(records: list[dict]) -> GenFeatures:
    """Inverse of :func:`features_to_records` — byte-for-byte the same shape as
    ``run_bakeoff_decision._features_from_records`` (the downstream reader), kept
    here so a roundtrip can be asserted without importing the scripts layer."""
    return {
        (rec["metric"], tuple(rec["stratum"])): [float(x) for x in rec["samples"]]
        for rec in records
    }


def city_features_to_records(by_city: Mapping[str, GenFeatures]) -> dict[str, list[dict]]:
    """``{city: GenFeatures}`` -> ``{city: [record, ...]}`` (cities sorted)."""
    return {city: features_to_records(by_city[city]) for city in sorted(by_city)}


def build_real_features_payload(
    *,
    meta: dict,
    real_by_city: Mapping[str, GenFeatures] | None = None,
    real_train_by_city: Mapping[str, GenFeatures] | None = None,
) -> dict:
    """Assemble the ``real-features.yaml`` payload.

    A half is included ONLY when computed: a held-out-only (or train-only) partial
    OMITS the other key ENTIRELY, so ``run_bakeoff_decision._load_real``'s STRICT
    read refuses it (missing key) rather than silently accepting an empty set —
    only a full run (both halves) yields the canonical artifact the memorization
    check accepts."""
    payload: dict[str, Any] = {"spec": REAL_FEATURES_SPEC, "meta": dict(meta)}
    if real_by_city is not None:
        payload["real_by_city"] = city_features_to_records(real_by_city)
    if real_train_by_city is not None:
        payload["real_train_by_city"] = city_features_to_records(real_train_by_city)
    return payload


def write_real_features(path: str | Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` as write-once YAML (atomic tmp + rename).

    Refuses to overwrite an existing file (``FileExistsError``): re-extracting real
    features means deleting the old artifact deliberately (the eval-set / gen-artifact
    write-once discipline), so a stale extraction is never silently clobbered."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"real-features artifact already exists at {path}; it is write-once — "
            "delete deliberately only to re-extract."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote real-features artifact -> %s", path)


def load_real_features(path: str | Path) -> dict:
    """Read a real-features YAML back into its raw mapping (for end-state verify)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
