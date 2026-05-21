"""Per-tile provenance writer.

Mirrors sub-D's pattern at ``src/cfm/data/sub_d/provenance.py``: the
on-disk YAML carries timestamps (extracted_utc) and verbatim sha256
values, but the self-integrity sha used in the digest chain strips both
classes of fields via ``SUB_E_EXCLUDED_FROM_SHA`` so reruns under live
clocks produce the same chain value. The neutral
``cfm.data.determinism.compute_sha256_excluding`` helper does the
strip-canonicalize-hash pipeline.

Spec §9.2 mandate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.io import canonicalize_yaml

#: Sub-E's exclusion table for self-integrity hashing. Per spec §9.2 the
#: inherited ``cfm.data.determinism`` grammar applies:
#:
#: - Entries under ``"*"`` apply to all file_keys; an entry starting with
#:   ``*`` is a final-segment suffix match.
#: - Entries under a specific file_key are exact dotted-path matches.
#:
#: Mirrors ``SUB_D_EXCLUDED_FROM_SHA`` at ``src/cfm/data/sub_d/provenance.py``.
#: ``manifest.yaml`` entries are present for spec-§9.2 completeness; the
#: helper ``manifest_sha256`` (in ``manifest.py``) reads from the same table.
SUB_E_EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
}


@dataclass(frozen=True)
class SubEInputDigests:
    release: str
    sub_c_manifest_sha256: str
    sub_c_features_parquet_sha256: str
    sub_c_crossings_parquet_sha256: str
    sub_d_manifest_sha256: str
    sub_d_macro_core_parquet_sha256: str
    boundary_vocab_sha256: str
    derivation_config_sha256: str


@dataclass(frozen=True)
class SubEVersions:
    sub_e_schema_version: str
    boundary_vocab_version: str
    boundary_derivation_version: str


@dataclass(frozen=True)
class SubEProvenance:
    tile_i: int
    tile_j: int
    extraction_commit_sha: str
    extracted_utc: str
    rerun_count: int
    rerun_reason: str
    inputs: SubEInputDigests
    versions: SubEVersions
    boundary_contract_parquet_sha256: str
    provenance_schema_version: str = "1.0"


def provenance_to_dict(prov: SubEProvenance) -> dict:
    """Serialise SubEProvenance to its on-disk YAML dict shape.

    Exposed publicly so callers (Task 10 orchestrator, Task 9 tests) can
    compute ``provenance_sha256(dict)`` against the same dict that
    ``write_provenance`` serialises, without re-loading the file.
    """
    return {
        "provenance_schema_version": prov.provenance_schema_version,
        "tile_i": prov.tile_i,
        "tile_j": prov.tile_j,
        "extraction": {
            "commit_sha": prov.extraction_commit_sha,
            "extracted_utc": prov.extracted_utc,
            "rerun_count": prov.rerun_count,
            "rerun_reason": prov.rerun_reason,
        },
        "inputs": asdict(prov.inputs),
        "versions": asdict(prov.versions),
        "outputs": {
            "boundary_contract_parquet_sha256": prov.boundary_contract_parquet_sha256,
        },
    }


def write_provenance(path: Path, prov: SubEProvenance) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(provenance_to_dict(prov)), encoding="utf-8")
    return path


def provenance_sha256(data: dict) -> str:
    """Compute the self-integrity sha for a provenance.yaml dict.

    Strips ``extraction.extracted_utc`` and final-segment ``*_sha256``
    fields per ``SUB_E_EXCLUDED_FROM_SHA``, canonicalises the remainder
    to YAML, and hashes the bytes. This is the value the region manifest
    records in ``tiles[*].provenance_sha256``.
    """
    return _compute_sha256_excluding(data, "provenance.yaml", SUB_E_EXCLUDED_FROM_SHA)
