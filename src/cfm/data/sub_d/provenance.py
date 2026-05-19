"""Per-tile provenance.yaml builder (sub-D spec §11.5, Task 11).

Sub-D writes one provenance.yaml per tile recording:

- ``extraction``: who/when/why this tile was derived (commit_sha,
  extracted_utc, rerun_count, rerun_reason).
- ``inputs``: sub-D's view of the upstream sub-C artifacts plus the locked
  macro-plan vocab/config it consumed. sha256 values are recorded verbatim
  from the caller; they were snapshotted by sub-D's reader at read time
  (``SubCTileInputs.digests``).
- ``versions``: locked vocab + per-namespace derivation versions that
  contributed to this tile's output (A1 discipline: per-namespace, not a
  single global derivation_version).
- ``outputs``: sha256 of the parquet/yaml files this tile wrote.

Two digest semantics coexist (tension flag B4):

- ``inputs.sub_c_*_sha256`` are *bytes-sha* values of upstream files. They
  flow through verbatim; sub-D does not re-hash them.
- ``provenance.yaml``'s OWN self-integrity sha (which the region manifest
  later records in ``manifest.tiles[*].provenance_sha256``) uses the
  *excluding-timestamp* semantics: strip ``extraction.extracted_utc`` and
  any final-segment ``*_sha256`` field, canonicalize the rest to YAML,
  sha256 the bytes.

``extraction.rerun_reason`` is INCLUDED in the self-sha (per sub-C F2
precedent — audit-trail purpose). Therefore it MUST NOT appear in
``SUB_D_EXCLUDED_FROM_SHA``.

Per spec §11.5 + tension flag B7, the artifact uses the namespaced
``provenance_schema_version`` field, NOT a bare ``schema_version``.
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.io import canonicalize_yaml

#: Version of the provenance.yaml artifact format. Bumped when the YAML
#: structure changes (new section, renamed field). Independent of vocab or
#: derivation versions per spec §12.
PROVENANCE_SCHEMA_VERSION: str = "1.0"

#: Sub-D's exclusion table for provenance.yaml self-integrity hashing.
#:
#: Wildcard semantics (shared neutral grammar from ``cfm.data.determinism``):
#:
#: - Entries under ``"*"`` apply to all file_keys; an entry that itself starts
#:   with ``*`` is a final-segment suffix match.
#: - Entries under a specific file_key (e.g. ``"provenance.yaml"``) are exact
#:   dotted-path matches.
#:
#: B4 invariant: ``extraction.rerun_reason`` is NOT in this table — it must
#: contribute to the sha so reruns are auditable.
SUB_D_EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
}


def build_tile_provenance(
    tile_i: int,
    tile_j: int,
    extraction: dict,
    inputs: dict,
    versions: dict,
    outputs: dict,
) -> dict:
    """Build the provenance.yaml dict for one tile.

    Parameters:
        tile_i, tile_j: tile coordinates.
        extraction: who/when/why; minimally ``commit_sha``, ``extracted_utc``,
            ``rerun_count``, ``rerun_reason``.
        inputs: ``release`` plus sub-C and locked-config sha256 values. These
            are recorded verbatim; the caller (the pipeline in Task 14) is
            responsible for sourcing them from ``SubCTileInputs.digests`` and
            from the locked-config file hashes.
        versions: locked vocab + per-namespace derivation versions. Minimally
            ``sub_d_schema_version``, ``macro_plan_vocab_version``, and the
            four ``{namespace}_vocab_version`` / ``{namespace}_derivation_version``
            pairs (zoning / cell_density / tile_population_density /
            road_skeleton).
        outputs: sha256 of the three per-tile sub-D output files
            (macro_core.parquet, derivation_evidence.parquet,
            effective_conditioning.yaml).

    Returns:
        The provenance dict, ready for ``write_provenance`` and
        ``provenance_sha256``.
    """
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "tile_i": int(tile_i),
        "tile_j": int(tile_j),
        "extraction": dict(extraction),
        "inputs": dict(inputs),
        "versions": dict(versions),
        "outputs": dict(outputs),
    }


def provenance_sha256(data: dict) -> str:
    """Compute the self-integrity sha for a provenance.yaml dict.

    Strips ``extraction.extracted_utc`` and final-segment ``*_sha256`` fields
    per ``SUB_D_EXCLUDED_FROM_SHA``, canonicalizes the remainder, and hashes
    the bytes. This is the value the region manifest later records in
    ``tiles[*].provenance_sha256`` (Task 12).
    """
    return _compute_sha256_excluding(data, "provenance.yaml", SUB_D_EXCLUDED_FROM_SHA)


def write_provenance(data: dict, path: Path) -> None:
    """Serialise *data* to *path* using the neutral canonical YAML helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(data), encoding="utf-8")
