"""Region manifest.yaml builder (sub-D spec §11.6, Task 12).

The region manifest is the rollup written by Task 14's pipeline after all
per-tile artifacts (macro_core.parquet, derivation_evidence.parquet,
effective_conditioning.yaml, provenance.yaml) are on disk. It carries:

- ``manifest_schema_version`` (namespaced per tension flag B7).
- ``sub_d_schema_version`` at top level (per spec §11.6).
- ``inputs``: sub-D's view of the consumed sub-C region.
- ``versions``: locked vocab + per-namespace derivation versions (A1).
- ``config_source`` + ``config``: the entire ``sub_c_manifest["config"]``
  dict copied verbatim (tension flag B6). build_manifest enforces equality
  with ``sub_c_manifest["config"]`` at build time; the Task 13 region
  validator cross-checks again at validation time (defense in depth).
- ``initial_extraction``: who/when/how long the first full extraction took.
- ``tiles``: per-tile inventory sorted by ``(tile_i, tile_j)``, each entry
  carrying ``provenance_sha256`` from Task 11's
  ``provenance_sha256(tile_prov)`` — the chain-of-custody anchor the Task 13
  validator uses to detect drift between manifest and on-disk provenance.

``_SUCCESS`` is written by ``write_success_marker`` only — never as a side
effect of ``write_manifest`` — so the Task 14 pipeline can gate it on the
cross-tile validator passing (spec §11.8 sub-C precedent).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.provenance import provenance_sha256

#: Version of the manifest.yaml artifact format. Bumped when the YAML
#: structure changes (new section, renamed field). Does NOT bump when vocab
#: or derivation versions change — those are independent per spec §12.
MANIFEST_SCHEMA_VERSION: str = "1.0"

#: Sub-D's umbrella code-level schema version. Recorded at the top level of
#: manifest.yaml (per spec §11.6) and in the ``versions`` block of
#: provenance.yaml (per spec §11.5). Single source of truth so the two
#: artifacts cannot drift apart.
SUB_D_SCHEMA_VERSION: str = "1.0"

#: Marker value written into the manifest's ``config_source`` field so a
#: reader (human or validator) can confirm where the ``config`` block was
#: copied from. The Task 13 validator asserts on this marker before
#: re-checking the config equality against the live sub-C manifest.
CONFIG_SOURCE_SUB_C_MANIFEST: str = "sub_c_manifest.config"


def build_manifest(
    release: str,
    region: str,
    region_crs: str,
    sub_c_manifest: dict,
    inputs: dict,
    versions: dict,
    config: dict,
    initial_extraction: dict,
    tile_provenances: list[dict],
) -> dict:
    """Build the region manifest.yaml dict.

    Parameters:
        release: release tag (e.g. ``"2026-04-15.0"``).
        region: region slug (e.g. ``"singapore"``).
        region_crs: EPSG code for the region (e.g. ``"EPSG:3414"``).
        sub_c_manifest: the upstream sub-C manifest dict. Used here only to
            enforce ``config == sub_c_manifest["config"]`` (tension flag
            B6 — schema-driven copy, not a hand-picked subset).
        inputs: sub-D's view of the consumed sub-C region, minimally
            ``sub_c_manifest_sha256`` and ``sub_c_region_dir``.
        versions: locked vocab + per-namespace derivation versions
            (macro_plan_vocab_version, zoning_vocab_version,
            zoning_derivation_version, cell_density_*, tile_population_density_*,
            road_skeleton_*). ``sub_d_schema_version`` is sourced from the
            module-level constant and emitted at the top level — callers do
            not pass it here.
        config: the config dict to record. Must equal
            ``sub_c_manifest["config"]`` byte-for-byte. ``SubDValidationError``
            is raised on mismatch — catches caller bugs (hand-picked subset,
            silent drift) at build time.
        initial_extraction: ``commit_sha``, ``started_utc``, ``completed_utc``,
            ``tile_count``.
        tile_provenances: per-tile provenance dicts from Task 11's
            ``build_tile_provenance``. Order is not significant; they are
            sorted by ``(tile_i, tile_j)`` inside ``aggregate_tile_inventory``.

    Returns:
        The manifest dict, ready for ``write_manifest`` and validation.

    Raises:
        SubDValidationError: if ``config != sub_c_manifest["config"]``.
            This is the B6 enforcement: copy verbatim or fail.
    """
    if config != sub_c_manifest["config"]:
        raise SubDValidationError(
            "manifest.config must equal sub_c_manifest['config'] verbatim "
            "(tension flag B6 — schema-driven copy, not a hand-picked subset). "
            f"sub-C keys: {sorted(sub_c_manifest['config'].keys())}; "
            f"sub-D-requested keys: {sorted(config.keys())}"
        )

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sub_d_schema_version": SUB_D_SCHEMA_VERSION,
        "release": str(release),
        "region": str(region),
        "region_crs": str(region_crs),
        "inputs": dict(inputs),
        "versions": dict(versions),
        "config_source": CONFIG_SOURCE_SUB_C_MANIFEST,
        "config": dict(config),
        "initial_extraction": dict(initial_extraction),
        "tiles": aggregate_tile_inventory(tile_provenances),
    }


def aggregate_tile_inventory(tile_provenances: list[dict]) -> list[dict]:
    """Build the sorted ``tiles[]`` list from per-tile provenance dicts.

    For each provenance dict (from ``build_tile_provenance``):

    - Compute the self-integrity sha via Task 11's ``provenance_sha256``,
      which strips ``extraction.extracted_utc`` + final-segment ``*_sha256``
      fields per ``SUB_D_EXCLUDED_FROM_SHA``.
    - Record ``{tile_i, tile_j, provenance_sha256}`` in the result.

    Result is sorted by ``(tile_i, tile_j)`` for byte-determinism; the
    Task 13 validator and any downstream reader can rely on this ordering
    without a second sort.
    """
    entries: list[dict] = []
    for prov in tile_provenances:
        entries.append(
            {
                "tile_i": int(prov["tile_i"]),
                "tile_j": int(prov["tile_j"]),
                "provenance_sha256": provenance_sha256(prov),
            }
        )
    return sorted(entries, key=lambda e: (e["tile_i"], e["tile_j"]))


def write_manifest(data: dict, path: Path) -> None:
    """Serialise *data* to *path* using the neutral canonical YAML helper.

    Tiles are NOT re-sorted here; ``build_manifest`` /
    ``aggregate_tile_inventory`` already sort them.

    Does NOT write ``_SUCCESS``. ``write_success_marker`` is a separate
    explicit call so the Task 14 pipeline can gate it on the cross-tile
    validator passing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(data), encoding="utf-8")


def read_manifest(path: Path) -> dict:
    """Load *path* (manifest.yaml) and return the parsed dict.

    No semantic validation here — the Task 13 region validator handles
    cross-checks (config drift, version namespaces, provenance sha chain).
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_success_marker(region_dir: Path) -> None:
    """Touch ``<region_dir>/_SUCCESS`` as a zero-byte sentinel file.

    Per spec §11.8 (sub-C precedent applied to sub-D): this MUST be called
    LAST — after ``write_manifest`` AND after the cross-tile validator
    passes. The Task 14 pipeline enforces that ordering.

    Idempotent: if ``_SUCCESS`` already exists it is overwritten with zero
    bytes (supports the full-re-extraction protocol).
    """
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "_SUCCESS").write_bytes(b"")
