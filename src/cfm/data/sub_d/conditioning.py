"""Per-tile effective_conditioning.yaml overlay (sub-D spec §11.4, Task 10).

Sub-D writes the complete consumer-facing tile conditioning vector. Every
field that sub-C owns flows through verbatim; sub-D fills the fields it
owns (currently just ``population_density_bucket``).

Copy rule (schema-driven, NOT a static allowlist of sub-C field names):

- Iterate ``meta["conditioning_per_tile"]`` items.
- Skip any key ending in ``_owner`` — those are markers about who fills a
  field, not conditioning values themselves.
- Skip any key in ``SUB_D_OWNED_FIELDS`` — sub-D will fill those with its
  own derivation.
- Copy everything else verbatim.

Then merge in ``manifest["conditioning_defaults"]`` (region-constant
sub-C-owned fields like ``country`` and ``climate_zone``) and finally add
sub-D's computed ``population_density_bucket``.

The schema-driven rule keeps the overlay safe against future sub-C
schema additions: a new sub-C-owned field appears in the output without
any code change in sub-D. A new sub-D-owned field requires bumping
``SUB_D_OWNED_FIELDS`` plus a corresponding sub-D derivation function.

Version field naming: per spec §11.4 + plan Task 11's discipline, sub-D
uses ``effective_conditioning_schema_version`` (namespaced), NOT a bare
``schema_version``. Each per-artifact format has its own version.
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.io import canonicalize_yaml

#: Version of the effective_conditioning.yaml artifact format. Bumped when
#: the YAML structure changes (e.g. new section, renamed field). Does NOT
#: bump when vocab or derivation versions change — those are independent
#: per spec §12.
EFFECTIVE_CONDITIONING_SCHEMA_VERSION: str = "1.0"

#: Conditioning fields sub-D owns and fills with its own derivation. Every
#: other field in sub-C's conditioning_per_tile / conditioning_defaults
#: flows through unchanged. Bumping this set is a sub-D schema change.
SUB_D_OWNED_FIELDS: set[str] = {"population_density_bucket"}


def _is_owner_marker(field_name: str) -> bool:
    """Return True for fields like ``population_density_bucket_owner`` whose
    purpose is to mark which sub-project owns a sibling field, not to carry
    conditioning data themselves.
    """
    return field_name.endswith("_owner")


def build_effective_conditioning(
    meta: dict,
    manifest: dict,
    population_density_bucket: int,
    versions: dict,
    digests: dict,
) -> dict:
    """Build the effective_conditioning.yaml dict for one tile.

    Parameters:
        meta: parsed sub-C ``meta.yaml`` for this tile. Reads
            ``tile_i``, ``tile_j``, and ``conditioning_per_tile``.
        manifest: parsed sub-C ``manifest.yaml`` for the region. Reads
            ``conditioning_defaults``.
        population_density_bucket: the sub-D-owned token_id derived
            upstream (typically by Task 14's pipeline: take the locked
            proxy from the macro vocab, look up the bucket whose
            ``[lower_inclusive, upper_exclusive)`` interval contains the
            tile's proxy value).
        versions: composite version dict; minimally
            ``sub_c_conditioning_schema_version``,
            ``tile_population_density_vocab_version``,
            ``tile_population_density_derivation_version``.
        digests: sub-C input digest dict; minimally
            ``manifest_sha256``, ``tile_meta_sha256``,
            ``tile_provenance_sha256``.
    """
    conditioning: dict = {}

    # Merge manifest-level defaults first (region-constant fields like
    # country, climate_zone). These are sub-C-owned by definition.
    for key, value in (manifest.get("conditioning_defaults") or {}).items():
        if _is_owner_marker(key) or key in SUB_D_OWNED_FIELDS:
            continue
        conditioning[key] = value

    # Then per-tile fields. Schema-driven copy.
    for key, value in (meta.get("conditioning_per_tile") or {}).items():
        if _is_owner_marker(key) or key in SUB_D_OWNED_FIELDS:
            continue
        conditioning[key] = value

    # Finally sub-D's filled field.
    conditioning["population_density_bucket"] = int(population_density_bucket)

    return {
        "effective_conditioning_schema_version": EFFECTIVE_CONDITIONING_SCHEMA_VERSION,
        "tile_i": int(meta["tile_i"]),
        "tile_j": int(meta["tile_j"]),
        "versions": dict(versions),
        "sub_c_inputs": dict(digests),
        "conditioning": conditioning,
    }


def write_effective_conditioning(data: dict, path: Path) -> None:
    """Serialise *data* to *path* using the neutral canonical YAML helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(data), encoding="utf-8")
