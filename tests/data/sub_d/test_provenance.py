"""Tests for the sub-D per-tile provenance.yaml (Task 11).

Sub-D's provenance has two digest semantics simultaneously (tension flag B4):

- ``inputs.sub_c_*_sha256`` are sha256(file_bytes) values snapshotted by sub-D's
  reader at read time. They are recorded verbatim from the caller. They are
  themselves stripped from the provenance.yaml self-integrity sha via the
  final-segment ``*_sha256`` rule — the chain-of-custody is preserved, but
  re-hashing the sha string would just be hashing a hash.
- ``provenance.yaml``'s OWN self-integrity sha (recorded later in
  ``manifest.tiles[*].provenance_sha256``) is computed via
  ``compute_sha256_excluding(data, "provenance.yaml", SUB_D_EXCLUDED_FROM_SHA)``.
  Sub-D's exclusion table strips ``extraction.extracted_utc`` (wall-clock
  timestamp, varies between reruns of identical inputs) and any final-segment
  ``*_sha256`` field.

``extraction.rerun_reason`` is INCLUDED in the sha (per sub-C F2 precedent,
audit-trail purpose). The exclusion table must NOT contain it.

Per spec §11.5 + tension flag B7, the artifact carries a namespaced
``provenance_schema_version`` field, NOT a bare ``schema_version``.
"""

from __future__ import annotations

import copy

from cfm.data.sub_d.provenance import (
    SUB_D_EXCLUDED_FROM_SHA,
    build_tile_provenance,
    provenance_sha256,
    write_provenance,
)


def _extraction_fixture() -> dict:
    return {
        "commit_sha": "abc123def456" + "0" * 28,  # 40-char hex
        "extracted_utc": "2026-05-19T12:00:00Z",
        "rerun_count": 0,
        "rerun_reason": "initial",
    }


def _inputs_fixture() -> dict:
    return {
        "release": "2026-04-15.0",
        "sub_c_manifest_sha256": "1" * 64,
        "sub_c_tile_provenance_sha256": "2" * 64,
        "sub_c_cells_parquet_sha256": "3" * 64,
        "sub_c_features_parquet_sha256": "4" * 64,
        "sub_c_crossings_parquet_sha256": "5" * 64,
        "sub_c_meta_yaml_sha256": "6" * 64,
        "macro_vocab_sha256": "7" * 64,
        "derivation_config_sha256": "8" * 64,
    }


def _versions_fixture() -> dict:
    return {
        "sub_d_schema_version": "1.0",
        "macro_plan_vocab_version": "1.0",
        "zoning_vocab_version": "1.0",
        "zoning_derivation_version": "1.0",
        "cell_density_vocab_version": "1.0",
        "cell_density_derivation_version": "1.0",
        "tile_population_density_vocab_version": "1.0",
        "tile_population_density_derivation_version": "1.0",
        "road_skeleton_vocab_version": "1.0",
        "road_skeleton_derivation_version": "1.0",
    }


def _outputs_fixture() -> dict:
    return {
        "macro_core_parquet_sha256": "a" * 64,
        "derivation_evidence_parquet_sha256": "b" * 64,
        "effective_conditioning_yaml_sha256": "c" * 64,
    }


def _build_fixture(**overrides) -> dict:
    """Build a tile-provenance dict using fixture defaults, with optional
    keyword overrides applied to ``build_tile_provenance``'s named args.
    """
    kwargs = {
        "tile_i": 12,
        "tile_j": 17,
        "extraction": _extraction_fixture(),
        "inputs": _inputs_fixture(),
        "versions": _versions_fixture(),
        "outputs": _outputs_fixture(),
    }
    kwargs.update(overrides)
    return build_tile_provenance(**kwargs)


def test_provenance_schema_uses_provenance_schema_version_not_bare_schema_version():
    """Per spec §11.5 + tension flag B7, sub-D's provenance.yaml carries a
    namespaced ``provenance_schema_version`` field. Sub-C's TileProvenance uses
    bare ``schema_version`` — sub-D does NOT inherit that. Each per-artifact
    format has its own version (manifest_schema_version,
    effective_conditioning_schema_version, provenance_schema_version).
    """
    data = _build_fixture()
    assert "provenance_schema_version" in data
    assert data["provenance_schema_version"] == "1.0"
    # Conversely: no bare schema_version at the top level.
    assert "schema_version" not in data


def test_provenance_records_sub_c_input_digests():
    """The ``inputs`` block records sub-D's view of the upstream sub-C
    artifacts: release tag plus all sha256s of the consumed sub-C files
    (manifest, per-tile provenance, cells/features/crossings parquet, meta.yaml)
    and the locked macro-plan artifacts (vocab + derivation config).

    These are recorded verbatim from the caller's digest dict — sub-D does NOT
    re-hash sub-C files here; the digests were snapshotted by sub-D's reader
    at read time (B4: bytes-sha semantics, not excluding-timestamp).
    """
    inputs = _inputs_fixture()
    data = _build_fixture(inputs=inputs)

    assert data["inputs"]["release"] == "2026-04-15.0"
    assert data["inputs"]["sub_c_manifest_sha256"] == "1" * 64
    assert data["inputs"]["sub_c_tile_provenance_sha256"] == "2" * 64
    assert data["inputs"]["sub_c_cells_parquet_sha256"] == "3" * 64
    assert data["inputs"]["sub_c_features_parquet_sha256"] == "4" * 64
    assert data["inputs"]["sub_c_crossings_parquet_sha256"] == "5" * 64
    assert data["inputs"]["sub_c_meta_yaml_sha256"] == "6" * 64
    assert data["inputs"]["macro_vocab_sha256"] == "7" * 64
    assert data["inputs"]["derivation_config_sha256"] == "8" * 64


def test_provenance_records_locked_vocab_and_derivation_versions():
    """The ``versions`` block records every locked vocab + derivation-version
    that contributed to the per-tile output. Per tension flag A1, derivation
    versions are per-namespace (zoning/cell_density/tile_population_density/
    road_skeleton) — there is NO single global ``derivation_version`` field.

    sub_d_schema_version (the sub-D code-level schema) is recorded alongside
    the per-namespace vocab/derivation versions.
    """
    data = _build_fixture()

    v = data["versions"]
    assert v["sub_d_schema_version"] == "1.0"
    assert v["macro_plan_vocab_version"] == "1.0"

    # Per-namespace vocab + derivation versions (A1 discipline).
    for namespace in (
        "zoning",
        "cell_density",
        "tile_population_density",
        "road_skeleton",
    ):
        assert f"{namespace}_vocab_version" in v, namespace
        assert f"{namespace}_derivation_version" in v, namespace

    # No single global derivation_version field — that would conflict with A1.
    assert "derivation_version" not in v


def test_provenance_sha_excludes_extracted_utc_and_output_sha_fields():
    """provenance_sha256 strips:

    - ``extraction.extracted_utc`` — wall-clock timestamp, varies between
      reruns of identical-input pipelines.
    - any final-segment ``*_sha256`` field (sub-C's chain-of-custody
      convention; re-hashing a digest string is meaningless).

    Two provenance dicts that differ ONLY in those fields must hash equal.
    """
    base = _build_fixture()

    # (1) Different extracted_utc — sha must NOT change.
    later = copy.deepcopy(base)
    later["extraction"]["extracted_utc"] = "2099-12-31T23:59:59Z"
    assert provenance_sha256(base) == provenance_sha256(later)

    # (2) Different output sha values — sha must NOT change (final-segment
    # *_sha256 stripped). This is the chain-of-custody rule: provenance.yaml
    # records the output digests for downstream readers, but its OWN
    # self-integrity sha cannot circularly include them.
    different_outputs = copy.deepcopy(base)
    different_outputs["outputs"]["macro_core_parquet_sha256"] = "f" * 64
    different_outputs["outputs"]["derivation_evidence_parquet_sha256"] = "e" * 64
    different_outputs["outputs"]["effective_conditioning_yaml_sha256"] = "d" * 64
    assert provenance_sha256(base) == provenance_sha256(different_outputs)

    # (3) Different input sha values — sha must NOT change either; same
    # final-segment rule applies. (Note: this means a sub-C input drift is
    # caught by the validator comparing inputs.* against the live sub-C
    # digests, not by the provenance self-sha. That is the intentional B4
    # division of labour: bytes-sha for upstream view, excluding-timestamp
    # for self-integrity.)
    different_inputs = copy.deepcopy(base)
    different_inputs["inputs"]["sub_c_manifest_sha256"] = "f" * 64
    assert provenance_sha256(base) == provenance_sha256(different_inputs)

    # Sanity-check the exclusion table contents — the assertions above rely
    # on this exact grammar.
    assert "*_sha256" in SUB_D_EXCLUDED_FROM_SHA["*"]
    assert "extraction.extracted_utc" in SUB_D_EXCLUDED_FROM_SHA["provenance.yaml"]


def test_provenance_sha_includes_rerun_reason():
    """Tension flag B4: ``extraction.rerun_reason`` is INCLUDED in the sha
    (audit-trail purpose, per sub-C F2 precedent). Changing rerun_reason
    must change the sha.

    Also pin the invariant in SUB_D_EXCLUDED_FROM_SHA directly — future
    edits that add rerun_reason to the exclusion table would silently
    drop the audit-trail guarantee, so the test asserts on the table too.
    """
    base = _build_fixture()
    changed = copy.deepcopy(base)
    changed["extraction"]["rerun_reason"] = "reran after sub-C v1.1 bump"

    assert provenance_sha256(base) != provenance_sha256(changed)

    # rerun_reason must NOT be in either the wildcard or the
    # provenance.yaml-keyed exclusion list.
    for entries in SUB_D_EXCLUDED_FROM_SHA.values():
        assert "extraction.rerun_reason" not in entries
        assert "*rerun_reason" not in entries


# ---------------------------------------------------------------------------
# Companion: write_provenance produces byte-deterministic canonical YAML.
# Not in the plan's 5 named tests, but small and pinned the same way Task 10
# did for write_effective_conditioning.
# ---------------------------------------------------------------------------


def test_write_provenance_is_byte_deterministic(tmp_path):
    data = _build_fixture()
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    write_provenance(data, a)
    write_provenance(data, b)
    assert a.read_bytes() == b.read_bytes()
