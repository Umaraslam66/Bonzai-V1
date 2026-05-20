from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from cfm.data.sub_e.provenance import (
    SubEInputDigests,
    SubEProvenance,
    SubEVersions,
    provenance_sha256,
    provenance_to_dict,
    write_provenance,
)


def _make_provenance(*, extracted_utc: str = "2026-05-21T12:00:00Z") -> SubEProvenance:
    return SubEProvenance(
        tile_i=12,
        tile_j=17,
        extraction_commit_sha="a" * 40,
        extracted_utc=extracted_utc,
        rerun_count=0,
        rerun_reason="initial",
        inputs=SubEInputDigests(
            release="2026-04-15.0",
            sub_c_manifest_sha256="b" * 64,
            sub_c_features_parquet_sha256="c" * 64,
            sub_c_crossings_parquet_sha256="d" * 64,
            sub_d_manifest_sha256="e" * 64,
            sub_d_macro_core_parquet_sha256="f" * 64,
            boundary_vocab_sha256="0" * 64,
            derivation_config_sha256="1" * 64,
        ),
        versions=SubEVersions(
            sub_e_schema_version="1.0",
            boundary_vocab_version="1.0",
            boundary_derivation_version="1.0",
        ),
        boundary_contract_parquet_sha256="2" * 64,
    )


def test_provenance_writes_canonical_yaml_with_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "provenance.yaml"
    write_provenance(p, _make_provenance())
    data = yaml.safe_load(p.read_text())
    assert data["tile_i"] == 12
    assert data["tile_j"] == 17
    assert data["versions"]["boundary_vocab_version"] == "1.0"
    assert data["inputs"]["sub_d_macro_core_parquet_sha256"] == "f" * 64
    assert data["outputs"]["boundary_contract_parquet_sha256"] == "2" * 64


def test_provenance_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    import hashlib

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    prov = _make_provenance()
    write_provenance(a, prov)
    write_provenance(b, prov)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_provenance_sha256_excludes_extracted_utc(tmp_path: Path) -> None:
    """Spec §9.2: extraction.extracted_utc is stripped before self-sha.

    Two provenance instances differing ONLY in extracted_utc must produce
    the same provenance_sha256, otherwise the digest chain breaks on every
    rerun under live clocks.
    """
    a = _make_provenance(extracted_utc="2026-05-21T12:00:00Z")
    b = _make_provenance(extracted_utc="2026-12-01T09:30:42Z")
    assert provenance_sha256(provenance_to_dict(a)) == provenance_sha256(provenance_to_dict(b))


def test_provenance_sha256_excludes_nested_sha_fields(tmp_path: Path) -> None:
    """Spec §9.2: final-segment *_sha256 fields are stripped before self-sha.

    Two provenance instances differing ONLY in a *_sha256 field (e.g. a
    different boundary_contract_parquet_sha256) must produce the same
    provenance_sha256.
    """
    a = _make_provenance()
    b = replace(a, boundary_contract_parquet_sha256="9" * 64)
    assert provenance_sha256(provenance_to_dict(a)) == provenance_sha256(provenance_to_dict(b))


def test_provenance_sha256_sensitive_to_semantic_changes(tmp_path: Path) -> None:
    """Inversely: non-excluded field changes MUST shift the self-sha.

    Guards against the table being too aggressive (over-stripping). A
    change in versions.boundary_vocab_version must produce a different
    sha.
    """
    a = _make_provenance()
    a_versions = replace(a.versions, boundary_vocab_version="2.0")
    b = replace(a, versions=a_versions)
    assert provenance_sha256(provenance_to_dict(a)) != provenance_sha256(provenance_to_dict(b))
