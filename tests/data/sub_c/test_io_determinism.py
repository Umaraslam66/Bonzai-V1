"""Tests for Task 8: io.py + determinism.py (sub-C tile extraction).

Covers:
- Determinism category E: pinned parquet writer kwargs + WKB byte-order + YAML canon
- Determinism category F: sha excludes timestamps but INCLUDES rerun_reason
"""

from __future__ import annotations

from shapely import wkb as shapely_wkb
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# io.py tests
# ---------------------------------------------------------------------------


def test_parquet_write_kwargs_pinned_per_spec_14_3() -> None:
    """_PARQUET_WRITE_KWARGS has the right keys per spec §14.3."""
    from cfm.data.sub_c.io import _PARQUET_WRITE_KWARGS

    assert _PARQUET_WRITE_KWARGS["compression"] == "snappy"
    assert _PARQUET_WRITE_KWARGS["row_group_size"] == 50_000
    assert _PARQUET_WRITE_KWARGS["data_page_size"] == 1_048_576
    assert _PARQUET_WRITE_KWARGS["write_batch_size"] == 10_000
    assert _PARQUET_WRITE_KWARGS["use_dictionary"] is True
    assert _PARQUET_WRITE_KWARGS["write_statistics"] is True
    assert _PARQUET_WRITE_KWARGS["use_compliant_nested_type"] is True
    assert _PARQUET_WRITE_KWARGS["version"] == "2.6"


def test_dump_wkb_byte_order_little_endian() -> None:
    """Point→WKB; first byte is 0x01 (little-endian NDR marker).

    Then roundtrip: wkb.loads(dump_wkb(p)) == p.
    """
    from cfm.data.sub_c.io import dump_wkb

    p = Point(1.0, 2.0)
    wkb_bytes = dump_wkb(p)

    # First byte must be 0x01 = little-endian / NDR
    assert wkb_bytes[0] == 0x01, f"Expected NDR byte-order marker 0x01, got {wkb_bytes[0]:#04x}"

    # Roundtrip
    recovered = shapely_wkb.loads(wkb_bytes)
    assert recovered == p


def test_canonicalize_yaml_sorted_keys_byte_deterministic() -> None:
    """dump same data twice; both runs produce identical bytes. Keys sorted."""
    from cfm.data.sub_c.io import canonicalize_yaml

    data = {"z_key": 3, "a_key": 1, "m_key": {"nested_b": 2, "nested_a": 1}}

    out1 = canonicalize_yaml(data)
    out2 = canonicalize_yaml(data)

    # Byte-identical across two calls
    assert out1 == out2, "canonicalize_yaml is not byte-deterministic"

    # Keys must be sorted (a_key before m_key before z_key)
    lines = out1.splitlines()
    key_lines = [line for line in lines if ":" in line and not line.startswith(" ")]
    keys_in_order = [line.split(":")[0].strip() for line in key_lines]
    assert keys_in_order == sorted(keys_in_order), f"Top-level keys are not sorted: {keys_in_order}"


# ---------------------------------------------------------------------------
# determinism.py tests
# ---------------------------------------------------------------------------


def test_sha256_excludes_sha256_fields_via_wildcard() -> None:
    """data = {"a": 1, "vocab_sha256": "abc"}; sha excluding "*_sha256" matches
    sha of {"a": 1}.
    """
    from cfm.data.sub_c.determinism import compute_sha256, compute_sha256_excluding
    from cfm.data.sub_c.io import canonicalize_yaml

    data_with_sha = {"a": 1, "vocab_sha256": "abc"}
    data_without_sha = {"a": 1}

    # sha of data_with_sha (excluding *_sha256 fields) should equal sha of data_without_sha
    sha_excluded = compute_sha256_excluding(data_with_sha, file_key="some_file.yaml")
    sha_clean = compute_sha256(canonicalize_yaml(data_without_sha).encode("utf-8"))

    assert sha_excluded == sha_clean


def test_sha256_excludes_started_utc_from_manifest() -> None:
    """sha excluding initial_extraction.started_utc for file_key='manifest.yaml'
    ignores the timestamp but keeps tile_count.
    """
    from cfm.data.sub_c.determinism import compute_sha256_excluding

    data_with_ts = {
        "initial_extraction": {
            "started_utc": "2026-01-01T00:00:00Z",
            "tile_count": 187,
        }
    }
    data_different_ts = {
        "initial_extraction": {
            "started_utc": "2026-06-15T12:34:56Z",
            "tile_count": 187,
        }
    }

    sha1 = compute_sha256_excluding(data_with_ts, file_key="manifest.yaml")
    sha2 = compute_sha256_excluding(data_different_ts, file_key="manifest.yaml")

    assert sha1 == sha2, "Different timestamps should not change the sha for manifest.yaml"


def test_sha256_does_not_exclude_rerun_reason() -> None:
    """rerun_reason IS included in sha (per F2 spec fix — not in EXCLUDED_FROM_SHA).

    data = {"extraction": {"rerun_reason": "initial"}} and
           {"extraction": {"rerun_reason": "audit"}}
    produce DIFFERENT shas.
    """
    from cfm.data.sub_c.determinism import compute_sha256_excluding

    data_initial = {"extraction": {"rerun_reason": "initial"}}
    data_audit = {"extraction": {"rerun_reason": "audit"}}

    sha_initial = compute_sha256_excluding(data_initial, file_key="provenance.yaml")
    sha_audit = compute_sha256_excluding(data_audit, file_key="provenance.yaml")

    assert sha_initial != sha_audit, (
        "rerun_reason must be INCLUDED in sha; different values must produce "
        "different digests. Check that 'rerun_reason' is NOT in EXCLUDED_FROM_SHA."
    )
