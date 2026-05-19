"""Tests for the neutral shared I/O helpers in cfm.data.io.

These tests pin the public contract that sub-C (and later sub-D) consume
via thin wrappers. Settings come from sub-C spec §14.3.
"""

from __future__ import annotations

from cfm.data.io import PARQUET_WRITE_KWARGS, canonicalize_yaml


def test_shared_parquet_write_kwargs_match_sub_c_contract():
    assert PARQUET_WRITE_KWARGS["compression"] == "snappy"
    assert PARQUET_WRITE_KWARGS["row_group_size"] == 50_000
    assert PARQUET_WRITE_KWARGS["data_page_size"] == 1_048_576
    assert PARQUET_WRITE_KWARGS["write_batch_size"] == 10_000
    assert PARQUET_WRITE_KWARGS["use_dictionary"] is True
    assert PARQUET_WRITE_KWARGS["write_statistics"] is True
    assert PARQUET_WRITE_KWARGS["use_compliant_nested_type"] is True
    assert PARQUET_WRITE_KWARGS["version"] == "2.6"


def test_shared_canonicalize_yaml_is_byte_stable_and_sorted():
    data = {"z": 1, "a": {"b": 2, "a": 1}}
    first = canonicalize_yaml(data)
    second = canonicalize_yaml(data)
    assert first == second
    assert first.splitlines()[0].startswith("a:")
