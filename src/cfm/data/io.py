"""Neutral I/O helpers shared by sub-C and later sidecar layers (sub-D, ...).

Carries the byte-deterministic parquet writer kwargs (spec §14.3) and the
canonical YAML serialiser. Sub-layer wrappers re-export these to preserve
their existing public names.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PARQUET_WRITE_KWARGS: dict = {
    "compression": "snappy",
    "row_group_size": 50_000,
    "data_page_size": 1_048_576,
    "write_batch_size": 10_000,
    "use_dictionary": True,
    "write_statistics": True,
    "use_compliant_nested_type": True,
    "version": "2.6",
}


def write_parquet(table: pa.Table, path: Path) -> None:
    """Write *table* to *path* using PARQUET_WRITE_KWARGS for byte determinism."""
    pq.write_table(table, path, **PARQUET_WRITE_KWARGS)


def canonicalize_yaml(data: dict) -> str:
    """Serialise *data* to a byte-deterministic YAML string.

    Locked settings per spec §14.3: SafeDumper, sorted keys, block style,
    unicode allowed, 2-space indent, 4096-column width. Equivalent to
    ``yaml.safe_dump`` with the same kwargs.
    """
    return yaml.dump(
        data,
        Dumper=yaml.SafeDumper,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        width=4096,
    )
