"""Write helpers and encoding-determinism primitives for sub-C tile extraction.

Per spec §14.3:
- _PARQUET_WRITE_KWARGS: pinned writer args for byte-deterministic parquet output.
- dump_wkb: explicit little-endian (NDR) WKB serialisation.
- canonicalize_yaml: byte-deterministic YAML serialisation (sorted keys, block style).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from shapely import wkb
from shapely.geometry.base import BaseGeometry

# DECISION: values match spec §14.3 verbatim. Snappy trades compression ratio for
# decompression speed, acceptable for tile data read frequently during training.
# Revisit if storage costs become a concern.
_PARQUET_WRITE_KWARGS: dict = {
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
    """Write *table* to *path* with pinned writer args from _PARQUET_WRITE_KWARGS."""
    pq.write_table(table, path, **_PARQUET_WRITE_KWARGS)


def dump_wkb(geom: BaseGeometry) -> bytes:
    """Serialise *geom* to WKB bytes with explicit little-endian (NDR) byte order.

    Per spec §14.3: byteorder=1 forces NDR regardless of platform default.
    The first byte of the result is always 0x01.
    """
    return wkb.dumps(geom, hex=False, byte_order=1)


def canonicalize_yaml(data: dict) -> str:
    """Serialise *data* to a byte-deterministic YAML string.

    Reuses B2's helper at cfm.data.vocab_derivation.canonicalize_yaml if
    available (identical settings). Falls back to a thin local version with
    the same locked yaml.dump settings per spec §14.3.
    """
    # Prefer B2's helper so there is exactly one canonical implementation.
    try:
        from cfm.data.vocab_derivation import canonicalize_yaml as b2_canon

        return b2_canon(data)
    except (ImportError, AttributeError):
        return yaml.dump(
            data,
            Dumper=yaml.SafeDumper,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
            width=4096,
        )
