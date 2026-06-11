"""atomic_write_text — crash-safe training-manifest writes (known_issues F17).

Three guards: (a) content round-trips (utf-8); (b) an injected failure between
temp-write and replace leaves the prior destination byte-untouched AND no temp
behind (the torn-state regime the fix targets — the test FAILS on a plain
``write_text``); (c) ``_write_training_manifest`` actually routes through it
(observed via a recorder on ``os.replace`` as seen by the atomic_io module),
and the written manifest still byte-round-trips through ``canonicalize_yaml``
(determinism preserved by the new write path).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from cfm.data.io import canonicalize_yaml
from cfm.data.training.atomic_io import atomic_write_text
from cfm.data.training.build_shards import _write_training_manifest
from cfm.data.training.shard_schema import TrainingShard

_TEXT = "héllo — ünïcode\n"  # non-ASCII: pins the utf-8 default


def test_round_trip_content_lands(tmp_path):
    path = tmp_path / "manifest.yaml"
    atomic_write_text(path, _TEXT)
    assert path.read_text(encoding="utf-8") == _TEXT
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.yaml"]  # no temp left


def test_injected_failure_leaves_original_untouched_and_no_temp(tmp_path, monkeypatch):
    """The torn-state regime: failure between write and replace must not corrupt
    the prior-good file nor leave a temp behind."""
    path = tmp_path / "manifest.yaml"
    original = "prior good bytes\n"
    path.write_text(original, encoding="utf-8")

    def boom(src, dst):
        raise OSError("injected failure between write and replace")

    monkeypatch.setattr("cfm.data.training.atomic_io.os.replace", boom)
    with pytest.raises(OSError, match="injected failure"):
        atomic_write_text(path, _TEXT)
    assert path.read_bytes() == original.encode("utf-8")  # byte-unchanged
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.yaml"]  # no temp left


def _one_shard() -> TrainingShard:
    """Smallest schema-valid shard (manifest reads only ids + lineage)."""
    return TrainingShard(
        region="singapore",
        tile_i=0,
        tile_j=0,
        tile_conditioning={
            "dominant_zoning_class": None,
            "modal_road_skeleton_class": None,
            "admin_region": None,
        },
        macro_tokens=(),
        cells=(),
        lineage=frozenset({("singapore", 0, 0)}),
    )


def test_write_training_manifest_uses_atomic_write(tmp_path, monkeypatch):
    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def recorder(src, dst):
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("cfm.data.training.atomic_io.os.replace", recorder)
    _write_training_manifest(
        tmp_path, "2026-04-15.0", "singapore", [_one_shard()], {(0, 0): "deadbeef"}
    )

    manifest_path = tmp_path / "training_manifest.yaml"
    assert calls and calls[-1][1] == manifest_path  # routed through atomic_io
    # Determinism preserved: bytes still equal canonicalize_yaml of the parsed dict.
    text = manifest_path.read_text(encoding="utf-8")
    assert canonicalize_yaml(yaml.safe_load(text)) == text
