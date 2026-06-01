"""known_issues #4 verification (training-scaffold Task 1).

Verifies the LIVE Phase-1 tokenizer path (sub-F encoder cascade #7) handles every
not-in-vocab building/POI/base class sub-C stores, on REAL Singapore data — the
verify-before-lock resolution of #4 (the Phase-0 cfm/tokenizer/encode.py it was
filed against is non-training-reachable, imported only by scripts/smoke.py).

One-source: builds semantic_tags via the SAME _semantic_tag_from_row sub-F's
encode_tile uses (pipeline_writer.py:42), never a reimplementation. The key map
is {0:highway, 1:building, 2:amenity (poi, BP4 fallback), 3:natural (base, BP4
fallback)}; value is always class_raw (None/sentinel -> "").
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.encoder import _resolve_semantic_tag_to_token_id
from cfm.data.sub_f.pipeline_writer import _semantic_tag_from_row
from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.eval.holdout.paths import sub_c_region_dir

_RELEASE, _REGION = "2026-04-15.0", "singapore"


def _distinct_real_semantic_tags() -> set[str]:
    """Every distinct semantic_tag sub-F builds from real sub-C rows, via the
    one-source constructor (never reimplemented here)."""
    region = sub_c_region_dir(_RELEASE, _REGION)
    tags: set[str] = set()
    for tile_dir in sorted(p for p in region.iterdir() if p.is_dir()):
        f = tile_dir / "features.parquet"
        if not f.exists():
            continue
        tbl = pq.ParquetFile(f).read(columns=["feature_class", "class_raw"])
        for row in tbl.to_pylist():
            tags.add(_semantic_tag_from_row(row))
    return tags


def test_unknown_classes_covered_nonvacuously_and_resolve_to_bp4():
    """(1) Counted, non-vacuous coverage: a non-zero count of distinct real
    semantic_tags must route to the <unknown_KEY> BP4 family (proving the
    unknown-class regime is exercised), and NONE may raise. Reports the count."""
    tag_to_id = vocab_tag_to_id()
    unknown_ids = {v for k, v in tag_to_id.items() if k.startswith("<unknown_")}
    distinct = _distinct_real_semantic_tags()
    routed_to_unknown = 0
    for tag in distinct:
        tid = _resolve_semantic_tag_to_token_id(tag)  # must not raise on any real tag
        if tid in unknown_ids:
            routed_to_unknown += 1
    print(
        f"[#4] distinct real semantic_tags={len(distinct)}; "
        f"routed to <unknown_KEY> (unknown-class regime exercised)={routed_to_unknown}"
    )
    assert routed_to_unknown > 0, "vacuous: no unknown-class regime present in real data"


def test_key_with_no_bp4_slot_raises_and_twin_resolves():
    """(2) Regime-distinguishing negative + must-pass twin: the encoder must
    FAIL LOUD on a key with no <unknown_KEY> slot (a true gap), not silently
    bucket everything; a key WITH a slot resolves (proves it isn't always-raising)."""
    tag_to_id = vocab_tag_to_id()
    with pytest.raises(KeyError):
        _resolve_semantic_tag_to_token_id("no_such_key=whatever")
    assert (
        _resolve_semantic_tag_to_token_id("building=__definitely_not_a_vocab_value__")
        == tag_to_id["<unknown_building>"]
    )


def test_unknown_building_roundtrip_is_known_lossy_collapse():
    """(3) Round-trip asserts the KNOWN loss: a not-in-vocab building class maps
    to the generic <unknown_building>, losing the specific class — the documented
    v1 limitation (protocol v3 §9 reported-not-gated), asserted explicitly."""
    tag_to_id = vocab_tag_to_id()
    tid = _resolve_semantic_tag_to_token_id("building=__definitely_not_a_vocab_value__")
    assert tid == tag_to_id["<unknown_building>"]
    assert "building=__definitely_not_a_vocab_value__" not in tag_to_id  # identity lost by design
