"""T12 CLI: scripts/sub_f/encode.py — GeoJSON geometry → token array.

Contract under test (§13.1 T12 lock):
  - Input: ONE GeoJSON geometry via stdin OR --geom-file.
  - --semantic-tag REQUIRED (passes through to encode_feature).
  - --inbound-bref / --outbound-bref take DIR:CLASS (e.g. E:MAJOR_ROAD) and are
    resolved via resolve_bref_tag — the exact production path. resolve_bref_tag
    raises only on a bad DIRECTION; it returns None for a non-emitting class, so
    the CLI adds a flag-provided⟹non-None guard for the bad-class case.
  - --cell-origin x,y is a CLI-side pre-translation (translate by -x,-y) applied
    before canonicalize + encode, faithful to how encode_cell positions a cell's
    features. Default (0,0).
  - --round-trip reports L_inf (vs the CANONICALIZED input) to stderr; the token
    array still goes to stdout (pipe-friendly).
  - Output: a JSON array of integer token IDs on stdout, one line.

Negatives assert a rule-specific stderr substring (the CLI-input analogue of the
T9/T10 leg-isolation standard): a missing-direction failure reports a format
problem, a bad-direction failure reports a direction problem, and a bad-class
failure reports a class problem — three distinct messages, so one cannot satisfy
another's assertion.
"""

from __future__ import annotations

import io
import json

import pytest
from shapely.geometry import LineString

from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature
from cfm.data.sub_f.vocab import vocab_tag_to_id
from scripts.sub_f.encode import main

_TAG = "highway=residential"


def _feed_geom(monkeypatch, geom_dict: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(geom_dict)))


def test_encode_stdin_case_a_emits_token_array(monkeypatch, capsys):
    geom = {"type": "LineString", "coordinates": [[0, 0], [10, 0], [10, 10]]}
    _feed_geom(monkeypatch, geom)

    rc = main(["--semantic-tag", _TAG])

    assert rc == 0
    tokens = json.loads(capsys.readouterr().out)
    assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
    assert tokens[0] == 509 and tokens[-1] == 510  # <feature> … <feature_end>
    # Full equivalence with the library path (Case A, no brefs).
    expected = encode_feature(
        canonicalize_geometry(LineString([(0, 0), (10, 0), (10, 10)])),
        semantic_tag=_TAG,
    ).tokens
    assert tokens == expected


def test_encode_geom_file(tmp_path, capsys):
    geom_file = tmp_path / "g.geojson"
    geom_file.write_text(json.dumps({"type": "LineString", "coordinates": [[0, 0], [10, 0]]}))

    rc = main(["--semantic-tag", _TAG, "--geom-file", str(geom_file)])

    assert rc == 0
    tokens = json.loads(capsys.readouterr().out)
    assert tokens[0] == 509 and tokens[-1] == 510


def test_encode_missing_semantic_tag_exits_2(monkeypatch):
    _feed_geom(monkeypatch, {"type": "LineString", "coordinates": [[0, 0], [10, 0]]})
    with pytest.raises(SystemExit) as exc:  # argparse required-arg error
        main([])
    assert exc.value.code == 2


def test_encode_malformed_json_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("definitely not json {["))
    rc = main(["--semantic-tag", _TAG])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "json" in err


def test_encode_inbound_bref_produces_case_c(monkeypatch, capsys):
    geom = {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}
    _feed_geom(monkeypatch, geom)

    rc = main(["--semantic-tag", _TAG, "--inbound-bref", "W:MINOR_ROAD"])

    assert rc == 0
    tokens = json.loads(capsys.readouterr().out)
    bref_id = vocab_tag_to_id()["<bref_W_MINOR>"]
    # Case C layout: [<feature>, semantic_tag, <bref_in>, anchor…]
    assert tokens[2] == bref_id


def test_encode_bref_missing_direction_exits_nonzero(monkeypatch, capsys):
    _feed_geom(monkeypatch, {"type": "LineString", "coordinates": [[0, 0], [10, 0]]})
    rc = main(["--semantic-tag", _TAG, "--inbound-bref", "MAJOR_ROAD"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    # Leg isolation: a FORMAT failure (no DIR:CLASS), not a direction/class one.
    assert "dir:class" in err or "separator" in err or "':'" in err


def test_encode_bref_invalid_direction_exits_nonzero(monkeypatch, capsys):
    _feed_geom(monkeypatch, {"type": "LineString", "coordinates": [[0, 0], [10, 0]]})
    rc = main(["--semantic-tag", _TAG, "--inbound-bref", "X:MAJOR_ROAD"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    # Leg isolation: resolve_bref_tag's DIRECTION error surfaced.
    assert "direction" in err


def test_encode_bref_invalid_class_exits_nonzero(monkeypatch, capsys):
    _feed_geom(monkeypatch, {"type": "LineString", "coordinates": [[0, 0], [10, 0]]})
    rc = main(["--semantic-tag", _TAG, "--inbound-bref", "E:NOT_A_CLASS"])
    assert rc != 0
    err = capsys.readouterr().err.lower()
    # Leg isolation: the flag-provided⟹non-None guard for a non-emitting class.
    # resolve_bref_tag returns None here (does NOT raise), so this must be a
    # CLASS message, distinct from the direction message above.
    assert "class" in err
    assert "direction" not in err


def test_encode_round_trip_reports_linf_to_stderr(monkeypatch, capsys):
    # Grid-aligned 0.5m coords → Case A round-trips ~exactly.
    geom = {"type": "LineString", "coordinates": [[0, 0], [10, 0], [10, 10]]}
    _feed_geom(monkeypatch, geom)

    rc = main(["--semantic-tag", _TAG, "--round-trip"])

    assert rc == 0
    captured = capsys.readouterr()
    # Tokens still on stdout (pipe-friendly).
    tokens = json.loads(captured.out)
    assert tokens[0] == 509
    # L_inf diagnostic on stderr.
    err = captured.err.lower()
    assert "round-trip" in err and "l_inf" in err
    assert "case=a" in err


def test_encode_cell_origin_pre_translation(monkeypatch, capsys):
    # A feature 1km from the projection origin, with --cell-origin set to its SW
    # corner, must produce the SAME tokens as the cell-local geometry at (0,0).
    far = {"type": "LineString", "coordinates": [[1000, 2000], [1010, 2000]]}
    _feed_geom(monkeypatch, far)

    rc = main(["--semantic-tag", _TAG, "--cell-origin", "1000,2000"])

    assert rc == 0
    tokens = json.loads(capsys.readouterr().out)
    expected = encode_feature(
        canonicalize_geometry(LineString([(0, 0), (10, 0)])),
        semantic_tag=_TAG,
    ).tokens
    assert tokens == expected
