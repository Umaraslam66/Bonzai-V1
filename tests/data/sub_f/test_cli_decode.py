"""T12 CLI: scripts/sub_f/decode.py — token-array → bare GeoJSON geometry.

Contract under test (§13.1 T12 lock):
  - Input: JSON int-array via stdin OR a positional token-file path.
  - Output: a BARE GeoJSON geometry (NOT a Feature / FeatureCollection) on
    stdout, one compact line (serialize_geojson). This is what `encode`
    consumes, so `decode tokens.json | encode --semantic-tag …` round-trips.
  - Errors: malformed JSON → nonzero exit + clear stderr; a token list that
    is not a well-formed feature (missing <feature>/<feature_end> markers,
    too short) → nonzero exit + clear stderr.

Negatives assert a rule-specific stderr substring (the CLI-input analogue of
the T9/T10 leg-isolation standard): a malformed-JSON failure must report a
JSON problem, NOT a marker problem, and vice-versa, so a different failure
mode cannot satisfy the assertion.
"""

from __future__ import annotations

import io
import json

from shapely.geometry import LineString

from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature
from scripts.sub_f.decode import main


def _valid_case_a_tokens() -> list[int]:
    """A well-formed Case-A (no-bref) feature token sequence."""
    geom = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    ef = encode_feature(canonicalize_geometry(geom), semantic_tag="highway=residential")
    return ef.tokens


def test_decode_stdin_emits_bare_geometry(monkeypatch, capsys):
    tokens = _valid_case_a_tokens()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(tokens)))

    rc = main([])

    assert rc == 0
    out = capsys.readouterr().out.strip()
    geom = json.loads(out)
    # BARE geometry contract: a geometry object, never a Feature wrapper.
    assert geom["type"] in {"Point", "LineString"}
    assert "coordinates" in geom
    assert geom["type"] not in {"Feature", "FeatureCollection"}
    assert "geometry" not in geom and "properties" not in geom
    # One compact line (serialize_geojson uses indent=None).
    assert "\n" not in out


def test_decode_positional_token_file(tmp_path, capsys):
    tokens = _valid_case_a_tokens()
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps(tokens))

    rc = main([str(token_file)])

    assert rc == 0
    geom = json.loads(capsys.readouterr().out)
    assert geom["type"] == "LineString"
    assert len(geom["coordinates"]) == 3


def test_decode_malformed_json_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json {{["))

    rc = main([])

    assert rc != 0
    err = capsys.readouterr().err.lower()
    # Rule isolation: this is a JSON-parse failure, not a token/marker failure.
    assert "json" in err
    assert "marker" not in err and "feature" not in err


def test_decode_non_int_array_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"type": "LineString"})))

    rc = main([])

    assert rc != 0
    err = capsys.readouterr().err.lower()
    # Input parsed as JSON but is not a list-of-ints token array.
    assert "array" in err or "list" in err or "int" in err


def test_decode_too_short_token_list_exits_nonzero(monkeypatch, capsys):
    # [509] = <feature> with no <feature_end>: a malformed feature, not a
    # JSON problem. Must fail for the decode/marker reason, not a JSON reason.
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([509])))

    rc = main([])

    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "json" not in err  # leg isolation: NOT a JSON failure
    assert "decode" in err or "feature" in err or "marker" in err
