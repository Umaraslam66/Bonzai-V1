"""T12 CLI: decode | encode round-trips (bare-geometry contract).

The §13.1 T12 decision that decode EMITS and encode CONSUMES a bare geometry is
load-bearing precisely so the two compose in a pipe. These tests pin that: a
Case-A feature's tokens survive `decode → encode` unchanged.

  - in-process: decode's stdout fed as encode's stdin (models the OS pipe; fast).
  - @slow: a literal `decode.py | encode.py` OS pipe via subprocess, which also
    exercises the `__main__` entry + the iCloud-safe sys.path inject.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString

from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature
from scripts.sub_f.decode import main as decode_main
from scripts.sub_f.encode import main as encode_main

_TAG = "highway=residential"
_REPO = Path(__file__).resolve().parents[3]


def _case_a_tokens() -> list[int]:
    return encode_feature(
        canonicalize_geometry(LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])),
        semantic_tag=_TAG,
    ).tokens


def test_decode_then_encode_roundtrips_case_a(tmp_path, monkeypatch, capsys):
    tokens1 = _case_a_tokens()
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps(tokens1))

    # decode (file) → bare geometry on stdout
    assert decode_main([str(token_file)]) == 0
    geom_json = capsys.readouterr().out

    # pipe: decode's stdout becomes encode's stdin
    monkeypatch.setattr("sys.stdin", io.StringIO(geom_json))
    assert encode_main(["--semantic-tag", _TAG]) == 0
    tokens2 = json.loads(capsys.readouterr().out)

    assert tokens2 == tokens1


@pytest.mark.slow
def test_decode_encode_os_pipe(tmp_path):
    tokens1 = _case_a_tokens()
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps(tokens1))

    decode_py = _REPO / "scripts" / "sub_f" / "decode.py"
    encode_py = _REPO / "scripts" / "sub_f" / "encode.py"

    p1 = subprocess.Popen([sys.executable, str(decode_py), str(token_file)], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        [sys.executable, str(encode_py), "--semantic-tag", _TAG],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
    )
    assert p1.stdout is not None
    p1.stdout.close()  # allow p1 to receive SIGPIPE if p2 exits
    out, _ = p2.communicate(timeout=60)
    p1.wait(timeout=60)

    assert p1.returncode == 0 and p2.returncode == 0
    assert json.loads(out) == tokens1
