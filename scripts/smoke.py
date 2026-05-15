"""Phase 0 smoke: load the single-cell fixture, round-trip, print a deterministic summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# iCloud Drive hides underscore-prefixed .pth files in synced .venv, so the
# editable install of cfm may not be on sys.path. Mirror pytest's
# `pythonpath = ["src"]` workaround so this script runs both locally and on
# Leonardo.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cfm.tokenizer import (  # noqa: E402
    Vocabulary,
    decode_cell,
    encode_cell,
    geometric_equal,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    vocab = Vocabulary.load(repo_root / "configs" / "tokenizer" / "vocab_phase0.yaml")
    fixture_path = repo_root / "tests" / "fixtures" / "single_cell" / "input.geojson"
    with fixture_path.open() as f:
        original = json.load(f)

    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    equal = geometric_equal(original, decoded, tol_m=0.5)

    token_bytes = ",".join(str(t) for t in encoded.tokens).encode("utf-8")
    digest = hashlib.sha256(token_bytes).hexdigest()[:16]

    print("Phase 0 smoke")
    print(f"  vocabulary size:     {len(vocab)}")
    print(f"  fixture features:    {len(original['features'])}")
    print(f"  encoded token count: {len(encoded.tokens)}")
    print(f"  token id sha256/16:  {digest}")
    print(f"  geometric_equal:     {equal}")

    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
