"""Reader + validator for the locked macro vocab artifact (Task 8).

The locked macro vocab is the Phase A -> Phase B handoff:
``configs/macro_plan/v1/macro_plan_vocab.yaml``. It is byte-identical to the
reviewed ``reports/phase-1-sub-D/macro_vocab_proposal.yaml`` modulo only the
``status: proposal`` -> ``status: locked`` line; ``promote_macro_vocab.py``
flips that single marker. Consumers go through this module to read token
IDs / names rather than parsing the YAML directly.

The vocab structure (per the proposal-index schema, locked verbatim):

```yaml
status: locked
analysis_version: 1.0
derivation_versions:
  zoning: "1.0"
  cell_density: "1.0"
  tile_population_density: "1.0"
  road_skeleton: "1.0"
locked_buckets:
  zoning: [{token_id, token_name, count}, ...]
  cell_density: [{token_id, token_name, lower_inclusive, upper_exclusive}, ...]
  road_skeleton: [{token_id, token_name, lower_inclusive, upper_exclusive}, ...]
  tile_population_density: [{token_id, token_name, lower_inclusive, upper_exclusive}, ...]
locked_proxy:
  tile_population_density: "<proxy_name>"
append_only_within_phase:
  cell_density: true
  road_skeleton: true
  tile_population_density: true
  zoning: true
namespace_files: [{filename, section_key, sha256}, ...]
# ...plus per_tile_evidence, zoning_orthogonality, input_digests,
# selected_layer3_tiles, etc. (carried verbatim from the proposal).
```
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.sub_d.errors import SubDValidationError

#: Reviewer-locked vocab namespaces. Every locked vocab must contain a
#: ``locked_buckets`` entry and an ``append_only_within_phase`` flag for
#: each of these names.
LOCKED_VOCAB_NAMESPACES: tuple[str, ...] = (
    "cell_density",
    "road_skeleton",
    "tile_population_density",
    "zoning",
)


def load_macro_vocab(path: Path) -> dict:
    """Load the locked macro vocab from *path* and validate its shape."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_macro_vocab(data)
    return data


def validate_macro_vocab(data: dict) -> None:
    """Raise ``SubDValidationError`` on malformed or unsafe vocab.

    Checks (all four are spec §11.7 requirements plus Phase 1 sub-D's
    append-only-within-phase contract):

    1. Required top-level keys present.
    2. ``status`` is ``"locked"`` (proposal artifacts must not be passed
       here; they go through ``promote_macro_vocab.py`` first).
    3. Every namespace has a non-empty ``locked_buckets`` list.
    4. Every namespace has ``append_only_within_phase: true``.
    5. Within each namespace, ``token_id`` values are unique.
    6. Within each namespace, ``token_name`` values are unique.
    7. ``tile_population_density`` carries a ``locked_proxy`` string.
    """
    required_top = {"status", "locked_buckets", "locked_proxy", "append_only_within_phase"}
    missing = required_top - data.keys()
    if missing:
        raise SubDValidationError(
            f"macro vocab missing required top-level keys: {sorted(missing)}"
        )
    if data["status"] != "locked":
        raise SubDValidationError(
            f"macro vocab status must be 'locked'; got {data['status']!r}. "
            "Did you forget to run promote_macro_vocab.py first?"
        )

    for namespace in LOCKED_VOCAB_NAMESPACES:
        buckets = data["locked_buckets"].get(namespace)
        if not buckets:
            raise SubDValidationError(
                f"macro vocab namespace {namespace!r} has empty locked_buckets"
            )
        flag = data["append_only_within_phase"].get(namespace)
        if flag is not True:
            raise SubDValidationError(
                f"macro vocab namespace {namespace!r} missing "
                f"append_only_within_phase=true; got {flag!r}"
            )
        ids = [entry["token_id"] for entry in buckets]
        if len(ids) != len(set(ids)):
            raise SubDValidationError(
                f"macro vocab namespace {namespace!r} has duplicate token_id: {ids}"
            )
        names = [entry["token_name"] for entry in buckets]
        if len(names) != len(set(names)):
            raise SubDValidationError(
                f"macro vocab namespace {namespace!r} has duplicate token_name: {names}"
            )

    if not data["locked_proxy"].get("tile_population_density"):
        raise SubDValidationError(
            "macro vocab missing locked_proxy[tile_population_density]"
        )


def token_name_to_id(section: str, token_name: str, vocab: dict) -> int:
    """Return the ``token_id`` for ``token_name`` in ``section``.

    Raises ``KeyError`` if the section or name is not in the vocab.
    """
    buckets = vocab["locked_buckets"].get(section)
    if buckets is None:
        raise KeyError(f"section {section!r} not in macro vocab")
    for entry in buckets:
        if entry["token_name"] == token_name:
            return int(entry["token_id"])
    raise KeyError(f"token_name {token_name!r} not in section {section!r}")


def token_id_to_name(section: str, token_id: int, vocab: dict) -> str:
    """Return the ``token_name`` for ``token_id`` in ``section``.

    Raises ``KeyError`` if the section or id is not in the vocab.
    """
    buckets = vocab["locked_buckets"].get(section)
    if buckets is None:
        raise KeyError(f"section {section!r} not in macro vocab")
    for entry in buckets:
        if int(entry["token_id"]) == int(token_id):
            return str(entry["token_name"])
    raise KeyError(f"token_id {token_id!r} not in section {section!r}")
