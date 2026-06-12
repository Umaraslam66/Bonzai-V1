"""The shared locked-YAML freeze grammar (W2; rule of three met).

ONE source for the grammar three sealed artifacts already share — and the one
NEW instances (the W3 shard cache is the fourth) build on instead of hand-rolling
a copy:

  * ``sha256_excluding_field`` — sha over the canonical YAML EXCLUDING the sha
    field itself (so the stamp can live inside the file it stamps);
  * ``stamp_and_seal``        — write-once: stamp the sha, write canonical
    bytes, touch the lock marker beside the file;
  * ``verify_sealed_yaml``    — the verified read for NEW instances (marker ->
    parse -> required key -> sha present -> sha match -> schema version; every
    failure raises the caller's error type, fail-closed).

EXISTING instances (holdout manifest / conditioning-floor artifact /
city-identity registry) delegate their sha functions HERE but keep their own
verified-read taxonomies: their check orders and exception shapes differ
observably (floor/registry refuse on the marker first; the holdout guard checks
the sha shape first and raises a structured ``HoldoutLeakError``), and those
behaviors are pinned by tests/data/test_locked_yaml.py — the extraction
deliberately preserves them rather than unifying them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml


def sha256_excluding_field(data: dict, sha_field: str) -> str:
    """SHA over the canonical YAML of ``data`` EXCLUDING ``sha_field`` itself."""
    payload = {k: v for k, v in data.items() if k != sha_field}
    return compute_sha256(canonicalize_yaml(payload).encode("utf-8"))


def stamp_and_seal(payload: dict, path: Path, *, sha_field: str, lock_name: str | None) -> None:
    """Stamp ``sha_field``, write ONCE as canonical YAML, touch the lock marker.

    ``lock_name=None`` writes no marker (the holdout-manifest case: its
    ``_EVAL_SET_LOCKED`` marker is managed separately from the freeze). Callers
    wanting a custom overwrite-refusal message pre-check ``path.exists()``
    themselves; this refusal is the generic backstop.
    """
    if path.exists():
        raise FileExistsError(
            f"sealed artifact already exists at {path}; it is write-once — delete "
            f"deliberately to re-freeze, never overwrite."
        )
    frozen = dict(payload)
    frozen[sha_field] = sha256_excluding_field(frozen, sha_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(frozen), encoding="utf-8")
    if lock_name is not None:
        (path.parent / lock_name).touch()


def verify_sealed_yaml(
    path: Path,
    *,
    sha_field: str,
    lock_name: str,
    schema_field: str,
    schema_version: str,
    required_key: str,
    error: type[Exception],
) -> dict:
    """Verified read for NEW sealed-YAML instances (fail-closed, one taxonomy).

    Check order (the floor-artifact taxonomy, inherited by new instances):
    file exists -> marker beside it -> YAML parses to a mapping with
    ``required_key`` -> ``sha_field`` present -> recomputed sha matches ->
    ``schema_field`` equals ``schema_version``. Every refusal raises ``error``.
    """
    path = Path(path)
    if not path.exists():
        raise error(f"sealed artifact {path} does not exist; refusing (fail-closed).")
    marker = path.parent / lock_name
    if not marker.exists():
        raise error(
            f"no {lock_name} marker beside the sealed artifact (expected {marker}); "
            f"refusing to read an unsealed artifact."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise error(
            f"malformed sealed artifact at {path}: unparseable YAML ({exc}); "
            f"refusing (fail-closed)."
        ) from exc
    if not isinstance(data, dict) or required_key not in data:
        raise error(
            f"malformed sealed artifact at {path}: expected a YAML mapping with a "
            f"{required_key!r} key (got {type(data).__name__}); refusing (fail-closed)."
        )
    stored = data.get(sha_field)
    if stored is None:
        raise error(
            f"sealed artifact {path} carries NO {sha_field} field — an unstamped "
            f"artifact is unverifiable; refusing (fail-closed)."
        )
    recomputed = sha256_excluding_field(data, sha_field)
    if stored != recomputed:
        raise error(
            f"sealed artifact sha mismatch at {path}: stored {sha_field}={stored!r} "
            f"but recomputed {recomputed!r} — the content was edited after the "
            f"freeze; refusing."
        )
    version = data.get(schema_field)
    if version != schema_version:
        raise error(
            f"sealed artifact {path} declares {schema_field}={version!r} but this "
            f"reader requires {schema_version!r}; refusing a version-skewed artifact."
        )
    return data
