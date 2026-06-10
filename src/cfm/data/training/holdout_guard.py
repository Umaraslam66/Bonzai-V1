"""Trigger-1 runtime holdout-audit wiring (pure core; no Lightning import).

The Lightning ``CellDataModule`` (Task 6) calls these at ``setup()`` BEFORE any
batch, fail-closed. Kept Lightning-free so the four regime-distinguishing tests
run anywhere (the DDP all-ranks halt is validated separately on 4xA100).

The load reads each shard's STAMPED lineage from the training manifest and NEVER
synthesizes one from a path: a tile lacking a ``lineage`` field reaches the audit
as a genuine ``None`` so G-F4 (fail-closed on absent lineage) can fire. Both this
exclusion layer and the build-time set-difference key on the SAME frozen holdout
manifest (one definition of "holdout").
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.eval.holdout.lineage_audit import (
    Artifact,
    HoldoutLeakError,
    LineageFailure,
    audit_no_holdout_leak,
)
from cfm.eval.holdout.manifest import manifest_sha256


def load_training_manifest(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def manifest_to_reachable(manifest: dict) -> list[Artifact]:
    """Build training-reachable Artifacts from the manifest's STAMPED lineage.

    A tile with no ``lineage`` field -> ``Artifact.lineage is None`` (G-F4),
    NEVER a path-synthesized value (that would make G-F4 vacuous)."""
    region = manifest.get("region", "?")
    out: list[Artifact] = []
    for t in manifest.get("tiles", []):
        path = f"{region}/{int(t['tile_i'])}_{int(t['tile_j'])}"
        if t.get("lineage") is not None:
            lineage = frozenset((str(r[0]), int(r[1]), int(r[2])) for r in t["lineage"])
        else:
            lineage = None  # absent -> genuine None (no synthesis)
        out.append(Artifact(path=path, lineage=lineage))
    return out


def run_holdout_audit(
    holdout_manifest: dict,
    reachable: list[Artifact],
    *,
    expected_schema_version: str = "2.0",
    manifest_path: Path | None = None,
) -> None:
    """Raise HoldoutLeakError on any leak/absent-lineage; return None iff clean.

    Fail-closed schema backstop: refuses a manifest whose ``manifest_schema_version``
    != ``expected_schema_version`` (default "2.0", the EU multi-region schema). This is
    the #16 backstop: a forgotten bake-off re-point that hands the stale SG 1.0 manifest
    is refused, never silently audited as the EU corpus.

    F9 reader-side integrity (Task 20): when ``manifest_path`` is given (the PRODUCTION
    reader, ``CellDataModule.setup``), two further checks run BEFORE the leak audit:

    - sha verification: the dict's stored ``manifest_sha256`` must equal the recomputed
      ``manifest_sha256(holdout_manifest)`` (the freeze grammar: sha over the canonical
      manifest EXCLUDING the field itself). A MISSING field is refused too — an
      unstamped manifest is unverifiable, so fail-closed.
    - lock marker: ``_EVAL_SET_LOCKED`` must exist beside the manifest
      (``manifest_path.parent``) — the write-once freeze seal travels with the file.

    ``manifest_path=None`` (legacy/synthetic callers) SKIPS only these two checks; the
    schema backstop + leak audit always run. Order: schema -> sha/marker -> leak."""
    got = holdout_manifest.get("manifest_schema_version")
    if got != expected_schema_version:
        raise HoldoutLeakError(
            [
                LineageFailure(
                    "<manifest>",
                    f"expected holdout manifest schema {expected_schema_version!r}, got {got!r} — "
                    "refusing to audit (fail-closed schema backstop). A non-multi-region (e.g. "
                    "stale SG 1.0) manifest must not be audited as the EU corpus; re-point to the "
                    "multi-region manifest or pass expected_schema_version explicitly.",
                )
            ],
            headline="held-out audit refused (schema mismatch):",
        )
    if manifest_path is not None:
        _verify_manifest_integrity(holdout_manifest, Path(manifest_path))
    audit_no_holdout_leak(holdout_manifest, reachable)


def _verify_manifest_integrity(holdout_manifest: dict, manifest_path: Path) -> None:
    """F9: recomputed-sha == stored ``manifest_sha256`` AND ``_EVAL_SET_LOCKED``
    beside the manifest; any failure -> HoldoutLeakError (fail-closed)."""
    stored = holdout_manifest.get("manifest_sha256")
    if stored is None:
        raise HoldoutLeakError(
            [
                LineageFailure(
                    str(manifest_path),
                    "holdout manifest carries NO manifest_sha256 field — an unstamped "
                    "manifest is unverifiable (fail-closed). Freeze it via "
                    "freeze_holdout_manifest before pointing training at it.",
                )
            ],
            headline="held-out audit refused (unverifiable manifest):",
        )
    recomputed = manifest_sha256(holdout_manifest)
    if recomputed != stored:
        raise HoldoutLeakError(
            [
                LineageFailure(
                    str(manifest_path),
                    f"holdout manifest sha mismatch: stored manifest_sha256={stored!r} but "
                    f"recomputed={recomputed!r} — the manifest was edited after freeze (or "
                    "corrupted). Refusing to audit against a tampered holdout.",
                )
            ],
            headline="held-out audit refused (sha mismatch):",
        )
    marker = manifest_path.parent / "_EVAL_SET_LOCKED"
    if not marker.exists():
        raise HoldoutLeakError(
            [
                LineageFailure(
                    str(manifest_path),
                    f"no _EVAL_SET_LOCKED marker beside the holdout manifest (expected "
                    f"{marker}) — the eval set is not sealed write-once; refusing to "
                    "audit against an unlocked holdout.",
                )
            ],
            headline="held-out audit refused (missing lock marker):",
        )
