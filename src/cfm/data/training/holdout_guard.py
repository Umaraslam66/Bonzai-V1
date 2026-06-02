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

from cfm.eval.holdout.lineage_audit import Artifact, audit_no_holdout_leak


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


def run_holdout_audit(holdout_manifest: dict, reachable: list[Artifact]) -> None:
    """Raise HoldoutLeakError on any leak/absent-lineage; return None iff clean."""
    audit_no_holdout_leak(holdout_manifest, reachable)
