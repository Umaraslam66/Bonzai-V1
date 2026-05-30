"""Sub-F region derivation pipeline orchestrator (Task 11).

The integration spine that composes the sub-F per-tile encode + the two
validators + provenance + region manifest + the `_SUCCESS` marker:

  require sub-D _SUCCESS  ->  require sub-E _SUCCESS  ->  per tile:
      encode_tile (T8.8)  ->  validate_inline (T9)  ->  write provenance.yaml
  ->  validate_cross_tile (T10)  ->  write manifest.yaml  ->  touch _SUCCESS

Halt-on-validator-fail discipline (mirrors src/cfm/data/sub_e/pipeline.py):
the cross-tile validator runs BEFORE `_SUCCESS` is written, and `_SUCCESS`
is touched LAST. Any validator exception propagates and the touch never
runs, so consumers never see a green-light marker for a run that did not
pass. There is no write-then-unlink recovery path and therefore no race
window in which a false-green `_SUCCESS` could be observed.

Per-cell sha: NOT recomputed here. `encode_tile` already writes the
per-cell `provenance_sha256` (the content-equivalence anchor,
sha256(big-endian-uint16 tokens + bytes([cell_i, cell_j])) per spec
§13.1, commit e556c31). T11 owns ONLY the TILE-level provenance.yaml
`provenance_sha256` (via provenance.py::provenance_sha256) and the region
manifest. The tile provenance dict carries tile identity so distinct
tiles produce distinct tile shas (T10 `_check_sha_uniqueness` verifies).

Restartability: every artifact this pipeline writes is byte-deterministic
given the same inputs and the same `extracted_utc`. `encode_tile` writes
via the byte-deterministic parquet writer; the tile provenance sha and the
manifest sha are content-derived with timestamps excluded
(SUB_F_EXCLUDED_FROM_SHA). The ONLY non-deterministic input is the wall
clock, threaded in via `PipelineConfig.extracted_utc` (defaults to the
live clock at run time, but a caller can pin it for a reproducible run).
Because no step reads or resumes from prior on-disk state, a partial run
that fails mid-pipeline cannot poison a later clean re-run: the clean
re-run overwrites every tile from scratch and produces byte-identical
output. No `_SUCCESS` is written on a partial run, so consumers treat the
output dir as incomplete and a re-run is mandatory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_f.manifest import build_region_manifest, task6_vocab_sources
from cfm.data.sub_f.pipeline_writer import encode_tile
from cfm.data.sub_f.provenance import provenance_sha256
from cfm.data.sub_f.validator_cross_tile import validate_cross_tile
from cfm.data.sub_f.validator_inline import validate_inline
from cfm.data.sub_f.versions import (
    SUB_F_ARTIFACT_FORMAT_VERSION,
    SUB_F_DERIVATION_VERSION,
    SUB_F_SCHEMA_VERSION,
    SUB_F_VALIDATOR_VERSION,
    SUB_F_VOCAB_VERSION,
    load_sub_f_source_version,
)

log = logging.getLogger(__name__)

# Warning-band diagnostic budgets — Halt-4 RE-LOCK (commit c1eb2a1); NOT the
# original 5792/5888. The locked action contract
# (configs/sub_f/sequence_length_analysis.yaml long_cell_diagnostic) pins the
# warning band to cells whose chunked length is in (5760, 6016]: 5760 is "2
# padding blocks below the padded budget" (6016 - 256) and 6016 is the padded
# budget. The orchestrator emits the warning-band cell count + per-type
# composition into the region run report after a successful derive.
_ALPHA_DROP_BUDGET_RAW = 5760
_ALPHA_DROP_BUDGET_PADDED = 6016


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single sub-F region derivation.

    ``extracted_utc`` is the wall-clock stamp written into every tile's
    provenance.yaml. It is excluded from the provenance sha
    (SUB_F_EXCLUDED_FROM_SHA), so it does not affect content-equivalence
    anchoring; it does, however, appear in the provenance.yaml bytes. Pin it
    (pass a fixed string) for a byte-reproducible run; leave it ``None`` to
    stamp the live clock at run time.
    """

    release: str
    region: str
    sub_c_region_dir: Path
    sub_d_region_dir: Path
    sub_e_region_dir: Path
    output_region_dir: Path
    extracted_utc: str | None = None
    run_alpha_drop_report: bool = True


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_sub_d_success_marker(sub_d_region_dir: Path) -> None:
    """Gate sub-F on sub-D's `_SUCCESS` marker.

    Raises FileNotFoundError if the marker is absent. Sub-F does not start
    derivation against a sub-D region whose validator has not closed.
    """
    marker = sub_d_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(
            f"sub-D _SUCCESS marker missing at {marker}; sub-F refuses to start"
        )


def require_sub_e_success_marker(sub_e_region_dir: Path) -> None:
    """Gate sub-F on sub-E's `_SUCCESS` marker.

    Raises FileNotFoundError if the marker is absent. Sub-F reads sub-E's
    boundary contracts; it does not start against a sub-E region whose
    validator has not closed.
    """
    marker = sub_e_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(
            f"sub-E _SUCCESS marker missing at {marker}; sub-F refuses to start"
        )


def _encode_tile_or_raise(
    sub_c_features_parquet: Path,
    sub_e_boundary_contract_parquet: Path,
    out_cells_parquet: Path,
) -> Path:
    """Indirect call to encode_tile so tests can monkey-patch to poison a tile.

    Keeping the encode behind a thin seam lets a halt-on-failure test inject a
    cells.parquet that violates an inline invariant (or fail mid-region) without
    rewriting the orchestrator.
    """
    return encode_tile(sub_c_features_parquet, sub_e_boundary_contract_parquet, out_cells_parquet)


def _validate_inline_or_raise(cells_path: Path) -> None:
    """Indirect call to validate_inline so tests can monkey-patch to simulate
    an inline-validator failure (halt-on-inline-fail test)."""
    validate_inline(cells_path)


def _build_tile_provenance(
    *,
    tile_name: str,
    tile_i: int,
    tile_j: int,
    release: str,
    extracted_utc: str,
) -> dict:
    """Build one tile's provenance dict (the canonical record-as-written).

    The six version axes are placed at the TOP LEVEL because T10's
    `_check_version_consistency` reads them via ``prov.get(axis)`` and
    `_check_sha_uniqueness` reads ``prov.get("provenance_sha256")`` at the top
    level. Tile identity (``tile_name``, ``tile_i``, ``tile_j``) is folded in
    so each tile's content differs and `provenance_sha256` is DISTINCT per
    tile. ``extraction.extracted_utc`` is excluded from the sha by
    SUB_F_EXCLUDED_FROM_SHA, so the sha is timestamp-stable across reruns.
    """
    data: dict = {
        "tile_name": tile_name,
        "tile_i": tile_i,
        "tile_j": tile_j,
        "release": release,
        "sub_f_artifact_format_version": SUB_F_ARTIFACT_FORMAT_VERSION,
        "sub_f_schema_version": SUB_F_SCHEMA_VERSION,
        "sub_f_vocab_version": SUB_F_VOCAB_VERSION,
        "sub_f_derivation_version": SUB_F_DERIVATION_VERSION,
        "sub_f_validator_version": SUB_F_VALIDATOR_VERSION,
        "sub_f_source_version": load_sub_f_source_version(),
        "extraction": {
            "extracted_utc": extracted_utc,
        },
    }
    data["provenance_sha256"] = provenance_sha256(data)
    return data


def derive_region(cfg: PipelineConfig) -> None:
    """Run the full sub-F derivation for a region.

    Halts on any validator failure; no `_SUCCESS` is written on error. See
    the module docstring for the ordering and restartability guarantees.
    """
    require_sub_d_success_marker(cfg.sub_d_region_dir)
    require_sub_e_success_marker(cfg.sub_e_region_dir)
    cfg.output_region_dir.mkdir(parents=True, exist_ok=True)

    extracted_utc = cfg.extracted_utc or _utc_now()

    # The sub-E region is the authoritative tile set: sub-F BP7 emission is
    # 100% sub-E-derived (close-checklist), and T10 cross-references each
    # sub-F tile against the identically-named sub-E tile dir.
    tile_dirs = sorted(cfg.sub_e_region_dir.glob("tile=*"))
    if not tile_dirs:
        raise FileNotFoundError(
            f"no tile=* dirs under sub-E region {cfg.sub_e_region_dir}; nothing to derive"
        )

    tile_entries: list[dict] = []
    for sub_e_tile_dir in tile_dirs:
        tile_name = sub_e_tile_dir.name
        tile_i, tile_j = _parse_tile_coords(tile_name)

        sub_c_features = cfg.sub_c_region_dir / tile_name / "features.parquet"
        sub_e_contract = sub_e_tile_dir / "boundary_contract.parquet"
        out_tile_dir = cfg.output_region_dir / tile_name
        out_tile_dir.mkdir(parents=True, exist_ok=True)
        out_cells = out_tile_dir / "cells.parquet"

        cells_path = _encode_tile_or_raise(sub_c_features, sub_e_contract, out_cells)
        _validate_inline_or_raise(cells_path)

        provenance = _build_tile_provenance(
            tile_name=tile_name,
            tile_i=tile_i,
            tile_j=tile_j,
            release=cfg.release,
            extracted_utc=extracted_utc,
        )
        (out_tile_dir / "provenance.yaml").write_text(canonicalize_yaml(provenance))

        tile_entries.append(
            {
                "tile_i": tile_i,
                "tile_j": tile_j,
                "tile_dir": tile_name,
                "provenance_sha256": provenance["provenance_sha256"],
            }
        )

    # Cross-tile validator runs BEFORE _SUCCESS (sub-E precedent + spec §11.8).
    # If it raises, the manifest + _SUCCESS are never written for this run.
    validate_cross_tile(cfg.output_region_dir, cfg.sub_e_region_dir)

    manifest = build_region_manifest(
        region=cfg.region,
        release=cfg.release,
        tile_entries=tile_entries,
        vocab_sources=task6_vocab_sources(),
    )
    (cfg.output_region_dir / "manifest.yaml").write_text(canonicalize_yaml(manifest))

    # _SUCCESS LAST — only after ALL validators pass. No partial _SUCCESS.
    (cfg.output_region_dir / "_SUCCESS").touch()

    if cfg.run_alpha_drop_report:
        _emit_alpha_drop_report(cfg)


def _parse_tile_coords(tile_name: str) -> tuple[int, int]:
    """Parse (tile_i, tile_j) from a `tile=EPSG3414_i<I>_j<J>` dir name.

    Mirrors scripts/sub_f/compute_alpha_drop_report.py's parser so the tile
    identity in provenance/manifest matches the convention used elsewhere.
    """
    parts = tile_name.replace("tile=", "").split("_")
    tile_i = int(parts[1].lstrip("i"))
    tile_j = int(parts[2].lstrip("j"))
    return tile_i, tile_j


def _emit_alpha_drop_report(cfg: PipelineConfig) -> None:
    """Run the warning-band alpha-drop diagnostic for this region (final step).

    Per the close-checklist obligation + the Halt-4 RE-LOCK action contract:
    after a successful derive, emit the warning-band cell count + per-type
    composition into the region run report at budgets
    (5760, 6016] (RE-LOCK commit c1eb2a1; NOT the original 5792/5888).

    Delegated to the script's thin importable entrypoint so the budget
    accounting (chunked, encoder-faithful per token_cost) stays in one place.
    Failures here are logged, NOT raised: the diagnostic is advisory and runs
    AFTER _SUCCESS, so a report glitch must not retroactively invalidate a
    region whose validators already passed.
    """
    from scripts.sub_f.compute_alpha_drop_report import run_alpha_drop_report

    try:
        result = run_alpha_drop_report(
            sub_c_region_dir=cfg.sub_c_region_dir,
            budget_raw=_ALPHA_DROP_BUDGET_RAW,
            budget_padded=_ALPHA_DROP_BUDGET_PADDED,
            label=f"warning_band_{cfg.region}",
        )
    except Exception:  # pragma: no cover - advisory diagnostic; never fatal
        log.exception(
            "alpha-drop warning-band report failed for region %s "
            "(advisory only; region derive already succeeded)",
            cfg.region,
        )
        return

    log.info(
        "alpha-drop warning band (%d, %d] for region %s: %d/%d cells "
        "(%.4f%%); per-type drop composition: %s",
        _ALPHA_DROP_BUDGET_RAW,
        _ALPHA_DROP_BUDGET_PADDED,
        cfg.region,
        result["n_cells_dropped"],
        result["n_cells_total"],
        result["drop_fraction_pct"],
        result["drop_set_by_type"],
    )
