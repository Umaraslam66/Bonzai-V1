"""Layer 3 integration: real cached Singapore.

Marked @pytest.mark.slow; excluded from default fast suite. Run explicitly
with `uv run pytest -m slow tests/data/sub_e/test_singapore_integration.py`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.pipeline import PipelineConfig, derive_region
from cfm.data.sub_e.rotation import GRID_SIZE, EdgeKind, cell_to_edge_ids
from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHED_SUB_C = REPO_ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
CACHED_SUB_D = REPO_ROOT / "data" / "processed" / "sub_d" / "2026-04-15.0" / "singapore"

# Empirical gate thresholds per spec §11.3 #1.
GATE_MAX_CLASS_FRACTION = 0.90
GATE_MIN_ACTIVE_CLASS_FRACTION = 0.02


pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class Layer3Run:
    """Layer-3 fixture artifacts — exposed so the determinism-rerun test can
    reuse the same filtered sub-D and sub-C inputs without rebuilding them.
    """

    out_root: Path
    filtered_sub_d: Path
    filtered_sub_c: Path


def _layer3_subset_tiles() -> list[tuple[int, int]]:
    """Read the Layer-3 subset from sub-D's locked macro_plan_vocab.yaml."""
    vocab_path = REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"
    data = yaml.safe_load(vocab_path.read_text())
    return [(t["tile_i"], t["tile_j"]) for t in data["selected_layer3_tiles"]]


@pytest.fixture(scope="module")
def sub_e_run_layer3(tmp_path_factory) -> Layer3Run:
    """Run sub-E end-to-end on the Layer-3 9-tile subset (real data).

    Strategy: build a filtered sub-D region directory containing only the
    Layer-3 tiles' subdirectories + a filtered manifest, then run derive_region
    against that filtered region. Same approach sub-D used for its own
    Layer-3 Task 15 integration test (see `tests/data/sub_d/test_singapore_integration.py`).

    Returns a ``Layer3Run`` exposing the output dir AND the filtered sub-D
    and sub-C input dirs, so the determinism-rerun test can re-invoke
    derive_region against the same inputs without rebuilding the filter.
    """
    if not (CACHED_SUB_D / "_SUCCESS").exists():
        pytest.fail(
            "sub-D cached Singapore output missing at "
            f"{CACHED_SUB_D} — run `scripts/derive_macro_plan.py --region "
            "singapore --release 2026-04-15.0 ...` first. Fail-loud per "
            "spec §11.3 + halt-on-validator-fail discipline; do not skip-"
            "and-continue (which would silently downgrade the de-risk run)."
        )

    out_root = tmp_path_factory.mktemp("sub_e_layer3")
    filtered_sub_d = tmp_path_factory.mktemp("sub_d_filtered") / "singapore"
    filtered_sub_c = tmp_path_factory.mktemp("sub_c_filtered") / "singapore"
    filtered_sub_d.mkdir(parents=True)
    filtered_sub_c.mkdir(parents=True)

    subset = _layer3_subset_tiles()

    # Symlink each Layer-3 tile dir; build a filtered manifest.yaml referencing
    # only the subset; copy sub-D _SUCCESS + sub-C _SUCCESS.
    sub_d_manifest = yaml.safe_load((CACHED_SUB_D / "manifest.yaml").read_text())
    sub_c_manifest = yaml.safe_load((CACHED_SUB_C / "manifest.yaml").read_text())

    sub_d_manifest["tiles"] = [
        t for t in sub_d_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    sub_d_manifest["initial_extraction"]["tile_count"] = len(sub_d_manifest["tiles"])
    sub_c_manifest["tiles"] = [
        t for t in sub_c_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]

    (filtered_sub_d / "manifest.yaml").write_text(yaml.safe_dump(sub_d_manifest))
    (filtered_sub_c / "manifest.yaml").write_text(yaml.safe_dump(sub_c_manifest))
    (filtered_sub_d / "_SUCCESS").touch()
    (filtered_sub_c / "_SUCCESS").touch()

    for ti, tj in subset:
        tile_name = f"tile=EPSG3414_i{ti}_j{tj}"
        (filtered_sub_d / tile_name).symlink_to(CACHED_SUB_D / tile_name, target_is_directory=True)
        (filtered_sub_c / tile_name).symlink_to(CACHED_SUB_C / tile_name, target_is_directory=True)

    derive_region(
        PipelineConfig(
            release="2026-04-15.0",
            region="singapore",
            sub_c_region_dir=filtered_sub_c,
            sub_d_region_dir=filtered_sub_d,
            output_region_dir=out_root,
            commit_sha="0" * 40,
            lever_3_collapse=False,
        )
    )
    return Layer3Run(
        out_root=out_root,
        filtered_sub_d=filtered_sub_d,
        filtered_sub_c=filtered_sub_c,
    )


def test_layer3_pipeline_writes_success(sub_e_run_layer3: Layer3Run) -> None:
    assert (sub_e_run_layer3.out_root / "_SUCCESS").exists()


def test_layer3_cross_tile_validator_passes_on_real_data(
    sub_e_run_layer3: Layer3Run,
) -> None:
    validate_extraction_cross_tile(sub_e_run_layer3.out_root)


def test_layer3_deterministic_rerun_same_process(
    sub_e_run_layer3: Layer3Run, tmp_path: Path
) -> None:
    """Rerun derive_region against the same filtered inputs → byte-identical
    boundary_contract.parquet outputs per tile (spec §11.3 #3).

    Earlier draft pytest.skip'd this test claiming a fixture refactor was
    needed; the refactor was just exposing filtered_sub_d + filtered_sub_c
    on the fixture's return value, which the Layer3Run dataclass now does.
    Implementing the rerun in-place catches floating-point/dict-iteration
    nondeterminism that synthetic same-process determinism tests can't
    surface (real-data shape exercises real-data hash distributions).
    """
    # First-run parquet shas from the module-scoped fixture.
    first_shas = {
        d.name: hashlib.sha256((d / "boundary_contract.parquet").read_bytes()).hexdigest()
        for d in sub_e_run_layer3.out_root.glob("tile=EPSG3414_*")
    }
    assert first_shas, "fixture produced no per-tile boundary_contract.parquet"

    # Rerun derive_region into a fresh output dir using the SAME filtered
    # inputs (symlinks to CACHED_SUB_D + CACHED_SUB_C subsets) so we
    # exercise the actual orchestrator path on the actual sub-D byte
    # content — not a re-fixturing.
    out_root_rerun = tmp_path / "sub_e_layer3_rerun"
    derive_region(
        PipelineConfig(
            release="2026-04-15.0",
            region="singapore",
            sub_c_region_dir=sub_e_run_layer3.filtered_sub_c,
            sub_d_region_dir=sub_e_run_layer3.filtered_sub_d,
            output_region_dir=out_root_rerun,
            commit_sha="0" * 40,
            lever_3_collapse=False,
        )
    )

    second_shas = {
        d.name: hashlib.sha256((d / "boundary_contract.parquet").read_bytes()).hexdigest()
        for d in out_root_rerun.glob("tile=EPSG3414_*")
    }
    assert set(first_shas) == set(second_shas), (
        f"rerun produced different tile set; only-in-first="
        f"{sorted(set(first_shas) - set(second_shas))}, only-in-second="
        f"{sorted(set(second_shas) - set(first_shas))}"
    )
    mismatches = [name for name in first_shas if first_shas[name] != second_shas[name]]
    assert not mismatches, (
        "boundary_contract.parquet bytes differ between runs on the same "
        f"filtered inputs at tiles: {mismatches}. "
        "Spec §11.3 #3 (same-process determinism) violated — likely "
        "data-dependent nondeterminism (dict iteration, hash randomization, "
        "floating-point reduction order) that synthetic fixtures miss."
    )


def test_layer3_external_edge_single_cell_membership(
    sub_e_run_layer3: Layer3Run,
) -> None:
    """Each external (lower_cell_i, lower_cell_j, axis) identity appears in
    exactly one cell's per-cell view per rotation's enumeration (spec §10.2
    #5 + §11.3 #4). This is the rotation-aware membership check, not the
    weaker slot_index-uniqueness check.

    Defense in depth: derive_region's cross-tile validator already enforces
    set-equality at pipeline time (so a mismatch would fail the fixture
    before this test runs), but asserting directly here catches drift
    modes the validator might not — and makes the test's name match its
    body (the earlier draft asserted only uniqueness despite the name).
    """
    # Rotation's external set is invariant across tiles in the 8x8 grid.
    expected_external = set()
    for ci in range(GRID_SIZE):
        for cj in range(GRID_SIZE):
            edges = cell_to_edge_ids(ci, cj)
            for edge in (edges.north, edges.south, edges.west, edges.east):
                li, lj, axis, kind = edge
                if kind is EdgeKind.EXTERNAL:
                    expected_external.add((li, lj, axis))
    assert len(expected_external) == 32, (
        f"rotation should emit 32 unique external tuples, got "
        f"{len(expected_external)} — rotation function bug"
    )

    for tile_dir in sub_e_run_layer3.out_root.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        slot_kinds = tbl.column("slot_kind").to_pylist()
        lower_is = tbl.column("lower_cell_i").to_pylist()
        lower_js = tbl.column("lower_cell_j").to_pylist()
        axes = tbl.column("axis").to_pylist()
        parquet_external = {
            (li, lj, ax)
            for sk, li, lj, ax in zip(slot_kinds, lower_is, lower_js, axes, strict=True)
            if sk == 2
        }
        assert parquet_external == expected_external, (
            f"tile {tile_dir.name}: external-edge set mismatch with rotation. "
            f"only-in-parquet={sorted(parquet_external - expected_external)}, "
            f"only-in-rotation={sorted(expected_external - parquet_external)}"
        )


def test_layer3_empirical_gate_real_distribution(
    sub_e_run_layer3: Layer3Run,
) -> None:
    """REAL empirical gate run on Layer-3 subset.

    Per spec §11.3 #1: ship iff no active class above 90% AND no active class
    below 2%. Halt on violation (memory `feedback_test_weakening_to_pass`).
    """
    counter: Counter[int] = Counter()
    for tile_dir in sub_e_run_layer3.out_root.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        scope_markers = tbl.column("scope_marker").to_pylist()
        boundary_classes = tbl.column("boundary_class_enum").to_pylist()
        for scope, cls in zip(scope_markers, boundary_classes, strict=True):
            if scope == 0 and cls is not None:  # active rows only
                counter[cls] += 1

    total_active = sum(counter.values())
    assert total_active > 0, "no active edges in Layer-3 subset — sub-D upstream failed"

    fractions = {cls: count / total_active for cls, count in counter.items()}

    # Print for reviewer visibility.
    print("\nLayer-3 boundary_class distribution:")
    for cls_id, frac in sorted(fractions.items()):
        cls_name = BoundaryClass(cls_id).name
        print(f"  {cls_name}: {frac:.4f} ({counter[cls_id]} of {total_active})")

    # Thread the violating class name and fraction directly into the assert
    # message — single-line CI failure should pinpoint which class violated
    # and by how much, not just "a class". The print above logs the full
    # distribution; the assert pinpoints the violator.
    max_cls = max(fractions, key=fractions.get)
    min_cls = min(fractions, key=fractions.get)
    max_frac = fractions[max_cls]
    min_frac = fractions[min_cls]

    assert max_frac <= GATE_MAX_CLASS_FRACTION, (
        f"empirical gate FAILED (max-concentration): "
        f"{BoundaryClass(max_cls).name} at {max_frac:.4f} > "
        f"{GATE_MAX_CLASS_FRACTION}. Halt and escalate per §5 reopen rule. "
        f"Full distribution: "
        f"{ {BoundaryClass(c).name: round(f, 4) for c, f in fractions.items()} }. "
        f"Do NOT weaken this threshold."
    )
    assert min_frac >= GATE_MIN_ACTIVE_CLASS_FRACTION, (
        f"empirical gate FAILED (min-presence): "
        f"{BoundaryClass(min_cls).name} at {min_frac:.4f} < "
        f"{GATE_MIN_ACTIVE_CLASS_FRACTION}. Halt and escalate per §5 reopen rule. "
        f"Full distribution: "
        f"{ {BoundaryClass(c).name: round(f, 4) for c, f in fractions.items()} }. "
        f"Do NOT weaken this threshold."
    )

    # Also pin the distribution as a golden file post-pass.
    golden = REPO_ROOT / "tests" / "golden" / "sub_e" / "empirical_gate"
    golden.mkdir(parents=True, exist_ok=True)
    (golden / "layer3_boundary_class_distribution.yaml").write_text(
        yaml.safe_dump(
            {
                "boundary_derivation_version": "1.0",
                "boundary_vocab_version": "1.0",
                "total_active_edges": total_active,
                "fractions": {
                    BoundaryClass(cls).name: round(frac, 6)
                    for cls, frac in sorted(fractions.items())
                },
            }
        )
    )


def test_layer3_writer_round_trips_major_and_minor(
    sub_e_run_layer3: Layer3Run,
) -> None:
    """Task-6 carry-forward writer-regression guard.

    A writer bug that always emits BoundaryClass.NONE (or coerces all
    active classes to a single value) would manifest in the empirical-gate
    test as a max-class-fraction violation, which routes into the §5 reopen
    pathway (a derivation-grouping concern). That's the wrong escalation
    path: the failure is structural (writer corruption), not semantic
    (derivation decision).

    This test asserts both BoundaryClass.MAJOR_ROAD and BoundaryClass.MINOR_ROAD
    appear at least once in the Layer-3 active-row distribution. If this
    fails, halt and diagnose the writer, not the class-grouping map.
    """
    counter: Counter[int] = Counter()
    for tile_dir in sub_e_run_layer3.out_root.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        scope_markers = tbl.column("scope_marker").to_pylist()
        boundary_classes = tbl.column("boundary_class_enum").to_pylist()
        for scope, cls in zip(scope_markers, boundary_classes, strict=True):
            if scope == 0 and cls is not None:
                counter[cls] += 1

    assert counter[int(BoundaryClass.MAJOR_ROAD)] > 0, (
        "Layer-3 distribution has zero MAJOR_ROAD active edges — likely "
        "writer regression (a derivation-grouping decision would still emit "
        "at least one MAJOR_ROAD on Singapore's 9-tile Layer-3 subset). "
        "Halt and diagnose the writer, NOT the class-grouping map."
    )
    assert counter[int(BoundaryClass.MINOR_ROAD)] > 0, (
        "Layer-3 distribution has zero MINOR_ROAD active edges — likely "
        "writer regression. Halt and diagnose the writer, NOT the "
        "class-grouping map."
    )


def test_layer3_lever_3_collapse_real_data(tmp_path_factory) -> None:
    """Lever-3 regression guard on real Singapore data.

    Runs the pipeline under `lever_3_collapse=True` over the Layer-3 9-tile
    subset. Asserts:

    - Pipeline writes `_SUCCESS` (validators pass under lever-3).
    - All `boundary_class_enum` values on-disk are null.
    - Cross-tile validator passes.

    Verifies that the day-9 lever-3 trigger path is mechanically pullable
    against real data, not just synthetic fixtures.
    """
    if not (CACHED_SUB_D / "_SUCCESS").exists():
        pytest.fail(
            "sub-D cached Singapore output missing at "
            f"{CACHED_SUB_D} — run `scripts/derive_macro_plan.py --region "
            "singapore --release 2026-04-15.0 ...` first. Fail-loud per "
            "spec §11.3 + halt-on-validator-fail discipline."
        )

    out_root = tmp_path_factory.mktemp("sub_e_layer3_lever_3")
    filtered_sub_d = tmp_path_factory.mktemp("sub_d_filtered_lever_3") / "singapore"
    filtered_sub_c = tmp_path_factory.mktemp("sub_c_filtered_lever_3") / "singapore"
    filtered_sub_d.mkdir(parents=True)
    filtered_sub_c.mkdir(parents=True)

    subset = _layer3_subset_tiles()

    sub_d_manifest = yaml.safe_load((CACHED_SUB_D / "manifest.yaml").read_text())
    sub_c_manifest = yaml.safe_load((CACHED_SUB_C / "manifest.yaml").read_text())
    sub_d_manifest["tiles"] = [
        t for t in sub_d_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    sub_d_manifest["initial_extraction"]["tile_count"] = len(sub_d_manifest["tiles"])
    sub_c_manifest["tiles"] = [
        t for t in sub_c_manifest["tiles"] if (t["tile_i"], t["tile_j"]) in subset
    ]
    (filtered_sub_d / "manifest.yaml").write_text(yaml.safe_dump(sub_d_manifest))
    (filtered_sub_c / "manifest.yaml").write_text(yaml.safe_dump(sub_c_manifest))
    (filtered_sub_d / "_SUCCESS").touch()
    (filtered_sub_c / "_SUCCESS").touch()
    for ti, tj in subset:
        tile_name = f"tile=EPSG3414_i{ti}_j{tj}"
        (filtered_sub_d / tile_name).symlink_to(CACHED_SUB_D / tile_name, target_is_directory=True)
        (filtered_sub_c / tile_name).symlink_to(CACHED_SUB_C / tile_name, target_is_directory=True)

    derive_region(
        PipelineConfig(
            release="2026-04-15.0",
            region="singapore",
            sub_c_region_dir=filtered_sub_c,
            sub_d_region_dir=filtered_sub_d,
            output_region_dir=out_root,
            commit_sha="0" * 40,
            lever_3_collapse=True,
        )
    )

    assert (out_root / "_SUCCESS").exists()
    validate_extraction_cross_tile(out_root)

    for tile_dir in out_root.glob("tile=EPSG3414_*"):
        tbl = pq.ParquetFile(tile_dir / "boundary_contract.parquet").read()
        values = tbl.column("boundary_class_enum").to_pylist()
        assert all(v is None for v in values), (
            f"lever-3 mode must produce uniform null in {tile_dir.name}"
        )
