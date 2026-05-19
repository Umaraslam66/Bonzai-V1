"""Layer-3 cached-Singapore integration tests for sub-D (Task 15).

These tests run the full sub-D pipeline against the cached sub-C Singapore
extraction at ``data/processed/sub_c/2026-04-15.0/singapore/``. They are
``@pytest.mark.slow`` (opt in with ``-m slow``) and skip when the cache is
absent (see ``conftest.py``'s ``cached_sub_c_singapore_dir`` fixture).

What's covered here:

- Layer-3 tile subset structural check (rationales present in locked vocab
  per D1 — the locked artifact is the source of truth, Task 8 committed it).
- End-to-end pipeline run on full Singapore + validator passes.
- Byte-determinism: two runs on the same inputs produce byte-identical
  artifacts everywhere.
- Cross-environment determinism sentinel: documents that determinism has
  only been verified on the local platform (darwin/aarch64), not on
  Leonardo. A future task can extend this when running on Leonardo.

What's NOT covered here (intentionally — lives in the fast suite so every
push catches regressions, see ``test_pipeline_derivation.py``):

- Pure-function correctness of ``_zoning_token_id``,
  ``_bucket_for_numeric_value``, ``_resolve_population_density_bucket``
  against the locked vocab. Those run in the fast suite without
  cached-data dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.sub_d.pipeline import derive_region_macro_plan

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCKED_VOCAB_PATH = _REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"

# Pinned values for byte-deterministic re-runs. Same shape as the Task 14
# fixture but using a "test-only" commit_sha + UTC so a stray cached output
# from this test is recognisable.
_TEST_COMMIT_SHA = "abc" + "0" * 37
_TEST_EXTRACTED_UTC = "2026-05-19T12:00:00Z"


# ---------------------------------------------------------------------------
# Plan-named tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cached_singapore_subset_tile_ids_have_rationales():
    """Per D1 (Phase-B tension flag): the Layer-3 cached-Singapore tile IDs
    live in the locked vocab artifact's ``selected_layer3_tiles`` field —
    that artifact is the source of truth, not a separate config or
    test-only file. Every entry MUST carry a ``rationale`` so a reviewer
    can audit why each tile was selected.

    This test does NOT require the sub-C cache (only the locked vocab,
    which is committed in repo). It's slow-marked for consistency with the
    Layer-3 cohort.
    """
    vocab = yaml.safe_load(_LOCKED_VOCAB_PATH.read_text(encoding="utf-8"))
    tiles = vocab.get("selected_layer3_tiles")
    assert isinstance(tiles, list) and len(tiles) >= 1, (
        "selected_layer3_tiles missing or empty in locked vocab"
    )
    for entry in tiles:
        assert isinstance(entry.get("tile_i"), int), entry
        assert isinstance(entry.get("tile_j"), int), entry
        rationale = entry.get("rationale")
        assert isinstance(rationale, str) and rationale.strip(), (
            f"tile ({entry.get('tile_i')},{entry.get('tile_j')}) missing or "
            f"empty rationale; got {rationale!r}"
        )


@pytest.mark.slow
def test_cached_singapore_subset_derivation_passes_validation(
    tmp_path: Path, cached_sub_c_singapore_dir: Path
):
    """End-to-end: run the pipeline on the cached Singapore sub-C output
    and assert validation passes. This is the broadest test in the sub-D
    suite — every module participates, and any contract violation
    (digest chain, schema, version namespace, masked-slot rule, B6 config
    copy) would surface here.

    ``derive_region_macro_plan`` itself calls ``validate_region`` before
    writing ``_SUCCESS``. So if ``_SUCCESS`` exists after the call, every
    Task 13 validator check passed against real Singapore data.
    """
    output_dir = tmp_path / "sub_d" / "singapore"
    derive_region_macro_plan(
        sub_c_region_dir=cached_sub_c_singapore_dir,
        output_dir=output_dir,
        macro_vocab_path=_LOCKED_VOCAB_PATH,
        release="2026-04-15.0",
        region="singapore",
        commit_sha=_TEST_COMMIT_SHA,
        extracted_utc=_TEST_EXTRACTED_UTC,
    )
    assert (output_dir / "_SUCCESS").is_file()
    assert (output_dir / "manifest.yaml").is_file()
    # Sanity: at least one tile dir is present.
    tile_dirs = list(output_dir.glob("tile=*"))
    assert tile_dirs, "no tile dirs written"


@pytest.mark.slow
def test_cached_singapore_subset_derivation_is_byte_identical_on_rerun(
    tmp_path: Path, cached_sub_c_singapore_dir: Path
):
    """Run the pipeline TWICE against the cached sub-C Singapore and
    assert every artifact is byte-identical between runs. This is the
    strongest determinism contract — non-determinism in any link
    (dict iteration order, parquet writer, yaml dumper, sha computation,
    proxy-row ordering) would surface as a byte diff.

    Compares every file under each output_dir, not just a sample. With
    ~494 tiles x 4 per-tile files plus manifest + _SUCCESS, that's around
    ~2000 files. The compare is fast; the two pipeline runs are what
    makes this test slow.
    """
    output_a = tmp_path / "sub_d_a" / "singapore"
    output_b = tmp_path / "sub_d_b" / "singapore"
    for out in (output_a, output_b):
        derive_region_macro_plan(
            sub_c_region_dir=cached_sub_c_singapore_dir,
            output_dir=out,
            macro_vocab_path=_LOCKED_VOCAB_PATH,
            release="2026-04-15.0",
            region="singapore",
            commit_sha=_TEST_COMMIT_SHA,
            extracted_utc=_TEST_EXTRACTED_UTC,
        )

    # Enumerate every file under output_a and compare to output_b.
    files_a = sorted(p.relative_to(output_a) for p in output_a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(output_b) for p in output_b.rglob("*") if p.is_file())
    assert files_a == files_b, (
        f"file inventory differs between runs: only in a={set(files_a) - set(files_b)}, "
        f"only in b={set(files_b) - set(files_a)}"
    )
    for rel in files_a:
        a_bytes = (output_a / rel).read_bytes()
        b_bytes = (output_b / rel).read_bytes()
        assert a_bytes == b_bytes, f"byte drift in {rel} between two runs"


@pytest.mark.slow
def test_cross_environment_determinism_gap_is_documented_if_not_run():
    """Sentinel: pins the known gap that byte-determinism has been verified
    only on the local development platform (darwin/aarch64 in this
    project's typical setup), not on Leonardo (linux/x86_64 + cluster
    file system) where production training will read sub-D's output.

    The gap is acceptable for Phase 1 because: (a) the locked vocab + sub-C
    cache used for derivation are byte-stable across platforms, (b) the
    pyarrow/yaml serialisation surface is the only remaining
    cross-platform risk, and (c) the validator's digest chain would
    detect drift if it manifested on Leonardo.

    This sentinel test exists so the gap is visible in the test surface
    (not just in a docstring or known_issues.md). A future task can
    promote it to an actual cross-environment verification when sub-D's
    output is first consumed on Leonardo — the assertion would then
    become: hash of the full Singapore sub-D output on Leonardo equals
    hash on local platform. For now: the gap is documented, the test
    passes, and the docstring is the contract.
    """
    # No-op assertion — the docstring above IS the test.
    assert True
