"""eu-train-union datamodule path (readiness-closure Task 11, F7).

The Task-8 per-city training manifests get their production consumer:
``_datamodule`` with ``cfg.train_set == "eu-train-union"`` resolves the train
cities from the G4 roll-up + multiregion holdout manifest and constructs the
CellDataModule in UNION mode (``training_manifests=[per-city paths]``).

Key contract under test:
  - held-out cities are ABSENT from the manifest list (whole-city exclusion);
  - the holdout ``held_out_cities`` read is STRICT (fail-closed caller pattern,
    never ``.get`` — a holdout mapping missing the key must raise, because
    ``train_cities`` itself uses ``.get(..., [])`` and would silently exclude
    nothing);
  - the union branch pins ``expected_holdout_schema="2.0"`` and the multiregion
    manifest DIRECTLY (not routed through ``holdout_manifest_for_region``);
  - the union branch NEVER rebuilds shards (missing manifests surface loudly in
    ``CellDataModule.setup()`` / the sbatch preamble existence check).

CellDataModule construction only STORES (IO happens in setup()), so a synthetic
2-city fixture suffices — no real corpus on disk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from cfm.data.training.paths import training_manifest_path
from cfm.training.config import ScaffoldConfig

_REPO = Path(__file__).resolve().parents[2]
_RELEASE = "2026-04-15.0"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_scaffold_union", _REPO / "scripts" / "train_scaffold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def union_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic 2-train-city world: g4 roll-up (aaa, bbb, heldout1 all validated)
    + multiregion holdout manifest holding out heldout1."""
    g4 = tmp_path / "g4_rollup.yaml"
    g4.write_text(
        yaml.safe_dump(
            {
                "per_city": [
                    {"name": "aaa", "validated": True},
                    {"name": "bbb", "validated": True},
                    {"name": "heldout1", "validated": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout_manifest.yaml"
    holdout.write_text(
        yaml.safe_dump({"holdout_schema_version": "2.0", "held_out_cities": ["heldout1"]}),
        encoding="utf-8",
    )
    return g4, holdout


def test_union_datamodule_constructs_from_per_city_manifests(
    monkeypatch: pytest.MonkeyPatch, union_fixtures: tuple[Path, Path]
) -> None:
    g4, holdout = union_fixtures
    mod = _load_module()
    monkeypatch.setattr(mod, "_g4_rollup_path", lambda: g4)
    monkeypatch.setattr(mod, "multiregion_holdout_manifest_path", lambda release: holdout)

    cfg = ScaffoldConfig(train_set="eu-train-union", release=_RELEASE)
    dm = mod._datamodule(cfg, build=False)

    # union mode: the per-city manifest paths for train_cities(...), held-out absent
    assert dm._train_manifests == [
        training_manifest_path(_RELEASE, "aaa"),
        training_manifest_path(_RELEASE, "bbb"),
    ]
    # multiregion manifest pinned DIRECTLY, with the 2.0 schema pin travelling with it
    assert dm._holdout_manifest == holdout
    assert dm._expected_holdout_schema == "2.0"


def test_union_holdout_missing_held_out_cities_raises(
    monkeypatch: pytest.MonkeyPatch,
    union_fixtures: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """STRICT read: a holdout manifest WITHOUT held_out_cities must raise — never
    fall through to train_cities' .get(..., []) and silently exclude nothing."""
    g4, _ = union_fixtures
    bad = tmp_path / "holdout_missing_key.yaml"
    bad.write_text(yaml.safe_dump({"holdout_schema_version": "2.0"}), encoding="utf-8")
    mod = _load_module()
    monkeypatch.setattr(mod, "_g4_rollup_path", lambda: g4)
    monkeypatch.setattr(mod, "multiregion_holdout_manifest_path", lambda release: bad)

    cfg = ScaffoldConfig(train_set="eu-train-union", release=_RELEASE)
    # ValueError ONLY: the shipped contract is the explicit raise in _union_datamodule,
    # never an incidental KeyError from a dict lookup.
    with pytest.raises(ValueError, match="held_out_cities"):
        mod._datamodule(cfg, build=False)


def test_union_branch_never_calls_single_region_build(
    monkeypatch: pytest.MonkeyPatch, union_fixtures: tuple[Path, Path]
) -> None:
    """build=True in union mode must NOT call build_training_shards (it RAISES for
    train cities — the I1 fail-closed boundary). The union never rebuilds."""
    g4, holdout = union_fixtures
    mod = _load_module()
    monkeypatch.setattr(mod, "_g4_rollup_path", lambda: g4)
    monkeypatch.setattr(mod, "multiregion_holdout_manifest_path", lambda release: holdout)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("build_training_shards must not be called on the union path")

    monkeypatch.setattr(mod, "build_training_shards", _boom)
    cfg = ScaffoldConfig(train_set="eu-train-union", release=_RELEASE)
    dm = mod._datamodule(cfg, build=True)  # build flag is a no-op on the union path
    assert dm._expected_holdout_schema == "2.0"
