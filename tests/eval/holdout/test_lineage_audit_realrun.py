from __future__ import annotations

import pytest
import yaml

from cfm.data.training.holdout_guard import run_holdout_audit
from cfm.eval.holdout.lineage_audit import Artifact, HoldoutLeakError
from cfm.eval.holdout.paths import holdout_manifest_path, multiregion_holdout_manifest_path

_REL = "2026-04-15.0"


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_realrun_fires_on_leaked_held_out_city_tile():
    m = _load(multiregion_holdout_manifest_path(_REL))  # frozen EU manifest, schema 2.0
    t = m["regions"]["krakow"]["tiles"][0]  # a REAL krakow held-out tile-ref
    leaked = Artifact("train/leak", frozenset({("krakow", int(t["tile_i"]), int(t["tile_j"]))}))
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(m, [leaked])  # default 2.0 -> schema ok -> leak caught


def test_realrun_passes_clean():
    m = _load(multiregion_holdout_manifest_path(_REL))
    run_holdout_audit(m, [Artifact("train/ok", frozenset({("hamburg", 1, 1)}))])  # no raise


def test_frozen_sg_manifest_refused_under_default_2_0():
    sg = _load(holdout_manifest_path(_REL))  # frozen SG manifest, schema 1.0
    with pytest.raises(HoldoutLeakError, match="schema"):  # #16 backstop firing on real stale
        run_holdout_audit(sg, [Artifact("train/ok", frozenset({("hamburg", 1, 1)}))])


def test_frozen_sg_manifest_passes_with_explicit_1_0():
    sg = _load(holdout_manifest_path(_REL))  # frozen SG manifest, schema 1.0
    run_holdout_audit(
        sg, [Artifact("train/ok", frozenset({("hamburg", 1, 1)}))], expected_schema_version="1.0"
    )  # legacy path works
