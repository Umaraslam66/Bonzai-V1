import dataclasses

from cfm.eval.holdout.labels import UNSCORED_V1_DIMENSIONS, TileLabels


def test_sg_constants_are_unscored_or_absent():
    # sub_c_morphology_class + admin_region (fields that REACH TileLabels) are UNSCORED
    assert "morphology_class" in UNSCORED_V1_DIMENSIONS
    assert "region" in UNSCORED_V1_DIMENSIONS
    # country / climate_zone / era_class are NOT TileLabels fields at all (structurally unscorable)
    fields = {f.name for f in dataclasses.fields(TileLabels)}
    assert {"country", "climate_zone", "era_class"}.isdisjoint(fields)
    # the scored morphology signal is the sub-D stratum, not the sub-C constant
    assert "morphology_stratum" in fields
