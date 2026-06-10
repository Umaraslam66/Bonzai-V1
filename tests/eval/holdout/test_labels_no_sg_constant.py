import dataclasses

from cfm.eval.holdout.labels import TileLabels


def test_sg_constants_are_unscored_or_absent():
    # country / climate_zone / era_class are NOT TileLabels fields at all (structurally unscorable)
    fields = {f.name for f in dataclasses.fields(TileLabels)}
    assert {"country", "climate_zone", "era_class"}.isdisjoint(fields)
    # the scored morphology signal is the sub-D stratum, not the sub-C constant
    assert "morphology_stratum" in fields
