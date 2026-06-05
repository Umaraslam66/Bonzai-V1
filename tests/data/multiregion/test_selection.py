"""Single-UTM filter — the centroid trap. The filter must reject on the FULL
bbox, not the centroid; a zone-straddling city (center in one zone, area spilling
into the next) must be REJECTED. Must-distinguish: accept a clean single-zone
city AND reject a straddler."""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.multiregion import selection


def test_accepts_clean_single_zone_city():
    # Berlin bbox — entirely within UTM zone 33 → EPSG:25833.
    ok, crs = selection.single_utm_zone_ok((13.0883, 52.3383, 13.7612, 52.6755))
    assert ok and crs == "EPSG:25833"


def test_rejects_zone_straddler_even_though_centroid_is_clean():
    # bbox spans the 12°E zone 32/33 boundary: W edge 11.9 (zone 32), E edge 12.1
    # (zone 33). Centroid 12.0 lands cleanly in a single zone, so a centroid-based
    # classifier would PASS this — the full-bbox check must REJECT it.
    bbox = (11.9, 52.0, 12.1, 52.5)
    assert selection._utm_zone(11.9) != selection._utm_zone(12.1)  # genuinely straddles
    ok, crs = selection.single_utm_zone_ok(bbox)
    assert not ok and crs is None


def test_rejects_southern_hemisphere():
    ok, _ = selection.single_utm_zone_ok((13.0, -1.0, 13.5, -0.5))
    assert not ok


def test_rejects_outside_european_zone_range():
    # A far-east longitude → UTM zone outside the ETRS89 European range.
    ok, _ = selection.single_utm_zone_ok((100.0, 1.0, 100.4, 1.4))
    assert not ok


def test_canary_manifest_labels_consistent_with_configs_and_filter():
    # The ratified labels live in the committed manifest (not the region configs).
    # Guard: complete axis coverage AND manifest projected_crs == the region
    # config's == single_utm_zone_ok(bbox) — manifest/config/filter cannot drift.
    repo = Path(__file__).resolve().parents[3]
    cities = selection.load_canary_manifest(repo / "configs" / "multiregion" / "canary_v1.yaml")
    by = {c["name"]: c for c in cities}
    assert set(by) == {"prague", "barcelona", "milton_keynes", "munich", "umea"}
    assert {c["morphology"] for c in cities} >= {
        "medieval-organic",
        "planned-grid",
        "modernist-sprawl",
        "mixed",
    }
    assert {c["density"] for c in cities} == {"dense-core", "moderate", "sparse"}
    assert len({c["projected_crs"] for c in cities}) == 5  # 5 distinct CRS paths
    for name, c in by.items():
        cfg = yaml.safe_load((repo / "configs" / "data" / "regions" / f"{name}.yaml").read_text())
        assert c["projected_crs"] == cfg["projected_crs"]
        ok, crs = selection.single_utm_zone_ok(tuple(cfg["fallback_bbox"]))
        assert ok and crs == c["projected_crs"]


def test_write_region_config_roundtrips_and_crs_matches_filter(tmp_path):
    ok, crs = selection.single_utm_zone_ok((13.0883, 52.3383, 13.7612, 52.6755))
    assert ok
    cand = selection.CityCandidate(
        name="berlin",
        country_code="DE",
        admin_level="region",
        bbox=(13.0883, 52.3383, 13.7612, 52.6755),
        morphology="mixed",
        density="dense-core",
        projected_crs=crs,
    )
    out = selection.write_region_config(cand, tmp_path)
    data = yaml.safe_load(Path(out).read_text())
    assert data["name"] == "berlin"
    assert data["projected_crs"] == crs == "EPSG:25833"
    assert data["admin"]["country_code"] == "DE"
    assert data["fallback_bbox"] == [13.0883, 52.3383, 13.7612, 52.6755]
