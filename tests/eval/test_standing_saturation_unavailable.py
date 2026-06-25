"""Saturation UNAVAILABLE (spec §3, D4): mamba-hybrid seed23's bake-off training log is
genuinely absent on disk. Saturation is recorded UNAVAILABLE — NOT guessed, NOT fabricated,
NOT hunted. Its perplexity-gap + geometry-validity still compute from the checkpoint, so the
seed still enters the comparison.
"""

from __future__ import annotations

from cfm.eval.standing.harness import render_table, run_saturation


def test_run_saturation_unavailable_for_mamba_seed23(tmp_path):
    # No version dir exists for this run; must NOT raise, must NOT touch disk hunting — it
    # records UNAVAILABLE from the committed registry.
    res = run_saturation(tmp_path, backbone="mamba-hybrid", seed=23)
    assert res["classification"] == "UNAVAILABLE"
    assert res.get("version_dir") is None


def test_run_saturation_still_fails_loud_on_unexpected_no_match(tmp_path):
    # A no-match for a (backbone,seed) NOT on the UNAVAILABLE registry and NOT overridden is a real
    # error (e.g. wrong logs-dir) and must still FAIL LOUD — UNAVAILABLE is registry-driven, not a
    # catch-all swallow. transformer-ar seed=99 is neither overridden nor unavailable.
    import pytest

    with pytest.raises(ValueError, match=r"(?i)no.*match"):
        run_saturation(tmp_path, backbone="transformer-ar", seed=99)


def _minimal_result(saturation):
    geom = {
        "self_term_frac": 1.0,
        "decode_frac": 1.0,
        "closure_gap_median": 0.02,
        "closure_within_5pct": 0.9,
        "median_components_per_segment": 1.0,
        "dangling_endpoint_frac": 0.9,
    }
    return {
        "ckpt_id": "mamba-hybrid-seed23",
        "meta": {
            "backbone": "mamba-hybrid",
            "d_model": 512,
            "n_layers": 24,
            "global_step": 112000,
            "conditioning_scheme": "value-char-v1",
            "conditioning_ablation": "full",
        },
        "perplexity_gap": {
            "n_cells": 8000,
            "cities": ["krakow"],
            "macro_shuffle_effective_fraction": 1.0,
            "macro_shuffle_floor": 0.95,
            "macro_only_reliable": True,
            "macro_only_primary": {
                "gap_nats_per_token": 0.006,
                "fraction_positive": 0.68,
                "sign_test_significant_at_p": True,
            },
            "full_secondary": {"gap_nats_per_token": 0.45, "fraction_positive": 0.98},
        },
        "saturation": saturation,
        "geometry_validity": {"dense_urban": geom},
    }


def test_render_table_handles_unavailable_saturation():
    r = _minimal_result(
        {
            "classification": "UNAVAILABLE",
            "version_dir": None,
            "reason": "no bake-off training log on disk",
        }
    )
    md = render_table(r)  # must NOT KeyError on the missing final_step/slope fields
    assert "UNAVAILABLE" in md
    assert "## (2) Saturation" in md


def test_render_table_still_renders_normal_saturation():
    r = _minimal_result(
        {
            "version_dir": "version_25",
            "classification": "PLATEAUED",
            "final_step": 112549,
            "final_loss": 2.0,
            "loss_at_80pct": 2.1,
            "loss_at_90pct": 2.05,
            "final_window_slope": 0.0008,
            "final_window_noise": 0.27,
            "plateau_threshold": 0.013,
            "n_window_points": 20,
        }
    )
    md = render_table(r)
    assert "PLATEAUED" in md
    assert "112549" in md
