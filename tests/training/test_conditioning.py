from __future__ import annotations

from cfm.data.training import conditioning as C
from cfm.eval.holdout import labels as L


def test_conditioning_derivation_is_the_same_function_as_labels():
    """Trigger 2: prove SAME-SOURCE structurally (fails the moment someone forks
    the derivation), not just equal-values-today."""
    assert C.derive_tile_conditioning is L._derive_tile_conditioning


def test_read_tile_labels_delegates_to_the_shared_derivation():
    """The eval's reader and the model conditioning both resolve to the shared
    core — so read_tile_labels must call _derive_tile_conditioning (one source)."""
    import inspect

    src = inspect.getsource(L.read_tile_labels)
    assert "_derive_tile_conditioning(" in src


def test_id_block_offset_is_above_sealed_subf_vocab():
    from cfm.data.sub_f.vocab import vocab_tag_to_id

    max_subf = max(vocab_tag_to_id().values())
    assert C.CONDITIONING_ID_BASE > max_subf  # appended above, never reindexes


def test_id_block_mapping_is_one_source_and_append_only():
    m = C.conditioning_field_to_id()
    assert all(v >= C.CONDITIONING_ID_BASE for v in m.values())
    # append-only contract: ids assigned in a fixed recorded order, contiguous
    assert list(m.values()) == sorted(m.values())
    assert list(m.values()) == list(range(C.CONDITIONING_ID_BASE, C.CONDITIONING_ID_BASE + len(m)))


def test_conditioning_prefix_carries_values_at_their_recorded_slots():
    """Hand-enumerated cross-ref WITHOUT using the builder in the expected
    computation (Gate 6): a known label set lands its values at the recorded
    field slots."""
    labels = L.TileLabels(
        tile_i=0,
        tile_j=0,
        population_density_bucket=2,
        cell_density_buckets=(1,),
        morphology_stratum=L.MorphologyStratum(
            dominant_zoning_class=3, modal_road_skeleton_class=1
        ),
        coastal_inland_river=0,
        admin_region="SG",
        sub_c_morphology_class="Asian-megacity",
    )
    prefix = C.conditioning_prefix_ids(labels, cell_density_bucket=1, seed=7)
    base, m = C.CONDITIONING_ID_BASE, C.conditioning_field_to_id()
    assert prefix[m["population_density_bucket"] - base] == 2
    assert prefix[m["zoning_class"] - base] == 3
    assert prefix[m["road_skeleton_class"] - base] == 1
    assert prefix[m["cell_density_bucket"] - base] == 1
    assert prefix[m["seed"] - base] == 7
