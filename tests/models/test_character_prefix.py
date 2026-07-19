"""Task 24b model-side teeth — the continuous character prefix position.

Axis separation (carried correction #9, the recurring conflation trap): this task
adds ONE POSITION (the 10th: 9 id positions + 1 continuous), never an embedding
ROW (the token-id span stays conditioning_id_span() == 576). The continuous
position's input embedding is ``Linear(7 -> d_model)`` of ``character_stats``;
a placeholder id occupies ids[9] and its token embedding is OVERWRITTEN by the
projection (the id value at that position is inert — proven below).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cfm.data.training.conditioning import (
    CHARACTER_PREFIX_POSITIONS,
    CHARACTER_STAT_CHANNELS,
    CONDITIONING_PREFIX_LEN,
    build_value_bearing_prefix,
    conditioning_id_span,
)
from cfm.data.training.datamodule import CHARACTER_PLACEHOLDER_ID
from cfm.models.backbone import build_backbone, subf_vocab_size
from cfm.models.micro_ar import MicroAR, MicroARConfig
from cfm.training.config import ScaffoldConfig


def _production_model(max_len: int = 32) -> tuple[MicroAR, ScaffoldConfig]:
    cfg = ScaffoldConfig(
        region="singapore",
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_len=max_len,
        accelerator="cpu",
        devices=1,
    )
    return build_backbone("transformer-ar", cfg), cfg


def _prefix_ids() -> list[int]:
    ids = build_value_bearing_prefix(
        population_density_bucket=0,
        zoning_class=1,
        road_skeleton_class=1,
        cell_density_bucket=2,
        region=None,
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=7,
        city_identity="singapore",
    )
    return [*ids, CHARACTER_PLACEHOLDER_ID]


_STATS = [0.9, 0.4, 0.3, 1.1, 0.7, 1.0, 1.0]


def test_positions_grow_by_one_and_embedding_rows_do_not():
    """THE axis-separation pin: positions = max_len + 9 + 1; embedding rows stay
    n_subf_vocab + 576 (no new vocabulary ids for the carrier)."""
    model, cfg = _production_model()
    assert CHARACTER_PREFIX_POSITIONS == 1
    assert (
        model.pos.num_embeddings
        == cfg.max_len + CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS
        == 42
    )
    assert model.embed.num_embeddings == subf_vocab_size() + conditioning_id_span()  # rows axis


def test_char_projection_is_linear_seven_to_d_model():
    model, cfg = _production_model()
    assert isinstance(model.char_proj, nn.Linear)
    assert model.char_proj.in_features == CHARACTER_STAT_CHANNELS == 7
    assert model.char_proj.out_features == cfg.d_model


def test_char_built_model_refuses_forward_without_char_stats():
    """Fail-loud: a model provisioned with the carrier must never silently train
    on the placeholder's token embedding (the silent-regime kill)."""
    model, _ = _production_model()
    ids = torch.tensor([[*_prefix_ids(), 1, 2, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="char_stats"):
        model(ids)


def test_hand_built_model_without_carrier_refuses_char_stats():
    """The mismatch is loud in BOTH directions."""
    m = MicroAR(
        MicroARConfig(d_model=32, n_layers=1, n_heads=2, n_subf_vocab=64, n_cond=8, max_len=32)
    )
    ids = torch.randint(0, 64, (1, 10))
    with pytest.raises(ValueError, match="char_stats"):
        m(ids, char_stats=torch.zeros(1, 7))
    m(ids)  # without stats the legacy fixture path is unchanged


def test_char_stats_actually_enter_the_forward():
    """Regime-distinguishing: two char_stats vectors must change the logits — a
    projection that is wired but ignored would pass every shape test."""
    model, _ = _production_model()
    ids = torch.tensor([[*_prefix_ids(), 1, 2, 3]], dtype=torch.long)
    a = model(ids, char_stats=torch.zeros(1, 7))
    b = model(ids, char_stats=torch.tensor([_STATS]))
    assert not torch.allclose(a, b)


def test_placeholder_token_id_is_inert_under_the_projection():
    """The overwrite proof: changing the placeholder id at position 9 must NOT
    change the logits when char_stats are given — the projection replaces the
    token embedding at that position entirely."""
    model, _ = _production_model()
    stats = torch.tensor([_STATS])
    base = [*_prefix_ids(), 1, 2, 3]
    swapped = list(base)
    swapped[CONDITIONING_PREFIX_LEN] = 5  # a different (valid) id in the placeholder slot
    model.eval()
    with torch.no_grad():
        a = model(torch.tensor([base], dtype=torch.long), char_stats=stats)
        b = model(torch.tensor([swapped], dtype=torch.long), char_stats=stats)
    assert torch.equal(a, b)


def test_zero_vector_and_absent_layer_vector_are_distinguishable_downstream():
    """The PI obligation, downstream half: the flag bit that separates a
    genuinely-zero cell from an absent-layer cell must reach the model as a
    DIFFERENT carrier state (different logits), or the flag carries nothing."""
    from cfm.data.training.build_shards import character_stats_for_cell

    zero_road = character_stats_for_cell([], [0.0])  # channel 4 == 0.0, flag 1
    absent = character_stats_for_cell([], [])  # channel 4 == 0.0, flag 0
    assert zero_road != absent  # differ only in the flag bit (proven data-side)
    model, _ = _production_model()
    ids = torch.tensor([[*_prefix_ids(), 1, 2, 3]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        a = model(ids, char_stats=torch.tensor([list(zero_road)]))
        b = model(ids, char_stats=torch.tensor([list(absent)]))
    assert not torch.allclose(a, b)


def test_char_proj_weight_receives_gradient_through_the_overwrite():
    """In-suite gradient pin (quality review #1): a future ``torch.no_grad()`` /
    ``.detach()`` around the embedding overwrite would leave the logits DIFFERENT
    (every other test here still passes) while silently cutting ``char_proj`` out
    of the graph — the loss still backprops through all other params. The pin:
    ``training_loss(...).loss.backward()`` must land a nonzero grad on
    ``char_proj.weight``."""
    model, _ = _production_model()
    body = torch.randint(0, 64, (1, 6))
    ids = torch.cat([torch.tensor([_prefix_ids()], dtype=torch.long), body], dim=1)
    out = model.training_loss(ids, prefix_len=torch.tensor([10]), char_stats=torch.tensor([_STATS]))
    out.loss.backward()
    grad = model.char_proj.weight.grad
    assert grad is not None
    assert float(grad.norm()) > 0.0


def test_config_refuses_char_position_without_char_stats():
    """Quality review #4: ``char_position`` set while ``n_char_stats == 0`` was
    silently ignored (the carrier never built) — now a loud config error."""
    with pytest.raises(ValueError, match="char_position"):
        MicroARConfig(
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_subf_vocab=64,
            n_cond=8,
            max_len=32,
            n_char_stats=0,
            char_position=9,
        )


def test_training_loss_masks_all_ten_prefix_targets():
    """Verified, not assumed (mini-spec §2): the mask keys on per-example
    prefix_len, so prefix_len=10 supervises exactly (T - 10) targets/example."""
    model, _ = _production_model()
    T, B = 20, 2
    prefix = _prefix_ids()
    body = torch.randint(0, 64, (B, T - len(prefix)))
    ids = torch.cat([torch.tensor([prefix, prefix]), body], dim=1)
    out = model.training_loss(
        ids,
        prefix_len=torch.tensor([10, 10]),
        char_stats=torch.tensor([_STATS, _STATS]),
    )
    assert out.loss.requires_grad
    assert out.n_supervised_positions == (T - 10) * B


def test_generation_threads_char_stats_and_is_seed_reproducible():
    from cfm.inference.generate import generate_cell_tokens

    model, _ = _production_model()
    prefix = _prefix_ids()
    a = generate_cell_tokens(model, prefix=prefix, max_new=8, seed=7, char_stats=_STATS)
    b = generate_cell_tokens(model, prefix=prefix, max_new=8, seed=7, char_stats=_STATS)
    assert a == b  # seeded -> reproducible
    assert len(a) == 8
    # without char_stats the char-built model refuses (no silent placeholder gen)
    with pytest.raises(ValueError, match="char_stats"):
        generate_cell_tokens(model, prefix=prefix, max_new=4, seed=7)


def test_generate_refuses_pre_24b_nine_id_prefix_with_char_stats():
    """Quality review #2: a caller passing the pre-24b 9-id prefix layout to a
    char-built generation would otherwise get its FIRST CELL TOKEN's embedding
    silently replaced by the projection (char_position == 9 lands on it). At
    generation time the prefix length is known — the layout check is loud."""
    from cfm.inference.generate import generate_cell_tokens

    model, _ = _production_model()
    nine_ids = _prefix_ids()[:-1]  # pre-24b layout: no placeholder slot
    assert len(nine_ids) == CONDITIONING_PREFIX_LEN == 9
    with pytest.raises(ValueError, match="10-position"):
        generate_cell_tokens(model, prefix=nine_ids, max_new=4, seed=7, char_stats=_STATS)
    # the 10-position layout passes the check and generates
    out = generate_cell_tokens(model, prefix=_prefix_ids(), max_new=4, seed=7, char_stats=_STATS)
    assert len(out) == 4
