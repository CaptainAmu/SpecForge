# coding=utf-8
"""FlashMap draft model, its two-stage training wrapper, and its registration."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.flashmap.model import BOUNDARY, DIAGONAL, OnlineFlashMapModel
from specforge.algorithms.flashmap.providers import (
    initialize_map_from_flow,
    populate_target_defaults,
)
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft import resolve_draft
from specforge.modeling.draft.flashmap import (
    DecodingErrorSchedule,
    FlashMapConfig,
    FlashMapDraftModel,
    create_context_mask,
)

VOCAB = 64
TARGET_HIDDEN = 32


def tiny_config(flow_map: bool = True) -> FlashMapConfig:
    config = FlashMapConfig(
        target_hidden_size=TARGET_HIDDEN,
        vocab_size=VOCAB,
        target_layer_ids=[0, 1],
        block_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        cond_dim=16,
        time_embed_dim=16,
        flow_map=flow_map,
    )
    config.architectures = ["FlashMapDraftModel"]
    return config


def randomize(model: nn.Module) -> nn.Module:
    """Move every weight off its init, so zero-initialized paths are live."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    return model


def wrapper(draft, stage="flow", flow=None, **kwargs):
    torch.manual_seed(1)
    embed = nn.Embedding(VOCAB, TARGET_HIDDEN)
    head = nn.Linear(TARGET_HIDDEN, VOCAB, bias=False)
    embed.requires_grad_(False)
    head.requires_grad_(False)
    options = dict(
        stage=stage,
        flow=flow,
        attention_backend="sdpa",
        num_anchors=6,
        loss_decay_gamma=7.0,
        objective_chunk_blocks=2,
    )
    options.update(kwargs)
    return OnlineFlashMapModel(draft, head, embed, **options)


def batch(seq_len: int = 24):
    torch.manual_seed(2)
    input_ids = torch.randint(0, VOCAB, (2, seq_len))
    loss_mask = torch.ones(2, seq_len)
    loss_mask[0, :5] = 0
    loss_mask[1, -6:] = 0
    hidden_states = torch.randn(2, seq_len, 2 * TARGET_HIDDEN)
    return input_ids, hidden_states, loss_mask


class FlashMapModelTest(unittest.TestCase):
    def test_registered_and_built_from_config(self):
        self.assertIs(resolve_draft("FlashMapDraftModel"), FlashMapDraftModel)
        model = AutoDraftModel.from_config(tiny_config())
        self.assertIsInstance(model, FlashMapDraftModel)

    def test_released_weight_names(self):
        keys = set(FlashMapDraftModel(tiny_config()).state_dict())
        for key in (
            "fc.weight",
            "hidden_norm.weight",
            "anchor_proj.bias",
            "anchor_norm.weight",
            "state_norm.weight",
            "noise_norm.weight",
            "noise_gate",
            "x_embed.weight",
            "time_embed.linear_1.weight",
            "jump_embed.linear_2.bias",
            "layers.1.self_attn_adaln.linear.weight",
            "layers.1.cross_attn.k_proj.weight",
            "layers.1.cross_attn.v_proj.weight",
            "layers.1.self_attn.q_norm.weight",
            "layers.1.mlp.fc2.bias",
            "final_adaln.linear.weight",
            "head.bias",
        ):
            self.assertIn(key, keys)
        self.assertFalse(any("embed_tokens" in key or "lm_head" in key for key in keys))
        flow_keys = set(FlashMapDraftModel(tiny_config(flow_map=False)).state_dict())
        self.assertEqual(
            {"noise_norm.weight", "noise_gate"}
            | {key for key in keys if key.startswith("jump_embed.")},
            keys - flow_keys,
        )

    def test_jump_grids_match_the_released_qwen3_schedule(self):
        grids = FlashMapConfig().jump_grids
        self.assertEqual([0.0, 1.0], grids["1"])
        self.assertAlmostEqual(0.8172569274902344, grids["2"][1], places=6)
        self.assertAlmostEqual(0.8379518389701843, grids["4"][3], places=6)

    def test_context_mask_sees_hidden_states_before_the_anchor_and_its_token(self):
        anchors = torch.tensor([[0, 3]])
        mask = create_context_mask(anchors, seq_len=5, block_size=2, use_flex=False)
        self.assertEqual((1, 1, 4, 10), tuple(mask.shape))
        second_block = mask[0, 0, 2].nonzero().flatten().tolist()
        self.assertEqual([0, 1, 2, 8], second_block)
        self.assertEqual([5], mask[0, 0, 0].nonzero().flatten().tolist())

    def test_initialized_map_is_its_flow_on_the_diagonal(self):
        flow = randomize(FlashMapDraftModel(tiny_config(flow_map=False)))
        draft = FlashMapDraftModel(tiny_config())
        initialize_map_from_flow(draft, flow)
        model = wrapper(draft, stage="map", flow=flow)
        input_ids, hidden_states, _ = batch()
        anchors = torch.tensor([[3, 9], [1, 12]])
        context = model._block_context(draft, input_ids, hidden_states, anchors)
        positions = (anchors.unsqueeze(-1) + torch.arange(1, 5)).flatten(1)
        state = torch.randn(2, 8, VOCAB)
        start = torch.rand(2, 8)
        expected = model._evaluate(flow, state, start, None, context, positions)
        # The end-time embedding is zero at init and the noise read-out is
        # gated by t - s, so delta_{s,s} is the flow's denoiser D_s.
        diagonal = model._evaluate(draft, state, start, start, context, positions)
        torch.testing.assert_close(diagonal, expected)
        jump = model._evaluate(
            draft, state, start, torch.ones_like(start), context, positions
        )
        self.assertFalse(torch.allclose(jump, expected))

    def test_flow_stage_trains_with_token_pooled_loss_terms(self):
        draft = randomize(FlashMapDraftModel(tiny_config(flow_map=False)))
        model = wrapper(draft)
        loss, accuracy, metrics = model(*batch())
        numerator, denominator = metrics["loss_terms"]
        torch.testing.assert_close(loss, numerator / denominator)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)
        bins = sum(
            metrics["ratio_metrics"][f"flashmap/ce_loss@tau{index}"][0]
            for index in range(4)
        )
        torch.testing.assert_close(bins, numerator.detach())
        numerator.backward()
        self.assertTrue(all(p.grad is not None for p in draft.parameters()))

    def test_map_stage_starts_at_zero_kl_on_the_diagonal(self):
        flow = randomize(FlashMapDraftModel(tiny_config(flow_map=False)))
        draft = FlashMapDraftModel(tiny_config())
        initialize_map_from_flow(draft, flow)
        model = wrapper(draft, stage="map", flow=flow, diagonal_fraction=1.0)
        _, _, metrics = model(*batch())
        kl, weight = metrics["ratio_metrics"]["map_kl"]
        self.assertLess(abs(float(kl / weight)), 1e-5)

    def test_map_stage_trains_only_the_student(self):
        flow = randomize(FlashMapDraftModel(tiny_config(flow_map=False)))
        draft = FlashMapDraftModel(tiny_config())
        initialize_map_from_flow(draft, flow)
        model = wrapper(draft, stage="map", flow=flow)
        self.assertFalse(any(name.startswith("_flow") for name in model.state_dict()))
        loss, _, metrics = model(*batch())
        metrics["loss_terms"][0].backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in flow.parameters()))
        self.assertIsNotNone(draft.jump_embed.linear_2.weight.grad)
        self.assertIsNotNone(draft.noise_norm.weight.grad)

    def test_semigroup_times_follow_the_official_batch_split(self):
        model = wrapper(
            FlashMapDraftModel(tiny_config()),
            stage="map",
            flow=FlashMapDraftModel(tiny_config(flow_map=False)),
            boundary_prob=0.25,
        )
        torch.manual_seed(3)
        kind, s, u, t = model._sample_map_times((4, 16), "cpu")
        self.assertEqual(32, int((kind == DIAGONAL).sum()))
        self.assertEqual(8, int((kind == BOUNDARY).sum()))
        boundary = kind == BOUNDARY
        self.assertTrue(torch.all(s[boundary] == 0) and torch.all(t[boundary] == 1))
        self.assertTrue(torch.equal(s[kind == DIAGONAL], t[kind == DIAGONAL]))
        self.assertTrue(torch.all((s <= u) & (u <= t)))

    def test_stage_and_network_must_agree(self):
        with self.assertRaisesRegex(ValueError, "flow_map"):
            wrapper(FlashMapDraftModel(tiny_config()))
        with self.assertRaisesRegex(ValueError, "frozen flow"):
            wrapper(FlashMapDraftModel(tiny_config()), stage="map")

    def test_schedule_inverts_decoding_error_rate(self):
        schedule = DecodingErrorSchedule(VOCAB)
        taus, times = schedule.sample((1000,), "cpu")
        self.assertTrue(torch.all((times >= 0) & (times <= 1)))
        order = taus.argsort()
        self.assertTrue(torch.all(times[order].diff() >= 0))


class FlashMapProviderTest(unittest.TestCase):
    def test_generated_config_describes_the_dit_not_the_target(self):
        payload = {
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "intermediate_size": 9728,
            "rope_theta": 1000000,
            "num_hidden_layers": 6,
        }
        config = SimpleNamespace(
            model=SimpleNamespace(target_model_path="Qwen/Qwen3-4B"),
            training=SimpleNamespace(flashmap_stage="map"),
        )
        populate_target_defaults(payload, SimpleNamespace(num_hidden_layers=36), config)
        self.assertEqual([1, 9, 17, 25, 33], payload["target_layer_ids"])
        self.assertEqual(2560, payload["target_hidden_size"])
        self.assertTrue(payload["flow_map"])
        generated = FlashMapConfig(**copy.deepcopy(payload))
        self.assertEqual(2048, generated.hidden_size)
        self.assertEqual(16, generated.num_attention_heads)
        self.assertEqual(10000.0, generated.rope_theta)

    def test_map_init_rejects_a_mismatched_flow(self):
        draft = FlashMapDraftModel(tiny_config())
        other = tiny_config(flow_map=False)
        other.num_hidden_layers = 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            initialize_map_from_flow(draft, FlashMapDraftModel(other))


if __name__ == "__main__":
    unittest.main()
