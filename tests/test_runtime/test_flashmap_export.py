# coding=utf-8
"""``specforge export --to hf`` of a FlashMap draft writes the FlashMap release
format (the published folders' config.json keys, release weight names, the
draft-vocabulary ranking), which loads back into SpecForge (CPU only)."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from specforge.export.to_hf import export_to_hf
from specforge.modeling.draft.flashmap import FlashMapConfig, FlashMapDraftModel

VOCAB = 64
# The keys of a published release folder's config.json (FlashMap-Qwen3-8B), less
# auto_map, which is written when the flashmap package is installed.
RELEASE_KEYS = {
    "architectures", "block_size", "cond_dim", "draft_vocab_size", "dspark_config",
    "dtype", "hidden_size", "intermediate_size", "jump_grids", "layer_norm_eps",
    "max_position_embeddings", "model_type", "noise_gate", "num_attention_heads",
    "num_hidden_layers", "num_steps", "qk_norm_eps", "rms_norm_eps", "rope_theta",
    "speculators_config", "speculators_model_type", "target_hidden_size",
    "target_layer_ids", "target_model", "tie_word_embeddings", "time_embed_dim",
    "time_scale", "time_schedule", "transformer_layer_config",
    "transformers_version", "vocab_size",
}  # fmt: skip


def _draft_config(root, flow_map=True):
    config = FlashMapConfig(
        target_hidden_size=32,
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
    path = os.path.join(root, "draft.json")
    config.to_json_file(path)
    return config, path


def _checkpoint(root, config, name="training_state.pt"):
    torch.manual_seed(0)
    model = FlashMapDraftModel(config).to(torch.bfloat16)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    path = os.path.join(root, name)
    torch.save({"draft_state_dict": state, "strategy": "flashmap"}, path)
    return state, path


def _ranking(root):
    ids = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(1))
    path = os.path.join(root, "ranking.safetensors")
    save_file({"ids": ids, "counts": torch.arange(VOCAB, 0, -1)}, path)
    return ids, path


class TestFlashMapReleaseExport(unittest.TestCase):
    def test_map_exports_to_the_release_format(self):
        with tempfile.TemporaryDirectory() as root:
            config, config_path = _draft_config(root)
            state, checkpoint = _checkpoint(root, config)
            ids, ranking = _ranking(root)
            out = os.path.join(root, "release")
            export_to_hf(
                checkpoint,
                config_path,
                out,
                target_model="toy/target",
                draft_vocab=ranking,
                draft_vocab_size=32,
            )
            with open(os.path.join(out, "config.json")) as handle:
                release = json.load(handle)
            self.assertEqual(RELEASE_KEYS, set(release) - {"auto_map"})
            self.assertEqual("flashmap", release["model_type"])
            self.assertEqual(32, release["draft_vocab_size"])
            self.assertEqual([0, 1], release["target_layer_ids"])
            self.assertEqual("toy/target", release["target_model"])
            verifier = release["speculators_config"]["verifier"]
            self.assertEqual("toy/target", verifier["name_or_path"])
            proposal = release["speculators_config"]["proposal_methods"][0]
            self.assertEqual(4, proposal["speculative_tokens"])
            # no tokenizer for the toy target: the last embedding row
            self.assertEqual(
                {"markov_rank": 1, "markov_head_type": "vanilla", "mask_token_id": 63},
                release["dspark_config"],
            )
            self.assertEqual(131072, release["max_position_embeddings"])
            weights = load_file(os.path.join(out, "model.safetensors"))
            self.assertEqual(set(state), set(weights))
            for name, tensor in state.items():
                self.assertTrue(torch.equal(tensor, weights[name]), name)
            shipped = load_file(os.path.join(out, "draft_vocab.safetensors"))
            self.assertTrue(torch.equal(ids, shipped["ids"]))
            # the release folder loads back into SpecForge (e.g. to warm-start)
            loaded = FlashMapDraftModel.from_pretrained(out)
            self.assertTrue(loaded.flow_map)
            for name, tensor in loaded.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor.to(torch.bfloat16), state[name]), name
                )

    def test_flow_exports_with_flow_map_false(self):
        with tempfile.TemporaryDirectory() as root:
            config, _ = _draft_config(root, flow_map=False)
            state, checkpoint = _checkpoint(root, config)
            out = os.path.join(root, "flow")
            # the map's draft config serves for the flow: the weights decide
            _, map_config_path = _draft_config(root, flow_map=True)
            export_to_hf(checkpoint, map_config_path, out, target_model="toy/target")
            with open(os.path.join(out, "config.json")) as handle:
                release = json.load(handle)
            self.assertFalse(release["flow_map"])
            self.assertNotIn("draft_vocab_size", release)
            self.assertNotIn(
                "jump_embed.linear_1.weight", load_file(f"{out}/model.safetensors")
            )
            self.assertEqual(set(state), set(load_file(f"{out}/model.safetensors")))

    def test_every_draft_weight_must_be_in_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            config, config_path = _draft_config(root)
            state, _ = _checkpoint(root, config)
            del state["x_embed.weight"]  # "*embed*", but trained
            path = os.path.join(root, "partial.pt")
            torch.save({"draft_state_dict": state, "strategy": "flashmap"}, path)
            with self.assertRaisesRegex(ValueError, "x_embed.weight"):
                export_to_hf(
                    path, config_path, os.path.join(root, "out"), target_model="t"
                )

    def test_options_belong_to_their_drafts(self):
        with tempfile.TemporaryDirectory() as root:
            config, config_path = _draft_config(root)
            _, checkpoint = _checkpoint(root, config)
            out = os.path.join(root, "out")
            with self.assertRaisesRegex(ValueError, "do not apply"):
                export_to_hf(checkpoint, config_path, out, vocab_mapping_path="m.pt")
            with self.assertRaisesRegex(ValueError, "needs a --draft-vocab"):
                export_to_hf(
                    checkpoint, config_path, out, target_model="t", draft_vocab_size=8
                )
            with self.assertRaisesRegex(ValueError, "needs the target model"):
                export_to_hf(checkpoint, config_path, out)


if __name__ == "__main__":
    unittest.main()
