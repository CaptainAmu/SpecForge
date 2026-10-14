"""The FlashMap sglang export: release names, norm weights and the sglang
config, from a tiny synthetic DeepSpec flow-map checkpoint (CPU only)."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from specforge.export.flashmap import export_flashmap_to_sglang

H, D, LAYERS, VOCAB = 8, 16, 2, 151936


def _deepspec_checkpoint(path):
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    state = {
        "embed_tokens.weight": rand(4, H),
        "lm_head.weight": rand(4, H),
        "fc.weight": rand(D, 5 * H),
        "hidden_norm.scale": rand(D),
        "anchor_proj.weight": rand(D, H),
        "anchor_proj.bias": rand(D),
        "anchor_norm.scale": rand(D),
        "state_norm.scale": rand(H),
        "noise_norm.scale": rand(H),
        "noise_gate": torch.ones(1, dtype=torch.bfloat16),
        "denoiser.x_embed.weight": rand(D, H),
        "denoiser.x_embed.bias": rand(D),
        "denoiser.head.weight": rand(H, D),
        "denoiser.head.bias": rand(H),
        "denoiser.final_norm.cond_proj.1.weight": rand(3 * D, 4),
        "denoiser.final_norm.cond_proj.1.bias": rand(3 * D),
    }
    for emb in ("temb", "temb_jump"):
        for i in (0, 2):
            state[f"denoiser.{emb}.mlp.{i}.weight"] = rand(4, 4)
            state[f"denoiser.{emb}.mlp.{i}.bias"] = rand(4)
    for n in range(LAYERS):
        p = f"denoiser.blocks.{n}."
        for norm in ("adaln_self", "adaln_cross", "adaln_mlp"):
            state[p + f"{norm}.cond_proj.1.weight"] = rand(3 * D, 4)
            state[p + f"{norm}.cond_proj.1.bias"] = rand(3 * D)
        for attn in ("self_attn", "cross_attn"):
            for w in ("wq", "wo"):
                state[p + f"{attn}.{w}.weight"] = rand(D, D)
            state[p + f"{attn}.wkv.weight"] = rand(2 * D, D)
            state[p + f"{attn}.qnorm.scale"] = rand(D // 2)
            state[p + f"{attn}.knorm.scale"] = rand(D // 2)
        state[p + "mlp.0.weight"] = rand(4 * D, D)
        state[p + "mlp.0.bias"] = rand(4 * D)
        state[p + "mlp.3.weight"] = rand(D, 4 * D)
        state[p + "mlp.3.bias"] = rand(D)
    config = {
        "architectures": ["Qwen3DFlashierSemigroupFlowMapModel"],
        "model_type": "qwen3",
        "hidden_size": H,
        "vocab_size": VOCAB,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "num_target_layers": 36,
        "block_size": 16,
        "dit_embed_dim": D,
        "dit_num_heads": 2,
        "dit_cond_dim": 4,
        "num_hidden_layers": LAYERS,
        "state_featurization": "posterior_noise",
        "noise_gate": "jump",
        "time_schedule": "decoding_error",
        "max_position_embeddings": 40960,
    }
    os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(config, handle)
    save_file(state, os.path.join(path, "model.safetensors"))
    return state


class TestFlashMapExport(unittest.TestCase):
    def test_deepspec_checkpoint_to_sglang(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _deepspec_checkpoint(os.path.join(tmp, "src"))
            ranking = torch.randperm(VOCAB)
            save_file({"ids": ranking}, os.path.join(tmp, "vocab.safetensors"))
            out = export_flashmap_to_sglang(
                os.path.join(tmp, "src"),
                os.path.join(tmp, "out"),
                draft_vocab=os.path.join(tmp, "vocab.safetensors"),
                num_steps=2,
            )
            state = load_file(os.path.join(out, "model.safetensors"))
            with open(os.path.join(out, "config.json")) as handle:
                config = json.load(handle)

        self.assertNotIn("embed_tokens.weight", state)
        self.assertNotIn("lm_head.weight", state)
        self.assertTrue(torch.equal(state["draft_vocab_ids"], ranking))
        # Norms carry the multiplicative weight 1 + scale, in bf16.
        self.assertTrue(
            torch.equal(state["state_norm.weight"], 1 + source["state_norm.scale"])
        )
        self.assertTrue(
            torch.equal(
                state["layers.1.cross_attn.k_norm.weight"],
                1 + source["denoiser.blocks.1.cross_attn.knorm.scale"],
            )
        )
        keys, values = source["denoiser.blocks.0.cross_attn.wkv.weight"].chunk(2)
        self.assertTrue(torch.equal(state["layers.0.cross_attn.k_proj.weight"], keys))
        self.assertTrue(torch.equal(state["layers.0.cross_attn.v_proj.weight"], values))
        self.assertTrue(
            torch.equal(
                state["layers.0.mlp.fc2.weight"],
                source["denoiser.blocks.0.mlp.3.weight"],
            )
        )
        self.assertIn("jump_embed.linear_2.bias", state)
        self.assertIn("final_adaln.linear.weight", state)
        # 22 weights outside the layers, 22 per layer and the ranking.
        self.assertEqual(len(state), 22 + 22 * LAYERS + 1)

        self.assertEqual(config["architectures"], ["FlashMapDraftModel"])
        self.assertEqual(config["num_attention_heads"], 2)
        self.assertEqual(config["head_dim"], D // 2)
        self.assertEqual(config["num_hidden_layers"], LAYERS)
        self.assertEqual(config["hidden_size"], H)
        self.assertEqual(config["draft_vocab_size"], 65536)
        self.assertEqual(config["markov_rank"], 1)
        flashmap = config["flashmap_config"]
        self.assertEqual(flashmap["num_steps"], 2)
        self.assertEqual(flashmap["hidden_size"], D)
        self.assertEqual(flashmap["jump_grids"]["2"][1], 0.8172569274902344)


if __name__ == "__main__":
    unittest.main()
