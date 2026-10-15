# coding=utf-8
"""FlashMap through the public CLI on CPU: flow stage, map stage distilled from
the flow's run directory, and HF export of both (the map in the FlashMap
release format).

Each leg is a ``specforge`` process, so the test takes minutes; it runs when
``SPECFORGE_RUN_CLI_TESTS=1``."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB = 500
TARGET_HIDDEN = 32


def _write_fixtures(root: str) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    target = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=TARGET_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            tie_word_embeddings=False,
        )
    )
    target.save_pretrained(os.path.join(root, "target"))
    draft = {
        "architectures": ["FlashMapDraftModel"],
        "model_type": "flashmap",
        "target_hidden_size": TARGET_HIDDEN,
        "vocab_size": VOCAB,
        "target_layer_ids": [0, 1],
        "block_size": 4,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "cond_dim": 16,
        "time_embed_dim": 16,
    }
    with open(os.path.join(root, "draft.json"), "w") as handle:
        json.dump(draft, handle)
    features = os.path.join(root, "features", "rows_0-2000")
    os.makedirs(features)
    for index in range(16):
        length = 24 + index % 8
        loss_mask = torch.ones(length, dtype=torch.long)
        loss_mask[:6] = 0
        torch.save(
            {
                "input_ids": torch.randint(0, VOCAB, (length,)),
                "loss_mask": loss_mask,
                "hidden_states": torch.randn(
                    1, length, 2 * TARGET_HIDDEN, dtype=torch.bfloat16
                ),
            },
            os.path.join(features, f"data_{index}.ckpt"),
        )


def _config(root: str, stage: str) -> str:
    lines = [
        "model:",
        f"  target_model_path: {root}/target",
        f"  draft_model_config: {root}/draft.json",
    ]
    if stage == "map":
        lines.append(f"  flashmap_flow_checkpoint: {root}/out/flashmap-flow")
    lines += [
        "data:",
        f"  hidden_states_path: {root}/features",
        "  max_length: 32",
        "  dataloader_num_workers: 0",
        "training:",
        "  strategy: flashmap",
        f"  flashmap_stage: {stage}",
        "  max_steps: 4",
        "  batch_size: 2",
        "  fsdp_sharding: NO_SHARD",
        "  learning_rate: 1.0e-3",
        "  warmup_ratio: 0.0",
        "  attention_backend: sdpa",
        "  num_anchors: 6",
        "  loss_decay_gamma: 7.0",
        "  save_interval: 2",
        "  log_interval: 2",
        f"run_id: flashmap-{stage}",
        f"output_dir: {root}/out/flashmap-{stage}",
    ]
    path = os.path.join(root, f"{stage}.yaml")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _specforge(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "specforge.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_CLI_TESTS") == "1",
    "set SPECFORGE_RUN_CLI_TESTS=1 to run the CLI pipeline",
)
class FlashMapCliPipelineTest(unittest.TestCase):
    def test_flow_then_map_then_export(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fixtures(root)
            for stage in ("flow", "map"):
                result = _specforge("train", "--config", _config(root, stage))
                self.assertEqual(0, result.returncode, result.stderr[-4000:])
                self.assertIn("step 4:", result.stdout)

            export = os.path.join(root, "export")
            result = _specforge(
                "export",
                "--to",
                "hf",
                "--checkpoint",
                f"{root}/out/flashmap-map",
                "--draft-config",
                f"{root}/draft.json",
                "--output-dir",
                export,
                "--target-model",
                f"{root}/target",
            )
            self.assertEqual(0, result.returncode, result.stderr[-4000:])
            with open(os.path.join(export, "config.json")) as handle:
                config = json.load(handle)
            self.assertEqual("flashmap", config["model_type"])
            self.assertNotIn("flow_map", config)  # a release config: a map
            verifier = config["speculators_config"]["verifier"]
            self.assertEqual(["Qwen3ForCausalLM"], verifier["architectures"])
            self.assertLess(config["dspark_config"]["mask_token_id"], VOCAB)
            weights = load_file(os.path.join(export, "model.safetensors"))
            self.assertIn("jump_embed.linear_2.weight", weights)
            self.assertIn("layers.1.cross_attn.k_proj.weight", weights)
            self.assertFalse(
                any("lm_head" in key or "embed_tokens" in key for key in weights)
            )

            flow_export = os.path.join(root, "flow-export")
            result = _specforge(
                "export",
                "--to",
                "hf",
                "--checkpoint",
                f"{root}/out/flashmap-flow",
                "--draft-config",
                f"{root}/draft.json",
                "--output-dir",
                flow_export,
                "--target-model",
                f"{root}/target",
            )
            self.assertEqual(0, result.returncode, result.stderr[-4000:])
            with open(os.path.join(flow_export, "config.json")) as handle:
                self.assertFalse(json.load(handle)["flow_map"])


if __name__ == "__main__":
    unittest.main()
