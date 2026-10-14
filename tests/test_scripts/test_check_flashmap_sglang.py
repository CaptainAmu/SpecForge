"""check_flashmap_sglang.py's target LM head loading, including targets with
tied embeddings (CPU only)."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gates" / "check_flashmap_sglang.py"
SPEC = importlib.util.spec_from_file_location("check_flashmap_sglang", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

EMBED = torch.arange(12, dtype=torch.float32).view(4, 3)
HEAD = -EMBED


def _target(path, *, tied, head, sharded):
    os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump({"tie_word_embeddings": tied}, handle)
    tensors = {"model.embed_tokens.weight": EMBED}
    if head:
        tensors["lm_head.weight"] = HEAD
    if not sharded:
        save_file(tensors, os.path.join(path, "model.safetensors"))
        return path
    weight_map = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        shard = f"model-0000{i + 1}.safetensors"
        save_file({name: tensor}, os.path.join(path, shard))
        weight_map[name] = shard
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as handle:
        json.dump({"weight_map": weight_map}, handle)
    return path


class TestTargetVocabWeights(unittest.TestCase):
    def _load(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = _target(os.path.join(tmp, "target"), **kwargs)
            return gate.target_vocab_weights(path, device="cpu")

    def test_untied_sharded_target_has_its_own_head(self):
        embed, head = self._load(tied=False, head=True, sharded=True)
        self.assertTrue(torch.equal(embed, EMBED))
        self.assertTrue(torch.equal(head, HEAD))

    def test_tied_target_without_head_uses_embedding(self):
        for sharded in (False, True):
            embed, head = self._load(tied=True, head=False, sharded=sharded)
            self.assertTrue(torch.equal(head, EMBED))

    def test_tied_target_ignores_a_stored_head(self):
        _, head = self._load(tied=True, head=True, sharded=False)
        self.assertTrue(torch.equal(head, EMBED))


if __name__ == "__main__":
    unittest.main()
