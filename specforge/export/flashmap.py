# coding=utf-8
"""Export a FlashMap draft for sglang (``specforge.serving.sglang.flashmap``).

The source is a FlashMap release folder (``config.json`` with ``model_type``
"flashmap" and ``model.safetensors`` with the drafter's own weights, without the
target's embedding and LM head) or a DeepSpec flow-map checkpoint, whose weights
are renamed to the release's names here. The output directory holds:

* ``config.json``: a Qwen3-shaped config so any sglang/transformers can parse
  it, with the fields sglang reads directly (the draft KV pool holds the DiT's
  cross-attention context, so heads, head_dim and layers are the DiT's; the
  DSPARK fields: block size, target layers, a rank-1 Markov head, which the
  draft replaces by an identity, and a mask token it never reads) and the
  release config under ``flashmap_config``;
* ``model.safetensors``: the release weights, plus ``draft_vocab_ids``, a
  ranking of every token id by how often the target produces it, when the
  draft predicts over a restricted vocabulary (``draft_vocab_size``, 65536 by
  default: 99.95% of Qwen3's response tokens, at no measured acceptance cost,
  while the draft's largest matmul shrinks by 2.3x). The ranking is the source
  folder's ``draft_vocab.safetensors`` (key ``ids``, as the releases ship it)
  unless ``--draft-vocab`` names another; a speculators-style ``d2t`` in the
  source restricts the vocabulary to its tokens.

    python -m specforge.export.flashmap --src FlashMap-Qwen3-8B --output-dir FlashMap-Qwen3-8B-sglang

``specforge export --to sglang`` routes FlashMap drafts here as well, from a
release-layout folder or a SpecForge training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from specforge.modeling.draft.flashmap import FlashMapConfig

ARCHITECTURE = "FlashMapDraftModel"
DEFAULT_DRAFT_VOCAB_SIZE = 65536
# The ranking a release folder ships (key "ids": every token id, most frequent first).
DRAFT_VOCAB_FILE = "draft_vocab.safetensors"
_LAYER_WEIGHTS = (
    tuple(
        f"{module}.{param}"
        for adaln in ("self_attn_adaln", "cross_attn_adaln", "mlp_adaln")
        for module, param in (
            (f"{adaln}.linear", "weight"),
            (f"{adaln}.linear", "bias"),
        )
    )
    + tuple(
        f"{attn}.{name}.weight"
        for attn in ("self_attn", "cross_attn")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    )
    + ("mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")
)
_GLOBAL_WEIGHTS = (
    "fc.weight", "hidden_norm.weight", "anchor_proj.weight", "anchor_proj.bias",
    "anchor_norm.weight", "state_norm.weight", "noise_norm.weight", "noise_gate",
    "x_embed.weight", "x_embed.bias", "time_embed.linear_1.weight",
    "time_embed.linear_1.bias", "time_embed.linear_2.weight",
    "time_embed.linear_2.bias", "jump_embed.linear_1.weight",
    "jump_embed.linear_1.bias", "jump_embed.linear_2.weight",
    "jump_embed.linear_2.bias", "final_adaln.linear.weight",
    "final_adaln.linear.bias", "head.weight", "head.bias",
)  # fmt: skip
_DEEPSPEC_MAPS = (
    "Qwen3DFlashierSemigroupFlowMapModel",
    "Qwen3DFlashierSolverFlowMapModel",
)


def _deepspec_name(name: str) -> Optional[str]:
    """The release name of a DeepSpec flow-map weight (None: dropped)."""
    plain = {
        "hidden_norm.scale": "hidden_norm.weight",
        "anchor_norm.scale": "anchor_norm.weight",
        "state_norm.scale": "state_norm.weight",
        "noise_norm.scale": "noise_norm.weight",
        "noise_gate": "noise_gate",
    }
    if name in plain:
        return plain[name]
    if name.startswith(("fc.", "anchor_proj.")):
        return name
    if not name.startswith("denoiser."):
        return None  # the target's embed_tokens and lm_head
    name = name.removeprefix("denoiser.")
    for old, new in (
        ("temb.mlp.0.", "time_embed.linear_1."),
        ("temb.mlp.2.", "time_embed.linear_2."),
        ("temb_jump.mlp.0.", "jump_embed.linear_1."),
        ("temb_jump.mlp.2.", "jump_embed.linear_2."),
        ("final_norm.cond_proj.1.", "final_adaln.linear."),
        ("x_embed.", "x_embed."),
        ("head.", "head."),
    ):
        if name.startswith(old):
            return new + name.removeprefix(old)
    _, layer, rest = name.split(".", 2)  # blocks.<i>.<rest>
    for old, new in (
        ("adaln_self.cond_proj.1.", "self_attn_adaln.linear."),
        ("adaln_cross.cond_proj.1.", "cross_attn_adaln.linear."),
        ("adaln_mlp.cond_proj.1.", "mlp_adaln.linear."),
        ("mlp.0.", "mlp.fc1."),
        ("mlp.3.", "mlp.fc2."),
        ("self_attn.wq.", "self_attn.q_proj."),
        ("self_attn.wo.", "self_attn.o_proj."),
        ("self_attn.qnorm.scale", "self_attn.q_norm.weight"),
        ("self_attn.knorm.scale", "self_attn.k_norm.weight"),
        ("cross_attn.wq.", "cross_attn.q_proj."),
        ("cross_attn.wo.", "cross_attn.o_proj."),
        ("cross_attn.qnorm.scale", "cross_attn.q_norm.weight"),
        ("cross_attn.knorm.scale", "cross_attn.k_norm.weight"),
    ):
        if rest.startswith(old):
            return f"layers.{layer}.{new}{rest.removeprefix(old)}"
    raise KeyError(f"no FlashMap name for DeepSpec weight {name!r}")


def from_deepspec(src: str):
    """A DeepSpec flow-map checkpoint as a release config and state dict.
    DeepSpec's RMSNorms scale by ``1 + scale``: the release stores that sum,
    in the weight's dtype, as DeepSpec evaluates it."""
    with open(os.path.join(src, "config.json")) as handle:
        source = json.load(handle)
    assert source["architectures"][0] in _DEEPSPEC_MAPS, source["architectures"]
    for key, value in (
        ("state_featurization", "posterior_noise"),
        ("noise_gate", "jump"),
        ("time_schedule", "decoding_error"),
    ):
        assert source.get(key) == value, (key, source.get(key))
    dim = int(source["dit_embed_dim"])
    config = dict(
        model_type="flashmap",
        architectures=[ARCHITECTURE],
        target_hidden_size=int(source["hidden_size"]),
        vocab_size=int(source["vocab_size"]),
        target_layer_ids=[int(i) for i in source["target_layer_ids"]],
        block_size=int(source["block_size"]),
        hidden_size=dim,
        num_hidden_layers=int(source["num_hidden_layers"]),
        num_attention_heads=int(source["dit_num_heads"]),
        intermediate_size=int(float(source.get("dit_mlp_ratio", 4.0)) * dim),
        cond_dim=int(source["dit_cond_dim"]),
        time_embed_dim=256,
        time_scale=float(source.get("dit_time_scale", 1000.0)),
        rope_theta=float(source.get("dit_rope_base", 10000.0)),
        rms_norm_eps=1e-6,
        qk_norm_eps=1e-5,
        layer_norm_eps=1e-6,
        noise_gate="jump",
        time_schedule="decoding_error",
        num_steps=1,
        jump_grids=FlashMapConfig(vocab_size=int(source["vocab_size"])).jump_grids,
        max_position_embeddings=int(source.get("max_position_embeddings", 40960)),
        num_target_layers=int(source.get("num_target_layers", 0)) or None,
    )
    state = {}
    for name, tensor in load_file(os.path.join(src, "model.safetensors")).items():
        if name.endswith(".wkv.weight"):  # keys and values, keys first
            stem = _deepspec_name(name.replace(".wkv.", ".wq.")).rsplit(".", 2)[0]
            keys, values = tensor.chunk(2, dim=0)
            state[f"{stem}.k_proj.weight"] = keys.contiguous()
            state[f"{stem}.v_proj.weight"] = values.contiguous()
            continue
        new = _deepspec_name(name)
        if new is not None:
            state[new] = (1 + tensor) if name.endswith(".scale") else tensor
    return config, state


def is_flashmap_folder(path: str) -> bool:
    """Whether ``path`` is a release-layout FlashMap folder."""
    try:
        with open(os.path.join(path, "config.json")) as handle:
            return json.load(handle).get("model_type") == "flashmap"
    except (OSError, ValueError):
        return False


def load_flashmap(src: str):
    """``(release config, state dict)`` of a release folder or DeepSpec map."""
    if not is_flashmap_folder(src):
        return from_deepspec(src)
    with open(os.path.join(src, "config.json")) as handle:
        config = json.load(handle)
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    for target_weight in ("embed_tokens.weight", "lm_head.weight"):
        state.pop(target_weight, None)
    return config, state


def check_weights(config: Dict, state: Dict) -> None:
    """Every weight the sglang draft loads, and nothing else: a key the loader
    did not expect would fail late, one it lacks would serve garbage."""
    expected = set(_GLOBAL_WEIGHTS) | {
        f"layers.{i}.{name}"
        for i in range(int(config["num_hidden_layers"]))
        for name in _LAYER_WEIGHTS
    }
    present = set(state) - {"draft_vocab_ids", "d2t", "t2d"}
    missing, unexpected = sorted(expected - present), sorted(present - expected)
    if missing or unexpected:
        raise ValueError(
            f"not a FlashMap state dict: missing {missing[:8]}, unexpected "
            f"{unexpected[:8]}"
        )


def mask_token_id(config: Dict) -> int:
    """A token id DSPARK may put in block slots the draft never reads: the
    config's ``mask_token_id``, else the first id past the target tokenizer's
    vocabulary (151669 for Qwen3), else the last embedding row."""
    if config.get("mask_token_id") is not None:
        return int(config["mask_token_id"])
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["target_model"])
        if len(tokenizer) < int(config["vocab_size"]):
            return len(tokenizer)
    except Exception:  # no target_model, or not loadable offline
        pass
    return int(config["vocab_size"]) - 1


def draft_vocab_ranking(src: str, state: Dict, path: Optional[str], vocab_size: int):
    """Token ids for the draft vocabulary, most frequent first: ``path``'s or
    the source folder's ranking of all ``vocab_size`` ids, a source ``d2t``'s
    tokens (``target_id = i + d2t[i]``), or None (no ranking found)."""
    d2t = state.pop("d2t", None)
    state.pop("t2d", None)
    if path is None and d2t is not None:
        return torch.arange(d2t.numel()) + d2t.to(torch.int64)
    if path is None and os.path.exists(os.path.join(src, DRAFT_VOCAB_FILE)):
        path = os.path.join(src, DRAFT_VOCAB_FILE)
    if path is None:
        return None
    ranking = load_file(path)["ids"].to(torch.int64)
    if ranking.numel() != vocab_size or not torch.equal(
        ranking.sort().values, torch.arange(vocab_size)
    ):
        raise ValueError(f"{path} is not a ranking of all {vocab_size} token ids")
    return ranking


def sglang_config(
    config: Dict, *, draft_vocab_size: int, num_steps: int, max_positions: int
) -> Dict:
    """The sglang draft config for a release ``config`` (see the module
    docstring)."""
    flashmap = {
        k: v for k, v in config.items() if k not in ("architectures", "auto_map")
    }
    flashmap["num_steps"] = int(num_steps)
    assert str(num_steps) in flashmap["jump_grids"], (num_steps, flashmap["jump_grids"])
    heads = int(config["num_attention_heads"])
    out = dict(
        architectures=[ARCHITECTURE],
        model_type="qwen3",
        dtype="bfloat16",
        hidden_size=int(config["target_hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=int(config["hidden_size"]) // heads,
        vocab_size=int(config["vocab_size"]),
        max_position_embeddings=int(max_positions),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        tie_word_embeddings=False,
        block_size=int(config["block_size"]),
        target_layer_ids=[int(i) for i in config["target_layer_ids"]],
        mask_token_id=mask_token_id(config),
        markov_rank=1,
        markov_head_type="vanilla",
        sample_from_anchor=True,
        enable_confidence_head=False,
        draft_vocab_size=int(draft_vocab_size),
        flashmap_config=flashmap,
    )
    if config.get("num_target_layers"):
        out["num_target_layers"] = int(config["num_target_layers"])
    return out


def export_flashmap_to_sglang(
    src: str,
    output_dir: str,
    *,
    draft_vocab: Optional[str] = None,
    draft_vocab_size: int = DEFAULT_DRAFT_VOCAB_SIZE,
    num_steps: int = 1,
    max_positions: Optional[int] = None,
) -> str:
    """Write an sglang FlashMap draft directory; returns ``output_dir``.

    ``draft_vocab`` is a safetensors file whose ``ids`` rank every token id,
    most frequent first (default: the source folder's ``draft_vocab.safetensors``);
    the draft predicts over the top ``draft_vocab_size`` (0: every token, as
    also without any ranking)."""
    config, state = load_flashmap(src)
    vocab_size = int(config["vocab_size"])
    ranking = draft_vocab_ranking(src, state, draft_vocab, vocab_size)
    check_weights(config, state)
    if ranking is None:
        draft_vocab_size = 0
    if draft_vocab_size:
        draft_vocab_size = min(draft_vocab_size, ranking.numel())
        state["draft_vocab_ids"] = ranking
    if max_positions is None:
        max_positions = int(config.get("max_position_embeddings") or 40960)
    out = sglang_config(
        config,
        draft_vocab_size=draft_vocab_size,
        num_steps=num_steps,
        max_positions=max_positions,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(out, handle, indent=2)
    tensors = {name: tensor.contiguous() for name, tensor in state.items()}
    save_file(
        tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--src", required=True, help="FlashMap release folder (or DeepSpec map)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--draft-vocab",
        help="safetensors file ranking every token id (key 'ids'); default: "
        f"<src>/{DRAFT_VOCAB_FILE} if present",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=DEFAULT_DRAFT_VOCAB_SIZE,
        help="draft over the ranking's top k tokens; 0: the full vocabulary",
    )
    parser.add_argument(
        "--num-steps", type=int, default=1, help="default jumps per draft"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="longest sequence served (default: the source's, else 40960)",
    )
    args = parser.parse_args(argv)
    out = export_flashmap_to_sglang(
        args.src,
        args.output_dir,
        draft_vocab=args.draft_vocab,
        draft_vocab_size=args.draft_vocab_size,
        num_steps=args.num_steps,
        max_positions=args.max_positions,
    )
    print(f"exported sglang FlashMap draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
