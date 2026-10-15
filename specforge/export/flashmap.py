# coding=utf-8
"""Write a trained FlashMap draft in the FlashMap release format.

``specforge export --to hf`` routes FlashMap drafts here. The output folder is
what the ``flashmap`` package (https://github.com/al-jshen/flashmap) serves in
SGLang (``flashmap[sglang]``) and vLLM (``flashmap[vllm]``) and loads with
Hugging Face ``transformers``, laid out as the published FlashMap models are:

* ``config.json``: ``model_type: "flashmap"`` with the drafter's fields, the
  speculators-style header (``speculators_model_type``, ``speculators_config``,
  an empty ``transformer_layer_config``) and ``draft_vocab_size`` that vLLM
  reads, and ``max_position_embeddings`` and ``dspark_config`` that SGLang reads;
* ``model.safetensors``: the draft's own weights under the release names (the
  training model already uses them), without the target's embedding and LM
  head, in bf16;
* with ``draft_vocab``, ``draft_vocab.safetensors``: the target's tokens ranked
  by frequency (``ids``, most frequent first, and their ``counts``); the serving
  integrations draft over its first ``draft_vocab_size`` tokens (65,536 by
  default), and the target still verifies every token;
* with the ``flashmap`` package installed, its model code
  (``configuration_flashmap.py``, ``modeling_flashmap.py`` and ``auto_map``), so
  that ``AutoModel.from_pretrained(..., trust_remote_code=True)`` works without
  the package.

A stage-1 flow exports the same way with ``"flow_map": false`` in its config; it
loads back into SpecForge (``model.flashmap_flow_checkpoint``) but is not served,
since only the distilled map drafts in one step.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
from typing import Dict, List, Optional

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

ARCHITECTURE = "FlashMapDraftModel"
# Draft vocabulary of the serving integrations when a ranking is shipped.
DEFAULT_DRAFT_VOCAB_SIZE = 65536
# Positions a served draft accepts: Qwen3's context with YaRN (and vLLM's default).
MAX_POSITION_EMBEDDINGS = 131072
DRAFT_VOCAB_FILE = "draft_vocab.safetensors"
# The flashmap package's model code, which a release folder ships for
# trust_remote_code.
REMOTE_CODE = ("configuration_flashmap.py", "modeling_flashmap.py")
# The drafter's own fields, as the release config.json names them.
_DRAFT_FIELDS = (
    "target_hidden_size",
    "vocab_size",
    "target_layer_ids",
    "block_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "cond_dim",
    "time_embed_dim",
    "time_scale",
    "rope_theta",
    "rms_norm_eps",
    "qk_norm_eps",
    "layer_norm_eps",
    "noise_gate",
    "time_schedule",
    "num_steps",
    "jump_grids",
)
_LAYER_WEIGHTS = (
    tuple(
        f"{adaln}.linear.{param}"
        for adaln in ("self_attn_adaln", "cross_attn_adaln", "mlp_adaln")
        for param in ("weight", "bias")
    )
    + tuple(
        f"{attn}.{name}.weight"
        for attn in ("self_attn", "cross_attn")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    )
    + ("mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")
)
_FLOW_WEIGHTS = (
    "fc.weight", "hidden_norm.weight", "anchor_proj.weight", "anchor_proj.bias",
    "anchor_norm.weight", "state_norm.weight", "x_embed.weight", "x_embed.bias",
    "time_embed.linear_1.weight", "time_embed.linear_1.bias",
    "time_embed.linear_2.weight", "time_embed.linear_2.bias",
    "final_adaln.linear.weight", "final_adaln.linear.bias", "head.weight", "head.bias",
)  # fmt: skip
# What only the map has: its end-time embedding and noise read-out.
_MAP_WEIGHTS = (
    "noise_norm.weight", "noise_gate", "jump_embed.linear_1.weight",
    "jump_embed.linear_1.bias", "jump_embed.linear_2.weight", "jump_embed.linear_2.bias",
)  # fmt: skip


def is_flashmap_config(draft_config_path: str) -> bool:
    """Whether the draft config at ``draft_config_path`` is a FlashMap draft's."""
    try:
        with open(draft_config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return False
    return (
        config.get("model_type") == "flashmap"
        or (config.get("architectures") or [None])[0] == ARCHITECTURE
    )


def release_weight_names(num_layers: int, flow_map: bool = True) -> set:
    """The names of a FlashMap draft's weights (a map's, or a stage-1 flow's)."""
    names = set(_FLOW_WEIGHTS) | {
        f"layers.{i}.{name}" for i in range(num_layers) for name in _LAYER_WEIGHTS
    }
    return names | set(_MAP_WEIGHTS) if flow_map else names


def mask_token_id(target_model: str, vocab_size: int) -> int:
    """A token id outside the target tokenizer's vocabulary (151669 for Qwen3):
    the first past it, or the last embedding row if the tokenizer is not
    reachable. DSPARK fills a draft block's slots with it; FlashMap never reads it."""
    try:
        tokens = len(transformers.AutoTokenizer.from_pretrained(target_model))
    except Exception:  # noqa: BLE001 (no tokenizer, or not reachable offline)
        return vocab_size - 1
    return tokens if tokens < vocab_size else vocab_size - 1


def target_architectures(target_model: str) -> List[str]:
    """The target's architectures from its config.json, or [] if it is not
    reachable (vLLM reads only the target's name)."""
    try:
        config, _ = transformers.PretrainedConfig.get_config_dict(target_model)
    except Exception:  # noqa: BLE001
        return []
    return list(config.get("architectures") or [])


def token_ranking(path: str, vocab_size: int) -> Dict[str, torch.Tensor]:
    """``ids`` (every token id, most frequent first) and their ``counts`` from a
    ranking file (safetensors with ``ids`` and, optionally, ``counts``), such as
    a released FlashMap folder's ``draft_vocab.safetensors`` for the same target."""
    ranking = load_file(path)
    ids = ranking["ids"].to(torch.int64)
    if ids.numel() != vocab_size or not torch.equal(
        ids.sort().values, torch.arange(vocab_size)
    ):
        raise ValueError(f"{path} is not a ranking of all {vocab_size} token ids")
    counts = ranking.get("counts")
    counts = torch.zeros_like(ids) if counts is None else counts.to(torch.int64)
    return {"ids": ids, "counts": counts}


def release_config(
    draft_config,
    target_model: str,
    *,
    draft_vocab_size: Optional[int] = None,
) -> dict:
    """The release config.json of a FlashMap draft config."""
    block_size = int(draft_config.block_size)
    vocab_size = int(draft_config.vocab_size)
    config = {key: getattr(draft_config, key) for key in _DRAFT_FIELDS}
    config.update(
        model_type="flashmap",
        architectures=[ARCHITECTURE],
        target_model=target_model,
        target_layer_ids=[int(i) for i in draft_config.target_layer_ids],
        jump_grids={str(k): list(v) for k, v in draft_config.jump_grids.items()},
        dtype="bfloat16",
        tie_word_embeddings=False,
        transformers_version=transformers.__version__,
        # What vLLM reads (with flashmap[vllm]): a speculators-style header.
        speculators_model_type="flashmap",
        speculators_config={
            "algorithm": "flashmap",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": block_size,
                    "accept_tolerance": 0.0,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "name_or_path": target_model,
                "architectures": target_architectures(target_model),
            },
        },
        transformer_layer_config={},
        # What SGLang's DSPARK worker reads (with flashmap[sglang]): the longest
        # context it may serve, and DSPARK's own fields: a rank-1 "vanilla"
        # Markov head, which flashmap replaces by an identity (FlashMap has
        # none), and the token DSPARK fills the block's slots with.
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        dspark_config={
            "markov_rank": 1,
            "markov_head_type": "vanilla",
            "mask_token_id": mask_token_id(target_model, vocab_size),
        },
    )
    if draft_vocab_size:
        config["draft_vocab_size"] = int(draft_vocab_size)
    return config


def export_flashmap_release(
    model,
    output_dir: str,
    *,
    target_model: Optional[str] = None,
    draft_vocab: Optional[str] = None,
    draft_vocab_size: Optional[int] = None,
) -> str:
    """Write ``model`` (a materialized ``FlashMapDraftModel``) as a FlashMap
    release folder; returns ``output_dir``.

    ``target_model`` (default: the draft config's ``target_model``) is the
    target the draft serves with. ``draft_vocab`` is a token ranking shipped as
    ``draft_vocab.safetensors``; the serving integrations draft over its first
    ``draft_vocab_size`` tokens (default 65,536). Without one they draft over
    the full vocabulary."""
    config = model.config
    target_model = target_model or getattr(config, "target_model", None)
    if not target_model:
        raise ValueError(
            "the FlashMap export needs the target model: set target_model in the "
            "draft config or pass --target-model"
        )
    flow_map = bool(getattr(model, "flow_map", True))
    weights = {
        name: tensor.detach().to("cpu", torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    expected = release_weight_names(int(config.num_hidden_layers), flow_map)
    if set(weights) != expected:
        raise ValueError(
            "the draft's weights differ from the FlashMap layout: missing "
            f"{sorted(expected - set(weights))[:8]}, unexpected "
            f"{sorted(set(weights) - expected)[:8]}"
        )
    vocab_size = int(config.vocab_size)
    ranking = token_ranking(draft_vocab, vocab_size) if draft_vocab else None
    if ranking is None and draft_vocab_size:
        raise ValueError("--draft-vocab-size needs a --draft-vocab ranking")
    if ranking is not None:
        draft_vocab_size = draft_vocab_size or DEFAULT_DRAFT_VOCAB_SIZE
        if draft_vocab_size >= vocab_size:
            draft_vocab_size = None  # the full vocabulary
    release = release_config(config, target_model, draft_vocab_size=draft_vocab_size)
    if not flow_map:
        release["flow_map"] = False
        logger.warning(
            "exporting a stage-1 FlashMap flow: it loads back into SpecForge "
            "(model.flashmap_flow_checkpoint) but only the distilled map is served"
        )

    os.makedirs(output_dir, exist_ok=True)
    code = importlib.util.find_spec("flashmap")
    if code is not None and code.origin is not None:
        for name in REMOTE_CODE:
            shutil.copy(
                os.path.join(os.path.dirname(code.origin), name),
                os.path.join(output_dir, name),
            )
        release["auto_map"] = {
            "AutoConfig": "configuration_flashmap.FlashMapConfig",
            "AutoModel": "modeling_flashmap.FlashMapDraftModel",
        }
    else:
        logger.info(
            "the flashmap package is not installed, so the folder ships without its "
            "model code: transformers loads it after `import flashmap`"
        )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(release, indent=2, sort_keys=True) + "\n")
    path = os.path.join(output_dir, "model.safetensors")
    save_file(weights, path, metadata={"format": "pt"})
    if ranking is not None:
        save_file(
            ranking,
            os.path.join(output_dir, DRAFT_VOCAB_FILE),
            metadata={
                "description": "target response-token frequency ranking: ids = "
                "token ids, most frequent first; counts[i] = occurrences of ids[i]",
                "target": target_model,
            },
        )
    with safe_open(path, "pt") as handle:
        for name in handle.keys():
            if not torch.equal(handle.get_tensor(name), weights[name]):
                raise ValueError(f"{name} was not saved as exported")
    return output_dir


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_DRAFT_VOCAB_SIZE",
    "export_flashmap_release",
    "is_flashmap_config",
    "release_config",
    "release_weight_names",
    "token_ranking",
]
