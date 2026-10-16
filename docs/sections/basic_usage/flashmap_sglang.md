# Serving FlashMap

SpecForge trains FlashMap drafts; the standalone
[flashmap](https://github.com/al-jshen/flashmap) package serves them. Its
`sglang` extra is an SGLang plugin that SGLang loads at startup: it adds the
FlashMap draft model to SGLang's DSPARK speculative worker without changing
SGLang, and serves a FlashMap release folder, local or on the Hugging Face Hub,
as it is. Install it in its own environment, since SpecForge pins another
SGLang version for training:

```bash
uv venv --python 3.12 .venv-sglang && source .venv-sglang/bin/activate
uv pip install "sglang==0.5.20" "flashmap[sglang]"

python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B
```

It needs `sglang==0.5.20` exactly and tensor parallelism 1. The block size
comes from the draft's `config.json`, and sampling at `temperature > 0` needs
no extra option. The package's
[Serve with SGLang](https://github.com/al-jshen/flashmap#serve-with-sglang)
section has the commands, and its `flashmap[vllm]` extra serves the same folders
in vLLM.

## Serving a SpecForge-trained draft

`specforge export --to hf` writes a trained FlashMap map in the FlashMap
release format, the layout of the published models, which the plugin serves
directly:

```bash
specforge export --to hf \
  --checkpoint ./outputs/qwen3-8b-flashmap-map-offline \
  --draft-config configs/qwen3-8b-flashmap.json \
  --draft-vocab FlashMap-Qwen3-8B/draft_vocab.safetensors \
  --output-dir ./exports/FlashMap-Qwen3-8B

python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path ./exports/FlashMap-Qwen3-8B
```

The folder holds:

- `config.json`: `model_type: flashmap` with the draft's fields, plus what the
  serving integrations read: `max_position_embeddings` and `dspark_config`
  (SGLang), and a speculators-style header and `draft_vocab_size` (vLLM);
- `model.safetensors`: the draft's own weights in bf16, without the target's
  embedding and LM head, which the draft borrows;
- with `--draft-vocab`, `draft_vocab.safetensors`: a ranking of the target's
  token ids by frequency (`ids`, most frequent first, and their `counts`), such
  as a published FlashMap folder's for the same target. The served draft
  predicts over its `--draft-vocab-size` most frequent tokens (65536 by
  default), which shrinks its output head; the target still verifies every
  token. Without a ranking the draft predicts over the full vocabulary;
- with the `flashmap` package installed in the exporting environment, the
  model code (`configuration_flashmap.py`, `modeling_flashmap.py`), so that
  `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` loads
  the folder without the package.

`--target-model` names the target the draft serves with when the draft config
has no `target_model`. A stage-1 flow exports the same way, with
`"flow_map": false` in its config; it loads back into SpecForge
(`model.flashmap_flow_checkpoint`) but is not served, since only the distilled
map drafts in one step.
