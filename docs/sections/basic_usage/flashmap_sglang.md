# Serving FlashMap with SGLang

FlashMap is a block-parallel speculative drafter: like DFlash it drafts a whole
block of 16 tokens after the last verified one, conditioned on the target's
hidden states, but its block is drawn from a distilled discrete flow map, one
network evaluation from Gaussian noise (one *jump*). Two jumps are more
accurate and somewhat slower.

SGLang has no built-in FlashMap model. SpecForge provides one,
`specforge.serving.sglang.flashmap.FlashMapDraftModel`, that runs under SGLang's
DSPARK speculative worker (which handles block drafting, KV injection of the
target's hidden states, verification and CUDA graphs). It is loaded through
SGLang's external model package hook, so no SGLang source change is needed.

## Requirements

- `sglang==0.5.20` (the DSPARK V2 worker protocol FlashMap plugs into), with its
  `sgl_kernel` (FlashAttention with paged KV) and Triton.
- A single GPU per draft (tensor parallelism is not supported by the draft).

## 1. Export

```bash
python -m specforge.export.flashmap \
  --src FlashMap-Qwen3-8B \
  --output-dir FlashMap-Qwen3-8B-sglang \
  --draft-vocab qwen3_8b_token_ranking.safetensors
```

`--src` is a FlashMap release folder (`config.json` with `model_type: flashmap`
and the drafter's weights, without the target's embedding and LM head); a
DeepSpec flow-map checkpoint directory also works. `--draft-vocab` is a
safetensors file whose `ids` tensor ranks every token id by how often the
target produces it; the draft then predicts over the top
`--draft-vocab-size` tokens (65536 by default, 99.95% of Qwen3's response
tokens), which shrinks the draft's largest matmul 2.3x at no measurable
acceptance cost. Without it the draft keeps the full vocabulary.
`--num-steps 2` makes two jumps the default.

## 2. Serve

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang \
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path FlashMap-Qwen3-8B-sglang \
  --speculative-dspark-block-size 16 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32
```

Environment knobs, read when the server starts:

| Variable | Default | Effect |
|---|---|---|
| `FLASHMAP_NUM_STEPS` | the export's `num_steps` (1) | jumps per draft |
| `FLASHMAP_DRAFT_VOCAB` | the export's `draft_vocab_size` | `0`: full vocabulary; `k`: the export ranking's top k |
| `FLASHMAP_JUMP_BANK` / `FLASHMAP_JUMP_TOPK` | 8192 / 16 | multi-jump noise bank size / tokens kept per intermediate prediction |
| `FLASHMAP_COMPILE` / `FLASHMAP_COMPILE_MODE` | 1 / default | torch.compile of the draft step |

Greedy (temperature 0) requests are lossless: every committed token is the
target's own greedy choice. Sampling requests are verified by DSPARK's
rejection sampling against the draft's logits.

## How the draft is made fast

- **One pass per block, in a CUDA graph.** All requests' blocks are drafted in
  one batched pass, compiled with torch.compile, which DSPARK captures in its
  draft CUDA graphs; drafting never synchronizes with the host.
- **Fused context write.** The target hidden states of every committed token
  are projected to all DiT layers' cross-attention keys/values in one matmul
  and written (k-normed, rotated, uncommitted verify columns skipped) by
  SGLang's fused DSpark KV kernel into the draft KV pool.
- **Paged cross-attention.** Each DiT layer's cross-attention reads the paged
  context directly with FlashAttention (SGLang's `req_to_token` rows as page
  tables), with one tile schedule for all layers.
- **Triton block self-attention.** Q/K norms, RoPE and the 16x16 attention of
  each block run in one kernel; Q, K and V are one matmul.
- **Constants folded.** Every AdaLN modulation depends only on the jump times
  and is computed at load. At `t = 0` the state's posterior is uniform whatever
  the noise, so its embedding is a constant too.
- **No vocabulary-sized noise.** The noise enters only through its embedding
  `x @ E ~ N(0, E^T E)`, drawn as `z @ L^T` with the Cholesky factor
  `L L^T = E^T E` (a hidden-size draw, not a vocabulary-size one).
- **Folded, restricted head.** The DiT's output head and the target's LM head
  are folded into one matrix over the draft vocabulary; drafted indices map
  back to token ids inside the graph.
- **One-pass sampling.** FlashMap has no Markov head; it stands in an identity
  for DSPARK's and samples (argmax or Gumbel) the whole block at once instead
  of position by position.
- **Banked multi-jump.** Later jumps need the vocabulary-sized state's
  embedding. A bank of precomputed noise draws carries everything about the
  noise over the whole vocabulary, so a jump reads only the embedding rows of
  the previous prediction's top tokens.

## Checks and benchmarks

```bash
# the served draft against a direct PyTorch implementation of the map
python scripts/gates/check_flashmap_sglang.py --draft FlashMap-Qwen3-8B-sglang --target Qwen/Qwen3-8B

# throughput/latency vs the target alone and DFlash (see the script header)
TARGET=Qwen/Qwen3-8B PROMPTS=bench_prompts OUT=logs/fm8b OUTPUT_DETAILS=1 \
SERVERS="target=target fm1=flashmap:FlashMap-Qwen3-8B-sglang fm2=flashmap:FlashMap-Qwen3-8B-sglang:FLASHMAP_NUM_STEPS=2 dflash=dflash:z-lab/Qwen3-8B-DFlash-b16" \
  benchmarks/flashmap/bench_sglang.sh
python benchmarks/flashmap/compare_outputs.py logs/fm8b   # greedy outputs vs the target's
```

## SGLang internals touched

Importing the package installs two hooks on `DSparkWorkerV2`, both active
only for FlashMap drafts: `init_cuda_graphs` turns off the draft attention
backend's per-step metadata (the draft never uses that backend), and
`_maybe_build_draft_sampler` wraps DSPARK's graph-folded sampler with the
one-pass sampler above.
