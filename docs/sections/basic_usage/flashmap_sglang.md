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

> **Inference only?** The standalone `flashmap` package carries the same
> integration as an SGLang plugin and serves a FlashMap release folder as it
> is, with no export step and no SpecForge install:
>
> ```bash
> uv pip install "sglang==0.5.20" "flashmap[sglang] @ git+https://github.com/al-jshen/flashmap"
> python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
>   --speculative-algorithm DSPARK --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B
> ```
>
> See [Serving with SGLang](https://github.com/al-jshen/flashmap/blob/master/docs/details.md#serving-with-sglang).
> This fork keeps its copy for training FlashMap, serving SpecForge exports and
> upstreaming the integration to SGLang.

## Requirements

- `sglang==0.5.20` exactly: the draft plugs into internals of its DSPARK V2
  worker, and it uses its `sgl_kernel` (FlashAttention with paged KV) and
  Triton. The package checks the version when sglang loads it and stops the
  server with the reason on a mismatch (`SPECFORGE_SGLANG_ANY_VERSION=1`
  skips the check). SpecForge itself pins sglang 0.5.18 for training, so
  install it into the serving environment without its dependencies:

  ```bash
  pip install "sglang==0.5.20"
  pip install --no-deps -e /path/to/SpecForge   # or PYTHONPATH=/path/to/SpecForge
  ```

- Tensor parallelism 1 (the draft is not sharded), a bf16 or e4m3 draft KV pool
  (`--speculative-draft-kv-cache-dtype fp8_e4m3`, independent of the target's
  KV dtype; other quantized KV dtypes are rejected), a target without
  vocabulary padding or muP logit scaling. Each is checked at startup.

## 1. Export

```bash
python -m specforge.export.flashmap --src FlashMap-Qwen3-8B --output-dir FlashMap-Qwen3-8B-sglang
# equivalently, also from a SpecForge FlashMap training checkpoint:
specforge export --to sglang --checkpoint <release folder or checkpoint> \
  --draft-config configs/qwen3-8b-flashmap.json --output-dir FlashMap-Qwen3-8B-sglang
```

`--src` is a FlashMap release folder (`config.json` with `model_type: flashmap`
and the drafter's weights, without the target's embedding and LM head); a
DeepSpec flow-map checkpoint directory also works. The draft predicts over the
target's `--draft-vocab-size` most frequent tokens (65536 by default), which
shrinks the draft's largest matmul; the target still verifies every token. The ranking comes from the release folder's
`draft_vocab.safetensors` (its `ids` tensor ranks every token id, most frequent
first), from `--draft-vocab <file>` (`--vocab-mapping` for `specforge export`),
or from a speculators-style `d2t` in the source; `--draft-vocab-size 0` keeps
the full vocabulary. `--num-steps 2` makes two jumps the default.

## 2. Serve

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang \
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
| `FLASHMAP_COMPILE` / `FLASHMAP_COMPILE_MODE` | 1 / default | torch.compile of the draft step; only modes without their own CUDA graphs (`default`, `max-autotune-no-cudagraphs`), since sglang captures the step |
| `FLASHMAP_FOLDED_NOISE` | 0 | opt-in (lossless): draw one jump's noise from the eigendecomposition of `E^T E`, folding its RMSNorm and `x_embed` into one GEMM |
| `FLASHMAP_FP8` | 0 | opt-in (Ada/Hopper): e4m3 copies of the draft's large GEMM weights (DiT, `x_embed`, folded head, `fc`, context and anchor projections), per output channel, with activations quantized per token |
| `FLASHMAP_FUSED_SAMPLER` | 0 | opt-in (lossless): one sampling kernel (argmax for greedy rows, Gumbel-max with in-kernel Philox otherwise), no noise buffer, full-vocabulary logits scattered only for sampling batches |
| `FLASHMAP_TREE_N` / `FLASHMAP_TREE_BUDGET` | 0 / 32 | tree drafting (below) |

`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (an sglang option) overlaps DSPARK's
verify planning with the previous step; it is optional, and the benchmark
script sets it for every speculative server.

`--speculative-dspark-block-size` is the number of drafts verified per request
(16, the whole block). At high concurrency the target's verify pass becomes
compute-bound and the last drafts of a block are rarely accepted, so a smaller
value (such as 15, DFlash's verify width) can raise throughput there; the map
still drafts its whole block, as it was trained, and only the first drafts are
verified.

An FP8 draft KV pool (`--speculative-draft-kv-cache-dtype fp8_e4m3`) stores the
cross-attention context in e4m3 (written by a Triton port of the fused write,
read by FA3 with the query quantized per request and head), which leaves more
memory for the target's KV cache.

### Tree drafting (opt-in)

`FLASHMAP_TREE_N=<N>` drafts up to N candidate blocks per request from
independent noise in one draft pass (their queries side by side in the
cross-attention, so the context is read once), merges them into a trie inside
the draft graph and keeps the 16 nodes the most candidates share; the target
verifies that tree (17 tokens with the root, DSPARK's own width) and the commit
reorders the verify window's `req_to_token` entries instead of moving KV. A batch
of `bs` requests drafts `clamp(FLASHMAP_TREE_BUDGET // bs, 1, N)` candidates each
(budget 32). `FLASHMAP_SPLITKV_VERIFY=1` with `--attention-backend triton` runs the
target's verify attention with a split-KV Triton kernel that applies the tree
mask only to each request's 17 x 17 block.

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang SGLANG_DSPARK_FOLDED_PROPOSAL=0 \
FLASHMAP_TREE_N=8 FLASHMAP_SPLITKV_VERIFY=1 \
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --attention-backend triton --page-size 1 \
  --speculative-algorithm DSPARK --speculative-draft-model-path FlashMap-Qwen3-8B-sglang \
  --speculative-dspark-block-size 16
```

Constraints: greedy requests only. The scheduler refuses a request that samples
(temperature > 0), asks for logprobs or constrains its output (JSON schema,
regex, EBNF, structural tag) when it arrives, answering it with an error that
says why, and keeps serving. Also `--page-size 1`,
`SGLANG_DSPARK_FOLDED_PROPOSAL=0`, a target attention backend that reads a tree
mask (triton, flashinfer or fa3), one jump and the full verify width. It pays
off at low concurrency.

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
  `L L^T = E^T E` (a hidden-size draw, not a vocabulary-size one). With
  `FLASHMAP_FOLDED_NOISE=1`, `E^T E = Q diag(lam) Q^T` instead, so the draw's
  RMS is `sum(lam z^2)` and its RMSNorm and `x_embed` fold into one matrix.
- **Folded, restricted head.** The DiT's output head and the target's LM head
  are folded into one matrix over the draft vocabulary; drafted indices map
  back to token ids inside the graph.
- **One-pass sampling.** FlashMap has no Markov head; it stands in an identity
  for DSPARK's and samples (argmax or Gumbel) the whole block at once instead
  of position by position (`FLASHMAP_FUSED_SAMPLER=1`: in one kernel).
- **Banked multi-jump.** Later jumps need the vocabulary-sized state's
  embedding. A bank of precomputed noise draws carries everything about the
  noise over the whole vocabulary, so a jump reads only the embedding rows of
  the previous prediction's top tokens.

## Checks and benchmarks

```bash
# the served draft against the training model's forward
python scripts/gates/check_flashmap_sglang.py --draft FlashMap-Qwen3-8B-sglang --target Qwen/Qwen3-8B

# throughput/latency vs the target alone and DFlash (see the script header)
TARGET=Qwen/Qwen3-8B PROMPTS=bench_prompts OUT=logs/fm8b OUTPUT_DETAILS=1 \
SERVERS="target=target fm1=flashmap:FlashMap-Qwen3-8B-sglang fm2=flashmap:FlashMap-Qwen3-8B-sglang:FLASHMAP_NUM_STEPS=2 dflash=dflash:z-lab/Qwen3-8B-DFlash-b16" \
  benchmarks/flashmap/bench_sglang.sh
python benchmarks/flashmap/compare_outputs.py logs/fm8b   # greedy outputs vs the target's
```

## Losslessness

Greedy outputs are the target's own, up to bf16 rounding: speculative
verification runs the target on 17-token windows and decoding alone on one
token, and their kernels round differently, so a near-tied argmax can flip.
`benchmarks/flashmap/check_greedy.py` teacher-forces a server's greedy outputs
through the target and reports how far each token is from its argmax. Sampling
requests are verified by DSPARK's rejection sampling against the draft's
logits.

## SGLang internals touched

Importing the package installs two hooks on `DSparkWorkerV2` (once, however
often it is imported), both active only for FlashMap drafts: `init_cuda_graphs`
turns off the draft attention backend's per-step metadata (the draft never uses
that backend), and `_maybe_build_draft_sampler` replaces DSPARK's graph-folded
sampler with the one-pass sampler above, which takes over its buffers. Tree
drafting (`FLASHMAP_TREE_N`) also replaces `DSparkWorkerV2._forward_decode` with
the tree step, has the target's FA3, FlashInfer and Triton backends capture tree
masks, and wraps `Scheduler._add_request_to_queue` to refuse what the tree step
cannot serve. No SGLang source file is patched.
