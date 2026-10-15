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
target's `--draft-vocab-size` most frequent tokens (65536 by default, 99.95% of
Qwen3's response tokens), which shrinks the draft's largest matmul 2.3x at no
measurable acceptance cost. The ranking comes from the release folder's
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
| `FLASHMAP_FOLDED_NOISE` | 0 | opt-in (lossless, measured on Qwen3-8B only): draw one jump's noise from the eigendecomposition of `E^T E`, folding its RMSNorm and `x_embed` into one GEMM (saves a 4096x4096 GEMM per step on Qwen3-8B) |
| `FLASHMAP_FP8` | 0 | opt-in (Ada/Hopper): e4m3 copies of the draft's large GEMM weights (DiT, `x_embed`, folded head, `fc`, context and anchor projections), per output channel, with activations quantized per token; acceptance unchanged, 5% off the draft step at batch 1-4, no measurable end-to-end gain (see Results) |
| `FLASHMAP_FUSED_SAMPLER` | 0 | opt-in (lossless, measured on Qwen3-8B only): one sampling kernel (argmax for greedy rows, Gumbel-max with in-kernel Philox otherwise), no noise buffer, full-vocabulary logits scattered only for sampling batches |

`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (an sglang option) overlaps DSPARK's
verify planning with the previous step; it is optional, and the benchmarks below
set it for every speculative server.

`--speculative-dspark-block-size` is the number of drafts verified per request
(16, the whole block, by default). At high concurrency the target's verify pass
becomes compute-bound and the last drafts of a block are rarely accepted, so a
smaller value raises throughput: 15 (the verify width DFlash uses) is as fast or
faster everywhere measured from concurrency 4 up (see Results); the map still
drafts its whole block, as it was trained, and only the first drafts are
verified.

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
# the served draft against a direct PyTorch implementation of the map
python scripts/gates/check_flashmap_sglang.py --draft FlashMap-Qwen3-8B-sglang --target Qwen/Qwen3-8B

# throughput/latency vs the target alone and DFlash (see the script header)
TARGET=Qwen/Qwen3-8B PROMPTS=bench_prompts OUT=logs/fm8b OUTPUT_DETAILS=1 \
SERVERS="target=target fm1=flashmap:FlashMap-Qwen3-8B-sglang fm2=flashmap:FlashMap-Qwen3-8B-sglang:FLASHMAP_NUM_STEPS=2 dflash=dflash:z-lab/Qwen3-8B-DFlash-b16" \
  benchmarks/flashmap/bench_sglang.sh
python benchmarks/flashmap/compare_outputs.py logs/fm8b   # greedy outputs vs the target's
```

## Results

One H100 PCIe 80GB (350 W), sglang 0.5.20, bf16 target and draft, outputs up to
1024 tokens, chat-templated prompts with thinking disabled: the first 16, 32 and
64 prompts of each task at concurrency 1, 4 and 16. Each cell is acceptance
length (bonus token included) / output throughput (speedup over the target
alone). DFlash is z-lab's `Qwen3-{8B,4B}-DFlash-b16` on SGLang's own DFLASH
worker; "DeepSpec" is the DeepSpec repository's FlashMap path under the same
DSPARK worker, venv and flags, for parity. FlashMap uses the 64k draft
vocabulary (the export default). Throughput at concurrency 16 varies by 5-10%
between runs; its FlashMap 1-jump, DeepSpec 1-jump and DFlash cells average two
runs.

**Qwen3-8B, greedy**

| concurrency 1 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 105 tok/s | 105 tok/s | 105 tok/s |
| DFlash | 6.18 / 491 (4.68x) | 6.33 / 487 (4.65x) | 2.23 / 178 (1.69x) |
| FlashMap, 1 jump | 7.72 / 657 (6.26x) | 5.70 / 483 (4.61x) | 2.32 / 200 (1.91x) |
| FlashMap, 2 jumps | 8.22 / 646 (6.16x) | 6.11 / 479 (4.57x) | 2.37 / 188 (1.80x) |
| DeepSpec, 1 jump | 7.71 / 657 (6.26x) | 5.71 / 486 (4.64x) | 2.33 / 201 (1.92x) |
| DeepSpec, 2 jumps | 8.24 / 649 (6.19x) | 6.10 / 476 (4.54x) | 2.36 / 187 (1.79x) |

| concurrency 4 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 385 tok/s | 384 tok/s | 327 tok/s |
| DFlash | 6.43 / 1703 (4.42x) | 6.14 / 1698 (4.43x) | 2.73 / 727 (2.23x) |
| FlashMap, 1 jump | 7.91 / 2224 (5.77x) | 5.53 / 1674 (4.36x) | 2.82 / 808 (2.47x) |
| FlashMap, 2 jumps | 8.40 / 2196 (5.70x) | 5.88 / 1623 (4.23x) | 2.84 / 756 (2.31x) |
| DeepSpec, 1 jump | 7.84 / 2243 (5.82x) | 5.58 / 1682 (4.38x) | 2.83 / 813 (2.49x) |
| DeepSpec, 2 jumps | 8.63 / 2246 (5.83x) | 5.85 / 1618 (4.22x) | 2.82 / 754 (2.31x) |

| concurrency 16 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 1282 tok/s | 1244 tok/s | 1061 tok/s |
| DFlash | 6.32 / 4415 (3.44x) | 6.50 / 4681 (3.76x) | 3.56 / 2344 (2.21x) |
| FlashMap, 1 jump | 7.90 / 5299 (4.13x) | 5.73 / 4223 (3.39x) | 3.53 / 2311 (2.18x) |
| FlashMap, 1 jump, verify width 15 | 7.69 / 5647 (4.41x) | 5.70 / 4403 (3.54x) | 3.52 / 2545 (2.40x) |
| FlashMap, 2 jumps | 8.43 / 5575 (4.35x) | 6.08 / 3988 (3.20x) | 3.65 / 2222 (2.09x) |
| DeepSpec, 1 jump | 7.83 / 5495 (4.29x) | 5.69 / 4211 (3.39x) | 3.53 / 2365 (2.23x) |
| DeepSpec, 2 jumps | 8.43 / 5393 (4.21x) | 6.10 / 4031 (3.24x) | 3.62 / 2192 (2.07x) |

**Qwen3-8B, temperature 1** (rejection sampling; one run, acceptance is itself random)

| | gsm8k c1 | humaneval c1 | mt-bench c1 | gsm8k c16 |
|---|---|---|---|---|
| target alone | 104 tok/s | 104 tok/s | 104 tok/s | 1280 tok/s |
| DFlash | 5.31 / 422 (4.05x) | 5.19 / 402 (3.86x) | 2.12 / 167 (1.60x) | 5.59 / 3296 (2.58x) |
| FlashMap, 1 jump | 5.97 / 503 (4.83x) | 4.75 / 402 (3.87x) | 2.25 / 192 (1.85x) | 6.62 / 4700 (3.67x) |
| FlashMap, 2 jumps | 6.78 / 524 (5.04x) | 5.07 / 395 (3.80x) | 2.32 / 183 (1.76x) | 7.05 / 4287 (3.35x) |
| DeepSpec, 1 jump | 6.25 / 530 (5.09x) | 4.95 / 419 (4.03x) | 2.22 / 190 (1.83x) | 6.63 / 4615 (3.61x) |
| DeepSpec, 2 jumps | 6.64 / 521 (5.01x) | 5.64 / 436 (4.20x) | 2.32 / 183 (1.76x) | 7.07 / 4487 (3.51x) |

**Verify width** (`--speculative-dspark-block-size G`, Qwen3-8B, 1 jump, greedy; one run each)

| G | gsm8k c16 | humaneval c16 | mt-bench c16 | gsm8k c4 | mt-bench c4 |
|---|---|---|---|---|---|
| 16 (default; c16 averaged) | 7.90 / 5299 | 5.73 / 4223 | 3.53 / 2311 | 7.91 / 2224 | 2.82 / 808 |
| 15 | 7.69 / 5647 | 5.70 / 4403 | 3.52 / 2545 | 7.81 / 2229 | 2.84 / 856 |
| 12 | 7.35 / 5264 | 5.54 / 4518 | 3.49 / 2543 | 7.36 / 2212 | 2.83 / 818 |
| 8 | 6.27 / 4903 | 5.03 / 4381 | 3.41 / 2592 | 6.27 / 1889 | 2.83 / 849 |

Verifying 15 drafts per request (16 tokens with the anchor, DFlash's verify
width) instead of 16 is as fast or faster in every measured cell, by 4-10% at
concurrency 16, where the target's verify pass is compute-bound. Narrower widths
help code (12) and chat (8) slightly more at concurrency 16 but cost acceptance
on gsm8k. Use G=15 from concurrency 4 up; G below 16 was not measured
at concurrency 1, where verifying is memory-bound and the full block costs
almost nothing extra.

**Opt-in variants** (`FLASHMAP_FOLDED_NOISE=1 FLASHMAP_FUSED_SAMPLER=1`, Qwen3-8B, 1 jump, greedy; one run)

| | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| concurrency 1 | 7.87 / 671 (6.40x) | 5.68 / 483 (4.61x) | 2.33 / 201 (1.92x) |
| concurrency 4 | 8.00 / 2257 (5.86x) | 5.54 / 1669 (4.35x) | 2.85 / 814 (2.49x) |
| concurrency 16 | 7.73 / 5551 (4.33x) | 5.70 / 4393 (3.53x) | 3.56 / 2370 (2.24x) |

Both are lossless (below) and as fast or slightly faster than the defaults
(+2% at concurrency 1 on gsm8k; the folded noise saves 20 us of the 1.04 ms
draft step at batch 1); they stay opt-in until measured on more settings.

**Qwen3-4B, greedy**

| concurrency 1 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 172 tok/s | 172 tok/s | 172 tok/s |
| DFlash | 6.01 / 752 (4.36x) | 6.39 / 798 (4.64x) | 2.30 / 292 (1.69x) |
| FlashMap, 1 jump | 7.33 / 928 (5.38x) | 5.74 / 733 (4.26x) | 2.50 / 323 (1.87x) |
| FlashMap, 2 jumps | 8.03 / 897 (5.20x) | 6.17 / 702 (4.08x) | 2.47 / 283 (1.64x) |
| DeepSpec, 1 jump | 7.40 / 941 (5.46x) | 5.66 / 728 (4.23x) | 2.47 / 320 (1.86x) |

| concurrency 4 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 626 tok/s | 635 tok/s | 550 tok/s |
| DFlash | 6.05 / 2555 (4.08x) | 6.17 / 2721 (4.29x) | 2.85 / 1224 (2.22x) |
| FlashMap, 1 jump | 7.48 / 3168 (5.06x) | 5.53 / 2463 (3.88x) | 3.07 / 1317 (2.39x) |
| FlashMap, 2 jumps | 8.32 / 3083 (4.92x) | 5.80 / 2230 (3.51x) | 3.10 / 1178 (2.14x) |
| DeepSpec, 1 jump | 7.61 / 3168 (5.06x) | 5.39 / 2375 (3.74x) | 3.04 / 1321 (2.40x) |

| concurrency 16 | gsm8k | humaneval | mt-bench |
|---|---|---|---|
| target alone | 2055 tok/s | 1666 tok/s | 1683 tok/s |
| DFlash | 6.22 / 6668 (3.24x) | 6.40 / 6318 (3.79x) | 3.59 / 3763 (2.24x) |
| FlashMap, 1 jump | 7.66 / 7798 (3.79x) | 5.67 / 5505 (3.30x) | 3.69 / 3861 (2.29x) |
| FlashMap, 2 jumps | 8.11 / 7269 (3.54x) | 6.05 / 4759 (2.86x) | 3.75 / 3420 (2.03x) |
| DeepSpec, 1 jump | 7.59 / 7926 (3.86x) | 5.73 / 5349 (3.21x) | 3.61 / 3804 (2.26x) |

**FP8 draft weights** (`FLASHMAP_FP8=1`, Qwen3-8B, 1 jump, greedy; one run each)

| | gsm8k c1 | humaneval c1 | mt-bench c1 | gsm8k c16 | mt-bench c16 | draft step bs 1 / 4 / 16 |
|---|---|---|---|---|---|---|
| bf16 | 7.96 / 672 | 5.70 / 483 | 2.32 / 200 | 7.81 / 5569 | 3.54 / 2380 | 1030 / 1155 / 1670 us |
| FP8 | 7.71 / 649 | 5.70 / 484 | 2.32 / 201 | 7.79 / 5660 | 3.51 / 2335 | 978 / 1113 / 1687 us |

Acceptance is unchanged within run-to-run noise, but the draft step only gets
5% faster at small batches (the per-token activation quantization and FP8 GEMMs
at 16 rows per request do not reach the halved weight traffic's bound), about
0.4% of a decode step, below the noise of end-to-end throughput; FP8 weights
stay opt-in.

**FP8 draft KV pool** (`--speculative-draft-kv-cache-dtype fp8_e4m3`, Qwen3-8B, 1 jump, greedy; one run each)

| | gsm8k c1 | humaneval c1 | gsm8k c16 | mt-bench c16 | arena-hard-v2 c16 |
|---|---|---|---|---|---|
| bf16 draft KV | 7.73 / 659 | 5.67 / 481 | 7.72 / 5774 | 3.52 / 2362 | 3.07 / 2285 |
| e4m3 draft KV | 7.71 / 647 | 5.73 / 481 | 7.81 / 5667 | 3.55 / 2379 | 3.06 / 2278 |

The cross-attention context is written e4m3 (scale 1; keys k-normed and
rotated, by a Triton port of the fused write) and read by FA3 with the query
quantized per request and head. Acceptance is unchanged and the draft pool
halves (7.1 to 3.6 GB at 311k tokens on Qwen3-8B, memory sglang can give to the
target's KV), but throughput is unchanged at these lengths (prompt plus output
up to ~2k tokens): the draft step's context reads are not yet the bottleneck.

**Losslessness.** Speculative verification runs the target on 17-token windows,
decoding alone on one token, and their bf16 kernels round differently, so a
near-tied argmax can flip and greedy texts part ways with the target-only
server's (DeepSpec's path diverges at the same places). `check_greedy.py`
teacher-forces each greedy output's token ids through the target instead: every
output token of every FlashMap server (8B at 1 and 2 jumps and with the opt-in
variants, 4B at 1 and 2 jumps; 16 prompts per task, about 18k tokens per
server) is the target's argmax or within 0.5 logit of it (a bf16 near-tie), as
are the target-only server's own outputs.

**Gaps.** Not measured: verify widths below 16 at concurrency 1, the opt-in
variants on Qwen3-4B, temperature 1 on Qwen3-4B, and the e4m3 draft KV pool at
long contexts (where its reads would matter). Tree drafting (several candidate
blocks verified as a tree, DeepSpec's best concurrency-1 configuration) is not
ported yet.

## SGLang internals touched

Importing the package installs two hooks on `DSparkWorkerV2` (once, however
often it is imported), both active only for FlashMap drafts: `init_cuda_graphs`
turns off the draft attention backend's per-step metadata (the draft never uses
that backend), and `_maybe_build_draft_sampler` replaces DSPARK's graph-folded
sampler with the one-pass sampler above, which takes over its buffers. No
SGLang source file is patched.
