# coding=utf-8
"""Triton kernels of the FlashMap sglang draft (see :mod:`.flashmap`).

They reproduce the rounding of the reference modules they replace: the RMSNorm
multiplies the bf16 input by its bf16-rounded scale ``weight * rsqrt(mean(x^2) +
eps)``, and the interleaved-pair RoPE rotates in float32 and rounds back to
bf16. The ops are registered as custom ops so torch.compile keeps them whole
inside the draft step it compiles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_self_attention_kernel(
    qkv_ptr,
    q_norm_ptr,
    k_norm_ptr,
    freqs_ptr,
    out_ptr,
    eps,
    sm_scale,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # One program per (block, head): the block's T queries attend to its T keys.
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    t = tl.arange(0, T)[:, None]
    i = tl.arange(0, HALF)[None, :]
    d = tl.arange(0, D)[None, :]
    # qkv is (blocks, T, 3, H, D); RoPE pairs dims (2i, 2i + 1).
    row = qkv_ptr + (b * T + t) * (3 * H * D) + h * D
    freqs = freqs_ptr + (b * T + t) * D + 2 * i  # (blocks, T, D / 2, [cos, sin])
    cos = tl.load(freqs)
    sin = tl.load(freqs + 1)

    q_even = tl.load(row + 2 * i).to(tl.float32)
    q_odd = tl.load(row + 2 * i + 1).to(tl.float32)
    k_even = tl.load(row + H * D + 2 * i).to(tl.float32)
    k_odd = tl.load(row + H * D + 2 * i + 1).to(tl.float32)

    # RMSNorm: x * bf16(weight * rsqrt(mean(x^2) + eps)), in bf16.
    q_rms = tl.rsqrt((tl.sum(q_even * q_even, 1) + tl.sum(q_odd * q_odd, 1)) / D + eps)[
        :, None
    ]
    k_rms = tl.rsqrt((tl.sum(k_even * k_even, 1) + tl.sum(k_odd * k_odd, 1)) / D + eps)[
        :, None
    ]
    q_even = _scale(q_even, tl.load(q_norm_ptr + 2 * i) * q_rms)
    q_odd = _scale(q_odd, tl.load(q_norm_ptr + 2 * i + 1) * q_rms)
    k_even = _scale(k_even, tl.load(k_norm_ptr + 2 * i) * k_rms)
    k_odd = _scale(k_odd, tl.load(k_norm_ptr + 2 * i + 1) * k_rms)

    # RoPE, rounded to bf16 as the attention receives it.
    q_rot_even = (q_even * cos - q_odd * sin).to(tl.bfloat16)
    q_rot_odd = (q_odd * cos + q_even * sin).to(tl.bfloat16)
    k_rot_even = (k_even * cos - k_odd * sin).to(tl.bfloat16)
    k_rot_odd = (k_odd * cos + k_even * sin).to(tl.bfloat16)

    scores = tl.dot(q_rot_even, tl.trans(k_rot_even)) + tl.dot(
        q_rot_odd, tl.trans(k_rot_odd)
    )
    scores = scores * sm_scale
    probs = tl.exp(scores - tl.max(scores, 1)[:, None])
    denom = tl.sum(probs, 1)[:, None]
    values = tl.load(row + 2 * H * D + d)
    out = tl.dot(probs.to(tl.bfloat16), values) / denom
    tl.store(out_ptr + (b * T + t) * (H * D) + h * D + d, out.to(tl.bfloat16))


@triton.jit
def _scale(x, scale):
    """``x`` (bf16 values in float32) times the bf16-rounded ``scale``, in bf16."""
    return (x * scale.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)


@torch.library.custom_op("specforge_flashmap::block_self_attention", mutates_args=())
def block_self_attention(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Self-attention within each draft block, from its projected ``qkv
    (blocks, T, 3, heads, head_dim)``: q/k RMSNorms with float32 weights, RoPE
    by ``freqs (blocks, T, head_dim / 2, [cos, sin])``, then unmasked softmax
    attention. Returns ``(blocks, T, heads, head_dim)``."""
    blocks, block_size, _, heads, head_dim = qkv.shape
    assert qkv.is_contiguous() and freqs.is_contiguous()
    assert freqs.dtype == torch.float32
    out = torch.empty(
        (blocks, block_size, heads, head_dim), dtype=qkv.dtype, device=qkv.device
    )
    _block_self_attention_kernel[(blocks * heads,)](
        qkv,
        q_norm,
        k_norm,
        freqs,
        out,
        eps,
        head_dim**-0.5,
        T=block_size,
        H=heads,
        D=head_dim,
    )
    return out


@block_self_attention.register_fake
def _(qkv, q_norm, k_norm, freqs, eps):
    blocks, block_size, _, heads, head_dim = qkv.shape
    return qkv.new_empty((blocks, block_size, heads, head_dim))


@triton.jit
def _rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, 0) / n_cols + eps)
    scale = tl.load(weight_ptr + cols, mask=mask, other=0.0) * rms
    scale = scale.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (x * scale).to(tl.bfloat16), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of a contiguous bf16 ``x (..., C)`` with a float32 ``weight``,
    in one launch."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert weight.dtype == torch.float32
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    if x.numel():
        _rms_norm_kernel[(x.numel() // n_cols,)](
            x, weight, out, n_cols, eps, BLOCK=triton.next_power_of_2(n_cols)
        )
    return out


@triton.jit
def _rms_norm_rope_kernel(
    x_ptr, weight_ptr, freqs_ptr, out_ptr, eps, H: tl.constexpr, D: tl.constexpr
):
    # One program per (token, head) of a contiguous (tokens, H, D) input.
    token = tl.program_id(0) // H
    head = tl.program_id(0) % H
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    row = x_ptr + (token * H + head) * D
    even = tl.load(row + 2 * i).to(tl.float32)
    odd = tl.load(row + 2 * i + 1).to(tl.float32)
    rms = tl.rsqrt((tl.sum(even * even, 0) + tl.sum(odd * odd, 0)) / D + eps)
    even = _scale(even, tl.load(weight_ptr + 2 * i) * rms)
    odd = _scale(odd, tl.load(weight_ptr + 2 * i + 1) * rms)
    cos = tl.load(freqs_ptr + token * D + 2 * i)
    sin = tl.load(freqs_ptr + token * D + 2 * i + 1)
    out = out_ptr + (token * H + head) * D
    tl.store(out + 2 * i, (even * cos - odd * sin).to(tl.bfloat16))
    tl.store(out + 2 * i + 1, (odd * cos + even * sin).to(tl.bfloat16))


@torch.library.custom_op("specforge_flashmap::rms_norm_rope", mutates_args=())
def rms_norm_rope(
    x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float
) -> torch.Tensor:
    """RoPE of RMSNorm(x) for ``x (..., T, heads, head_dim)`` and ``freqs (...,
    T, head_dim / 2, [cos, sin])``, in one launch."""
    *_, heads, head_dim = x.shape
    assert x.is_contiguous() and freqs.is_contiguous()
    assert freqs.dtype == torch.float32
    out = torch.empty_like(x)
    _rms_norm_rope_kernel[(x.numel() // head_dim,)](
        x, weight, freqs, out, eps, H=heads, D=head_dim
    )
    return out


@rms_norm_rope.register_fake
def _(x, weight, freqs, eps):
    return torch.empty_like(x)


__all__ = ["block_self_attention", "rms_norm", "rms_norm_rope"]


@triton.jit
def _sample_tiles_kernel(
    logits_ptr,
    temperatures_ptr,
    greedy_ptr,
    seed_ptr,
    best_ptr,
    best_index_ptr,
    V,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    # One program per (row, tile of the vocabulary): the tile's best key and
    # its first index. A greedy row's key is its logit; a sampled row's is its
    # logit over the temperature plus Gumbel noise from in-kernel Philox
    # (Gumbel-max: the argmax is a sample of softmax(logits / T)).
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    key = tl.load(logits_ptr + row * V + offs, mask=mask, other=float("-inf"))
    key = key.to(tl.float32)
    if tl.load(greedy_ptr + row) == 0:
        uniform = tl.rand(tl.load(seed_ptr), row * V + offs)
        key = key / tl.load(temperatures_ptr + row) - tl.log(-tl.log(uniform))
    key = tl.where(mask, key, float("-inf"))
    best = tl.max(key, axis=0)
    index = tl.min(tl.where(key == best, offs, V), axis=0)
    tl.store(best_ptr + row * n_tiles + tile, best)
    tl.store(best_index_ptr + row * n_tiles + tile, index)


def sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: torch.Tensor,
    seed: torch.Tensor,
    block_v: int = 4096,
) -> torch.Tensor:
    """A token per row of contiguous ``logits (rows, V)``: the argmax for
    ``greedy`` rows, else a sample of ``softmax(logits / temperature)``, with
    Philox noise seeded by the int64 scalar tensor ``seed`` (advance it between
    calls, in-graph, for fresh noise). Reads the logits once; no noise buffer."""
    rows, vocab = logits.shape
    assert logits.is_contiguous()
    n_tiles = triton.cdiv(vocab, block_v)
    best = torch.empty((rows, n_tiles), dtype=torch.float32, device=logits.device)
    best_index = torch.empty((rows, n_tiles), dtype=torch.int64, device=logits.device)
    _sample_tiles_kernel[(rows, n_tiles)](
        logits, temperatures, greedy, seed, best, best_index, vocab, n_tiles,
        BLOCK_V=block_v,
    )  # fmt: skip
    return best_index.gather(1, best.argmax(dim=1, keepdim=True)).squeeze(1)


__all__ += ["sample_rows"]
