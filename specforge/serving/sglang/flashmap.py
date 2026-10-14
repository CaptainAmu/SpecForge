# coding=utf-8
"""FlashMap as an sglang draft model, served by sglang's DSPARK worker.

Export a FlashMap checkpoint with ``python -m specforge.export.flashmap`` and
serve it with

    SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang \\
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B \\
        --speculative-algorithm DSPARK --speculative-draft-model-path <export>

DSPARK drives a block draft through a small protocol, implemented here:

* ``write_target_hidden_kv``: DSPARK hands over the target's captured hidden
  states for every committed position. FlashMap's cross-attention context (the
  keys and values each DiT layer projects from them) goes into the draft KV
  pool, which the export sizes to the DiT's layers, heads and head_dim. All
  layers are projected in one matmul and written, k-normed and rotated, by
  sglang's fused DSpark kernel, which also skips the uncommitted verify
  columns, so the write never synchronizes with the host.
* ``forward``: one block of ``block_size`` drafts after each request's anchor
  (the last verified token), for the whole batch in one pass that DSPARK
  captures in a CUDA graph. It is compiled with torch.compile, and
  - the anchor's own context entry is written into its verify slot, so the
    cross-attention reads every request's paged context directly
    (FlashAttention with sglang's ``req_to_token`` rows as page tables, one
    tile schedule for all layers);
  - self-attention within each block is one Triton kernel
    (:mod:`.flashmap_kernels`), its q/k/v one matmul;
  - every AdaLN modulation depends only on the jump's times, which are fixed,
    so they are computed once at load;
  - one jump starts from noise at ``t = 0``, where the state's posterior is
    uniform whatever the noise: its embedding is a constant, and the noise
    enters only through ``x @ E ~ N(0, E^T E)``, drawn as ``z @ L^T`` with
    ``L L^T = E^T E`` (Cholesky), so no vocabulary-sized noise is ever formed;
  - more jumps (``FLASHMAP_NUM_STEPS``) take their vocabulary-sized inputs from
    a bank of precomputed noise draws (:mod:`.flashmap_jump_bank`) and read
    only the embedding rows of each jump's top predictions.
* ``compute_base_logits``: the DiT's output head (dim -> hidden) folded into the
  target's LM head (hidden -> vocab) as one smaller matmul, restricted to the
  draft vocabulary (the target's most frequent tokens, 64k by default, stored
  in the export); drafted indices map back to token ids in the draft graph.

The draft borrows the target's token embedding and LM head (DSPARK attaches
them); its Markov head is an identity (:mod:`.flashmap_sampler`), so DSPARK
samples and verifies the map's logits unchanged, in one pass per block.

Environment knobs: ``FLASHMAP_NUM_STEPS`` (jumps per draft; default the
export's ``num_steps``), ``FLASHMAP_DRAFT_VOCAB`` (0: the full vocabulary,
``<k>``: the export ranking's top k), ``FLASHMAP_JUMP_BANK`` /
``FLASHMAP_JUMP_TOPK`` (bank draws, 8192; support per jump, 16),
``FLASHMAP_COMPILE`` (0 disables torch.compile) and ``FLASHMAP_COMPILE_MODE``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
    fused_kv_norm_rope_write,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.dspark_components import dspark_worker_v2
from torch import nn

from specforge.serving.sglang.flashmap_jump_bank import JumpNoiseBank, top_predictions
from specforge.serving.sglang.flashmap_kernels import (
    block_self_attention,
    rms_norm,
    rms_norm_rope,
)
from specforge.serving.sglang.flashmap_sampler import (
    FlashMapDraftSampler,
    IdentityMarkov,
)

logger = logging.getLogger(__name__)


# A custom op, so torch.compile traces the draft step through it in one graph.
@torch.library.custom_op("flashmap::paged_cross_attention", mutates_args=())
def _paged_cross_attention(
    query: torch.Tensor,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    scheduler_metadata: torch.Tensor,
) -> torch.Tensor:
    """``query (bs, K, heads, head_dim)`` onto each request's first
    ``cache_seqlens`` pool rows, listed by ``page_table`` (page size 1)."""
    return flash_attn_with_kvcache(
        q=query,
        k_cache=key_buffer.unsqueeze(1),
        v_cache=value_buffer.unsqueeze(1),
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        causal=False,
        scheduler_metadata=scheduler_metadata,
    )


@_paged_cross_attention.register_fake
def _(query, key_buffer, value_buffer, cache_seqlens, page_table, scheduler_metadata):
    return torch.empty_like(query)


def _cross_attention_schedule(query_shape, cache_seqlens, page_table):
    """FlashAttention's tile schedule for ``_paged_cross_attention``, computed
    once per draft step for all layers (it depends only on shapes and lengths)."""
    bsz, block_size, heads, head_dim = query_shape
    return get_scheduler_metadata(
        batch_size=bsz,
        max_seqlen_q=block_size,
        max_seqlen_k=page_table.shape[1],
        num_heads=heads,
        num_heads_k=heads,
        headdim=head_dim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=torch.bfloat16,
        page_size=1,
        causal=False,
    )


def rope_freqs(positions: torch.Tensor, head_dim: int, base: float) -> torch.Tensor:
    """Positions ``(...)`` -> ``(..., head_dim // 2, [cos, sin])`` in float32;
    RoPE rotates interleaved pairs of channels ``(2i, 2i + 1)``."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )
    angles = positions.float()[..., None] * inv_freq
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FlashMap's RMSNorm, ``x * bf16(weight * rsqrt(mean(x^2) + eps))``, with a
    float32 ``weight`` (torch version of ``flashmap_kernels.rms_norm``)."""
    mean_sq = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x * (weight * torch.rsqrt(mean_sq + eps)).to(x.dtype)


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Norm(nn.Module):
    """FlashMap's RMSNorm (most are applied by the fused kernels instead)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight.float(), self.eps)


class _TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, cond_dim)
        self.linear_2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = _timestep_embedding(t, self.embed_dim).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class _AdaLN(nn.Module):
    """AdaLN-zero's conditioning projection: ``(scale, shift, gate)``."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 3 * dim)

    def forward(self, cond: torch.Tensor):
        return self.linear(F.silu(cond)).chunk(3, dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = _Norm(dim // heads, eps)
        self.k_norm = _Norm(dim // heads, eps)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, cond_dim: int, eps: float):
        super().__init__()
        self.self_attn_adaln = _AdaLN(dim, cond_dim)
        self.self_attn = _Attention(dim, heads, eps)
        self.cross_attn_adaln = _AdaLN(dim, cond_dim)
        self.cross_attn = _Attention(dim, heads, eps)
        self.mlp_adaln = _AdaLN(dim, cond_dim)
        self.mlp = _MLP(dim, mlp_dim)


class _FoldedHead(nn.Module):
    """The draft's head as DSPARK reads it: the DiT head folded into the
    target's LM head over the draft vocabulary. ``org_vocab_size`` is the full
    vocabulary, which DSPARK's sampling buffers are sized to."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, vocab_size: int):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.org_vocab_size = int(vocab_size)


class FlashMapDraftModel(nn.Module):
    # Read by the DSPARK worker and the hooks below.
    flashmap = True
    uses_own_vocab_modules = False
    supports_pre_gather_target_hidden_projection = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        self.config = config
        fm = dict(config.flashmap_config)
        self.vocab_size = int(fm["vocab_size"])
        self.target_hidden_size = int(fm["target_hidden_size"])
        self.dim = int(fm["hidden_size"])
        self.num_heads = int(fm["num_attention_heads"])
        self.head_dim = self.dim // self.num_heads
        self.block_size = self.gamma = int(fm["block_size"])
        self.num_stages = int(fm["num_hidden_layers"])
        self.rope_theta = float(fm["rope_theta"])
        self.time_scale = float(fm["time_scale"])
        self.rms_norm_eps = float(fm["rms_norm_eps"])
        self.qk_norm_eps = float(fm["qk_norm_eps"])
        self.layer_norm_eps = float(fm["layer_norm_eps"])
        assert fm["noise_gate"] == "jump", fm["noise_gate"]
        cond_dim = int(fm["cond_dim"])
        context_features = len(fm["target_layer_ids"]) * self.target_hidden_size

        # The released weights, under their release names.
        self.fc = nn.Linear(context_features, self.dim, bias=False)
        eps = self.rms_norm_eps
        self.hidden_norm = _Norm(self.dim, eps)
        self.anchor_proj = nn.Linear(self.target_hidden_size, self.dim)
        self.anchor_norm = _Norm(self.dim, eps)
        self.state_norm = _Norm(self.target_hidden_size, eps)
        self.noise_norm = _Norm(self.target_hidden_size, eps)
        self.noise_gate = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.x_embed = nn.Linear(self.target_hidden_size, self.dim)
        self.time_embed = _TimeEmbedding(int(fm["time_embed_dim"]), cond_dim)
        self.jump_embed = _TimeEmbedding(int(fm["time_embed_dim"]), cond_dim)
        self.layers = nn.ModuleList(
            _Layer(
                self.dim,
                self.num_heads,
                int(fm["intermediate_size"]),
                cond_dim,
                self.qk_norm_eps,
            )
            for _ in range(self.num_stages)
        )
        self.final_adaln = _AdaLN(self.dim, cond_dim)
        self.head = nn.Linear(self.dim, self.target_hidden_size)

        # DSPARK's draft protocol.
        self.sample_from_anchor = True
        self.markov_head = IdentityMarkov(vocab_size=self.vocab_size)
        self.confidence_head = None
        self.logits_mup_width_multiplier = None
        self.lm_head = None  # the folded head, once the target's is attached
        self.embedding = None  # the target's token embedding (not a submodule)
        self.hot_ids = None  # draft vocabulary -> token ids (None: all tokens)
        self.folded_head = None

        self.num_steps = int(os.environ.get("FLASHMAP_NUM_STEPS") or fm["num_steps"])
        self.grid = [float(t) for t in fm["jump_grids"][str(self.num_steps)]]
        assert self.grid[0] == 0.0 and self.grid[-1] == 1.0, self.grid
        self.jump_bank_size = int(os.environ.get("FLASHMAP_JUMP_BANK", "8192"))
        self.jump_topk = int(os.environ.get("FLASHMAP_JUMP_TOPK", "16"))
        self.jump_bank = None
        size = os.environ.get("FLASHMAP_DRAFT_VOCAB")
        size = int(size if size is not None else getattr(config, "draft_vocab_size", 0))
        self.draft_vocab_size = size if 0 < size < self.vocab_size else 0
        self._vocab_ranking = None  # the export's draft_vocab_ids
        # Scales the starting noise; checks set 0 for deterministic drafts.
        self.noise_temperature = 1.0
        self._constants = None
        self._fused_write = None  # (pool id, kernel arguments)
        self._draft = self._draft_impl
        self._anchor_kv = self._anchor_kv_impl
        if os.environ.get("FLASHMAP_COMPILE", "1") == "1":
            mode = os.environ.get("FLASHMAP_COMPILE_MODE") or None
            self._draft = torch.compile(self._draft_impl, dynamic=True, mode=mode)
            self._anchor_kv = torch.compile(self._anchor_kv_impl, dynamic=True)
        logger.info(
            "FlashMap draft: %d jump(s) on %s, block_size=%d, draft vocabulary %s",
            self.num_steps,
            self.grid,
            self.block_size,
            self.draft_vocab_size or "full",
        )

    # --- loading -----------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name == "draft_vocab_ids":
                self._vocab_ranking = tensor
                continue
            if name not in params:
                raise KeyError(f"unexpected FlashMap weight {name!r}")
            with torch.no_grad():
                params[name].copy_(tensor.to(params[name].dtype))
            loaded.add(name)
        missing = sorted(
            n for n in params if n not in loaded and not n.startswith("markov_head.")
        )
        if missing:
            raise KeyError(f"FlashMap weights missing from the checkpoint: {missing}")
        layers = self.layers
        # Every layer's context keys and values in one matmul (rows [k; v] per
        # layer); each k_proj/v_proj weight becomes a view of its rows.
        self._ctx_kv_weight = torch.cat(
            [
                torch.cat(
                    [layer.cross_attn.k_proj.weight, layer.cross_attn.v_proj.weight]
                )
                for layer in layers
            ]
        )
        for i, layer in enumerate(layers):
            rows = self._ctx_kv_weight[2 * self.dim * i : 2 * self.dim * (i + 1)]
            layer.cross_attn.k_proj.weight.data = rows[: self.dim]
            layer.cross_attn.v_proj.weight.data = rows[self.dim :]
        # Likewise each self-attention's queries, keys and values.
        self._self_qkv_weights = []
        for layer in layers:
            attn = layer.self_attn
            qkv = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
            )
            attn.q_proj.weight.data = qkv[: self.dim]
            attn.k_proj.weight.data = qkv[self.dim : 2 * self.dim]
            attn.v_proj.weight.data = qkv[2 * self.dim :]
            self._self_qkv_weights.append(qkv)
        # The norms' weights in float32, as the kernels take them.
        self._self_norm_weights = [
            (
                layer.self_attn.q_norm.weight.float(),
                layer.self_attn.k_norm.weight.float(),
            )
            for layer in layers
        ]
        self._cross_q_norm_weights = [
            layer.cross_attn.q_norm.weight.float() for layer in layers
        ]
        self._hidden_norm_weight = self.hidden_norm.weight.float()

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module):
        """DSPARK hands over the target's embedding and LM head; everything the
        draft step reads from them is precomputed here, before sglang sizes its
        KV pools. A target with tied embeddings (Qwen3-4B) hands over its
        embedding as the LM head; one without a head of its own falls back to
        the embedding too."""
        self.embedding = embed_tokens.weight.detach()
        assert self.embedding.shape == (self.vocab_size, self.target_hidden_size)
        head = getattr(lm_head, "weight", None)
        self.prepare(self.embedding if head is None else head.detach())

    @torch.no_grad()
    def prepare(self, lm_head_weight: torch.Tensor) -> None:
        if self.draft_vocab_size:
            assert self._vocab_ranking is not None, (
                f"a draft vocabulary of {self.draft_vocab_size} tokens, but the "
                "export has no draft_vocab_ids (set FLASHMAP_DRAFT_VOCAB=0)"
            )
            self.hot_ids = (
                self._vocab_ranking[: self.draft_vocab_size]
                .to(device=self.embedding.device, dtype=torch.int64)
                .sort()
                .values
            )
        self.folded_head = self._fold_lm_head(lm_head_weight, self.hot_ids)
        self.lm_head = _FoldedHead(*self.folded_head, self.vocab_size)
        self.markov_head.hot_ids = self.hot_ids
        self._constants = self._jump_constants()
        if self.num_steps > 1:
            self.jump_bank = JumpNoiseBank(
                self,
                self.grid,
                size=self.jump_bank_size,
                temperature=self.noise_temperature,
            )
            logger.info(
                "FlashMap: %d jumps from a bank of %d noise draws (%.0f MB), "
                "predictions cut to %d tokens",
                self.num_steps,
                self.jump_bank_size,
                self.jump_bank.nbytes() / 2**20,
                self.jump_topk,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fold_lm_head(self, lm_head, rows: Optional[torch.Tensor], chunk: int = 8192):
        """The DiT head (dim -> hidden, with bias) and the LM head (hidden ->
        vocab) as one ``(vocab rows, dim)`` projection and bias; there is no
        nonlinearity between them (Qwen3 has no final-logit softcap). Folded
        ``chunk`` rows at a time in float32."""
        count = lm_head.shape[0] if rows is None else rows.numel()
        weight = lm_head.new_empty(count, self.dim)
        bias = lm_head.new_empty(count)
        head_weight, head_bias = self.head.weight.float(), self.head.bias.float()
        for start in range(0, count, chunk):
            if rows is None:
                part = lm_head[start : start + chunk].float()
            else:
                part = lm_head[rows[start : start + chunk]].float()
            weight[start : start + chunk] = (part @ head_weight).to(lm_head.dtype)
            bias[start : start + chunk] = (part @ head_bias).to(lm_head.dtype)
        return weight, bias

    def _jump_constants(self):
        """Everything a draft needs that depends only on the jump times: per
        jump the AdaLN modulations ``(1 + scale, shift, gate)`` of every norm;
        the embedded starting state's constant (the uniform posterior's
        embedding) and the Cholesky factor of ``E^T E`` its noise is drawn
        with."""
        device = self.fc.weight.device
        per_jump = []
        for start, end in zip(self.grid[:-1], self.grid[1:]):
            times = torch.tensor([[start, end]], device=device) * self.time_scale
            cond = self.time_embed(times[:, 0]) + self.jump_embed(times[:, 1])
            modulations = []
            for layer in self.layers:
                adalns = (
                    layer.self_attn_adaln,
                    layer.cross_attn_adaln,
                    layer.mlp_adaln,
                )
                chunks = [adaln(cond) for adaln in adalns]
                modulations.append(
                    tuple((1 + scale, shift, gate) for scale, shift, gate in chunks)
                )
            scale, shift, _ = self.final_adaln(cond)
            per_jump.append((modulations, (1 + scale, shift)))
        embedding = self.embedding
        posterior = torch.full(
            (1, self.vocab_size), 1.0 / self.vocab_size, device=device
        ).to(embedding.dtype)
        uniform = self.state_norm(posterior @ embedding)[0]
        # E^T E accumulated in float64 a chunk of rows at a time.
        gram = torch.zeros(
            self.target_hidden_size,
            self.target_hidden_size,
            dtype=torch.float64,
            device=device,
        )
        for start in range(0, self.vocab_size, 8192):
            part = embedding[start : start + 8192].double()
            gram += part.T @ part
        noise_factor = torch.linalg.cholesky(gram).T.to(embedding.dtype)
        return per_jump, uniform, noise_factor

    def noise_gate_at(self, start: float, end: float) -> torch.Tensor:
        """The weight of the state's noise term for a jump from ``start`` to
        ``end``: ``noise_gate * (end - start)``."""
        return self.noise_gate * (end - start)

    # --- DSPARK protocol ------------------------------------------------------

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Its presence tells DSPARK the draft embeds its own block in-graph, so
        # it skips staging the target's embedding of the block every step; the
        # draft never reads it (the anchor is embedded by _anchor_kv).
        return F.embedding(input_ids, self.embedding)

    def prune_to_ctx_kv_injection(self) -> None:
        pass

    def compute_base_logits(self, hidden: torch.Tensor):
        weight, bias = self.folded_head
        return F.linear(hidden, weight, bias), None

    @torch.no_grad()
    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        assert not target_hidden_is_projected
        if target_hidden.shape[0] == 0:
            return
        meta, k_norm_weights, cos_sin, kv_size = self._fused_write_args(pool)
        # k-norm, RoPE and the write for every layer in one launch, skipping
        # the uncommitted verify columns.
        commit = cache_loc_2d is not None and commit_lens is not None
        context = rms_norm(
            self.fc(target_hidden.to(self._ctx_kv_weight.dtype)),
            self._hidden_norm_weight,
            self.rms_norm_eps,
        )
        fused_kv_norm_rope_write(
            F.linear(context, self._ctx_kv_weight),
            meta,
            k_norm_weights,
            cos_sin,
            positions,
            cache_loc_2d.reshape(-1) if commit else cache_loc,
            self.num_stages,
            kv_size,
            self.head_dim,
            self.qk_norm_eps,
            commit_lens=commit_lens if commit else None,
            locs_row_width=cache_loc_2d.shape[1] if commit else None,
            is_neox_style=False,  # FlashMap rotates interleaved pairs
        )

    def _fused_write_args(self, pool):
        """``fused_kv_norm_rope_write``'s arguments for ``pool``."""
        if self._fused_write is not None and self._fused_write[0] == id(pool):
            return self._fused_write[1]
        buffers = [
            (pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id))
            for layer_id in range(self.num_stages)
        ]
        for buf in (buf for pair in buffers for buf in pair):
            assert buf.is_cuda and buf.dtype == torch.bfloat16, buf.dtype
            assert buf.shape[1:] == (self.num_heads, self.head_dim), buf.shape
            assert buf.stride(1) == self.head_dim and buf.stride(2) == 1
        device = self._ctx_kv_weight.device
        meta = torch.tensor(
            [
                [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                for k, v in buffers
            ],
            dtype=torch.int64,
            device=device,
        )
        k_norm_weights = torch.stack(
            [layer.cross_attn.k_norm.weight.float() for layer in self.layers]
        )
        # Row p: cos then sin of p times each frequency.
        max_positions = int(self.config.max_position_embeddings) + self.block_size + 1
        freqs = rope_freqs(
            torch.arange(max_positions, device=device), self.head_dim, self.rope_theta
        )
        cos_sin = torch.cat([freqs[..., 0], freqs[..., 1]], dim=-1).contiguous()
        args = (meta, k_norm_weights, cos_sin, self.num_heads * self.head_dim)
        self._fused_write = (id(pool), args)
        return args

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        """A block for every request; returns the DiT's final hidden states,
        which ``compute_base_logits`` (or the folded sampler) maps to logits.

        DSPARK asks for ``gamma`` drafts per request, the block size unless
        ``--speculative-num-draft-tokens`` verifies fewer (at high concurrency
        the target's verify is compute-bound and the last drafts are rarely
        accepted); the map always drafts its whole block, as it was trained,
        and hands over the first ``gamma``."""
        del input_embeds, get_embedding, pp_proxy_tensors
        bsz, block_size = forward_batch.batch_size, self.block_size
        gamma = positions.numel() // bsz
        pool = get_token_to_kv_pool()
        anchor_positions = positions.view(bsz, gamma)[:, 0]
        block_positions = anchor_positions[:, None] + torch.arange(
            block_size, device=positions.device
        )
        # The anchor's context entry goes into its own slot, the first of the
        # request's verify window, so each request attends to positions
        # 0..anchor; committing the verified tokens later overwrites the slot.
        meta, k_norm_weights, cos_sin, kv_size = self._fused_write_args(pool)
        fused_kv_norm_rope_write(
            self._anchor_kv(input_ids.view(bsz, gamma)[:, 0]),
            meta,
            k_norm_weights,
            cos_sin,
            anchor_positions,
            forward_batch.out_cache_loc.view(bsz, gamma)[:, 0],
            self.num_stages,
            kv_size,
            self.head_dim,
            self.qk_norm_eps,
            is_neox_style=False,
        )
        page_table = get_req_to_token_pool().req_to_token[
            forward_batch.req_pool_indices
        ]
        cache_seqlens = (forward_batch.seq_lens + 1).to(torch.int32)
        schedule = _cross_attention_schedule(
            (bsz, block_size, self.num_heads, self.head_dim), cache_seqlens, page_table
        )
        hidden = self._draft(
            block_positions,
            page_table,
            cache_seqlens,
            schedule,
            [pool.get_key_buffer(i) for i in range(self.num_stages)],
            [pool.get_value_buffer(i) for i in range(self.num_stages)],
        )
        if gamma < block_size:
            hidden = hidden.view(bsz, block_size, -1)[:, :gamma].reshape(
                bsz * gamma, -1
            )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def _anchor_kv_impl(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """Every layer's context key/value row for each anchor token, before
        k-norm and RoPE (the fused write applies those)."""
        anchor = self.anchor_norm(
            self.anchor_proj(F.embedding(anchor_ids, self.embedding))
        )
        return F.linear(anchor, self._ctx_kv_weight)

    def _draft_impl(
        self, block_positions, page_table, cache_seqlens, schedule, keys, values
    ) -> torch.Tensor:
        """The block's ``num_steps`` jumps, a pure tensor function so it can be
        compiled; returns the last jump's hidden states ``(bs * K, dim)``."""
        _, uniform, noise_factor = self._constants
        bsz, block_size = block_positions.shape
        slots = bsz * block_size
        # The slots sit at anchor + 1 .. anchor + K.
        freqs = rope_freqs(block_positions + 1, self.head_dim, self.rope_theta)
        attend = (freqs, page_table, cache_seqlens, schedule, keys, values)
        if self.num_steps == 1:
            # One jump from t = 0: the noise drawn in the embedding's space.
            noise = self.noise_temperature * torch.randn(
                bsz,
                block_size,
                noise_factor.shape[0],
                device=block_positions.device,
                dtype=noise_factor.dtype,
            )
            gate = self.noise_gate_at(self.grid[0], self.grid[1])
            embedded = uniform + gate * self.noise_norm(noise @ noise_factor)
            return self._denoise(0, embedded, *attend).reshape(slots, -1)
        # Banked jumps (flashmap_jump_bank.py): no vocabulary-sized state, the
        # predictions between jumps from the draft vocabulary's head.
        bank = self.jump_bank
        rows, seeds = bank.draw(slots, block_positions.device)
        embedded = bank.first_input(self, rows, uniform)
        weight, bias = self.folded_head
        supports = []
        for step in range(self.num_steps):
            if step:
                embedded = bank.jump_input(self, step, rows, seeds, supports)
            hidden = self._denoise(
                step, embedded.view(bsz, block_size, -1), *attend
            ).reshape(slots, -1)
            if step + 1 < self.num_steps:
                logits = F.linear(hidden, weight, bias)
                supports.append(top_predictions(logits, self.jump_topk, self.hot_ids))
        return hidden

    def _denoise(
        self, step, embedded, freqs, page_table, cache_seqlens, schedule, keys, values
    ) -> torch.Tensor:
        """One evaluation of the map at jump ``step``'s times on the embedded
        state ``(bs, K, hidden)``; returns the final AdaLN's output."""
        modulations, (final_scale, final_shift) = self._constants[0][step]
        bsz, block_size, _ = embedded.shape
        shape = (bsz, block_size, self.num_heads, self.head_dim)
        eps = self.layer_norm_eps
        hidden = self.x_embed(embedded)
        for i, layer in enumerate(self.layers):
            (s_scale, s_shift, s_gate), (c_scale, c_shift, c_gate), mlp = modulations[i]
            m_scale, m_shift, m_gate = mlp
            # Self-attention within each request's block.
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * s_scale + s_shift
            qkv = F.linear(normed, self._self_qkv_weights[i])
            attended = block_self_attention(
                qkv.view(bsz, block_size, 3, self.num_heads, self.head_dim),
                *self._self_norm_weights[i],
                freqs,
                self.qk_norm_eps,
            )
            hidden = hidden + s_gate * layer.self_attn.o_proj(
                attended.reshape(bsz, block_size, self.dim)
            )
            # Cross-attention onto the paged context.
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * c_scale + c_shift
            attn = layer.cross_attn
            query = rms_norm_rope(
                attn.q_proj(normed).view(shape),
                self._cross_q_norm_weights[i],
                freqs,
                self.qk_norm_eps,
            )
            context = _paged_cross_attention(
                query, keys[i], values[i], cache_seqlens, page_table, schedule
            )
            hidden = hidden + c_gate * attn.o_proj(
                context.reshape(bsz, block_size, self.dim)
            )
            normed = F.layer_norm(hidden, (self.dim,), eps=eps) * m_scale + m_shift
            hidden = hidden + m_gate * layer.mlp.fc2(F.gelu(layer.mlp.fc1(normed)))
        return F.layer_norm(hidden, (self.dim,), eps=eps) * final_scale + final_shift


EntryClass = [FlashMapDraftModel]


# --- DSPARK worker hooks, for FlashMap drafts only -----------------------------

_dspark_init_cuda_graphs = dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs
_dspark_build_draft_sampler = dspark_worker_v2.DSparkWorkerV2._maybe_build_draft_sampler


def _no_attention_metadata(*args, **kwargs) -> None:
    return None


def _init_cuda_graphs(self):
    if getattr(self.draft_model, "flashmap", False):
        # FlashMap attends through its own paged cross-attention and never
        # reads the draft attention backend: skip the metadata it prepares
        # every step.
        backend = self.draft_model_runner.attn_backend
        assert backend is not self.model_runner.attn_backend
        for name in (
            "init_forward_metadata",
            "init_forward_metadata_capture_cuda_graph",
            "init_forward_metadata_replay_cuda_graph",
        ):
            setattr(backend, name, _no_attention_metadata)
    return _dspark_init_cuda_graphs(self)


def _maybe_build_draft_sampler(self, *, available_memory_gb: float):
    sampler = _dspark_build_draft_sampler(self, available_memory_gb=available_memory_gb)
    if sampler is None or not getattr(self.draft_model, "flashmap", False):
        return sampler
    max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)
    sampler = FlashMapDraftSampler(sampler, max_bs=max_bs, device=self.device)
    logger.info(
        "FlashMap draft sampler: one pass over the block (folded sampling %s)",
        sampler.folded_sampling,
    )
    return sampler


dspark_worker_v2.DSparkWorkerV2.init_cuda_graphs = _init_cuda_graphs
dspark_worker_v2.DSparkWorkerV2._maybe_build_draft_sampler = _maybe_build_draft_sampler
