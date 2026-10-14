# coding=utf-8
"""FlashMap draft model: a block-parallel drafter drawn from a flow map.

FlashMap drafts the ``block_size`` tokens after the last verified token (the
anchor) in one network evaluation: a DiT jumps a Gaussian noise state over the
target vocabulary straight to per-slot token distributions. The network is
trained in two stages (``specforge/algorithms/flashmap``): first as the
denoiser of a flow over one-hot tokens, ``I_t = (1 - t) z + t onehot(y)``, then
distilled into the flow map ``delta_{s,t}``, which also conditions on the time
it jumps to (Flow Map Language Models, arXiv:2602.16813, eqs. 18-19).

Like DFlash, a block conditions on the fused target hidden states strictly
before its anchor, here through cross-attention, plus the anchor token's own
embedding. The target's frozen embedding and lm_head are not draft weights:
the embedding featurizes the noisy state and the anchor, and the lm_head turns
the DiT output into logits. Weight names follow the released FlashMap
checkpoints, so a trained map exports to the serving layout unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import PretrainedConfig, PreTrainedModel
from transformers import initialization as init
from transformers.integrations.flex_attention import compile_friendly_flex_attention

from .flex_attention_backend import flex_attention_backend
from .registry import register_draft

AttentionMask = Union[BlockMask, torch.Tensor, None]
ContextKV = Sequence[Tuple[torch.Tensor, torch.Tensor]]

_TIME_SCHEDULES = ("decoding_error",)
_NOISE_GATES = ("jump",)


def decoding_error_rate(
    t: torch.Tensor, vocab_size: int, z_points: int = 2001, z_max: float = 10.0
) -> torch.Tensor:
    """``P_e(t)``: the chance the argmax of ``(1 - t) z + t e_y`` is not ``y``.

    Decoding is correct iff ``z_y + t / (1 - t)`` beats the largest of the other
    ``V - 1`` standard normals, so ``P_correct = E_z[Phi(z + c)^(V - 1)]`` with
    ``c = t / (1 - t)``, integrated by the trapezoid rule in float64.
    """
    t = t.to(torch.float64).clamp(0.0, 1.0)
    z = torch.linspace(-z_max, z_max, z_points, dtype=torch.float64, device=t.device)
    log_phi = -0.5 * z**2 - 0.5 * math.log(2 * math.pi)
    c = t / (1.0 - t).clamp_min(torch.finfo(torch.float64).tiny)
    c = torch.where(t >= 1.0, torch.full_like(c, float("inf")), c)
    log_integrand = (vocab_size - 1) * torch.special.log_ndtr(
        z[None, :] + c[:, None]
    ) + log_phi[None, :]
    p_correct = torch.trapezoid(log_integrand.exp(), z, dim=-1)
    return (1.0 - p_correct).clamp(0.0, 1.0)


class DecodingErrorSchedule:
    """The time reparameterization of Flow Map Language Models, eq. (25).

    ``tau(t) = (P_e(0) - P_e(t)) / P_e(0)`` makes equal steps in ``tau`` resolve
    equally many token identities. Training draws ``tau ~ U[0, 1]`` and maps it
    back through ``t(tau)``; drafting jumps over the grid ``t_n = t(n / N)``.
    Both maps are tabulated once and evaluated by linear interpolation.

    A plain object, not a module: casting the draft to bf16 would otherwise
    destroy the float64 tables.
    """

    def __init__(self, vocab_size: int, table_size: int = 4001):
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        t_table = torch.linspace(0.0, 1.0, table_size, dtype=torch.float64)
        p_e = decoding_error_rate(t_table, vocab_size)
        p_e0 = 1.0 - 1.0 / vocab_size
        tau_table = ((p_e0 - p_e) / p_e0).clamp(0.0, 1.0)
        tau_table[0], tau_table[-1] = 0.0, 1.0
        tau_table = torch.cummax(tau_table, dim=0).values
        self._tables = {torch.device("cpu"): (t_table, tau_table)}

    def _tables_on(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        if device not in self._tables:
            t_table, tau_table = self._tables[torch.device("cpu")]
            self._tables[device] = (t_table.to(device), tau_table.to(device))
        return self._tables[device]

    @staticmethod
    def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        x64 = x.to(torch.float64).clamp(xs[0], xs[-1])
        idx = torch.searchsorted(xs, x64, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        w = torch.where(x1 > x0, (x64 - x0) / (x1 - x0), torch.zeros_like(x64))
        return (y0 + w * (y1 - y0)).float()

    def t(self, tau: torch.Tensor) -> torch.Tensor:
        t_table, tau_table = self._tables_on(tau.device)
        return self._interp(tau, tau_table, t_table)

    def sample(self, shape, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(tau, t)`` with ``tau ~ U[0, 1]``, as float32."""
        tau = torch.rand(tuple(shape), dtype=torch.float64, device=device)
        return tau.float(), self.t(tau)

    def jump_grid(self, num_steps: int) -> list:
        """Jump times for ``num_steps`` jumps, the ends set to exactly 0 and 1.

        The inverse saturates just short of ``t = 1``, the endpoint a map is
        trained and drafts at.
        """
        tau = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
        grid = self.t(tau)
        grid[0], grid[-1] = 0.0, 1.0
        return grid.tolist()


class FlashMapConfig(PretrainedConfig):
    r"""Configuration of a FlashMap draft model.

    ``hidden_size`` is the DiT width; ``target_hidden_size`` and ``vocab_size``
    describe the target, whose embedding and lm_head the draft borrows.
    ``target_layer_ids`` index target decoder layers as DFlash does
    (``hidden_states[i + 1]`` of a transformers forward).

    ``flow_map`` selects the network: ``True`` is the two-time flow map that is
    served; ``False`` is the stage-1 flow it is distilled from, which has no
    end-time embedding and reads the state through its posterior alone.
    ``jump_grids`` holds the drafting times for 1 to 4 jumps.
    """

    model_type = "flashmap"

    def __init__(
        self,
        target_model: Optional[str] = None,
        target_hidden_size: int = 4096,
        vocab_size: int = 151936,
        target_layer_ids: Sequence[int] = (1, 9, 17, 25, 33),
        block_size: int = 16,
        hidden_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
        cond_dim: int = 256,
        time_embed_dim: int = 256,
        time_scale: float = 1000.0,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm_eps: float = 1e-5,
        layer_norm_eps: float = 1e-6,
        noise_gate: str = "jump",
        time_schedule: str = "decoding_error",
        num_steps: int = 1,
        jump_grids: Optional[dict] = None,
        flow_map: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if hidden_size % num_attention_heads:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_attention_heads {num_attention_heads}"
            )
        if noise_gate not in _NOISE_GATES:
            raise ValueError(f"noise_gate must be one of {_NOISE_GATES}")
        if time_schedule not in _TIME_SCHEDULES:
            raise ValueError(f"time_schedule must be one of {_TIME_SCHEDULES}")
        self.target_model = target_model
        self.target_hidden_size = int(target_hidden_size)
        self.vocab_size = int(vocab_size)
        self.target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.time_scale = float(time_scale)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.qk_norm_eps = float(qk_norm_eps)
        self.layer_norm_eps = float(layer_norm_eps)
        self.noise_gate = noise_gate
        self.time_schedule = time_schedule
        self.num_steps = int(num_steps)
        if jump_grids is None:
            schedule = DecodingErrorSchedule(self.vocab_size)
            jump_grids = {str(n): schedule.jump_grid(n) for n in range(1, 5)}
        self.jump_grids = {str(key): list(value) for key, value in jump_grids.items()}
        self.flow_map = bool(flow_map)
        self.initializer_range = float(initializer_range)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


def rope_freqs(position_ids: torch.Tensor, head_dim: int, base: float) -> torch.Tensor:
    """Position ids ``(B, T)`` -> rotary ``(cos, sin)`` table ``(B, T, head_dim // 2, 2)``."""
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim)
    )
    angles = position_ids.float()[..., None] * inv_freq
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved channel pairs of ``x (B, T, heads, head_dim)``."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs = freqs[:, :, None]
    rotated = torch.stack(
        [
            pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1],
            pairs[..., 1] * freqs[..., 0] + pairs[..., 0] * freqs[..., 1],
        ],
        dim=-1,
    )
    return rotated.flatten(3).type_as(x)


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[..., None] * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


def flex_kernel_options() -> dict:
    """Flex Attention kernel options: ``SPECFORGE_FLEX_ATTENTION_BACKEND`` when
    set, else the general Triton kernel, which DFlash training also pins
    because AUTO may route short query batches to flex-decoding, whose configs
    can be empty for a sparse block mask."""
    if torch.__version__ >= "2.11":
        return {"BACKEND": flex_attention_backend() or "TRITON"}
    return {"FORCE_USE_FLEX_ATTENTION": True}


def create_context_mask(
    anchor_positions: torch.Tensor,
    seq_len: int,
    block_size: int,
    use_flex: bool,
) -> AttentionMask:
    """Cross-attention mask from draft slots onto ``[hidden states, anchors]``.

    Keys ``[0, seq_len)`` are fused target hidden states and keys
    ``[seq_len, 2 * seq_len)`` token embeddings. A block anchored at ``a`` sees
    hidden states before ``a`` and the embedding of the token at ``a``: what is
    known when drafting after that token.
    """
    bsz, num_blocks = anchor_positions.shape
    if use_flex:
        # A tensor rather than the int, so torch.compile does not specialize
        # (and recompile) on every sequence length.
        offset = torch.full(
            (), seq_len, dtype=anchor_positions.dtype, device=anchor_positions.device
        )

        def mask_mod(b, h, q_idx, kv_idx):
            del h
            anchor = anchor_positions[b, q_idx // block_size]
            return (kv_idx < anchor) | (kv_idx == anchor + offset)

        kwargs = {}
        if flex_attention_backend() == "FLASH":
            # FLASH requires a minimum of this block size.
            kwargs["BLOCK_SIZE"] = (256, 128)
        return create_block_mask(
            mask_mod,
            B=bsz,
            H=None,
            Q_LEN=num_blocks * block_size,
            KV_LEN=2 * seq_len,
            device=anchor_positions.device,
            **kwargs,
        )
    anchor = anchor_positions.repeat_interleave(block_size, dim=1).unsqueeze(-1)
    kv_idx = torch.arange(2 * seq_len, device=anchor_positions.device).view(1, 1, -1)
    return ((kv_idx < anchor) | (kv_idx == anchor + seq_len)).unsqueeze(1)


class FlashMapRMSNorm(nn.Module):
    """RMSNorm whose statistics and gain are applied in float32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        scale = self.weight.float() * torch.rsqrt(variance + self.eps)
        return x * scale.to(x.dtype)


class FlashMapTimestepEmbedding(nn.Module):
    def __init__(self, cond_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, cond_dim)
        self.linear_2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        embedding = timestep_embedding(t, self.embed_dim)
        return self.linear_2(
            F.silu(self.linear_1(embedding.to(self.linear_1.weight.dtype)))
        )


class FlashMapAdaLayerNorm(nn.Module):
    """AdaLN-zero: a non-affine LayerNorm modulated by the time conditioning,
    returning the modulated input and the residual gate."""

    def __init__(self, dim: int, cond_dim: int, eps: float):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.linear = nn.Linear(cond_dim, 3 * dim)
        self.linear._flashmap_zero_init = True

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.linear(F.silu(cond)).chunk(3, dim=-1)
        x = F.layer_norm(x, (self.dim,), eps=self.eps)
        return x * (1 + scale) + shift, gate


class FlashMapAttention(nn.Module):
    """QK-normed rotary attention. Keys and values come from ``project_kv``, so
    cross-attention context is projected once and shared by every evaluation."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)
        self.k_norm = FlashMapRMSNorm(self.head_dim, config.qk_norm_eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:2], self.num_heads, self.head_dim)

    def project_kv(
        self, x: torch.Tensor, freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = apply_rotary_emb(self.k_norm(self._heads(self.k_proj(x))), freqs)
        values = self._heads(self.v_proj(x))
        return keys.transpose(1, 2), values.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: AttentionMask = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        query = apply_rotary_emb(self.q_norm(self._heads(self.q_proj(x))), freqs)
        query = query.transpose(1, 2)
        if isinstance(mask, BlockMask):
            output = compile_friendly_flex_attention(
                query,
                keys,
                values,
                block_mask=mask,
                kernel_options=flex_kernel_options(),
            )
        else:
            output = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        return self.o_proj(output.transpose(1, 2).reshape(bsz, seq_len, -1))


class FlashMapMLP(nn.Module):
    def __init__(self, config: FlashMapConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class FlashMapDecoderLayer(nn.Module):
    """Self-attention within each draft block, cross-attention onto the context,
    and an MLP, each modulated by AdaLN-zero on the time conditioning."""

    def __init__(self, config: FlashMapConfig):
        super().__init__()
        dim, cond_dim, eps = config.hidden_size, config.cond_dim, config.layer_norm_eps
        self.self_attn_adaln = FlashMapAdaLayerNorm(dim, cond_dim, eps)
        self.self_attn = FlashMapAttention(config)
        self.cross_attn_adaln = FlashMapAdaLayerNorm(dim, cond_dim, eps)
        self.cross_attn = FlashMapAttention(config)
        self.mlp_adaln = FlashMapAdaLayerNorm(dim, cond_dim, eps)
        self.mlp = FlashMapMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        freqs: torch.Tensor,
        block_size: int,
        context_kv: Tuple[torch.Tensor, torch.Tensor],
        context_mask: AttentionMask = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        # A block attends only to itself, so self-attention folds the blocks
        # into the batch instead of running under a mask.
        h, gate = self.self_attn_adaln(x, cond)
        h = h.reshape(-1, block_size, dim)
        block_freqs = freqs.reshape(-1, block_size, *freqs.shape[2:])
        keys, values = self.self_attn.project_kv(h, block_freqs)
        h = self.self_attn(h, block_freqs, keys, values)
        x = x + gate * h.reshape(bsz, seq_len, dim)

        h, gate = self.cross_attn_adaln(x, cond)
        x = x + gate * self.cross_attn(h, freqs, *context_kv, mask=context_mask)

        h, gate = self.mlp_adaln(x, cond)
        return x + gate * self.mlp(h)


class FlashMapPreTrainedModel(PreTrainedModel):
    config_class = FlashMapConfig
    base_model_prefix = "flashmap"
    # One FSDP unit for the whole DiT: the cross-attention context is projected
    # once per step (``project_context``, outside the layers' forward) and read
    # by every evaluation, so a layer must not be a sharded unit of its own.
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        # transformers' init functions skip tensors loaded from a checkpoint.
        if isinstance(module, nn.Linear):
            # AdaLN modulations, the output head and the end-time embedding
            # start at zero: every block starts as the identity, the draft as
            # the uniform distribution, and a map as the flow it came from.
            if getattr(module, "_flashmap_torch_init", False):
                # nn.Linear's own init, which the end-time embedding's first
                # layer keeps in the reference (it is added after the rest of
                # the network is initialized).
                init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    bound = 1 / math.sqrt(module.in_features)
                    init.uniform_(module.bias, -bound, bound)
                return
            if getattr(module, "_flashmap_zero_init", False):
                init.zeros_(module.weight)
            else:
                init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, FlashMapRMSNorm):
            init.ones_(module.weight)


@register_draft
class FlashMapDraftModel(FlashMapPreTrainedModel):
    """The FlashMap DiT (or, with ``config.flow_map=False``, its stage-1 flow).

    ``forward`` maps a featurized state at start times ``s`` (and, for a map,
    end times ``t``) to target-hidden-sized outputs; the caller applies the
    target lm_head. ``embed_state`` and ``embed_context`` take the target
    embedding explicitly because it is not part of the draft.
    """

    def __init__(self, config: FlashMapConfig) -> None:
        super().__init__(config)
        target_hidden = config.target_hidden_size
        dim = config.hidden_size
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.flow_map = config.flow_map
        self.head_dim = dim // config.num_attention_heads

        self.fc = nn.Linear(len(self.target_layer_ids) * target_hidden, dim, bias=False)
        self.hidden_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)
        self.anchor_proj = nn.Linear(target_hidden, dim)
        self.anchor_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)
        # The frozen target embedding has an arbitrary scale, so the embedded
        # state is normalized before the DiT reads it.
        self.state_norm = FlashMapRMSNorm(target_hidden, config.rms_norm_eps)
        if self.flow_map:
            # At s = 0 the posterior is uniform whatever the noise, so a map
            # jumping from there sees its noise only through a linear read-out,
            # weighted by noise_gate * (t - s): zero on the diagonal, where the
            # map is its flow's denoiser. The gate is fixed at 1.
            self.noise_norm = FlashMapRMSNorm(target_hidden, config.rms_norm_eps)
            self.register_buffer("noise_gate", torch.ones(1))
        self.x_embed = nn.Linear(target_hidden, dim)
        self.time_embed = FlashMapTimestepEmbedding(
            config.cond_dim, config.time_embed_dim
        )
        if self.flow_map:
            self.jump_embed = FlashMapTimestepEmbedding(
                config.cond_dim, config.time_embed_dim
            )
            self.jump_embed.linear_1._flashmap_torch_init = True
            self.jump_embed.linear_2._flashmap_zero_init = True
        self.layers = nn.ModuleList(
            [FlashMapDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_adaln = FlashMapAdaLayerNorm(
            dim, config.cond_dim, config.layer_norm_eps
        )
        self.head = nn.Linear(dim, target_hidden)
        self.head._flashmap_zero_init = True
        self.post_init()

    @staticmethod
    def adapt_config_to_checkpoint(config: FlashMapConfig, state_dict) -> None:
        """A flow and its map share one draft config; the weights tell which a
        checkpoint holds (only a map has the end-time embedding)."""
        config.flow_map = any(key.startswith("jump_embed.") for key in state_dict)

    def freqs(self, position_ids: torch.Tensor) -> torch.Tensor:
        return rope_freqs(position_ids, self.head_dim, self.config.rope_theta)

    def embed_state(
        self,
        state: torch.Tensor,
        start: torch.Tensor,
        end: Optional[torch.Tensor],
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Featurize a state ``(B, T, V)`` at per-slot times ``(B, T)``.

        The state is read through ``softmax(s x / (1 - s)^2)``, the exact
        posterior over the clean token under a uniform prior, embedded by the
        target embedding. A map jumping to ``end`` adds the gated linear
        read-out of the state itself.
        """
        one_minus_s = (1.0 - start).clamp_min(1e-3)
        evidence = (start / one_minus_s**2).unsqueeze(-1)
        posterior = (state.float() * evidence).softmax(dim=-1)
        embedded = self.state_norm(posterior.to(embed_weight.dtype) @ embed_weight)
        if self.flow_map and end is not None:
            gate = (self.noise_gate * (end - start)).unsqueeze(-1)
            noise = self.noise_norm(state.to(embed_weight.dtype) @ embed_weight)
            embedded = embedded + gate.to(embedded.dtype) * noise
        return embedded

    def embed_context(
        self, target_hidden: torch.Tensor, anchor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention context ``[fused hidden states, anchor embeddings]``
        for every position, ``(B, 2 S, hidden_size)``."""
        hidden = self.hidden_norm(self.fc(target_hidden))
        anchors = self.anchor_norm(self.anchor_proj(anchor_embeddings))
        return torch.cat([hidden, anchors], dim=1)

    def project_context(
        self, context: torch.Tensor, position_ids: torch.Tensor
    ) -> list:
        """Per-layer cross-attention keys and values of the context."""
        freqs = self.freqs(position_ids)
        return [layer.cross_attn.project_kv(context, freqs) for layer in self.layers]

    def forward(
        self,
        state_embedding: torch.Tensor,
        start: torch.Tensor,
        position_ids: torch.Tensor,
        context_kv: ContextKV,
        context_mask: AttentionMask = None,
        end: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``start``, ``end`` and ``position_ids`` are per slot, ``(B, T)``;
        ``T`` is a whole number of blocks."""
        cond = self.time_embed(start * self.config.time_scale)
        if end is not None:
            if not self.flow_map:
                raise ValueError("a flow (flow_map=False) takes no end time")
            cond = cond + self.jump_embed(end * self.config.time_scale)
        freqs = self.freqs(position_ids)
        hidden = self.x_embed(state_embedding)
        for layer, layer_kv in zip(self.layers, context_kv):
            hidden = layer(hidden, cond, freqs, self.block_size, layer_kv, context_mask)
        hidden, _ = self.final_adaln(hidden, cond)
        return self.head(hidden)


__all__ = [
    "DecodingErrorSchedule",
    "FlashMapConfig",
    "FlashMapDraftModel",
    "create_context_mask",
    "decoding_error_rate",
]
