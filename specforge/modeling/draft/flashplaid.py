# coding=utf-8
"""FlashPlaid draft model: a block drafter that diffuses in a learned 16-d token space.

FlashPlaid drafts the ``block_size`` tokens after the anchor like FlashMap, but
its state is not a point on the vocabulary simplex: each draft slot is a
``latent_dim``-dimensional embedding (Plaid / RePlaid, arXiv:2605.18530),
``z_t = alpha_t e + sigma_t eps`` with ``e = E_h[y]`` a unit-norm row of a
learned embedding table. The network is a variational diffusion model trained
with Plaid's NELBO (``specforge/algorithms/flashplaid``); its learned noise
schedule ``gamma(t) = gamma_0 + (gamma_1 - gamma_0) gamma_tilde(t)`` belongs to
the draft so that it is optimized and checkpointed with it.

The DiT, its cross-attention onto the fused target hidden states before the
anchor, and the anchor-token context entry are FlashMap's. Only the state read
(three bias-free input projections of a ``latent_dim`` state, RePlaid's design
without self-conditioning) and the noise-level conditioning differ. Its output
is target-hidden-sized: the caller applies the frozen target lm_head and adds
Plaid's output-prior logits.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers import initialization as init

from .flashmap import (
    AttentionMask,
    ContextKV,
    FlashMapAdaLayerNorm,
    FlashMapDecoderLayer,
    FlashMapRMSNorm,
    rope_freqs,
)
from .registry import register_draft


class FlashPlaidConfig(PretrainedConfig):
    r"""Configuration of a FlashPlaid draft model.

    ``hidden_size`` is the DiT width and ``latent_dim`` the per-token diffusion
    space; ``target_hidden_size`` and ``vocab_size`` describe the target, whose
    embedding (for the anchor) and lm_head (for the logits) the draft borrows.
    ``gamma_0_init`` / ``gamma_1_init`` start the learned noise-schedule
    endpoints (Plaid's defaults), ``schedule_hidden`` is the width of its
    monotone interior network.
    """

    model_type = "flashplaid"

    def __init__(
        self,
        target_model: Optional[str] = None,
        target_hidden_size: int = 4096,
        vocab_size: int = 151936,
        target_layer_ids: Sequence[int] = (1, 9, 17, 25, 33),
        block_size: int = 16,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        cond_dim: int = 256,
        latent_dim: int = 16,
        gamma_0_init: float = -3.0,
        gamma_1_init: float = 6.0,
        schedule_hidden: int = 1024,
        posterior_input: bool = True,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm_eps: float = 1e-5,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if hidden_size % num_attention_heads:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_attention_heads {num_attention_heads}"
            )
        if not gamma_0_init < gamma_1_init:
            raise ValueError("gamma_0_init must be smaller than gamma_1_init")
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
        self.latent_dim = int(latent_dim)
        self.gamma_0_init = float(gamma_0_init)
        self.gamma_1_init = float(gamma_1_init)
        self.schedule_hidden = int(schedule_hidden)
        self.posterior_input = bool(posterior_input)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.qk_norm_eps = float(qk_norm_eps)
        self.layer_norm_eps = float(layer_norm_eps)
        self.initializer_range = float(initializer_range)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class PlaidNoiseSchedule(nn.Module):
    """Plaid's learned noise schedule ``gamma(t) = gamma_0 + (gamma_1 -
    gamma_0) gamma_tilde(t)``, evaluated in float64.

    ``gamma_tilde`` is a monotone 1 -> hidden -> 1 tanh network (softplus
    weights; the output layer scaled by 0.01) normalized to ``gamma_tilde(0) =
    0`` and ``gamma_tilde(1) = 1``, so the endpoints alone set the schedule's
    range (Plaid ``lib/models.py`` ``NoiseSchedule`` and ``GammaBounds``).
    """

    def __init__(self, hidden: int, gamma_0: float, gamma_1: float):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden, 1))
        self.b1 = nn.Parameter(torch.randn(hidden))
        self.W2 = nn.Parameter(torch.randn(1, hidden))
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))

    def _raw(self, t: torch.Tensor) -> torch.Tensor:
        w1 = F.softplus(self.W1.double())
        w2 = 0.01 * F.softplus(self.W2.double())
        h = torch.tanh((t.double().reshape(-1, 1) - 0.5) @ w1.T + self.b1.double())
        return (h @ w2.T).reshape(t.shape)

    def gamma_tilde(self, t: torch.Tensor) -> torch.Tensor:
        ends = self._raw(torch.tensor([0.0, 1.0], dtype=torch.float64, device=t.device))
        return (self._raw(t) - ends[0]) / (ends[1] - ends[0])

    def bounds(self):
        return self.gamma_0.double(), self.gamma_1.double()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        gamma_0, gamma_1 = self.bounds()
        return gamma_0 + (gamma_1 - gamma_0) * self.gamma_tilde(t)


class GammaEmbedding(nn.Module):
    """Noise-level conditioning: Plaid's sinusoidal features of ``gamma``
    (32 frequencies ``exp(linspace(-5, 5))``), then an MLP to ``cond_dim``."""

    def __init__(self, cond_dim: int, num_frequencies: int = 32):
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.linspace(-5.0, 5.0, num_frequencies).exp(),
            persistent=False,
        )
        self.linear_1 = nn.Linear(2 * num_frequencies, cond_dim)
        self.linear_2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        args = gamma.float()[..., None] * self.frequencies.float()
        features = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.linear_2(F.silu(self.linear_1(features.to(self.linear_1.weight.dtype))))


def alpha_sigma(gamma: torch.Tensor):
    """``(alpha, sigma)`` of the variance-preserving process at noise level
    ``gamma``: ``alpha^2 = sigmoid(-gamma)``, ``sigma^2 = sigmoid(gamma)``."""
    return torch.sigmoid(-gamma).sqrt(), torch.sigmoid(gamma).sqrt()


class FlashPlaidPreTrainedModel(PreTrainedModel):
    config_class = FlashPlaidConfig
    base_model_prefix = "flashplaid"
    # One FSDP unit, as for FlashMap: the context is projected once per step.
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if getattr(module, "_flashmap_zero_init", False):
                init.zeros_(module.weight)
            else:
                init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Rows are normalized when read; training starts from a PCA of the
            # target's embedding instead (flashplaid providers).
            init.normal_(module.weight, mean=0.0, std=1.0)
        elif isinstance(module, FlashMapRMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, PlaidNoiseSchedule):
            for parameter in (module.W1, module.b1, module.W2):
                init.normal_(parameter, mean=0.0, std=1.0)
            init.constant_(module.gamma_0, self.config.gamma_0_init)
            init.constant_(module.gamma_1, self.config.gamma_1_init)


@register_draft
class FlashPlaidDraftModel(FlashPlaidPreTrainedModel):
    """The FlashPlaid DiT, its latent embedding table and its noise schedule.

    ``forward`` maps a latent state ``z (B, T, latent_dim)`` at per-slot noise
    levels ``gamma (B, T)`` to target-hidden-sized outputs; the caller applies
    the target lm_head and :meth:`output_prior_logits`.
    """

    # Kept in float32 when the draft is cast to bf16 (see restore_fp32_modules).
    _fp32_modules = ("latent_embed", "noise_schedule")

    def __init__(self, config: FlashPlaidConfig) -> None:
        super().__init__(config)
        target_hidden = config.target_hidden_size
        dim = config.hidden_size
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.latent_dim = config.latent_dim
        self.head_dim = dim // config.num_attention_heads

        # FlashMap's context path (same weight names).
        self.fc = nn.Linear(len(self.target_layer_ids) * target_hidden, dim, bias=False)
        self.hidden_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)
        self.anchor_proj = nn.Linear(target_hidden, dim)
        self.anchor_norm = FlashMapRMSNorm(dim, config.rms_norm_eps)

        # The diffusion space: a learned table of unit-norm token embeddings
        # and the schedule that noises them.
        self.latent_embed = nn.Embedding(config.vocab_size, config.latent_dim)
        self.noise_schedule = PlaidNoiseSchedule(
            config.schedule_hidden, config.gamma_0_init, config.gamma_1_init
        )

        # State read: bias-free projections of the rescaled state and of its
        # posterior-mean embedding, plus noise-level conditioning via AdaLN.
        self.in_z = nn.Linear(config.latent_dim, dim, bias=False)
        if config.posterior_input:
            self.in_post = nn.Linear(config.latent_dim, dim, bias=False)
        self.gamma_embed = GammaEmbedding(config.cond_dim)
        self.layers = nn.ModuleList(
            [FlashMapDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_adaln = FlashMapAdaLayerNorm(dim, config.cond_dim, config.layer_norm_eps)
        self.head = nn.Linear(dim, target_hidden)
        self.head._flashmap_zero_init = True
        self.post_init()

    @staticmethod
    def may_be_missing_from_checkpoint(key: str) -> bool:
        """Every weight is trained, including the "*embed*" ones."""
        del key
        return False

    def restore_fp32_modules(self) -> "FlashPlaidDraftModel":
        """Return the embedding table and the noise schedule to float32 after
        the draft was cast to the training dtype."""
        for name in self._fp32_modules:
            getattr(self, name).float()
        return self

    def aux_parameters(self):
        """``{name: parameter}`` of the embedding table and the schedule, which
        train at their own learning rate (Plaid: 1e-2)."""
        return {
            f"{module}.{name}": parameter
            for module in self._fp32_modules
            for name, parameter in getattr(self, module).named_parameters()
        }

    def latent_weight(self) -> torch.Tensor:
        """The embedding table with unit-norm rows, ``(V, latent_dim)`` float32."""
        weight = self.latent_embed.weight.float()
        return weight / (weight.norm(dim=-1, keepdim=True) + 1e-8)

    def freqs(self, position_ids: torch.Tensor) -> torch.Tensor:
        return rope_freqs(position_ids, self.head_dim, self.config.rope_theta)

    def embed_context(
        self, target_hidden: torch.Tensor, anchor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """FlashMap's cross-attention context ``[fused hidden states, anchor
        embeddings]``, ``(B, 2 S, hidden_size)``."""
        hidden = self.hidden_norm(self.fc(target_hidden))
        anchors = self.anchor_norm(self.anchor_proj(anchor_embeddings))
        return torch.cat([hidden, anchors], dim=1)

    def project_context(self, context: torch.Tensor, position_ids: torch.Tensor) -> list:
        freqs = self.freqs(position_ids)
        return [layer.cross_attn.project_kv(context, freqs) for layer in self.layers]

    @staticmethod
    def output_prior_logits(
        z: torch.Tensor, gamma: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Plaid's output prior: ``log N(z; alpha e_v, sigma^2 I)`` up to a
        constant, ``(alpha / sigma^2) <z, e_v>`` for unit-norm ``e_v``.
        ``z (..., latent_dim)``, ``gamma (...)`` -> ``(..., V)`` in the dtype of
        ``latent`` (float32 in training)."""
        alpha, sigma = alpha_sigma(gamma.to(latent.dtype))
        return (alpha / sigma**2).unsqueeze(-1) * (z.to(latent.dtype) @ latent.T)

    def posterior_mean(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
        latent: torch.Tensor,
        chunk_slots: int = 2048,
    ) -> torch.Tensor:
        """``softmax_V(output prior) @ E_h``: the clean embedding a uniform token
        prior implies for ``z``, ``(B, T, latent_dim)``. Chunked over slots with
        activation checkpointing, so no ``(B, T, V)`` tensor is kept."""

        def chunk(z_chunk, gamma_chunk, latent):
            logits = self.output_prior_logits(z_chunk, gamma_chunk, latent)
            return logits.softmax(dim=-1) @ latent

        outputs = []
        for start in range(0, z.shape[1], chunk_slots):
            stop = start + chunk_slots
            args = (z[:, start:stop], gamma[:, start:stop], latent)
            if torch.is_grad_enabled() and latent.requires_grad:
                outputs.append(checkpoint(chunk, *args, use_reentrant=False))
            else:
                outputs.append(chunk(*args))
        return torch.cat(outputs, dim=1)

    def embed_state(
        self, z: torch.Tensor, gamma: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """DiT input for a latent state ``z (B, T, latent_dim)`` at ``gamma (B,
        T)``: Plaid's rescale ``z / sqrt(alpha^2 / d + sigma^2)`` (the state's
        per-coordinate scale) and, optionally, its posterior-mean embedding."""
        alpha, sigma = alpha_sigma(gamma.float())
        scale = (alpha**2 / self.latent_dim + sigma**2).sqrt().unsqueeze(-1)
        dtype = self.in_z.weight.dtype
        hidden = self.in_z((z.float() / scale).to(dtype))
        if self.config.posterior_input:
            mean = self.posterior_mean(z, gamma, latent)
            hidden = hidden + self.in_post((mean * math.sqrt(self.latent_dim)).to(dtype))
        return hidden

    def forward(
        self,
        z: torch.Tensor,
        gamma: torch.Tensor,
        position_ids: torch.Tensor,
        context_kv: ContextKV,
        context_mask: AttentionMask = None,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``z (B, T, latent_dim)``; ``gamma`` and ``position_ids`` are per slot,
        ``(B, T)``, and ``T`` is a whole number of blocks. Returns ``(B, T,
        target_hidden_size)``."""
        if latent is None:
            latent = self.latent_weight()
        cond = self.gamma_embed(gamma)
        freqs = self.freqs(position_ids)
        hidden = self.embed_state(z, gamma, latent)
        for layer, layer_kv in zip(self.layers, context_kv):
            hidden = layer(hidden, cond, freqs, self.block_size, layer_kv, context_mask)
        hidden, _ = self.final_adaln(hidden, cond)
        return self.head(hidden)


__all__ = [
    "FlashPlaidConfig",
    "FlashPlaidDraftModel",
    "GammaEmbedding",
    "PlaidNoiseSchedule",
    "alpha_sigma",
]
