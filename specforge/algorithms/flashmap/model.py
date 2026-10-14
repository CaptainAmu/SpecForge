# coding=utf-8
"""FlashMap training wrapper: the flow objective and its flow-map distillation.

Stage ``flow`` trains the DiT as the denoiser ``D_t(x) = E[x1 | I_t = x]`` of
the linear path ``I_t = (1 - t) z + t onehot(y)`` over each anchor block's 16
draft tokens: cross entropy against the clean tokens, every block noised to
its own time ``t(tau)``, ``tau ~ U[0, 1]`` (the decoding-error schedule).

Stage ``map`` distils that flow into the two-time denoiser ``delta_{s,t}``
with the semigroup objective of Flow Map Language Models (arXiv:2602.16813,
eq. 23 and Algorithm 7): half the blocks train the diagonal ``delta_{t,t}``
against the frozen flow's ``D_t``; the rest train a jump ``s -> t`` against
two half-jumps of the current map under stop-gradient, with a fraction of them
on the drafting jump ``(0, 1)``. The student starts as a copy of the flow plus
an end-time embedding with a zero output layer, so on the diagonal it starts as
the flow.

Anchor sampling, block labels and slot-decay weights are DFlash/DSpark's; the
target embedding and lm_head are frozen, as for every DFlash-family wrapper.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    build_block_labels,
    sample_anchor_positions,
    slot_decay_weights,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.flashmap import (
    DecodingErrorSchedule,
    FlashMapDraftModel,
    create_context_mask,
)

STAGES = ("flow", "map")
# Block categories of the semigroup objective's batch split.
DIAGONAL, BOUNDARY, OFF_DIAGONAL = 0, 1, 2
NUM_TAU_BINS = 4


class OnlineFlashMapModel(nn.Module):
    """FlashMap training forward for either stage; returns
    ``(loss, accuracy, metrics)`` like the DFlash-family wrappers."""

    def __init__(
        self,
        draft_model: FlashMapDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        *,
        stage: str = "flow",
        flow: Optional[FlashMapDraftModel] = None,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 7.0,
        objective_chunk_blocks: int = 128,
        diagonal_fraction: float = 0.5,
        boundary_prob: float = 0.0625,
        antithetic_sampling: bool = True,
    ):
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        if draft_model.flow_map != (stage == "map"):
            raise ValueError(
                f"stage {stage!r} trains a draft with flow_map={stage == 'map'}"
            )
        if stage == "map" and (flow is None or flow.flow_map):
            raise ValueError("stage 'map' distils a frozen flow (flow_map=False)")
        if attention_backend not in ("flex_attention", "sdpa"):
            raise ValueError("FlashMap supports flex_attention and sdpa attention")
        if not 0.0 <= diagonal_fraction <= 1.0:
            raise ValueError("diagonal_fraction must be in [0, 1]")
        if not 0.0 <= boundary_prob <= 1.0:
            raise ValueError("boundary_prob must be in [0, 1]")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.stage = stage
        self.block_size = int(draft_model.block_size)
        self.attention_backend = attention_backend
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.diagonal_fraction = float(diagonal_fraction)
        self.boundary_prob = float(boundary_prob)
        self.antithetic_sampling = bool(antithetic_sampling)
        self.vocab_size = int(draft_model.config.vocab_size)
        self.time_schedule = DecodingErrorSchedule(self.vocab_size)
        # The frozen flow is a training-time teacher, not part of what this run
        # trains or checkpoints: kept out of parameters() and state_dict().
        object.__setattr__(
            self, "_flow", None if flow is None else flow.requires_grad_(False).eval()
        )

    @property
    def flow(self) -> Optional[FlashMapDraftModel]:
        return self._flow

    # ------------------------------------------------------------------
    # Shared pieces
    # ------------------------------------------------------------------

    def _sample_anchor_positions(self, seq_len, loss_mask, device, max_valid_anchors):
        # DFlash's anchor sampler; a block's slots then draft the tokens after
        # its anchor, labelled and decay-weighted as in DSpark.
        return sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            num_anchors=self.num_anchors,
            max_valid_anchors=max_valid_anchors,
        )

    def _loss_weights(self, eval_mask: torch.Tensor) -> torch.Tensor:
        return slot_decay_weights(eval_mask, self.block_size, self.loss_decay_gamma)

    def _block_context(self, model, input_ids, hidden_states, anchor_positions):
        """Cross-attention keys/values of ``model`` over every position, and
        the mask restricting each block to what is known at its anchor."""
        bsz, seq_len = input_ids.shape
        context = model.embed_context(hidden_states, self.embed_tokens(input_ids))
        positions = torch.arange(seq_len, device=input_ids.device).repeat(2)
        context_kv = model.project_context(
            context, positions.unsqueeze(0).expand(bsz, -1)
        )
        context_mask = create_context_mask(
            anchor_positions,
            seq_len,
            self.block_size,
            use_flex=self.attention_backend == "flex_attention",
        )
        return context_kv, context_mask

    def _evaluate(self, model, state, start, end, context, position_ids):
        """``model``'s target-hidden-sized output for ``state`` at ``start``
        (jumping to ``end`` for a map); one network evaluation."""
        state_embedding = model.embed_state(state, start, end, self.embed_tokens.weight)
        return model(
            state_embedding,
            start,
            position_ids,
            context[0],
            context[1],
            end=end,
        )

    def _probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float().softmax(dim=-1)

    def _noise(self, shape, dtype, device) -> torch.Tensor:
        return torch.randn(*shape, self.vocab_size, dtype=dtype, device=device)

    def _interpolant(self, target_ids, slot_times, dtype) -> torch.Tensor:
        """``I_t = (1 - t) z + t onehot(y)`` at per-slot times ``(B, T)``."""
        bsz = target_ids.shape[0]
        flat_ids = target_ids.reshape(bsz, -1)
        with torch.no_grad():
            state = self._noise(flat_ids.shape, dtype, flat_ids.device)
            state *= (1.0 - slot_times).unsqueeze(-1).to(dtype)
            state.scatter_add_(
                -1, flat_ids.unsqueeze(-1), slot_times.unsqueeze(-1).to(dtype)
            )
        return state

    def _per_slot(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.repeat_interleave(self.block_size, dim=1)

    # ------------------------------------------------------------------
    # Stage 1: the flow
    # ------------------------------------------------------------------

    def _flow_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        tau_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        logits = self.lm_head(hidden)
        cross_entropy = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).view_as(loss_weights)
        with torch.no_grad():
            correct = (logits.argmax(dim=-1) == target_ids) & eval_mask
            # The loss is dominated by how noisy a block was, so it is also
            # reported per quartile of tau (0: pure noise, 3: nearly clean).
            bins = F.one_hot(tau_bins, NUM_TAU_BINS).float().unsqueeze(-2)
            bin_weights = loss_weights.unsqueeze(-1) * bins
            bin_loss = (cross_entropy.detach().unsqueeze(-1) * bin_weights).sum(
                dim=(0, 1, 2)
            )
        return (
            (cross_entropy * loss_weights).sum(),
            correct.float().sum(),
            eval_mask.float().sum(),
            bin_loss,
            bin_weights.sum(dim=(0, 1, 2)),
        )

    def _flow_forward(self, input_ids, hidden_states, anchor_positions, labels):
        target_ids, eval_mask, position_ids = labels
        bsz, num_blocks = anchor_positions.shape
        taus, times = self.time_schedule.sample((bsz, num_blocks), input_ids.device)
        slot_times = self._per_slot(times)
        state = self._interpolant(
            target_ids, slot_times, self.embed_tokens.weight.dtype
        )
        context = self._block_context(
            self.draft_model, input_ids, hidden_states, anchor_positions
        )
        hidden = self._evaluate(
            self.draft_model, state, slot_times, None, context, position_ids
        )
        del state
        loss_weights = self._loss_weights(eval_mask)
        tau_bins = (taus * NUM_TAU_BINS).long().clamp(0, NUM_TAU_BINS - 1)
        loss_num, correct, eval_den, bin_loss, bin_den = checkpointed_chunk_reduce(
            self._flow_chunk_terms,
            hidden.reshape(bsz, num_blocks, self.block_size, -1),
            target_ids,
            loss_weights,
            eval_mask,
            tau_bins,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        loss_den = loss_weights.sum()
        ratio_metrics = {
            "ce_loss": (loss_num.detach(), loss_den.detach()),
            "acc": (correct, eval_den),
        }
        for index in range(NUM_TAU_BINS):
            ratio_metrics[f"flashmap/ce_loss@tau{index}"] = (
                bin_loss[index],
                bin_den[index],
            )
        return loss_num, loss_den, correct, eval_den, ratio_metrics

    # ------------------------------------------------------------------
    # Stage 2: semigroup flow-map distillation
    # ------------------------------------------------------------------

    def _uniform(self, shape, device) -> torch.Tensor:
        """``U[0, 1)`` in float64, stratified over the batch with antithetic
        sampling (the official FMLM ``_sample_t_interval``)."""
        n = math.prod(shape)
        draws = torch.rand(n, dtype=torch.float64, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, dtype=torch.float64, device=device)
            draws = ((draws + offset) / n)[torch.randperm(n, device=device)]
        return draws.view(shape)

    def _split(self, shape, device) -> torch.Tensor:
        """Each block's category with exact counts placed at random: a
        ``diagonal_fraction`` diagonal and ``boundary_prob`` of the rest on the
        drafting jump ``(0, 1)``."""
        n = math.prod(shape)
        num_diagonal = int(n * self.diagonal_fraction)
        num_boundary = (
            int(
                n
                * (
                    self.diagonal_fraction
                    + (1.0 - self.diagonal_fraction) * self.boundary_prob
                )
            )
            - num_diagonal
        )
        kind = torch.full((n,), OFF_DIAGONAL, dtype=torch.long, device=device)
        kind[:num_diagonal] = DIAGONAL
        kind[num_diagonal : num_diagonal + num_boundary] = BOUNDARY
        return kind[torch.randperm(n, device=device)].view(shape)

    def _sample_map_times(self, shape, device):
        """Per-block category and times ``s <= u <= t`` (float64), drawn in the
        schedule's ``tau``: the diagonal at ``tau(s) = tau(t) ~ U[0, 1]``; jumps
        of ``h ~ U[0, 1]`` from ``tau(s) ~ U[0, 1 - h]``; ``tau(u)`` the
        midpoint."""
        kind = self._split(shape, device)
        diagonal, boundary = kind == DIAGONAL, kind == BOUNDARY
        tau_diagonal = self._uniform(shape, device)
        h = self._uniform(shape, device)
        tau_s_off = self._uniform(shape, device) * (1.0 - h)
        tau_t_off = tau_s_off + h
        zero, one = torch.zeros_like(h), torch.ones_like(h)
        tau_s = torch.where(
            diagonal, tau_diagonal, torch.where(boundary, zero, tau_s_off)
        )
        tau_t = torch.where(
            diagonal, tau_diagonal, torch.where(boundary, one, tau_t_off)
        )
        tau_s, tau_t = tau_s.clamp(0.0, 1.0), tau_t.clamp(0.0, 1.0)
        tau_u = 0.5 * (tau_s + tau_t)
        s, u, t = (self.time_schedule.t(tau).double() for tau in (tau_s, tau_u, tau_t))
        # The schedule's inverse stops just short of t = 1; the drafting jump
        # ends there exactly.
        s = torch.where(boundary, zero, s)
        t = torch.where(boundary, one, t)
        return kind, s, u, t

    @torch.no_grad()
    def _semigroup_target(
        self, state, kind, s, u, t, flow_context, context, position_ids
    ) -> torch.Tensor:
        """The frozen flow's ``D_s(I_s)`` on the diagonal; elsewhere the map's
        own two half-jumps ``gamma delta_{s,u} + (1 - gamma) delta_{u,t}``
        (eq. 87), with ``gamma = (1 - t)(u - s) / ((1 - u)(t - s))``."""
        s_slot, u_slot, t_slot = (self._per_slot(x).float() for x in (s, u, t))
        target = self._probabilities(
            self._evaluate(self.flow, state, s_slot, None, flow_context, position_ids)
        )
        delta_su = self._probabilities(
            self._evaluate(
                self.draft_model, state, s_slot, u_slot, context, position_ids
            )
        )
        # X_{s,u}(I_s) by eq. (19), with the official 1e-8 guard on 1 - s.
        coef_state = self._per_slot((1.0 - u) / (1.0 - s + 1e-8)).float().unsqueeze(-1)
        coef_delta = self._per_slot((u - s) / (1.0 - s + 1e-8)).float().unsqueeze(-1)
        x_su = coef_state * state + coef_delta * delta_su
        delta_ut = self._probabilities(
            self._evaluate(
                self.draft_model, x_su, u_slot, t_slot, context, position_ids
            )
        )
        del x_su
        # The guard also makes gamma = 0 on the diagonal and at t = 1.
        gamma = (1.0 - t) * (u - s) / ((1.0 - u) * (t - s) + 1e-8)
        semigroup = delta_ut.lerp_(
            delta_su, self._per_slot(gamma).float().unsqueeze(-1)
        )
        del delta_su
        on_diagonal = self._per_slot(kind == DIAGONAL).unsqueeze(-1)
        return torch.where(on_diagonal, target, semigroup)

    def _map_chunk_terms(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        target_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        eval_mask: torch.Tensor,
        groups: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        log_probs = self.lm_head(hidden).float().log_softmax(dim=-1)
        cross_entropy = -(target * log_probs).sum(dim=-1)
        with torch.no_grad():
            entropy = -(target * target.clamp_min(1e-9).log()).sum(dim=-1)
            kl = (cross_entropy.detach() - entropy) * loss_weights
            correct = ((log_probs.argmax(dim=-1) == target_ids) & eval_mask).float()
            group_weights = loss_weights.unsqueeze(-1) * groups.unsqueeze(-2)
            group_kl = (kl.unsqueeze(-1) * groups.unsqueeze(-2)).sum(dim=(0, 1, 2))
            group_mask = eval_mask.unsqueeze(-1) & groups.unsqueeze(-2)
        return (
            (cross_entropy * loss_weights).sum(),
            (entropy * loss_weights).sum(),
            correct.sum(),
            eval_mask.float().sum(),
            group_kl,
            group_weights.sum(dim=(0, 1, 2)),
            (correct.unsqueeze(-1) * group_mask).sum(dim=(0, 1, 2)),
            group_mask.float().sum(dim=(0, 1, 2)),
        )

    def _map_forward(self, input_ids, hidden_states, anchor_positions, labels):
        target_ids, eval_mask, position_ids = labels
        bsz, num_blocks = anchor_positions.shape
        # Categories and times are drawn for every anchor slot of the padded
        # batch, bsz * num_anchors, as DeepSpec does: a row with few valid
        # anchors still gets its share of drafting-jump blocks.
        kind, s, u, t = (
            draws[:, :num_blocks]
            for draws in self._sample_map_times(
                (bsz, max(self.num_anchors, num_blocks)), input_ids.device
            )
        )
        # One interpolant per block, at its start time.
        state = self._interpolant(target_ids, self._per_slot(s).float(), torch.float32)
        with torch.no_grad():
            flow_context = self._block_context(
                self.flow, input_ids, hidden_states, anchor_positions
            )
        context = self._block_context(
            self.draft_model, input_ids, hidden_states, anchor_positions
        )
        target = self._semigroup_target(
            state, kind, s, u, t, flow_context, context, position_ids
        )
        del flow_context
        # The one evaluation with gradients: delta_{s,t}(I_s) for every block.
        hidden = self._evaluate(
            self.draft_model,
            state,
            self._per_slot(s).float(),
            self._per_slot(t).float(),
            context,
            position_ids,
        )
        del state
        # As the official offdiag loss, "off" includes the boundary blocks; the
        # boundary is the jump drafting takes, so it is also reported alone.
        groups = torch.stack(
            [kind == DIAGONAL, kind != DIAGONAL, kind == BOUNDARY], dim=-1
        )
        loss_weights = self._loss_weights(eval_mask)
        shape = (bsz, num_blocks, self.block_size)
        (
            loss_num,
            entropy_num,
            correct,
            eval_den,
            group_kl,
            group_den,
            group_correct,
            group_count,
        ) = checkpointed_chunk_reduce(
            self._map_chunk_terms,
            hidden.reshape(*shape, -1),
            target.view(*shape, -1),
            target_ids,
            loss_weights,
            eval_mask,
            groups,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        loss_den = loss_weights.sum()
        ratio_metrics = {
            "map_loss": (loss_num.detach(), loss_den.detach()),
            "map_kl": ((loss_num - entropy_num).detach(), loss_den.detach()),
            "teacher_entropy": (entropy_num, loss_den.detach()),
            "acc": (correct, eval_den),
        }
        for index, name in enumerate(("diag", "off", "boundary")):
            ratio_metrics[f"flashmap/map_kl@{name}"] = (
                group_kl[index],
                group_den[index],
            )
            ratio_metrics[f"flashmap/acc@{name}"] = (
                group_correct[index],
                group_count[index],
            )
        return loss_num, loss_den, correct, eval_den, ratio_metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        max_valid_anchors: Optional[int] = None,
        collect_detailed_metrics: bool = True,
        selector_loss_alpha: Optional[float] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        # The DFlash step's selector weight and teacher states do not apply.
        del collect_detailed_metrics, selector_loss_alpha, target_last_hidden_states
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            input_ids.shape[1],
            loss_mask,
            input_ids.device,
            max_valid_anchors=max_valid_anchors,
        )
        target_ids, eval_mask, _ = build_block_labels(
            input_ids, loss_mask, anchor_positions, block_keep_mask, self.block_size
        )
        offsets = torch.arange(1, self.block_size + 1, device=input_ids.device)
        position_ids = (anchor_positions.unsqueeze(-1) + offsets).flatten(1)
        stage_forward = (
            self._flow_forward if self.stage == "flow" else self._map_forward
        )
        loss_num, loss_den, correct, eval_den, ratio_metrics = stage_forward(
            input_ids,
            hidden_states,
            anchor_positions,
            (target_ids, eval_mask, position_ids),
        )
        metrics: Dict[str, object] = {
            "accuracy_denom": eval_den.detach(),
            "ratio_metrics": ratio_metrics,
            "loss_terms": (loss_num, loss_den.detach()),
        }
        loss = loss_num / loss_den.clamp_min(torch.finfo(loss_den.dtype).tiny)
        accuracy = correct / eval_den.clamp_min(1.0)
        return loss, accuracy, metrics


__all__ = ["OnlineFlashMapModel", "STAGES"]
