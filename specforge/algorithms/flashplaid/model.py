# coding=utf-8
"""FlashPlaid training wrapper: the NELBO of a conditional latent diffusion.

Stage ``teacher`` trains the draft as Plaid's variational diffusion model over
each anchor block's ``K`` draft tokens, conditioned on the target hidden
states before the anchor (FlashMap's context). A block's clean latent is
``e = E_h[y] (K, d)``; it is noised to ``z_t = alpha_t e + sigma_t eps`` and
the NELBO (Plaid / RePlaid eq. 4, App. B) is, per slot,

    prior + reconstruction + diffusion
    = KL(q(z_1 | y) || N(0, I)) - log <x(z_0), y> - 1/2 SNR'(t) |x(z_t) E_h - e|^2

with ``x = softmax(lm_head(DiT) + output prior)``. Each block is either a
reconstruction row (``t = 0``) or a diffusion row (low-discrepancy ``t``); the
two are importance-weighted by their batch shares, which Plaid adapts to the
ratio of the two terms' standard deviations. Every slot's share is weighted by
DFlash's slot decay ``w_k`` and pooled over tokens (``loss_terms``); the
unweighted NELBO is reported in bits per token. The schedule's endpoints
minimize the loss and its interior the loss variance (Plaid's gradient hook).

Options (all off by default): ``t1_ce_weight`` adds a cross-entropy at the
drafting state (``z ~ N(0, I)`` at ``gamma_1``) to the training loss;
``t_focus_start`` / ``t_focus_mass`` draw the diffusion times with density
biased to ``[t_focus_start, 1]`` and importance-weight them, so the NELBO
estimate stays unbiased; ``split_std="plaid"`` adapts the reconstruction share
from the standard deviations of the raw sub-batch mean losses, as Plaid's code
(``"scaled"``: of the per-block importance-scaled estimates). In evaluation the
wrapper also drafts every block in one step from ``N(0, I)`` and reports
per-position conditional accuracies (``acc_corrects`` / ``acc_denoms``), from
which the evaluator's ``simulated_acc_len`` is the expected number of
accepted draft tokens.

``ce_all_weight`` adds the cross-entropy of every diffusion row at its own t,
importance-weighted like the diffusion term (an average over t ~ U[0, 1]), with
no extra forward pass. ``objective="ce"`` replaces the prior and diffusion terms
by that cross-entropy (reconstruction rows unchanged); the NELBO is still
computed and reported. The schedule then gets no training signal and must be
frozen (FlashMap-like flow training on the latent path).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    build_block_labels,
    sample_anchor_positions,
    slot_decay_weights,
)
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.flashmap import create_context_mask
from specforge.modeling.draft.flashplaid import FlashPlaidDraftModel, alpha_sigma

STAGES = ("teacher",)
SPLIT_STD = ("scaled", "plaid")
OBJECTIVES = ("nelbo", "ce")
NUM_TIME_BINS = 4
LN2 = math.log(2.0)


def prior_loss(gamma_1: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Per-slot ``KL(N(alpha_1 e, sigma_1^2 I) || N(0, I))`` for unit-norm ``e``."""
    alpha_sq = torch.sigmoid(-gamma_1)
    sigma_sq = torch.sigmoid(gamma_1)
    return 0.5 * (alpha_sq + latent_dim * (sigma_sq - 1.0 - sigma_sq.log()))


def slot_nelbo_terms(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    e: torch.Tensor,
    latent: torch.Tensor,
    neg_snr_prime: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-slot reconstruction CE and diffusion loss of Plaid's NELBO.

    ``logits (..., V)`` are the full pre-softmax logits (head + output prior),
    ``e (..., d)`` the clean latents of ``target_ids (...)``, ``latent (V, d)``
    the unit-norm table and ``neg_snr_prime (...)`` ``-SNR'(t) >= 0``. Returns
    ``(recon, diffusion, correct)``, each ``(...)``: ``recon = -log x[y]``,
    ``diffusion = -1/2 SNR'(t) |x E_h - e|^2``, ``correct = argmax x == y``.
    """
    dtype = latent.dtype  # float32 in training
    log_probs = logits.to(dtype).log_softmax(dim=-1)
    recon = -log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    e_hat = log_probs.exp() @ latent
    diffusion = 0.5 * neg_snr_prime.to(dtype) * (e_hat - e.to(dtype)).pow(2).sum(dim=-1)
    with torch.no_grad():
        correct = log_probs.argmax(dim=-1) == target_ids
    return recon, diffusion, correct


class OnlineFlashPlaidModel(nn.Module):
    """FlashPlaid training forward; returns ``(loss, accuracy, metrics)`` like
    the DFlash-family wrappers."""

    def __init__(
        self,
        draft_model: FlashPlaidDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        *,
        stage: str = "teacher",
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 7.0,
        objective_chunk_blocks: int = 64,
        adaptive_recon: bool = True,
        recon_fraction: float = 0.125,
        std_ema: float = 0.997,
        t1_ce_weight: float = 0.0,
        t_focus_start: Optional[float] = None,
        t_focus_mass: Optional[float] = None,
        split_std: str = "scaled",
        simulate_acceptance: bool = True,
        objective: str = "nelbo",
        ce_all_weight: float = 0.0,
    ):
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        if attention_backend not in ("flex_attention", "sdpa"):
            raise ValueError("FlashPlaid supports flex_attention and sdpa attention")
        if not 0.0 < recon_fraction < 1.0:
            raise ValueError("recon_fraction must be in (0, 1)")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")
        if t1_ce_weight < 0:
            raise ValueError("t1_ce_weight must be >= 0")
        if (t_focus_start is None) != (t_focus_mass is None):
            raise ValueError("t_focus_start and t_focus_mass go together")
        if t_focus_start is not None and not (
            0.0 < t_focus_start < 1.0 and 0.0 < t_focus_mass < 1.0
        ):
            raise ValueError("t_focus_start and t_focus_mass must be in (0, 1)")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if ce_all_weight < 0:
            raise ValueError("ce_all_weight must be >= 0")
        if split_std not in SPLIT_STD:
            raise ValueError(f"split_std must be one of {SPLIT_STD}, got {split_std!r}")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.stage = stage
        self.block_size = int(draft_model.block_size)
        self.latent_dim = int(draft_model.latent_dim)
        self.attention_backend = attention_backend
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.adaptive_recon = bool(adaptive_recon)
        self.recon_fraction = float(recon_fraction)
        self.std_ema = float(std_ema)
        self.t1_ce_weight = float(t1_ce_weight)
        self.t_focus = None if t_focus_start is None else (float(t_focus_start), float(t_focus_mass))
        self.split_std = split_std
        self.simulate_acceptance = bool(simulate_acceptance)
        self.objective = objective
        self.ce_all_weight = float(ce_all_weight)
        # Running first/second moments of the per-block reconstruction and
        # diffusion estimates (Plaid's adaptive batch split); rank-local and
        # not checkpointed: a resume restarts them from recon_fraction.
        self._moments = None

    # ------------------------------------------------------------------
    # Pieces shared with FlashMap
    # ------------------------------------------------------------------

    def _sample_anchor_positions(self, seq_len, loss_mask, device, max_valid_anchors):
        return sample_anchor_positions(
            seq_len,
            loss_mask,
            device,
            num_anchors=self.num_anchors,
            max_valid_anchors=max_valid_anchors,
        )

    def _block_context(self, input_ids, hidden_states, anchor_positions):
        bsz, seq_len = input_ids.shape
        model = self.draft_model
        context = model.embed_context(hidden_states, self.embed_tokens(input_ids))
        positions = torch.arange(seq_len, device=input_ids.device).repeat(2)
        context_kv = model.project_context(context, positions.unsqueeze(0).expand(bsz, -1))
        context_mask = create_context_mask(
            anchor_positions,
            seq_len,
            self.block_size,
            use_flex=self.attention_backend == "flex_attention",
        )
        return context_kv, context_mask

    def _per_slot(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.repeat_interleave(self.block_size, dim=1)

    # ------------------------------------------------------------------
    # The batch split and times
    # ------------------------------------------------------------------

    def _current_recon_fraction(self) -> float:
        if not self.adaptive_recon or self._moments is None:
            return self.recon_fraction
        stds = []
        for mean, second, weight in self._moments:
            mean, second = mean / weight, second / weight
            stds.append(max(second - mean * mean, 0.0) ** 0.5)
        recon_std, diffusion_std = stds
        return recon_std / (1e-8 + recon_std + diffusion_std)

    def _update_moments(self, recon_values, diffusion_values) -> None:
        """``"scaled"``: running moments of the per-block values. ``"plaid"``:
        running moments of each sub-batch's mean (Plaid's ``reconst_ema`` /
        ``diffusion_ema``), averaged over ranks."""
        if not (self.training and self.adaptive_recon):
            return
        decay = self.std_ema
        if self._moments is None:
            self._moments = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        if self.split_std == "plaid":
            means = torch.stack(
                [values.double().mean() for values in (recon_values, diffusion_values)]
            )
            stats = torch.cat([means, means.pow(2)])
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
                stats = stats / dist.get_world_size()
            stats = stats.tolist()
            for index, moments in enumerate(self._moments):
                moments[0] = decay * moments[0] + (1 - decay) * stats[index]
                moments[1] = decay * moments[1] + (1 - decay) * stats[index + 2]
                moments[2] = decay * moments[2] + (1 - decay)
            return
        for moments, values in zip(self._moments, (recon_values, diffusion_values)):
            if values.numel() == 0:
                continue
            values = values.double()
            moments[0] = decay * moments[0] + (1 - decay) * float(values.mean())
            moments[1] = decay * moments[1] + (1 - decay) * float(values.pow(2).mean())
            moments[2] = decay * moments[2] + (1 - decay)

    def _focus_times(self, strata: torch.Tensor):
        """Map stratified uniforms to ``t`` with density ``(1 - m) / s`` on
        ``[0, s)`` and ``m / (1 - s)`` on ``[s, 1]``; returns ``(t, 1 / q(t))``."""
        start, mass = self.t_focus
        low = strata < 1.0 - mass
        times = torch.where(
            low,
            strata * start / (1.0 - mass),
            start + (strata - (1.0 - mass)) * (1.0 - start) / mass,
        )
        weights = torch.where(
            low,
            torch.full_like(strata, start / (1.0 - mass)),
            torch.full_like(strata, (1.0 - start) / mass),
        )
        return times, weights

    def _split_and_times(self, shape, device):
        """``(recon (B, N) bool, t (B, N) float64, importance (B, N) float64,
        recon share p)``. A uniform random subset of the blocks are
        reconstruction rows (at least one, and at least one diffusion row); the
        rest get stratified ``t ~ U[0, 1]`` (or, with ``t_focus``, stratified
        ``t ~ q`` and importance weights ``1 / q(t)``; 1 elsewhere)."""
        total = math.prod(shape)
        if total < 2:
            raise ValueError("FlashPlaid needs at least two blocks per batch")
        num_recon = int(round(total * self._current_recon_fraction()))
        num_recon = min(max(num_recon, 1), total - 1)
        order = torch.randperm(total, device=device)
        recon = torch.zeros(total, dtype=torch.bool, device=device)
        recon[order[:num_recon]] = True
        num_diffusion = total - num_recon
        strata = torch.rand(num_diffusion, dtype=torch.float64, device=device)
        strata = (strata + torch.arange(num_diffusion, device=device)) / num_diffusion
        strata = strata[torch.randperm(num_diffusion, device=device)]
        weights = torch.ones_like(strata)
        if self.t_focus is not None:
            strata, weights = self._focus_times(strata)
        times = torch.zeros(total, dtype=torch.float64, device=device)
        times[order[num_recon:]] = strata
        importance = torch.ones(total, dtype=torch.float64, device=device)
        importance[order[num_recon:]] = weights
        return recon.view(shape), times.view(shape), importance.view(shape), num_recon / total

    # ------------------------------------------------------------------
    # Stage 1: the teacher's NELBO
    # ------------------------------------------------------------------

    def _chunk_terms(
        self,
        hidden: torch.Tensor,
        z: torch.Tensor,
        gamma: torch.Tensor,
        neg_snr_prime: torch.Tensor,
        target_ids: torch.Tensor,
        weights: torch.Tensor,
        eval_mask: torch.Tensor,
        recon_rows: torch.Tensor,
        importance: torch.Tensor,
        time_bins: torch.Tensor,
        block_index: torch.Tensor,
        *,
        latent: torch.Tensor,
        prior: torch.Tensor,
        recon_scale: float,
        diffusion_scale: float,
        num_blocks: int,
    ) -> Tuple[torch.Tensor, ...]:
        # hidden (B, n, K, Ht); z (B, n, K, d); per-block tensors (B, n).
        bsz = hidden.shape[0]
        slot_gamma = gamma.unsqueeze(-1).expand(-1, -1, self.block_size)
        logits = self.lm_head(hidden).float()
        logits = logits + self.draft_model.output_prior_logits(z, slot_gamma, latent)
        e = latent[target_ids]
        recon, diffusion, correct = slot_nelbo_terms(
            logits,
            target_ids,
            e,
            latent,
            neg_snr_prime.unsqueeze(-1).float(),
        )
        mask = eval_mask.float()
        is_recon = recon_rows.unsqueeze(-1)
        # 1 / q(t) of a biased time draw (1 for uniform t and recon rows).
        weighted_diffusion = diffusion * importance.unsqueeze(-1).float()
        # Importance-weighted estimate of each slot's NELBO share.
        estimate = prior + torch.where(
            is_recon, recon * recon_scale, weighted_diffusion * diffusion_scale
        )
        # Cross-entropy of the diffusion rows at their own t (``recon`` holds
        # -log x[y] for every row), as an average over t ~ U[0, 1].
        diffusion_ce = torch.where(
            is_recon, 0.0, recon * importance.unsqueeze(-1).float() * diffusion_scale
        )
        if self.objective == "ce":
            objective = torch.where(is_recon, recon * recon_scale, diffusion_ce)
        else:
            objective = estimate + self.ce_all_weight * diffusion_ce
        weighted_num = (objective * weights).sum()
        with torch.no_grad():
            nelbo_num = (estimate * mask).sum()
            recon_num = (torch.where(is_recon, recon * recon_scale, 0.0) * mask).sum()
            diffusion_num = (
                torch.where(is_recon, 0.0, weighted_diffusion * diffusion_scale) * mask
            ).sum()
            ce_all_num = (diffusion_ce * mask).sum()
            diffusion_slots = mask * (~is_recon).float()
            diffusion_correct = (correct.float() * diffusion_slots).sum()
            bins = F.one_hot(time_bins, NUM_TIME_BINS).float().unsqueeze(-2)
            bin_den = (diffusion_slots.unsqueeze(-1) * bins).sum(dim=(0, 1, 2))
            bin_correct = ((correct.float() * diffusion_slots).unsqueeze(-1) * bins).sum(
                dim=(0, 1, 2)
            )
            bin_diffusion = (
                (diffusion * diffusion_slots).unsqueeze(-1) * bins
            ).sum(dim=(0, 1, 2))
        # Per-block weighted diffusion and reconstruction estimates, scattered
        # to their places so that summing over chunks yields all blocks.
        block_diffusion = torch.zeros(
            bsz, num_blocks, dtype=torch.float32, device=hidden.device
        )
        # The diffusion rows' term in the objective, for the variance hook and
        # the adaptive split (the cross-entropy when objective="ce").
        row_diffusion = (
            diffusion_ce / diffusion_scale if self.objective == "ce" else weighted_diffusion
        )
        block_diffusion.scatter_(
            1, block_index, (row_diffusion.detach() * weights).sum(dim=-1)
        )
        block_recon = torch.zeros_like(block_diffusion)
        block_recon.scatter_(1, block_index, (recon.detach() * weights).sum(dim=-1))
        return (
            weighted_num,
            nelbo_num,
            recon_num,
            diffusion_num,
            ce_all_num,
            diffusion_correct,
            diffusion_slots.sum(),
            bin_correct,
            bin_diffusion,
            bin_den,
            block_diffusion,
            block_recon,
        )

    def _teacher_forward(self, input_ids, hidden_states, anchor_positions, labels):
        target_ids, eval_mask, position_ids = labels
        bsz, num_blocks = anchor_positions.shape
        device = input_ids.device
        model = self.draft_model
        schedule = model.noise_schedule

        recon_rows, times, importance, recon_share = self._split_and_times(
            (bsz, num_blocks), device
        )
        # gamma'(t) by autograd, also under the evaluator's no_grad.
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            times = times.requires_grad_(True)
            gamma = schedule(times)
            gamma_prime = torch.autograd.grad(
                gamma.sum(), [times], create_graph=grad_enabled
            )[0]
        if not grad_enabled:
            gamma, gamma_prime = gamma.detach(), gamma_prime.detach()
        # Reconstruction rows sit at t = 0, where gamma = gamma_0 exactly.
        _, gamma_1 = schedule.bounds()

        # Plaid's variance hook: on diffusion rows the schedule's gradient is
        # scaled by 2 * (the block's diffusion loss), i.e. the interior
        # minimizes E[loss^2] (its variance, as the endpoints fix its mean).
        hook_state: Dict[str, torch.Tensor] = {}
        diffusion_rows = (~recon_rows).double()

        def variance_hook(grad: torch.Tensor) -> torch.Tensor:
            factor = 2.0 * hook_state["block_diffusion"].to(grad.dtype)
            return grad * (1.0 + diffusion_rows * (factor - 1.0))

        if gamma.requires_grad:
            gamma.register_hook(variance_hook)
            gamma_prime.register_hook(variance_hook)
        neg_snr_prime = gamma_prime * torch.exp(-gamma)  # -SNR'(t) >= 0

        latent = model.latent_weight()
        alpha, sigma = alpha_sigma(gamma.float())
        e = latent[target_ids]  # (B, N, K, d)
        noise = torch.randn_like(e)
        z = alpha[..., None, None] * e + sigma[..., None, None] * noise
        z_slots = z.reshape(bsz, num_blocks * self.block_size, self.latent_dim)
        context_kv, context_mask = self._block_context(
            input_ids, hidden_states, anchor_positions
        )
        hidden = model(
            z_slots,
            self._per_slot(gamma).float(),
            position_ids,
            context_kv,
            context_mask,
            latent=latent,
        )

        weights = slot_decay_weights(eval_mask, self.block_size, self.loss_decay_gamma)
        time_bins = (times.detach() * NUM_TIME_BINS).long().clamp(0, NUM_TIME_BINS - 1)
        block_index = torch.arange(num_blocks, device=device).expand(bsz, -1)
        (
            weighted_num,
            nelbo_num,
            recon_num,
            diffusion_num,
            ce_all_num,
            diffusion_correct,
            diffusion_den,
            bin_correct,
            bin_diffusion,
            bin_den,
            block_diffusion,
            block_recon,
        ) = checkpointed_chunk_reduce(
            self._chunk_terms_bound(
                latent=latent,
                prior=prior_loss(gamma_1, self.latent_dim).float(),
                recon_scale=1.0 / recon_share,
                diffusion_scale=1.0 / (1.0 - recon_share),
                num_blocks=num_blocks,
            ),
            hidden.reshape(bsz, num_blocks, self.block_size, -1),
            z,
            gamma,
            neg_snr_prime,
            target_ids,
            weights,
            eval_mask,
            recon_rows,
            importance,
            time_bins,
            block_index,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        hook_state["block_diffusion"] = block_diffusion.detach().double()
        if self.split_std == "plaid":
            self._update_moments(block_recon[recon_rows], block_diffusion[~recon_rows])
        else:
            self._update_moments(
                block_recon[recon_rows] / recon_share,
                block_diffusion[~recon_rows] / (1.0 - recon_share),
            )

        # The drafting state: every block from N(0, I) at gamma_1, for the
        # training cross-entropy and the evaluation's simulated acceptance.
        noise_end = None
        want_ce = self.training and self.t1_ce_weight > 0
        want_sim = (not self.training) and self.simulate_acceptance
        if want_ce or want_sim:
            noise_end = self._noise_end_terms(
                z_slots, gamma_1, position_ids, context_kv, context_mask, latent,
                target_ids, weights, eval_mask, bsz, num_blocks,
            )
            if want_ce:
                weighted_num = weighted_num + self.t1_ce_weight * noise_end[0]

        loss_den = weights.sum()
        token_den = eval_mask.float().sum()
        with torch.no_grad():
            gamma_0_value, gamma_1_value = schedule.bounds()
            one = torch.ones((), device=device)
            ratio_metrics = {
                "nelbo_bpt": (nelbo_num / LN2, token_den),
                "flashplaid/recon_bpt": (recon_num / LN2, token_den),
                "flashplaid/diffusion_bpt": (diffusion_num / LN2, token_den),
                "flashplaid/ce_all": (ce_all_num, token_den),
                "flashplaid/prior_bpt": (
                    prior_loss(gamma_1_value, self.latent_dim).float() / LN2 * one,
                    one,
                ),
                "acc": (diffusion_correct, diffusion_den),
                "flashplaid/gamma_0": (gamma_0_value.float() * one, one),
                "flashplaid/gamma_1": (gamma_1_value.float() * one, one),
                "flashplaid/recon_share": (recon_share * one, one),
                "flashplaid/latent_eff_rank": (self._effective_rank(latent) * one, one),
            }
            if noise_end is not None:
                ratio_metrics["flashplaid/ce_t1"] = (noise_end[0].detach(), loss_den)
            for index in range(NUM_TIME_BINS):
                ratio_metrics[f"flashplaid/acc@t{index}"] = (
                    bin_correct[index],
                    bin_den[index],
                )
                ratio_metrics[f"flashplaid/diffusion@t{index}"] = (
                    bin_diffusion[index],
                    bin_den[index],
                )
        extra = {}
        if want_sim:
            extra["acc_corrects"] = list(noise_end[1].double().unbind())
            extra["acc_denoms"] = list(noise_end[2].double().unbind())
        return weighted_num, loss_den, diffusion_correct, diffusion_den, ratio_metrics, extra

    def _noise_end_terms(
        self, z_like, gamma_1, position_ids, context_kv, context_mask, latent,
        target_ids, weights, eval_mask, bsz, num_blocks,
    ):
        """One-step drafts of every block from ``z ~ N(0, I)`` at ``gamma_1``:
        ``(sum_k w_k CE_k, conditional corrects (K,), conditional denominators
        (K,))``. Slot ``k`` counts in the denominator when slots ``< k`` were
        all correct and it is a label, and in the numerator when it is correct
        too, so the product of the ratios up to ``k`` is the probability that
        the first ``k`` drafts are accepted (against the data token)."""
        model = self.draft_model
        z = torch.randn_like(z_like)
        gamma = gamma_1.detach().float().expand(bsz, z.shape[1])
        hidden = model(z, gamma, position_ids, context_kv, context_mask, latent=latent)
        hidden = hidden.reshape(bsz, num_blocks, self.block_size, -1)
        z = z.reshape(bsz, num_blocks, self.block_size, self.latent_dim)

        def terms(hidden, z, target_ids, weights, eval_mask):
            slot_gamma = gamma_1.detach().float().expand(z.shape[:-1])
            logits = self.lm_head(hidden).float()
            logits = logits + model.output_prior_logits(z, slot_gamma, latent)
            log_probs = logits.log_softmax(dim=-1)
            ce = -log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                valid = eval_mask.float()
                correct = (log_probs.argmax(dim=-1) == target_ids).float() * valid
                before = torch.cumprod(
                    torch.cat([torch.ones_like(correct[..., :1]), correct[..., :-1]], dim=-1),
                    dim=-1,
                )
                corrects = (before * correct).sum(dim=(0, 1))
                denoms = (before * valid).sum(dim=(0, 1))
            return (ce * weights).sum(), corrects, denoms

        return checkpointed_chunk_reduce(
            terms, hidden, z, target_ids, weights, eval_mask,
            chunk_size=self.objective_chunk_blocks, dim=1,
        )

    def _chunk_terms_bound(self, **constants):
        def terms(*aligned):
            return self._chunk_terms(*aligned, **constants)

        return terms

    @staticmethod
    @torch.no_grad()
    def _effective_rank(latent: torch.Tensor) -> torch.Tensor:
        """Participation ratio of the embedding table's covariance spectrum."""
        centered = latent - latent.mean(dim=0, keepdim=True)
        eigenvalues = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0.0)
        return eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum().clamp_min(1e-12)

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
        del collect_detailed_metrics, selector_loss_alpha, target_last_hidden_states
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            input_ids.shape[1], loss_mask, input_ids.device, max_valid_anchors
        )
        target_ids, eval_mask, _ = build_block_labels(
            input_ids, loss_mask, anchor_positions, block_keep_mask, self.block_size
        )
        offsets = torch.arange(1, self.block_size + 1, device=input_ids.device)
        position_ids = (anchor_positions.unsqueeze(-1) + offsets).flatten(1)
        loss_num, loss_den, correct, eval_den, ratio_metrics, extra = self._teacher_forward(
            input_ids,
            hidden_states,
            anchor_positions,
            (target_ids, eval_mask, position_ids),
        )
        metrics: Dict[str, object] = {
            "accuracy_denom": eval_den.detach(),
            "ratio_metrics": ratio_metrics,
            "loss_terms": (loss_num, loss_den.detach()),
            **extra,
        }
        loss = loss_num / loss_den.clamp_min(torch.finfo(loss_den.dtype).tiny)
        accuracy = correct / eval_den.clamp_min(1.0)
        return loss, accuracy, metrics


__all__ = ["OnlineFlashPlaidModel", "STAGES", "prior_loss", "slot_nelbo_terms"]
