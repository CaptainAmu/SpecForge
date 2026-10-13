# coding=utf-8
"""DSPARK's Markov head and draft sampler, specialized for FlashMap.

sglang's DSPARK worker requires a Markov head (a low-rank bias on each draft
position's logits from the previous token) and samples the block one position
at a time through it. FlashMap has no such head: its block positions are
independent given the noise. :class:`IdentityMarkov` stands in for the head
(rank 1, never loaded, contributes nothing) so DSPARK samples the map's logits
unchanged and verifies them with its usual rejection sampling.

With no bias, the positions are independent, yet DSPARK's graph-folded sampler
would still walk them one at a time (an exponential-noise fill and two
online-softmax kernels per position, or an argmax per position for greedy
batches). :class:`FlashMapDraftSampler` samples the whole block at once inside
the draft's CUDA graph: the logits written straight into DSPARK's buffer by one
GEMM against the folded head, then one argmax (greedy) or one
``SampleStepTokens`` pass with fresh noise per row and each request's
temperature, which is exactly DSPARK's per-position sampling since the rows are
independent; drafted indices of a restricted vocabulary are mapped back to
token ids in the graph too.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import SampleStepTokens
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite


class IdentityMarkov(VanillaMarkov):
    """A Markov head that leaves the draft's logits unchanged.

    With a restricted draft vocabulary (``hot_ids``, set by the draft) its
    eager block sampler also maps drafted indices back to token ids and
    scatters the logits back over the full vocabulary (-inf elsewhere, so
    rejection sampling stays exact)."""

    def __init__(self, *, vocab_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=1)
        for weight in self.parameters():
            weight.requires_grad_(False)
            torch.nn.init.zeros_(weight)
        self.hot_ids = None

    def apply_step_logits(self, logits, *args, **kwargs):
        return logits

    def apply_block_logits(self, base_logits, *args, **kwargs):
        return base_logits

    def sample_block_greedy_fused(self, *args, **kwargs):
        return None  # the caller falls back to sample_block

    def sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
        collect_corrected: bool = True,
    ):
        del first_prev_tokens, hidden_states
        tokens = torch.stack(
            [sampler(base_logits[:, i], i) for i in range(base_logits.shape[1])], dim=1
        )
        if self.hot_ids is None:
            return tokens, base_logits if collect_corrected else None
        logits = None
        if collect_corrected:
            logits = base_logits.new_full(
                (*base_logits.shape[:2], self.vocab_size), float("-inf")
            )
            logits.index_copy_(2, self.hot_ids, base_logits)
        return self.hot_ids[tokens], logits


class FlashMapDraftSampler(DsparkDraftSampler):
    """DSPARK's graph-folded draft sampler for a FlashMap draft: the whole block
    in one pass (see the module docstring). Same buffers and contract; its
    sampling logits stay over the full vocabulary, -inf outside a restricted
    draft vocabulary."""

    def __init__(self, sampler: DsparkDraftSampler, *, max_bs: int, device):
        self.__dict__.update(sampler.__dict__)
        assert self.sample_from_anchor and self.confidence_out is None
        if self.folded_sampling:
            draft_vocab = self.model.folded_head[0].shape[0]
            # Exponential noise for every row of the block, not one position's.
            self.row_noise = torch.empty(
                (int(max_bs) * self.gamma, draft_vocab),
                dtype=torch.float32,
                device=device,
            )
            self.exp_noise = None
            if self.model.hot_ids is not None:
                self.corrected_out.fill_(float("-inf"))

    def __call__(self, hidden_states, input_ids):
        del input_ids
        bs = hidden_states.shape[0] // self.query_token_num
        rows = bs * self.gamma
        weight, bias = self.model.folded_head
        hot_ids = self.model.hot_ids
        if self.folded_sampling:
            if hot_ids is None:
                logits = torch.addmm(
                    bias, hidden_states, weight.t(), out=self.corrected_out[:rows]
                )
            else:
                logits = torch.addmm(bias, hidden_states, weight.t())
                self.corrected_out[:rows].index_copy_(1, hot_ids, logits)
            tokens = SampleStepTokens.execute(
                step_logits=logits,
                temperatures=self.temperatures[:bs].repeat_interleave(self.gamma),
                greedy_mask=self.greedy_mask[:bs].repeat_interleave(self.gamma),
                exp_noise=self.row_noise[:rows].exponential_(),
            )
            tokens = self._tp_sync.sync(SpecTpSyncSite.DSPARK_GRAPH_SAMPLE, tokens)
        else:
            logits = torch.addmm(bias, hidden_states, weight.t())
            tokens = self._tp_sync.sync(
                SpecTpSyncSite.DSPARK_GRAPH_GREEDY, logits.argmax(dim=-1)
            )
        if hot_ids is not None:
            tokens = hot_ids[tokens]
        self.out[:rows].copy_(tokens.reshape(-1))


__all__ = ["FlashMapDraftSampler", "IdentityMarkov"]
