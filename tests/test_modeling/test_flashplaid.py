# coding=utf-8
"""FlashPlaid draft model, its NELBO training wrapper, and its registration."""

import math
import unittest

import torch
from torch import nn

from specforge.algorithms.flashplaid.model import (
    OnlineFlashPlaidModel,
    prior_loss,
    slot_nelbo_terms,
)
from specforge.algorithms.flashplaid.providers import (
    initialize_latent_from_target,
    tag_aux_parameters,
)
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft import resolve_draft
from specforge.modeling.draft.flashplaid import (
    FlashPlaidConfig,
    FlashPlaidDraftModel,
    PlaidNoiseSchedule,
    alpha_sigma,
)
from specforge.optimizer import BF16Optimizer

VOCAB = 64
TARGET_HIDDEN = 32
LATENT = 4


def tiny_config(**overrides) -> FlashPlaidConfig:
    options = dict(
        target_hidden_size=TARGET_HIDDEN,
        vocab_size=VOCAB,
        target_layer_ids=[0, 1],
        block_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        cond_dim=16,
        latent_dim=LATENT,
        schedule_hidden=16,
    )
    options.update(overrides)
    config = FlashPlaidConfig(**options)
    config.architectures = ["FlashPlaidDraftModel"]
    return config


def randomize(model: nn.Module) -> nn.Module:
    """Move every DiT weight off its init, so zero-initialized paths are live
    (the schedule keeps its normalized init)."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("noise_schedule."):
                parameter.normal_(std=0.05)
    return model


def wrapper(draft, **kwargs):
    torch.manual_seed(1)
    embed = nn.Embedding(VOCAB, TARGET_HIDDEN)
    head = nn.Linear(TARGET_HIDDEN, VOCAB, bias=False)
    embed.requires_grad_(False)
    head.requires_grad_(False)
    options = dict(
        attention_backend="sdpa",
        num_anchors=6,
        loss_decay_gamma=7.0,
        objective_chunk_blocks=2,
    )
    options.update(kwargs)
    return OnlineFlashPlaidModel(draft, head, embed, **options)


def batch(seq_len: int = 24):
    torch.manual_seed(2)
    input_ids = torch.randint(0, VOCAB, (2, seq_len))
    loss_mask = torch.ones(2, seq_len)
    loss_mask[0, :5] = 0
    loss_mask[1, -6:] = 0
    hidden_states = torch.randn(2, seq_len, 2 * TARGET_HIDDEN)
    return input_ids, hidden_states, loss_mask


class FlashPlaidModelTest(unittest.TestCase):
    def test_registered_and_built_from_config(self):
        self.assertIs(resolve_draft("FlashPlaidDraftModel"), FlashPlaidDraftModel)
        model = AutoDraftModel.from_config(tiny_config())
        self.assertIsInstance(model, FlashPlaidDraftModel)
        self.assertEqual((VOCAB, LATENT), tuple(model.latent_embed.weight.shape))
        self.assertAlmostEqual(float(model.noise_schedule.gamma_0), -3.0)
        self.assertAlmostEqual(float(model.noise_schedule.gamma_1), 6.0)

    def test_schedule_is_monotone_with_fixed_endpoints_and_autograd_derivative(self):
        torch.manual_seed(0)
        schedule = PlaidNoiseSchedule(64, -3.0, 6.0)
        t = torch.linspace(0, 1, 201, dtype=torch.float64)
        gamma = schedule(t)
        self.assertAlmostEqual(float(gamma[0]), -3.0, places=10)
        self.assertAlmostEqual(float(gamma[-1]), 6.0, places=10)
        self.assertTrue(torch.all(gamma.diff() > 0))
        times = t[1:-1].clone().requires_grad_(True)
        derivative = torch.autograd.grad(schedule(times).sum(), [times])[0]
        step = 1e-6
        finite = (schedule(times.detach() + step) - schedule(times.detach() - step)) / (
            2 * step
        )
        torch.testing.assert_close(derivative, finite, rtol=1e-5, atol=1e-6)

    def test_output_prior_is_the_gaussian_log_density_up_to_a_constant(self):
        torch.manual_seed(0)
        latent = torch.nn.functional.normalize(torch.randn(VOCAB, LATENT), dim=-1)
        z = torch.randn(3, LATENT)
        gamma = torch.tensor([-2.0, 0.5, 4.0])
        logits = FlashPlaidDraftModel.output_prior_logits(z, gamma, latent)
        alpha, sigma = alpha_sigma(gamma)
        log_density = -((z[:, None, :] - alpha[:, None, None] * latent) ** 2).sum(-1) / (
            2 * sigma[:, None] ** 2
        )
        torch.testing.assert_close(
            logits - logits[:, :1], log_density - log_density[:, :1], rtol=1e-4, atol=1e-4
        )

    def test_nelbo_upper_bounds_the_entropy_with_an_oracle_denoiser(self):
        """With the exact posterior as denoiser and a wide linear schedule, the
        expected NELBO of a single token is the entropy of its distribution up
        to the (tiny) endpoint terms: a check of every factor of the bound."""
        torch.manual_seed(0)
        vocab, dim, samples = 8, 8, 200_000
        gamma_0, gamma_1 = -10.0, 10.0
        latent = torch.nn.functional.normalize(torch.randn(vocab, dim, dtype=torch.float64), dim=-1)
        probs = torch.softmax(torch.randn(vocab, dtype=torch.float64), dim=-1)
        entropy = float(-(probs * probs.log()).sum())

        def oracle_terms(t):
            y = torch.multinomial(probs, samples, replacement=True)
            gamma = gamma_0 + (gamma_1 - gamma_0) * t
            alpha, sigma = alpha_sigma(gamma)
            e = latent[y]
            z = alpha[:, None] * e + sigma[:, None] * torch.randn(samples, dim, dtype=torch.float64)
            logits = probs.log() + FlashPlaidDraftModel.output_prior_logits(z, gamma, latent).double()
            neg_snr_prime = (gamma_1 - gamma_0) * torch.exp(-gamma)
            return slot_nelbo_terms(logits, y, e, latent, neg_snr_prime)

        strata = (torch.rand(samples, dtype=torch.float64) + torch.arange(samples)) / samples
        _, diffusion, _ = oracle_terms(strata)
        recon, _, _ = oracle_terms(torch.zeros(samples, dtype=torch.float64))
        prior = float(prior_loss(torch.tensor(gamma_1, dtype=torch.float64), dim))
        nelbo = prior + float(recon.mean()) + float(diffusion.mean())
        stderr = float(diffusion.std()) / math.sqrt(samples)
        self.assertGreater(nelbo, entropy - 4 * stderr)
        self.assertLess(nelbo, entropy + 4 * stderr + 0.02)

    def test_teacher_trains_with_token_pooled_loss_terms(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        model = wrapper(draft)
        loss, accuracy, metrics = model(*batch())
        numerator, denominator = metrics["loss_terms"]
        torch.testing.assert_close(loss, numerator / denominator)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)
        ratios = metrics["ratio_metrics"]
        for key in ("nelbo_bpt", "acc", "flashplaid/gamma_0", "flashplaid/latent_eff_rank"):
            self.assertIn(key, ratios)
        nelbo, tokens = ratios["nelbo_bpt"]
        parts = sum(
            ratios[f"flashplaid/{name}_bpt"][0] for name in ("recon", "diffusion")
        ) + ratios["flashplaid/prior_bpt"][0] * tokens
        torch.testing.assert_close(nelbo, parts, rtol=1e-4, atol=1e-4)
        numerator.backward()
        missing = [n for n, p in draft.named_parameters() if p.grad is None]
        self.assertEqual([], missing)

    def test_evaluation_runs_under_no_grad(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        model = wrapper(draft).eval()
        with torch.no_grad():
            loss, _, _ = model(*batch())
        self.assertTrue(torch.isfinite(loss))

    def test_padded_blocks_carry_no_weight(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        model = wrapper(draft, adaptive_recon=False)
        input_ids, hidden_states, loss_mask = batch()
        anchors = torch.tensor([[5, 9, 12], [2, 0, 0]])
        keep = torch.tensor([[True, True, True], [True, False, False]])
        results = []
        for padded in (0, 7):
            model._sample_anchor_positions = lambda *args, padded=padded, **kwargs: (
                torch.where(keep, anchors, padded),
                keep,
            )
            torch.manual_seed(5)
            _, _, metrics = model(input_ids, hidden_states, loss_mask)
            results.append(metrics["loss_terms"])
        torch.testing.assert_close(results[0][0], results[1][0])
        expected = 4 * float(torch.exp(-torch.arange(4.0) / 7.0).sum())
        self.assertAlmostEqual(float(results[0][1]), expected, places=4)

    def test_latent_table_starts_as_the_target_embedding_pca(self):
        torch.manual_seed(0)
        draft = FlashPlaidDraftModel(tiny_config())
        target = torch.randn(VOCAB, TARGET_HIDDEN) @ torch.diag(torch.linspace(3, 0.1, TARGET_HIDDEN))
        initialize_latent_from_target(draft, target)
        first = draft.latent_embed.weight.detach().clone()
        torch.testing.assert_close(first.norm(dim=-1), torch.ones(VOCAB), atol=1e-5, rtol=0)
        initialize_latent_from_target(draft, target)
        torch.testing.assert_close(first, draft.latent_embed.weight.detach())

    def test_target_projection_table_starts_as_the_pca_table_and_trains(self):
        torch.manual_seed(0)
        target = torch.randn(VOCAB, TARGET_HIDDEN) @ torch.diag(torch.linspace(3, 0.1, TARGET_HIDDEN))
        table = FlashPlaidDraftModel(tiny_config())
        initialize_latent_from_target(table, target)
        draft = FlashPlaidDraftModel(tiny_config(latent_param="target_projection"))
        self.assertFalse(hasattr(draft, "latent_embed"))
        self.assertEqual(("latent_proj", "noise_schedule"), draft._fp32_modules)
        with self.assertRaises(RuntimeError):
            draft.latent_weight()
        draft.set_target_embedding(target)
        initialize_latent_from_target(draft, target)
        torch.testing.assert_close(draft.latent_weight(), table.latent_weight(), atol=1e-5, rtol=0)
        self.assertNotIn("latent_proj.target_embedding", draft.state_dict())
        randomize(draft)
        model = wrapper(draft)
        loss, _, metrics = model(*batch())
        self.assertTrue(torch.isfinite(loss))
        metrics["loss_terms"][0].backward()
        missing = [n for n, p in draft.named_parameters() if p.grad is None]
        self.assertEqual([], missing)
        self.assertGreater(float(draft.latent_proj.weight.grad.abs().sum()), 0.0)

    def test_focused_times_are_importance_weighted_to_uniform(self):
        model = wrapper(FlashPlaidDraftModel(tiny_config()), t_focus_start=0.8, t_focus_mass=0.6)
        torch.manual_seed(0)
        recon, times, importance, _ = model._split_and_times((4, 500), torch.device("cpu"))
        diffusion = ~recon
        t, w = times[diffusion], importance[diffusion]
        self.assertTrue(torch.all(importance[recon] == 1))
        self.assertAlmostEqual(float((t >= 0.8).double().mean()), 0.6, places=2)
        # E_q[1 / q] = 1 and E_q[f / q] = E_U[f]: the estimate stays unbiased.
        self.assertAlmostEqual(float(w.mean()), 1.0, places=2)
        self.assertAlmostEqual(float((w * t).mean()), 0.5, places=2)

    def test_t1_cross_entropy_joins_the_training_loss(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        losses = []
        for weight in (0.0, 0.5):
            model = wrapper(draft, t1_ce_weight=weight)
            torch.manual_seed(7)
            loss, _, metrics = model(*batch())
            losses.append(float(loss))
        self.assertIn("flashplaid/ce_t1", metrics["ratio_metrics"])
        ce, den = metrics["ratio_metrics"]["flashplaid/ce_t1"]
        self.assertAlmostEqual(losses[1], losses[0] + 0.5 * float(ce / den), places=3)
        draft.zero_grad()
        metrics["loss_terms"][0].backward()
        self.assertEqual([], [n for n, p in draft.named_parameters() if p.grad is None])

    def test_evaluation_reports_conditional_per_position_accuracy(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        model = wrapper(draft).eval()
        with torch.no_grad():
            _, _, metrics = model(*batch())
        corrects = torch.stack(metrics["acc_corrects"])
        denoms = torch.stack(metrics["acc_denoms"])
        self.assertEqual((4,), tuple(corrects.shape))
        self.assertTrue(torch.all(corrects <= denoms))
        # Slot k + 1 is only counted after slot k was correct.
        self.assertTrue(torch.all(denoms[1:] <= corrects[:-1] + 1e-9))
        model.train()
        _, _, metrics = model(*batch())
        self.assertNotIn("acc_corrects", metrics)

    def test_plaid_split_tracks_raw_sub_batch_means(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        model = wrapper(draft, split_std="plaid")
        for _ in range(3):
            model(*batch())
        self.assertEqual(3, len(model._moments[0]))
        self.assertTrue(0.0 < model._current_recon_fraction() < 1.0)

    def test_cross_entropy_objectives(self):
        draft = randomize(FlashPlaidDraftModel(tiny_config()))
        results = {}
        for name, options in (
            ("nelbo", {}),
            ("ce_all", {"ce_all_weight": 0.5}),
            ("ce", {"objective": "ce"}),
        ):
            model = wrapper(draft, adaptive_recon=False, **options)
            torch.manual_seed(11)
            loss, _, metrics = model(*batch())
            results[name] = (float(loss), metrics)
        ratios = results["nelbo"][1]["ratio_metrics"]
        ce_all, tokens = ratios["flashplaid/ce_all"]
        _, weight_sum = results["nelbo"][1]["loss_terms"]
        # The NELBO is reported whatever the objective.
        torch.testing.assert_close(
            results["ce"][1]["ratio_metrics"]["nelbo_bpt"][0], ratios["nelbo_bpt"][0]
        )
        self.assertNotAlmostEqual(results["ce"][0], results["nelbo"][0], places=3)
        self.assertGreater(results["ce_all"][0], results["nelbo"][0])
        # objective="ce" trains no schedule: freeze it, every other weight learns.
        draft.zero_grad()
        draft.noise_schedule.requires_grad_(False)
        model = wrapper(draft, objective="ce")
        _, _, metrics = model(*batch())
        metrics["loss_terms"][0].backward()
        missing = [n for n, p in draft.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual([], missing)

    def test_aux_parameters_get_their_own_optimizer_groups(self):
        draft = FlashPlaidDraftModel(tiny_config())
        tag_aux_parameters(draft, 1e-2)
        optimizer = BF16Optimizer(
            draft,
            lr=3e-4,
            weight_decay=0.1,
            decay_matrices_only=True,
            total_steps=10,
            warmup_ratio=0.2,
            lr_scheduler="cosine",
        )
        groups = optimizer.optimizer.param_groups
        aux = {
            id(master)
            for parameter, master in zip(optimizer.model_params, optimizer.fp32_params)
            if hasattr(parameter, "_specforge_lr")
        }
        self.assertEqual(len(aux), len(draft.aux_parameters()))
        tagged = [g for g in groups if all(id(p) in aux for p in g["params"])]
        self.assertEqual({1e-3, 0.0}, {g["weight_decay"] for g in tagged})
        # Warmup scales the tagged groups like the main one.
        ratios = {g["lr"] / g["initial_lr"] for g in groups}
        self.assertEqual(1, len({round(r, 8) for r in ratios}))
        base = {round(g["initial_lr"], 10) for g in groups}
        self.assertEqual({3e-4, 1e-2}, base)


if __name__ == "__main__":
    unittest.main()
