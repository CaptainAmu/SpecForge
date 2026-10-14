"""Pieces of the FlashMap sglang draft that run without a server: the
multi-jump noise bank (pure torch), the draft-vocabulary mapping of the eager
and folded samplers, and the fused sampling kernel (CUDA)."""

import unittest

import torch
import torch.nn.functional as F

from specforge.serving.sglang.flashmap_jump_bank import (
    JumpNoiseBank,
    jump_coefficients,
    token_normals,
)

try:
    from specforge.serving.sglang.flashmap_sampler import IdentityMarkov, _FullLogits
except Exception:  # sglang (0.5.20) not installed
    IdentityMarkov = None


class _Draft:
    """The slice of FlashMapDraftModel the bank reads, on a tiny vocabulary."""

    def __init__(self, vocab=40, hidden=8):
        generator = torch.Generator().manual_seed(0)
        self.embedding = torch.randn(vocab, hidden, generator=generator)
        self.gate = torch.ones(1)

    def state_norm(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=1e-6)

    noise_norm = state_norm

    def noise_gate_at(self, start, end):
        return self.gate * (end - start)


class TestJumpBank(unittest.TestCase):
    grid = [0.0, 0.8172569274902344, 1.0]

    def test_coefficients(self):
        (alpha0, betas0), (alpha1, betas1) = jump_coefficients(self.grid)
        self.assertEqual((alpha0, betas0), (1.0, []))
        self.assertAlmostEqual(alpha1, 1.0 - self.grid[1])
        self.assertEqual(len(betas1), 1)
        self.assertAlmostEqual(betas1[0], self.grid[1])

    def test_token_normals_are_a_function_of_seed_and_token(self):
        seeds = torch.arange(4096)
        ids = torch.randint(0, 151936, (4096, 16))
        first, again = token_normals(seeds, ids), token_normals(seeds, ids)
        self.assertTrue(torch.equal(first, again))
        self.assertLess(abs(first.mean().item()), 0.02)
        self.assertLess(abs(first.std().item() - 1.0), 0.02)

    def test_jump_input_is_exact_without_predictions(self):
        # With all-zero predictions the support terms vanish, so jump 1's input
        # must equal the literal embedding of x_1 = alpha_1 z over the vocabulary.
        draft = _Draft()
        bank = JumpNoiseBank(draft, self.grid, size=8, temperature=1.0, seed=3, chunk=4)
        generator = torch.Generator().manual_seed(3)
        z = torch.cat([torch.randn(4, 40, generator=generator) for _ in range(2)])
        rows = torch.tensor([0, 5, 7])
        seeds = torch.tensor([1, 2, 3])
        ids = torch.zeros(3, 4, dtype=torch.int64)
        supports = [(ids, torch.zeros(3, 4))]
        banked = bank.jump_input(draft, 1, rows, seeds, supports)
        t, (alpha, _) = self.grid[1], jump_coefficients(self.grid)[1]
        state = alpha * z[rows]
        posterior = (t / (1 - t) ** 2 * state).softmax(-1) @ draft.embedding
        literal = draft.state_norm(posterior) + draft.noise_gate_at(
            t, 1.0
        ) * draft.noise_norm(state @ draft.embedding)
        torch.testing.assert_close(banked, literal, rtol=1e-4, atol=1e-4)


@unittest.skipIf(IdentityMarkov is None, "needs sglang 0.5.20")
class TestDraftVocabulary(unittest.TestCase):
    hot_ids = torch.tensor([1, 4, 7])

    def test_eager_sampler_maps_tokens_and_scatters_logits(self):
        head = IdentityMarkov(vocab_size=10)
        head.hot_ids = self.hot_ids
        logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]])
        tokens, full = head.sample_block(
            logits,
            first_prev_tokens=None,
            hidden_states=None,
            sampler=lambda step_logits, step: step_logits.argmax(-1),
        )
        self.assertEqual(tokens.tolist(), [[4, 1]])
        self.assertEqual(full.shape, (1, 2, 10))
        self.assertTrue(torch.equal(full[..., self.hot_ids], logits))
        others = torch.ones(10, dtype=torch.bool)
        others[self.hot_ids] = False
        self.assertTrue(torch.isneginf(full[..., others]).all())

    def test_folded_logits_scatter_on_read(self):
        full = torch.zeros(4, 10)
        hot = torch.arange(12, dtype=torch.float32).view(4, 3)
        lazy = _FullLogits(full, hot, self.hot_ids)
        rows = lazy[:2]
        self.assertTrue(torch.equal(rows[:, self.hot_ids], hot[:2]))
        self.assertTrue(torch.isneginf(rows[:, 0]).all())
        self.assertTrue(torch.isneginf(full[2:]).all())  # rows past the read stay -inf


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestSampleRows(unittest.TestCase):
    def test_greedy_rows_argmax_and_sampled_rows_follow_softmax(self):
        from specforge.serving.sglang.flashmap_kernels import sample_rows

        rows, vocab = 4096, 5000
        logits = torch.randn(rows, vocab, device="cuda").to(torch.bfloat16)
        logits[:, :4] = torch.tensor([3.0, 2.0, 1.0, 0.0], device="cuda").to(
            torch.bfloat16
        )
        logits[:, 4:] -= 30  # four live tokens
        greedy = torch.zeros(rows, dtype=torch.bool, device="cuda")
        greedy[:64] = True
        temperatures = torch.full((rows,), 2.0, device="cuda")
        seed = torch.zeros((), dtype=torch.int64, device="cuda")
        tokens = sample_rows(logits, temperatures, greedy, seed)
        self.assertTrue(torch.equal(tokens[:64], logits[:64].float().argmax(-1)))
        counts = torch.bincount(tokens[64:], minlength=vocab)[:4].float()
        expected = torch.softmax(torch.tensor([3.0, 2.0, 1.0, 0.0]) / 2.0, 0)
        torch.testing.assert_close(
            counts.cpu() / counts.sum().cpu(), expected, atol=0.03, rtol=0
        )
        seed.add_(1)
        self.assertFalse(
            torch.equal(sample_rows(logits, temperatures, greedy, seed), tokens)
        )


if __name__ == "__main__":
    unittest.main()
