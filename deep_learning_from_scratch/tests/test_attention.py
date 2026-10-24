"""
Unit tests for ScaledDotProductAttention and CustomMultiHeadAttention.
Verifies attention weights sum to 1.0, shape consistency, and strict causality mask invariance.
"""

import os
import sys
import unittest
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.attention import ScaledDotProductAttention, CustomMultiHeadAttention


class TestAttention(unittest.TestCase):

    def test_scaled_dot_product_shapes_and_probabilities(self):
        """Verify attention weights sum to 1.0 across key dimension."""
        torch.manual_seed(42)
        b, h, q_len, k_len, d_k = 2, 4, 8, 8, 16
        q = torch.randn(b, h, q_len, d_k)
        k = torch.randn(b, h, k_len, d_k)
        v = torch.randn(b, h, k_len, d_k)

        attn = ScaledDotProductAttention(dropout=0.0)
        context, weights = attn(q, k, v)

        self.assertEqual(context.shape, (b, h, q_len, d_k))
        self.assertEqual(weights.shape, (b, h, q_len, k_len))

        # Softmax sum should equal 1.0 along key dimension
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

    def test_multi_head_attention_shapes(self):
        """Verify CustomMultiHeadAttention preserves (B, T, D) shape."""
        torch.manual_seed(42)
        b, t, d_model, n_heads = 4, 12, 64, 4
        x = torch.randn(b, t, d_model)

        mha = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads)
        out, weights = mha(x)

        self.assertEqual(out.shape, (b, t, d_model))
        self.assertEqual(weights.shape, (b, n_heads, t, t))

    def test_strict_causality_masking(self):
        r"""
        Verify that in causal mode, token at position t cannot attend to or receive gradients
        from future positions t + k.

        If we compute the loss strictly from the output at position t=1:
            \mathcal{L} = \text{output}[:, 1, :].sum()
        Then \frac{\partial \mathcal{L}}{\partial x[:, t, :]} MUST be zero for all t > 1.
        """
        torch.manual_seed(42)
        b, t, d_model, n_heads = 2, 5, 32, 4

        # Input tensor requiring gradients
        x = torch.randn(b, t, d_model, requires_grad=True)

        mha = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        out, _ = mha(x, is_causal=True)

        # Loss depending ONLY on the 2nd token (index 1)
        loss = out[:, 1, :].sum()
        loss.backward()

        # Check gradients for position t=0 and t=1 (should have non-zero gradients)
        self.assertFalse(torch.all(x.grad[:, 0, :] == 0), "Past token should have non-zero gradient")
        self.assertFalse(torch.all(x.grad[:, 1, :] == 0), "Current token should have non-zero gradient")

        # Check gradients for future positions t=2, 3, 4 (MUST be strictly 0)
        future_grads = x.grad[:, 2:, :]
        max_future_grad = torch.max(torch.abs(future_grads)).item()
        self.assertEqual(
            max_future_grad,
            0.0,
            f"Causal leak detected! Future token gradient should be 0.0, got {max_future_grad}",
        )


if __name__ == "__main__":
    unittest.main()
