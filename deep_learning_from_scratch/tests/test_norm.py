"""
Unit tests for CustomLayerNorm and CustomRMSNorm.
Verifies numerical parity with torch.nn.LayerNorm and gradient correctness.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.normalization import CustomLayerNorm, CustomRMSNorm


class TestNormalization(unittest.TestCase):

    def test_layernorm_forward_parity(self):
        """Verify CustomLayerNorm produces identical results to torch.nn.LayerNorm."""
        torch.manual_seed(42)
        dim = 64
        x = torch.randn(8, 16, dim)

        custom_ln = CustomLayerNorm(dim, eps=1e-5)
        torch_ln = nn.LayerNorm(dim, eps=1e-5)

        # Sync weights and biases
        with torch.no_grad():
            custom_ln.weight.copy_(torch_ln.weight)
            custom_ln.bias.copy_(torch_ln.bias)

        out_custom = custom_ln(x)
        out_torch = torch_ln(x)

        max_diff = torch.max(torch.abs(out_custom - out_torch)).item()
        self.assertLess(max_diff, 1e-5, f"LayerNorm forward mismatch: {max_diff}")

    def test_layernorm_backward_parity(self):
        """Verify backward gradients of CustomLayerNorm match torch.nn.LayerNorm."""
        torch.manual_seed(42)
        dim = 32
        x1 = torch.randn(4, 10, dim, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        custom_ln = CustomLayerNorm(dim, eps=1e-5)
        torch_ln = nn.LayerNorm(dim, eps=1e-5)

        with torch.no_grad():
            custom_ln.weight.copy_(torch_ln.weight)
            custom_ln.bias.copy_(torch_ln.bias)

        out1 = custom_ln(x1)
        out2 = torch_ln(x2)

        loss1 = out1.sum()
        loss2 = out2.sum()

        loss1.backward()
        loss2.backward()

        grad_diff = torch.max(torch.abs(x1.grad - x2.grad)).item()
        self.assertLess(grad_diff, 1e-5, f"LayerNorm gradient mismatch: {grad_diff}")

    def test_rmsnorm_invariance(self):
        """Verify CustomRMSNorm normalizes root-mean-square to 1."""
        torch.manual_seed(42)
        dim = 64
        x = torch.randn(4, 8, dim) * 5.0  # High variance input

        rms_norm = CustomRMSNorm(dim)
        # Set weight to 1
        with torch.no_grad():
            rms_norm.weight.fill_(1.0)

        out = rms_norm(x)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        # RMS should be approximately 1.0 everywhere
        max_deviation = torch.max(torch.abs(rms - 1.0)).item()
        self.assertLess(max_deviation, 1e-3, f"RMSNorm failed RMS invariance: {max_deviation}")


if __name__ == "__main__":
    unittest.main()
