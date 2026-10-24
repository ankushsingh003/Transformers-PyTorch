"""
Unit tests for Custom Training Loop components:
- Mathematical equivalence of Gradient Accumulation vs Full Batch
- Automatic Mixed Precision (AMP) execution
"""

import os
import sys
import unittest
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gpt import GPT, GPTConfig


class TestTrainingLoop(unittest.TestCase):

    def test_gradient_accumulation_mathematical_equivalence(self):
        r"""
        Verify that accumulating gradients across N micro-batches of size B
        produces identical gradients to a single full batch of size N * B.

        Mathematical condition:
            \nabla_W \left( \frac{1}{N} \sum_{i=1}^N \mathcal{L}_i \right)
            = \frac{1}{N} \sum_{i=1}^N \nabla_W \mathcal{L}_i
        """
        torch.manual_seed(1337)
        vocab_size = 50
        block_size = 16
        micro_batch_size = 4
        accum_steps = 3
        full_batch_size = micro_batch_size * accum_steps

        # Generate deterministic synthetic data
        full_inputs = torch.randint(0, vocab_size, (full_batch_size, block_size))
        full_targets = torch.randint(0, vocab_size, (full_batch_size, block_size))

        config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=block_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )

        # Base model
        model_full = GPT(config)
        # Exact clone for accumulation test
        model_accum = copy.deepcopy(model_full)

        # --- 1. Full Batch Reference Pass ---
        model_full.zero_grad(set_to_none=True)
        _, loss_full = model_full(full_inputs, full_targets)
        loss_full.backward()
        grads_full = {name: p.grad.clone() for name, p in model_full.named_parameters() if p.grad is not None}

        # --- 2. Micro-Batch Gradient Accumulation Pass ---
        model_accum.zero_grad(set_to_none=True)
        for i in range(accum_steps):
            start = i * micro_batch_size
            end = start + micro_batch_size
            mb_inputs = full_inputs[start:end]
            mb_targets = full_targets[start:end]

            _, mb_loss = model_accum(mb_inputs, mb_targets)
            scaled_loss = mb_loss / accum_steps
            scaled_loss.backward()

        grads_accum = {name: p.grad.clone() for name, p in model_accum.named_parameters() if p.grad is not None}

        # Compare gradients across all parameters
        for name in grads_full:
            diff = torch.max(torch.abs(grads_full[name] - grads_accum[name])).item()
            self.assertLess(
                diff,
                1e-5,
                f"Gradient mismatch for parameter '{name}': max diff = {diff}",
            )

    def test_amp_autocast_forward_backward(self):
        """Verify AMP autocasting runs forward and backward without errors."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16

        config = GPTConfig(
            vocab_size=30,
            max_seq_len=10,
            d_model=16,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        model = GPT(config).to(device)
        inputs = torch.randint(0, 30, (2, 10), device=device)
        targets = torch.randint(0, 30, (2, 10), device=device)

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            logits, loss = model(inputs, targets=targets)

        self.assertIsNotNone(loss)
        loss.backward()

        # Ensure parameters received gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
