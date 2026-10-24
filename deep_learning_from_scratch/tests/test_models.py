"""
Unit tests for end-to-end model architectures:
- GPT Decoder-Only Causal Transformer
- TransformerSeq2Seq Encoder-Decoder
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gpt import GPT, GPTConfig
from models.seq2seq import TransformerSeq2Seq


class TestModels(unittest.TestCase):

    def test_gpt_forward_backward_and_generate(self):
        """Verify GPT forward loss, backward gradients, and autoregressive generation."""
        torch.manual_seed(42)
        vocab_size = 50
        max_seq_len = 16

        config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=32,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
        )
        model = GPT(config)

        # Forward with targets
        idx = torch.randint(0, vocab_size, (2, 8))
        targets = torch.randint(0, vocab_size, (2, 8))
        logits, loss = model(idx, targets=targets)

        self.assertEqual(logits.shape, (2, 8, vocab_size))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        # Backward
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Parameter {name} has None gradient")

        # Autoregressive generation
        prompt = torch.randint(0, vocab_size, (1, 4))
        generated = model.generate(prompt, max_new_tokens=6, temperature=1.0, greedy=False)
        self.assertEqual(generated.shape, (1, 10))

    def test_seq2seq_forward_and_generate(self):
        """Verify TransformerSeq2Seq forward loss and greedy generation."""
        torch.manual_seed(42)
        src_vocab = 20
        tgt_vocab = 20

        model = TransformerSeq2Seq(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=32,
            n_heads=2,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )

        src = torch.randint(1, src_vocab, (2, 6))
        tgt = torch.randint(1, tgt_vocab, (2, 5))

        logits = model(src, tgt)
        self.assertEqual(logits.shape, (2, 5, tgt_vocab))

        # Generation
        out = model.generate(src, start_symbol=1, end_symbol=2, max_len=8)
        self.assertEqual(out.shape[0], 2)
        self.assertGreaterEqual(out.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
