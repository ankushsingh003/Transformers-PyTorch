"""
Encoder-Decoder Sequence-to-Sequence Transformer from 'Attention Is All You Need' (Vaswani et al., 2017).
Implemented from scratch using custom sub-modules.
"""

# Import typing utilities for optional values and tuple types
from typing import Optional, Tuple
# Import math module for sqrt scaling
import math
# Import PyTorch core library
import torch
# Import PyTorch neural network classes
import torch.nn as nn
# Import PyTorch functional operations
import torch.nn.functional as F

# Import our custom normalization layer
from .normalization import CustomLayerNorm
# Import our custom sinusoidal positional encoding
from .embeddings import SinusoidalPositionalEncoding
# Import our custom transformer block
from .transformer_block import TransformerBlock


# Define the full Encoder-Decoder Transformer architecture
class TransformerSeq2Seq(nn.Module):
    r"""
    The original Transformer Encoder-Decoder architecture (Vaswani et al., 2017).

    Encoder:
        - Source Token Embeddings + Positional Encodings
        - N x Encoder Blocks (Self-Attention + FFN)
    Decoder:
        - Target Token Embeddings + Positional Encodings
        - N x Decoder Blocks (Causal Self-Attention + Cross-Attention + FFN)
    Output:
        - Projection to target vocabulary
    """

    # Constructor initializing Encoder and Decoder stacks
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 1000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        # Call base nn.Module constructor
        super().__init__()
        # Store model hidden dimension
        self.d_model = d_model
        # Store padding token index (ignored during attention and loss)
        self.pad_idx = pad_idx

        # Source language/token embedding table
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        # Target language/token embedding table
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        # Sinusoidal positional encoding shared across encoder and decoder
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # Encoder Stack: List of N TransformerBlocks without cross-attention
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                pre_ln=True,
                has_cross_attention=False,
            )
            for _ in range(n_encoder_layers)
        ])
        # Layer normalization applied to final encoder representations
        self.enc_ln = CustomLayerNorm(d_model)

        # Decoder Stack: List of N TransformerBlocks with cross-attention enabled
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                pre_ln=True,
                has_cross_attention=True,
            )
            for _ in range(n_decoder_layers)
        ])
        # Layer normalization applied to final decoder representations
        self.dec_ln = CustomLayerNorm(d_model)

        # Final linear projection from d_model hidden representations to target vocabulary logits
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize all model weights using Xavier uniform initialization
        self._init_weights()

    # Weight initialization helper
    def _init_weights(self):
        # Iterate over all parameter tensors
        for p in self.parameters():
            # If parameter has 2 or more dimensions (weight matrices)
            if p.dim() > 1:
                # Apply Xavier uniform initialization
                nn.init.xavier_uniform_(p)

    # Helper method to create a boolean mask for padding tokens
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask for source padding tokens: (B, 1, 1, src_len)."""
        # Return True for real tokens, False for padding tokens pad_idx (0)
        # Unsqueeze dimensions to broadcast with shape: (batch_size, 1, 1, src_len)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    # Encode source tokens into contextual memory representations
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run input through encoder stack."""
        # Embed tokens, scale by sqrt(d_model) as in original paper, and add positional encoding
        x = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        # Pass sequentially through each encoder transformer block
        for block in self.encoder_blocks:
            # Bidirectional attention across all source tokens (is_causal=False)
            x, _ = block(x, self_mask=src_mask, is_causal=False)
        # Apply final encoder layer normalization and return
        return self.enc_ln(x)

    # Decode target tokens by attending to target and cross-attending to encoder memory
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run target through decoder stack with cross-attention to memory."""
        # Embed target tokens, scale by sqrt(d_model), and add positional encoding
        x = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        # Pass through each decoder block
        for block in self.decoder_blocks:
            # Causal self-attention over target tokens + cross-attention over encoder memory
            x, _ = block(x, context=memory, self_mask=tgt_mask, cross_mask=src_mask, is_causal=True)
        # Apply final decoder layer normalization and return
        return self.dec_ln(x)

    # Full forward pass of Seq2Seq model
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through entire Seq2Seq Transformer.

        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        # If no source mask provided, automatically create one based on padding tokens
        if src_mask is None:
            # Generate padding mask
            src_mask = self.make_src_mask(src)

        # Run source sequence through encoder stack to produce contextual memory
        memory = self.encode(src, src_mask)
        # Run target sequence through decoder stack using encoder memory
        dec_out = self.decode(tgt, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        # Project decoder output representations to vocabulary logits
        logits = self.lm_head(dec_out)
        # Return predicted logits
        return logits

    # Autoregressive generation method for translating or generating from source
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        start_symbol: int,
        end_symbol: int,
        max_len: int = 50,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation given source sequence.
        """
        # Switch model to evaluation mode
        self.eval()
        # Get device of input
        device = src.device
        # Build source padding mask
        src_mask = self.make_src_mask(src)
        # Encode source once (compute encoder memory once upfront for all steps)
        memory = self.encode(src, src_mask)

        # Initialize output sequence with start token: shape (batch_size, 1)
        ys = torch.ones(src.size(0), 1, dtype=torch.long, device=device).fill_(start_symbol)

        # Iteratively generate one token at a time up to max_len
        for _ in range(max_len - 1):
            # Decode current target sequence ys with cross-attention to memory
            out = self.decode(ys, memory, src_mask=src_mask)
            # Project the last generated token position to vocabulary logits
            logits = self.lm_head(out[:, -1, :])
            # Select the token with maximum probability (greedy argmax)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # Append selected token to the running target sequence
            ys = torch.cat([ys, next_token], dim=1)

            # If all sequences in the batch have produced the end symbol, break early
            if (next_token == end_symbol).all():
                # Stop generating
                break

        # Return full generated target sequence
        return ys
