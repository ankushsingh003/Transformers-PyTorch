"""
Transformer Block implementation supporting:
- Pre-LayerNorm (standard in GPT-2/3, LLaMA) and Post-LayerNorm (Vaswani et al., 2017)
- Self-Attention and optional Cross-Attention (Encoder-Decoder)
- CustomLayerNorm or CustomRMSNorm
"""

# Import typing utilities for optional arguments and tuple outputs
from typing import Optional, Tuple
# Import PyTorch core library
import torch
# Import neural network building blocks from PyTorch
import torch.nn as nn

# Import our from-scratch Multi-Head Attention module
from .attention import CustomMultiHeadAttention
# Import our from-scratch Position-Wise Feed-Forward module
from .feed_forward import CustomFeedForward
# Import our from-scratch CustomLayerNorm and CustomRMSNorm normalization modules
from .normalization import CustomLayerNorm, CustomRMSNorm


# Define a single unified Transformer Block
class TransformerBlock(nn.Module):
    r"""
    A single unified Transformer Block.

    Architecture (Pre-LN):
        1. x = x + SelfAttention(Norm1(x))
        2. If cross_attention: x = x + CrossAttention(Norm_cross(x), context)
        3. x = x + FeedForward(Norm2(x))

    Args:
        d_model (int): Hidden dimension size.
        n_heads (int): Number of attention heads.
        d_ff (int, optional): Feed-forward hidden dimension.
        dropout (float): Dropout probability.
        norm_type (str): 'layernorm' or 'rmsnorm'.
        pre_ln (bool): If True, use Pre-LN (recommended for training stability).
        has_cross_attention (bool): If True, adds an encoder-decoder cross-attention layer.
        activation (str): 'gelu' or 'relu'.
    """

    # Constructor method initializing block sub-layers
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        pre_ln: bool = True,
        has_cross_attention: bool = False,
        activation: str = "gelu",
    ):
        # Initialize base nn.Module
        super().__init__()
        # Store whether to use Pre-LN (LayerNorm before sub-layer) or Post-LN (LayerNorm after residual add)
        self.pre_ln = pre_ln
        # Store whether this block includes cross-attention (for encoder-decoder models)
        self.has_cross_attention = has_cross_attention

        # Factory function to build either CustomRMSNorm or CustomLayerNorm
        def build_norm():
            # Check if modern RMSNorm was selected
            if norm_type.lower() == "rmsnorm":
                # Return RMSNorm module
                return CustomRMSNorm(d_model)
            # Otherwise return standard LayerNorm module
            return CustomLayerNorm(d_model)

        # First normalization layer (applied to input before/after self-attention)
        self.norm1 = build_norm()
        # Self-attention module: tokens attend to other tokens in the same sequence
        self.self_attn = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # If cross-attention is needed (e.g., in a translation decoder attending to encoder outputs)
        if self.has_cross_attention:
            # Normalization layer before cross-attention
            self.norm_cross = build_norm()
            # Cross-attention module: queries come from target, keys and values come from encoder context
            self.cross_attn = CustomMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # Second normalization layer (applied before/after feed-forward network)
        self.norm2 = build_norm()
        # Position-wise feed-forward network (two linear layers with GELU/ReLU)
        self.mlp = CustomFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

    # Forward pass through the Transformer Block
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            context: Optional encoder hidden states (batch_size, context_len, d_model)
            self_mask: Optional attention mask for self-attention
            cross_mask: Optional attention mask for cross-attention
            is_causal: Whether self-attention is causal (autoregressive)

        Returns:
            x: Transformed tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Self-attention weights
        """
        # --- 1. Self-Attention Sub-Layer ---
        # Check if Pre-LN topology is active (standard in modern Transformers like GPT-2 and LLaMA)
        if self.pre_ln:
            # Normalize x FIRST before self-attention
            normed = self.norm1(x)
            # Run normalized representation through self-attention
            attn_out, attn_weights = self.self_attn(normed, mask=self_mask, is_causal=is_causal)
            # Add attention output to original unnormalized x (Residual Connection: x + f(LN(x)))
            x = x + attn_out
        else:
            # Post-LN (Vaswani 2017): Run unnormalized x through self-attention
            attn_out, attn_weights = self.self_attn(x, mask=self_mask, is_causal=is_causal)
            # Add to residual and THEN normalize: LN(x + f(x))
            x = self.norm1(x + attn_out)

        # --- 2. Cross-Attention Sub-Layer (Optional, used in Seq2Seq decoders) ---
        if self.has_cross_attention:
            # Ensure encoder context was provided
            assert context is not None, "Context must be provided when has_cross_attention=True"
            # In Pre-LN mode
            if self.pre_ln:
                # Normalize x before cross-attention
                normed_cross = self.norm_cross(x)
                # Compute cross-attention: query=normed_cross, key=context, value=context
                cross_out, _ = self.cross_attn(query=normed_cross, key=context, value=context, mask=cross_mask)
                # Residual addition
                x = x + cross_out
            else:
                # Post-LN cross-attention
                cross_out, _ = self.cross_attn(query=x, key=context, value=context, mask=cross_mask)
                # Residual addition followed by normalization
                x = self.norm_cross(x + cross_out)

        # --- 3. Position-Wise Feed-Forward Sub-Layer ---
        # Pre-LN mode for MLP
        if self.pre_ln:
            # Normalize representation before feed-forward network
            normed_mlp = self.norm2(x)
            # Pass through MLP and add back to residual stream
            x = x + self.mlp(normed_mlp)
        else:
            # Post-LN mode: pass through MLP, add to residual, and normalize
            x = self.norm2(x + self.mlp(x))

        # Return updated hidden representations and self-attention weights
        return x, attn_weights
