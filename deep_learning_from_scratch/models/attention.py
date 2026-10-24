"""
Attention Mechanisms implemented strictly from scratch using tensor operations:
- ScaledDotProductAttention: Core scaled query-key dot-product with masking & dropout
- CustomMultiHeadAttention: Multi-head parallelization with linear projections
"""

# Import the standard math module for calculating square roots
import math
# Import typing helpers for clean type hints
from typing import Optional, Tuple
# Import PyTorch core library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn


# Scaled Dot-Product Attention: the mathematical core of the Transformer
class ScaledDotProductAttention(nn.Module):
    r"""
    Scaled Dot-Product Attention from 'Attention Is All You Need' (Vaswani et al., 2017).

    Mathematical formulation:
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V

    Args:
        dropout (float): Dropout probability applied to attention weights.
    """

    # Constructor method for attention module
    def __init__(self, dropout: float = 0.0):
        # Initialize base nn.Module
        super().__init__()
        # Create dropout layer applied to attention probability maps to prevent co-adaptation of features
        self.dropout = nn.Dropout(p=dropout)

    # Forward pass calculating attention weights and weighted context vectors
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            query: Shape (batch_size, n_heads, q_len, head_dim)
            key: Shape (batch_size, n_heads, k_len, head_dim)
            value: Shape (batch_size, n_heads, k_len, head_dim)
            mask: Optional mask tensor broadcastable to (batch_size, n_heads, q_len, k_len).
                  Can be a boolean tensor (True = keep, False = mask out) or float tensor (0 = keep, -inf = mask out).

        Returns:
            context: Shape (batch_size, n_heads, q_len, head_dim)
            attention_weights: Shape (batch_size, n_heads, q_len, k_len)
        """
        # Extract the dimension size of each head from the last axis of query
        head_dim = query.size(-1)
        # Calculate scaling factor: 1 / sqrt(head_dim) to counteract large dot product magnitudes
        scale = 1.0 / math.sqrt(head_dim)
        # Compute raw attention scores: query * key^T
        # Transpose key's last two dimensions: (B, H, K, D) -> (B, H, D, K)
        # Batched matmul: (B, H, Q, D) x (B, H, D, K) -> (B, H, Q, K)
        # Multiply by scale to prevent vanishing gradients in softmax
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Check if an attention mask (such as causal or padding mask) is supplied
        if mask is not None:
            # Check if mask is a boolean tensor
            if mask.dtype == torch.bool:
                # Replace positions where mask is False (~mask) with negative infinity (-inf)
                # Softmax of -inf equals exactly 0.0, completely blocking information flow
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                # If mask is already a float tensor (with 0 and -inf), add it directly
                scores = scores + mask

        # Apply softmax along the last dimension (over key sequence length) to turn scores into probabilities (sum to 1.0)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to the attention weight probabilities
        attention_weights_dropped = self.dropout(attention_weights)

        # Multiply attention probabilities by value vectors
        # Batched matmul: (B, H, Q, K) x (B, H, K, D) -> (B, H, Q, D)
        context = torch.matmul(attention_weights_dropped, value)

        # Return both the weighted context representation and the attention weights (for visualization/inspection)
        return context, attention_weights


# Multi-Head Attention: divides embedding dimension into multiple independent heads
class CustomMultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention implemented from scratch without torch.nn.MultiheadAttention.

    Instead of performing a single attention function with d_model-dimensional queries,
    keys, and values, we project the queries, keys, and values h times with different,
    learned linear projections to d_k, d_k, and d_v dimensions, respectively.

    Mathematical formulation:
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
        \text{where} \; \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)

    Args:
        d_model (int): Total hidden embedding dimension.
        n_heads (int): Number of parallel attention heads. Must divide d_model evenly.
        dropout (float): Dropout probability for attention weights and output projection.
        bias (bool): Whether to include bias terms in linear projections.
    """

    # Constructor method for multi-head attention module
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        # Call base class constructor
        super().__init__()
        # Assert that d_model is evenly divisible by the number of heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Store model dimension
        self.d_model = d_model
        # Store number of heads
        self.n_heads = n_heads
        # Compute individual head dimension: e.g. 64 / 4 = 16
        self.head_dim = d_model // n_heads

        # Linear projection for Queries: projects (B, T, d_model) -> (B, T, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        # Linear projection for Keys: projects (B, T, d_model) -> (B, T, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        # Linear projection for Values: projects (B, T, d_model) -> (B, T, d_model)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Instantiate our from-scratch ScaledDotProductAttention module
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Output linear projection: projects concatenated heads back to d_model: (B, T, d_model) -> (B, T, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # Residual dropout applied to the final output of the attention block
        self.resid_dropout = nn.Dropout(p=dropout)

        # Initialize parameters with Xavier uniform initialization
        self._reset_parameters()

    # Method to initialize weight matrices and zero-out biases
    def _reset_parameters(self):
        # Xavier uniform initialization for query projection weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        # Xavier uniform initialization for key projection weights
        nn.init.xavier_uniform_(self.k_proj.weight)
        # Xavier uniform initialization for value projection weights
        nn.init.xavier_uniform_(self.v_proj.weight)
        # Xavier uniform initialization for output projection weights
        nn.init.xavier_uniform_(self.out_proj.weight)
        # If bias is enabled, zero out all biases
        if self.q_proj.bias is not None:
            # Zero out query bias
            nn.init.zeros_(self.q_proj.bias)
            # Zero out key bias
            nn.init.zeros_(self.k_proj.bias)
            # Zero out value bias
            nn.init.zeros_(self.v_proj.bias)
            # Zero out output projection bias
            nn.init.zeros_(self.out_proj.bias)

    # Helper method to split and reshape tensor across multiple heads
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, head_dim)
        """
        # Unpack batch_size and seq_len from input tensor shape
        batch_size, seq_len, _ = x.shape
        # Reshape the last dimension (d_model) into (n_heads, head_dim)
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # Transpose dimensions 1 and 2 so heads dimension comes before sequence length
        # Result shape: (batch_size, n_heads, seq_len, head_dim)
        return x.transpose(1, 2)

    # Helper method to concatenate and combine heads back into a single tensor
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        """
        # Unpack batch_size and seq_len from head tensor shape
        batch_size, _, seq_len, _ = x.shape
        # Transpose back: (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        # .contiguous() ensures tensor memory is laid out sequentially for .view()
        x = x.transpose(1, 2).contiguous()
        # Flatten (n_heads, head_dim) back into original d_model dimension
        return x.view(batch_size, seq_len, self.d_model)

    # Forward pass of Multi-Head Attention
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            query: Tensor of shape (batch_size, q_len, d_model)
            key: Optional tensor of shape (batch_size, k_len, d_model). Defaults to query (self-attention).
            value: Optional tensor of shape (batch_size, k_len, d_model). Defaults to key.
            mask: Optional mask broadcastable to (batch_size, n_heads, q_len, k_len) or (q_len, k_len).
            is_causal: If True, automatically creates and applies a causal lower-triangular mask.

        Returns:
            output: Tensor of shape (batch_size, q_len, d_model)
            attn_weights: Tensor of shape (batch_size, n_heads, q_len, k_len)
        """
        # In self-attention, query, key, and value all originate from the same input tensor
        if key is None:
            # Set key equal to query
            key = query
        # If value is not provided, set value equal to key
        if value is None:
            # Set value equal to key
            value = key

        # Get sequence length of query
        q_len = query.size(1)
        # Get sequence length of key
        k_len = key.size(1)

        # Project Query and split into parallel heads: (B, T, D) -> (B, H, Q, head_dim)
        q = self._split_heads(self.q_proj(query))
        # Project Key and split into parallel heads: (B, T, D) -> (B, H, K, head_dim)
        k = self._split_heads(self.k_proj(key))
        # Project Value and split into parallel heads: (B, T, D) -> (B, H, K, head_dim)
        v = self._split_heads(self.v_proj(value))

        # Check if autoregressive causal masking is requested
        if is_causal:
            # Construct a lower-triangular matrix of 1s (True) and 0s (False)
            # Position (i, j) is True if i >= j (past/present token), and False if i < j (future token)
            causal_mask = torch.tril(torch.ones((q_len, k_len), device=query.device, dtype=torch.bool))
            # If an existing mask was provided, combine with bitwise AND
            if mask is not None:
                mask = mask & causal_mask
            else:
                # Otherwise use the causal mask directly
                mask = causal_mask

        # Perform scaled dot-product attention in parallel across all heads
        context, attn_weights = self.attention(q, k, v, mask=mask)

        # Stitch all head representations back together into shape (batch_size, seq_len, d_model)
        combined = self._combine_heads(context)
        # Apply the final output projection matrix W_O and dropout
        output = self.resid_dropout(self.out_proj(combined))

        # Return projected output and attention weights
        return output, attn_weights
