"""
Decoder-Only Causal Transformer (Radford et al., 2019 / GPT-2 architecture).
Implemented strictly from scratch using custom modules without external libraries.
"""

# Import typing annotations for hints and documentation
from typing import Optional, Tuple, Dict, Any
# Import math module for mathematical formulas (square root)
import math
# Import PyTorch core library
import torch
# Import neural network base classes and layers
import torch.nn as nn
# Import functional neural network operations (cross entropy, softmax)
import torch.nn.functional as F

# Import our custom normalization layers
from .normalization import CustomLayerNorm, CustomRMSNorm
# Import our custom positional embeddings
from .embeddings import LearnablePositionalEmbedding, SinusoidalPositionalEncoding
# Import our custom transformer block
from .transformer_block import TransformerBlock


# Configuration class holding all model hyperparameters
class GPTConfig:
    """Hyperparameters configuration for GPT model."""

    # Constructor setting default hyperparameter values
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        pre_ln: bool = True,
        tie_weights: bool = True,
        positional_encoding: str = "learned",  # 'learned' or 'sinusoidal'
        activation: str = "gelu",
    ):
        # Total number of unique tokens in vocabulary
        self.vocab_size = vocab_size
        # Maximum sequence length (context window) model can process
        self.max_seq_len = max_seq_len
        # Hidden embedding dimension
        self.d_model = d_model
        # Number of parallel attention heads
        self.n_heads = n_heads
        # Number of stacked transformer blocks (depth)
        self.n_layers = n_layers
        # Inner feed-forward expansion dimension (defaults to 4 * d_model)
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        # Dropout probability
        self.dropout = dropout
        # Type of normalization: 'layernorm' or 'rmsnorm'
        self.norm_type = norm_type
        # Whether to use Pre-LN (True) or Post-LN (False)
        self.pre_ln = pre_ln
        # Whether to tie input token embedding weights with output LM head weights
        self.tie_weights = tie_weights
        # Whether positional encoding is learned or sinusoidal
        self.positional_encoding = positional_encoding
        # Activation function ('gelu' or 'relu')
        self.activation = activation


# The complete Decoder-Only Causal Transformer (GPT-2 style)
class GPT(nn.Module):
    r"""
    Decoder-Only Causal Transformer for autoregressive language modeling.
    Follows Radford et al. (2019) 'Language Models are Unsupervised Multitask Learners'.

    Components:
        1. Token Embedding & Positional Embedding
        2. Stack of N Transformer Blocks with causal self-attention
        3. Final Normalization layer
        4. Linear Language Modeling Head (d_model -> vocab_size)
    """

    # Constructor initializing all model submodules
    def __init__(self, config: GPTConfig):
        # Call base nn.Module constructor
        super().__init__()
        # Store configuration object
        self.config = config

        # Token embedding lookup table: maps integer token ID to d_model vector
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Choose positional encoding based on config
        if config.positional_encoding == "learned":
            # Instantiate learned positional embedding (lookup table)
            self.pos_emb = LearnablePositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                dropout=config.dropout,
            )
        elif config.positional_encoding == "sinusoidal":
            # Instantiate fixed sinusoidal positional encoding
            self.pos_emb = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            # Raise error for unknown type
            raise ValueError(f"Unknown positional encoding: {config.positional_encoding}")

        # Dropout layer applied after embedding sum
        self.drop = nn.Dropout(config.dropout)

        # Stack of N Transformer Blocks stored in a ModuleList
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                norm_type=config.norm_type,
                pre_ln=config.pre_ln,
                has_cross_attention=False,
                activation=config.activation,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization layer after exiting the transformer block stack
        if config.norm_type.lower() == "rmsnorm":
            # Use RMSNorm if specified
            self.ln_f = CustomRMSNorm(config.d_model)
        else:
            # Use standard LayerNorm by default
            self.ln_f = CustomLayerNorm(config.d_model)

        # Output projection head: maps d_model representation to vocabulary size logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the weight tensor between token embedding and LM projection head
        # This reduces parameter count significantly and improves language model performance
        if config.tie_weights:
            # Point the LM head weight directly to the token embedding weight tensor
            self.lm_head.weight = self.tok_emb.weight

        # Apply parameter initialization recursively to all modules
        self.apply(self._init_weights)

        # Apply GPT-2's special initialization scaling for residual projections:
        # Scale standard deviation by 1 / sqrt(2 * n_layers) to prevent variance growth at depth
        for pn, p in self.named_parameters():
            # Check if parameter is an output projection or second feed-forward weight matrix
            if pn.endswith("out_proj.weight") or pn.endswith("w2.weight"):
                # Re-initialize with scaled standard deviation
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    # Weight initialization helper function
    def _init_weights(self, module):
        # If the module is a linear layer
        if isinstance(module, nn.Linear):
            # Initialize weights from normal distribution with mean 0.0 and std 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # If bias is present, initialize to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # If the module is an embedding layer
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights from normal distribution with mean 0.0 and std 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Forward pass of GPT model
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: Tensor of token indices of shape (batch_size, seq_len)
            targets: Optional ground-truth token indices of shape (batch_size, seq_len)

        Returns:
            logits: Predicted next-token logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy scalar loss (if targets is provided) else None
        """
        # Unpack batch size (b) and sequence length (t) from input tensor shape
        b, t = idx.shape
        # Assert that sequence length does not exceed maximum context window
        assert t <= self.config.max_seq_len, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.max_seq_len}"
        )

        # Convert token indices to d_model-dimensional embedding vectors: (b, t) -> (b, t, d_model)
        tok_embeddings = self.tok_emb(idx)
        # Add positional embeddings to token embeddings: (b, t, d_model)
        x = self.pos_emb(tok_embeddings)

        # Pass through each Transformer block sequentially with causal attention enabled
        for block in self.blocks:
            # Execute block with is_causal=True to mask future tokens
            x, _ = block(x, is_causal=True)

        # Apply final normalization layer to the output activations
        x = self.ln_f(x)

        # Project final activations to vocabulary logits: (b, t, d_model) -> (b, t, vocab_size)
        logits = self.lm_head(x)

        # Initialize loss to None
        loss = None
        # If training targets are provided, compute Cross-Entropy loss
        if targets is not None:
            # Flatten logits to (b * t, vocab_size) and targets to (b * t), ignore padding index -1
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return predicted logits and loss
        return logits, loss

    # Autoregressive text generation method (disables gradient tracking for efficiency)
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive next-token generation loop.

        Args:
            idx: Conditioning prompt token indices of shape (batch_size, seq_len).
            max_new_tokens: Number of subsequent tokens to generate.
            temperature: Softmax temperature scale (>0). Lower = more deterministic.
            top_k: If set, retain only top_k most probable tokens.
            greedy: If True, greedily pick argmax token at each step.

        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens)
        """
        # Loop for the number of new tokens to produce
        for _ in range(max_new_tokens):
            # If sequence exceeds max context length, crop from the left to keep only the last max_seq_len tokens
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len :]

            # Forward pass through model to get logits for all tokens in context
            logits, _ = self(idx_cond)
            # Slice only the logits for the very last token position: (batch_size, vocab_size)
            logits = logits[:, -1, :]

            # If greedy decoding is chosen, pick the token with the highest logit
            if greedy:
                # Get index of maximum logit: (batch_size, 1)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Scale logits by temperature (higher temp = more diverse, lower temp = more focused)
                logits = logits / max(temperature, 1e-5)

                # If top-k filtering is requested, zero out all probabilities outside the top k
                if top_k is not None and top_k > 0:
                    # Find the k-th highest logit value
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Set any logit lower than the k-th value to -infinity
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Apply softmax to convert scaled logits into valid probability distribution
                probs = F.softmax(logits, dim=-1)
                # Sample 1 token from the probability distribution for each batch element
                idx_next = torch.multinomial(probs, num_samples=1)

            # Concatenate the newly sampled token index to the running sequence: (batch_size, seq_len + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        # Return full generated token sequence
        return idx

    # Configures AdamW optimizer with weight decay applied selectively
    def configure_optimizers(
        self,
        weight_decay: float = 0.01,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """
        Constructs AdamW optimizer with weight decay applied only to 2D weight matrices (linear/embeddings),
        exempting 1D parameters (biases, layernorm weights). Follows the standard GPT-2 recipe.
        """
        # List of parameters that should undergo L2 weight decay
        decay_params = []
        # List of parameters that should NOT undergo weight decay
        no_decay_params = []

        # Iterate through all named parameters in the model
        for name, param in self.named_parameters():
            # Skip parameters that are frozen / do not require gradients
            if not param.requires_grad:
                continue
            # 2D weight matrices (linear layer weights, embedding tables) should decay
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                # 1D vectors (biases, LayerNorm scale and shift parameters) should NOT decay
                no_decay_params.append(param)

        # Create parameter groups for PyTorch optimizer
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Check if fused AdamW kernel is supported (CUDA only)
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        # Use fused kernel if running on CUDA to speed up optimizer step
        use_fused = fused_available and device_type == "cuda"
        # Set fused argument dictionary
        extra_args = dict(fused=True) if use_fused else dict()

        # Construct and return AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    # Helper method to calculate total parameter count
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Calculate total number of model parameters."""
        # Sum total number of elements in all parameters
        n_params = sum(p.numel() for p in self.parameters())
        # Optionally subtract positional embedding count for fair comparison
        if non_embedding and hasattr(self, "pos_emb") and hasattr(self.pos_emb, "pos_emb"):
            n_params -= self.pos_emb.pos_emb.weight.numel()
        # Return parameter count
        return n_params
