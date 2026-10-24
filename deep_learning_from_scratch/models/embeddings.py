"""
Positional Encodings and Embeddings implemented from first principles:
- SinusoidalPositionalEncoding: Fixed geometric harmonic series (Vaswani et al., 2017)
- LearnablePositionalEmbedding: Learned position lookup table (Radford et al., 2019 / GPT-2)
"""

# Import the math module for mathematical constants like pi and logarithms
import math
# Import PyTorch core library for tensor calculations
import torch
# Import neural network building blocks from PyTorch
import torch.nn as nn


# Define the Sinusoidal Positional Encoding class inheriting from PyTorch's base neural network Module
class SinusoidalPositionalEncoding(nn.Module):
    r"""
    Sinusoidal Positional Encoding from 'Attention Is All You Need' (Vaswani et al., 2017).

    Formulation:
        PE(pos, 2i)   = \sin(pos / 10000^{2i / d_{model}})
        PE(pos, 2i+1) = \cos(pos / 10000^{2i / d_{model}})

    Args:
        d_model (int): Dimension of the model embeddings.
        max_len (int): Maximum sequence length to precompute encodings for.
        dropout (float): Dropout probability applied to sum of embeddings and PE.
    """

    # Constructor method to initialize parameters, precomputed values, and layers
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        # Call the parent class (nn.Module) constructor to properly initialize PyTorch internals
        super().__init__()
        # Initialize a dropout layer to randomly zero some elements to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Allocate a 2D tensor of zeros of shape (max_len, d_model) to store positional values
        pe = torch.zeros(max_len, d_model)
        # Create a 1D column vector containing integer positions from 0 up to max_len - 1
        # Shape: (max_len, 1) -> unsqueeze adds an explicit dimension for broadcasting
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Compute the geometric frequencies across embedding dimensions (div_term)
        # Instead of computing 10000^(2i/d_model) directly, we compute in log-space: exp(2i * -log(10000)/d_model)
        # This prevents floating-point numerical underflow/overflow for large dimensions
        # Shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        # Fill the EVEN indices (0, 2, 4, ...) across the embedding dimension using the sine function
        # position * div_term broadcasts (max_len, 1) * (d_model // 2) -> (max_len, d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Fill the ODD indices (1, 3, 5, ...) across the embedding dimension using the cosine function
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a leading batch dimension of size 1 so it can broadcast across any batch size
        # Shape transforms from (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register 'pe' as a persistent model buffer (saved in state_dict, moves with .to(device), but NOT trained by optimizer)
        self.register_buffer("pe", pe)

    # Forward method executed whenever data passes through this layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token embedding tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional information added
        """
        # Extract the sequence length (number of tokens in current batch) from dimension 1
        seq_len = x.size(1)
        # Slice the precomputed table up to current seq_len and add element-wise to input token embeddings
        # Broadcasts: (batch_size, seq_len, d_model) + (1, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]
        # Apply dropout regularization to the combined representation and return
        return self.dropout(x)


# Define Learnable Positional Embedding (used in GPT-2 and BERT) where positions are learned parameters
class LearnablePositionalEmbedding(nn.Module):
    r"""
    Learnable Positional Embedding as used in GPT-2 (Radford et al., 2019) and BERT (Devlin et al., 2018).

    Args:
        max_seq_len (int): Maximum sequence length supported.
        d_model (int): Hidden embedding dimension.
        dropout (float): Dropout probability.
    """

    # Constructor method to set up learned weights and configuration
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        # Call the base nn.Module constructor
        super().__init__()
        # Store the maximum sequence length as an instance variable
        self.max_seq_len = max_seq_len
        # Create an embedding lookup table with max_seq_len rows and d_model columns
        # Each position index (0, 1, 2, ...) has its own trainable d_model-dimensional vector
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Create a dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the embedding weights with a normal distribution centered at 0 with standard deviation 0.02
        # This matches the standard weight initialization scheme from the original GPT-2 paper
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    # Forward method executed on every forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional vectors added
        """
        # Get the sequence length of the incoming token embeddings
        seq_len = x.size(1)
        # Ensure that the incoming sequence does not exceed the model's maximum supported context window
        assert seq_len <= self.max_seq_len, (
            f"Input sequence length {seq_len} exceeds max supported length {self.max_seq_len}"
        )

        # Generate a 1D tensor of integer positions: [0, 1, 2, ..., seq_len - 1]
        # Match device (CPU or GPU) to the device of the input tensor x
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # Look up the learned embedding vector for each position index -> shape: (seq_len, d_model)
        pos_embeddings = self.pos_emb(positions)
        # Add an extra dimension at the front to shape (1, seq_len, d_model) and add to x via broadcasting
        x = x + pos_embeddings.unsqueeze(0)
        # Apply dropout regularization to the sum and return the final representation
        return self.dropout(x)
