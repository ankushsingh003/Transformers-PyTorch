"""
Position-Wise Feed-Forward Network implemented from first principles.
Supports GELU (Gaussian Error Linear Unit, standard in GPT-2/3) and ReLU.
"""

# Import Python's built-in math module for mathematical constants (pi, sqrt)
import math
# Import PyTorch core library
import torch
# Import PyTorch neural network module
import torch.nn as nn


# Define CustomGELU activation function implemented from scratch without torch.nn.GELU
class CustomGELU(nn.Module):
    r"""
    Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).
    Implemented from scratch using the accurate tanh approximation:
        0.5 * x * (1 + \tanh(\sqrt{2/\pi} * (x + 0.044715 * x^3)))
    """

    # Forward computation of GELU activation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the cubic term: x^3
        # Compute the argument inside tanh: sqrt(2/pi) * (x + 0.044715 * x^3)
        # Compute tanh of the argument, add 1.0, and multiply by 0.5 * x
        # This provides a smooth non-linearity where negative values have non-zero gradients
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Position-Wise Feed-Forward Network (FFN) applied to every token position identically and independently
class CustomFeedForward(nn.Module):
    r"""
    Position-Wise Feed-Forward Network (FFN).

    Mathematical formulation:
        \text{FFN}(x) = \text{Dropout}(\text{Activation}(x W_1 + b_1)) W_2 + b_2

    Args:
        d_model (int): Hidden dimension size.
        d_ff (int, optional): Inner feed-forward expansion dimension. Defaults to 4 * d_model.
        dropout (float): Dropout probability.
        activation (str): 'gelu' or 'relu'. Defaults to 'gelu'.
        bias (bool): Whether linear layers include bias. Defaults to True.
    """

    # Constructor method initializing linear projections and activation
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        # Call parent nn.Module constructor
        super().__init__()
        # If inner dimension d_ff is not specified, default to 4 times the embedding dimension (standard in Transformer papers)
        if d_ff is None:
            # Set expansion dimension to 4 * d_model (e.g. 4 * 64 = 256)
            d_ff = 4 * d_model

        # First linear transformation: expands from d_model up to d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        # Second linear transformation: projects back down from d_ff to d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        # Dropout layer applied after activation to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Check which activation function was requested
        if activation.lower() == "gelu":
            # Instantiate our from-scratch GELU activation
            self.activation = CustomGELU()
        elif activation.lower() == "relu":
            # Instantiate standard Rectified Linear Unit (max(0, x))
            self.activation = nn.ReLU()
        else:
            # Raise an error if an unsupported activation string was provided
            raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu' or 'relu'.")

        # Initialize linear layer weights using Xavier uniform initialization
        self._reset_parameters()

    # Helper method to initialize weights
    def _reset_parameters(self):
        # Xavier uniform initialization for w1 weight matrix
        nn.init.xavier_uniform_(self.w1.weight)
        # Xavier uniform initialization for w2 weight matrix
        nn.init.xavier_uniform_(self.w2.weight)
        # If bias terms exist, initialize them to zero
        if self.w1.bias is not None:
            # Zero out w1 bias
            nn.init.zeros_(self.w1.bias)
            # Zero out w2 bias
            nn.init.zeros_(self.w2.bias)

    # Forward pass of Feed-Forward Network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Pass input through first linear expansion layer: (B, T, d_model) -> (B, T, d_ff)
        # Then pass through activation function (GELU or ReLU)
        hidden = self.activation(self.w1(x))
        # Apply dropout to the hidden representations
        hidden = self.dropout(hidden)
        # Project back down to model dimension: (B, T, d_ff) -> (B, T, d_model)
        output = self.w2(hidden)
        # Return the transformed representation
        return output
