"""
Custom Normalization Layers implemented from first principles.
- CustomLayerNorm: Standard Layer Normalization (Ba, Kiros, & Hinton, 2016)
- CustomRMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
"""

# Import the core PyTorch library for tensor mathematics
import torch
# Import the neural network module from PyTorch containing base Module and Parameter
import torch.nn as nn


# Define CustomLayerNorm class implementing standard Layer Normalization from scratch
class CustomLayerNorm(nn.Module):
    r"""
    Custom Layer Normalization implemented from scratch without torch.nn.LayerNorm.

    Mathematical formulation:
        \mu = \frac{1}{d} \sum_{i=1}^d x_i
        \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2
        \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
        y = \gamma \odot \hat{x} + \beta

    Args:
        normalized_shape (int or tuple): Input shape from an expected input of size.
        eps (float): Numerical stability epsilon (default: 1e-5).
        elementwise_affine (bool): Whether to learn \gamma (weight) and \beta (bias).
    """

    # Initialization method to set up parameters and configuration
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        # Call base class constructor to initialize PyTorch's internal hooks and state tracking
        super().__init__()
        # Store the feature dimension size that needs to be normalized (e.g., d_model = 64)
        self.normalized_shape = normalized_shape
        # Store epsilon: a tiny positive constant (1e-5) added to variance to prevent division by zero
        self.eps = eps
        # Store boolean flag indicating whether learnable scale (gamma) and shift (beta) parameters are used
        self.elementwise_affine = elementwise_affine

        # If affine parameters are enabled, create learnable scale (weight) and shift (bias)
        if self.elementwise_affine:
            # Initialize learnable scale parameter gamma with all 1s (shape matches feature dimension)
            # nn.Parameter wraps a tensor so PyTorch tracks it as a trainable model weight
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            # Initialize learnable shift parameter beta with all 0s (shape matches feature dimension)
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            # If affine is disabled, register weight and bias as None so PyTorch knows they don't exist
            self.register_parameter("weight", None)
            # Register bias as None
            self.register_parameter("bias", None)

    # Forward pass executed when calling this module on an input tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the mean (average) across the last feature dimension (dim=-1)
        # keepdim=True retains the dimension with size 1 so broadcasting works during subtraction: (B, T, D) - (B, T, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate variance across the last dimension
        # unbiased=False computes population variance (dividing by N, not N-1), matching PyTorch's native nn.LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Subtract mean (center at zero) and divide by standard deviation sqrt(var + eps) (scale to unit variance)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # If learnable scale (weight) and shift (bias) are enabled, apply affine transformation
        if self.elementwise_affine:
            # Multiply by gamma (weight) and add beta (bias) element-wise
            return self.weight * x_norm + self.bias
        # If affine is disabled, return the zero-mean unit-variance normalized tensor directly
        return x_norm


# Define CustomRMSNorm class implementing Root Mean Square Layer Normalization (used in LLaMA, Mistral, Gemma)
class CustomRMSNorm(nn.Module):
    r"""
    Root Mean Square Layer Normalization (RMSNorm) from Zhang & Sennrich (2019).
    Widely used in modern LLMs (LLaMA, Mistral, Gemma) for faster computation and stability.

    Mathematical formulation:
        \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
        \bar{x} = \frac{x}{\text{RMS}(x)}
        y = \gamma \odot \bar{x}

    Args:
        dim (int): Hidden dimension size.
        eps (float): Numerical stability epsilon (default: 1e-6).
    """

    # Initialization method for RMSNorm
    def __init__(self, dim: int, eps: float = 1e-6):
        # Call parent nn.Module constructor
        super().__init__()
        # Store the hidden dimension size
        self.dim = dim
        # Store epsilon to avoid division by zero
        self.eps = eps
        # Create learnable scaling parameter gamma (initialized to 1.0)
        # RMSNorm does NOT use a bias parameter, only scale!
        self.weight = nn.Parameter(torch.ones(dim))

    # Forward pass computation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the Root Mean Square of the vector across the last dimension
        # Square each element (x ** 2), take average across dim=-1, add eps, and take square root
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Scale the input tensor by dividing by its RMS, then multiply by the learnable weight vector
        return (x / rms) * self.weight
