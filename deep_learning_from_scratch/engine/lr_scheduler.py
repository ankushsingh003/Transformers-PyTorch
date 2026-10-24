"""
Learning Rate Schedulers implemented from scratch:
- CosineWarmupScheduler: Linear warmup followed by Cosine Annealing (Radford et al. 2019 / Loshchilov & Hutter)
- NoamScheduler: Inverse square-root schedule with warmup from 'Attention Is All You Need' (Vaswani et al. 2017)
"""

# Import math module for cosine and pi calculations
import math
# Import List type hint
from typing import List
# Import PyTorch Optimizer base class
from torch.optim import Optimizer


# Cosine Annealing with Warmup scheduler (used in modern LLMs like GPT-2/3, LLaMA)
class CosineWarmupScheduler:
    r"""
    Linear Warmup followed by Cosine Annealing decay down to min_lr.

    Formulation:
        If step < warmup_steps:
            lr = max_lr * (step + 1) / warmup_steps
        Else:
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + cos(pi * decay_ratio))
            lr = min_lr + coeff * (max_lr - min_lr)

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps to linearly ramp up the learning rate.
        max_steps (int): Total number of training steps.
        min_lr (float): Minimum learning rate floor (default: 0.1 * max_lr).
    """

    # Constructor method initializing schedule parameters
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
    ):
        # Store wrapped PyTorch optimizer (e.g. AdamW)
        self.optimizer = optimizer
        # Store number of warmup steps (at least 1 to avoid division by zero)
        self.warmup_steps = max(1, warmup_steps)
        # Store total planned training steps
        self.max_steps = max_steps
        # Store minimum learning rate floor
        self.min_lr = min_lr
        # Record initial peak learning rates from each parameter group in optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        # Initialize internal step counter to 0
        self.current_step = 0

    # Calculate current learning rate for a given base learning rate
    def get_lr(self, base_lr: float) -> float:
        # Retrieve current step index
        step = self.current_step
        # 1. Warmup phase: linearly ramp up from 0 to base_lr
        if step < self.warmup_steps:
            # Linear interpolation: base_lr * (current_step / warmup_steps)
            return base_lr * (step + 1) / self.warmup_steps
        # If training has surpassed max_steps, clamp to min_lr floor
        if step > self.max_steps:
            # Return floor rate
            return self.min_lr

        # 2. Cosine decay phase: smoothly decay from base_lr down to min_lr
        # Compute progress ratio between 0.0 and 1.0
        decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        # Compute cosine decay coefficient between 1.0 (at start) and 0.0 (at end)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # Interpolate between min_lr and base_lr using coeff
        return self.min_lr + coeff * (base_lr - self.min_lr)

    # Step function called after each optimizer step to update learning rates
    def step(self):
        """Update learning rate for all parameter groups."""
        # Loop through parameter groups and their corresponding base learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            # Update the learning rate parameter in the optimizer dictionary
            param_group["lr"] = self.get_lr(base_lr)
        # Increment step counter by 1
        self.current_step += 1

    # Return current learning rates across all parameter groups
    def get_last_lr(self) -> List[float]:
        # Return list of current learning rates
        return [group["lr"] for group in self.optimizer.param_groups]

    # Save scheduler state for checkpointing
    def state_dict(self) -> dict:
        # Return serializable dictionary of internal state
        return {"current_step": self.current_step, "base_lrs": self.base_lrs}

    # Restore scheduler state from a checkpoint
    def load_state_dict(self, state_dict: dict):
        # Restore current step
        self.current_step = state_dict["current_step"]
        # Restore base learning rates
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)


# Noam Scheduler: the exact formula from 'Attention Is All You Need' (Vaswani et al., 2017)
class NoamScheduler:
    r"""
    The original Transformer learning rate scheduler from 'Attention Is All You Need' (Vaswani et al., 2017).

    Formulation:
        lr = d_model^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        d_model (int): Model embedding dimension.
        warmup_steps (int): Number of warmup steps (typically 4000).
        scale_factor (float): Multiplier factor (default: 1.0).
    """

    # Constructor method for Noam scheduler
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        scale_factor: float = 1.0,
    ):
        # Store wrapped optimizer
        self.optimizer = optimizer
        # Store model dimension
        self.d_model = d_model
        # Store warmup steps count
        self.warmup_steps = warmup_steps
        # Store scaling multiplier
        self.scale_factor = scale_factor
        # Initialize step counter to 0
        self.current_step = 0

    # Calculate learning rate according to the formula
    def get_lr(self) -> float:
        # Prevent division by zero on step 0
        step = max(1, self.current_step)
        # Compute first term: 1 / sqrt(step)
        arg1 = step ** (-0.5)
        # Compute second term: step * (warmup_steps ^ -1.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        # Compute final rate: scale * (1 / sqrt(d_model)) * min(arg1, arg2)
        return self.scale_factor * (self.d_model ** (-0.5)) * min(arg1, arg2)

    # Step function updating optimizer parameter groups
    def step(self):
        # Calculate new learning rate
        lr = self.get_lr()
        # Apply new rate to all parameter groups
        for param_group in self.optimizer.param_groups:
            # Set learning rate
            param_group["lr"] = lr
        # Increment step count
        self.current_step += 1

    # Return current learning rate
    def get_last_lr(self) -> List[float]:
        # Return list of learning rates
        return [group["lr"] for group in self.optimizer.param_groups]

    # Return serializable state dictionary
    def state_dict(self) -> dict:
        # Return dictionary with current step
        return {"current_step": self.current_step}

    # Load scheduler state from dictionary
    def load_state_dict(self, state_dict: dict):
        # Restore step
        self.current_step = state_dict["current_step"]
