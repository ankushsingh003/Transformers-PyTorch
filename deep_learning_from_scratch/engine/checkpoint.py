"""
Resumable checkpointing system for model weights, optimizer, scaler, and training state.
"""

# Import os module for interacting with the file system (directory creation, file renaming)
import os
# Import typing annotations for optional values and dictionaries
from typing import Optional, Dict, Any
# Import PyTorch core library
import torch
# Import neural network base class
import torch.nn as nn


# CheckpointManager handles saving and loading full training sessions safely
class CheckpointManager:
    """Manages saving and restoring training state atomically."""

    # Constructor initializing target checkpoint directory
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        # Store checkpoint folder path
        self.checkpoint_dir = checkpoint_dir
        # Create directory on disk if it does not exist yet
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Method to save complete training state atomically to disk
    def save(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        val_loss: float = float("inf"),
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a complete checkpoint dictionary to disk."""
        # Join folder path with filename to form full destination path
        filepath = os.path.join(self.checkpoint_dir, filename)
        # Construct dictionary packaging all critical training objects
        checkpoint = {
            # Current training epoch number
            "epoch": epoch,
            # Current global training step count
            "step": step,
            # Validation loss score for this checkpoint
            "val_loss": val_loss,
            # Model weights and persistent buffers
            "model_state_dict": model.state_dict(),
            # Optimizer internal state (Adam moving averages of gradients and squared gradients)
            "optimizer_state_dict": optimizer.state_dict(),
            # Loss scaler state (current scale factor for mixed precision)
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            # Learning rate scheduler step position
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            # Any supplementary metadata such as model configuration
            "extra_state": extra_state or {},
        }
        # Define temporary filename to ensure atomic write operations
        temp_filepath = filepath + ".tmp"
        # Serialize and write the checkpoint dictionary to the temporary file
        torch.save(checkpoint, temp_filepath)
        # If an older checkpoint with the destination name already exists, remove it
        if os.path.exists(filepath):
            # Delete old file
            os.remove(filepath)
        # Atomically rename temporary file to official destination file (prevents corruption if interrupted)
        os.rename(temp_filepath, filepath)
        # Return path of saved checkpoint file
        return filepath

    # Method to load and restore training state from disk
    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Restore training state from a saved checkpoint."""
        # Check if the requested checkpoint file exists on disk
        if not os.path.exists(filepath):
            # Raise an error if file is not found
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        # Load checkpoint dictionary into memory, mapping tensors to specified device (e.g., 'cpu' or 'cuda')
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        # Restore learned parameters into the model architecture
        model.load_state_dict(checkpoint["model_state_dict"])
        # Restore optimizer momentum and moving average states if optimizer was provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            # Load optimizer state dictionary
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore scaler state if scaler was provided
        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            # Load scaler state dictionary
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Restore learning rate scheduler step counter if scheduler was provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            # Load scheduler state dictionary
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return full checkpoint dictionary containing epoch, step, and extra metadata
        return checkpoint
