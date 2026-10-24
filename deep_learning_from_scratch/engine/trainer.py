"""
Custom Training Loop implemented completely from scratch.
Features:
- Native Automatic Mixed Precision (AMP) via torch.amp.autocast and GradScaler
- Multi-step Gradient Accumulation with remainder step handling
- Gradient Norm Clipping for numerical stability
- Throughput tracking (tokens/sec, step latency)
- Loss and Perplexity tracking with checkpointing
"""

# Import os for path handling
import os
# Import time module to benchmark step latency and compute token throughput
import time
# Import math module for exponentiation (perplexity calculation)
import math
# Import typing annotations
from typing import Optional, Callable, Dict, Any
# Import PyTorch core library
import torch
# Import neural network building blocks
import torch.nn as nn
# Import DataLoader for batching and shuffling
from torch.utils.data import DataLoader

# Import our custom CheckpointManager to save and restore training sessions
from .checkpoint import CheckpointManager


# CustomTrainer: A completely library-free training engine
class CustomTrainer:
    r"""
    A production-grade, library-free training engine.

    Gradient Accumulation Logic:
        Accumulates gradients across `grad_accum_steps` micro-batches before calling
        `optimizer.step()` and `optimizer.zero_grad()`. This simulates a batch size of
        `batch_size * grad_accum_steps` with constant memory footprint.

    Mixed Precision Logic:
        Runs forward pass under `torch.amp.autocast`, scales loss with `GradScaler` to prevent
        gradient underflow, unscales gradients before clipping, and steps the optimizer.

    Args:
        model: nn.Module to train.
        optimizer: Optimizer (e.g. AdamW).
        train_loader: DataLoader yielding (inputs, targets) batches.
        val_loader: Optional DataLoader for validation.
        scheduler: Optional learning rate scheduler.
        device: 'cuda', 'cpu', or torch.device.
        grad_accum_steps: Number of micro-batches to accumulate gradients over.
        use_amp: Whether to enable automatic mixed precision.
        amp_dtype: torch.float16 or torch.bfloat16.
        max_grad_norm: Maximum L2 gradient norm for clipping (0 to disable).
        checkpoint_dir: Directory to save checkpoints.
    """

    # Constructor initializing model, optimizer, data loaders, and training hyperparameters
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
    ):
        # Store model to be trained
        self.model = model
        # Store optimizer
        self.optimizer = optimizer
        # Store training DataLoader
        self.train_loader = train_loader
        # Store optional validation DataLoader
        self.val_loader = val_loader
        # Store optional learning rate scheduler
        self.scheduler = scheduler

        # Resolve target device: automatically detect CUDA GPU, otherwise fall back to CPU
        if device is None:
            # Check CUDA availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Convert string to torch.device if needed
            self.device = torch.device(device) if isinstance(device, str) else device

        # Store device type string ('cuda' or 'cpu') for torch.amp.autocast compatibility
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        # Move all model parameters and buffers to the target hardware device
        self.model.to(self.device)

        # Store gradient accumulation steps (must be at least 1)
        self.grad_accum_steps = max(1, grad_accum_steps)
        # Store maximum gradient norm threshold (1.0 is standard to prevent exploding gradients)
        self.max_grad_norm = max_grad_norm
        # Store boolean flag whether Automatic Mixed Precision is active
        self.use_amp = use_amp

        # Determine optimal floating-point precision format for AMP
        if amp_dtype is None:
            # If running on CUDA GPU
            if self.device_type == "cuda":
                # Use bfloat16 if GPU architecture supports it (Ampere or newer), otherwise float16
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                # On modern CPUs, PyTorch natively supports bfloat16 autocast
                self.amp_dtype = torch.bfloat16
        else:
            # Use explicitly provided dtype
            self.amp_dtype = amp_dtype

        # Set up Loss Scaler for Mixed Precision
        # Float16 has a narrow dynamic exponent range, so gradients can underflow to zero
        # GradScaler multiplies the loss by a large factor (e.g. 65536) before backward pass
        # On bfloat16 or CPU, the exponent range is large enough that scaling is not needed
        if self.device_type == "cuda" and self.use_amp and self.amp_dtype == torch.float16:
            try:
                # Use PyTorch 2.x standard torch.amp.GradScaler
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (AttributeError, TypeError):
                # Fallback to legacy torch.cuda.amp.GradScaler
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            # On CPU or bfloat16, define a lightweight dummy scaler that performs regular unscaled operations
            class DummyGradScaler:
                # Loss scaling is a no-op
                def scale(self, loss):
                    return loss
                # Optimizer step executes directly
                def step(self, optimizer):
                    optimizer.step()
                # Update is a no-op
                def update(self):
                    pass
                # Unscale is a no-op
                def unscale_(self, optimizer):
                    pass
                # Empty state dict
                def state_dict(self):
                    return {}
                # Empty load state dict
                def load_state_dict(self, state_dict):
                    pass

            # Assign dummy scaler instance
            self.scaler = DummyGradScaler()

        # Instantiate CheckpointManager to save models
        self.ckpt_manager = CheckpointManager(checkpoint_dir)
        # Track lowest validation loss observed so far
        self.best_val_loss = float("inf")
        # Global optimizer step counter
        self.global_step = 0

    # Executes a full training pass across one epoch of data
    def train_epoch(
        self,
        epoch: int,
        max_steps: Optional[int] = None,
        step_callback: Optional[Callable[[int, float], None]] = None,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Executes one training epoch with gradient accumulation and AMP.
        """
        # Set model into training mode (enables dropout)
        self.model.train()
        # Accumulator for total unscaled loss across the epoch
        total_loss = 0.0
        # Running loss accumulator for reporting progress in log intervals
        running_loss = 0.0
        # Counter for micro-batches processed since the last log message
        micro_batches_in_window = 0
        # Counter for total tokens processed for throughput calculation
        total_tokens = 0
        # Record epoch start timestamp
        epoch_start_time = time.time()
        # Record interval start timestamp
        step_start_time = time.time()

        # Total number of micro-batches in the training DataLoader
        num_batches = len(self.train_loader)
        # Clear existing gradients before starting epoch
        # set_to_none=True frees memory instead of writing zeros
        self.optimizer.zero_grad(set_to_none=True)

        # Loop through batches yielded by DataLoader
        for micro_idx, batch in enumerate(self.train_loader):
            # Extract inputs and ground-truth targets from batch
            if isinstance(batch, (list, tuple)):
                # Move input and target tensors to hardware device
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                # Unsupervised inputs
                inputs = batch.to(self.device)
                targets = None

            # Count total number of tokens in this micro-batch: batch_size * sequence_length
            batch_tokens = inputs.numel()
            # Accumulate token count
            total_tokens += batch_tokens

            # --- 1. Forward Pass under Automatic Mixed Precision context ---
            with torch.amp.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                # If target tokens exist, compute loss
                if targets is not None:
                    # Forward pass returning logits and cross-entropy loss
                    _, loss = self.model(inputs, targets=targets)
                else:
                    # Forward pass without targets
                    logits, _ = self.model(inputs)
                    loss = logits.mean()

                # Scale loss by 1 / grad_accum_steps so accumulated gradients equal the average across the effective batch
                scaled_loss = loss / self.grad_accum_steps

            # --- 2. Backward Pass with Loss Scaling ---
            # Scaler scales the loss to prevent underflow, then calls .backward() to compute gradients
            self.scaler.scale(scaled_loss).backward()
            # Track unscaled loss for metrics logging
            running_loss += loss.item()
            # Increment micro-batch window counter
            micro_batches_in_window += 1
            # Add to epoch total loss
            total_loss += loss.item()

            # --- 3. Optimizer Step on Accumulation Boundary ---
            # Check if this micro-batch marks an accumulation boundary or the end of the dataset
            is_accum_step = ((micro_idx + 1) % self.grad_accum_steps == 0) or ((micro_idx + 1) == num_batches)
            if is_accum_step:
                # Unscale the gradients before clipping norm (so norm calculation uses real mathematical magnitudes)
                self.scaler.unscale_(self.optimizer)

                # Gradient norm clipping: if global L2 norm exceeds max_grad_norm, rescale gradients downwards
                if self.max_grad_norm > 0:
                    # Clip gradients in-place and return original norm
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    grad_norm = 0.0

                # Optimizer step: scaler checks for Inf/NaN gradients; if clean, calls optimizer.step(), else skips
                self.scaler.step(self.optimizer)
                # Update scaler scale factor (increases if clean, decreases if Inf/NaN detected)
                self.scaler.update()
                # Reset gradients to None for the next accumulation cycle
                self.optimizer.zero_grad(set_to_none=True)

                # Step the learning rate scheduler if present
                if self.scheduler is not None:
                    self.scheduler.step()

                # Increment global training step counter
                self.global_step += 1

                # Periodic console logging
                if self.global_step % log_interval == 0 or (micro_idx + 1) == num_batches:
                    # Compute duration of this logging interval in seconds
                    step_duration = time.time() - step_start_time
                    # Compute processing speed in tokens per second
                    tokens_per_sec = (batch_tokens * self.grad_accum_steps) / max(step_duration, 1e-5)
                    # Compute average micro-batch loss across this window
                    avg_micro_loss = running_loss / max(1, micro_batches_in_window)
                    # Retrieve current learning rate
                    current_lr = (
                        self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]
                    )
                    # Calculate perplexity = exp(loss), capped at 20.0 to prevent overflow in print
                    perplexity = math.exp(min(avg_micro_loss, 20.0))

                    # Print step metrics
                    print(
                        f"Epoch [{epoch}] Step [{self.global_step}]: "
                        f"Loss = {avg_micro_loss:.4f} | "
                        f"PPL = {perplexity:.2f} | "
                        f"LR = {current_lr:.6e} | "
                        f"GradNorm = {float(grad_norm):.2f} | "
                        f"Speed = {tokens_per_sec:.0f} tok/s",
                        flush=True,
                    )
                    # Reset running loss accumulator
                    running_loss = 0.0
                    # Reset micro-batch window counter
                    micro_batches_in_window = 0
                    # Reset interval timer
                    step_start_time = time.time()

                # Execute optional user callback (e.g. logging to tensorboard or wandb)
                if step_callback is not None:
                    step_callback(self.global_step, loss.item())

                # Check if total step limit was reached
                if max_steps is not None and self.global_step >= max_steps:
                    # Break out of batch loop
                    break

        # Compute average loss across all micro-batches in the epoch
        avg_loss = total_loss / max(1, num_batches)
        # Compute total duration of epoch
        epoch_duration = time.time() - epoch_start_time
        # Return epoch metrics dictionary
        return {
            "train_loss": avg_loss,
            "train_ppl": math.exp(min(avg_loss, 20.0)),
            "tokens_per_sec": total_tokens / max(epoch_duration, 1e-5),
            "epoch_duration": epoch_duration,
        }

    # Evaluate model on validation data without tracking gradients
    @torch.no_grad()
    def evaluate(self, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation dataset.
        """
        # Use provided DataLoader or fall back to self.val_loader
        loader = val_loader if val_loader is not None else self.val_loader
        # If no validation set is available, return empty dictionary
        if loader is None:
            return {}

        # Set model into evaluation mode (disables dropout)
        self.model.eval()
        # Accumulator for total validation loss
        total_loss = 0.0
        # Batch counter
        num_batches = 0

        # Loop through validation batches
        for batch in loader:
            # Extract inputs and targets
            if isinstance(batch, (list, tuple)):
                # Move to hardware device
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None

            # Forward pass under AMP context
            with torch.amp.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                if targets is not None:
                    # Compute validation loss
                    _, loss = self.model(inputs, targets=targets)
                else:
                    logits, _ = self.model(inputs)
                    loss = logits.mean()

            # Accumulate loss
            total_loss += loss.item()
            # Increment batch count
            num_batches += 1

        # Compute average validation loss
        avg_loss = total_loss / max(1, num_batches)
        # Compute validation perplexity
        ppl = math.exp(min(avg_loss, 20.0))
        # Return metrics dictionary
        return {"val_loss": avg_loss, "val_ppl": ppl}

    # High-level method running training across all epochs
    def fit(
        self,
        epochs: int,
        max_steps: Optional[int] = None,
        save_best: bool = True,
        step_callback: Optional[Callable[[int, float], None]] = None,
        eval_interval: int = 1,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        """
        Complete training run across epochs.
        """
        # Dictionary to store loss and perplexity history curves
        history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

        # Print header summary of training configuration
        print(
            f"\n>>> Starting Training on {self.device_type.upper()} "
            f"(AMP: {self.use_amp}, Dtype: {self.amp_dtype}, GradAccum: {self.grad_accum_steps} steps) <<<",
            flush=True,
        )

        # Iterate through epoch numbers
        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(
                epoch=epoch,
                max_steps=max_steps,
                step_callback=step_callback,
                log_interval=log_interval,
            )
            # Record training metrics
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_ppl"].append(train_metrics["train_ppl"])

            # Run validation evaluation at specified intervals or at the final epoch
            if self.val_loader is not None and (epoch % eval_interval == 0 or epoch == epochs):
                # Evaluate model on validation set
                val_metrics = self.evaluate()
                # Record validation metrics
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_ppl"].append(val_metrics["val_ppl"])
                # Print validation summary
                print(
                    f"\n--- Epoch {epoch} Validation: Loss = {val_metrics['val_loss']:.4f} | "
                    f"PPL = {val_metrics['val_ppl']:.2f} ---\n",
                    flush=True,
                )

                # Check if current model achieved a new lowest validation loss
                if save_best and val_metrics["val_loss"] < self.best_val_loss:
                    # Update lowest validation loss
                    self.best_val_loss = val_metrics["val_loss"]
                    # Extract config dictionary if model has one
                    extra_cfg = {"config": self.model.config.__dict__} if hasattr(self.model, "config") else {}
                    # Save best checkpoint atomically
                    ckpt_path = self.ckpt_manager.save(
                        filename="best_model.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.global_step,
                        val_loss=self.best_val_loss,
                        extra_state=extra_cfg,
                    )
                    # Print checkpoint confirmation
                    print(f"[*] New best model saved to: {ckpt_path}\n", flush=True)

            # Check if global step limit has been reached
            if max_steps is not None and self.global_step >= max_steps:
                # Print completion notification
                print(f"Reached max_steps ({max_steps}). Finishing training.")
                # Terminate epoch loop
                break

        # Return full training history
        return history
