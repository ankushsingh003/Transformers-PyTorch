"""
Research Paper Reproduction:
'Language Models are Unsupervised Multitask Learners' (Radford et al., 2019 / GPT-2)
Implemented completely from scratch without external libraries.

Features demonstrated:
1. Pure PyTorch Decoder-Only Transformer (Custom LayerNorm, Attention, FFN, Pre-LN)
2. Custom Training Engine with Automatic Mixed Precision (AMP) and Gradient Accumulation
3. Validation Perplexity evaluation & Autoregressive text generation
"""

# Import os module for managing directories and file paths
import os
# Import sys module to modify module resolution paths
import sys
# Import argparse module for parsing command-line flags and parameters
import argparse
# Import PyTorch core library
import torch
# Import DataLoader for batching and shuffling training examples
from torch.utils.data import DataLoader

# Insert directory path so local module folders ('models', 'engine', 'data') are discovered
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import GPT model class and GPTConfig container
from models.gpt import GPT, GPTConfig
# Import our custom library-free training engine
from engine.trainer import CustomTrainer
# Import custom Cosine Annealing with linear warmup scheduler
from engine.lr_scheduler import CosineWarmupScheduler
# Import dataset utilities and TextDataset class
from data.dataset import get_shakespeare_data, TextDataset


# Function to parse and validate command-line arguments
def parse_args():
    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Reproduce GPT-2 Causal LM on Tiny Shakespeare")
    # Argument: number of passes over the training dataset
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    # Argument: maximum training steps (optimizer updates) to run
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum training steps (for quick reproduction)")
    # Argument: micro-batch size processed in a single forward/backward pass
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size")
    # Argument: number of micro-batches to accumulate before performing optimizer step
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    # Argument: context window sequence length (block size)
    parser.add_argument("--block-size", type=int, default=128, help="Context sequence length")
    # Argument: hidden embedding dimension (d_model)
    parser.add_argument("--d-model", type=int, default=128, help="Hidden dimension size")
    # Argument: number of parallel attention heads
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    # Argument: number of transformer block layers
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    # Argument: peak learning rate reached at the end of warmup
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    # Argument: number of warmup steps for learning rate schedule
    parser.add_argument("--warmup-steps", type=int, default=30, help="Learning rate warmup steps")
    # Argument: choice of normalization layer ('layernorm' or 'rmsnorm')
    parser.add_argument("--norm-type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])
    # Argument: target device ('cuda', 'cpu', or 'auto')
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    # Argument: flag to disable Automatic Mixed Precision
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    # Argument: prompt string for testing text generation
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Prompt for generation demo")
    # Argument: cap maximum samples loaded for fast iteration (0 for all)
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples to load (0 for all)")
    # Argument: interval frequency for printing training progress
    parser.add_argument("--log-interval", type=int, default=5, help="Step log interval")
    # Parse arguments and return
    return parser.parse_args()


# Main reproduction function
def main():
    # Parse command-line arguments
    args = parse_args()
    # Print benchmark banner
    print("=" * 70, flush=True)
    print(" REPRODUCING: 'Language Models are Unsupervised Multitask Learners' (GPT-2)", flush=True)
    print("=" * 70, flush=True)

    # 1. Resolve compute hardware device
    if args.device == "auto":
        # Automatically choose CUDA GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Use explicitly provided device string
        device = torch.device(args.device)
    # Print selected hardware device
    print(f"[*] Running on device: {device}", flush=True)

    # 2. Data Preparation and Tokenization
    print("\n[1/5] Loading and Tokenizing Dataset...", flush=True)
    # Load Shakespeare tokens and fitted character tokenizer
    train_tokens, val_tokens, tokenizer = get_shakespeare_data(data_dir="data")
    # Save tokenizer vocabulary to disk for future inference
    tokenizer.save(os.path.join("data", "tokenizer.json"))

    # Resolve optional sample limits for fast training
    max_train_samples = args.max_samples if args.max_samples > 0 else None
    # Use 1/5th of max samples for validation
    max_val_samples = (args.max_samples // 5) if args.max_samples > 0 else None

    # Construct PyTorch TextDataset for training
    train_dataset = TextDataset(train_tokens, block_size=args.block_size, max_samples=max_train_samples)
    # Construct PyTorch TextDataset for validation
    val_dataset = TextDataset(val_tokens, block_size=args.block_size, max_samples=max_val_samples)

    # Build training DataLoader with shuffling enabled
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    # Build validation DataLoader without shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # 3. Model Architecture Instantiation
    print("\n[2/5] Initializing Scratch GPT Architecture...")
    # Define hyperparameter configuration
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        norm_type=args.norm_type,
        pre_ln=True,
        dropout=0.1,
    )
    # Instantiate GPT model and move parameters to target device
    model = GPT(config).to(device)
    # Count model parameters
    num_params = model.get_num_params()
    # Print model parameter count
    print(f"[*] Model instantiated: {num_params:,} parameters ({num_params / 1e6:.2f}M)")

    # 4. Qualitative sample BEFORE training (demonstrates baseline untrained noise)
    print("\n[3/5] Sampling Untrained Model (Baseline Output):")
    # Prompt text to condition generation
    sample_prompt = args.prompt
    # Encode prompt string into 2D tensor of token IDs
    input_ids = torch.tensor([tokenizer.encode(sample_prompt)], dtype=torch.long, device=device)
    # Generate 60 tokens from untrained model
    raw_gen = model.generate(input_ids, max_new_tokens=60, temperature=0.8, top_k=10)
    # Print prompt
    print(f"Prompt: '{sample_prompt}'")
    # Print untrained output
    print(f"Untrained Generation:\n---\n{tokenizer.decode(raw_gen[0].tolist())}\n---")

    # 5. Training Engine Setup (AMP + Gradient Accumulation)
    print("\n[4/5] Setting up Custom Training Engine (AMP + Grad Accumulation)...")
    # Compute effective batch size: micro_batch * accumulation_steps
    effective_batch_size = args.batch_size * args.grad_accum
    # Print batch size statistics
    print(
        f"[*] Micro-batch size: {args.batch_size} | "
        f"Grad Accumulation Steps: {args.grad_accum} | "
        f"Effective Batch Size: {effective_batch_size}"
    )

    # Configure AdamW optimizer with selective weight decay
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=args.lr,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    # Calculate total steps for scheduler
    total_steps = min(args.max_steps, args.epochs * (len(train_loader) // args.grad_accum))
    # Configure Cosine Warmup learning rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=total_steps,
        min_lr=args.lr * 0.1,
    )

    # Instantiate our CustomTrainer
    trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints",
    )

    # 6. Execute Training Loop
    print("\n[5/5] Executing Training Loop...")
    # Train model across specified epochs and steps
    history = trainer.fit(
        epochs=args.epochs,
        max_steps=args.max_steps,
        save_best=True,
        eval_interval=1,
        log_interval=args.log_interval,
    )

    # 7. Final Evaluation & Generation Demonstration
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE — EVALUATION & GENERATION DEMO")
    print("=" * 70)

    # Evaluate final validation metrics
    val_metrics = trainer.evaluate()
    # Initial loss
    initial_train_loss = history["train_loss"][0] if history["train_loss"] else 0.0
    # Final validation loss
    final_val_loss = val_metrics.get("val_loss", 0.0)
    # Final validation perplexity
    final_val_ppl = val_metrics.get("val_ppl", 0.0)

    # Print quantitative loss and perplexity improvements
    print(f"[*] Initial Train Loss: {initial_train_loss:.4f}")
    print(f"[*] Final Validation Loss: {final_val_loss:.4f}")
    print(f"[*] Final Validation Perplexity: {final_val_ppl:.2f}")

    # Generate qualitative text samples at multiple temperatures
    print("\n[Trained Text Generation Demo]")
    for temp in [0.7, 0.9]:
        # Print temperature banner
        print(f"\n--- Sampling with Temperature = {temp} ---")
        # Autoregressively sample 150 new characters
        gen_tokens = model.generate(input_ids, max_new_tokens=150, temperature=temp, top_k=15)
        # Decode and print generated text
        print(tokenizer.decode(gen_tokens[0].tolist()))
    # Print footer divider
    print("=" * 70)


# Boilerplate execution
if __name__ == "__main__":
    # Call main
    main()
