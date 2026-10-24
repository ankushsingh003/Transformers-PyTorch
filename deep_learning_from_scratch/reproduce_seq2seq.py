"""
Research Paper Demonstration:
'Attention Is All You Need' (Vaswani et al., 2017)
Full Encoder-Decoder Transformer with Cross-Attention and NoamScheduler.
"""

# Import os module for managing file paths
import os
# Import sys module to modify module discovery path
import sys
# Import PyTorch core tensor library
import torch
# Import neural network modules from PyTorch
import torch.nn as nn
# Import DataLoader for batching and iterating through dataset
from torch.utils.data import DataLoader

# Insert current directory into sys.path so 'models', 'engine', and 'data' can be imported directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our from-scratch Encoder-Decoder Transformer model
from models.seq2seq import TransformerSeq2Seq
# Import our from-scratch Noam learning rate scheduler
from engine.lr_scheduler import NoamScheduler
# Import synthetic sequence reversal dataset
from data.dataset import Seq2SeqSyntheticDataset


# Main execution function for Seq2Seq paper demonstration
def main():
    # Print demonstration header banner
    print("=" * 70)
    print(" REPRODUCING: 'Attention Is All You Need' (Vaswani et al., 2017)")
    print(" Architecture: Full Encoder-Decoder with Cross-Attention")
    print("=" * 70)

    # Automatically resolve compute device (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the selected hardware device
    print(f"[*] Running on device: {device}")

    # 1. Dataset Setup: Sequence Reversal Task
    # Special reserved tokens: 0 = PAD (padding), 1 = SOS (Start of Sequence), 2 = EOS (End of Sequence)
    # Total vocabulary size is 25 (content tokens range from 3 to 24)
    vocab_size = 25
    # Sequence length of tokens to reverse
    seq_len = 8
    # Instantiate synthetic dataset generating 1,200 random sequences and their reversed targets
    dataset = Seq2SeqSyntheticDataset(num_samples=1200, seq_len=seq_len, vocab_size=vocab_size)
    # Wrap dataset in a PyTorch DataLoader with batch size 32 and shuffling enabled
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Model Architecture: Instantiate TransformerSeq2Seq
    model = TransformerSeq2Seq(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        dropout=0.1,
        pad_idx=0,
    ).to(device)

    # Instantiate Adam optimizer following original paper hyperparameters (beta1=0.9, beta2=0.98, eps=1e-9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    # Instantiate Noam learning rate scheduler with 100 warmup steps
    scheduler = NoamScheduler(optimizer, d_model=64, warmup_steps=100)
    # Cross-Entropy loss criterion, ignoring padding tokens (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Print start of training notification
    print(f"[*] Training Encoder-Decoder Seq2Seq on sequence reversal task...")
    # Put model into training mode
    model.train()
    # Iterate across 10 training epochs
    for epoch in range(1, 11):
        # Accumulator for total epoch loss
        total_loss = 0.0
        # Iterate over batches: source sequence, target input (with SOS), target output (with EOS)
        for src, tgt_in, tgt_out in loader:
            # Move all three tensors to compute device
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            # Clear existing parameter gradients
            optimizer.zero_grad()
            # Forward pass: encoder processes src, decoder processes tgt_in with cross-attention to encoder
            logits = model(src, tgt_in)
            # Compute cross-entropy loss against ground-truth target output tokens
            loss = criterion(logits.view(-1, vocab_size), tgt_out.view(-1))
            # Backward pass computing parameter gradients
            loss.backward()
            # Clip gradient norm at 1.0 to ensure stable optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Optimizer updates model weights
            optimizer.step()
            # Scheduler adjusts learning rate according to Noam formula
            scheduler.step()

            # Add batch loss to epoch total
            total_loss += loss.item()

        # Compute average loss across all batches in this epoch
        avg_loss = total_loss / len(loader)
        # Print progress report periodically
        if epoch % 2 == 0 or epoch == 1:
            # Print epoch, loss, and current learning rate
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 3. Test Generation: Autoregressively decode test sequences
    print("\n[Testing Autoregressive Seq2Seq Generation]")
    # Generate 3 random test source sequences
    test_src = torch.randint(3, vocab_size, (3, seq_len), device=device)
    # Execute greedy generation starting with SOS symbol (1) and stopping at EOS symbol (2)
    generated = model.generate(test_src, start_symbol=1, end_symbol=2, max_len=seq_len + 3)

    # Inspect each test sample prediction
    for i in range(3):
        # Convert source tensor to Python list
        src_list = test_src[i].tolist()
        # Compute ground truth expected reversed sequence
        expected = list(reversed(src_list))
        # Filter out special tokens (0=PAD, 1=SOS, 2=EOS) from model predictions
        pred_list = [t for t in generated[i].tolist() if t not in (0, 1, 2)]
        # Print sample index
        print(f"Sample {i+1}:")
        # Print original source sequence
        print(f"  Source:   {src_list}")
        # Print mathematically expected reversed sequence
        print(f"  Expected: {expected}")
        # Print actual model predicted sequence
        print(f"  Model:    {pred_list}")
        # Print whether prediction matches expected ground truth exactly
        print(f"  Match:    {'MATCH!' if expected == pred_list else 'Partial'}")

    # Print footer divider
    print("=" * 70)


# Boilerplate direct execution check
if __name__ == "__main__":
    # Call main function
    main()
