"""
Interactive Text Generation CLI for Scratch GPT.
Load saved checkpoints or run with initial weights to generate text with sampling.
"""

# Import os module for checking file existence and path joins
import os
# Import sys module to modify Python's import search path
import sys
# Import argparse module for parsing command-line flags and parameters
import argparse
# Import PyTorch core library
import torch

# Add current directory to sys.path so modules like 'models' and 'data' can be imported directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import GPT model class and GPTConfig hyperparameter container
from models.gpt import GPT, GPTConfig
# Import CharacterTokenizer for converting text to token indices and vice versa
from data.dataset import CharacterTokenizer


# Main execution function for interactive text generation CLI
def main():
    # Initialize command-line argument parser with a description
    parser = argparse.ArgumentParser(description="Generate text using Scratch GPT")
    # Argument: starting text prompt for model completion
    parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Input prompt")
    # Argument: number of subsequent characters/tokens to generate
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Number of tokens to generate")
    # Argument: softmax temperature scale (higher = creative/random, lower = deterministic)
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    # Argument: top-k filtering parameter (restricts sampling to k highest probability tokens)
    parser.add_argument("--top-k", type=int, default=10, help="Top-k filtering threshold")
    # Argument: boolean flag to enable greedy decoding (always picking argmax token)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    # Argument: path to saved model checkpoint file (.pt)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    # Argument: path to saved tokenizer JSON vocabulary file
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json", help="Path to tokenizer JSON")
    # Parse provided command-line arguments
    args = parser.parse_args()

    # Automatically resolve hardware device: use CUDA GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate CharacterTokenizer
    tokenizer = CharacterTokenizer()
    # Check if saved tokenizer vocabulary file exists on disk
    if os.path.exists(args.tokenizer):
        # Load saved vocabulary from JSON
        tokenizer.load(args.tokenizer)
        # Get vocabulary size
        vocab_size = tokenizer.vocab_size
        # Print confirmation
        print(f"Loaded tokenizer with vocab size: {vocab_size}")
    else:
        # If no saved tokenizer file exists, fall back to printable ASCII characters
        print("Tokenizer not found at default path, using basic ASCII fallback.")
        # Create standard ASCII string covering letters, digits, punctuation, and newline
        ascii_chars = "".join(chr(i) for i in range(32, 127)) + "\n"
        # Fit tokenizer on fallback ASCII characters
        tokenizer.fit(ascii_chars)
        # Get vocabulary size
        vocab_size = tokenizer.vocab_size

    # Variable to hold loaded checkpoint dictionary
    ckpt = None
    # Check if the checkpoint file exists on disk
    if os.path.exists(args.checkpoint):
        # Print status notification
        print(f"Loading checkpoint from: {args.checkpoint}")
        # Load checkpoint dictionary from file, mapping weights onto target device
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # If checkpoint contains saved model configuration in extra_state, restore it directly
    if ckpt is not None and "extra_state" in ckpt and "config" in ckpt["extra_state"]:
        # Extract configuration dictionary
        cfg_dict = ckpt["extra_state"]["config"]
        # Reconstruct GPTConfig object from dictionary
        config = GPTConfig(**cfg_dict)
    elif ckpt is not None:
        # If no config dict was saved, dynamically deduce model architecture dimensions from weight tensor shapes
        sd = ckpt["model_state_dict"]
        # Retrieve token embedding weight matrix
        tok_w = sd["tok_emb.weight"]
        # Vocabulary size is row count, d_model is column count
        v_size, d_m = tok_w.shape
        # Context window size is row count of positional embedding table
        max_len = sd["pos_emb.pos_emb.weight"].shape[0] if "pos_emb.pos_emb.weight" in sd else 256
        # Count number of unique block layer indices present in state dict keys
        n_lay = len(set(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")))
        # Construct matching GPTConfig
        config = GPTConfig(
            vocab_size=v_size,
            max_seq_len=max_len,
            d_model=d_m,
            n_heads=4,
            n_layers=n_lay,
        )
    else:
        # Default configuration if generating with uninitialized weights
        config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=256,
            d_model=128,
            n_heads=4,
            n_layers=4,
        )

    # Instantiate GPT model architecture with determined config and move to target device
    model = GPT(config).to(device)

    # If checkpoint was loaded, load weights into model
    if ckpt is not None:
        # Load weights state dictionary into model
        model.load_state_dict(ckpt["model_state_dict"])
        # Print confirmation message with epoch and validation loss
        print(f"Model restored from epoch {ckpt.get('epoch', '?')} (val_loss: {ckpt.get('val_loss', 0.0):.4f})")
    else:
        # Notify user that model will generate from scratch
        print(f"Notice: Checkpoint '{args.checkpoint}' not found. Generating with initialized weights.")

    # Set model into evaluation mode (disables dropout layers during inference)
    model.eval()

    # Convert prompt string into list of integer token IDs
    encoded_prompt = tokenizer.encode(args.prompt)
    # If prompt characters were not found, default to token 0
    if not encoded_prompt:
        encoded_prompt = [0]
    # Wrap token IDs in a 2D PyTorch tensor with batch dimension 1: shape (1, prompt_len)
    input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    # Print summary header of generation run
    print("\n" + "=" * 60)
    # Print the user's input prompt
    print(f"Prompt: {args.prompt}")
    # Print sampling parameter settings
    print(f"Settings: Temperature={args.temperature}, Top-K={args.top_k}, Greedy={args.greedy}")
    print("=" * 60)

    # Execute autoregressive generation loop
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    # Decode generated token IDs back into string text
    generated_text = tokenizer.decode(out[0].tolist())
    # Print generated output header
    print("\n[Generated Output]:")
    # Print the decoded text string
    print(generated_text)
    # Print footer divider
    print("\n" + "=" * 60)


# Standard Python boilerplate to execute main() when script is run directly
if __name__ == "__main__":
    # Call main function
    main()
