"""
Data handling and tokenization:
- CharacterTokenizer: Vocabulary mapping for character-level language modeling
- TextDataset: Autoregressive next-token prediction dataset (x_t, x_{t+1})
- Dataset utilities: Tiny Shakespeare downloader/generator
- Seq2SeqSyntheticDataset: Sequence reversal / transduction for Attention Is All You Need
"""

# Import os module for file path operations and directory creation
import os
# Import urllib.request to download datasets from the web
import urllib.request
# Import json module to serialize and deserialize the tokenizer vocabulary
import json
# Import typing annotations for tuples, lists, and optional arguments
from typing import Tuple, List, Optional
# Import PyTorch core library
import torch
# Import Dataset base class from PyTorch for creating custom datasets
from torch.utils.data import Dataset


# Character-level tokenizer converting text strings into integer sequences and vice versa
class CharacterTokenizer:
    """
    Character-level tokenizer with encode/decode methods and JSON persistence.
    """

    # Constructor method optionally taking initial text to build vocabulary
    def __init__(self, text: Optional[str] = None):
        # Dictionary mapping string characters to integer indices: {'a': 0, 'b': 1, ...}
        self.stoi = {}
        # Dictionary mapping integer indices back to string characters: {0: 'a', 1: 'b', ...}
        self.itos = {}
        # If training text was provided, fit vocabulary immediately
        if text is not None:
            # Fit vocabulary from input text
            self.fit(text)

    # Method to discover unique characters and assign integer IDs
    def fit(self, text: str):
        # Extract all unique characters from the text and sort them alphabetically
        chars = sorted(list(set(text)))
        # Build string-to-integer mapping
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # Build integer-to-string mapping
        self.itos = {i: ch for i, ch in enumerate(chars)}

    # Property returning the total number of unique tokens in the vocabulary
    @property
    def vocab_size(self) -> int:
        # Return length of the vocabulary dictionary
        return len(self.stoi)

    # Method to encode a string into a list of integer token IDs
    def encode(self, s: str) -> List[int]:
        # Look up each character in stoi dictionary and return list of integers
        return [self.stoi[c] for c in s if c in self.stoi]

    # Method to decode a list of integer token IDs back into a human-readable string
    def decode(self, indices: List[int]) -> str:
        # Look up each integer index in itos dictionary and join into a single string
        return "".join([self.itos.get(i, "") for i in indices])

    # Method to save the tokenizer vocabulary to a JSON file
    def save(self, filepath: str):
        # Prepare dictionary structure with string keys for JSON compatibility
        data = {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}
        # Open file for writing in UTF-8 encoding
        with open(filepath, "w", encoding="utf-8") as f:
            # Dump dictionary formatted with indentation
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Method to load saved vocabulary from a JSON file
    def load(self, filepath: str):
        # Open file for reading
        with open(filepath, "r", encoding="utf-8") as f:
            # Load JSON content
            data = json.load(f)
        # Restore string-to-integer dictionary
        self.stoi = data["stoi"]
        # Restore integer-to-string dictionary (converting keys back from string to int)
        self.itos = {int(k): v for k, v in data["itos"].items()}


# PyTorch Dataset for autoregressive causal language modeling
class TextDataset(Dataset):
    """
    Autoregressive Language Modeling Dataset.
    Extracts chunks of length `block_size`:
      x = tokens[idx : idx + block_size]
      y = tokens[idx + 1 : idx + block_size + 1]
    """

    # Constructor method taking a 1D token tensor and sequence length
    def __init__(self, tokens: torch.Tensor, block_size: int, max_samples: Optional[int] = None):
        # Store full 1D tensor of token integers
        self.tokens = tokens
        # Store the context window length (e.g. 64 or 128 tokens)
        self.block_size = block_size
        # Compute maximum number of overlapping chunks available in the stream
        n = max(0, len(tokens) - block_size)
        # If a sample cap is specified, limit total dataset size for rapid iteration
        if max_samples is not None and max_samples > 0:
            # Take minimum of available chunks and max_samples
            self.n_samples = min(n, max_samples)
        else:
            # Use all available chunks
            self.n_samples = n

    # Returns the total number of samples in the dataset
    def __len__(self) -> int:
        # Return sample count
        return self.n_samples

    # Retrieves a single training example (input sequence x and shifted target sequence y)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Slice a window of block_size + 1 consecutive tokens from the stream
        chunk = self.tokens[idx : idx + self.block_size + 1]
        # Input x is all tokens except the last one (length = block_size)
        x = chunk[:-1]
        # Target y is all tokens shifted right by 1 position (length = block_size)
        y = chunk[1:]
        # Return input and target pair
        return x, y


# Helper function to download or create Tiny Shakespeare dataset and tokenize it
def get_shakespeare_data(
    data_dir: str = "data",
    val_split: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, CharacterTokenizer]:
    """
    Downloads or generates the classic Tiny Shakespeare benchmark dataset,
    builds the character tokenizer, and splits into train and validation tensors.
    """
    # Create the data directory if it does not already exist
    os.makedirs(data_dir, exist_ok=True)
    # Define local file path for Shakespeare text file
    filepath = os.path.join(data_dir, "tinyshakespeare.txt")

    # If the dataset file does not exist locally, download it
    if not os.path.exists(filepath):
        # URL of Karpathy's Tiny Shakespeare corpus on GitHub
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        # Print progress notification
        print(f"Downloading Tiny Shakespeare dataset from {url}...")
        try:
            # Download file from URL to local filepath
            urllib.request.urlretrieve(url, filepath)
            # Print completion message
            print("Download complete.")
        except Exception as e:
            # If download fails (e.g., offline or network error), use fallback text
            print(f"Download failed ({e}). Generating fallback text corpus...")
            # Repeated dramatic script text for offline training
            sample_text = (
                "First Citizen:\n"
                "Before we proceed any further, hear me speak.\n\n"
                "All:\n"
                "Speak, speak.\n\n"
                "First Citizen:\n"
                "You are all resolved rather to die than to famish?\n\n"
                "All:\n"
                "Resolved. resolved.\n\n"
                "First Citizen:\n"
                "First, you know Caius Marcius is chief enemy to the people.\n"
            ) * 500
            # Write fallback text to disk
            with open(filepath, "w", encoding="utf-8") as f:
                # Write string to file
                f.write(sample_text)

    # Read the full corpus text from disk
    with open(filepath, "r", encoding="utf-8") as f:
        # Read entire file into string
        text = f.read()

    # Instantiate and fit character tokenizer on the corpus
    tokenizer = CharacterTokenizer(text)
    # Encode the entire corpus text into a 1D PyTorch tensor of integer IDs
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Compute number of tokens for validation set (e.g. 10% of corpus)
    n_val = int(len(encoded) * val_split)
    # Train tokens: all tokens except the final n_val tokens (90%)
    train_tokens = encoded[:-n_val]
    # Validation tokens: the final n_val tokens (10%)
    val_tokens = encoded[-n_val:]

    # Print summary statistics
    print(
        f"Corpus loaded: {len(text):,} characters | "
        f"Vocab size: {tokenizer.vocab_size} | "
        f"Train tokens: {len(train_tokens):,} | "
        f"Val tokens: {len(val_tokens):,}"
    )

    # Return training token tensor, validation token tensor, and fitted tokenizer
    return train_tokens, val_tokens, tokenizer


# Synthetic dataset for sequence transduction / reversal to test Encoder-Decoder models
class Seq2SeqSyntheticDataset(Dataset):
    """
    Synthetic dataset for sequence reversal/copying to validate Encoder-Decoder (Vaswani et al., 2017).
    Task: Given sequence [1, 5, 9, 3, ...], target is reversed sequence [..., 3, 9, 5, 1].
    """

    # Constructor generating synthetic sequences
    def __init__(self, num_samples: int = 1000, seq_len: int = 10, vocab_size: int = 20):
        # Store total sample count
        self.num_samples = num_samples
        # Store sequence length
        self.seq_len = seq_len
        # Store vocabulary size
        self.vocab_size = vocab_size

        # Special token convention: 0 = PAD, 1 = Start-of-Sequence (SOS), 2 = End-of-Sequence (EOS)
        # Content tokens range from 3 up to vocab_size - 1
        self.data = []
        # Generate samples iteratively
        for _ in range(num_samples):
            # Generate random source integer sequence: shape (seq_len,)
            src = torch.randint(3, vocab_size, (seq_len,))
            # Reverse sequence elements to create target content
            tgt_content = torch.flip(src, dims=[0])
            # Target input sequence begins with SOS token (1) followed by content
            tgt_in = torch.cat([torch.tensor([1]), tgt_content])
            # Target output sequence ends with EOS token (2) after content (for calculating loss)
            tgt_out = torch.cat([tgt_content, torch.tensor([2])])
            # Store tuple in list
            self.data.append((src, tgt_in, tgt_out))

    # Return dataset size
    def __len__(self) -> int:
        # Return total sample count
        return self.num_samples

    # Return sample at specified index
    def __getitem__(self, idx: int):
        # Unpack tuple
        src, tgt_in, tgt_out = self.data[idx]
        # Return source, decoder input, and target output
        return src, tgt_in, tgt_out
