import os
import numpy as np
from tokenizers import Tokenizer

def convert_split_to_bins(
    text_path,
    tokenizer_path,
    output_dir,
    prefix="train",
    tokens_per_shard=50_000_000,
    chunk_size=10000,
):
    """Streams a text file, tokenizes it in batches, and saves it into

    binary shards (.bin) or a single binary file (if tokens_per_shard is None).
    """
    if not os.path.exists(text_path):
        print(f"Skipping {prefix}: File not found at {text_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    shard_idx = 0
    current_shard_tokens = []
    total_tokens = 0

    def flush_shard(tokens, idx):
        if not tokens:
            return
        if tokens_per_shard is None:
            shard_path = os.path.join(output_dir, f"{prefix}.bin")
        else:
            shard_path = os.path.join(output_dir, f"{prefix}_shard_{idx:03d}.bin")
            
        token_array = np.array(tokens, dtype=np.uint32)
        token_array.tofile(shard_path)
        print(f"Saved {prefix} file/shard to {shard_path} with {len(token_array):,} tokens")

    print(f"\nProcessing {prefix}.txt...")
    with open(text_path, "r", encoding="utf-8") as f:
        batch_text = []
        for line in f:
            cleaned = line.strip()
            if cleaned:
                batch_text.append(cleaned)

            if len(batch_text) >= chunk_size:
                encoded = tokenizer.encode_batch(batch_text)
                for enc in encoded:
                    current_shard_tokens.extend(enc.ids)
                    total_tokens += len(enc.ids)
                batch_text = []

                # Check if current shard is full (only if tokens_per_shard is set)
                if tokens_per_shard and len(current_shard_tokens) >= tokens_per_shard:
                    flush_shard(current_shard_tokens, shard_idx)
                    current_shard_tokens = []
                    shard_idx += 1

        # Process final batch of text
        if batch_text:
            encoded = tokenizer.encode_batch(batch_text)
            for enc in encoded:
                current_shard_tokens.extend(enc.ids)
                total_tokens += len(enc.ids)

    # Flush any remaining tokens
    if current_shard_tokens:
        flush_shard(current_shard_tokens, shard_idx if tokens_per_shard else 0)

    print(f"Completed {prefix}! Total tokens processed: {total_tokens:,}")

# --- Base Paths Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer_runs/bpe_vocab_7500/tokenizer.json")
DATA_DIR = os.path.join(BASE_DIR, "data/final_data_set")
OUTPUT_DIR = os.path.join(BASE_DIR, "token_shards")

# 1. Convert train.txt into multiple shards (~50M tokens each)
convert_split_to_bins(
    text_path=os.path.join(DATA_DIR, "train.txt"),
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    prefix="train",
    tokens_per_shard=50_000_000
)

# 2. Convert val.txt into a single val.bin
convert_split_to_bins(
    text_path=os.path.join(DATA_DIR, "val.txt"),
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    prefix="val",
    tokens_per_shard=None  # Single file output
)

# 3. Convert test.txt into a single test.bin
convert_split_to_bins(
    text_path=os.path.join(DATA_DIR, "test.txt"),
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    prefix="test",
    tokens_per_shard=None  # Single file output
)