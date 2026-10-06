"""
count_tokens.py
=======================================================
Counts total tokens, words, and manual/public split breakdown
for any trained local tokenizer model (from tokenizers or transformers)
or HuggingFace pretrained tokenizer (e.g. sarvamai/sarvam-1).

Usage:
    python count_tokens.py
"""

import os
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus.txt"
MODEL_PATH   = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/bpe_vocab_12500/tokenizer.json"
BATCH_SIZE   = 125_000
# ──────────────────────────────────────────────────────────────────────────────


def load_tokenizer(model_path: str):
    """Load tokenizer from local JSON file or HuggingFace hub."""
    if os.path.exists(model_path):
        print(f"Loading local trained tokenizer from: {model_path}")
        return Tokenizer.from_file(model_path)
    else:
        print(f"Loading pretrained HuggingFace tokenizer for: {model_path}")
        return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def count_tokens(dataset_file: str, model_path: str, batch_size: int = BATCH_SIZE):
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        return

    tok = load_tokenizer(model_path)
    is_hf = hasattr(tok, "batch_encode_plus")

    total_tokens = 0
    manual_tokens = 0
    public_tokens = 0
    total_words = 0
    total_lines = 0

    batch_texts = []
    batch_labels = []

    print(f"Counting tokens in '{dataset_file}'...\n")

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting tokens"):
            raw_str = line.strip()
            if not raw_str:
                continue

            label = ""
            if "\t" in raw_str:
                label, text = raw_str.split("\t", 1)
                text = text.strip()
            else:
                parts = raw_str.split(maxsplit=1)
                if parts[0] in ("manual", "public"):
                    label = parts[0]
                    text = parts[1].strip() if len(parts) > 1 else ""
                else:
                    text = raw_str

            if not text:
                continue

            batch_texts.append(text)
            batch_labels.append(label)
            total_lines += 1
            total_words += len(text.split())

            if len(batch_texts) >= batch_size:
                if is_hf:
                    encodings = tok(batch_texts, add_special_tokens=False)["input_ids"]
                    for enc, lbl in zip(encodings, batch_labels):
                        n_toks = len(enc)
                        total_tokens += n_toks
                        if lbl == "manual":
                            manual_tokens += n_toks
                        elif lbl == "public":
                            public_tokens += n_toks
                else:
                    encodings = tok.encode_batch(batch_texts)
                    for enc, lbl in zip(encodings, batch_labels):
                        n_toks = len(enc.ids)
                        total_tokens += n_toks
                        if lbl == "manual":
                            manual_tokens += n_toks
                        elif lbl == "public":
                            public_tokens += n_toks

                batch_texts = []
                batch_labels = []

        # Flush final batch
        if batch_texts:
            if is_hf:
                encodings = tok(batch_texts, add_special_tokens=False)["input_ids"]
                for enc, lbl in zip(encodings, batch_labels):
                    n_toks = len(enc)
                    total_tokens += n_toks
                    if lbl == "manual":
                        manual_tokens += n_toks
                    elif lbl == "public":
                        public_tokens += n_toks
            else:
                encodings = tok.encode_batch(batch_texts)
                for enc, lbl in zip(encodings, batch_labels):
                    n_toks = len(enc.ids)
                    total_tokens += n_toks
                    if lbl == "manual":
                        manual_tokens += n_toks
                    elif lbl == "public":
                        public_tokens += n_toks

    print("\n" + "=" * 60)
    print(f"TOKEN COUNT REPORT")
    print("=" * 60)
    print(f"  Model Used            : {model_path}")
    print(f"  Total Lines Processed : {total_lines:,}")
    print(f"  Total Words           : {total_words:,}")
    print(f"  Total BPE Tokens      : {total_tokens:,}")
    if manual_tokens or public_tokens:
        print(f"    - Manual Data Tokens: {manual_tokens:,} ({manual_tokens / total_tokens * 100:.2f}%)")
        print(f"    - Public Data Tokens: {public_tokens:,} ({public_tokens / total_tokens * 100:.2f}%)")
    print(f"  Average Fertility     : {total_tokens / total_words:.4f} tokens/word")
    print("=" * 60)


if __name__ == "__main__":
    count_tokens(DATASET_FILE, MODEL_PATH)