import os
import sys
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

DATASET_FILE = "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/data/processed/nepali_corpus.txt"

VOCAB_MODELS = {
    "5000": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/bpe_vocab_5000/tokenizer.json",
    "7500": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/bpe_vocab_7500/tokenizer.json",
    "10000": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/bpe_vocab_10000/tokenizer.json",
    "12500": "/home/sanjana/Documents/7/LMA/individual-project-sanjanasheela/language_L/tokenizer_runs/bpe_vocab_12500/tokenizer.json",
}

def count_single_vocab(vocab_size: str, model_path: str, dataset_file: str, batch_size: int = 100_000):
    if not os.path.exists(dataset_file):
        return
    tok = Tokenizer.from_file(model_path)
    
    total_tokens = 0
    manual_tokens = 0
    public_tokens = 0
    total_words = 0
    total_lines = 0

    batch_texts = []
    batch_labels = []

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            raw_str = line.strip()
            if not raw_str:
                continue

            parts = raw_str.split(maxsplit=1)
            src = parts[0]
            text = parts[1].strip() if len(parts) > 1 else ""

            if not text:
                continue

            total_lines += 1
            total_words += len(text.split())
            batch_texts.append(text)
            batch_labels.append(src)

            if len(batch_texts) >= batch_size:
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

        if batch_texts:
            encodings = tok.encode_batch(batch_texts)
            for enc, lbl in zip(encodings, batch_labels):
                n_toks = len(enc.ids)
                total_tokens += n_toks
                if lbl == "manual":
                    manual_tokens += n_toks
                elif lbl == "public":
                    public_tokens += n_toks

    print(f"Vocab {vocab_size}: Total Tokens = {total_tokens:,} | Fertility = {total_tokens / total_words:.4f} | Manual Tokens = {manual_tokens:,} | Public Tokens = {public_tokens:,}")

if __name__ == "__main__":
    for v, p in VOCAB_MODELS.items():
        count_single_vocab(v, p, DATASET_FILE)
