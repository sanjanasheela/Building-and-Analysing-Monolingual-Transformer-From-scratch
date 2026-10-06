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

def analyze_corpus(dataset_file: str, batch_size: int = 250_000):
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        return

    print("Loading tokenizers...")
    tokenizers = {vocab: Tokenizer.from_file(path) for vocab, path in VOCAB_MODELS.items()}

    lines_by_source = {}
    words_by_source = {}
    tokens_by_vocab_source = {v: {} for v in VOCAB_MODELS}

    total_lines = 0
    total_words = 0

    batch_texts = []
    batch_sources = []

    print(f"Processing '{dataset_file}'...\n")

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Corpus"):
            raw_str = line.strip()
            if not raw_str:
                continue

            parts = raw_str.split(maxsplit=1)
            source = parts[0]
            text = parts[1].strip() if len(parts) > 1 else ""

            if not text:
                continue

            word_count = len(text.split())

            lines_by_source[source] = lines_by_source.get(source, 0) + 1
            words_by_source[source] = words_by_source.get(source, 0) + word_count
            total_lines += 1
            total_words += word_count

            batch_texts.append(text)
            batch_sources.append(source)

            if len(batch_texts) >= batch_size:
                for vocab, tok in tokenizers.items():
                    encodings = tok.encode_batch(batch_texts)
                    for enc, src in zip(encodings, batch_sources):
                        n_toks = len(enc.ids)
                        tokens_by_vocab_source[vocab][src] = tokens_by_vocab_source[vocab].get(src, 0) + n_toks

                batch_texts = []
                batch_sources = []

        if batch_texts:
            for vocab, tok in tokenizers.items():
                encodings = tok.encode_batch(batch_texts)
                for enc, src in zip(encodings, batch_sources):
                    n_toks = len(enc.ids)
                    tokens_by_vocab_source[vocab][src] = tokens_by_vocab_source[vocab].get(src, 0) + n_toks

    print("\n" + "=" * 85)
    print("DETAILED SOURCE BREAKDOWN (LINES & WORDS)")
    print("=" * 85)
    print(f"{'Source Tag':<15} | {'Line Count':<15} | {'Line %':<10} | {'Word Count':<18} | {'Word %':<10}")
    print("-" * 85)
    for src in sorted(lines_by_source.keys()):
        l_cnt = lines_by_source[src]
        l_pct = (l_cnt / total_lines) * 100
        w_cnt = words_by_source[src]
        w_pct = (w_cnt / total_words) * 100
        print(f"{src:<15} | {l_cnt:<15,} | {l_pct:<9.2f}% | {w_cnt:<18,} | {w_pct:<9.2f}%")
    print("-" * 85)
    print(f"{'TOTAL':<15} | {total_lines:<15,} | {'100.00%':<10} | {total_words:<18,} | {'100.00%':<10}")
    print("=" * 85)

    print("\n" + "=" * 95)
    print("TOKEN SWEEP ACROSS VOCAB SIZES (5000, 7500, 10000, 12500)")
    print("=" * 95)

    header = f"{'Vocab Size':<12} | {'Total Tokens':<18} | {'Fertility':<12} | {'Manual Tokens':<18} | {'Public Tokens':<18}"
    print(header)
    print("-" * 95)
    for vocab in ["5000", "7500", "10000", "12500"]:
        tot_toks = sum(tokens_by_vocab_source[vocab].values())
        fertility = tot_toks / total_words
        man_toks = tokens_by_vocab_source[vocab].get("manual", 0)
        pub_toks = tokens_by_vocab_source[vocab].get("public", 0)
        print(f"{vocab:<12} | {tot_toks:<18,} | {fertility:<12.4f} | {man_toks:<18,} | {pub_toks:<18,}")
    print("=" * 95)

if __name__ == "__main__":
    analyze_corpus(DATASET_FILE)
