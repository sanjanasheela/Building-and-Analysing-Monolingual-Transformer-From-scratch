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

def analyze_corpus_sources_and_tokens(dataset_file: str, batch_size: int = 100_000):
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        return

    print("Loading tokenizers...")
    tokenizers = {vocab: Tokenizer.from_file(path) for vocab, path in VOCAB_MODELS.items()}

    # Sources tracking
    lines_by_source = {}
    words_by_source = {}
    tokens_by_vocab_source = {v: {} for v in VOCAB_MODELS}

    total_lines = 0
    total_words = 0

    batch_texts = []
    batch_sources = []

    print(f"Analyzing source breakdown and token counts across all vocabulary sizes on '{dataset_file}'...\n")

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading & Tokenizing Corpus"):
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

        # Flush final batch
        if batch_texts:
            for vocab, tok in tokenizers.items():
                encodings = tok.encode_batch(batch_texts)
                for enc, src in zip(encodings, batch_sources):
                    n_toks = len(enc.ids)
                    tokens_by_vocab_source[vocab][src] = tokens_by_vocab_source[vocab].get(src, 0) + n_toks

    print("\n" + "=" * 80)
    print("DETAILED SOURCE BREAKDOWN (LINES & WORDS)")
    print("=" * 80)
    print(f"{'Source Tag':<15} | {'Line Count':<15} | {'Line %':<10} | {'Word Count':<15} | {'Word %':<10}")
    print("-" * 80)
    for src in sorted(lines_by_source.keys()):
        l_cnt = lines_by_source[src]
        l_pct = (l_cnt / total_lines) * 100
        w_cnt = words_by_source[src]
        w_pct = (w_cnt / total_words) * 100
        print(f"{src:<15} | {l_cnt:<15,} | {l_pct:<9.2f}% | {w_cnt:<15,} | {w_pct:<9.2f}%")
    print("-" * 80)
    print(f"{'TOTAL':<15} | {total_lines:<15,} | {'100.00%':<10} | {total_words:<15,} | {'100.00%':<10}")
    print("=" * 80)

    print("\n" + "=" * 90)
    print("TOKEN SWEEP Across Vocab Sizes (5000, 7500, 10000, 12500)")
    print("=" * 90)

    for vocab in ["5000", "7500", "10000", "12500"]:
        tot_toks = sum(tokens_by_vocab_source[vocab].values())
        fertility = tot_toks / total_words
        print(f"\n--- VOCAB SIZE: {vocab} ---")
        print(f"Total Tokens: {tot_toks:,} | Fertility: {fertility:.4f} tokens/word")
        print(f"{'Source Tag':<15} | {'Token Count':<20} | {'Token Share %':<15}")
        print("-" * 55)
        for src in sorted(tokens_by_vocab_source[vocab].keys()):
            t_cnt = tokens_by_vocab_source[vocab][src]
            t_pct = (t_cnt / tot_toks) * 100
            print(f"{src:<15} | {t_cnt:<20,} | {t_pct:<14.2f}%")

if __name__ == "__main__":
    analyze_corpus_sources_and_tokens(DATASET_FILE)
